from __future__ import annotations

import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Iterable

from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.config import Settings
from app.schemas import PaperSource


UNKNOWN = "\u672a\u77e5"


QA_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "\u4f60\u662f\u4e25\u8c28\u7684\u4e2d\u6587\u6587\u732e\u95ee\u7b54\u52a9\u624b\u3002"
            "\u53ea\u80fd\u4f9d\u636e\u7ed9\u5b9a\u7684\u8bba\u6587\u6458\u8981\u8d44\u6599\u56de\u7b54\u3002"
            "\u5982\u679c\u8d44\u6599\u4e0d\u8db3\u4ee5\u652f\u6301\u7ed3\u8bba\uff0c\u8bf7\u660e\u786e\u8bf4\u660e\u672a\u5728\u77e5\u8bc6\u5e93\u4e2d\u627e\u5230\u5145\u5206\u4f9d\u636e\u3002"
            "\u56de\u7b54\u8981\u7ed3\u6784\u6e05\u6670\uff0c\u5e76\u5728\u5173\u952e\u89c2\u70b9\u540e\u6807\u6ce8\u6765\u6e90\u7f16\u53f7\uff0c\u4f8b\u5982 [1]\u3001[2]\u3002",
        ),
        (
            "human",
            "\u95ee\u9898\uff1a{question}\n\n"
            "\u53ef\u7528\u8bba\u6587\u8d44\u6599\uff1a\n{context}\n\n"
            "\u8bf7\u7ed9\u51fa\u7b54\u6848\uff0c\u5e76\u5c3d\u91cf\u6307\u51fa\u76f8\u5173\u8bba\u6587\u4e3b\u9898\u3001\u671f\u520a\u6216\u5e74\u4efd\u5dee\u5f02\u3002",
        ),
    ]
)


class LocalHashingEmbeddings(Embeddings):
    def __init__(self, dimensions: int = 768):
        self.dimensions = dimensions

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in _tokenize_for_hashing(text):
            digest = hashlib.md5(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "little") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]


class SimplePaperStore:
    def __init__(self, index_path: str | Path, embeddings: Embeddings):
        self.index_path = Path(index_path)
        self.embeddings = embeddings
        self.records = self._load_records()

    def _load_records(self) -> list[dict]:
        if not self.index_path.exists():
            return []
        with self.index_path.open("r", encoding="utf-8") as file:
            data = json.load(file)
        return data.get("records", [])

    def count(self) -> int:
        return len(self.records)

    def journals(self) -> list[str]:
        journals = {
            record.get("metadata", {}).get("journal")
            for record in self.records
            if record.get("metadata", {}).get("journal")
        }
        return sorted(journals)

    def similarity_search_with_score(self, query: str, k: int) -> list[tuple[Document, float]]:
        query_embedding = self.embeddings.embed_query(query)
        scored: list[tuple[float, dict]] = []
        for record in self.records:
            score = _cosine_similarity(query_embedding, record["embedding"])
            scored.append((score, record))
        scored.sort(key=lambda item: item[0], reverse=True)
        results = []
        for score, record in scored[:k]:
            doc = Document(
                page_content=record["page_content"],
                metadata=record.get("metadata", {}),
            )
            distance = 1.0 - score
            results.append((doc, distance))
        return results


class PaperRAG:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.embeddings = create_embeddings(settings)
        self.store = SimplePaperStore(settings.simple_index_path, self.embeddings)
        self.llm = create_chat_model(settings)
        self.qa_chain = QA_PROMPT | self.llm | StrOutputParser()

    def document_count(self) -> int:
        return self.store.count()

    def ask(
        self,
        question: str,
        k: int = 5,
        journal: str | None = None,
        year_from: int | None = None,
        year_to: int | None = None,
    ) -> tuple[str, list[PaperSource]]:
        docs = self._search_documents(question, k, journal, year_from, year_to)
        if not docs:
            return "\u672a\u5728\u77e5\u8bc6\u5e93\u4e2d\u68c0\u7d22\u5230\u8db3\u591f\u76f8\u5173\u7684\u8bba\u6587\u8d44\u6599\u3002", []
        context = self._format_context([doc for doc, _distance in docs])
        answer = self.qa_chain.invoke({"question": question, "context": context})
        return answer, [self._to_source(doc, distance) for doc, distance in docs]

    def similar(
        self,
        query: str,
        k: int = 5,
        journal: str | None = None,
        year_from: int | None = None,
        year_to: int | None = None,
    ) -> list[PaperSource]:
        docs = self._search_documents(query, k, journal, year_from, year_to)
        return [self._to_source(doc, distance) for doc, distance in docs]

    def journals(self) -> list[str]:
        return self.store.journals()

    def _search_documents(
        self,
        query: str,
        k: int,
        journal: str | None,
        year_from: int | None,
        year_to: int | None,
    ) -> list[tuple[Document, float]]:
        fetch_k = min(max(k * 5, k), 100)
        docs = self.store.similarity_search_with_score(query, k=fetch_k)
        filtered = [
            (doc, distance)
            for doc, distance in docs
            if _metadata_matches(doc.metadata, journal, year_from, year_to)
        ]
        return filtered[:k]

    @staticmethod
    def _format_context(documents: Iterable[Document]) -> str:
        chunks = []
        for index, doc in enumerate(documents, start=1):
            meta = doc.metadata
            chunks.append(
                "\n".join(
                    [
                        f"[{index}] {meta.get('title', '')}",
                        f"\u671f\u520a\uff1a{meta.get('journal', '')}",
                        f"\u65f6\u95f4\uff1a{meta.get('published_at') or UNKNOWN}",
                        f"\u5f15\u7528\uff1a{meta.get('citation_count', UNKNOWN)}",
                        f"\u8d44\u6599\uff1a{doc.page_content}",
                    ]
                )
            )
        return "\n\n".join(chunks)

    @staticmethod
    def _to_source(doc: Document, distance: float | None = None) -> PaperSource:
        meta = doc.metadata
        return PaperSource(
            paper_id=str(meta.get("paper_id", "")),
            title=str(meta.get("title", "")),
            journal=str(meta.get("journal", "")),
            published_at=meta.get("published_at") or None,
            year=meta.get("year"),
            authors=meta.get("authors") or None,
            citation_count=meta.get("citation_count"),
            source_sheet=meta.get("source_sheet"),
            source_row=meta.get("source_row"),
            distance=distance,
            abstract_preview=_extract_abstract_preview(doc.page_content),
        )


def build_simple_index(documents: list[Document], embeddings: Embeddings, index_path: str | Path) -> None:
    path = Path(index_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    vectors = embeddings.embed_documents([doc.page_content for doc in documents])
    records = [
        {
            "page_content": doc.page_content,
            "metadata": doc.metadata,
            "embedding": vector,
        }
        for doc, vector in zip(documents, vectors)
    ]
    with path.open("w", encoding="utf-8") as file:
        json.dump({"records": records}, file, ensure_ascii=False)



def configure_network(settings: Settings) -> None:
    if settings.disable_proxy:
        for key in [
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "ALL_PROXY",
            "http_proxy",
            "https_proxy",
            "all_proxy",
        ]:
            os.environ.pop(key, None)
def create_embeddings(settings: Settings) -> Embeddings:
    configure_network(settings)
    if settings.embeddings_provider == "local":
        return LocalHashingEmbeddings()

    if settings.embeddings_provider == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        api_key = settings.effective_gemini_api_key
        if not api_key:
            raise ValueError("GEMINI_API_KEY is required when EMBEDDING_PROVIDER=gemini")
        os.environ.setdefault("GOOGLE_API_KEY", api_key)
        return GoogleGenerativeAIEmbeddings(
            model=settings.gemini_embedding_model,
            google_api_key=api_key,
        )

    if settings.embeddings_provider == "qwen":
        if settings.qwen_api_key:
            os.environ.setdefault("DASHSCOPE_API_KEY", settings.qwen_api_key)
        return DashScopeEmbeddings(
            model=settings.qwen_embedding_model,
            dashscope_api_key=settings.qwen_api_key or None,
        )

    raise ValueError(f"Unsupported EMBEDDING_PROVIDER: {settings.embedding_provider}")


def create_chat_model(settings: Settings):
    configure_network(settings)
    if settings.provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        api_key = settings.effective_gemini_api_key
        if not api_key:
            raise ValueError("GEMINI_API_KEY is required when LLM_PROVIDER=gemini")
        os.environ.setdefault("GOOGLE_API_KEY", api_key)
        return ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            temperature=settings.temperature,
            google_api_key=api_key,
        )

    if settings.provider == "qwen":
        if settings.qwen_api_key:
            os.environ.setdefault("DASHSCOPE_API_KEY", settings.qwen_api_key)
        return ChatTongyi(
            model_name=settings.qwen_model,
            temperature=settings.temperature,
            dashscope_api_key=settings.qwen_api_key or None,
        )

    raise ValueError(f"Unsupported LLM_PROVIDER: {settings.llm_provider}")


def _metadata_matches(metadata: dict, journal: str | None, year_from: int | None, year_to: int | None) -> bool:
    if journal and metadata.get("journal") != journal:
        return False
    year = metadata.get("year")
    if year is None:
        return year_from is None and year_to is None
    if year_from is not None and int(year) < year_from:
        return False
    if year_to is not None and int(year) > year_to:
        return False
    return True


def _extract_abstract_preview(content: str, limit: int = 180) -> str:
    match = re.search(r"\u6458\u8981[:\uff1a](.*)", content, flags=re.S)
    text = match.group(1).strip() if match else content.strip()
    return text[:limit] + ("..." if len(text) > limit else "")


def _tokenize_for_hashing(text: str) -> list[str]:
    lowered = text.lower()
    tokens = re.findall(r"[a-z0-9_]+", lowered)
    cjk_chars = re.findall(r"[\u4e00-\u9fff]", text)
    tokens.extend(cjk_chars)
    tokens.extend("".join(pair) for pair in zip(cjk_chars, cjk_chars[1:]))
    return tokens


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))

