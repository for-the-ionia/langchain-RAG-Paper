from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pandas as pd
try:
    from langchain_core.documents import Document
except ModuleNotFoundError:
    from dataclasses import dataclass

    @dataclass
    class Document:  # type: ignore[no-redef]
        page_content: str
        metadata: dict[str, Any]


FIELD_ALIASES = {
    "title": ["\u7bc7\u540d", "\u6807\u9898", "\u8bba\u6587\u6807\u9898", "\u9898\u540d", "title", "paper_title"],
    "abstract": ["\u6458\u8981", "abstract", "summary"],
    "authors": ["\u4f5c\u8005", "authors", "author"],
    "journal": ["\u671f\u520a", "\u520a\u540d", "journal", "source"],
    "published_at": ["\u65f6\u95f4", "\u53d1\u8868\u65f6\u95f4", "\u65e5\u671f", "\u51fa\u7248\u65f6\u95f4", "year", "published_at"],
    "citation_count": ["\u5f15\u7528", "\u88ab\u5f15", "\u88ab\u5f15\u91cf", "citations", "citation_count"],
}


LABELS = {
    "title": "\u8bba\u6587\u6807\u9898",
    "authors": "\u4f5c\u8005",
    "journal": "\u671f\u520a",
    "published_at": "\u53d1\u8868\u65f6\u95f4",
    "citation_count": "\u5f15\u7528\u6b21\u6570",
    "abstract": "\u6458\u8981",
    "unknown": "\u672a\u77e5",
}


def _clean_cell(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    return " ".join(text.split())


def _find_column(columns: list[str], logical_name: str) -> str | None:
    normalized = {str(column).strip().lower(): column for column in columns}
    for alias in FIELD_ALIASES[logical_name]:
        key = alias.strip().lower()
        if key in normalized:
            return normalized[key]
    return None


def _safe_int(value: Any) -> int | None:
    text = _clean_cell(value)
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def _normalize_date(value: Any) -> tuple[str, int | None]:
    if pd.isna(value):
        return "", None
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.notna(parsed):
        return parsed.strftime("%Y-%m-%d"), int(parsed.year)

    text = _clean_cell(value)
    year = None
    for token in text.replace("/", "-").split("-"):
        if token.isdigit() and len(token) == 4:
            year = int(token)
            break
    return text, year


def _paper_id(title: str, journal: str, published_at: str, abstract: str) -> str:
    raw = f"{title}|{journal}|{published_at}|{abstract}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def _build_content(metadata: dict[str, Any], abstract: str) -> str:
    citation = metadata.get("citation_count")
    lines = [
        f"{LABELS['title']}:{metadata.get('title', '')}",
        f"{LABELS['authors']}:{metadata.get('authors', '') or LABELS['unknown']}",
        f"{LABELS['journal']}:{metadata.get('journal', '')}",
        f"{LABELS['published_at']}:{metadata.get('published_at', '') or LABELS['unknown']}",
        f"{LABELS['citation_count']}:{citation if citation is not None else LABELS['unknown']}",
        f"{LABELS['abstract']}:{abstract}",
    ]
    return "\n".join(lines)


def load_excel_documents(excel_path: str | Path) -> list[Document]:
    path = Path(excel_path)
    if not path.exists():
        raise FileNotFoundError(f"Excel file does not exist: {path}")

    workbook = pd.ExcelFile(path)
    documents: list[Document] = []
    seen_ids: set[str] = set()

    for sheet_name in workbook.sheet_names:
        frame = pd.read_excel(path, sheet_name=sheet_name)
        columns = [str(column).strip() for column in frame.columns]
        mapped = {name: _find_column(columns, name) for name in FIELD_ALIASES}

        if not mapped["title"] or not mapped["abstract"]:
            continue

        for row_index, row in frame.iterrows():
            title = _clean_cell(row.get(mapped["title"]))
            abstract = _clean_cell(row.get(mapped["abstract"]))
            if not title or not abstract:
                continue

            journal = _clean_cell(row.get(mapped["journal"])) if mapped["journal"] else sheet_name
            authors = _clean_cell(row.get(mapped["authors"])) if mapped["authors"] else ""
            published_at, year = (
                _normalize_date(row.get(mapped["published_at"]))
                if mapped["published_at"]
                else ("", None)
            )
            citation_count = (
                _safe_int(row.get(mapped["citation_count"]))
                if mapped["citation_count"]
                else None
            )

            doc_id = _paper_id(title, journal, published_at, abstract)
            if doc_id in seen_ids:
                doc_id = f"{doc_id}-{row_index + 2}"
            seen_ids.add(doc_id)

            metadata = {
                "paper_id": doc_id,
                "title": title,
                "authors": authors,
                "journal": journal,
                "published_at": published_at,
                "source_sheet": sheet_name,
                "source_row": int(row_index) + 2,
            }
            if year is not None:
                metadata["year"] = year
            if citation_count is not None:
                metadata["citation_count"] = citation_count
            documents.append(Document(page_content=_build_content(metadata, abstract), metadata=metadata))

    if not documents:
        raise ValueError("No valid paper records were parsed from Excel. Check title and abstract columns.")

    return documents

