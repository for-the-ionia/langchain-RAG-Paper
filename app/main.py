from functools import lru_cache

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.config import Settings, get_settings
from app.rag import PaperRAG
from app.schemas import (
    AskRequest,
    AskResponse,
    HealthResponse,
    SimilarRequest,
    SimilarResponse,
)


@lru_cache
def get_rag() -> PaperRAG:
    return PaperRAG(get_settings())


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title=settings.api_title, version=settings.api_version)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    return app


app = create_app()


@app.get("/api/health", response_model=HealthResponse)
def health(
    settings: Settings = Depends(get_settings),
    rag: PaperRAG = Depends(get_rag),
) -> HealthResponse:
    count = rag.document_count()
    chat_model = settings.gemini_model if settings.provider == "gemini" else settings.qwen_model
    embedding_model = (
        "local-hashing"
        if settings.embeddings_provider == "local"
        else settings.gemini_embedding_model
        if settings.embeddings_provider == "gemini"
        else settings.qwen_embedding_model
    )
    return HealthResponse(
        status="ok" if count > 0 else "empty",
        collection=settings.chroma_collection,
        document_count=count,
        detail={
            "chroma_dir": str(settings.chroma_dir),
            "provider": settings.provider,
            "embedding_provider": settings.embeddings_provider,
            "chat_model": chat_model,
            "embedding_model": embedding_model,
        },
    )


@app.post("/api/ask", response_model=AskResponse)
def ask(request: AskRequest, rag: PaperRAG = Depends(get_rag)) -> AskResponse:
    try:
        answer, sources = rag.ask(
            question=request.question,
            k=request.k,
            journal=request.journal,
            year_from=request.year_from,
            year_to=request.year_to,
        )
        return AskResponse(answer=answer, sources=sources)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/similar", response_model=SimilarResponse)
def similar(request: SimilarRequest, rag: PaperRAG = Depends(get_rag)) -> SimilarResponse:
    try:
        results = rag.similar(
            query=request.query,
            k=request.k,
            journal=request.journal,
            year_from=request.year_from,
            year_to=request.year_to,
        )
        return SimilarResponse(query=request.query, results=results)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/journals", response_model=list[str])
def journals(rag: PaperRAG = Depends(get_rag)) -> list[str]:
    try:
        return rag.journals()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

