from typing import Any

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="用户问题")
    k: int = Field(default=5, ge=1, le=20, description="召回论文数量")
    journal: str | None = Field(default=None, description="按期刊过滤")
    year_from: int | None = Field(default=None, description="起始年份")
    year_to: int | None = Field(default=None, description="结束年份")


class SimilarRequest(BaseModel):
    query: str = Field(..., min_length=1, description="标题、摘要或检索问题")
    k: int = Field(default=5, ge=1, le=20, description="返回相似论文数量")
    journal: str | None = Field(default=None, description="按期刊过滤")
    year_from: int | None = None
    year_to: int | None = None


class PaperSource(BaseModel):
    paper_id: str
    title: str
    journal: str
    published_at: str | None = None
    year: int | None = None
    authors: str | None = None
    citation_count: int | None = None
    source_sheet: str | None = None
    source_row: int | None = None
    distance: float | None = None
    abstract_preview: str | None = None


class AskResponse(BaseModel):
    answer: str
    sources: list[PaperSource]


class SimilarResponse(BaseModel):
    query: str
    results: list[PaperSource]


class HealthResponse(BaseModel):
    status: str
    collection: str
    document_count: int
    detail: dict[str, Any] = Field(default_factory=dict)
