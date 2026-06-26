from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    api_title: str = "Paper RAG API"
    api_version: str = "0.1.0"
    allow_origins: str = "*"

    llm_provider: str = Field(default="gemini", alias="LLM_PROVIDER")
    embedding_provider: str = Field(default="local", alias="EMBEDDING_PROVIDER")
    temperature: float = Field(default=0.2, alias="MODEL_TEMPERATURE")
    disable_proxy: bool = Field(default=True, alias="DISABLE_PROXY")

    qwen_api_key: str = Field(default="", alias="QWEN_API_KEY")
    qwen_model: str = Field(default="qwen-plus", alias="QWEN_MODEL")
    qwen_embedding_model: str = Field(default="text-embedding-v4", alias="QWEN_EMBEDDING_MODEL")

    gemini_api_key: str = Field(default="", alias="GEMINI_API_KEY")
    google_api_key: str = Field(default="", alias="GOOGLE_API_KEY")
    gemini_model: str = Field(default="gemini-flash-latest", alias="GEMINI_MODEL")
    gemini_embedding_model: str = Field(default="models/gemini-embedding-001", alias="GEMINI_EMBEDDING_MODEL")

    chroma_dir: Path = Field(default=Path("storage/chroma"), alias="CHROMA_DIR")
    chroma_collection: str = Field(default="paper_abstracts", alias="CHROMA_COLLECTION")
    simple_index_path: Path = Field(default=Path("storage/paper_index.json"), alias="SIMPLE_INDEX_PATH")

    @property
    def provider(self) -> str:
        return self.llm_provider.strip().lower()

    @property
    def effective_gemini_api_key(self) -> str:
        return self.gemini_api_key or self.google_api_key

    @property
    def embeddings_provider(self) -> str:
        return self.embedding_provider.strip().lower()

    @property
    def cors_origins(self) -> list[str]:
        if self.allow_origins.strip() == "*":
            return ["*"]
        return [item.strip() for item in self.allow_origins.split(",") if item.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()



