from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path
from pydantic import Field


class Settings(BaseSettings):
    
    openai_api_key: str
    embedding_api_key: str
    openai_base_url: str
    es_url: str
    es_index: str
    embedding_dim: int
    max_tokens_chunk: int
    chunk_overlap: int
    top_k: int
    retrieval_k: int
    dense_k: int
    sparse_k: int
    num_candidates: int
    hybrid_dense_weight: float
    llm_model: str
    embedding_model_doc: str
    embedding_model_query: str
    embedding_concurrency: int
    enable_reranker: bool
    reranker_model: str
    reranker_batch_size: int
    jwt_secret: str
    jwt_expire_hours: int = 720
    chat_history_messages: int = Field(default=12, ge=0, le=100)
    chat_history_max_chars: int = Field(default=12000, ge=256)
    chat_rewrite_timeout_seconds: float = Field(default=10.0, gt=0)
    database_url: str = "postgresql+asyncpg://rag:rag@localhost:5433/rag_app"
    cors_origins: list[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]


    model_config = SettingsConfigDict(
        env_file=Path(__file__).resolve().parent.parent / ".env",
        extra="ignore",
    )

settings = Settings()
