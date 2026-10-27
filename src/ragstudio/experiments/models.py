from pydantic import BaseModel, Field


class PipelineConfig(BaseModel):
    name: str
    chunking_strategy: str = "fixed-size"
    chunk_size: int = Field(default=500, ge=100)
    overlap: int = Field(default=50, ge=0)
    retrieval_method: str = "hybrid_rrf"
    top_k: int = Field(default=5, ge=1)
    reranking: bool = True
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    generation_model: str = "gpt-4o-mini"
    query_rewrite: bool = False


class BenchmarkEntry(BaseModel):
    question: str
    answer: str | None = None
    relevant_passages: list[str] = []
