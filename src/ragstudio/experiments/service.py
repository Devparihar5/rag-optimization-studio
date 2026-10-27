from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class DatasetCreateRequest(BaseModel):
    name: str = Field(min_length=3, max_length=200)
    description: str = Field(default="")
    source_type: str = Field(default="pdf")


class DatasetRecord(BaseModel):
    id: str
    name: str
    description: str
    source_type: str
    created_at: str
    document_count: int = 0


class BenchmarkQuestion(BaseModel):
    question: str
    answer: str | None = None
    relevant_passages: list[str] = Field(default_factory=list)


class BenchmarkCreateRequest(BaseModel):
    name: str = Field(min_length=3, max_length=200)
    dataset_id: str
    questions: list[BenchmarkQuestion] = Field(default_factory=list)


class BenchmarkRecord(BaseModel):
    id: str
    name: str
    dataset_id: str
    question_count: int
    created_at: str


class ExperimentConfig(BaseModel):
    chunking_strategy: str = Field(default="fixed-size")
    chunk_size: int = Field(default=500, ge=100, le=2000)
    overlap: int = Field(default=50, ge=0, le=500)
    retrieval_method: str = Field(default="hybrid_rrf")
    top_k: int = Field(default=5, ge=1, le=50)
    reranking: bool = Field(default=True)
    embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    generation_model: str = Field(default="gpt-4o-mini")
    query_rewrite: bool = Field(default=False)
    notes: str | None = None


class ExperimentCreateRequest(BaseModel):
    name: str = Field(min_length=3, max_length=200)
    dataset_id: str
    benchmark_id: str | None = None
    config: ExperimentConfig = Field(default_factory=ExperimentConfig)


class ResultRow(BaseModel):
    question: str
    recall_at_5: float
    precision_at_5: float
    mrr: float
    latency_ms: int
    token_cost: float
    answer_quality: float


class ExperimentResult(BaseModel):
    experiment_id: str
    status: str
    metrics: dict[str, float]
    rows: list[ResultRow] = Field(default_factory=list)
    generated_at: str


class ExperimentRecord(BaseModel):
    id: str
    name: str
    dataset_id: str
    benchmark_id: str | None
    config: ExperimentConfig
    status: str
    created_at: str
    results: list[ResultRow] = Field(default_factory=list)
    summary: dict[str, float] = Field(default_factory=dict)


class ComparisonRequest(BaseModel):
    experiment_ids: list[str] = Field(default_factory=list)


class ComparisonResult(BaseModel):
    experiment_ids: list[str]
    winners: dict[str, str]
    best_by_recall: str | None = None
    best_by_quality: str | None = None
    summary: dict[str, dict[str, float]] = Field(default_factory=dict)
