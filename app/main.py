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


class ExperimentRecord(BaseModel):
    id: str
    name: str
    dataset_id: str
    benchmark_id: str | None
    config: ExperimentConfig
    status: str
    created_at: str


class ExperimentService:
    def __init__(self) -> None:
        self._datasets: dict[str, DatasetRecord] = {}
        self._benchmarks: dict[str, BenchmarkRecord] = {}
        self._experiments: dict[str, ExperimentRecord] = {}
        self._seed_demo_data()

    def _seed_demo_data(self) -> None:
        dataset_id = "dataset-demo"
        self._datasets[dataset_id] = DatasetRecord(
            id=dataset_id,
            name="Technical Docs",
            description="Internal documentation corpus for an enterprise knowledge base.",
            source_type="pdf",
            created_at=self._now_iso(),
            document_count=18,
        )

        benchmark_id = "benchmark-demo"
        self._benchmarks[benchmark_id] = BenchmarkRecord(
            id=benchmark_id,
            name="Support Questions",
            dataset_id=dataset_id,
            question_count=3,
            created_at=self._now_iso(),
        )

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    def get_dashboard_summary(self) -> dict[str, Any]:
        return {
            "datasets": len(self._datasets),
            "benchmarks": len(self._benchmarks),
            "experiments": len(self._experiments),
            "active_experiments": sum(1 for e in self._experiments.values() if e.status == "running"),
        }

    def list_datasets(self) -> list[dict[str, Any]]:
        return [dataset.model_dump() for dataset in self._datasets.values()]

    def create_dataset(self, payload: DatasetCreateRequest) -> dict[str, Any]:
        dataset_id = f"dataset-{uuid4().hex[:8]}"
        dataset = DatasetRecord(
            id=dataset_id,
            name=payload.name,
            description=payload.description,
            source_type=payload.source_type,
            created_at=self._now_iso(),
            document_count=0,
        )
        self._datasets[dataset_id] = dataset
        return dataset.model_dump()

    def list_benchmarks(self) -> list[dict[str, Any]]:
        return [benchmark.model_dump() for benchmark in self._benchmarks.values()]

    def create_benchmark(self, payload: BenchmarkCreateRequest) -> dict[str, Any]:
        benchmark_id = f"benchmark-{uuid4().hex[:8]}"
        benchmark = BenchmarkRecord(
            id=benchmark_id,
            name=payload.name,
            dataset_id=payload.dataset_id,
            question_count=len(payload.questions),
            created_at=self._now_iso(),
        )
        self._benchmarks[benchmark_id] = benchmark
        return benchmark.model_dump()

    def list_experiments(self) -> list[dict[str, Any]]:
        return [experiment.model_dump() for experiment in self._experiments.values()]

    def create_experiment(self, payload: ExperimentCreateRequest) -> dict[str, Any]:
        experiment_id = f"exp-{uuid4().hex[:8]}"
        experiment = ExperimentRecord(
            id=experiment_id,
            name=payload.name,
            dataset_id=payload.dataset_id,
            benchmark_id=payload.benchmark_id,
            config=payload.config,
            status="created",
            created_at=self._now_iso(),
        )
        self._experiments[experiment_id] = experiment
        return experiment.model_dump()

    def get_experiment(self, experiment_id: str) -> dict[str, Any] | None:
        experiment = self._experiments.get(experiment_id)
        return None if experiment is None else experiment.model_dump()
