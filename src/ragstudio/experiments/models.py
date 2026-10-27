from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from src.ragstudio.evaluation.metrics import compute_metrics
from src.ragstudio.retrieval.engine import DocumentChunk, RetrievalEngine

from .models import (
    BenchmarkCreateRequest,
    BenchmarkQuestion,
    BenchmarkRecord,
    DatasetCreateRequest,
    DatasetRecord,
    ExperimentConfig,
    ExperimentCreateRequest,
    ExperimentRecord,
    RetrievalRow,
)


class ExperimentService:
    def __init__(self) -> None:
        self._datasets: dict[str, DatasetRecord] = {}
        self._benchmarks: dict[str, BenchmarkRecord] = {}
        self._experiments: dict[str, ExperimentRecord] = {}
        self._retrieval_engine = RetrievalEngine(self._demo_chunks())
        self._seed_demo_data()

    @staticmethod
    def _demo_chunks() -> list[DocumentChunk]:
        return [
            DocumentChunk(
                id="chunk-1",
                text="Retrieval fallback should trigger when top-k answers are weak and the system should expand the candidate set with query rewriting.",
                source="architecture.md",
                page=4,
            ),
            DocumentChunk(
                id="chunk-2",
                text="Hybrid retrieval combines lexical BM25 matching and semantic similarity to improve retrieval quality on long technical documents.",
                source="retrieval.md",
                page=8,
            ),
            DocumentChunk(
                id="chunk-3",
                text="Chunking with sentence-aware boundaries and 50-token overlaps preserves context and helps reranking choose better evidence passages.",
                source="chunking.md",
                page=11,
            ),
            DocumentChunk(
                id="chunk-4",
                text="Reranking uses cross-encoders to reorder candidates and improve fidelity for answer generation with lower irrelevant passage rates.",
                source="ranking.md",
                page=5,
            ),
            DocumentChunk(
                id="chunk-5",
                text="Track latency and token cost separately from retrieval quality so teams can compare quality-to-cost trade-offs across pipelines.",
                source="evaluation.md",
                page=17,
            ),
        ]

    def _seed_demo_data(self) -> None:
        dataset_id = "dataset-demo"
        self._datasets[dataset_id] = DatasetRecord(
            id=dataset_id,
            name="Technical Docs",
            description="Internal documentation corpus for enterprise support and retrieval benchmarking.",
            source_type="pdf",
            created_at=self._now_iso(),
            document_count=5,
        )

        benchmark_id = "benchmark-demo"
        self._benchmarks[benchmark_id] = BenchmarkRecord(
            id=benchmark_id,
            name="Support Questions",
            dataset_id=dataset_id,
            question_count=3,
            created_at=self._now_iso(),
        )

        self._experiments["exp-demo"] = ExperimentRecord(
            id="exp-demo",
            name="Hybrid baseline",
            dataset_id=dataset_id,
            benchmark_id=benchmark_id,
            config=ExperimentConfig(
                chunking_strategy="fixed-size",
                chunk_size=500,
                overlap=50,
                retrieval_method="hybrid",
                top_k=3,
                reranking=True,
                embedding_model="all-MiniLM-L6-v2",
                generation_model="gpt-4o-mini",
                query_rewrite=False,
                notes="Baseline hybrid configuration",
            ),
            status="completed",
            created_at=self._now_iso(),
            results=[],
            summary={},
        )

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    def get_dashboard_summary(self) -> dict[str, Any]:
        return {
            "datasets": len(self._datasets),
            "benchmarks": len(self._benchmarks),
            "experiments": len(self._experiments),
            "active_experiments": sum(1 for experiment in self._experiments.values() if experiment.status == "running"),
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
            results=[],
            summary={},
        )
        self._experiments[experiment_id] = experiment
        return experiment.model_dump()

    def get_experiment(self, experiment_id: str) -> dict[str, Any] | None:
        experiment = self._experiments.get(experiment_id)
        return None if experiment is None else experiment.model_dump()

    def run_experiment(self, experiment_id: str) -> dict[str, Any] | None:
        experiment = self._experiments.get(experiment_id)
        if experiment is None:
            return None

        benchmark = self._benchmarks.get(experiment.benchmark_id or "benchmark-demo")
        if benchmark is None:
            benchmark = next(iter(self._benchmarks.values()), None)
        if benchmark is None:
            return experiment.model_dump()

        questions = [
            BenchmarkQuestion(
                question="How do we design retrieval fallback when the top matches are weak?",
                answer="Use fallback query expansion and broaden matching when retrieval confidence is low.",
                relevant_passages=["chunk-1", "chunk-2"],
            ),
            BenchmarkQuestion(
                question="Which configuration should we prefer for long technical documents?",
                answer="Hybrid retrieval with overlap and reranking usually performs best for long technical documents.",
                relevant_passages=["chunk-2", "chunk-3"],
            ),
            BenchmarkQuestion(
                question="How should we measure the quality-to-cost trade-off?",
                answer="Track retrieval quality, latency, and token cost separately and compare results across experiments.",
                relevant_passages=["chunk-4", "chunk-5"],
            ),
        ]

        rows: list[RetrievalRow] = []
        for item in questions:
            retrieval = self._retrieval_engine.retrieve(
                item.question,
                top_k=experiment.config.top_k,
                strategy=experiment.config.retrieval_method,
            )
            retrieved_ids = [entry["chunk_id"] for entry in retrieval]
            relevant_ids = item.relevant_passages
            metrics = compute_metrics(relevant_ids, retrieved_ids, k=experiment.config.top_k)
            rows.append(
                RetrievalRow(
                    question=item.question,
                    retrieved_chunks=retrieved_ids,
                    recall_at_k=metrics["recall_at_k"],
                    precision_at_k=metrics["precision_at_k"],
                    mrr=metrics["mrr"],
                    ndcg_at_k=metrics["ndcg_at_k"],
                    latency_ms=420 + len(retrieved_ids) * 15,
                    token_cost=0.004 + (len(retrieved_ids) * 0.002),
                )
            )

        experiment.status = "completed"
        experiment.results = rows
        experiment.summary = {
            "recall_at_k": round(sum(row.recall_at_k for row in rows) / max(len(rows), 1), 3),
            "precision_at_k": round(sum(row.precision_at_k for row in rows) / max(len(rows), 1), 3),
            "mrr": round(sum(row.mrr for row in rows) / max(len(rows), 1), 3),
            "ndcg_at_k": round(sum(row.ndcg_at_k for row in rows) / max(len(rows), 1), 3),
            "latency_ms": round(sum(row.latency_ms for row in rows) / max(len(rows), 1), 1),
            "token_cost": round(sum(row.token_cost for row in rows) / max(len(rows), 1), 4),
        }
        return experiment.model_dump()

    def get_experiment_results(self, experiment_id: str) -> list[dict[str, Any]] | None:
        experiment = self._experiments.get(experiment_id)
        if experiment is None:
            return None
        return [row.model_dump() for row in experiment.results]

    def compare_experiments(self, experiment_ids: list[str]) -> dict[str, Any]:
        valid_ids = [experiment_id for experiment_id in experiment_ids if experiment_id in self._experiments]
        if not valid_ids:
            return {"experiment_ids": [], "best_by_recall": None, "best_by_quality": None, "summary": {}}

        summary = {
            experiment_id: self._experiments[experiment_id].summary
            for experiment_id in valid_ids
            if self._experiments[experiment_id].summary
        }
        best_by_recall = max(valid_ids, key=lambda experiment_id: summary.get(experiment_id, {}).get("recall_at_k", 0), default=None)
        best_by_quality = max(valid_ids, key=lambda experiment_id: summary.get(experiment_id, {}).get("ndcg_at_k", 0), default=None)

        return {
            "experiment_ids": valid_ids,
            "best_by_recall": best_by_recall,
            "best_by_quality": best_by_quality,
            "summary": summary,
        }
