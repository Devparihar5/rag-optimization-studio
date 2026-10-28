from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from starlette.requests import Request

# ---------------------------------------------------------------------------
# App instance
# ---------------------------------------------------------------------------

app = FastAPI(
    title="RAG Optimization Studio",
    description="Benchmarking platform for RAG pipeline experiments",
    version="0.1.0",
)

# ---------------------------------------------------------------------------
# Static files + templates
# ---------------------------------------------------------------------------

_BASE = Path(__file__).parent
app.mount("/static", StaticFiles(directory=_BASE / "static"), name="static")
templates = Jinja2Templates(directory=str(_BASE / "templates"))

# ---------------------------------------------------------------------------
# In-memory state
# ---------------------------------------------------------------------------

_datasets: dict[str, dict[str, Any]] = {}
_benchmarks: dict[str, dict[str, Any]] = {}
_experiments: dict[str, dict[str, Any]] = {}

# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


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
    summary: dict[str, float] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def dashboard(request: Request) -> HTMLResponse:
    # Starlette >= 1.10 uses TemplateResponse(request, name, context)
    return templates.TemplateResponse(request, "index.html")


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


@app.get("/api/health", tags=["system"])
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/overview", tags=["system"])
async def overview() -> dict[str, Any]:
    return {
        "datasets": len(_datasets),
        "benchmarks": len(_benchmarks),
        "experiments": len(_experiments),
        "running": sum(1 for e in _experiments.values() if e["status"] == "running"),
    }


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------


@app.post("/api/datasets", response_model=DatasetRecord, status_code=201, tags=["datasets"])
async def create_dataset(body: DatasetCreateRequest) -> DatasetRecord:
    record = DatasetRecord(
        id=str(uuid.uuid4()),
        name=body.name,
        description=body.description,
        source_type=body.source_type,
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    _datasets[record.id] = record.model_dump()
    return record


@app.get("/api/datasets", response_model=list[DatasetRecord], tags=["datasets"])
async def list_datasets() -> list[DatasetRecord]:
    return [DatasetRecord(**d) for d in _datasets.values()]


@app.get("/api/datasets/{dataset_id}", response_model=DatasetRecord, tags=["datasets"])
async def get_dataset(dataset_id: str) -> DatasetRecord:
    if dataset_id not in _datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return DatasetRecord(**_datasets[dataset_id])


@app.delete("/api/datasets/{dataset_id}", status_code=204, tags=["datasets"])
async def delete_dataset(dataset_id: str) -> None:
    if dataset_id not in _datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")
    del _datasets[dataset_id]


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------


@app.post("/api/benchmarks", response_model=BenchmarkRecord, status_code=201, tags=["benchmarks"])
async def create_benchmark(body: BenchmarkCreateRequest) -> BenchmarkRecord:
    if body.dataset_id not in _datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")
    record = BenchmarkRecord(
        id=str(uuid.uuid4()),
        name=body.name,
        dataset_id=body.dataset_id,
        question_count=len(body.questions),
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    _benchmarks[record.id] = record.model_dump()
    return record


@app.get("/api/benchmarks", response_model=list[BenchmarkRecord], tags=["benchmarks"])
async def list_benchmarks() -> list[BenchmarkRecord]:
    return [BenchmarkRecord(**b) for b in _benchmarks.values()]


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------


@app.post("/api/experiments", response_model=ExperimentRecord, status_code=201, tags=["experiments"])
async def create_experiment(body: ExperimentCreateRequest) -> ExperimentRecord:
    if body.dataset_id not in _datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")
    if body.benchmark_id and body.benchmark_id not in _benchmarks:
        raise HTTPException(status_code=404, detail="Benchmark not found")

    record = ExperimentRecord(
        id=str(uuid.uuid4()),
        name=body.name,
        dataset_id=body.dataset_id,
        benchmark_id=body.benchmark_id,
        config=body.config,
        status="pending",
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    _experiments[record.id] = record.model_dump()
    return record


@app.delete("/api/benchmarks/{benchmark_id}", status_code=204, tags=["benchmarks"])
async def delete_benchmark(benchmark_id: str) -> None:
    # Experiments keep their configuration and become unlinked from this benchmark.
    with connect() as db:
        cursor = db.execute("DELETE FROM benchmarks WHERE id=?", (benchmark_id,))
        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="Benchmark not found")


@app.get("/api/experiments", response_model=list[ExperimentRecord], tags=["experiments"])
async def list_experiments() -> list[ExperimentRecord]:
    return [ExperimentRecord(**e) for e in _experiments.values()]


@app.get("/api/experiments/{experiment_id}", response_model=ExperimentRecord, tags=["experiments"])
async def get_experiment(experiment_id: str) -> ExperimentRecord:
    if experiment_id not in _experiments:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return ExperimentRecord(**_experiments[experiment_id])


@app.delete("/api/experiments/{experiment_id}", status_code=204, tags=["experiments"])
async def delete_experiment(experiment_id: str) -> None:
    if experiment_id not in _experiments:
        raise HTTPException(status_code=404, detail="Experiment not found")
    del _experiments[experiment_id]


__all__ = ["app"]
