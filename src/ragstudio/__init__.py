from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path

from src.ragstudio.experiments.models import (
    BenchmarkCreateRequest,
    DatasetCreateRequest,
    ExperimentCreateRequest,
)
from src.ragstudio.experiments.service import ExperimentService

BASE_DIR = Path(__file__).resolve().parent
app = FastAPI(
    title="RAG Optimization Studio",
    version="0.4.0",
    description="Single-server RAG benchmarking platform.",
)

app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

service = ExperimentService()


@app.get("/")
async def dashboard(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "project_name": "RAG Optimization Studio"},
    )


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "service": "rag-optimization-studio",
        "single_server_mode": True,
        "version": "0.4.0",
    }


@app.get("/api/overview")
async def overview():
    return {
        "title": "RAG Optimization Studio",
        "summary": "Experimentation and benchmarking platform for retrieval-augmented generation pipelines.",
        "mvp_status": "in_progress",
        "single_server": True,
        "focus_areas": [
            "dataset ingestion",
            "benchmark design and relevance labels",
            "chunking and indexing",
            "dense + BM25 + hybrid retrieval",
            "reranking and evaluation",
            "result comparison and reproducibility",
        ],
        "key_metrics": [
            "Recall@k",
            "Precision@k",
            "MRR",
            "nDCG@k",
            "Latency",
            "Token cost",
        ],
        "stats": service.get_dashboard_summary(),
    }


@app.get("/api/datasets")
async def list_datasets():
    return {"datasets": service.list_datasets()}


@app.post("/api/datasets")
async def create_dataset(payload: DatasetCreateRequest):
    return {"dataset": service.create_dataset(payload)}


@app.get("/api/benchmarks")
async def list_benchmarks():
    return {"benchmarks": service.list_benchmarks()}


@app.post("/api/benchmarks")
async def create_benchmark(payload: BenchmarkCreateRequest):
    return {"benchmark": service.create_benchmark(payload)}


@app.get("/api/experiments")
async def list_experiments():
    return {"experiments": service.list_experiments()}


@app.post("/api/experiments")
async def create_experiment(payload: ExperimentCreateRequest):
    return {"experiment": service.create_experiment(payload)}


@app.get("/api/experiments/{experiment_id}")
async def get_experiment(experiment_id: str):
    experiment = service.get_experiment(experiment_id)
    if experiment is None:
        return {"detail": "Experiment not found"}
    return experiment


@app.post("/api/experiments/{experiment_id}/run")
async def run_experiment(experiment_id: str):
    result = service.run_experiment(experiment_id)
    if result is None:
        return {"detail": "Experiment not found"}
    return {"experiment": result}


@app.get("/api/experiments/{experiment_id}/results")
async def get_results(experiment_id: str):
    result = service.get_experiment_results(experiment_id)
    if result is None:
        return {"detail": "Experiment not found"}
    return {"results": result}


@app.post("/api/comparisons")
async def compare_experiments(payload: dict):
    experiment_ids = payload.get("experiment_ids", [])
    return {"comparison": service.compare_experiments(experiment_ids)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
