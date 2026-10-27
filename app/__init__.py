from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from uuid import uuid4
from datetime import datetime, timezone
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
app = FastAPI(
    title="RAG Optimization Studio",
    version="0.1.0",
    description="Single-server experimentation platform for benchmarking RAG pipelines.",
)

app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# In-memory experiment store for the MVP prototype.
EXPERIMENTS: dict[str, dict] = {}


class ExperimentConfig(BaseModel):
    chunking_strategy: str = Field(default="fixed-size", description="Chunking method used for the corpus")
    chunk_size: int = Field(default=500, ge=100, le=2000)
    overlap: int = Field(default=50, ge=0, le=500)
    retrieval_method: str = Field(default="hybrid_rrf", description="dense, bm25, or hybrid_rrf")
    top_k: int = Field(default=5, ge=1, le=50)
    reranking: bool = Field(default=True)
    embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    generation_model: str = Field(default="gpt-4o-mini")
    query_rewrite: bool = Field(default=False)
    notes: str | None = None


class ExperimentCreateRequest(BaseModel):
    name: str = Field(min_length=3)
    dataset_name: str = Field(default="technical-docs")
    config: ExperimentConfig = Field(default_factory=ExperimentConfig)


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
        "timestamp": datetime.now(timezone.utc).isoformat(),
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
    }


@app.get("/api/experiments")
async def list_experiments():
    return {
        "experiments": [
            {
                "id": exp_id,
                **payload,
            }
            for exp_id, payload in EXPERIMENTS.items()
        ]
    }


@app.post("/api/experiments")
async def create_experiment(payload: ExperimentCreateRequest):
    experiment_id = f"exp-{uuid4().hex[:8]}"
    now = datetime.now(timezone.utc).isoformat()

    EXPERIMENTS[experiment_id] = {
        "id": experiment_id,
        "name": payload.name,
        "dataset_name": payload.dataset_name,
        "config": payload.config.model_dump(),
        "status": "created",
        "created_at": now,
    }

    return {
        "id": experiment_id,
        "status": "created",
        "created_at": now,
        "message": "Experiment configuration saved and ready to run.",
        "config": payload.config.model_dump(),
    }


@app.get("/api/experiments/{experiment_id}")
async def get_experiment(experiment_id: str):
    if experiment_id not in EXPERIMENTS:
        return {"detail": "Experiment not found"}
    return EXPERIMENTS[experiment_id]


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
