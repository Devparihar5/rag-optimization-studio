from __future__ import annotations

import io
import json
import math
import re
import time
import uuid
from collections import Counter
from datetime import datetime, timezone
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field, model_validator
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request

try:
    from ragstudio.storage import connect, data_dir
    from ragstudio.model_registry import (
        router as model_router,
        get_model_runtime_config,
        get_embedding_model_runtime_config,
    )
except ModuleNotFoundError:
    # Supports running directly from a source checkout without an editable install.
    from src.ragstudio.storage import connect, data_dir
    from src.ragstudio.model_registry import (
        router as model_router,
        get_model_runtime_config,
        get_embedding_model_runtime_config,
    )

app = FastAPI(
    title="RAG Optimization Studio",
    description="Benchmarking platform for RAG pipeline experiments",
    version="0.2.0",
)
app.include_router(model_router)

_BASE = Path(__file__).parent
app.mount("/static", StaticFiles(directory=_BASE / "static"), name="static")
templates = Jinja2Templates(directory=str(_BASE / "templates"))

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
SUPPORTED_EXTENSIONS = {".pdf", ".md", ".txt", ".html", ".htm"}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_load(value: str, fallback: Any) -> Any:
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return fallback


def dataset_record(row: Any) -> dict[str, Any]:
    result = dict(row)
    result["document_count"] = int(result.pop("document_count", 0))
    return result


def get_dataset_row(db: Any, dataset_id: str) -> Any:
    return db.execute(
        """SELECT d.*, (SELECT COUNT(*) FROM documents doc WHERE doc.dataset_id=d.id) AS document_count
           FROM datasets d WHERE d.id=?""",
        (dataset_id,),
    ).fetchone()


def extract_text(filename: str, content: bytes) -> str:
    suffix = Path(filename).suffix.lower()
    decoded = content.decode("utf-8", errors="replace")
    if suffix in {".txt", ".md"}:
        return decoded
    if suffix in {".html", ".htm"}:
        class TextExtractor(HTMLParser):
            def __init__(self) -> None:
                super().__init__()
                self.parts: list[str] = []
                self.skip_depth = 0

            def handle_starttag(self, tag: str, attrs: Any) -> None:
                if tag.lower() in {"script", "style", "noscript"}:
                    self.skip_depth += 1

            def handle_endtag(self, tag: str) -> None:
                if tag.lower() in {"script", "style", "noscript"} and self.skip_depth:
                    self.skip_depth -= 1

            def handle_data(self, data: str) -> None:
                if not self.skip_depth and data.strip():
                    self.parts.append(data.strip())

        parser = TextExtractor()
        parser.feed(decoded)
        return "\n".join(parser.parts)
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise ValueError("PDF support is unavailable. Install the pypdf dependency.") from exc
        try:
            reader = PdfReader(io.BytesIO(content))
            return "\n".join(page.extract_text() or "" for page in reader.pages).strip()
        except Exception as exc:
            raise ValueError("Could not read this PDF. Check that the file is a valid PDF.") from exc
    raise ValueError("Unsupported file type. Upload PDF, Markdown, TXT, or HTML files.")


class DatasetCreateRequest(BaseModel):
    name: str = Field(min_length=3, max_length=200)
    description: str = Field(default="", max_length=2000)
    source_type: str = Field(default="pdf")


class DatasetRecord(BaseModel):
    id: str
    name: str
    description: str
    source_type: str
    created_at: str
    document_count: int = 0


class DocumentRecord(BaseModel):
    id: str
    dataset_id: str
    filename: str
    media_type: str
    size_bytes: int
    created_at: str


class BenchmarkQuestion(BaseModel):
    question: str = Field(min_length=1, max_length=5000)
    answer: str | None = None
    relevant_passages: list[str] = Field(default_factory=list)


class BenchmarkCreateRequest(BaseModel):
    name: str = Field(min_length=3, max_length=200)
    dataset_id: str
    questions: list[BenchmarkQuestion] = Field(min_length=1, max_length=1000)


class BenchmarkUpdateRequest(BaseModel):
    name: str = Field(min_length=3, max_length=200)
    questions: list[BenchmarkQuestion] = Field(min_length=1, max_length=1000)


class ExperimentCompareRequest(BaseModel):
    experiment_ids: list[str] = Field(min_length=2, max_length=10)


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
    # These options are deliberately off until real reranking/rewrite providers are wired.
    reranking: bool = Field(default=False)
    embedding_model: str = Field(min_length=1, max_length=300)
    generation_model: str = Field(min_length=1, max_length=300)
    query_rewrite: bool = Field(default=False)
    notes: str | None = None

    @model_validator(mode="after")
    def validate_supported_options(self) -> "ExperimentConfig":
        if self.chunking_strategy not in {"fixed-size", "sentence-aware", "recursive"}:
            raise ValueError("Unsupported chunking strategy.")
        if self.retrieval_method not in {"dense", "bm25", "hybrid_rrf"}:
            raise ValueError("Unsupported retrieval method.")
        if self.overlap >= self.chunk_size:
            raise ValueError("Overlap must be smaller than chunk size.")
        if self.reranking:
            raise ValueError("Reranking is not implemented yet. Turn it off before saving this experiment.")
        if self.query_rewrite:
            raise ValueError("Query rewriting is not implemented yet. Turn it off before saving this experiment.")
        return self


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
    started_at: str | None = None
    completed_at: str | None = None
    error_message: str | None = None
    summary: dict[str, Any] = Field(default_factory=dict)


def benchmark_record(row: Any) -> dict[str, Any]:
    questions = json_load(row["questions_json"], [])
    return {
        "id": row["id"], "name": row["name"], "dataset_id": row["dataset_id"],
        "question_count": len(questions), "created_at": row["created_at"],
    }


def experiment_record(row: Any) -> dict[str, Any]:
    return {
        "id": row["id"], "name": row["name"], "dataset_id": row["dataset_id"],
        "benchmark_id": row["benchmark_id"],
        "config": json_load(row["config_json"], {}),
        "status": row["status"], "created_at": row["created_at"],
        "started_at": row["started_at"] if "started_at" in row.keys() else None,
        "completed_at": row["completed_at"] if "completed_at" in row.keys() else None,
        "error_message": row["error_message"] if "error_message" in row.keys() else None,
        "summary": json_load(row["summary_json"], {}),
    }


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def dashboard(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "index.html")


@app.get("/api/health", tags=["system"])
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/overview", tags=["system"])
async def overview() -> dict[str, int]:
    with connect() as db:
        return {
            "datasets": db.execute("SELECT COUNT(*) FROM datasets").fetchone()[0],
            "benchmarks": db.execute("SELECT COUNT(*) FROM benchmarks").fetchone()[0],
            "experiments": db.execute("SELECT COUNT(*) FROM experiments").fetchone()[0],
            "running": db.execute("SELECT COUNT(*) FROM experiments WHERE status='running'").fetchone()[0],
        }


@app.post("/api/datasets", response_model=DatasetRecord, status_code=201, tags=["datasets"])
async def create_dataset(body: DatasetCreateRequest) -> dict[str, Any]:
    record = {
        "id": str(uuid.uuid4()), "name": body.name, "description": body.description,
        "source_type": body.source_type, "created_at": now_iso(), "document_count": 0,
    }
    with connect() as db:
        db.execute(
            "INSERT INTO datasets(id,name,description,source_type,created_at) VALUES(?,?,?,?,?)",
            (record["id"], record["name"], record["description"], record["source_type"], record["created_at"]),
        )
    return record


@app.get("/api/datasets", response_model=list[DatasetRecord], tags=["datasets"])
async def list_datasets() -> list[dict[str, Any]]:
    with connect() as db:
        rows = db.execute(
            """SELECT d.*, (SELECT COUNT(*) FROM documents doc WHERE doc.dataset_id=d.id) AS document_count
               FROM datasets d ORDER BY d.created_at DESC"""
        ).fetchall()
    return [dataset_record(row) for row in rows]


@app.get("/api/datasets/{dataset_id}", response_model=DatasetRecord, tags=["datasets"])
async def get_dataset(dataset_id: str) -> dict[str, Any]:
    with connect() as db:
        row = get_dataset_row(db, dataset_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return dataset_record(row)


@app.get("/api/datasets/{dataset_id}/documents", response_model=list[DocumentRecord], tags=["datasets"])
async def list_documents(dataset_id: str) -> list[dict[str, Any]]:
    with connect() as db:
        if db.execute("SELECT 1 FROM datasets WHERE id=?", (dataset_id,)).fetchone() is None:
            raise HTTPException(status_code=404, detail="Dataset not found")
        rows = db.execute(
            """SELECT id,dataset_id,filename,media_type,size_bytes,created_at
               FROM documents WHERE dataset_id=? ORDER BY created_at DESC""",
            (dataset_id,),
        ).fetchall()
    return [dict(row) for row in rows]


@app.post("/api/datasets/{dataset_id}/documents", tags=["datasets"])
async def upload_documents(dataset_id: str, files: list[UploadFile] = File(...)) -> dict[str, Any]:
    if not files:
        raise HTTPException(status_code=400, detail="Choose at least one file to upload.")
    with connect() as db:
        if db.execute("SELECT 1 FROM datasets WHERE id=?", (dataset_id,)).fetchone() is None:
            raise HTTPException(status_code=404, detail="Dataset not found")

    saved: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    target_dir = data_dir() / "documents" / dataset_id
    for upload in files:
        filename = Path(upload.filename or "").name
        suffix = Path(filename).suffix.lower()
        if not filename or suffix not in SUPPORTED_EXTENSIONS:
            failures.append({"filename": filename or "unnamed file", "error": "Unsupported file type. Use PDF, MD, TXT, or HTML."})
            continue
        content = await upload.read(MAX_UPLOAD_BYTES + 1)
        if len(content) > MAX_UPLOAD_BYTES:
            failures.append({"filename": filename, "error": "File exceeds the 10 MB limit."})
            continue
        try:
            extracted = extract_text(filename, content)
        except ValueError as exc:
            failures.append({"filename": filename, "error": str(exc)})
            continue

        document_id = str(uuid.uuid4())
        safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", filename).strip("._") or "document"
        stored_path = target_dir / f"{document_id}_{safe_name}"
        target_dir.mkdir(parents=True, exist_ok=True)
        stored_path.write_bytes(content)
        record = {
            "id": document_id, "dataset_id": dataset_id, "filename": filename,
            "media_type": upload.content_type or "application/octet-stream",
            "size_bytes": len(content), "stored_path": str(stored_path),
            "extracted_text": extracted, "created_at": now_iso(),
        }
        try:
            with connect() as db:
                db.execute(
                    """INSERT INTO documents
                       (id,dataset_id,filename,media_type,size_bytes,stored_path,extracted_text,created_at)
                       VALUES(?,?,?,?,?,?,?,?)""",
                    tuple(record[key] for key in ("id","dataset_id","filename","media_type","size_bytes","stored_path","extracted_text","created_at")),
                )
        except Exception:
            stored_path.unlink(missing_ok=True)
            raise
        saved.append({key: record[key] for key in ("id","dataset_id","filename","media_type","size_bytes","created_at")})
    return {"uploaded": saved, "failed": failures, "document_count": len(saved)}


@app.delete("/api/datasets/{dataset_id}/documents/{document_id}", status_code=204, tags=["datasets"])
async def delete_document(dataset_id: str, document_id: str) -> None:
    with connect() as db:
        row = db.execute(
            "SELECT stored_path FROM documents WHERE id=? AND dataset_id=?", (document_id, dataset_id)
        ).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="Document not found")
        path = Path(row["stored_path"])
        db.execute("DELETE FROM documents WHERE id=? AND dataset_id=?", (document_id, dataset_id))
    path.unlink(missing_ok=True)


@app.delete("/api/datasets/{dataset_id}", status_code=204, tags=["datasets"])
async def delete_dataset(dataset_id: str) -> None:
    with connect() as db:
        if db.execute("SELECT 1 FROM datasets WHERE id=?", (dataset_id,)).fetchone() is None:
            raise HTTPException(status_code=404, detail="Dataset not found")
        paths = [Path(row["stored_path"]) for row in db.execute(
            "SELECT stored_path FROM documents WHERE dataset_id=?", (dataset_id,)
        ).fetchall()]
        db.execute("DELETE FROM datasets WHERE id=?", (dataset_id,))
    for path in paths:
        path.unlink(missing_ok=True)
    folder = data_dir() / "documents" / dataset_id
    if folder.exists():
        try:
            folder.rmdir()
        except OSError:
            pass


@app.post("/api/benchmarks", response_model=BenchmarkRecord, status_code=201, tags=["benchmarks"])
async def create_benchmark(body: BenchmarkCreateRequest) -> dict[str, Any]:
    benchmark_id = str(uuid.uuid4())
    created_at = now_iso()
    questions = [question.model_dump() for question in body.questions]
    with connect() as db:
        if db.execute("SELECT 1 FROM datasets WHERE id=?", (body.dataset_id,)).fetchone() is None:
            raise HTTPException(status_code=404, detail="Dataset not found")
        db.execute(
            "INSERT INTO benchmarks(id,name,dataset_id,questions_json,created_at) VALUES(?,?,?,?,?)",
            (benchmark_id, body.name, body.dataset_id, json.dumps(questions), created_at),
        )
    return {"id": benchmark_id, "name": body.name, "dataset_id": body.dataset_id,
            "question_count": len(questions), "created_at": created_at}


@app.get("/api/benchmarks", response_model=list[BenchmarkRecord], tags=["benchmarks"])
async def list_benchmarks() -> list[dict[str, Any]]:
    with connect() as db:
        rows = db.execute("SELECT * FROM benchmarks ORDER BY created_at DESC").fetchall()
    return [benchmark_record(row) for row in rows]


@app.put("/api/benchmarks/{benchmark_id}", response_model=BenchmarkRecord, tags=["benchmarks"])
async def update_benchmark(benchmark_id: str, body: BenchmarkUpdateRequest) -> dict[str, Any]:
    with connect() as db:
        row = db.execute("SELECT * FROM benchmarks WHERE id=?", (benchmark_id,)).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="Benchmark not found")
        questions = [question.model_dump() for question in body.questions]
        db.execute(
            "UPDATE benchmarks SET name=?,questions_json=? WHERE id=?",
            (body.name, json.dumps(questions), benchmark_id),
        )
        updated = db.execute("SELECT * FROM benchmarks WHERE id=?", (benchmark_id,)).fetchone()
    return benchmark_record(updated)


@app.get("/api/benchmarks/{benchmark_id}", tags=["benchmarks"])
async def get_benchmark(benchmark_id: str) -> dict[str, Any]:
    with connect() as db:
        row = db.execute("SELECT * FROM benchmarks WHERE id=?", (benchmark_id,)).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Benchmark not found")
    result = benchmark_record(row)
    result["questions"] = json_load(row["questions_json"], [])
    return result


@app.post("/api/experiments", response_model=ExperimentRecord, status_code=201, tags=["experiments"])
async def create_experiment(body: ExperimentCreateRequest) -> dict[str, Any]:
    experiment_id = str(uuid.uuid4())
    created_at = now_iso()
    with connect() as db:
        if db.execute("SELECT 1 FROM datasets WHERE id=?", (body.dataset_id,)).fetchone() is None:
            raise HTTPException(status_code=404, detail="Dataset not found")
        # A saved experiment must point to real configured models, never raw model names.
        get_model_runtime_config(body.config.generation_model)
        get_embedding_model_runtime_config(body.config.embedding_model)
        if body.config.overlap >= body.config.chunk_size:
            raise HTTPException(status_code=422, detail="Overlap must be smaller than chunk size.")
        if body.benchmark_id:
            benchmark = db.execute(
                "SELECT dataset_id FROM benchmarks WHERE id=?", (body.benchmark_id,)
            ).fetchone()
            if benchmark is None:
                raise HTTPException(status_code=404, detail="Benchmark not found")
            if benchmark["dataset_id"] != body.dataset_id:
                raise HTTPException(status_code=409, detail="Benchmark must belong to the selected dataset")
        db.execute(
            """INSERT INTO experiments
               (id,name,dataset_id,benchmark_id,config_json,status,created_at,summary_json)
               VALUES(?,?,?,?,?,?,?,?)""",
            (experiment_id, body.name, body.dataset_id, body.benchmark_id,
             body.config.model_dump_json(), "pending", created_at, "{}"),
        )
    return {
        "id": experiment_id, "name": body.name, "dataset_id": body.dataset_id,
        "benchmark_id": body.benchmark_id, "config": body.config.model_dump(),
        "status": "pending", "created_at": created_at, "summary": {},
    }


@app.delete("/api/benchmarks/{benchmark_id}", status_code=204, tags=["benchmarks"])
async def delete_benchmark(benchmark_id: str) -> None:
    # Experiments keep their configuration and become unlinked from this benchmark.
    with connect() as db:
        cursor = db.execute("DELETE FROM benchmarks WHERE id=?", (benchmark_id,))
        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="Benchmark not found")


@app.get("/api/experiments", response_model=list[ExperimentRecord], tags=["experiments"])
async def list_experiments() -> list[dict[str, Any]]:
    with connect() as db:
        rows = db.execute("SELECT * FROM experiments ORDER BY created_at DESC").fetchall()
    return [experiment_record(row) for row in rows]


@app.get("/api/experiments/{experiment_id}", response_model=ExperimentRecord, tags=["experiments"])
async def get_experiment(experiment_id: str) -> dict[str, Any]:
    with connect() as db:
        row = db.execute("SELECT * FROM experiments WHERE id=?", (experiment_id,)).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return experiment_record(row)


@app.delete("/api/experiments/{experiment_id}", status_code=204, tags=["experiments"])
async def delete_experiment(experiment_id: str) -> None:
    with connect() as db:
        cursor = db.execute("DELETE FROM experiments WHERE id=?", (experiment_id,))
        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="Experiment not found")




# ---- Retrieval, provider inference, execution and evaluation -----------------

def _provider_json(url: str, payload: dict[str, Any], headers: dict[str, str] | None = None) -> dict[str, Any]:
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **(headers or {})},
        method="POST",
    )
    try:
        with urlopen(request, timeout=90) as response:
            result = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        # Do not echo request headers or provider secrets into persisted error messages.
        try:
            detail = json.loads(exc.read().decode("utf-8")).get("error", {}).get("message", "")
        except Exception:
            detail = ""
        raise RuntimeError(f"Provider request failed (HTTP {exc.code}). {detail[:400]}".strip()) from exc
    except (URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not reach the configured model provider: {type(exc).__name__}.") from exc
    if not isinstance(result, dict):
        raise RuntimeError("Provider returned an unexpected response.")
    return result


def _embed_texts(runtime: dict[str, Any], texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    provider, model = runtime["provider"], runtime["model_id"]
    if provider == "openai":
        result = _provider_json(
            "https://api.openai.com/v1/embeddings",
            {"model": model, "input": texts},
            {"Authorization": f"Bearer {runtime['api_key']}"},
        )
        return [item["embedding"] for item in sorted(result.get("data", []), key=lambda item: item.get("index", 0))]
    if provider == "gemini":
        vectors = []
        for text_value in texts:
            result = _provider_json(
                f"https://generativelanguage.googleapis.com/v1beta/models/{quote(model, safe='')}:embedContent?{urlencode({'key': runtime['api_key']})}",
                {"content": {"parts": [{"text": text_value}]}},
            )
            vectors.append(result.get("embedding", {}).get("values", []))
        return vectors
    if provider == "ollama":
        base = runtime.get("base_url") or "http://localhost:11434"
        result = _provider_json(f"{base.rstrip('/')}/api/embed", {"model": model, "input": texts})
        vectors = result.get("embeddings")
        if isinstance(vectors, list) and len(vectors) == len(texts):
            return vectors
        # Older Ollama versions expose one prompt per request.
        vectors = []
        for text_value in texts:
            item = _provider_json(f"{base.rstrip('/')}/api/embeddings", {"model": model, "prompt": text_value})
            vectors.append(item.get("embedding", []))
        return vectors
    raise RuntimeError("Unsupported embedding provider.")


def _generate_answer(runtime: dict[str, Any], question: str, evidence: list[dict[str, Any]]) -> tuple[str, dict[str, int]]:
    context = "\n\n".join(
        f"[{index + 1}] {item['text']}" for index, item in enumerate(evidence)
    ) or "(No evidence retrieved.)"
    prompt = (
        "Answer the question using only the evidence below. If the evidence does not contain "
        "the answer, say that the available documents do not provide enough information. "
        "Do not invent facts.\n\nEVIDENCE:\n" + context + "\n\nQUESTION:\n" + question
    )
    provider, model = runtime["provider"], runtime["model_id"]
    usage: dict[str, int] = {}
    if provider == "openai":
        result = _provider_json(
            "https://api.openai.com/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0},
            {"Authorization": f"Bearer {runtime['api_key']}"},
        )
        answer = result.get("choices", [{}])[0].get("message", {}).get("content", "")
        raw_usage = result.get("usage", {})
        for key, source in (("prompt_tokens", "prompt_tokens"), ("completion_tokens", "completion_tokens"), ("total_tokens", "total_tokens")):
            if isinstance(raw_usage.get(source), int):
                usage[key] = raw_usage[source]
        return str(answer).strip(), usage
    if provider == "gemini":
        result = _provider_json(
            f"https://generativelanguage.googleapis.com/v1beta/models/{quote(model, safe='')}:generateContent?{urlencode({'key': runtime['api_key']})}",
            {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"temperature": 0}},
        )
        parts = result.get("candidates", [{}])[0].get("content", {}).get("parts", [])
        answer = "\n".join(part.get("text", "") for part in parts if isinstance(part, dict))
        raw_usage = result.get("usageMetadata", {})
        for key, source in (("prompt_tokens", "promptTokenCount"), ("completion_tokens", "candidatesTokenCount"), ("total_tokens", "totalTokenCount")):
            if isinstance(raw_usage.get(source), int):
                usage[key] = raw_usage[source]
        return answer.strip(), usage
    if provider == "ollama":
        base = runtime.get("base_url") or "http://localhost:11434"
        result = _provider_json(
            f"{base.rstrip('/')}/api/chat",
            {"model": model, "messages": [{"role": "user", "content": prompt}], "stream": False, "options": {"temperature": 0}},
        )
        answer = result.get("message", {}).get("content", "")
        for key, source in (("prompt_tokens", "prompt_eval_count"), ("completion_tokens", "eval_count")):
            if isinstance(result.get(source), int):
                usage[key] = result[source]
        if usage:
            usage["total_tokens"] = usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)
        return str(answer).strip(), usage
    raise RuntimeError("Unsupported generation provider.")


def _tokens(text_value: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text_value.lower())


def _chunk_document(text_value: str, strategy: str, chunk_size: int, overlap: int) -> list[str]:
    words = text_value.split()
    if not words:
        return []
    if strategy == "fixed-size":
        chunks = []
        step = chunk_size - overlap
        for start in range(0, len(words), step):
            part = " ".join(words[start:start + chunk_size])
            if part:
                chunks.append(part)
        return chunks
    if strategy == "sentence-aware":
        units = re.split(r"(?<=[.!?])\s+", text_value.strip())
    elif strategy == "recursive":
        units = [piece.strip() for piece in re.split(r"\n\s*\n|(?<=[.!?])\s+", text_value.strip()) if piece.strip()]
    else:
        raise ValueError("Unsupported chunking strategy.")
    chunks: list[str] = []
    current: list[str] = []
    for unit in units:
        unit_words = unit.split()
        while len(unit_words) > chunk_size:
            if current:
                chunks.append(" ".join(current))
                current = current[-overlap:] if overlap else []
            chunks.append(" ".join(unit_words[:chunk_size]))
            unit_words = unit_words[chunk_size - overlap if overlap else chunk_size:]
        if current and len(current) + len(unit_words) > chunk_size:
            chunks.append(" ".join(current))
            current = current[-overlap:] if overlap else []
        current.extend(unit_words)
    if current:
        chunks.append(" ".join(current))
    # Avoid empty/duplicate tail chunks caused by very short source documents.
    return list(dict.fromkeys(part for part in chunks if part.strip()))


def _cosine(left: list[float], right: list[float]) -> float:
    if not left or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    norm_left = math.sqrt(sum(value * value for value in left))
    norm_right = math.sqrt(sum(value * value for value in right))
    return dot / (norm_left * norm_right) if norm_left and norm_right else 0.0


def _bm25_scores(query: str, texts: list[str]) -> list[float]:
    query_terms = set(_tokens(query))
    tokenized = [_tokens(text_value) for text_value in texts]
    if not texts:
        return []
    avg_length = sum(map(len, tokenized)) / len(tokenized) or 1.0
    doc_frequency = Counter(term for terms in tokenized for term in set(terms))
    scores = []
    for terms in tokenized:
        frequencies = Counter(terms)
        score = 0.0
        for term in query_terms:
            frequency = frequencies[term]
            if not frequency:
                continue
            idf = math.log(1 + (len(texts) - doc_frequency[term] + 0.5) / (doc_frequency[term] + 0.5))
            denominator = frequency + 1.2 * (1 - 0.75 + 0.75 * len(terms) / avg_length)
            score += idf * frequency * 2.2 / denominator
        scores.append(score)
    return scores


def _rank_chunks(query: str, chunks: list[dict[str, Any]], method: str, embedding_runtime: dict[str, Any] | None, top_k: int) -> list[dict[str, Any]]:
    texts = [item["text"] for item in chunks]
    bm25 = _bm25_scores(query, texts) if method in {"bm25", "hybrid_rrf"} else []
    dense: list[float] = []
    if method in {"dense", "hybrid_rrf"}:
        if embedding_runtime is None:
            raise RuntimeError("A configured embedding model is required for dense retrieval.")
        vectors = _embed_texts(embedding_runtime, texts + [query])
        if len(vectors) != len(texts) + 1 or not vectors[-1]:
            raise RuntimeError("The embedding provider returned incomplete vectors.")
        dense = [_cosine(vector, vectors[-1]) for vector in vectors[:-1]]
    if method == "dense":
        scores = dense
    elif method == "bm25":
        scores = bm25
    else:
        dense_order = sorted(range(len(chunks)), key=lambda index: dense[index], reverse=True)
        bm25_order = sorted(range(len(chunks)), key=lambda index: bm25[index], reverse=True)
        fused = [0.0] * len(chunks)
        for order in (dense_order, bm25_order):
            for rank, index in enumerate(order, start=1):
                fused[index] += 1.0 / (60 + rank)
        scores = fused
    order = sorted(range(len(chunks)), key=lambda index: scores[index], reverse=True)[:top_k]
    return [{**chunks[index], "score": float(scores[index]), "rank": rank} for rank, index in enumerate(order, start=1)]


def _is_relevant(chunk_text: str, references: list[str]) -> bool:
    normalized_chunk = " ".join(chunk_text.lower().split())
    return any(
        " ".join(reference.lower().split()) in normalized_chunk
        or normalized_chunk in " ".join(reference.lower().split())
        for reference in references if reference.strip()
    )


def _question_metrics(retrieved: list[dict[str, Any]], all_chunks: list[dict[str, Any]], references: list[str], k: int) -> dict[str, float | None]:
    if not references:
        return {"precision_at_k": None, "recall_at_k": None, "mrr": None, "ndcg_at_k": None}
    relevant_total = sum(_is_relevant(chunk["text"], references) for chunk in all_chunks)
    hits = [1 if _is_relevant(item["text"], references) else 0 for item in retrieved[:k]]
    precision = sum(hits) / k
    recall = sum(hits) / relevant_total if relevant_total else 0.0
    reciprocal_rank = next((1.0 / index for index, hit in enumerate(hits, start=1) if hit), 0.0)
    dcg = sum(hit / math.log2(index + 1) for index, hit in enumerate(hits, start=1))
    ideal_hits = min(k, relevant_total)
    idcg = sum(1.0 / math.log2(index + 1) for index in range(1, ideal_hits + 1))
    return {
        "precision_at_k": precision,
        "recall_at_k": recall,
        "mrr": reciprocal_rank,
        "ndcg_at_k": dcg / idcg if idcg else 0.0,
    }


def _run_experiment_sync(experiment_id: str) -> dict[str, Any]:
    with connect() as db:
        experiment = db.execute("SELECT * FROM experiments WHERE id=?", (experiment_id,)).fetchone()
        if experiment is None:
            raise RuntimeError("Experiment not found.")
        config = json_load(experiment["config_json"], {})
        dataset_id = experiment["dataset_id"]
        benchmark_id = experiment["benchmark_id"]
        documents = db.execute(
            "SELECT id,filename,extracted_text FROM documents WHERE dataset_id=? ORDER BY created_at",
            (dataset_id,),
        ).fetchall()
        benchmark = db.execute("SELECT questions_json FROM benchmarks WHERE id=?", (benchmark_id,)).fetchone() if benchmark_id else None
    if not benchmark:
        raise RuntimeError("Select a benchmark with evaluation questions before running this experiment.")
    questions = json_load(benchmark["questions_json"], [])
    if not questions:
        raise RuntimeError("The selected benchmark contains no questions.")
    if not documents:
        raise RuntimeError("The selected dataset has no uploaded documents.")
    if config.get("reranking") or config.get("query_rewrite"):
        raise RuntimeError("Reranking and query rewriting are not implemented; disable them before running.")
    chunks: list[dict[str, Any]] = []
    for document in documents:
        for index, chunk_text in enumerate(_chunk_document(
            document["extracted_text"], config["chunking_strategy"], config["chunk_size"], config["overlap"]
        )):
            chunks.append({
                "document_id": document["id"], "filename": document["filename"],
                "chunk_index": index, "text": chunk_text,
            })
    if not chunks:
        raise RuntimeError("No text could be chunked from the selected dataset.")
    generation_runtime = get_model_runtime_config(config["generation_model"])
    embedding_runtime = (
        get_embedding_model_runtime_config(config["embedding_model"])
        if config["retrieval_method"] in {"dense", "hybrid_rrf"} else None
    )
    question_results = []
    for question_item in questions:
        question = str(question_item.get("question", "")).strip()
        if not question:
            continue
        started = time.perf_counter()
        retrieved = _rank_chunks(question, chunks, config["retrieval_method"], embedding_runtime, config["top_k"])
        answer, usage = _generate_answer(generation_runtime, question, retrieved)
        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        references = question_item.get("relevant_passages") or []
        metrics = _question_metrics(retrieved, chunks, references, config["top_k"])
        question_results.append({
            "question": question,
            "expected_answer": question_item.get("answer"),
            "generated_answer": answer,
            "retrieved_evidence": retrieved,
            "metrics": metrics,
            "latency_ms": latency_ms,
            "token_usage": usage,
            "cost_usd": None,
        })
    if not question_results:
        raise RuntimeError("The benchmark contains no valid questions.")
    metric_names = ("precision_at_k", "recall_at_k", "mrr", "ndcg_at_k")
    summary: dict[str, Any] = {}
    for metric in metric_names:
        values = [item["metrics"][metric] for item in question_results if item["metrics"][metric] is not None]
        summary[metric] = round(sum(values) / len(values), 4) if values else None
    summary["evaluated_questions"] = len(question_results)
    summary["questions_with_relevance_labels"] = sum(bool(item.get("metrics", {}).get("mrr") is not None) for item in question_results)
    summary["average_latency_ms"] = round(sum(item["latency_ms"] for item in question_results) / len(question_results), 2)
    usage_items = [item["token_usage"] for item in question_results if item["token_usage"]]
    summary["total_tokens"] = sum(item.get("total_tokens", 0) for item in usage_items) if usage_items else None
    summary["cost_usd"] = None  # No provider-specific pricing table is configured.
    completed_at = now_iso()
    result_payload = {
        "experiment_id": experiment_id,
        "dataset_id": dataset_id,
        "benchmark_id": benchmark_id,
        "retrieval_method": config["retrieval_method"],
        "chunk_count": len(chunks),
        "summary": summary,
        "questions": question_results,
        "completed_at": completed_at,
        "metric_definitions": {
            "precision_at_k": "Relevant retrieved chunks divided by configured k; relevance requires a provided relevant_passages string to be contained in the chunk (or vice versa).",
            "recall_at_k": "Relevant retrieved chunks divided by all chunks matching the provided relevant_passages strings.",
            "mrr": "Reciprocal rank of the first relevant retrieved chunk.",
            "ndcg_at_k": "Binary-relevance NDCG at k using the same explicit passage-match rule.",
            "latency_ms": "Measured wall-clock time for retrieval plus generation per question.",
            "cost_usd": "Unavailable because provider pricing is not configured.",
        },
    }
    with connect() as db:
        db.execute(
            "UPDATE experiments SET status='done',completed_at=?,error_message=NULL,summary_json=?,results_json=? WHERE id=?",
            (completed_at, json.dumps(summary), json.dumps(result_payload), experiment_id),
        )
    return result_payload


@app.post("/api/experiments/{experiment_id}/run", tags=["experiments"])
async def run_experiment(experiment_id: str) -> dict[str, Any]:
    with connect() as db:
        row = db.execute("SELECT status,benchmark_id FROM experiments WHERE id=?", (experiment_id,)).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="Experiment not found")
        if row["status"] == "running":
            raise HTTPException(status_code=409, detail="This experiment is already running.")
        if not row["benchmark_id"]:
            raise HTTPException(status_code=409, detail="Select a benchmark with questions before running.")
        db.execute(
            "UPDATE experiments SET status='running',started_at=?,completed_at=NULL,error_message=NULL WHERE id=?",
            (now_iso(), experiment_id),
        )
    try:
        result = await run_in_threadpool(_run_experiment_sync, experiment_id)
        return {"status": "done", "results": result}
    except Exception as exc:
        message = str(exc)[:1000] or "Experiment execution failed."
        with connect() as db:
            db.execute(
                "UPDATE experiments SET status='failed',completed_at=?,error_message=? WHERE id=?",
                (now_iso(), message, experiment_id),
            )
        raise HTTPException(status_code=502, detail=message) from exc


@app.get("/api/experiments/{experiment_id}/results", tags=["experiments"])
async def get_experiment_results(experiment_id: str) -> dict[str, Any]:
    with connect() as db:
        row = db.execute(
            "SELECT status,error_message,results_json FROM experiments WHERE id=?", (experiment_id,)
        ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Experiment not found")
    if row["status"] != "done" or not row["results_json"]:
        return {"experiment_id": experiment_id, "status": row["status"], "error": row["error_message"], "results": None}
    return {"experiment_id": experiment_id, "status": row["status"], "error": None, "results": json_load(row["results_json"], {})}


@app.post("/api/experiments/compare", tags=["experiments"])
async def compare_experiments(body: ExperimentCompareRequest) -> dict[str, Any]:
    if len(set(body.experiment_ids)) != len(body.experiment_ids):
        raise HTTPException(status_code=422, detail="Experiment IDs must be unique.")
    placeholders = ",".join("?" for _ in body.experiment_ids)
    with connect() as db:
        rows = db.execute(
            f"SELECT id,name,dataset_id,benchmark_id,config_json,status,summary_json FROM experiments WHERE id IN ({placeholders})",
            tuple(body.experiment_ids),
        ).fetchall()
    by_id = {row["id"]: row for row in rows}
    if len(by_id) != len(body.experiment_ids):
        raise HTTPException(status_code=404, detail="One or more experiments were not found.")
    ordered = [by_id[item] for item in body.experiment_ids]
    if any(row["status"] != "done" for row in ordered):
        raise HTTPException(status_code=409, detail="Only completed experiments can be compared.")
    if len({row["dataset_id"] for row in ordered}) != 1 or len({row["benchmark_id"] for row in ordered}) != 1:
        raise HTTPException(status_code=409, detail="Experiments must use the same dataset and benchmark to be compared.")
    configs = [json_load(row["config_json"], {}) for row in ordered]
    keys = sorted(set().union(*(config.keys() for config in configs)))
    differences = {
        key: {row["id"]: config.get(key) for row, config in zip(ordered, configs)}
        for key in keys if len({json.dumps(config.get(key), sort_keys=True) for config in configs}) > 1
    }
    return {
        "dataset_id": ordered[0]["dataset_id"],
        "benchmark_id": ordered[0]["benchmark_id"],
        "configuration_differences": differences,
        "experiments": [
            {"id": row["id"], "name": row["name"], "summary": json_load(row["summary_json"], {})}
            for row in ordered
        ],
        "metric_deltas_vs_first": {
            row["id"]: {
                metric: (
                    round(json_load(row["summary_json"], {}).get(metric) - json_load(ordered[0]["summary_json"], {}).get(metric), 4)
                    if isinstance(json_load(row["summary_json"], {}).get(metric), (int, float))
                    and isinstance(json_load(ordered[0]["summary_json"], {}).get(metric), (int, float))
                    else None
                )
                for metric in ("precision_at_k", "recall_at_k", "mrr", "ndcg_at_k", "average_latency_ms")
            }
            for row in ordered[1:]
        },
    }


__all__ = ["app"]
