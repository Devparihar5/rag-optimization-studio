from __future__ import annotations

import io
import json
import re
import uuid
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from starlette.requests import Request

try:
    from ragstudio.storage import connect, data_dir
    from ragstudio.model_registry import router as model_router
except ModuleNotFoundError:
    # Supports running directly from a source checkout without an editable install.
    from src.ragstudio.storage import connect, data_dir
    from src.ragstudio.model_registry import router as model_router

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


__all__ = ["app"]
