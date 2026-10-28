"""Provider-backed model discovery and encrypted local model configuration."""
from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen

from cryptography.fernet import Fernet, InvalidToken
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

try:
    from ragstudio.storage import connect, data_dir
except ModuleNotFoundError:
    from src.ragstudio.storage import connect, data_dir

router = APIRouter(tags=["models"])
Provider = Literal["openai", "gemini", "ollama"]


class ModelDiscoveryRequest(BaseModel):
    provider: Provider
    api_key: str | None = Field(default=None, max_length=4096)
    base_url: str | None = Field(default=None, max_length=500)


class ModelSaveRequest(ModelDiscoveryRequest):
    model_id: str = Field(min_length=1, max_length=300)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_model_table() -> None:
    with connect() as db:
        db.execute(
            """CREATE TABLE IF NOT EXISTS configured_models (
                id TEXT PRIMARY KEY,
                provider TEXT NOT NULL,
                model_id TEXT NOT NULL,
                display_name TEXT NOT NULL,
                encrypted_api_key TEXT,
                base_url TEXT,
                created_at TEXT NOT NULL,
                UNIQUE(provider, model_id)
            )"""
        )


def _fernet() -> Fernet:
    root = data_dir()
    root.mkdir(parents=True, exist_ok=True)
    key_path = root / ".model_encryption_key"
    if not key_path.exists():
        key_path.write_bytes(Fernet.generate_key())
        try:
            key_path.chmod(0o600)
        except OSError:
            pass
    try:
        key_path.chmod(0o600)
    except OSError:
        pass
    return Fernet(key_path.read_bytes())


def encrypt_api_key(value: str | None) -> str | None:
    if not value:
        return None
    return _fernet().encrypt(value.encode("utf-8")).decode("ascii")


def decrypt_api_key(value: str | None) -> str | None:
    if not value:
        return None
    try:
        return _fernet().decrypt(value.encode("ascii")).decode("utf-8")
    except (InvalidToken, OSError, ValueError) as exc:
        raise HTTPException(
            status_code=500,
            detail="Could not decrypt a saved provider key. Check the local model encryption key file.",
        ) from exc


def _json_request(url: str, headers: dict[str, str] | None = None) -> dict[str, Any]:
    request = Request(url, headers={"Accept": "application/json", **(headers or {})})
    try:
        with urlopen(request, timeout=15) as response:
            payload = json.loads(response.read().decode("utf-8"))
            if not isinstance(payload, dict):
                raise HTTPException(status_code=502, detail="Provider returned an unexpected response.")
            return payload
    except HTTPError as exc:
        # Never return provider URLs or request headers, which can contain credentials.
        if exc.code in (400, 401, 403):
            raise HTTPException(status_code=401, detail="Provider rejected this API key. Check the key and its permissions.") from exc
        if exc.code == 429:
            raise HTTPException(status_code=429, detail="Provider rate limit reached. Try again shortly.") from exc
        raise HTTPException(status_code=502, detail=f"Provider request failed (HTTP {exc.code}).") from exc
    except (URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=502, detail="Could not reach the provider. Check your connection and provider URL.") from exc


def _ollama_url(base_url: str | None) -> str:
    url = (base_url or "http://localhost:11434").strip().rstrip("/")
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise HTTPException(status_code=422, detail="Enter a valid Ollama URL.")
    if parsed.hostname not in {"localhost", "127.0.0.1", "::1", "host.docker.internal"}:
        raise HTTPException(status_code=422, detail="Ollama URL must point to this machine or host.docker.internal.")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise HTTPException(status_code=422, detail="Ollama URL must not contain credentials, query parameters, or fragments.")
    return url


def discover_models(provider: Provider, api_key: str | None, base_url: str | None) -> list[dict[str, str]]:
    """Return models this credential can access that are suitable for generation."""
    if provider == "openai":
        if not api_key or not api_key.strip():
            raise HTTPException(status_code=422, detail="Enter an OpenAI API key first.")
        payload = _json_request("https://api.openai.com/v1/models", {"Authorization": f"Bearer {api_key.strip()}"})
        excluded = ("embedding", "whisper", "tts", "dall-e", "moderation", "transcribe", "realtime", "search", "babbage", "davinci", "audio", "image")
        ids = [
            item.get("id", "") for item in payload.get("data", [])
            if isinstance(item, dict) and isinstance(item.get("id"), str)
            and not any(token in item["id"].lower() for token in excluded)
        ]
        return [{"id": model_id, "name": model_id} for model_id in sorted(set(ids))]
    if provider == "gemini":
        if not api_key or not api_key.strip():
            raise HTTPException(status_code=422, detail="Enter a Gemini API key first.")
        url = "https://generativelanguage.googleapis.com/v1beta/models?" + urlencode({"key": api_key.strip()})
        payload = _json_request(url)
        models = []
        for item in payload.get("models", []):
            if not isinstance(item, dict):
                continue
            methods = item.get("supportedGenerationMethods", [])
            name = item.get("name", "")
            if isinstance(name, str) and "generateContent" in methods:
                model_id = name.removeprefix("models/")
                models.append({"id": model_id, "name": item.get("displayName") or model_id})
        return sorted(models, key=lambda item: item["id"])
    if provider == "ollama":
        url = _ollama_url(base_url)
        payload = _json_request(f"{url}/api/tags")
        models = []
        for item in payload.get("models", []):
            if isinstance(item, dict) and isinstance(item.get("name"), str):
                name = item["name"]
                models.append({"id": name, "name": name})
        return sorted(models, key=lambda item: item["id"])
    raise HTTPException(status_code=422, detail="Unsupported model provider.")


def get_model_runtime_config(configured_model_id: str) -> dict[str, Any]:
    """Resolve a saved model for server-side inference without exposing its key over the API."""
    ensure_model_table()
    with connect() as db:
        row = db.execute(
            "SELECT * FROM configured_models WHERE id=?", (configured_model_id,)
        ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Configured model not found")
    return {
        "id": row["id"],
        "provider": row["provider"],
        "model_id": row["model_id"],
        "api_key": decrypt_api_key(row["encrypted_api_key"]),
        "base_url": row["base_url"],
    }


def public_model(row: Any) -> dict[str, Any]:
    return {
        "id": row["id"],
        "provider": row["provider"],
        "model_id": row["model_id"],
        "name": row["display_name"],
        "base_url": row["base_url"],
        "created_at": row["created_at"],
        "has_api_key": bool(row["encrypted_api_key"]),
    }


@router.post("/api/models/discover")
async def model_discover(body: ModelDiscoveryRequest) -> dict[str, Any]:
    models = discover_models(body.provider, body.api_key, body.base_url)
    return {"provider": body.provider, "models": models, "count": len(models)}


@router.get("/api/models")
async def list_configured_models() -> list[dict[str, Any]]:
    ensure_model_table()
    with connect() as db:
        rows = db.execute("SELECT * FROM configured_models ORDER BY created_at DESC").fetchall()
    return [public_model(row) for row in rows]


@router.post("/api/models", status_code=201)
async def save_configured_model(body: ModelSaveRequest) -> dict[str, Any]:
    models = discover_models(body.provider, body.api_key, body.base_url)
    chosen = next((item for item in models if item["id"] == body.model_id), None)
    if chosen is None:
        raise HTTPException(
            status_code=422,
            detail="This model is not available for the supplied credentials. Verify the key and load models again.",
        )
    ensure_model_table()
    model_id = str(uuid.uuid4())
    created_at = now_iso()
    base_url = _ollama_url(body.base_url) if body.provider == "ollama" else None
    encrypted_key = encrypt_api_key(body.api_key.strip() if body.provider != "ollama" and body.api_key else None)
    try:
        with connect() as db:
            db.execute(
                """INSERT INTO configured_models
                   (id,provider,model_id,display_name,encrypted_api_key,base_url,created_at)
                   VALUES(?,?,?,?,?,?,?)""",
                (model_id, body.provider, chosen["id"], chosen["name"], encrypted_key, base_url, created_at),
            )
            row = db.execute("SELECT * FROM configured_models WHERE id=?", (model_id,)).fetchone()
    except sqlite3.IntegrityError as exc:
        raise HTTPException(status_code=409, detail="This model is already configured.") from exc
    return public_model(row)


@router.delete("/api/models/{configured_model_id}", status_code=204)
async def delete_configured_model(configured_model_id: str) -> None:
    ensure_model_table()
    with connect() as db:
        cursor = db.execute("DELETE FROM configured_models WHERE id=?", (configured_model_id,))
        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="Configured model not found")
