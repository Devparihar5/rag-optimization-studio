from __future__ import annotations

import importlib
import sqlite3
from pathlib import Path

from fastapi.testclient import TestClient


def _client(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("RAGSTUDIO_DATA_DIR", str(tmp_path))
    from app import main
    from ragstudio import model_registry, storage

    importlib.reload(storage)
    importlib.reload(model_registry)
    importlib.reload(main)
    return TestClient(main.app), model_registry


def test_model_discovery_and_saved_keys_are_not_exposed(tmp_path: Path, monkeypatch) -> None:
    client, model_registry = _client(tmp_path, monkeypatch)

    def fake_discover(provider, api_key, base_url):
        assert provider == "openai"
        assert api_key == "test-secret-key"
        return [{"id": "gpt-test-model", "name": "GPT Test Model"}]

    monkeypatch.setattr(model_registry, "discover_models", fake_discover)
    discovery = client.post("/api/models/discover", json={
        "provider": "openai", "api_key": "test-secret-key"
    })
    assert discovery.status_code == 200
    assert discovery.json()["models"] == [{"id": "gpt-test-model", "name": "GPT Test Model"}]

    saved = client.post("/api/models", json={
        "provider": "openai", "api_key": "test-secret-key", "model_id": "gpt-test-model"
    })
    assert saved.status_code == 201
    saved_json = saved.json()
    assert saved_json["model_id"] == "gpt-test-model"
    assert saved_json["has_api_key"] is True
    assert "test-secret-key" not in saved.text

    listed = client.get("/api/models")
    assert listed.status_code == 200
    assert len(listed.json()) == 1
    assert "test-secret-key" not in listed.text

    connection = sqlite3.connect(tmp_path / "ragstudio.sqlite3")
    try:
        stored_key = connection.execute(
            "SELECT encrypted_api_key FROM configured_models WHERE id=?",
            (saved_json["id"],),
        ).fetchone()[0]
    finally:
        connection.close()
    assert stored_key != "test-secret-key"
    assert "test-secret-key" not in stored_key

    deleted = client.delete(f"/api/models/{saved_json['id']}")
    assert deleted.status_code == 204
    assert client.get("/api/models").json() == []


def test_only_models_returned_by_provider_can_be_saved(tmp_path: Path, monkeypatch) -> None:
    client, model_registry = _client(tmp_path, monkeypatch)
    monkeypatch.setattr(
        model_registry,
        "discover_models",
        lambda provider, api_key, base_url: [{"id": "allowed-model", "name": "Allowed"}],
    )

    response = client.post("/api/models", json={
        "provider": "gemini", "api_key": "valid-key", "model_id": "not-available"
    })
    assert response.status_code == 422
    assert "not available" in response.json()["detail"]


def test_local_model_does_not_require_api_key(tmp_path: Path, monkeypatch) -> None:
    client, model_registry = _client(tmp_path, monkeypatch)
    monkeypatch.setattr(
        model_registry,
        "discover_models",
        lambda provider, api_key, base_url: [{"id": "qwen2.5:7b", "name": "qwen2.5:7b"}],
    )

    response = client.post("/api/models", json={
        "provider": "ollama",
        "base_url": "http://localhost:11434",
        "model_id": "qwen2.5:7b",
    })
    assert response.status_code == 201
    assert response.json()["provider"] == "ollama"
    assert response.json()["has_api_key"] is False



def test_embedding_model_discovery_and_storage(tmp_path: Path, monkeypatch) -> None:
    client, model_registry = _client(tmp_path, monkeypatch)
    monkeypatch.setattr(
        model_registry,
        "discover_embedding_models",
        lambda provider, api_key, base_url: [{"id": "text-embedding-3-small", "name": "Text Embedding 3 Small"}],
    )

    discovery = client.post("/api/embedding-models/discover", json={
        "provider": "openai", "api_key": "test-embedding-key"
    })
    assert discovery.status_code == 200
    assert discovery.json()["models"][0]["id"] == "text-embedding-3-small"

    saved = client.post("/api/embedding-models", json={
        "provider": "openai",
        "api_key": "test-embedding-key",
        "model_id": "text-embedding-3-small",
    })
    assert saved.status_code == 201
    saved_json = saved.json()
    assert saved_json["has_api_key"] is True
    assert "test-embedding-key" not in saved.text

    listed = client.get("/api/embedding-models")
    assert listed.status_code == 200
    assert len(listed.json()) == 1
    assert listed.json()[0]["model_id"] == "text-embedding-3-small"

    deleted = client.delete(f"/api/embedding-models/{saved_json['id']}")
    assert deleted.status_code == 204
    assert client.get("/api/embedding-models").json() == []


def test_embedding_model_cannot_save_model_not_discovered(tmp_path: Path, monkeypatch) -> None:
    client, model_registry = _client(tmp_path, monkeypatch)
    monkeypatch.setattr(
        model_registry,
        "discover_embedding_models",
        lambda provider, api_key, base_url: [{"id": "text-embedding-3-small", "name": "Text Embedding 3 Small"}],
    )
    response = client.post("/api/embedding-models", json={
        "provider": "openai",
        "api_key": "test-key",
        "model_id": "gpt-4o",
    })
    assert response.status_code == 422


def test_experiment_requires_explicit_generation_and_embedding_models(tmp_path: Path, monkeypatch) -> None:
    client, _ = _client(tmp_path, monkeypatch)
    response = client.post("/api/experiments", json={
        "name": "Missing model selection",
        "dataset_id": "dataset-does-not-matter-for-validation",
    })
    assert response.status_code == 422
    details = response.json()["detail"]
    missing_fields = {item["loc"][-1] for item in details}
    assert {"generation_model", "embedding_model"}.issubset(missing_fields)
