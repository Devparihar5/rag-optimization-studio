from __future__ import annotations

import importlib
import sqlite3
from pathlib import Path

from fastapi.testclient import TestClient


def _client(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("RAGSTUDIO_DATA_DIR", str(tmp_path))
    import src.ragstudio.storage as storage
    import src.ragstudio.model_registry as model_registry
    import app.main as main

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


def test_benchmark_questions_create_edit_and_reopen(tmp_path: Path, monkeypatch) -> None:
    client, _ = _client(tmp_path, monkeypatch)
    dataset = client.post("/api/datasets", json={"name": "qa dataset", "description": "", "source_type": "text"})
    assert dataset.status_code == 201
    dataset_id = dataset.json()["id"]
    created = client.post("/api/benchmarks", json={
        "name": "support QA",
        "dataset_id": dataset_id,
        "questions": [{"question": "What is the policy?", "answer": "Refunds are allowed.", "relevant_passages": ["Refunds are allowed within 30 days."]}],
    })
    assert created.status_code == 201
    benchmark_id = created.json()["id"]
    reopened = client.get(f"/api/benchmarks/{benchmark_id}")
    assert reopened.json()["questions"][0]["answer"] == "Refunds are allowed."
    updated = client.put(f"/api/benchmarks/{benchmark_id}", json={
        "name": "support QA v2",
        "questions": [{"question": "How long is the refund window?", "answer": "30 days", "relevant_passages": ["Refunds are allowed within 30 days."]}],
    })
    assert updated.status_code == 200
    assert updated.json()["question_count"] == 1
    assert client.get(f"/api/benchmarks/{benchmark_id}").json()["questions"][0]["answer"] == "30 days"


def test_retrieval_strategies_and_chunking_are_deterministic(tmp_path: Path, monkeypatch) -> None:
    client, _ = _client(tmp_path, monkeypatch)
    import app.main as main

    chunks = [
        {"text": "cat cat lexical match", "filename": "a.txt", "document_id": "a", "chunk_index": 0},
        {"text": "feline semantic match", "filename": "b.txt", "document_id": "b", "chunk_index": 0},
        {"text": "unrelated ocean passage", "filename": "c.txt", "document_id": "c", "chunk_index": 0},
    ]
    assert main._bm25_scores("cat", [item["text"] for item in chunks])[0] > main._bm25_scores("cat", [item["text"] for item in chunks])[1]
    monkeypatch.setattr(main, "_embed_texts", lambda runtime, texts: [
        [1.0, 0.0] if text == "cat" or "feline" in text else ([0.8, 0.2] if "cat cat" in text else [0.0, 1.0])
        for text in texts
    ])
    dense = main._rank_chunks("cat", chunks, "dense", {"provider": "test"}, 1)
    lexical = main._rank_chunks("cat", chunks, "bm25", None, 1)
    hybrid = main._rank_chunks("cat", chunks, "hybrid_rrf", {"provider": "test"}, 2)
    assert dense[0]["filename"] == "b.txt"
    assert lexical[0]["filename"] == "a.txt"
    assert len(hybrid) == 2
    assert main._chunk_document("One short sentence. Another sentence.", "sentence-aware", 100, 0) == [
        "One short sentence. Another sentence."
    ]
    assert main._question_metrics([], chunks, [], 2)["mrr"] is None


def test_experiment_execution_persists_real_results_and_failure(tmp_path: Path, monkeypatch) -> None:
    client, _ = _client(tmp_path, monkeypatch)
    import app.main as main

    monkeypatch.setattr(main, "get_model_runtime_config", lambda model_id: {
        "id": model_id, "provider": "openai", "model_id": "mock-generation", "api_key": "test", "base_url": None
    })
    monkeypatch.setattr(main, "get_embedding_model_runtime_config", lambda model_id: {
        "id": model_id, "provider": "openai", "model_id": "mock-embedding", "api_key": "test", "base_url": None
    })
    monkeypatch.setattr(main, "_generate_answer", lambda runtime, question, evidence: ("The sky is blue.", {"total_tokens": 7}))
    dataset = client.post("/api/datasets", json={"name": "sky docs", "description": "", "source_type": "text"}).json()
    uploaded = client.post(
        f"/api/datasets/{dataset['id']}/documents",
        files=[("files", ("sky.txt", b"The sky is blue.", "text/plain"))],
    )
    assert uploaded.status_code == 200
    benchmark = client.post("/api/benchmarks", json={
        "name": "sky QA",
        "dataset_id": dataset["id"],
        "questions": [{"question": "What color is the sky?", "answer": "Blue.", "relevant_passages": ["The sky is blue."]}],
    }).json()
    config = {
        "chunking_strategy": "fixed-size", "chunk_size": 100, "overlap": 0,
        "retrieval_method": "bm25", "top_k": 1, "reranking": False,
        "query_rewrite": False, "embedding_model": "embedding-config-id",
        "generation_model": "generation-config-id",
    }
    experiment = client.post("/api/experiments", json={
        "name": "bm25 run", "dataset_id": dataset["id"], "benchmark_id": benchmark["id"], "config": config,
    })
    assert experiment.status_code == 201
    experiment_id = experiment.json()["id"]
    run = client.post(f"/api/experiments/{experiment_id}/run")
    assert run.status_code == 200
    assert run.json()["status"] == "done"
    results = client.get(f"/api/experiments/{experiment_id}/results").json()["results"]
    assert results["questions"][0]["generated_answer"] == "The sky is blue."
    assert results["summary"]["mrr"] == 1.0
    assert results["summary"]["cost_usd"] is None

    monkeypatch.setattr(main, "_generate_answer", lambda *args: (_ for _ in ()).throw(RuntimeError("mock provider failure")))
    failed = client.post(f"/api/experiments/{experiment_id}/run")
    assert failed.status_code == 502
    assert client.get(f"/api/experiments/{experiment_id}/results").json()["status"] == "failed"
