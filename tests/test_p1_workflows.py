from __future__ import annotations

import importlib
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
    return TestClient(main.app), main


def _dataset(client: TestClient, name: str = "test-dataset") -> str:
    response = client.post("/api/datasets", json={"name": name, "description": "", "source_type": "text"})
    assert response.status_code == 201
    return response.json()["id"]


def _benchmark(client: TestClient, dataset_id: str, name: str = "test-benchmark") -> str:
    response = client.post("/api/benchmarks", json={
        "name": name,
        "dataset_id": dataset_id,
        "questions": [{
            "question": "What does the document explain?",
            "answer": "It explains the system.",
            "relevant_passages": ["The document explains the system."],
        }],
    })
    assert response.status_code == 201
    return response.json()["id"]


def _experiment(client: TestClient, dataset_id: str, benchmark_id: str, name: str = "test-run") -> str:
    response = client.post("/api/experiments", json={
        "name": name,
        "dataset_id": dataset_id,
        "benchmark_id": benchmark_id,
        "config": {
            "embedding_model": "configured-embedding-id",
            "generation_model": "configured-generation-id",
            "overlap": 20,
            "chunk_size": 100,
            "reranking": False,
        },
    })
    assert response.status_code == 201, response.text
    return response.json()["id"]


def test_benchmark_requires_questions_and_can_be_edited(tmp_path: Path, monkeypatch) -> None:
    client, _ = _client(tmp_path, monkeypatch)
    dataset_id = _dataset(client)

    empty = client.post("/api/benchmarks", json={
        "name": "empty-benchmark", "dataset_id": dataset_id, "questions": [],
    })
    assert empty.status_code == 422

    benchmark_id = _benchmark(client, dataset_id)
    detail = client.get(f"/api/benchmarks/{benchmark_id}")
    assert detail.status_code == 200
    assert detail.json()["questions"][0]["question"] == "What does the document explain?"

    updated = client.put(f"/api/benchmarks/{benchmark_id}", json={
        "name": "updated-benchmark",
        "dataset_id": dataset_id,
        "questions": [{"question": "What is the system?", "answer": "A test system.", "relevant_passages": []}],
    })
    assert updated.status_code == 200
    assert client.get(f"/api/benchmarks/{benchmark_id}").json()["questions"][0]["answer"] == "A test system."


def test_cross_dataset_benchmark_is_rejected(tmp_path: Path, monkeypatch) -> None:
    client, _ = _client(tmp_path, monkeypatch)
    first_dataset = _dataset(client, "dataset-one")
    second_dataset = _dataset(client, "dataset-two")
    benchmark_id = _benchmark(client, first_dataset)
    response = client.post("/api/experiments", json={
        "name": "invalid-link",
        "dataset_id": second_dataset,
        "benchmark_id": benchmark_id,
        "config": {"embedding_model": "embedding-id", "generation_model": "generation-id"},
    })
    assert response.status_code == 409


def test_run_persists_results_and_comparison(tmp_path: Path, monkeypatch) -> None:
    client, main = _client(tmp_path, monkeypatch)
    dataset_id = _dataset(client)
    benchmark_id = _benchmark(client, dataset_id)

    def fake_execute(dataset, documents, questions, config):
        return {
            "cases": [{
                "question": questions[0]["question"],
                "answer": "A real answer from the mocked executor.",
                "retrieved": [],
                "metrics": {"precision_at_k": None, "recall_at_k": None, "mrr": None, "ndcg_at_k": None},
                "latency_ms": 12.5,
                "cost": None,
            }],
            "metrics": {"mrr": None, "avg_latency_ms": 12.5, "cost": None},
            "question_count": 1,
            "evaluated_count": 0,
        }

    monkeypatch.setattr(main, "execute_experiment", fake_execute)
    first = _experiment(client, dataset_id, benchmark_id, "run-one")
    second = _experiment(client, dataset_id, benchmark_id, "run-two")
    for experiment_id in (first, second):
        response = client.post(f"/api/experiments/{experiment_id}/run")
        assert response.status_code == 202
        results = client.get(f"/api/experiments/{experiment_id}/results")
        assert results.status_code == 200
        assert results.json()["status"] == "done"
        assert results.json()["results"]["cases"][0]["answer"].startswith("A real answer")

    compared = client.post("/api/experiments/compare", json={"experiment_ids": [first, second]})
    assert compared.status_code == 200
    assert len(compared.json()["experiments"]) == 2


def test_failed_run_persists_failure_details(tmp_path: Path, monkeypatch) -> None:
    client, main = _client(tmp_path, monkeypatch)
    dataset_id = _dataset(client)
    benchmark_id = _benchmark(client, dataset_id)
    experiment_id = _experiment(client, dataset_id, benchmark_id)
    monkeypatch.setattr(main, "execute_experiment", lambda *args: (_ for _ in ()).throw(RuntimeError("provider unavailable")))

    response = client.post(f"/api/experiments/{experiment_id}/run")
    assert response.status_code == 202
    result = client.get(f"/api/experiments/{experiment_id}/results").json()
    assert result["status"] == "failed"
    assert "provider unavailable" in result["error"]
