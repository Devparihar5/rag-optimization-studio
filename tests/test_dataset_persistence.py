from __future__ import annotations

import importlib
from pathlib import Path

from fastapi.testclient import TestClient


def test_dataset_upload_persistence_and_cascade(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("RAGSTUDIO_DATA_DIR", str(tmp_path))
    import src.ragstudio.storage as storage
    import app.main as main
    importlib.reload(storage)
    importlib.reload(main)

    client = TestClient(main.app)
    dataset = client.post("/api/datasets", json={
        "name": "sample corpus", "description": "test", "source_type": "text"
    })
    assert dataset.status_code == 201
    dataset_id = dataset.json()["id"]

    uploaded = client.post(
        f"/api/datasets/{dataset_id}/documents",
        files=[("files", ("notes.md", b"# Intro\nPersistent content", "text/markdown"))],
    )
    assert uploaded.status_code == 200
    assert uploaded.json()["document_count"] == 1
    assert uploaded.json()["uploaded"][0]["filename"] == "notes.md"
    assert client.get(f"/api/datasets/{dataset_id}").json()["document_count"] == 1
    assert client.get(f"/api/datasets/{dataset_id}/documents").json()[0]["filename"] == "notes.md"

    benchmark = client.post("/api/benchmarks", json={
        "name": "sample questions", "dataset_id": dataset_id,
        "questions": [{"question": "What is persistent?", "answer": "Data stored beyond process lifetime"}],
    })
    assert benchmark.status_code == 201
    assert client.get(f"/api/benchmarks/{benchmark.json()['id']}").json()["question_count"] == 1

    experiment = client.post("/api/experiments", json={
        "name": "baseline", "dataset_id": dataset_id, "benchmark_id": benchmark.json()["id"],
        "config": {"embedding_model": "configured-embedding", "generation_model": "configured-generation"}
    })
    assert experiment.status_code == 201
    experiment_id = experiment.json()["id"]

    # Re-import the application against the same data directory to simulate a restart.
    importlib.reload(main)
    restarted = TestClient(main.app)
    assert restarted.get("/api/datasets").json()[0]["id"] == dataset_id
    assert restarted.get(f"/api/experiments/{experiment_id}").status_code == 200

    deleted = restarted.delete(f"/api/datasets/{dataset_id}")
    assert deleted.status_code == 204
    assert restarted.get("/api/datasets").json() == []
    assert restarted.get("/api/benchmarks").json() == []
    assert restarted.get("/api/experiments").json() == []
    assert restarted.get(f"/api/datasets/{dataset_id}/documents").status_code == 404


def test_rejects_benchmark_from_another_dataset(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("RAGSTUDIO_DATA_DIR", str(tmp_path))
    import src.ragstudio.storage as storage
    import app.main as main
    importlib.reload(storage)
    importlib.reload(main)
    client = TestClient(main.app)
    dataset_ids = [
        client.post("/api/datasets", json={"name": name, "source_type": "text"}).json()["id"]
        for name in ("first corpus", "second corpus")
    ]
    benchmark = client.post("/api/benchmarks", json={
        "name": "first benchmark", "dataset_id": dataset_ids[0], "questions": [{"question": "What is this corpus?"}]
    }).json()
    response = client.post("/api/experiments", json={
        "name": "invalid pairing", "dataset_id": dataset_ids[1], "benchmark_id": benchmark["id"],
        "config": {"embedding_model": "configured-embedding", "generation_model": "configured-generation"}
    })
    assert response.status_code == 409


def test_upload_rejects_unsupported_files(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("RAGSTUDIO_DATA_DIR", str(tmp_path))
    import src.ragstudio.storage as storage
    import app.main as main
    importlib.reload(storage)
    importlib.reload(main)
    client = TestClient(main.app)
    dataset_id = client.post("/api/datasets", json={"name": "sample corpus"}).json()["id"]
    response = client.post(
        f"/api/datasets/{dataset_id}/documents",
        files=[("files", ("payload.exe", b"not allowed", "application/octet-stream"))],
    )
    assert response.status_code == 200
    assert response.json()["uploaded"] == []
    assert response.json()["failed"][0]["filename"] == "payload.exe"

def test_deleting_benchmark_unlinks_experiment(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("RAGSTUDIO_DATA_DIR", str(tmp_path))
    import importlib
    import src.ragstudio.storage as storage
    import app.main as main
    importlib.reload(storage)
    importlib.reload(main)
    client = TestClient(main.app)
    dataset_id = client.post("/api/datasets", json={"name": "sample corpus"}).json()["id"]
    benchmark = client.post("/api/benchmarks", json={
        "name": "sample benchmark", "dataset_id": dataset_id, "questions": [{"question": "What is this corpus?"}]
    }).json()
    experiment = client.post("/api/experiments", json={
        "name": "sample experiment", "dataset_id": dataset_id, "benchmark_id": benchmark["id"],
        "config": {"embedding_model": "configured-embedding", "generation_model": "configured-generation"}
    }).json()

    response = client.delete(f"/api/benchmarks/{benchmark['id']}")
    assert response.status_code == 204
    assert client.get(f"/api/benchmarks/{benchmark['id']}").status_code == 404
    assert client.get(f"/api/experiments/{experiment['id']}").json()["benchmark_id"] is None
