"""SQLite persistence for RAG Studio's datasets, documents, benchmarks, and experiments."""
from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


def data_dir() -> Path:
    configured = os.getenv("RAGSTUDIO_DATA_DIR")
    return Path(configured).expanduser().resolve() if configured else Path(__file__).resolve().parents[2] / ".data"


def database_path() -> Path:
    return data_dir() / "ragstudio.sqlite3"


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    root = data_dir()
    root.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database_path(), timeout=15)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA journal_mode = WAL")
    try:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS datasets (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT NOT NULL DEFAULT '',
                source_type TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                dataset_id TEXT NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
                filename TEXT NOT NULL,
                media_type TEXT NOT NULL,
                size_bytes INTEGER NOT NULL,
                stored_path TEXT NOT NULL,
                extracted_text TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_documents_dataset_id ON documents(dataset_id);
            CREATE TABLE IF NOT EXISTS benchmarks (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                dataset_id TEXT NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
                questions_json TEXT NOT NULL DEFAULT '[]',
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_benchmarks_dataset_id ON benchmarks(dataset_id);
            CREATE TABLE IF NOT EXISTS experiments (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                dataset_id TEXT NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
                benchmark_id TEXT REFERENCES benchmarks(id) ON DELETE SET NULL,
                config_json TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                summary_json TEXT NOT NULL DEFAULT '{}',
                result_json TEXT NOT NULL DEFAULT '{}',
                error TEXT,
                started_at TEXT,
                finished_at TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_experiments_dataset_id ON experiments(dataset_id);
            """
        )
        # Migrate existing local databases without requiring users to delete data.
        columns = {row[1] for row in connection.execute("PRAGMA table_info(experiments)").fetchall()}
        for name, definition in (
            ("result_json", "TEXT NOT NULL DEFAULT '{}'"),
            ("error", "TEXT"),
            ("started_at", "TEXT"),
            ("finished_at", "TEXT"),
        ):
            if name not in columns:
                connection.execute(f"ALTER TABLE experiments ADD COLUMN {name} {definition}")
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()
