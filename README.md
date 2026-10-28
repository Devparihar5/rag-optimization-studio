# RAG Optimization Studio

A single-server experimentation platform for benchmarking and improving retrieval-augmented generation (RAG) pipelines.

This repository starts with a working FastAPI app that serves both the API and the web dashboard from one process. That keeps the first implementation simple while still matching the product direction for a modern RAG evaluation studio.

## Product direction

- Upload datasets and benchmark questions
- Configure chunking, embeddings, retrieval, reranking, and generation
- Run reproducible experiments
- Compare retrieval quality, answer quality, latency, and cost
- Inspect failures and improve the system iteratively

## Single-server architecture

Instead of running separate backend and frontend servers, the app serves:

- API routes under `/api/*`
- Frontend assets under `/static/*`
- The dashboard at `/`

This follows the requested design of keeping backend and frontend on one server.

## Full application architecture

```text
┌──────────────────────────────────────────────────────────────────────┐
│ Browser / User                                                       │
│ - Opens dashboard                                                    │
│ - Views overview and experiments                                     │
│ - Creates datasets, benchmarks, and experiments                     │
└───────────────────────────────┬──────────────────────────────────────┘
                                │ HTTP / JSON
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│ FastAPI Application (single process)                                │
│ app/main.py                                                         │
│ - serves /                                                          │
│ - serves /api/*                                                     │
│ - serves /static/*                                                  │
│ - renders dashboard templates                                       │
│ - exposes health and summary endpoints                               │
└───────────────────────────────┬──────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│ Application Service Layer                                            │
│ src/ragstudio/                                                       │
│ - experiments/service.py                                            │
│ - dataset management                                                 │
│ - benchmark management                                               │
│ - experiment lifecycle                                               │
│ - dashboard summary generation                                        │
└───────────────┬───────────────────────────────────────────────┬──────┘
                │                                               │
                ▼                                               ▼
┌──────────────────────────────┐                 ┌──────────────────────────────┐
│ Domain & Config Models        │                 │ Retrieval + Evaluation Stack │
│ src/ragstudio/experiments/   │                 │ src/ragstudio/              │
│ models.py                    │                 │ retrieval/                  │
│ - PipelineConfig             │                 │ evaluation/                 │
│ - BenchmarkEntry              │                 │ - chunking                  │
│ - Experiment configuration   │                 │ - embeddings               │
│                              │                 │ - retrieval                 │
│                              │                 │ - reranking                 │
│                              │                 │ - metrics                   │
└──────────────────────────────┘                 └──────────────────────────────┘
                                │
                                ▼
                    ┌──────────────────────┐
                    │ In-memory Runtime    │
                    │ State                │
                    │ - datasets           │
                    │ - benchmarks         │
                    │ - experiments       │
                    └──────────────────────┘
```

This repository is intentionally structured around the same single-server mental model: the dashboard, API, and orchestration logic live together while the domain and retrieval/evaluation logic stay organized as separate modules.

## Tech stack

- Python + FastAPI
- Jinja2 templates for the dashboard
- Static frontend assets served by the same app
- Pydantic models for experiment configs

## Dataset uploads and persistence

Datasets, uploaded documents, benchmark questions, and experiment configurations are stored in SQLite rather than process memory. The default data directory is `.data/` at the repository root. Set `RAGSTUDIO_DATA_DIR` to move the database and uploaded-file storage to a persistent volume in deployment.

Open **Datasets**, create a dataset, then select **Documents** on its row to upload one or more PDF, Markdown, TXT, or HTML files. Files are limited to 10 MB each. Text and HTML are extracted directly; PDFs are extracted with pypdf. Each upload reports per-file failures so valid files in the same batch can still be saved.

Deleting a dataset cascades to its documents, benchmarks, and experiments. The associated uploaded files are removed from disk. An experiment's benchmark must belong to the same dataset; invalid pairings return HTTP 409.


## Model providers and credentials

Open **Models** to configure generation and embedding models before creating an experiment.

- Choose **Generation model** or **Embedding model**, then select OpenAI, Google Gemini, or local Ollama.
- **OpenAI** and **Google Gemini**: enter an API key, select **Verify connection & load models**, and choose from compatible models returned by that provider for the supplied key. Gemini generation models must advertise `generateContent`; Gemini embedding models must advertise `embedContent`. OpenAI embedding discovery filters for embedding model IDs.
- **Ollama (local)**: run Ollama on your machine and install models with `ollama pull <model>`. The app discovers installed models at `http://localhost:11434` and checks Ollama model capabilities before listing an embedding model.
- Select configured generation and embedding models in the experiment form. The selectors submit the saved configuration IDs instead of arbitrary model names.
- Provider API keys are encrypted before being stored in SQLite. The encryption key is kept in `.data/.model_encryption_key` with restrictive file permissions and is never returned by the API. Back up this file with the database if you need to preserve access to saved credentials. Protect the entire data directory and do not commit it.

Model discovery validates credentials and lists compatible models. Experiment execution is a separate roadmap item; configuring a model does not itself run an inference, create embeddings, or execute a benchmark.

## Benchmarks, retrieval, and experiment execution

- Create benchmarks with one or more evaluation questions. Each question can include an expected answer and one or more relevant passage strings. Questions can be edited after saving or imported from JSON or CSV. CSV headers must include `question`; optional headers are `answer` (or `expected_answer`) and `relevant_passages` (separate passages with `||`). Import errors include the row where possible.
- Run an experiment from the Experiments table. A run requires uploaded text documents and a benchmark with questions. Status, start/completion timestamps, failure details, summary metrics, and per-question results persist in SQLite.
- Implemented retrieval choices: **BM25** (lexical term-frequency ranking), **Dense** (provider embedding cosine similarity), and **Hybrid RRF** (reciprocal-rank fusion of BM25 and dense rankings). Dense and hybrid embed dataset chunks once per run and reuse those vectors across questions.
- Implemented chunking choices: fixed-size word windows, sentence-aware grouping, and recursive paragraph/sentence splitting. Chunk size/overlap and top-k are validated.
- Reranking and query rewriting are shown as unavailable in the UI and rejected by the API while no real implementation is connected. They are not silently simulated.
- Retrieval metrics use explicit `relevant_passages` labels: a chunk is relevant when the normalized reference passage is contained in the chunk (or vice versa). Precision@k, Recall@k, MRR, and binary NDCG@k are unavailable when no relevance labels are supplied. Generated answers and retrieved evidence are shown per question. Latency is measured; token usage is shown when returned by the provider. Cost is unavailable unless provider pricing is configured.
- Compare two or more completed runs only when they share the same dataset and benchmark. The comparison shows metric summaries, configuration differences, and metric deltas against the first selected run.

For production deployments, mount a persistent volume at the configured data directory and back it up along with the application.

## Local development

1. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the app:
   ```bash
   uvicorn app.main:app --reload
   ```

4. Open the dashboard:
   - http://localhost:8000
   - API docs: http://localhost:8000/docs

## Project structure

```text
app/
  main.py
  static/
    app.js
    styles.css
  templates/
    index.html
src/
  ragstudio/
    __init__.py
    core/
      config.py
    experiments/
      __init__.py
      models.py
      service.py
    retrieval/
      engine.py
    evaluation/
      metrics.py
```

## MVP roadmap embedded in the code

The starter backend includes:

- health and overview endpoints
- experiment creation flow
- sample experiment list
- dashboard UI for comparing a few pipeline shapes

This is intentionally a strong foundation for the eventual full evaluation engine.
