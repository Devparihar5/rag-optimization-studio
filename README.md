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

Open **Models** to connect a generation model before creating an experiment.

- **OpenAI** and **Google Gemini**: enter an API key, select **Verify connection & load models**, and choose from models returned by that provider for the supplied key. Gemini models are limited to those advertising `generateContent` support; non-generation OpenAI model families are filtered out.
- **Ollama (local)**: run Ollama on your machine, install a model with `ollama pull <model>`, then discover models at `http://localhost:11434` (or `http://host.docker.internal:11434` when the app runs in a container).
- Select a configured model in the experiment form. The selector stores the configured model's internal ID rather than asking users to type an arbitrary model name.
- Provider API keys are encrypted before being stored in SQLite. The encryption key is kept in `.data/.model_encryption_key` with restrictive file permissions and is never returned by the API. Back up this file with the database if you need to preserve access to saved credentials. Protect the entire data directory and do not commit it.

Model discovery validates credentials and lists accessible models. Experiment execution is a separate roadmap item; configuring a model does not itself run an inference or benchmark.

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
