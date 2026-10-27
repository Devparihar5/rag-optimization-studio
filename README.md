# RAG Optimization Studio

A single-server experimentation platform for benchmarking and improving retrieval-augmented generation (RAG) pipelines.

This repository starts with a working FastAPI app that serves both the API and the web dashboard from one process. That keeps the first implementation simple while still matching the product direction described in the project plan.

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

## Tech stack

- Python + FastAPI
- Jinja2 templates for the dashboard
- Static frontend assets served by the same app
- Pydantic models for experiment configs

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
```

## MVP roadmap embedded in the code

The starter backend includes:

- health and overview endpoints
- experiment creation flow
- sample experiment list
- dashboard UI for comparing a few pipeline shapes

This is intentionally a strong foundation for the eventual full evaluation engine.
