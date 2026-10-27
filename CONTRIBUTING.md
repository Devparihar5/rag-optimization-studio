# Contributing

Thanks for your interest in contributing to RAG Optimization Studio.

## Development setup

1. Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -e .[dev]
   ```

2. Start the app:
   ```bash
   uvicorn app.main:app --reload
   ```

3. Run tests:
   ```bash
   pytest
   ```

## Code standards

- Keep functions small and readable.
- Prefer type annotations.
- Avoid mutable default arguments.
- Add tests for new behavior.
- Keep the architecture layered.

## Pull requests

- Open an issue before large changes.
- Keep PRs focused.
- Include a short summary and validation steps.
