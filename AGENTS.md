# Repository Guidelines

## Project Structure & Module Organization
The FastAPI backend lives in `app/`, with ingestion logic in `processing.py` and format-specific transforms under `processors/`. Shared settings load from `config.yml`, and SQLite artifacts plus uploads land in `data/` (`data/uploads/` is created on boot). Runtime logs roll to `logs/`. Copy `.env.example` to `.env` and set `INGEST_TOKEN` before ingest testing. The Next.js dashboard sits in `frontend/`, with routes under `frontend/src/app/` and reusable UI in `frontend/src/components/`. Supporting configs live beside Docker assets (`docker-compose.yml`, `Caddyfile`) and screenshots in `screenshots/` illustrate expected UI. Python tests reside in `tests/`.

## Build, Test, and Development Commands
From the repo root run `python -m venv .venv && source .venv/bin/activate` followed by `pip install -r requirements.txt` to set up the API. Use `uvicorn app.main:app --reload` for a live-reloading server. `pytest` executes the backend suite with verbose output. For full-stack validation prefer `docker compose up -d` to launch API, web, and Caddy together. Inside `frontend/`, install dependencies with `pnpm install`; `pnpm dev` serves the dashboard, `pnpm build` produces a production bundle, and `pnpm start` runs the compiled app.

## Coding Style & Naming Conventions
Python modules use snake_case filenames and 4-space indentation. Follow PEP 8 with explicit type hints and docstrings mirroring the existing endpoints (`ingest`, `health`). Keep logging via the shared `logging` logger rather than `print`. Frontend files adopt PascalCase for React components, camelCase for hooks/utilities, and Tailwind utility classes defined in `components.json`. Run `pnpm lint` and `pnpm type-check` before pushing; fix issues instead of suppressing rules.

## Testing Guidelines
Add PyTest cases under `tests/`, mirroring the import routing fixtures already present. Use descriptive `test_*` function names and mark longer flows with `@pytest.mark.slow` or `integration` so they can be filtered. Ensure new processor logic persists data under `data/` deterministically to keep tests hermetic. When touching the frontend, at minimum rerun `pnpm lint`, `pnpm type-check`, and verify dashboard flows against a local export.

## Commit & Pull Request Guidelines
Commits in history are short, imperative phrases (“fix types”, “volume & max weight ...”). Match that tone, optionally appending issue numbers. Each PR should include: a concise summary, screenshots for UI shifts, notes on migrations or data resets, and a checklist of commands run (`pytest`, `pnpm lint`, `docker compose up`). Reference any related GitHub issues and call out follow-up tasks to ease review.
