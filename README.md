# HP Docs Chatbot

A chatbot that answers questions about two public HP documents using Retrieval-Augmented Generation:

- HP ENVY 6000 All-in-One series User Guide
- OMEN 17.3 inch Gaming Laptop PC Maintenance and Service Guide

> Status: Phase 0 (foundation). Ingestion, retrieval and chat are not implemented yet.

## Quick start

Requirements: Docker with Compose v2.

```bash
cp .env.example .env
docker compose up --build
```

- Frontend: http://localhost:8501
- API: http://localhost:8000 (`/health`, `/ready`)

## Development

Requirements: Python 3.12, [uv](https://docs.astral.sh/uv/), Docker.

```bash
make setup   # install dependencies and pre-commit hooks
make check   # lint + typecheck + unit tests (coverage gate) + security
make help    # all targets
```

Run the API locally without Docker (needs the variables from `.env.example`):

```bash
cd backend
uv run uvicorn app.main:create_app --factory --no-server-header
```

## Architecture

See [docs/adr](docs/adr) for the architecture decision records.

## Data and privacy

The HP PDFs are copyrighted by HP and are not part of this repository; ingestion downloads them from their public URLs and verifies pinned checksums.

## License

MIT, see [LICENSE](LICENSE).
