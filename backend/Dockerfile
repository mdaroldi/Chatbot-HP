# syntax=docker/dockerfile:1

# Base images pinned by digest; Dependabot proposes updates.
ARG PYTHON_IMAGE=python:3.12.16-slim-trixie@sha256:5e2bdb1b0d542d5039cffb8181c65930093949194a65c365dfb1b0fe13a4fa5b
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.11.3@sha256:90bbb3c16635e9627f49eec6539f956d70746c409209041800a0280b93152823

FROM ${UV_IMAGE} AS uv

# ---------------------------------------------------------------- builder
FROM ${PYTHON_IMAGE} AS builder
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv
WORKDIR /build
COPY pyproject.toml uv.lock .python-version ./
# Runtime dependencies only, exactly as locked.
RUN uv sync --frozen --no-dev --no-install-project

# ---------------------------------------------------------------- models
# Every model in models.toml, fetched at its pinned revision and checked file by file
# (ADR 0008): a changed, missing or extra file fails the build. Only the modules the
# store needs are copied, so code changes elsewhere do not rerun this step. The cache
# mount keeps downloads between local builds; the store itself holds real files.
FROM ${PYTHON_IMAGE} AS models
COPY --from=builder /opt/venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH \
    HF_HUB_DISABLE_TELEMETRY=1 \
    HF_HUB_DISABLE_PROGRESS_BARS=1
WORKDIR /build
COPY models.toml ./
COPY app/__init__.py ./app/
COPY app/domain/__init__.py app/domain/ingestion.py ./app/domain/
COPY app/infra/__init__.py app/infra/model_files.py app/infra/model_store.py ./app/infra/
RUN --mount=type=cache,target=/root/.cache/hpdocs-models \
    python -c "import sys; from app.infra.model_store import main; sys.exit(main(sys.argv[1:]))" \
        --cache /root/.cache/hpdocs-models --target /opt/hpdocs/models

# ---------------------------------------------------------------- runtime
FROM ${PYTHON_IMAGE} AS runtime
# Models are read from the baked store only: the hub library is offline and silent,
# and onnxruntime's telemetry is off (compose sets the same, so either holds alone).
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    EMBEDDING_CACHE_DIR=/opt/hpdocs/models \
    HF_HUB_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    HF_HUB_DISABLE_PROGRESS_BARS=1 \
    ORT_DISABLE_TELEMETRY=1
# 1. Apply pending Debian security updates. Added for CVE-2026-76844 and
#    CVE-2026-85091 (zlib1g; fixed in 1:1.3.dfsg+really1.3.1-1+deb13u1), which the
#    pinned base image still ships. Remove the `apt-get update/upgrade` lines once
#    the PYTHON_IMAGE digest is bumped to a base that already contains that zlib1g
#    or newer, and `make scan-images` passes without them.
# 2. Drop pip and ensurepip's bundled wheels: no Python package managers at runtime.
# 3. Add the numeric non-root user the container runs as.
# 4. Create the corpus cache mount point owned by that user: a named volume copies
#    its mount point's owner on first use, and a root-owned one is not writable.
RUN apt-get update \
    && apt-get upgrade --yes --no-install-recommends \
    && rm -rf /var/lib/apt/lists/* \
    && python -m pip uninstall --yes pip \
    && rm -rf /usr/local/lib/python3.12/ensurepip \
    && groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid 10001 --no-create-home --shell /usr/sbin/nologin app \
    && mkdir -p /var/cache/hpdocs/corpus \
    && chown 10001:10001 /var/cache/hpdocs/corpus
COPY --from=builder /opt/venv /opt/venv
# Owned by root: the app user can read the models, never change them.
COPY --from=models /opt/hpdocs/models /opt/hpdocs/models
WORKDIR /app
COPY alembic.ini ./
COPY alembic ./alembic
COPY app ./app
# The corpus allowlist and the model pins, read from the directory above app/.
COPY corpus.toml models.toml ./
COPY --chmod=0755 docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
USER 10001:10001
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2).status == 200 else 1)"]
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
