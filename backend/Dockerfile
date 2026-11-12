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

# ---------------------------------------------------------------- runtime
FROM ${PYTHON_IMAGE} AS runtime
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
# No package managers at runtime: drop pip and ensurepip's bundled wheels; add the
# numeric non-root user the container runs as.
RUN python -m pip uninstall --yes pip \
    && rm -rf /usr/local/lib/python3.12/ensurepip \
    && groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid 10001 --no-create-home --shell /usr/sbin/nologin app
COPY --from=builder /opt/venv /opt/venv
WORKDIR /app
COPY alembic.ini ./
COPY alembic ./alembic
COPY app ./app
COPY --chmod=0755 docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
USER 10001:10001
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2).status == 200 else 1)"]
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
