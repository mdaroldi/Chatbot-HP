#!/bin/sh
# Replace this shell with uvicorn so it receives signals directly (compose
# `init: true` adds a minimal init as PID 1 to forward them). Migrations run in the
# one-shot `migrate` service before the backend starts (ADR 0007).
set -eu

exec uvicorn app.main:create_app --factory --no-server-header --host 0.0.0.0 --port 8000
