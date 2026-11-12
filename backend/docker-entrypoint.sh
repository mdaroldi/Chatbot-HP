#!/bin/sh
# Apply migrations, then replace this shell with uvicorn so it receives signals
# directly (compose `init: true` adds a minimal init as PID 1 to forward them).
set -eu

alembic -c /app/alembic.ini upgrade head
exec uvicorn app.main:create_app --factory --no-server-header --host 0.0.0.0 --port 8000
