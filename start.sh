#!/usr/bin/env bash
# Komute-Local: start the verification node in the foreground (Linux).
#
# systemd runs this script (service "komute", installed by setup.sh). Use systemctl, not this script, on a server:
#   sudo systemctl start|stop|restart|status komute
#   journalctl -u komute -f
# To run it by hand (service stopped first):  sudo -u komute /opt/komute/app/start.sh
#
# Settings come from /etc/komute/komute.env (or the file in $KOMUTE_ENV_FILE). One worker only:
# one verification at a time per node; a second call gets 503 with Retry-After.
set -euo pipefail

cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
ENV_FILE=${KOMUTE_ENV_FILE:-/etc/komute/komute.env}
PY=${KOMUTE_PYTHON:-/opt/komute/venv/bin/python}

fail() { printf 'ERROR: %s\n' "$1" >&2; exit 1; }

[ -r "$ENV_FILE" ] || fail "Cannot read $ENV_FILE. Run: sudo ./setup.sh"
set -a
# shellcheck disable=SC1090
. "$ENV_FILE"
set +a
[ -x "$PY" ] || fail "Python not found at $PY. Run: sudo ./setup.sh"

export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUTF8=1
export MALLOC_ARENA_MAX=2   # glibc: fewer malloc arenas keeps memory flat across requests

# Refuse to start with a changed or missing model file.
"$PY" -c 'from pathlib import Path
from backend.core.registry import verify_lock
verify_lock(Path("backend/models"), Path("backend/models/models.lock"))' \
  || fail "A model file is missing or does not match backend/models/models.lock. Run: sudo ./setup.sh"

HOST=${KOMUTE_HOST:-127.0.0.1}
PORT=${KOMUTE_PORT:-8000}
printf 'Starting Komute-Local on http://%s:%s (env=%s)\n' "$HOST" "$PORT" "${KOMUTE_ENV:-dev}"
exec "$PY" -m uvicorn backend.app.main:create_app --factory --host "$HOST" --port "$PORT" --workers 1
