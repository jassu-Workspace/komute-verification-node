#!/usr/bin/env bash
# Komute-Local: set up an Ubuntu server (22.04 or 24.04, x86_64) as a verification node.
#
#   sudo ./setup.sh              # run from the unpacked source folder; safe to run again (upgrades in place)
#   sudo ./setup.sh --no-start   # install everything, but do not start the service
#
# What it does:
#   1. Checks the platform: root, Ubuntu, x86_64, at least 2 GB RAM, 3 GB free disk, network to GitHub and PyPI.
#   2. Installs OS packages (curl, rsync, ca-certificates) and creates the system user "komute".
#   3. Copies the code to /opt/komute/app (models and keys are never overwritten).
#   4. Installs uv, Python 3.12 (in /opt/komute/python) and the venv (/opt/komute/venv) with requirements.lock.
#   5. Downloads the four ONNX models; each must match backend/models/models.lock (SHA-256) or it is deleted.
#   6. Creates /etc/komute/komute.env once (API key, storage and manifest keys; values are never printed).
#   7. Installs the systemd service "komute" (runs start.sh), starts it and waits for GET /health.
set -euo pipefail

APP=/opt/komute/app
VENV=/opt/komute/venv
PYDIR=/opt/komute/python
DATA=/var/lib/komute
ENV_DIR=/etc/komute
ENV_FILE=$ENV_DIR/komute.env
UNIT=/etc/systemd/system/komute.service
SRC=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
START=1
for a in "$@"; do
  case "$a" in
    --no-start) START=0 ;;
    *) echo "unknown option: $a (use --no-start)" >&2; exit 2 ;;
  esac
done

step() { printf '\n==> %s\n' "$1"; }
ok()   { printf '    ok  %s\n' "$1"; }
warn() { printf '    !!  %s\n' "$1"; }
fail() { printf '\nERROR: %s\n' "$1" >&2; exit 1; }

# ---------------------------------------------------------------- 1. platform
step "Checking the platform"
[ "$(id -u)" -eq 0 ] || fail "Run as root: sudo ./setup.sh"
. /etc/os-release
[ "${ID:-}" = "ubuntu" ] || fail "This script is for Ubuntu (found: ${PRETTY_NAME:-unknown})."
case "${VERSION_ID:-}" in 22.04|24.04) ;; *) warn "Tested for Ubuntu 22.04 and 24.04; this is ${VERSION_ID:-unknown}." ;; esac
[ "$(uname -m)" = "x86_64" ] || fail "An x86_64 server is required (found $(uname -m))."
ram_mb=$(awk '/^MemTotal:/ {print int($2/1024)}' /proc/meminfo)
# 2 GB plans report about 1.9 GB after the kernel's reserve. One verification peaks near 1.2 GB.
[ "$ram_mb" -ge 1800 ] || fail "At least 2 GB RAM is required (found ${ram_mb} MB). Resize the server first."
mkdir -p /opt/komute
free_mb=$(df -Pm /opt/komute | awk 'NR==2 {print $4}')
[ "$free_mb" -ge 3072 ] || fail "At least 3 GB free disk is required in /opt (found ${free_mb} MB)."
[ -f "$SRC/backend/models/models.lock" ] && [ -f "$SRC/requirements.lock" ] && [ -f "$SRC/start.sh" ] \
  || fail "Run this script from the unpacked Komute-Local source folder (backend/, requirements.lock, start.sh)."
ok "Ubuntu ${VERSION_ID}, x86_64, ${ram_mb} MB RAM, ${free_mb} MB free"

step "Installing OS packages"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq --no-install-recommends ca-certificates curl rsync >/dev/null
for h in https://github.com https://pypi.org/simple/ https://files.pythonhosted.org; do
  curl -sS -o /dev/null --max-time 20 "$h" || fail "Cannot reach $h. Check the network or set HTTPS_PROXY."
done
ok "curl, rsync, ca-certificates; GitHub and PyPI reachable"

# ---------------------------------------------------------------- 2. user and folders
step "Creating the service user and folders"
id komute >/dev/null 2>&1 || useradd --system --home-dir "$DATA" --shell /usr/sbin/nologin komute
install -d -o root -g root -m 0755 /opt/komute "$APP"
install -d -o komute -g komute -m 0750 "$DATA" "$DATA/storage"
install -d -o root -g komute -m 0750 "$ENV_DIR"
ok "user komute; data in $DATA (only komute can read it)"

# ---------------------------------------------------------------- 3. code
step "Copying the code to $APP"
if [ "$SRC" != "$APP" ]; then
  for d in backend storage scripts; do
    # --delete removes old code; downloaded models (*.onnx) and caches are kept.
    rsync -a --delete --exclude '__pycache__' --exclude '*.onnx' "$SRC/$d/" "$APP/$d/"
  done
  install -m 0644 "$SRC/requirements.lock" "$SRC/requirements.txt" "$SRC/.env.example" "$APP/"
  install -m 0755 "$SRC/start.sh" "$SRC/setup.sh" "$APP/"
fi
chown -R root:root "$APP"
ok "code in place (private data folders are never copied)"

# ---------------------------------------------------------------- 4. uv, Python 3.12, dependencies
step "Installing uv, Python 3.12 and dependencies"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin UV_NO_MODIFY_PATH=1 sh >/dev/null
fi
command -v uv >/dev/null 2>&1 || fail "uv could not be installed. See https://docs.astral.sh/uv/"
export UV_PYTHON_INSTALL_DIR=$PYDIR  # outside /root, so the komute user can run it
uv python install 3.12 >/dev/null
if ! "$VENV/bin/python" -c 'import sys; sys.exit(sys.version_info[:2] != (3, 12))' 2>/dev/null; then
  uv venv --python 3.12 --clear "$VENV" >/dev/null
fi
PY=$VENV/bin/python
# requirements.lock: the exact tested versions (all have Linux x86_64 wheels).
uv pip install --python "$PY" --only-binary :all: -r "$APP/requirements.lock" >/dev/null
chmod -R a+rX "$PYDIR" "$VENV"
(cd "$APP" && "$PY" -c 'import backend.core, cv2, onnxruntime, zxingcpp, fastapi, uvicorn') \
  || fail "The backend does not import. See the error above."
ok "uv $(uv --version | awk '{print $2}'), Python $("$PY" -c 'import platform; print(platform.python_version())'), packages from requirements.lock"

# ---------------------------------------------------------------- 5. models
step "Installing model files (checked against models.lock)"
declare -A URL=(
  [face_detection_yunet_2023mar.onnx]=https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
  [yolo-v9-t-384-license-plates-end2end.onnx]=https://github.com/ankandrew/open-image-models/releases/download/assets/yolo-v9-t-384-license-plates-end2end.onnx
  [cct_xs_v2_global.onnx]=https://github.com/ankandrew/fast-plate-ocr/releases/download/arg-plates/cct_xs_v2_global.onnx
  [w600k_r50.onnx]=https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip
)
MODELS=$APP/backend/models
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
while IFS='=' read -r name want; do
  dest=$MODELS/$name
  if [ -f "$dest" ] && [ "$(sha256sum "$dest" | cut -d' ' -f1)" = "$want" ]; then ok "$name (present, hash matches)"; continue; fi
  url=${URL[$name]:-}
  [ -n "$url" ] || fail "$name is in models.lock but has no download source. Copy the file to $MODELS."
  printf '    ..  downloading %s\n' "$name"
  if [[ "$url" == *.zip ]]; then
    curl -fsSL --retry 3 --connect-timeout 20 -o "$TMP/m.zip" "$url"
    "$PY" -c 'import sys, zipfile, shutil
z = zipfile.ZipFile(sys.argv[1]); n = next(e for e in z.namelist() if e.split("/")[-1] == sys.argv[2])
with z.open(n) as s, open(sys.argv[3], "wb") as d: shutil.copyfileobj(s, d)' "$TMP/m.zip" "$name" "$TMP/part"
  else
    curl -fsSL --retry 3 --connect-timeout 20 -o "$TMP/part" "$url"
  fi
  got=$(sha256sum "$TMP/part" | cut -d' ' -f1)
  [ "$got" = "$want" ] || fail "$name hash mismatch (expected $want, got $got). The file was discarded."
  install -m 0644 "$TMP/part" "$dest"
  ok "$name (downloaded, hash matches)"
done < <("$PY" -c 'import json, sys
for k, v in json.load(open(sys.argv[1])).items(): print(f"{k}={v.lower()}")' "$MODELS/models.lock")

# ---------------------------------------------------------------- 6. settings and keys
step "Preparing settings"
if [ -f "$ENV_FILE" ]; then
  ok "$ENV_FILE exists (kept; values not shown)"
else
  "$PY" - "$ENV_FILE" <<'PYEOF'
import base64, secrets, sys
b64 = lambda: base64.b64encode(secrets.token_bytes(32)).decode()
lines = [
    "# Komute-Local node settings. Read by start.sh. Keep this file secret (root:komute 0640).",
    "# dev = liveness skipped (no liveness model yet). production needs KOMUTE_LIVENESS_MODEL.",
    "KOMUTE_ENV=dev",
    "KOMUTE_DEV_SKIP_LIVENESS=true",
    f"KOMUTE_API_KEY={secrets.token_urlsafe(32)}",
    f"KOMUTE_STORAGE_KEY_B64={b64()}",
    "KOMUTE_STORAGE_KEY_ID=srv-s1",
    f"KOMUTE_MANIFEST_KEY_B64={b64()}",
    "KOMUTE_MANIFEST_KEY_ID=srv-m1",
    "KOMUTE_STORAGE_ROOT=/var/lib/komute/storage",
    "KOMUTE_FACE_MATCH_THRESHOLD=0.35",
    "# Address the node listens on. 127.0.0.1 = only programs on this server can call it.",
    "KOMUTE_HOST=127.0.0.1",
    "KOMUTE_PORT=8000",
]
with open(sys.argv[1], "x", encoding="utf-8", newline="\n") as f:
    f.write("\n".join(lines) + "\n")
PYEOF
  ok "$ENV_FILE created with new keys (values not shown)"
fi
chown root:komute "$ENV_FILE"
chmod 0640 "$ENV_FILE"
runuser -u komute -- bash -c 'set -a; . "$1"; set +a; cd "$2"; exec "$3" -c "
from pathlib import Path
from backend.core.registry import verify_lock
from backend.core.config import Settings
verify_lock(Path(\"backend/models\"), Path(\"backend/models/models.lock\")); Settings()"' _ "$ENV_FILE" "$APP" "$PY" \
  || fail "Model lock or settings check failed. See the error above."
ok "model hashes and settings valid"

# ---------------------------------------------------------------- 7. service
step "Installing the systemd service"
cat > "$UNIT" <<UNITEOF
[Unit]
Description=Komute-Local verification node
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=komute
Group=komute
WorkingDirectory=$APP
ExecStart=$APP/start.sh
Restart=on-failure
RestartSec=5
# Memory budget: one verification peaks near 1.2 GB.
MemoryMax=1300M
NoNewPrivileges=true
PrivateTmp=true
ProtectHome=true
ProtectSystem=strict
ReadWritePaths=$DATA

[Install]
WantedBy=multi-user.target
UNITEOF
systemctl daemon-reload
systemctl enable komute >/dev/null 2>&1
ok "service komute enabled (starts at boot)"

if [ "$START" -eq 0 ]; then
  printf '\nSetup complete. Start the node with: sudo systemctl start komute\n'
  exit 0
fi
systemctl restart komute
port=$(awk -F= '/^KOMUTE_PORT=/ {print $2}' "$ENV_FILE"); port=${port:-8000}
for _ in $(seq 1 90); do
  if curl -fsS "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
    ok "node healthy at http://127.0.0.1:$port"
    printf '\nSetup complete.\n  Logs:     journalctl -u komute -f\n  API key:  sudo grep ^KOMUTE_API_KEY= %s\n' "$ENV_FILE"
    exit 0
  fi
  systemctl is-active --quiet komute || break
  sleep 2
done
journalctl -u komute -n 40 --no-pager || true
fail "The node did not become healthy. See the log above."
