#!/usr/bin/env bash
# Heimdal installation helper for a fresh Ubuntu (native or WSL2) environment.
# Ensures Python 3.11+ and pip are present, then runs the developer setup.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if ! command -v python3 >/dev/null 2>&1; then
  echo "[heimdal] installing python3 (requires sudo)"
  sudo apt-get update
  sudo apt-get install -y python3 python3-venv python3-pip
fi

PY_OK="$(python3 -c 'import sys; print(1 if sys.version_info >= (3, 11) else 0)')"
if [ "$PY_OK" != "1" ]; then
  echo "[heimdal] Python 3.11+ is required; please install it and re-run." >&2
  exit 1
fi

echo "[heimdal] Ollama is optional. Install from https://ollama.com if you want"
echo "[heimdal] a local model backend; Heimdal runs offline without it."

exec bash "$REPO_ROOT/scripts/dev_setup.sh"
