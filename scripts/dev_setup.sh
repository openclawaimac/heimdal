#!/usr/bin/env bash
# Heimdal developer setup for Ubuntu native and WSL2.
# Creates a virtualenv, installs Heimdal, and runs the doctor.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-python3}"
echo "[heimdal] using $($PYTHON --version)"

if [ ! -d ".venv" ]; then
  echo "[heimdal] creating virtualenv .venv"
  "$PYTHON" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

echo "[heimdal] installing dependencies"
pip install --upgrade pip >/dev/null
pip install -e .

# WSL2 note: keep the storage tree on the Linux filesystem, not /mnt/c.
if grep -qiE "microsoft|wsl" /proc/version 2>/dev/null; then
  echo "[heimdal] WSL2 detected - keep ./storage on the Linux filesystem"
fi

echo "[heimdal] running doctor"
heimdal doctor

echo "[heimdal] setup complete. Try: heimdal run demo"
