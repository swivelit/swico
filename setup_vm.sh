#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if ! python3 -m venv --help >/dev/null 2>&1; then
  echo "python3-venv is missing. Run: sudo apt-get update && sudo apt-get install -y python3-venv"
  exit 1
fi

FREE_KB="$(df -Pk . | awk 'NR==2 {print $4}')"
if [ "$FREE_KB" -lt 5242880 ]; then
  echo "At least 5 GiB of free disk is required before installing the CPU training environment."
  exit 1
fi

python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade --no-cache-dir pip setuptools wheel
python -m pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install --no-cache-dir -r requirements-cpu.txt

python - <<'PY'
import importlib

packages = [
    "torch",
    "sentence_transformers",
    "datasets",
    "transformers",
    "faiss",
    "pandas",
    "numpy",
    "psutil",
    "dotenv",
    "peft",
]
for package in packages:
    module = importlib.import_module(package)
    print(f"{package}: {getattr(module, '__version__', 'installed')}")
PY

python -m unittest discover -s tests -v
python verify_install.py
printf '%s\n' "CPU training environment is ready."
