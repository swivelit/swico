#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [ ! -x .venv/bin/python ]; then
  echo "Training environment is missing. Run ./setup_vm.sh first."
  exit 1
fi

export SWICO_CPU_THREADS="${SWICO_CPU_THREADS:-8}"
export OMP_NUM_THREADS="$SWICO_CPU_THREADS"
export MKL_NUM_THREADS="$SWICO_CPU_THREADS"
export OPENBLAS_NUM_THREADS="$SWICO_CPU_THREADS"
export NUMEXPR_NUM_THREADS="$SWICO_CPU_THREADS"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_TELEMETRY=1
export PYTHONUNBUFFERED=1
export MALLOC_ARENA_MAX=2
export HF_HOME="$ROOT_DIR/.hf_cache"

PROFILE="${SWICO_PROFILE:-vm}"
exec .venv/bin/python train_vm.py --profile "$PROFILE" --resume "$@"
