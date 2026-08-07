#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [ ! -x .venv/bin/python ]; then
  echo "Training environment is missing. Run ./setup_vm.sh first."
  exit 1
fi

export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_TELEMETRY=1
export PYTHONUNBUFFERED=1
export MALLOC_ARENA_MAX=2
export HF_HOME="${HF_HOME:-$ROOT_DIR/.hf_cache}"

HAS_ENV_FILE=false
for argument in "$@"; do
  case "$argument" in
    --env-file|--env-file=*)
      HAS_ENV_FILE=true
      break
      ;;
  esac
done

if [ "$HAS_ENV_FILE" = true ]; then
  exec .venv/bin/python qwen_train_vm.py "$@"
fi

ENV_FILE="${SWICO_QWEN_ENV_FILE:-$ROOT_DIR/qwen_training.env}"
if [ ! -f "$ENV_FILE" ]; then
  echo "Qwen training configuration file is missing: $ENV_FILE"
  echo "Copy qwen_training.env.example to qwen_training.env, then edit the values."
  exit 1
fi

exec .venv/bin/python qwen_train_vm.py --env-file "$ENV_FILE" "$@"
