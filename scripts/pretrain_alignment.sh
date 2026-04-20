#!/usr/bin/env bash
set -euo pipefail

# Run identical text alignment (linear adapter)
# Usage:
#   ./scripts/pretrain_alignment.sh \
#     --large-model Qwen/Qwen3-Embedding-8B \
#     --small-model Qwen/Qwen3-Embedding-0.6B \
#     --task NFCorpus \
#     --sample-size 5000 \
#     --eval-ratio 0.1 \
#     --batch-size 64 \
#     --epochs 5

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

uv run python scripts/pretrain_alignment.py "$@"
