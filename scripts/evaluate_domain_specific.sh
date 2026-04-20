#!/bin/bash
# Re-evaluate existing only-domain adapters on ALL MAIR domains.
#
# This script uses pre-trained only-domain adapter weights
# (from run_only_domain_training.sh) and evaluates them on ALL 126 tasks
# — not just the training domain.
#
# For training domain tasks: uses eval-split queries only (no data leakage)
# For non-training domains:  uses all queries (never seen in training)
#
# Results are saved in eval_results_all_domains/ next to existing eval_results/.
#
# Multi-GPU parallelism: by default all available GPUs are used.
# Each domain experiment is dispatched to a separate process pinned to one GPU.
# With 8 GPUs and 11 domains, all evaluations finish in roughly 2 sequential waves.
#
# Usage:
#   # Evaluate all only-domain experiments using all available GPUs (auto-detect)
#   bash scripts/evaluate_domain_specific.sh --train-ratio 0.2
#
#   # Use only 4 GPUs
#   bash scripts/evaluate_domain_specific.sh --train-ratio 0.2 --num-workers 4
#
#   # Evaluate specific domains only
#   bash scripts/evaluate_domain_specific.sh --domains "code web finance"
#
#   # Resume interrupted evaluation
#   bash scripts/evaluate_domain_specific.sh --train-ratio 0.2 --resume
#
#   # Dry run: show what would be done
#   bash scripts/evaluate_domain_specific.sh --dry-run

set -e

cd "$(dirname "$0")/.."

# ============== Default values ==============
LARGE_MODEL="${LARGE_MODEL:-Qwen/Qwen3-Embedding-8B}"
SMALL_MODEL="${SMALL_MODEL:-Qwen/Qwen3-Embedding-0.6B}"
TRAINING_MODE="${TRAINING_MODE:-both}"
LABEL_NEGATIVE_STRATEGY="${LABEL_NEGATIVE_STRATEGY:-topk_percpos}"
LABEL_NUM_NEGATIVES="${LABEL_NUM_NEGATIVES:-5}"
LABEL_HARD_NEGATIVE_TOP_K="${LABEL_HARD_NEGATIVE_TOP_K:-2000}"
LABEL_HARD_NEGATIVE_PERC_MARGIN="${LABEL_HARD_NEGATIVE_PERC_MARGIN:-0.95}"
OUTPUT_DIR="${OUTPUT_DIR:-results/era}"
LABEL_TRAIN_RATIOS="${LABEL_TRAIN_RATIOS:-${TRAIN_RATIO:-}}"
NUM_WORKERS="${NUM_WORKERS:-}"
EXTRA_ARGS=""

# ============== Parse arguments ==============
while [[ $# -gt 0 ]]; do
    case $1 in
        --domains)
            EXTRA_ARGS="${EXTRA_ARGS} --domains $2"
            shift 2
            ;;
        --train-ratio)
            LABEL_TRAIN_RATIOS="$2"
            shift 2
            ;;
        --train-ratios)
            LABEL_TRAIN_RATIOS="$2"
            shift 2
            ;;
        --mode|--training-mode)
            TRAINING_MODE="$2"
            shift 2
            ;;
        --label-negative-strategy)
            LABEL_NEGATIVE_STRATEGY="$2"
            shift 2
            ;;
        --label-num-negatives)
            LABEL_NUM_NEGATIVES="$2"
            shift 2
            ;;
        --label-hard-negative-top-k)
            LABEL_HARD_NEGATIVE_TOP_K="$2"
            shift 2
            ;;
        --label-hard-negative-perc-margin)
            LABEL_HARD_NEGATIVE_PERC_MARGIN="$2"
            shift 2
            ;;
        --large-model)
            LARGE_MODEL="$2"
            shift 2
            ;;
        --small-model)
            SMALL_MODEL="$2"
            shift 2
            ;;
        --resume)
            EXTRA_ARGS="${EXTRA_ARGS} --resume"
            shift
            ;;
        --dry-run)
            EXTRA_ARGS="${EXTRA_ARGS} --dry-run"
            shift
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --num-workers)
            NUM_WORKERS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: bash scripts/evaluate_domain_specific.sh [OPTIONS]"
            exit 1
            ;;
    esac
done

# Convert train ratios to array
read -ra RATIO_ARRAY <<< "${LABEL_TRAIN_RATIOS}"
NUM_RATIOS=${#RATIO_ARRAY[@]}

WORKERS_ARG=""
if [[ -n "${NUM_WORKERS}" ]]; then
    WORKERS_ARG="--num-workers ${NUM_WORKERS}"
fi

echo "============================================================"
echo "  Only-Domain Full Evaluation"
echo "============================================================"
echo "  Large model:     ${LARGE_MODEL}"
echo "  Small model:     ${SMALL_MODEL}"
echo "  Training mode:   ${TRAINING_MODE}"
echo "  Output dir:      ${OUTPUT_DIR}"
echo "  Train ratio(s):  ${LABEL_TRAIN_RATIOS:-any}"
echo "  Neg strategy:    ${LABEL_NEGATIVE_STRATEGY}"
echo "  Num negatives:   ${LABEL_NUM_NEGATIVES}"
if [[ "${LABEL_NEGATIVE_STRATEGY}" == "naive_topk" || "${LABEL_NEGATIVE_STRATEGY}" == "topk_percpos" ]]; then
    echo "  Hard-neg top-k:  ${LABEL_HARD_NEGATIVE_TOP_K}"
fi
if [[ "${LABEL_NEGATIVE_STRATEGY}" == "topk_percpos" ]]; then
    echo "  Hard-neg margin: ${LABEL_HARD_NEGATIVE_PERC_MARGIN}"
fi
echo "  Num workers:     ${NUM_WORKERS:-auto (all GPUs)}"
echo "============================================================"

if [[ ${NUM_RATIOS} -eq 0 ]]; then
    # No ratio filter: evaluate all experiments regardless of train ratio
    python scripts/evaluate_domain_specific.py \
        --large-model "${LARGE_MODEL}" \
        --small-model "${SMALL_MODEL}" \
        --output-dir "${OUTPUT_DIR}" \
        --training-mode "${TRAINING_MODE}" \
        --label-negative-strategy "${LABEL_NEGATIVE_STRATEGY}" \
        --label-num-negatives "${LABEL_NUM_NEGATIVES}" \
        --label-hard-negative-top-k "${LABEL_HARD_NEGATIVE_TOP_K}" \
        --label-hard-negative-perc-margin "${LABEL_HARD_NEGATIVE_PERC_MARGIN}" \
        ${WORKERS_ARG} \
        ${EXTRA_ARGS}
else
    for RATIO in "${RATIO_ARRAY[@]}"; do
        echo ""
        echo "============================================================"
        echo "  Train ratio: ${RATIO}"
        echo "============================================================"
        python scripts/evaluate_domain_specific.py \
            --large-model "${LARGE_MODEL}" \
            --small-model "${SMALL_MODEL}" \
            --output-dir "${OUTPUT_DIR}" \
            --training-mode "${TRAINING_MODE}" \
            --label-negative-strategy "${LABEL_NEGATIVE_STRATEGY}" \
            --label-num-negatives "${LABEL_NUM_NEGATIVES}" \
            --label-hard-negative-top-k "${LABEL_HARD_NEGATIVE_TOP_K}" \
            --label-hard-negative-perc-margin "${LABEL_HARD_NEGATIVE_PERC_MARGIN}" \
            --train-ratio "${RATIO}" \
            ${WORKERS_ARG} \
            ${EXTRA_ARGS}
    done
fi

echo ""
echo "============================================================"
echo "  Only-Domain Full Evaluation Complete"
echo "============================================================"
