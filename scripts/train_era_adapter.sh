#!/bin/bash
# Unified adapter training script
# This script runs the unified training pipeline with different modes
# Eval modes: in_domain, out_of_domain, only_domain

set -e

cd "$(dirname "$0")/.."

# Default values (all overridable via env vars or CLI flags)
TRAINING_MODE="${TRAINING_MODE:-both}" # Options: alignment_only, label_only, both
TASKS="${TASKS:-}"
LABEL_TRAIN_RATIOS="${LABEL_TRAIN_RATIOS:-0.1}"
LABEL_VAL_RATIO="${LABEL_VAL_RATIO:-0.1}"
LABEL_EVAL_RATIO="${LABEL_EVAL_RATIO:-0.5}"
LABEL_NEGATIVE_STRATEGY="${LABEL_NEGATIVE_STRATEGY:-topk_percpos}"
LABEL_NUM_NEGATIVES="${LABEL_NUM_NEGATIVES:-5}"
LABEL_HARD_NEGATIVE_TOP_K="${LABEL_HARD_NEGATIVE_TOP_K:-2000}"
LABEL_HARD_NEGATIVE_PERC_MARGIN="${LABEL_HARD_NEGATIVE_PERC_MARGIN:-0.95}"
LABEL_LR="${LABEL_LR:-1e-5}"
LABEL_WEIGHT_DECAY="${LABEL_WEIGHT_DECAY:-1e-4}"
EVAL_AFTER_TRAINING="${EVAL_AFTER_TRAINING:-true}"
LARGE_MODEL="${LARGE_MODEL:-Qwen/Qwen3-Embedding-8B}"
SMALL_MODEL="${SMALL_MODEL:-Qwen/Qwen3-Embedding-0.6B}"
EVAL_MODE="${EVAL_MODE:-in_domain}"
HOLDOUT_DOMAIN="${HOLDOUT_DOMAIN:-}"
TRAIN_DOMAINS="${TRAIN_DOMAINS:-}"
PRETRAINED_ADAPTER_PATH="${PRETRAINED_ADAPTER_PATH:-}"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)
            TRAINING_MODE="$2"
            shift 2
            ;;
        --tasks)
            TASKS="$2"
            shift 2
            ;;
        --train-ratio|--train-ratios)
            LABEL_TRAIN_RATIOS="$2"
            shift 2
            ;;
        --eval-ratio)
            LABEL_EVAL_RATIO="$2"
            shift 2
            ;;
        --val-ratio)
            LABEL_VAL_RATIO="$2"
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
        --label-lr)
            LABEL_LR="$2"
            shift 2
            ;;
        --label-weight-decay)
            LABEL_WEIGHT_DECAY="$2"
            shift 2
            ;;
        --eval)
            EVAL_AFTER_TRAINING="true"
            shift
            ;;
        --large-model)
            LARGE_MODEL="$2"
            shift 2
            ;;
        --small-model)
            SMALL_MODEL="$2"
            shift 2
            ;;
        --eval-mode)
            EVAL_MODE="$2"
            shift 2
            ;;
        --holdout-domain)
            HOLDOUT_DOMAIN="$2"
            shift 2
            ;;
        --train-domains)
            TRAIN_DOMAINS="$2"
            shift 2
            ;;
        --pretrained-adapter-path)
            PRETRAINED_ADAPTER_PATH="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Convert LABEL_TRAIN_RATIOS and LABEL_NUM_NEGATIVES to arrays
read -ra RATIO_ARRAY <<< "${LABEL_TRAIN_RATIOS}"
read -ra NUM_NEGATIVES_ARRAY <<< "${LABEL_NUM_NEGATIVES}"

echo "============================================"
echo "Unified Adapter Training"
echo "============================================"
echo "Large model: ${LARGE_MODEL}"
echo "Small model: ${SMALL_MODEL}"
echo "Mode: ${TRAINING_MODE}"
echo "Tasks: ${TASKS:-ALL}"
echo "Label train ratio(s): ${LABEL_TRAIN_RATIOS}"
echo "Label eval ratio: ${LABEL_EVAL_RATIO}"
echo "Label val ratio: ${LABEL_VAL_RATIO}"
echo "Label negative strategy: ${LABEL_NEGATIVE_STRATEGY}"
echo "Label # negatives: ${LABEL_NUM_NEGATIVES} (sweep: ${NUM_NEGATIVES_ARRAY[*]})"
if [[ "${LABEL_NEGATIVE_STRATEGY}" == "naive_topk" || "${LABEL_NEGATIVE_STRATEGY}" == "topk_percpos" ]]; then
    echo "Label hard-negative top-k: ${LABEL_HARD_NEGATIVE_TOP_K}"
    if [[ "${LABEL_NEGATIVE_STRATEGY}" == "naive_topk" ]]; then
        echo "Label hard-negative selection: take the top ${LABEL_NUM_NEGATIVES} directly"
    fi
    if [[ "${LABEL_NEGATIVE_STRATEGY}" == "topk_percpos" ]]; then
        echo "Label hard-negative sampling pool: top-$((2 * LABEL_NUM_NEGATIVES)) after threshold"
        echo "Label hard-negative perc margin: ${LABEL_HARD_NEGATIVE_PERC_MARGIN}"
    fi
fi
echo "Label learning rate: ${LABEL_LR}"
echo "Label weight decay: ${LABEL_WEIGHT_DECAY}"
echo "Eval after training: ${EVAL_AFTER_TRAINING}"
echo "Eval mode: ${EVAL_MODE}"
if [[ -n "${HOLDOUT_DOMAIN}" ]]; then
    echo "Holdout domain: ${HOLDOUT_DOMAIN}"
fi
if [[ -n "${TRAIN_DOMAINS}" ]]; then
    echo "Train domains: ${TRAIN_DOMAINS}"
fi
if [[ -n "${PRETRAINED_ADAPTER_PATH}" ]]; then
    echo "Pretrained adapter: ${PRETRAINED_ADAPTER_PATH}"
fi
echo "============================================"

# Build tasks argument (omit if not set to use all tasks)
TASKS_ARG=""
if [[ -n "${TASKS}" ]]; then
    TASKS_ARG="--tasks ${TASKS}"
fi

# Build eval argument
EVAL_ARG=""
if [[ "${EVAL_AFTER_TRAINING}" == "true" ]]; then
    EVAL_ARG="--eval-after-training"
fi

# Build eval-mode arguments
EVAL_MODE_ARGS=""
if [[ "${EVAL_MODE}" != "in_domain" ]]; then
    EVAL_MODE_ARGS="--eval-mode ${EVAL_MODE}"
fi
if [[ -n "${HOLDOUT_DOMAIN}" ]]; then
    EVAL_MODE_ARGS="${EVAL_MODE_ARGS} --holdout-domain ${HOLDOUT_DOMAIN}"
fi
if [[ -n "${TRAIN_DOMAINS}" ]]; then
    EVAL_MODE_ARGS="${EVAL_MODE_ARGS} --train-domains ${TRAIN_DOMAINS}"
fi

PRETRAINED_ARG=""
if [[ -n "${PRETRAINED_ADAPTER_PATH}" ]]; then
    PRETRAINED_ARG="--pretrained-adapter-path ${PRETRAINED_ADAPTER_PATH}"
fi

# Run training for each combination of train ratio and num negatives
for RATIO in "${RATIO_ARRAY[@]}"; do
    for N_NEG in "${NUM_NEGATIVES_ARRAY[@]}"; do
        echo ""
        echo "============================================"
        echo "Running with train ratio: ${RATIO} (${RATIO_ARRAY[*]}), num negatives: ${N_NEG} (${NUM_NEGATIVES_ARRAY[*]})"
        echo "============================================"

        python scripts/train_era_adapter.py \
            ${TASKS_ARG} \
            --large-model ${LARGE_MODEL} \
            --small-model ${SMALL_MODEL} \
            --training-mode ${TRAINING_MODE} \
            --label-train-ratio ${RATIO} \
            --label-eval-ratio ${LABEL_EVAL_RATIO} \
            --label-val-ratio ${LABEL_VAL_RATIO} \
            --label-negative-strategy ${LABEL_NEGATIVE_STRATEGY} \
            --label-num-negatives ${N_NEG} \
            --label-hard-negative-top-k ${LABEL_HARD_NEGATIVE_TOP_K} \
            --label-hard-negative-perc-margin ${LABEL_HARD_NEGATIVE_PERC_MARGIN} \
            --label-lr ${LABEL_LR} \
            --label-weight-decay ${LABEL_WEIGHT_DECAY} \
            --alignment-epochs 100 \
            --label-epochs 1000 \
            --batch-size 256 \
            --checkpoint-interval 10 \
            --output-dir results/era \
            ${PRETRAINED_ARG} \
            ${EVAL_ARG} \
            ${EVAL_MODE_ARGS}

        echo "============================================"
        echo "Completed train ratio: ${RATIO}, num negatives: ${N_NEG}"
        echo "============================================"
    done
done
