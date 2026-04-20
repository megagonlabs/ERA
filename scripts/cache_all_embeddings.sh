#!/bin/bash
# Cache embeddings for all 126 MAIR tasks for both 0.6B and 8B models
# Uses 8 GPUs in parallel: 0.6B on GPUs 0-5 (1 per category), 8B on GPUs 0-5 concurrently
set -e
cd "$(dirname "$0")/.."

source ../.venv/bin/activate

export HF_HOME="/local/scratch/${USER}/huggingface"
export HF_DATASETS_CACHE="/local/scratch/${USER}/huggingface/datasets"
mkdir -p "$HF_DATASETS_CACHE"

DOMAINS=(academic code finance legal medical web)
SMALL_MODEL="Qwen/Qwen3-Embedding-0.6B"
LARGE_MODEL="Qwen/Qwen3-Embedding-8B"
OUTPUT="results/no_adapter"

echo "=========================================="
echo " Phase 1: Caching 0.6B embeddings (GPUs 0-5)"
echo "=========================================="
pids=()
for i in "${!DOMAINS[@]}"; do
    domain="${DOMAINS[$i]}"
    gpu=$i
    echo "  GPU $gpu: 0.6B / $domain"
    CUDA_VISIBLE_DEVICES=$gpu python scripts/evaluate_zero_shot.py \
        --model_name "$SMALL_MODEL" \
        --model_type hf \
        --mair_category "$domain" \
        --output_dir "$OUTPUT" \
        > "logs/cache_0.6B_${domain}.log" 2>&1 &
    pids+=($!)
done

echo "Waiting for 0.6B caching to complete..."
for pid in "${pids[@]}"; do wait "$pid"; done
echo "0.6B caching done."

echo "=========================================="
echo " Phase 2: Caching 8B embeddings (GPUs 0-5)"
echo "=========================================="
pids=()
for i in "${!DOMAINS[@]}"; do
    domain="${DOMAINS[$i]}"
    gpu=$i
    echo "  GPU $gpu: 8B / $domain"
    CUDA_VISIBLE_DEVICES=$gpu python scripts/evaluate_zero_shot.py \
        --model_name "$LARGE_MODEL" \
        --model_type hf \
        --mair_category "$domain" \
        --output_dir "$OUTPUT" \
        > "logs/cache_8B_${domain}.log" 2>&1 &
    pids+=($!)
done

echo "Waiting for 8B caching to complete..."
for pid in "${pids[@]}"; do wait "$pid"; done
echo "8B caching done."

echo "All embeddings cached."
