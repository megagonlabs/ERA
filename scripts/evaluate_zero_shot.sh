#!/bin/bash

# MAIR Benchmark Execution Script
# Usage: 
#   bash scripts/run_mair_benchmark.sh [gpu_ids] [category] [instruction_mode] [force_recache]
#
# Arguments:
#   gpu_ids: GPU ID(s) to use (default: 0)
#            - Single GPU: 0
#            - Multiple GPUs (Data Parallel): 0,1,2,3
#   category: MAIR category to run (optional, e.g., 'math', 'code')
#   force_recache: 'force' to recompute and overwrite cache (default: use cache)
#
# Examples:
#   ./scripts/run_mair_benchmark.sh 0                    # Run default tasks with instructions on GPU 0
#   ./scripts/run_mair_benchmark.sh 0,1,2,3              # Run with data parallel on 4 GPUs
#   ./scripts/run_mair_benchmark.sh 0 math               # Run math category with instructions
#   ./scripts/run_mair_benchmark.sh 0,1 math            # Run math category with 2 GPUs
#   ./scripts/run_mair_benchmark.sh 0 math force         # Run with FORCE RECACHE (ignore existing cache)

# Parse arguments
GPU_IDS="${1:-0}"
CATEGORY="${2:-}"
FORCE_RECACHE="${3:-}"

# Use local disk for HuggingFace cache to avoid NFS issues
# This prevents "Device or resource busy" errors with .nfs* files
export HF_HOME="/local/scratch/${USER}/huggingface"
export HF_DATASETS_CACHE="/local/scratch/${USER}/huggingface/datasets"
export HF_HUB_CACHE="/local/scratch/${USER}/huggingface/hub"
mkdir -p "$HF_DATASETS_CACHE" "$HF_HUB_CACHE"
echo "Using local HuggingFace cache: $HF_HOME"

# Set GPU(s)
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
if [[ "$GPU_IDS" == *,* ]]; then
    NUM_GPUS=$(echo $GPU_IDS | tr ',' '\n' | wc -l)
    echo "Setting CUDA_VISIBLE_DEVICES to $GPU_IDS (Data Parallel: $NUM_GPUS GPUs)"
else
    echo "Setting CUDA_VISIBLE_DEVICES to $GPU_IDS (Single GPU)"
fi

# Determine force recache flag
if [ "$FORCE_RECACHE" == "force" ] || [ "$FORCE_RECACHE" == "recache" ]; then
    FORCE_RECACHE_FLAG="--force_recache"
    echo "Cache mode: FORCE RECACHE (ignore existing cache and recompute)"
else
    FORCE_RECACHE_FLAG=""
    echo "Cache mode: USE CACHE (use existing cache if available)"
fi

MODELS=(
    # "BAAI/bge-m3"
    # "Qwen/Qwen3-Embedding-0.6B"
    # "Qwen/Qwen3-Embedding-4B"
    # "Qwen/Qwen3-Embedding-8B"
    # "text-embedding-3-small"
    "text-embedding-3-large"
)

# ----- MAIR Task Selection -----
# Choose which tasks to run by commenting/uncommenting sections below.
# You can also use category mode by passing category as argument: bash run_mair_benchmark.sh 0 math

# Math tasks (12 tasks)
MATH_TASKS=(
    "Competition-Math"
    "ProofWiki_Proof"
    "ProofWiki_Reference"
    "Stacks_Proof"
    "Stacks_Reference"
    "Stein_Proof"
    "Stein_Reference"
    "Trench_Proof"
    "Trench_Reference"
    "TAD"
    "TAS2"
    "StackMathQA"
)

# Code tasks (10 tasks)
CODE_TASKS=(
    "APPS"
    "CodeEditSearch"
    "CodeSearchNet"
    "Conala"
    "HumanEval-X"
    "LeetCode"
    "MBPP"
    "RepoBench"
    "TLDR"
    "SWE-Bench-Lite"
)

# Finance tasks (7 tasks)
FINANCE_TASKS=(
    "Apple"
    "ConvFinQA"
    "FinQA"
    "FinanceBench"
    "HC3Finance"
    "TAT-DQA"
    "Trade-the-event"
)

# Knowledge tasks (8 tasks)
KNOWLEDGE_TASKS=(
    "AY2"
    "ELI5"
    "Fever"
    "TREx"
    "WnCw"
    "WnWi"
    "WoW"
    "zsRE"
)

# Law tasks (10 tasks)
LAW_TASKS=(
    "AILA2019-Case"
    "AILA2019-Statutes"
    "BSARD"
    "BillSum"
    "CUAD"
    "GerDaLIR"
    "LeCaRDv2"
    "LegalQuAD"
    "REGIR-EU2UK"
    "REGIR-UK2EU"
)

# Web tasks (10 tasks)
WEB_TASKS=(
    "ArguAna"
    "CQADupStack"
    "FiQA"
    "NFCorpus"
    "Quora"
    "SciDocs"
    "SciFact"
    "TopiOCQA"
    "Touche"
    "Trec-Covid"
)

# Science tasks (10 tasks)
SCIENCE_TASKS=(
    "ACORDAR"
    "CPCD"
    "ChroniclingAmericaQA"
    "Monant"
    "NTCIR"
    "PointRec"
    "ProCIS-Dialog"
    "ProCIS-Turn"
    "QuanTemp"
    "WebTableSearch"
)

# Database tasks (7 tasks)
DATABASE_TASKS=(
    "CARE"
    "MISeD" #
    "SParC"
    "SParC-SQL"
    "Spider"
    "Spider-SQL"
    "LitSearch"
)

# TREC tasks (39 tasks)
TREC_TASKS=(
    "CAsT_2019"
    "CAsT_2020"
    "CAsT_2021"
    "CAsT_2022"
    "Core_2017"
    "Microblog_2011"
    "Microblog_2012"
    "Microblog_2013"
    "Microblog_2014"
    "PrecisionMedicine_2017"
    "PrecisionMedicine_2018"
    "PrecisionMedicine_2019"
    "PrecisionMedicine-Article_2019"
    "PrecisionMedicine-Article_2020"
    "CliniDS_2014"
    "CliniDS_2015"
    "CliniDS_2016"
    "ClinicalTrials_2021"
    "ClinicalTrials_2022"
    "ClinicalTrials_2023"
    "DD_2015"
    "DD_2016"
    "DD_2017"
    "FairRanking_2020"
    "FairRanking_2021"
    "FairRanking_2022"
    "Genomics-AdHoc_2004"
    "Genomics-AdHoc_2005"
    "Genomics-AdHoc_2006"
    "Genomics-AdHoc_2007"
    "TREC-Legal_2011"
    "NeuCLIR-Tech_2023"
    "NeuCLIR_2022"
    "NeuCLIR_2023"
    "ProductSearch_2023"
    "ToT_2023"
    "ToT_2024"
)

# Tool tasks (8 tasks)
TOOL_TASKS=(
    "FoodAPI"
    "HuggingfaceAPI"
    "PytorchAPI"
    "SpotifyAPI"
    "TMDB"
    "TensorAPI"
    "ToolBench"
    "WeatherAPI"
)

# Special tasks (7 tasks)
SPECIAL_TASKS=(
    "ExcluIR"
    "Core17"
    "News21"
    "Robust04"
    "InstructIR"
    "NevIR"
    "IFEval"
)

# ----- Select which categories to run -----
# Comment/uncomment the categories you want to run
SELECTED_TASKS=()
SELECTED_TASKS+=("${MATH_TASKS[@]}")
SELECTED_TASKS+=("${CODE_TASKS[@]}")
SELECTED_TASKS+=("${FINANCE_TASKS[@]}")
SELECTED_TASKS+=("${KNOWLEDGE_TASKS[@]}")
SELECTED_TASKS+=("${LAW_TASKS[@]}")
SELECTED_TASKS+=("${WEB_TASKS[@]}")        # Default: Web category
SELECTED_TASKS+=("${SCIENCE_TASKS[@]}")
SELECTED_TASKS+=("${DATABASE_TASKS[@]}")
SELECTED_TASKS+=("${TREC_TASKS[@]}")
SELECTED_TASKS+=("${TOOL_TASKS[@]}")
SELECTED_TASKS+=("${SPECIAL_TASKS[@]}")

# Or manually specify individual tasks:
# SELECTED_TASKS=("SciFact" "ArguAna" "NFCorpus")

# Convert array to space-separated string
DEFAULT_TASKS="${SELECTED_TASKS[*]}"

OUTPUT_BASE="results"
# --------------------------------------------------

# Activate virtual environment if needed
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Show settings
echo "=================================================="
echo "MAIR Benchmark Runner (with Data Parallel Support)"
echo "=================================================="
echo ""
echo "Settings:"
echo "  GPU(s): $GPU_IDS"
if [[ "$GPU_IDS" == *,* ]]; then
    echo "  Data Parallel: ENABLED"
fi
echo "  Category: ${CATEGORY:-'(default tasks)'}"
echo ""
echo "Available categories: math, code, finance, knowledge, law, web, science, database, trec, tool, special"
echo "Note: For large models (4B+), use multiple GPUs (e.g., 0,1,2,3) for faster inference"
echo ""

for MODEL in "${MODELS[@]}"; do
    echo "=================================================="
    echo "Processing Model: $MODEL"
    echo "=================================================="

    case "$MODEL" in
        "text-embedding-3-large" | "text-embedding-3-small")
            echo "Running MAIR with OpenAI model..."
            if [ -z "$OPENAI_API_KEY" ]; then
                echo "Skip: OPENAI_API_KEY is not set."
                continue
            fi
            
            if [ -n "$CATEGORY" ]; then
                python scripts/evaluate_zero_shot.py \
                    --model_name "$MODEL" \
                    --model_type openai \
                    --benchmark mair \
                    --mair_category "$CATEGORY" \
                    --batch_size 32 \
                    --output_dir "${OUTPUT_BASE}" \
                    $FORCE_RECACHE_FLAG
            else
                python scripts/evaluate_zero_shot.py \
                    --model_name "$MODEL" \
                    --model_type openai \
                    --benchmark mair \
                    --tasks $DEFAULT_TASKS \
                    --batch_size 32 \
                    --output_dir "${OUTPUT_BASE}" \
                    $FORCE_RECACHE_FLAG
            fi
            ;;

        "BAAI/bge-m3")
            echo "Running MAIR with BGE-M3..."
            
            if [ -n "$CATEGORY" ]; then
                python scripts/evaluate_zero_shot.py \
                    --model_name "$MODEL" \
                    --model_type hf \
                    --benchmark mair \
                    --mair_category "$CATEGORY" \
                    --batch_size 64 \
                    --output_dir "${OUTPUT_BASE}" \
                    $FORCE_RECACHE_FLAG
            else
                python scripts/evaluate_zero_shot.py \
                    --model_name "$MODEL" \
                    --model_type hf \
                    --benchmark mair \
                    --tasks $DEFAULT_TASKS \
                    --batch_size 64 \
                    --output_dir "${OUTPUT_BASE}" \
                    $FORCE_RECACHE_FLAG
            fi
            ;;

        "Qwen/Qwen3-Embedding-"*)
            echo "Running MAIR with Qwen3 Embedding..."
            
            # Set batch size based on model size and number of GPUs
            # Batch size will be automatically multiplied by num_gpus in LocalHFEmbedder
            if [[ "$MODEL" == *"8B"* ]]; then
                BASE_BATCH_SIZE=4
            elif [[ "$MODEL" == *"4B"* ]]; then
                BASE_BATCH_SIZE=16
            elif [[ "$MODEL" == *"0.6B"* ]]; then
                BASE_BATCH_SIZE=64
            else
                BASE_BATCH_SIZE=16
            fi
            
            # For multi-GPU, use base batch size (will be multiplied internally)
            if [[ "$GPU_IDS" == *,* ]]; then
                BATCH_SIZE=$BASE_BATCH_SIZE
                echo "Using base batch_size=$BATCH_SIZE per GPU for $MODEL (Data Parallel)"
            else
                BATCH_SIZE=$BASE_BATCH_SIZE
                echo "Using batch_size=$BATCH_SIZE for $MODEL (Single GPU)"
            fi
            
            if [ -n "$CATEGORY" ]; then
                python scripts/evaluate_zero_shot.py \
                    --model_name "$MODEL" \
                    --model_type hf \
                    --benchmark mair \
                    --mair_category "$CATEGORY" \
                    --batch_size $BATCH_SIZE \
                    --output_dir "${OUTPUT_BASE}" \
                    $FORCE_RECACHE_FLAG
            else
                python scripts/evaluate_zero_shot.py \
                    --model_name "$MODEL" \
                    --model_type hf \
                    --benchmark mair \
                    --tasks $DEFAULT_TASKS \
                    --batch_size $BATCH_SIZE \
                    --output_dir "${OUTPUT_BASE}" \
                    $FORCE_RECACHE_FLAG
            fi
            ;;

        *)
            echo "Error: Unknown model: $MODEL"
            ;;
    esac
done

echo ""
echo "=================================================="
echo "MAIR Benchmark Complete"
echo "=================================================="
echo "Results saved in: ${OUTPUT_BASE}/mair/"
echo "  -> with_instruction/{model_name}/"
