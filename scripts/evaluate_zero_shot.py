import argparse
import os
import sys
from pathlib import Path

# Add project root (paper_release/) to path so that src/ is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.wrappers import OpenAIEmbedder, LocalHFEmbedder
from src.patch_transformers import patch_dynamic_cache
from src.cache_config import EMBEDDING_CACHE_DIR


def _resolve_benchmark_cache_dir(cache_dir: str, benchmark_name: str) -> str:
    """Resolve cache dir for a benchmark without duplicating benchmark suffixes."""
    normalized = os.path.normpath(cache_dir)
    if os.path.basename(normalized) == benchmark_name:
        return normalized
    return os.path.join(normalized, benchmark_name)

def main():
    # Apply patches
    patch_dynamic_cache()

    parser = argparse.ArgumentParser(description="Retrieval Benchmark Framework")
    
    parser.add_argument("--model_name", type=str, required=True, 
                        help="Model name (e.g., text-embedding-3-large, nvidia/NV-Embed-v2, bm25)")
    parser.add_argument("--model_type", type=str, choices=["openai", "hf"], required=True,
                        help="Type of the model provider")
    parser.add_argument("--tasks", type=str, nargs="+", default=["NFCorpus"],
                        help="List of MTEB tasks to evaluate")
    parser.add_argument("--batch_size", type=int, default=32, help="Inference batch size")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save results")
    
    # Benchmark type selection
    parser.add_argument("--mair_category", type=str, default=None,
                        help="MAIR category to evaluate (e.g., 'math', 'code', 'finance'). "
                             "Use --list_mair to see all categories.")
    parser.add_argument("--list_mair", action="store_true",
                        help="List all available MAIR tasks and categories")
    
    # Cache options
    parser.add_argument("--cache_dir", type=str, default=EMBEDDING_CACHE_DIR, 
                        help="Directory to cache embeddings")
    parser.add_argument("--no_cache", action="store_true",
                        help="Disable embedding cache")
    parser.add_argument("--force_recache", action="store_true",
                        help="Force recompute embeddings and overwrite cache (default: use cache if available)")
    
    args = parser.parse_args()
    
    # Handle --list_mair
    if args.list_mair:
        from src.evaluation.mair_evaluator import list_mair_categories
        categories = list_mair_categories()
        print("\n=== Available MAIR Tasks by Category ===\n")
        for category, tasks in categories.items():
            print(f"{category.upper()} ({len(tasks)} tasks):")
            for task in tasks:
                print(f"  - {task}")
            print()
        return

    # 1. Initialize Model
    print(f"Initializing model: {args.model_name} ({args.model_type})")
    
    try:
        if args.model_type == "openai":
            model = OpenAIEmbedder(model_name=args.model_name)
        elif args.model_type == "hf":
            model = LocalHFEmbedder(model_name_or_path=args.model_name, use_fp16=True)
        else:
            raise ValueError("Unknown model type")
    except Exception as e:
        print(f"Error initializing model: {e}")
        sys.exit(1)

    # 2. Setup Evaluation
    os.makedirs(args.output_dir, exist_ok=True)
    cache_enabled = not args.no_cache
    force_recache = args.force_recache
    
    if args.no_cache and args.force_recache:
        print("Warning: --no_cache and --force_recache both specified. Using --no_cache (cache disabled).")
        force_recache = False
    
    if cache_enabled:
        if force_recache:
            print(f"Embedding cache: FORCE RECACHE mode - will recompute and overwrite existing cache")
        else:
            print(f"Embedding cache enabled: {args.cache_dir}")
    else:
        print("Embedding cache disabled")
    
    # 3. Run MAIR benchmark
    from src.evaluation.mair_evaluator import MAIREvaluator, get_mair_tasks_by_category, ALL_MAIR_TASKS

    # Add benchmark type to paths
    mair_cache_dir = _resolve_benchmark_cache_dir(args.cache_dir, "mair")
    mair_output_dir = os.path.join(args.output_dir, "mair")

    # Determine which tasks to run
    if args.mair_category:
        tasks = get_mair_tasks_by_category(args.mair_category)
        print(f"Running MAIR category '{args.mair_category}': {len(tasks)} tasks")
    else:
        tasks = args.tasks
        # Validate tasks
        for task in tasks:
            if task not in ALL_MAIR_TASKS:
                print(f"Warning: '{task}' is not a valid MAIR task. Use --list_mair to see available tasks.")

    # Set experiment name based on instruction usage
    experiment_name = "with_instruction"

    evaluator = MAIREvaluator(
        model,
        experiment_name=experiment_name,
        cache_enabled=cache_enabled,
        cache_dir=mair_cache_dir,
        use_instructions=True,
        force_recache=force_recache
    )

    print(f"Starting MAIR evaluation on tasks: {tasks}")
    print(f"Instructions: enabled")
    print(f"Results will be saved to: {mair_output_dir}/{experiment_name}/")
    try:
        evaluator.run(tasks=tasks, output_folder=mair_output_dir, batch_size=args.batch_size)
        print(f"MAIR Evaluation finished. Results saved to {mair_output_dir}")
    except Exception as e:
        print(f"Error during MAIR evaluation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
