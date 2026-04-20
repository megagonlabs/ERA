#!/usr/bin/env python3
"""
Batch evaluation of a trained adapter on all (or selected) MAIR tasks.

Evaluates a single adapter weight file across multiple MAIR tasks and
saves per-task metrics + an aggregated summary.

Usage Examples:
    # Evaluate on all MAIR tasks
    python scripts/run_adapter_batch_evaluation.py \
        --adapter-path results/era/.../adapter__....pt

    # Evaluate on specific tasks
    python scripts/run_adapter_batch_evaluation.py \
        --adapter-path results/.../adapter.pt \
        --tasks NFCorpus SciFact ArguAna

    # Evaluate on tasks from specific domains
    python scripts/run_adapter_batch_evaluation.py \
        --adapter-path results/.../adapter.pt \
        --domains web finance

    # Evaluate without instructions
    python scripts/run_adapter_batch_evaluation.py \
        --adapter-path results/.../adapter.pt

    # Resume from a partial run (skip already-evaluated tasks)
    python scripts/run_adapter_batch_evaluation.py \
        --adapter-path results/.../adapter.pt \
        --resume
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.adapter.adapted_embedder import AdaptedQueryEmbedder, NoAdapterEmbedder
from src.cache_config import EMBEDDING_CACHE_DIR
from src.evaluation.mair_evaluator import (
    MAIREvaluator,
    MAIRDataset,
    MAIR_TASKS,
    ALL_MAIR_TASKS,
)

AVAILABLE_DOMAINS = list(MAIR_TASKS.keys())


def _load_eval_query_ids_from_metadata(adapter_path: str | None) -> dict[str, list[str]] | None:
    """Load eval query IDs from nearby unified training metadata when available."""
    if not adapter_path:
        return None

    adapter_file = Path(adapter_path).resolve()
    candidate_dirs = [adapter_file.parent, adapter_file.parent.parent, adapter_file.parent.parent.parent]
    for directory in candidate_dirs:
        meta_path = directory / "era_meta.json"
        if not meta_path.exists():
            continue
        try:
            with meta_path.open(encoding="utf-8") as f:
                meta = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"WARNING: Failed to read {meta_path}: {exc}")
            continue

        label_result = meta.get("label_result") or {}
        eval_ids = label_result.get("eval_query_ids_per_task")
        if isinstance(eval_ids, dict) and eval_ids:
            print(f"Loaded eval query IDs for {len(eval_ids)} task(s) from: {meta_path}")
            return eval_ids

    return None


def get_tasks_for_domains(domains: list) -> list:
    """Get all task names for the specified domains."""
    tasks = []
    for domain in domains:
        if domain not in MAIR_TASKS:
            raise ValueError(f"Unknown domain: {domain}. Available: {AVAILABLE_DOMAINS}")
        tasks.extend(MAIR_TASKS[domain])
    return tasks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch evaluate a trained adapter on MAIR tasks.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available Domains: {', '.join(AVAILABLE_DOMAINS)}
Total MAIR Tasks: {len(ALL_MAIR_TASKS)}

Examples:
  # All tasks (default)
  python run_adapter_batch_evaluation.py --adapter-path results/.../adapter.pt

  # Specific domain
  python run_adapter_batch_evaluation.py --adapter-path results/.../adapter.pt --domains web

  # Resume interrupted run
  python run_adapter_batch_evaluation.py --adapter-path results/.../adapter.pt --resume
        """,
    )
    # Model config
    parser.add_argument("--large-model", type=str, default="Qwen/Qwen3-Embedding-8B")
    parser.add_argument("--small-model", type=str, default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--adapter-path", type=str, default=None,
                        help="Path to the trained adapter .pt file. "
                             "Omit to run without adapter (no_adapter baseline).")

    # Task selection (mutually exclusive convenience)
    parser.add_argument("--tasks", type=str, nargs="+", default=None,
                        help="Specific MAIR tasks to evaluate. Default: all tasks")
    parser.add_argument("--domains", type=str, nargs="+", default=None,
                        help="Evaluate tasks from these domains")

    # Paths
    parser.add_argument("--cache-dir", type=str, default=EMBEDDING_CACHE_DIR)
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for results. Default: next to adapter file")

    # Resume support
    parser.add_argument("--resume", action="store_true", default=False,
                        help="Skip tasks that already have results in output-dir")

    return parser.parse_args()


def evaluate_adapter_on_tasks(
    adapter_path: str | None,
    large_model_name: str,
    small_model_name: str,
    task_names: list,
    cache_dir: str = EMBEDDING_CACHE_DIR,
    output_dir: str | None = None,
    resume: bool = False,
    eval_query_ids_per_task: dict | None = None,
) -> dict:
    """
    Evaluate a trained adapter (or no adapter baseline) on multiple MAIR tasks.

    Args:
        adapter_path: Path to adapter .pt file, or None to run without adapter
            (no_adapter baseline). When None, large_model_name must equal
            small_model_name, otherwise a ValueError is raised.
        large_model_name: Large model name.
        small_model_name: Small model name.
        task_names: List of MAIR task names to evaluate.
        cache_dir: Embedding cache directory.
        output_dir: Where to save results. If None, creates a directory next to
            adapter_path (adapter mode) or under ``results/no_adapter/`` (no-adapter
            mode).
        resume: If True, skip tasks with existing results.
        eval_query_ids_per_task: Optional dict of {task_name: [eval_qid, ...]}. When
            provided, only the specified query IDs are used for evaluation (training
            queries excluded).

    Returns:
        Dictionary of {task_name: {metric: score}}.
    """
    start_time = time.time()

    no_adapter_mode = adapter_path is None

    # Validate no-adapter mode
    if no_adapter_mode and large_model_name != small_model_name:
        raise ValueError(
            f"No adapter specified (no_adapter mode): large_model ({large_model_name}) and "
            f"small_model ({small_model_name}) must be the same. "
            f"Without an adapter there is no transformation between the two spaces."
        )

    # Determine output directory
    if output_dir is None:
        if no_adapter_mode:
            safe_name = small_model_name.replace("/", "_").replace("\\", "_")
            output_dir = str(Path("results") / "no_adapter" / safe_name / "with_instruction")
        else:
            adapter_dir = Path(adapter_path).parent
            output_dir = str(adapter_dir / "eval_results" / "with_instruction")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load model
    description = "with_instruction"
    if no_adapter_mode:
        model = NoAdapterEmbedder(
            model_name=small_model_name,
            cache_dir=cache_dir,
            description=description,
        )
        print(f"[No-Adapter Baseline] Using {small_model_name} for both queries and corpus.")
    else:
        model = AdaptedQueryEmbedder(
            large_model_name=large_model_name,
            small_model_name=small_model_name,
            adapter_path=adapter_path,
            cache_dir=cache_dir,
            description=description,
        )

    # Determine which tasks to skip (resume mode)
    skip_tasks = set()
    if resume:
        summary_path = output_path / "summary.json"
        if summary_path.exists():
            with open(summary_path) as f:
                existing = json.load(f)
            for task in existing.get("per_task", {}):
                if "error" not in existing["per_task"][task]:
                    skip_tasks.add(task)
            print(f"[Resume] Skipping {len(skip_tasks)} already-evaluated tasks")

    # Evaluate task by task
    all_results = {}
    total = len(task_names)
    succeeded = 0
    failed = 0

    for i, task_name in enumerate(task_names, 1):
        print(f"\n{'=' * 60}")
        print(f"[{i}/{total}] Evaluating: {task_name}")
        print(f"{'=' * 60}")

        if task_name in skip_tasks:
            print(f"  [Skip] Already evaluated (--resume)")
            continue

        try:
            # Load dataset
            dataset = MAIRDataset(task_name, cache_dir=None)
            dataset.load()

            full_query_items = list(dataset.queries.items())
            query_items = full_query_items
            eval_query_ids = None

            # Filter to eval-only queries for metrics, but preload caches using the full task.
            if eval_query_ids_per_task and task_name in eval_query_ids_per_task:
                eval_query_ids = list(eval_query_ids_per_task[task_name])
                allowed = set(eval_query_ids)
                query_items = [(qid, text) for qid, text in query_items if qid in allowed]
                print(f"  [Eval filter] Using {len(query_items)}/{len(dataset.queries)} eval queries")

            all_query_texts = [text for _, text in full_query_items]
            all_corpus_texts = list(dataset.corpus.values())

            # Determine instruction
            instruction = None
            per_query_instructions = None

            if dataset.instructions:
                unique_instructions = set(dataset.instructions.values())
                if len(unique_instructions) == 1:
                    instruction = list(unique_instructions)[0]
                elif len(unique_instructions) > 1:
                    per_query_instructions = [
                        dataset.instructions.get(qid, "") for qid, _ in full_query_items
                    ]

            # Pre-load caches
            model.load_full_cache(
                task_name, all_query_texts, all_corpus_texts,
                instruction, per_query_instructions,
            )

            # Create evaluator per task (reuses the same model)
            evaluator = MAIREvaluator(
                model=model,
                use_instructions=True,
                cache_enabled=True,
                cache_dir=cache_dir,
                force_recache=False,
                use_subdirs=False,
            )

            results = evaluator.run(
                tasks=[task_name],
                output_folder=str(output_path / "tasks"),
                batch_size=64,
                eval_query_ids_per_task={task_name: eval_query_ids} if eval_query_ids is not None else None,
            )

            task_results = results.get(task_name, {})
            all_results[task_name] = task_results
            succeeded += 1

            print(f"  Results for {task_name}:")
            for metric, value in task_results.items():
                if isinstance(value, (int, float)):
                    print(f"    {metric}: {value:.4f}")

        except Exception as e:
            import traceback
            traceback.print_exc()
            all_results[task_name] = {"error": str(e)}
            failed += 1

        # Save incremental summary after each task
        _save_summary(all_results, output_path, adapter_path, time.time() - start_time)

    elapsed = time.time() - start_time
    _save_summary(all_results, output_path, adapter_path, elapsed)

    print(f"\n{'=' * 60}")
    print("Batch Evaluation Summary")
    print(f"{'=' * 60}")
    print(f"  Tasks evaluated: {succeeded}/{total}")
    print(f"  Tasks failed: {failed}")
    print(f"  Total time: {elapsed:.1f}s")
    print(f"  Results saved to: {output_path}")

    # Print aggregated metrics
    _print_aggregated_metrics(all_results)

    return all_results


def _save_summary(
    all_results: dict,
    output_path: Path,
    adapter_path: str,
    elapsed: float,
):
    """Save summary JSON with per-task and aggregated metrics."""
    # Compute aggregated metrics
    metric_sums = {}
    metric_counts = {}
    for task, results in all_results.items():
        if "error" in results:
            continue
        for metric, value in results.items():
            if isinstance(value, (int, float)):
                metric_sums[metric] = metric_sums.get(metric, 0.0) + value
                metric_counts[metric] = metric_counts.get(metric, 0) + 1

    aggregated = {}
    for metric in sorted(metric_sums.keys()):
        aggregated[metric] = round(metric_sums[metric] / metric_counts[metric], 6)

    # Domain-level aggregation
    domain_results = {}
    for domain, tasks in MAIR_TASKS.items():
        domain_metrics = {}
        domain_count = 0
        for task in tasks:
            if task in all_results and "error" not in all_results[task]:
                domain_count += 1
                for metric, value in all_results[task].items():
                    if isinstance(value, (int, float)):
                        domain_metrics[metric] = domain_metrics.get(metric, 0.0) + value
        if domain_count > 0:
            domain_results[domain] = {
                "num_tasks": domain_count,
                "metrics": {
                    m: round(v / domain_count, 6) for m, v in domain_metrics.items()
                },
            }

    summary = {
        "adapter_path": adapter_path,
        "use_instruction": True,
        "num_tasks_evaluated": sum(1 for r in all_results.values() if "error" not in r),
        "num_tasks_failed": sum(1 for r in all_results.values() if "error" in r),
        "elapsed_sec": round(elapsed, 2),
        "aggregated": aggregated,
        "per_domain": domain_results,
        "per_task": all_results,
    }

    summary_path = output_path / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def _print_aggregated_metrics(all_results: dict):
    """Print aggregated metrics across tasks."""
    metric_sums = {}
    metric_counts = {}
    for task, results in all_results.items():
        if "error" in results:
            continue
        for metric, value in results.items():
            if isinstance(value, (int, float)):
                metric_sums[metric] = metric_sums.get(metric, 0.0) + value
                metric_counts[metric] = metric_counts.get(metric, 0) + 1

    if not metric_sums:
        print("  No metrics to aggregate.")
        return

    n = next(iter(metric_counts.values()))
    print(f"\n  Aggregated over {n} tasks:")
    for metric in sorted(metric_sums.keys()):
        avg = metric_sums[metric] / metric_counts[metric]
        print(f"    {metric}: {avg:.4f}")

    # Per-domain summary for key metric
    key_metric = "ndcg@10" if "ndcg@10" in metric_sums else next(iter(metric_sums.keys()))
    print(f"\n  Per-domain {key_metric}:")
    for domain, tasks in MAIR_TASKS.items():
        values = [
            all_results[t][key_metric]
            for t in tasks
            if t in all_results and "error" not in all_results[t] and key_metric in all_results[t]
        ]
        if values:
            avg = sum(values) / len(values)
            print(f"    {domain:12s}: {avg:.4f} ({len(values)} tasks)")


def main() -> int:
    args = parse_args()

    # Determine tasks
    if args.tasks:
        task_names = args.tasks
    elif args.domains:
        task_names = get_tasks_for_domains(args.domains)
    else:
        task_names = list(ALL_MAIR_TASKS)

    print(f"Adapter: {args.adapter_path if args.adapter_path else '[none — no_adapter baseline]'}")
    print(f"Tasks: {len(task_names)}")
    print(f"Instructions: enabled")

    eval_query_ids_per_task = _load_eval_query_ids_from_metadata(args.adapter_path)
    if eval_query_ids_per_task is not None:
        filtered_task_count = sum(1 for task in task_names if task in eval_query_ids_per_task)
        print(f"Eval query filters available for {filtered_task_count}/{len(task_names)} selected task(s)")

    # Validate no-adapter mode before running
    if args.adapter_path is None and args.large_model != args.small_model:
        print(f"ERROR: No adapter specified but large_model ({args.large_model}) != "
              f"small_model ({args.small_model}).")
        print("Without an adapter, both models must be the same. "
              "Use --small-model to set the same model for both, or provide --adapter-path.")
        return 1

    evaluate_adapter_on_tasks(
        adapter_path=args.adapter_path,
        large_model_name=args.large_model,
        small_model_name=args.small_model,
        task_names=task_names,
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        resume=args.resume,
        eval_query_ids_per_task=eval_query_ids_per_task,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
