#!/usr/bin/env python3
"""
Re-evaluate existing LODO adapters on ALL MAIR domains (not just the held-out one).

For each LODO experiment (holdout domain X):
  - Training tasks (all except X): eval using only eval-split queries (no leakage)
  - Held-out tasks (domain X): eval using ALL queries (never seen in training)

Results are saved in ``eval_results_all_domains/`` next to the existing
``eval_results/`` directory.

Multi-GPU parallelism is supported: each domain experiment is dispatched to a
separate process pinned to one GPU.  With 8 GPUs and 11 domains, all 11 runs
finish in roughly 2 sequential waves instead of 11.

Usage:
    # Evaluate all LODO experiments (auto-detect GPUs)
    python scripts/run_lodo_full_evaluation.py --train-ratio 0.2

    # Limit parallelism (e.g. 4 GPUs)
    python scripts/run_lodo_full_evaluation.py --train-ratio 0.2 --num-workers 4

    # Evaluate specific held-out domains only
    python scripts/run_lodo_full_evaluation.py --domains code web finance

    # Resume partial evaluation (skip already-evaluated tasks)
    python scripts/run_lodo_full_evaluation.py --resume

    # Dry run: print what would be evaluated
    python scripts/run_lodo_full_evaluation.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.mair_evaluator import MAIR_TASKS, ALL_MAIR_TASKS

ALL_DOMAINS = list(MAIR_TASKS.keys())


def _num_available_gpus() -> int:
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return len([l for l in result.stdout.strip().splitlines() if l.strip()])
    except Exception:
        pass
    return 1


def _eval_worker(job: dict) -> dict:
    """Worker function executed in a child process pinned to one GPU.

    Each child process starts fresh (spawn), so setting CUDA_VISIBLE_DEVICES
    here — before any CUDA-aware import — reliably pins this process to the
    requested GPU.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(job["gpu_id"])

    # Re-add project root to sys.path (child process may not inherit it)
    project_root = job["project_root"]
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from scripts.evaluate_era_adapter import evaluate_adapter_on_tasks
    from src.evaluation.mair_evaluator import ALL_MAIR_TASKS as _ALL_TASKS

    domain = job["domain"]
    exp_dir = Path(job["exp_dir"])
    print(f"\n[GPU {job['gpu_id']}] Starting holdout={domain}: {exp_dir.name}", flush=True)

    try:
        evaluate_adapter_on_tasks(
            adapter_path=job["adapter_path"],
            large_model_name=job["large_model"],
            small_model_name=job["small_model"],
            task_names=list(_ALL_TASKS),
            output_dir=job["eval_output_dir"],
            resume=job["resume"],
            eval_query_ids_per_task=job["eval_query_ids"],
            **({"cache_dir": job["cache_dir"]} if job["cache_dir"] else {}),
        )
        print(f"\n[GPU {job['gpu_id']}] Completed holdout={domain}", flush=True)
        return {"domain": domain, "success": True}
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"\n[GPU {job['gpu_id']}] FAILED holdout={domain}: {e}\n{tb}", flush=True)
        return {"domain": domain, "success": False, "error": str(e)}


def safe_name(name: str) -> str:
    return name.replace("/", "__").replace(" ", "_")


def extract_domain_from_dir(dir_name: str) -> str | None:
    m = re.search(r"ood_([a-zA-Z]+)_\d+_tasks", dir_name)
    return m.group(1) if m else None


def extract_train_ratio_from_dir(dir_name: str) -> str | None:
    m = re.search(r"__train([0-9.]+)__", dir_name)
    return m.group(1) if m else None


def build_label_tag(negative_strategy: str, num_negatives: int) -> str:
    return f"losscontrastive__neg{negative_strategy}__nneg{num_negatives}"


def discover_lodo_dirs(
    results_base: Path,
    large_model: str,
    small_model: str,
    train_ratios: list[str] | None = None,
    training_mode: str | None = None,
    label_negative_strategy: str = "random",
    label_num_negatives: int = 5,
) -> dict[tuple[str, str], Path]:
    """Auto-discover LODO experiment directories, keyed by (holdout domain, train_ratio)."""
    model_pair = f"{safe_name(large_model)}__to__{safe_name(small_model)}"
    search_dir = results_base / model_pair / "with_instruction" / "linear"

    ratio_filters = {f"__train{r}__" for r in train_ratios} if train_ratios else None
    label_tag = build_label_tag(label_negative_strategy, label_num_negatives)
    lodo_dirs: dict[tuple[str, str], Path] = {}

    if not search_dir.is_dir():
        return lodo_dirs

    for exp_dir in sorted(search_dir.iterdir()):
        if not exp_dir.is_dir() or "ood" not in exp_dir.name:
            continue
        if training_mode and not exp_dir.name.startswith(f"{training_mode}__"):
            continue
        if ratio_filters and not exp_dir.name.startswith("alignment_only__"):
            if not any(rf in exp_dir.name for rf in ratio_filters):
                continue
        if not exp_dir.name.startswith("alignment_only__") and label_tag not in exp_dir.name:
            continue

        domain = extract_domain_from_dir(exp_dir.name)
        ratio = extract_train_ratio_from_dir(exp_dir.name) or "unknown"
        if not domain:
            continue

        key = (domain, ratio)
        # Keep latest run if multiple exist for the same (domain, ratio)
        if key not in lodo_dirs or exp_dir.stat().st_mtime > lodo_dirs[key].stat().st_mtime:
            lodo_dirs[key] = exp_dir

    return lodo_dirs


def find_adapter_path(exp_dir: Path) -> str | None:
    """Find the final unified adapter .pt file in the experiment directory."""
    # The final adapter is at the root level of the experiment dir
    pt_files = [f for f in exp_dir.glob("adapter__*.pt") if "epoch" not in f.name]
    if pt_files:
        return str(pt_files[0])
    return None


def load_eval_query_ids(exp_dir: Path) -> dict[str, list[str]]:
    """Load eval query IDs from era_meta.json."""
    meta_path = exp_dir / "era_meta.json"
    if not meta_path.is_file():
        return {}
    with open(meta_path) as f:
        meta = json.load(f)
    return (meta.get("label_result") or {}).get("eval_query_ids_per_task", {})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-evaluate LODO adapters on ALL MAIR domains.",
    )
    parser.add_argument("--large-model", type=str, default="Qwen/Qwen3-Embedding-8B")
    parser.add_argument("--small-model", type=str, default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--output-dir", type=str, default="results/era",
                        help="Base output directory (same as training)")
    parser.add_argument("--train-ratio", type=str, default=None,
                        help="Filter experiments by a single train ratio (e.g. '0.2'). "
                             "Use --train-ratios for multiple ratios.")
    parser.add_argument("--train-ratios", type=str, nargs="+", default=None,
                        help="Filter experiments by multiple train ratios (e.g. 0.05 0.1 0.2 0.4).")
    parser.add_argument("--training-mode", type=str, default="both",
                        choices=["alignment_only", "label_only", "both"],
                        help="Filter experiments by unified training mode")
    parser.add_argument("--label-negative-strategy", type=str, default="topk_percpos",
                        choices=["random", "naive_topk", "topk_percpos"],
                        help="Filter label-phase experiments by negative sampling strategy")
    parser.add_argument("--label-num-negatives", type=int, default=5,
                        help="Filter label-phase experiments by number of negatives")
    parser.add_argument("--label-hard-negative-top-k", type=int, default=2000,
                        help="Hard-negative retrieval depth (default: 2000)")
    parser.add_argument("--label-hard-negative-perc-margin", type=float, default=0.95,
                        help="topk_percpos margin (default: 0.95)")
    parser.add_argument("--domains", type=str, nargs="+", default=None,
                        help="Only evaluate these held-out domains (default: all found)")
    parser.add_argument("--resume", action="store_true", default=False,
                        help="Skip already-evaluated tasks within each experiment")
    parser.add_argument("--dry-run", action="store_true", default=False,
                        help="Print what would be evaluated without running")
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument(
        "--num-workers", type=int, default=None,
        help="Number of parallel workers (= GPUs to use). "
             "Defaults to the number of available GPUs, or 1 if no GPU is found.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results_base = Path(args.output_dir)

    # Merge --train-ratio and --train-ratios into a single list
    train_ratios: list[str] | None = None
    if args.train_ratios:
        train_ratios = args.train_ratios
    elif args.train_ratio:
        train_ratios = [args.train_ratio]

    # Discover experiments
    lodo_dirs = discover_lodo_dirs(
        results_base, args.large_model, args.small_model,
        train_ratios,
        training_mode=args.training_mode,
        label_negative_strategy=args.label_negative_strategy,
        label_num_negatives=args.label_num_negatives,
    )

    if args.domains:
        lodo_dirs = {key: p for key, p in lodo_dirs.items() if key[0] in args.domains}

    if not lodo_dirs:
        print("No LODO experiments found. Run training first.")
        return 1

    # Determine parallelism
    num_gpus = _num_available_gpus()
    num_workers = args.num_workers if args.num_workers is not None else num_gpus
    num_workers = max(1, min(num_workers, len(lodo_dirs)))

    print("=" * 70)
    print("  LODO Full-Domain Evaluation")
    print("=" * 70)
    print(f"  Experiments found: {len(lodo_dirs)}")
    for (domain, ratio), exp_dir in sorted(lodo_dirs.items()):
        print(f"    {domain:>12s} (train={ratio}): {exp_dir.name}")
    print(f"  Total MAIR tasks per experiment: {len(ALL_MAIR_TASKS)}")
    if train_ratios:
        print(f"  Train ratio filter: {', '.join(train_ratios)}")
    if args.training_mode:
        print(f"  Training mode filter: {args.training_mode}")
    print(
        "  Label filter: "
        f"{build_label_tag(args.label_negative_strategy, args.label_num_negatives)}"
    )
    if args.label_negative_strategy in ("naive_topk", "topk_percpos"):
        print(f"  Hard-neg top-k:  {args.label_hard_negative_top_k}")
    if args.label_negative_strategy == "topk_percpos":
        print(f"  Hard-neg margin: {args.label_hard_negative_perc_margin}")
    print(f"  GPUs available:  {num_gpus}")
    print(f"  Parallel workers: {num_workers}")
    print(f"  Resume: {args.resume}")
    print(f"  Dry run: {args.dry_run}")
    print("=" * 70)

    # Build job list (assign GPUs round-robin)
    project_root = str(Path(__file__).parent.parent)
    jobs = []
    for i, ((domain, ratio), exp_dir) in enumerate(sorted(lodo_dirs.items())):
        adapter_path = find_adapter_path(exp_dir)
        if not adapter_path:
            print(f"  WARNING: No adapter .pt file found in {exp_dir} — skipping {domain} (train={ratio})")
            continue

        eval_query_ids = load_eval_query_ids(exp_dir)
        holdout_tasks = MAIR_TASKS.get(domain, [])

        print(f"  {domain:>12s} (train={ratio}): adapter={Path(adapter_path).name[:60]}...")
        print(f"    train-filter tasks={len(eval_query_ids)}, held-out tasks={len(holdout_tasks)}")

        if args.dry_run:
            print(f"    [DRY RUN] Would evaluate {len(ALL_MAIR_TASKS)} tasks on GPU {i % num_workers}")
            continue

        jobs.append({
            "domain": domain,
            "train_ratio": ratio,
            "exp_dir": str(exp_dir),
            "adapter_path": adapter_path,
            "eval_query_ids": eval_query_ids,
            "eval_output_dir": str(exp_dir / "eval_results_all_domains"),
            "large_model": args.large_model,
            "small_model": args.small_model,
            "resume": args.resume,
            "cache_dir": args.cache_dir,
            "gpu_id": i % num_workers,       # round-robin GPU assignment
            "project_root": project_root,
        })

    if args.dry_run:
        return 0

    if not jobs:
        print("No valid jobs to run.")
        return 1

    # Run in parallel using spawn to safely set CUDA_VISIBLE_DEVICES per worker
    print(f"\nLaunching {len(jobs)} experiments across {num_workers} workers...")
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(processes=num_workers) as pool:
        outcomes = pool.map(_eval_worker, jobs)

    succeeded = sum(1 for r in outcomes if r.get("success"))
    failed    = sum(1 for r in outcomes if not r.get("success"))
    failed_domains = [r["domain"] for r in outcomes if not r.get("success")]

    print(f"\n{'=' * 70}")
    print("  Full-Domain Evaluation Summary")
    print(f"{'=' * 70}")
    print(f"  Succeeded: {succeeded}/{len(jobs)}")
    print(f"  Failed:    {failed}/{len(jobs)}")
    if failed_domains:
        print(f"  Failed domains: {failed_domains}")
    print(f"{'=' * 70}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
