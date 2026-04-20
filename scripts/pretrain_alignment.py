#!/usr/bin/env python3
"""
Run identical text alignment for query-side adapter initialization.
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.adapter.identical_text_alignment import AlignmentConfig, train_identical_text_alignment
from src.cache_config import EMBEDDING_CACHE_DIR, DATASET_CACHE_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a linear adapter via identical text alignment.")
    parser.add_argument("--large-model", type=str, default="Qwen/Qwen3-Embedding-8B")
    parser.add_argument("--small-model", type=str, default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--tasks", type=str, nargs="+", default=["NFCorpus"], help="List of task names to sample from")
    parser.add_argument("--samples-per-task", type=int, default=1000, help="Number of samples per task")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--checkpoint-interval", type=int, default=50, help="Save checkpoint every N epochs")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--eval-ratio", type=float, default=0.1)
    parser.add_argument("--loss-type", type=str, default="cosine", choices=["cosine", "mse", "smooth_l1"], help="Loss function type")
    parser.add_argument("--cache-dir", type=str, default=EMBEDDING_CACHE_DIR)
    parser.add_argument("--dataset-cache-dir", type=str, default=None, help="Path to dataset cache (default: from .env or cache/datasets/mair)")
    parser.add_argument("--output-dir", type=str, default="results/adapter_alignment")
    parser.add_argument("--max-length", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    config = AlignmentConfig(
        large_model_name=args.large_model,
        small_model_name=args.small_model,
        task_names=args.tasks,
        samples_per_task=args.samples_per_task,
        seed=args.seed,
        batch_size=args.batch_size,
        epochs=args.epochs,
        checkpoint_interval=args.checkpoint_interval,
        lr=args.lr,
        weight_decay=args.weight_decay,
        eval_ratio=args.eval_ratio,
        loss_type=args.loss_type,
        cache_dir=args.cache_dir,
        dataset_cache_dir=args.dataset_cache_dir,
        output_dir=args.output_dir,
        max_length=args.max_length,
    )

    result = train_identical_text_alignment(config)
    print("Training completed.")
    print(f"Adapter saved to: {result['adapter_path']}")
    print(f"Metadata saved to: {result['meta_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
