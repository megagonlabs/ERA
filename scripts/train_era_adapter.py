#!/usr/bin/env python3
"""
Train the ERA (Embedding-Retrieval Adapter) pipeline.

This script combines identical text alignment (self-supervised) with
label-based training on query-document pairs.

Usage Examples:
    # Default: both alignment and label training (in-domain)
    python scripts/train_era_adapter.py --tasks NFCorpus SciFact
    
    # Alignment only (self-supervised)
    python scripts/train_era_adapter.py --tasks NFCorpus --training-mode alignment_only
    
    # Label training only (supervised)
    python scripts/train_era_adapter.py --tasks NFCorpus --training-mode label_only
    
    # Label training only, initialised from pre-trained alignment weights
    python scripts/train_era_adapter.py --training-mode label_only \
        --pretrained-adapter-path results/.../alignment/adapter_*.pt
    
    # Custom train/eval split ratio (default 50:50)
    python scripts/train_era_adapter.py --tasks NFCorpus --label-train-ratio 0.8
    
    # Out-of-domain evaluation: train on all domains except 'web', eval on 'web'
    python scripts/train_era_adapter.py --eval-mode out_of_domain --holdout-domain web
    
    # Train on specific domains, hold out another for evaluation
    python scripts/train_era_adapter.py --train-domains math code --holdout-domain finance --eval-mode out_of_domain
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.adapter.era_training import ERAConfig, train_era
from src.cache_config import EMBEDDING_CACHE_DIR, DATASET_CACHE_DIR
from src.evaluation.mair_evaluator import MAIR_TASKS, ALL_MAIR_TASKS


# Available MAIR domains
AVAILABLE_DOMAINS = list(MAIR_TASKS.keys())


def get_tasks_for_domains(domains: list) -> list:
    """Get all task names for the specified domains."""
    tasks = []
    for domain in domains:
        if domain not in MAIR_TASKS:
            raise ValueError(f"Unknown domain: {domain}. Available: {AVAILABLE_DOMAINS}")
        tasks.extend(MAIR_TASKS[domain])
    return tasks


def get_tasks_excluding_domains(exclude_domains: list) -> list:
    """Get all task names excluding specified domains."""
    tasks = []
    for domain, domain_tasks in MAIR_TASKS.items():
        if domain not in exclude_domains:
            tasks.extend(domain_tasks)
    return tasks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified adapter training: alignment + label training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Training Modes:
  both           First alignment pretraining, then label finetuning (default)
  alignment_only Only self-supervised identical text alignment
  label_only     Only supervised label training on query-doc pairs

Evaluation Modes:
  in_domain      Split queries within each task. Eval set is fixed by --label-eval-ratio (default 0.5).
                 Training set size is controlled by --label-train-ratio (0.1~0.5).
  out_of_domain  Train on all tasks except holdout domain, eval on holdout domain
  only_domain    Train and eval on a single domain only (with query splitting)

Negative Strategies:
    random         Sample negatives uniformly from non-relevant documents
    naive_topk     Retrieve hard negatives from the teacher similarity top-k
    topk_percpos   Filter top-k negatives by a percentage of the positive score

Available Domains: {', '.join(AVAILABLE_DOMAINS)}

Examples:
  # In-domain: Run both phases on NFCorpus with 50:50 query split
  python scripts/train_era_adapter.py --tasks NFCorpus

  # In-domain: vary training data size while keeping eval fixed at 50%%
  python scripts/train_era_adapter.py --tasks NFCorpus --label-train-ratio 0.3 --label-eval-ratio 0.5

  # Sweep over training sizes
  for ratio in 0.1 0.2 0.3 0.4 0.5; do
    python scripts/train_era_adapter.py --label-train-ratio $ratio --label-eval-ratio 0.5
  done

  # Out-of-domain: Train on all domains except 'web', eval on 'web'
  python scripts/train_era_adapter.py --eval-mode out_of_domain --holdout-domain web

  # Out-of-domain: Train on specific domains, hold out 'finance'
  python scripts/train_era_adapter.py --train-domains math code knowledge --holdout-domain finance --eval-mode out_of_domain

  # Only-domain: Train and evaluate on 'web' domain only
  python scripts/train_era_adapter.py --eval-mode only_domain --holdout-domain web
        """
    )
    
    # Model configuration
    parser.add_argument("--large-model", type=str, default="Qwen/Qwen3-Embedding-8B",
                        help="Large model for query encoding")
    parser.add_argument("--small-model", type=str, default="Qwen/Qwen3-Embedding-0.6B",
                        help="Small model for document encoding")
    
    # Task configuration
    parser.add_argument("--tasks", type=str, nargs="+", default=None,
                        help="List of MAIR task names (overrides --train-domains). Default: all MAIR tasks")
    parser.add_argument("--train-domains", type=str, nargs="+", default=None,
                        help=f"Domains to use for training. Available: {', '.join(AVAILABLE_DOMAINS)}")
    
    # Evaluation mode
    eval_group = parser.add_argument_group("Evaluation Mode")
    eval_group.add_argument("--eval-mode", type=str, default="in_domain",
                            choices=["in_domain", "out_of_domain", "only_domain"],
                            help="Evaluation mode: 'in_domain' splits queries, 'out_of_domain' uses holdout domain, 'only_domain' trains+evals on one domain")
    eval_group.add_argument("--holdout-domain", type=str, default=None,
                            help=f"Domain for out-of-domain or only-domain evaluation. Available: {', '.join(AVAILABLE_DOMAINS)}")
    eval_group.add_argument("--eval-tasks", type=str, nargs="+", default=None,
                            help="Specific tasks for evaluation (overrides --holdout-domain)")
    
    # Training mode
    parser.add_argument("--training-mode", type=str, default="both",
                        choices=["both", "alignment_only", "label_only"],
                        help="Training mode: 'both', 'alignment_only', or 'label_only'")
    
    # === Alignment configuration ===
    align_group = parser.add_argument_group("Alignment Training (Phase 1)")
    align_group.add_argument("--alignment-samples-per-task", type=int, default=1000,
                             help="Number of corpus samples per task for alignment")
    align_group.add_argument("--alignment-epochs", type=int, default=100,
                             help="Number of alignment training epochs")
    align_group.add_argument("--alignment-lr", type=float, default=1e-3,
                             help="Learning rate for alignment")
    align_group.add_argument("--alignment-weight-decay", type=float, default=0.01,
                             help="Weight decay for alignment")
    align_group.add_argument("--alignment-eval-ratio", type=float, default=0.1,
                             help="Ratio of samples for alignment evaluation")
    align_group.add_argument("--alignment-loss-type", type=str, default="cosine",
                             choices=["cosine", "mse", "smooth_l1"],
                             help="Loss function for alignment")
    align_group.add_argument("--alignment-early-stopping", dest="alignment_early_stopping",
                             action="store_true", default=False,
                             help="Enable early stopping for alignment training (default: off)")
    align_group.add_argument("--no-alignment-early-stopping", dest="alignment_early_stopping",
                             action="store_false",
                             help="Disable early stopping for alignment training")
    align_group.add_argument("--alignment-early-stopping-patience", type=int, default=20,
                             help="Patience for alignment training early stopping")
    
    # === Label training configuration ===
    label_group = parser.add_argument_group("Label Training (Phase 2)")
    label_group.add_argument("--label-train-ratio", type=float, default=0.1,
                             help="Ratio of total queries for training (must be <= 1 - val-ratio - eval-ratio)")
    label_group.add_argument("--label-val-ratio", type=float, default=0.1,
                             help="Ratio of total queries reserved for validation (fixed, default 0.1)")
    label_group.add_argument("--label-eval-ratio", type=float, default=0.5,
                             help="Ratio of total queries reserved for evaluation/test (fixed eval set)")
    label_group.add_argument("--label-num-negatives", type=int, default=5,
                             help="Number of negative samples per positive")
    label_group.add_argument("--label-negative-strategy", type=str, default="random",
                             choices=["random", "naive_topk", "topk_percpos"],
                             help="Negative sampling strategy")
    label_group.add_argument("--label-hard-negative-top-k", type=int, default=2000,
                             help="Initial top-k retrieval size for hard-negative mining")
    label_group.add_argument("--label-hard-negative-perc-margin", type=float, default=0.95,
                             help="Maximum negative score ratio relative to the positive anchor for topk-percpos")
    label_group.add_argument("--label-epochs", type=int, default=10,
                             help="Number of label training epochs")
    label_group.add_argument("--label-lr", type=float, default=1e-3,
                             help="Learning rate for label training")
    label_group.add_argument("--label-weight-decay", type=float, default=0.01,
                             help="Weight decay for label training")
    label_group.add_argument("--label-warmup-ratio", type=float, default=0.1,
                             help="Warmup ratio for label training")
    label_group.add_argument("--label-temperature", type=float, default=0.05,
                             help="Temperature for contrastive loss")
    label_group.add_argument("--label-early-stopping", dest="label_early_stopping",
                             action="store_true", default=True,
                             help="Enable early stopping for label training (default: on)")
    label_group.add_argument("--no-label-early-stopping", dest="label_early_stopping",
                             action="store_false",
                             help="Disable early stopping for label training")
    label_group.add_argument("--label-early-stopping-patience", type=int, default=5,
                             help="Patience for label training early stopping")
    
    # === Shared configuration ===
    shared_group = parser.add_argument_group("Shared Configuration")
    shared_group.add_argument("--seed", type=int, default=42,
                              help="Random seed")
    shared_group.add_argument("--batch-size", type=int, default=256,
                              help="Batch size for training")
    shared_group.add_argument("--checkpoint-interval", type=int, default=50,
                              help="Save checkpoint every N epochs")
    
    # Paths
    path_group = parser.add_argument_group("Paths")
    path_group.add_argument("--cache-dir", type=str, default=EMBEDDING_CACHE_DIR,
                            help="Directory for embedding cache")
    path_group.add_argument("--dataset-cache-dir", type=str, default=None,
                            help="Directory for dataset cache (default: from .env or cache/datasets/mair)")
    path_group.add_argument("--output-dir", type=str, default="results/era",
                            help="Directory for output")
    path_group.add_argument("--pretrained-adapter-path", type=str, default=None,
                            help="Path to pretrained adapter weights (e.g. alignment weights from a prior run). "
                                 "Used as initialization for label training in label_only mode.")
    
    # Post-training evaluation
    eval_post_group = parser.add_argument_group("Post-Training Evaluation")
    eval_post_group.add_argument("--eval-after-training", action="store_true", default=False,
                                 help="Run retrieval evaluation on all tasks after training. Adapter inference uses all visible GPUs when available.")
    eval_post_group.add_argument("--eval-tasks-override", type=str, nargs="+", default=None,
                                 help="Custom list of tasks for post-training eval (default: all MAIR tasks)")
    
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    
    # Determine training tasks
    if args.tasks:
        # Explicit task list provided
        train_tasks = args.tasks
    elif args.train_domains:
        # Get tasks from specified domains
        train_tasks = get_tasks_for_domains(args.train_domains)
    elif args.eval_mode == "out_of_domain" and args.holdout_domain:
        # Exclude holdout domain
        train_tasks = get_tasks_excluding_domains([args.holdout_domain])
    elif args.eval_mode == "only_domain" and args.holdout_domain:
        # Train only on the specified domain
        train_tasks = get_tasks_for_domains([args.holdout_domain])
    else:
        # Default to all MAIR tasks
        train_tasks = list(ALL_MAIR_TASKS)
    
    # Determine evaluation tasks (for out_of_domain and only_domain modes)
    eval_tasks = None
    if args.eval_mode == "out_of_domain":
        if args.eval_tasks:
            # Explicit eval tasks
            eval_tasks = args.eval_tasks
        elif args.holdout_domain:
            # Get tasks from holdout domain
            eval_tasks = get_tasks_for_domains([args.holdout_domain])
        else:
            print("Warning: out_of_domain mode requires --holdout-domain or --eval-tasks")
            print("Falling back to in_domain mode")
            args.eval_mode = "in_domain"
    elif args.eval_mode == "only_domain":
        if args.eval_tasks:
            eval_tasks = args.eval_tasks
        elif args.holdout_domain:
            # Eval on the same domain we trained on (with query splitting)
            eval_tasks = get_tasks_for_domains([args.holdout_domain])
        else:
            print("Warning: only_domain mode requires --holdout-domain or --eval-tasks")
            print("Falling back to in_domain mode")
            args.eval_mode = "in_domain"
    
    print("=" * 70)
    print("Configuration Summary")
    print("=" * 70)
    print(f"Eval mode: {args.eval_mode}")
    print(f"Training tasks ({len(train_tasks)}): {train_tasks[:5]}{'...' if len(train_tasks) > 5 else ''}")
    if eval_tasks:
        print(f"Eval tasks ({len(eval_tasks)}): {eval_tasks[:5]}{'...' if len(eval_tasks) > 5 else ''}")
    print("=" * 70)
    
    config = ERAConfig(
        large_model_name=args.large_model,
        small_model_name=args.small_model,
        task_names=train_tasks,
        training_mode=args.training_mode,
        eval_mode=args.eval_mode,
        eval_task_names=eval_tasks,
        # Alignment params
        alignment_samples_per_task=args.alignment_samples_per_task,
        alignment_epochs=args.alignment_epochs,
        alignment_lr=args.alignment_lr,
        alignment_weight_decay=args.alignment_weight_decay,
        alignment_eval_ratio=args.alignment_eval_ratio,
        alignment_loss_type=args.alignment_loss_type,
        alignment_early_stopping=args.alignment_early_stopping,
        alignment_early_stopping_patience=args.alignment_early_stopping_patience,
        holdout_domain=args.holdout_domain,
        # Label params
        label_train_ratio=args.label_train_ratio,
        label_val_ratio=args.label_val_ratio,
        label_eval_ratio=args.label_eval_ratio,
        label_num_negatives=args.label_num_negatives,
        label_negative_strategy=args.label_negative_strategy,
        label_hard_negative_top_k=args.label_hard_negative_top_k,
        label_hard_negative_perc_margin=args.label_hard_negative_perc_margin,
        label_epochs=args.label_epochs,
        label_lr=args.label_lr,
        label_weight_decay=args.label_weight_decay,
        label_warmup_ratio=args.label_warmup_ratio,
        label_temperature=args.label_temperature,
        label_early_stopping=args.label_early_stopping,
        label_early_stopping_patience=args.label_early_stopping_patience,
        # Shared params
        seed=args.seed,
        batch_size=args.batch_size,
        checkpoint_interval=args.checkpoint_interval,
        cache_dir=args.cache_dir,
        dataset_cache_dir=args.dataset_cache_dir,
        output_dir=args.output_dir,
        pretrained_adapter_path=args.pretrained_adapter_path,
        # Post-training evaluation
        eval_after_training=args.eval_after_training,
        eval_tasks_override=args.eval_tasks_override,
    )
    
    result = train_era(config)
    
    print("\n" + "=" * 70)
    print("Summary:")
    print("=" * 70)
    print(f"Training mode: {config.training_mode}")
    print(f"Final adapter: {result['final_adapter_path']}")
    if result.get('alignment_result'):
        print(f"Alignment adapter: {result['alignment_result']['adapter_path']}")
    if result.get('label_result'):
        print(f"Label adapter: {result['label_result']['adapter_path']}")
    print(f"Total time: {result['total_elapsed_sec']:.2f}s")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
