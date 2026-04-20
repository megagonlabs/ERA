"""
Unified Adapter Training Pipeline.

Combines identical text alignment (self-supervised) and label-based training
for query-side adapter learning. Supports flexible training modes:
- alignment_only: Only use identical text alignment (self-supervised)
- label_only: Only use labeled query-document pairs
- both: First alignment, then label training (default)
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import List, Optional, Dict, Any, Literal

from ..cache_config import EMBEDDING_CACHE_DIR, DATASET_CACHE_DIR
from .identical_text_alignment import AlignmentConfig, train_identical_text_alignment, _build_adapter_filename
from .label_training import LabelTrainingConfig, train_with_labels


TrainingMode = Literal["both", "alignment_only", "label_only"]


@dataclass
class ERAConfig:
    """Unified configuration for adapter training pipeline."""
    
    # Model configuration
    large_model_name: str
    small_model_name: str
    
    # Task configuration
    task_names: List[str]  # List of MAIR task names for training
    
    # Training mode
    training_mode: TrainingMode = "both"  # "both", "alignment_only", "label_only"
    
    # Evaluation mode (for label training)
    eval_mode: str = "in_domain"  # "in_domain", "out_of_domain", or "only_domain"
    eval_task_names: Optional[List[str]] = None  # Tasks for out-of-domain / only-domain evaluation
    holdout_domain: Optional[str] = None  # Domain name for OOD / only-domain evaluation
    
    # Post-training evaluation
    eval_after_training: bool = False  # Run retrieval eval on all tasks after training
    eval_tasks_override: Optional[List[str]] = None  # Custom eval task list (None = all)
    
    # === Alignment configuration ===
    alignment_samples_per_task: int = 1000
    alignment_epochs: int = 10
    alignment_lr: float = 1e-3
    alignment_weight_decay: float = 0.01
    alignment_eval_ratio: float = 0.1
    alignment_loss_type: str = "cosine"  # "cosine", "mse", "smooth_l1"
    
    # === Label training configuration ===
    label_train_ratio: float = 0.1  # Train/eval split ratio for queries
    label_val_ratio: float = 0.1    # Fraction of queries reserved for validation (fixed)
    label_eval_ratio: float = 0.5   # Fraction of queries reserved for evaluation/test (fixed)
    label_num_negatives: int = 5
    label_negative_strategy: str = "random"  # "random", "naive_topk", or "topk_percpos"
    label_hard_negative_top_k: int = 2000
    label_hard_negative_perc_margin: float = 0.95
    label_epochs: int = 1000
    label_lr: float = 1e-5  # Usually lower than alignment
    label_weight_decay: float = 1e-4
    label_warmup_ratio: float = 0.1
    label_temperature: float = 0.05
    # Early stopping for label training (enabled by default)
    label_early_stopping: bool = True
    label_early_stopping_patience: int = 5
    label_early_stopping_min_delta: float = 0.0
    
    # === Shared configuration ===
    seed: int = 42
    batch_size: int = 256
    checkpoint_interval: int = 50

    # Pretrained adapter path (e.g. alignment weights from a prior run)
    pretrained_adapter_path: Optional[str] = None

    # Early stopping for alignment (disabled by default)
    alignment_early_stopping: bool = False
    alignment_early_stopping_patience: int = 5
    alignment_early_stopping_min_delta: float = 0.0

    # Paths
    cache_dir: str = EMBEDDING_CACHE_DIR
    dataset_cache_dir: Optional[str] = None
    output_dir: str = "results/era"
    compact_output_layout: bool = False


def _safe_name(name: str) -> str:
    """Create safe filename from model/task names."""
    return name.replace("/", "__").replace(" ", "_")


def _format_float(value: float) -> str:
    """Format float for filename, stripping trailing zeros."""
    formatted = f"{value:.10f}"
    formatted = formatted.rstrip("0").rstrip(".")
    return "0" if formatted in {"", "-0"} else formatted


def _build_alignment_result_tag(loss_type: str, lr: float, weight_decay: float) -> Optional[str]:
    """Build a compact directory tag for alignment settings.

    The historical default configuration keeps the legacy directory layout so
    existing results remain addressable.
    """
    if loss_type == "cosine" and abs(lr - 1e-3) <= 1e-12 and abs(weight_decay - 0.01) <= 1e-12:
        return None
    return f"alignloss{loss_type}__awd{_format_float(weight_decay)}__alr{_format_float(lr)}"


def _build_ratio_tag(train_ratio: float, val_ratio: float, eval_ratio: float) -> str:
    """Build a label split tag for directory names."""
    ratio_tag = f"train{_format_float(train_ratio)}"
    if abs(val_ratio - 0.1) > 1e-9:
        ratio_tag += f"_val{_format_float(val_ratio)}"
    if abs(eval_ratio - 0.5) > 1e-9:
        ratio_tag += f"_eval{_format_float(eval_ratio)}"
    return ratio_tag


def _build_alignment_run_tag(config: ERAConfig) -> str:
    """Build a directory tag capturing alignment-only hyperparameters."""
    parts = [
        f"alignloss{config.alignment_loss_type}",
        f"ansamp{config.alignment_samples_per_task}",
        f"awd{_format_float(config.alignment_weight_decay)}",
        f"alr{_format_float(config.alignment_lr)}",
        f"aep{config.alignment_epochs}",
    ]
    if abs(config.alignment_eval_ratio - 0.1) > 1e-9:
        parts.append(f"aeval{_format_float(config.alignment_eval_ratio)}")
    return "__".join(parts)


def _build_label_result_tag(
    negative_strategy: str,
    num_negatives: int,
    lr: float,
    weight_decay: float,
    hard_negative_top_k: int,
    hard_negative_perc_margin: float,
) -> Optional[str]:
    """Build a compact directory tag for unified label-training settings.

    Unified runs always include the label loss / negative-sampling identity so
    runs with different label settings never share the same output directory.
    Label learning rate and weight decay are always appended so the optimizer
    configuration is fully recoverable from the path, even for default values.
    """
    tag = f"losscontrastive__neg{negative_strategy}__nneg{num_negatives}"
    if negative_strategy == "naive_topk":
        tag += f"__topk{hard_negative_top_k}"
    elif negative_strategy == "topk_percpos":
        tag += f"__topk{hard_negative_top_k}__pm{_format_float(hard_negative_perc_margin)}"
    tag += f"__lwd{_format_float(weight_decay)}"
    tag += f"__llr{_format_float(lr)}"
    return tag


def _compute_eval_mode_str(config: ERAConfig) -> str:
    """Compute the evaluation-mode portion used in unified run directory names."""
    eval_mode_str = config.eval_mode
    if config.eval_mode == "out_of_domain" and config.eval_task_names:
        if config.holdout_domain:
            eval_mode_str = f"ood_{config.holdout_domain}_{len(config.eval_task_names)}_tasks"
        else:
            eval_mode_str = f"ood_{len(config.eval_task_names)}_tasks"
    elif config.eval_mode == "only_domain" and config.holdout_domain:
        num_tasks = len(config.eval_task_names) if config.eval_task_names else len(config.task_names)
        eval_mode_str = f"onlydomain_{config.holdout_domain}_{num_tasks}_tasks"
    return eval_mode_str


def _build_run_config_fingerprint(config: ERAConfig) -> str:
    """Build a short stable fingerprint to prevent cross-run metadata collisions."""
    payload: Dict[str, Any] = {
        "training_mode": config.training_mode,
        "eval_mode": config.eval_mode,
        "eval_task_names": config.eval_task_names,
        "holdout_domain": config.holdout_domain,
        "eval_after_training": config.eval_after_training,
        "eval_tasks_override": config.eval_tasks_override,
        "task_names": config.task_names,
        "shared": {
            "seed": config.seed,
            "batch_size": config.batch_size,
            "checkpoint_interval": config.checkpoint_interval,
        },
    }
    if config.training_mode in ("both", "alignment_only"):
        payload["alignment"] = {
            "samples_per_task": config.alignment_samples_per_task,
            "epochs": config.alignment_epochs,
            "lr": config.alignment_lr,
            "weight_decay": config.alignment_weight_decay,
            "eval_ratio": config.alignment_eval_ratio,
            "loss_type": config.alignment_loss_type,
            "early_stopping": config.alignment_early_stopping,
            "early_stopping_patience": config.alignment_early_stopping_patience,
            "early_stopping_min_delta": config.alignment_early_stopping_min_delta,
        }
    if config.training_mode in ("both", "label_only"):
        payload["label"] = {
            "train_ratio": config.label_train_ratio,
            "val_ratio": config.label_val_ratio,
            "eval_ratio": config.label_eval_ratio,
            "num_negatives": config.label_num_negatives,
            "negative_strategy": config.label_negative_strategy,
            "hard_negative_top_k": config.label_hard_negative_top_k,
            "hard_negative_perc_margin": config.label_hard_negative_perc_margin,
            "epochs": config.label_epochs,
            "lr": config.label_lr,
            "weight_decay": config.label_weight_decay,
            "warmup_ratio": config.label_warmup_ratio,
            "temperature": config.label_temperature,
            "early_stopping": config.label_early_stopping,
            "early_stopping_patience": config.label_early_stopping_patience,
            "early_stopping_min_delta": config.label_early_stopping_min_delta,
        }

    serialized = json.dumps(payload, sort_keys=True, ensure_ascii=True)
    return hashlib.sha1(serialized.encode("utf-8")).hexdigest()[:10]


def _build_run_dir_name(config: ERAConfig) -> str:
    """Build a run directory name that stays stable and collision-free across modes."""
    parts = [config.training_mode, _compute_eval_mode_str(config)]

    if config.training_mode == "alignment_only":
        parts.append(_build_alignment_run_tag(config))
    else:
        parts.append(
            _build_ratio_tag(
                config.label_train_ratio,
                config.label_val_ratio,
                config.label_eval_ratio,
            )
        )
        if config.training_mode == "both":
            parts.append(_build_alignment_run_tag(config))

    if config.training_mode in ("both", "label_only"):
        label_tag = _build_label_result_tag(
            config.label_negative_strategy,
            config.label_num_negatives,
            config.label_lr,
            config.label_weight_decay,
            config.label_hard_negative_top_k,
            config.label_hard_negative_perc_margin,
        )
        if label_tag is not None:
            parts.append(label_tag)

    parts.append(f"cfg{_build_run_config_fingerprint(config)}")
    parts.append(_compute_task_str(config.task_names))
    return "__".join(parts)


def _build_unified_adapter_filename(config: ERAConfig) -> str:
    """Build a combined adapter filename encoding all hyperparameters.

    For 'both' mode the filename includes both alignment and label params.
    For single-phase modes only the relevant phase params are included.
    Shared params (batch_size, seed) appear once at the end.
    """
    sections: list[str] = ["adapter"]

    # --- alignment section ---
    if config.training_mode in ("both", "alignment_only"):
        align_parts = [
            f"loss{config.alignment_loss_type}",
            f"nsamp{config.alignment_samples_per_task}",
            f"wd{_format_float(config.alignment_weight_decay)}",
            f"lr{_format_float(config.alignment_lr)}",
            f"ep{config.alignment_epochs}",
        ]
        sections.append("align[" + "__".join(align_parts) + "]")

    # --- label section ---
    if config.training_mode in ("both", "label_only"):
        label_parts = [
            "losscontrastive",
            f"neg{config.label_negative_strategy}",
            f"nneg{config.label_num_negatives}",
            f"train{_format_float(config.label_train_ratio)}",
        ]
        if config.label_negative_strategy == "naive_topk":
            label_parts.append(f"topk{config.label_hard_negative_top_k}")
        elif config.label_negative_strategy == "topk_percpos":
            label_parts.append(f"topk{config.label_hard_negative_top_k}")
            label_parts.append(f"pm{_format_float(config.label_hard_negative_perc_margin)}")
        # Include val_ratio only when it differs from default (0.1)
        if abs(config.label_val_ratio - 0.1) > 1e-9:
            label_parts.append(f"val{_format_float(config.label_val_ratio)}")
        # Include eval_ratio only when it differs from default (0.5)
        if abs(config.label_eval_ratio - 0.5) > 1e-9:
            label_parts.append(f"eval{_format_float(config.label_eval_ratio)}")
        label_parts += [
            f"wd{_format_float(config.label_weight_decay)}",
            f"lr{_format_float(config.label_lr)}",
            f"ep{config.label_epochs}",
        ]
        label_parts.append(f"temp{_format_float(config.label_temperature)}")
        sections.append("label[" + "__".join(label_parts) + "]")

    # --- shared params ---
    sections.append(f"bs{config.batch_size}")
    sections.append(f"seed{config.seed}")

    return "__".join(sections) + ".pt"


def _compute_task_str(task_names: List[str]) -> str:
    """Compute the task string suffix used in directory names."""
    return "_".join(task_names) if len(task_names) <= 3 else f"multi_{len(task_names)}_tasks"


def _get_layout_root(config: ERAConfig) -> Path:
    """Return the directory that directly contains per-run directories."""
    if config.compact_output_layout:
        return Path(config.output_dir) / "linear"

    output_base = Path(config.output_dir)
    run_name = f"{_safe_name(config.large_model_name)}__to__{_safe_name(config.small_model_name)}"
    return output_base / run_name / "with_instruction" / "linear"


def _get_run_dir(config: ERAConfig) -> Path:
    """Return the concrete run directory for the current configuration."""
    return _get_layout_root(config) / _build_run_dir_name(config)


def _find_existing_alignment_weights(
    config: ERAConfig,
    alignment_config: AlignmentConfig,
) -> Optional[str]:
    """Search for existing alignment weights that match the current alignment config.

    Alignment training is independent of ``label_train_ratio`` (and other
    label-only parameters), so weights produced by a prior run with the same
    alignment hyper-parameters can safely be reused.  The search looks for a
    matching *alignment adapter filename* (which encodes all alignment
    hyper-parameters) under sibling experiment directories of the same
    model-pair / instruction-mode / adapter-type hierarchy.

    Search order:
            1. Current run's own ``alignment/`` sub-directory.
            2. Canonical ``alignment_only`` sibling directories with the same task suffix.
            3. Other sibling directories with the same task suffix (``train0.5`` preferred first).

    Returns:
        Absolute path string to the matching ``.pt`` file, or ``None``.
    """
    expected_filename = _build_adapter_filename(alignment_config)
    task_str = _compute_task_str(config.task_names)

    current_run_dir = _get_run_dir(config)
    current_adapter_path = current_run_dir / "alignment" / expected_filename
    if current_adapter_path.is_file():
        return str(current_adapter_path)

    candidates: List[Path] = []
    if config.compact_output_layout:
        current_exp_dir = Path(config.output_dir)
        hp_search_root = current_exp_dir.parent
        if not hp_search_root.is_dir():
            return None
        for experiment_dir in sorted(hp_search_root.iterdir()):
            if not experiment_dir.is_dir() or experiment_dir == current_exp_dir:
                continue
            layout_root = experiment_dir
            if not layout_root.is_dir():
                continue
            for child in sorted(layout_root.iterdir()):
                if not child.is_dir():
                    continue
                if not child.name.endswith(f"__{task_str}"):
                    continue
                candidates.append(child)
    else:
        parent_dir = _get_layout_root(config)
        if not parent_dir.is_dir():
            return None
        for child in sorted(parent_dir.iterdir()):
            if not child.is_dir():
                continue
            if child == current_run_dir:
                continue
            if not child.name.endswith(f"__{task_str}"):
                continue
            candidates.append(child)

    if not candidates:
        return None

    # Sort candidates: prefer canonical alignment-only runs first, then the
    # common train0.5 siblings, then any remaining directory.
    def _sort_key(d: Path) -> tuple:
        name = d.name
        if name.startswith("alignment_only__"):
            return (0, name)
        if "__train0.5__" in name:
            return (1, name)
        return (2, name)

    candidates.sort(key=_sort_key)

    for candidate in candidates:
        adapter_path = candidate / "alignment" / expected_filename
        if adapter_path.is_file():
            return str(adapter_path)

    return None


def train_era(config: ERAConfig) -> Dict[str, Any]:
    """
    Run unified adapter training pipeline.
    
    The pipeline supports three modes:
    - "both": First alignment pretraining, then label finetuning
    - "alignment_only": Only self-supervised alignment
    - "label_only": Only supervised label training
    
    Args:
        config: Unified training configuration
    
    Returns:
        Dictionary with training results, paths, and histories
    """
    start_time = time.time()
    results = {
        "training_mode": config.training_mode,
        "alignment_result": None,
        "label_result": None,
        "final_adapter_path": None,
    }
    
    print("=" * 70)
    print("Unified Adapter Training Pipeline")
    print("=" * 70)
    print(f"Training mode: {config.training_mode}")
    print(f"Eval mode: {config.eval_mode}")
    print(f"Large model: {config.large_model_name}")
    print(f"Small model: {config.small_model_name}")
    print(f"Training tasks: {config.task_names}")
    if config.eval_task_names:
        print(f"Eval tasks (held out): {config.eval_task_names}")
    print()
    
    # Create output directory structure
    # L1: model pair  /  L2: instruction mode  /  L3: adapter type  /  L4: experiment spec
    run_dir = _get_run_dir(config)
    run_dir.mkdir(parents=True, exist_ok=True)
    
    alignment_adapter_path = None
    
    # If a pretrained adapter path is explicitly provided, use it directly.
    # This allows sharing alignment weights across different training runs
    # (e.g. global alignment once, then per-domain label tuning), and it
    # suppresses Phase 1 because the caller already chose the alignment state.
    if config.pretrained_adapter_path:
        alignment_adapter_path = config.pretrained_adapter_path
        print(f"\nUsing externally provided pretrained adapter:")
        print(f"  {alignment_adapter_path}")
    
    # Use default dataset cache dir if not specified
    dataset_cache_dir = config.dataset_cache_dir or DATASET_CACHE_DIR
    
    # Phase 1: Alignment Training (if needed)
    if config.training_mode in ["both", "alignment_only"] and not alignment_adapter_path:
        print("\n" + "=" * 70)
        print("Phase 1: Identical Text Alignment")
        print("=" * 70)
        
        alignment_config = AlignmentConfig(
            large_model_name=config.large_model_name,
            small_model_name=config.small_model_name,
            task_names=config.task_names,
            samples_per_task=config.alignment_samples_per_task,
            seed=config.seed,
            batch_size=config.batch_size,
            epochs=config.alignment_epochs,
            lr=config.alignment_lr,
            weight_decay=config.alignment_weight_decay,
            eval_ratio=config.alignment_eval_ratio,
            loss_type=config.alignment_loss_type,
            checkpoint_interval=config.checkpoint_interval,
            cache_dir=config.cache_dir,
            dataset_cache_dir=dataset_cache_dir,
            output_dir=str(run_dir / "alignment"),
            flat_output=True,
            early_stopping=config.alignment_early_stopping,
            early_stopping_patience=config.alignment_early_stopping_patience,
            early_stopping_min_delta=config.alignment_early_stopping_min_delta,
        )

        # In "both" mode, try to reuse existing alignment weights to avoid
        # redundant computation.  Alignment is independent of label_train_ratio
        # so weights from any sibling run with identical alignment hyper-params
        # (encoded in the filename) are safe to reuse.
        reused_alignment_path: Optional[str] = None
        if config.training_mode == "both":
            reused_alignment_path = _find_existing_alignment_weights(
                config, alignment_config,
            )

        if reused_alignment_path is not None:
            print(f"\n>>> Reusing existing alignment weights (skipping Phase 1):")
            print(f"    {reused_alignment_path}")
            alignment_adapter_path = reused_alignment_path
            results["alignment_result"] = {
                "adapter_path": reused_alignment_path,
                "reused": True,
                "source_path": reused_alignment_path,
            }
        else:
            alignment_result = train_identical_text_alignment(alignment_config)
            results["alignment_result"] = alignment_result
            alignment_adapter_path = alignment_result["adapter_path"]

        if config.training_mode == "alignment_only":
            results["final_adapter_path"] = alignment_adapter_path
            print(f"\nAlignment-only training completed.")
            print(f"Final adapter: {alignment_adapter_path}")
    elif config.training_mode == "alignment_only" and alignment_adapter_path:
        results["alignment_result"] = {
            "adapter_path": alignment_adapter_path,
            "reused": True,
            "source_path": alignment_adapter_path,
            "provided_pretrained_adapter": True,
        }
        results["final_adapter_path"] = alignment_adapter_path
        print("\nSkipping Phase 1 because pretrained_adapter_path was provided.")
        print(f"Final adapter: {alignment_adapter_path}")
    
    # Phase 2: Label Training (if needed)
    if config.training_mode in ["both", "label_only"]:
        print("\n" + "=" * 70)
        print("Phase 2: Label-based Training")
        print("=" * 70)
        
        label_config = LabelTrainingConfig(
            large_model_name=config.large_model_name,
            small_model_name=config.small_model_name,
            task_names=config.task_names,
            eval_mode=config.eval_mode,
            eval_task_names=config.eval_task_names,
            train_ratio=config.label_train_ratio,
            val_ratio=config.label_val_ratio,
            eval_ratio=config.label_eval_ratio,
            seed=config.seed,
            num_negatives=config.label_num_negatives,
            negative_strategy=config.label_negative_strategy,
            hard_negative_top_k=config.label_hard_negative_top_k,
            hard_negative_perc_margin=config.label_hard_negative_perc_margin,
            batch_size=config.batch_size,
            epochs=config.label_epochs,
            lr=config.label_lr,
            weight_decay=config.label_weight_decay,
            warmup_ratio=config.label_warmup_ratio,
            temperature=config.label_temperature,
            cache_dir=config.cache_dir,
            dataset_cache_dir=config.dataset_cache_dir,
            output_dir=str(run_dir / "label"),
            checkpoint_interval=config.checkpoint_interval,
            pretrained_adapter_path=alignment_adapter_path,  # Use alignment as init for "both"
            flat_output=True,
            early_stopping=config.label_early_stopping,
            early_stopping_patience=config.label_early_stopping_patience,
            early_stopping_min_delta=config.label_early_stopping_min_delta,
        )
        
        label_result = train_with_labels(label_config)
        results["label_result"] = label_result
        results["final_adapter_path"] = label_result["adapter_path"]
    
    # Copy final adapter to run_dir with a unified filename encoding all config
    if results["final_adapter_path"]:
        unified_filename = _build_unified_adapter_filename(config)
        unified_adapter_path = run_dir / unified_filename
        src_path = Path(results["final_adapter_path"])
        if src_path != unified_adapter_path:
            shutil.copy2(str(src_path), str(unified_adapter_path))
            print(f"\nFinal adapter (unified name): {unified_adapter_path}")
        results["final_adapter_path"] = str(unified_adapter_path)
    
    # Save unified metadata
    total_time = round(time.time() - start_time, 2)
    results["total_elapsed_sec"] = total_time
    results["config"] = asdict(config)
    
    meta_path = run_dir / "era_meta.json"
    with meta_path.open("w", encoding="utf-8") as f:
        # Convert to JSON-serializable format
        json_results = {
            k: v for k, v in results.items() 
            if k != "config"  # Config is already serializable
        }
        json_results["config"] = asdict(config)
        json.dump(json_results, f, ensure_ascii=False, indent=2)
    
    results["meta_path"] = str(meta_path)
    
    # Phase 3: Post-training evaluation (if requested)
    if config.eval_after_training and results["final_adapter_path"]:
        print("\n" + "=" * 70)
        print("Phase 3: Post-Training Retrieval Evaluation")
        print("=" * 70)
        
        try:
            # Import here to avoid circular dependency
            sys_path = str(Path(__file__).parent.parent.parent)
            import sys
            import torch
            if sys_path not in sys.path:
                sys.path.insert(0, sys_path)
            from scripts.run_adapter_batch_evaluation import evaluate_adapter_on_tasks
            from ..evaluation.mair_evaluator import ALL_MAIR_TASKS, MAIRDataset
            from .label_training import split_queries_by_id

            visible_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
            if visible_gpus > 1:
                print(f"\nPost-training retrieval eval will use all visible GPUs ({visible_gpus}) for adapter inference.")
            elif visible_gpus == 1:
                print("\nPost-training retrieval eval will run on a single visible GPU.")
            else:
                print("\nPost-training retrieval eval will run on CPU.")

            # eval_tasks_override always wins; otherwise mode-specific default.
            if config.eval_tasks_override:
                eval_task_list = config.eval_tasks_override
            elif config.eval_mode in ("out_of_domain", "only_domain") and config.eval_task_names:
                eval_task_list = config.eval_task_names
            else:
                eval_task_list = list(ALL_MAIR_TASKS)
            eval_output_dir = str(run_dir / "eval_results")

            # Compute eval query IDs / eval task list to avoid data leakage.
            # For ALL modes, training tasks had their queries split (train/eval).
            # We must evaluate ONLY on the eval portion so results are comparable
            # across in_domain / out_of_domain / alignment_only / label_only / both.
            eval_query_ids_per_task: Optional[Dict[str, List[str]]] = None

            # Reuse eval_query_ids from label training if available (avoids reloading datasets)
            label_eval_ids = (results.get("label_result") or {}).get("eval_query_ids_per_task")
            if label_eval_ids:
                eval_query_ids_per_task = label_eval_ids
                print("\nReusing eval query splits from label training cache.")
                for tn, eqids in label_eval_ids.items():
                    print(f"  {tn}: {len(eqids)} eval queries")
            elif config.task_names:
                print("\nComputing eval query splits for leakage-free evaluation...")
                eval_query_ids_per_task = {}
                for task_name in config.task_names:
                    try:
                        ds = MAIRDataset(task_name, cache_dir=config.dataset_cache_dir)
                        ds.load()
                        _, _, eval_qids = split_queries_by_id(
                            ds.qrels,
                            config.label_train_ratio,
                            config.seed,
                            eval_ratio=config.label_eval_ratio,
                            val_ratio=config.label_val_ratio,
                        )
                        eval_query_ids_per_task[task_name] = eval_qids
                        print(f"  {task_name}: {len(eval_qids)} eval queries "
                              f"(train_ratio={config.label_train_ratio}, "
                              f"val_ratio={config.label_val_ratio}, "
                              f"eval_ratio={config.label_eval_ratio}, seed={config.seed})")
                    except Exception as e:
                        print(f"  Warning: Could not compute eval split for {task_name}: {e}")
            if config.eval_mode == "out_of_domain":
                if config.eval_task_names:
                    print(f"\n[out_of_domain] Evaluating only held-out tasks "
                          f"({len(eval_task_list)} tasks): {eval_task_list}")
                else:
                    print("\n[out_of_domain] Warning: eval_task_names not set; "
                          "falling back to all tasks.")
            elif config.eval_mode == "only_domain":
                if config.eval_task_names:
                    print(f"\n[only_domain] Evaluating on target domain tasks "
                          f"({len(eval_task_list)} tasks): {eval_task_list}")
                else:
                    print("\n[only_domain] Warning: eval_task_names not set; "
                          "falling back to all tasks.")

            eval_results = evaluate_adapter_on_tasks(
                adapter_path=results["final_adapter_path"],
                large_model_name=config.large_model_name,
                small_model_name=config.small_model_name,
                task_names=eval_task_list,
                cache_dir=config.cache_dir,
                output_dir=eval_output_dir,
                eval_query_ids_per_task=eval_query_ids_per_task,
            )
            results["eval_results_path"] = eval_output_dir
            results["eval_num_tasks"] = sum(1 for r in eval_results.values() if "error" not in r)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Post-training evaluation failed: {e}")
            results["eval_error"] = str(e)
    
    print("\n" + "=" * 70)
    print("Unified Training Pipeline Completed!")
    print("=" * 70)
    print(f"Training mode: {config.training_mode}")
    print(f"Final adapter: {results['final_adapter_path']}")
    print(f"Metadata: {meta_path}")
    print(f"Total time: {total_time:.2f}s")
    
    return results


def create_era_config_from_args(args) -> ERAConfig:
    """
    Create ERAConfig from argparse namespace.
    
    This utility function helps convert command-line arguments to config.
    """
    return ERAConfig(
        large_model_name=args.large_model,
        small_model_name=args.small_model,
        task_names=args.tasks,
        training_mode=args.training_mode,
        # Alignment params
        alignment_samples_per_task=getattr(args, 'alignment_samples_per_task', 1000),
        alignment_epochs=getattr(args, 'alignment_epochs', 10),
        alignment_lr=getattr(args, 'alignment_lr', 1e-3),
        alignment_weight_decay=getattr(args, 'alignment_weight_decay', 0.01),
        alignment_eval_ratio=getattr(args, 'alignment_eval_ratio', 0.1),
        alignment_loss_type=getattr(args, 'alignment_loss_type', 'cosine'),
        alignment_early_stopping=getattr(args, 'alignment_early_stopping', False),
        alignment_early_stopping_patience=getattr(args, 'alignment_early_stopping_patience', 5),
        # Label params
        label_train_ratio=getattr(args, 'label_train_ratio', 0.5),
        label_val_ratio=getattr(args, 'label_val_ratio', 0.1),
        label_eval_ratio=getattr(args, 'label_eval_ratio', 0.5),
        label_num_negatives=getattr(args, 'label_num_negatives', 5),
        label_negative_strategy=getattr(args, 'label_negative_strategy', 'random'),
        label_hard_negative_top_k=getattr(args, 'label_hard_negative_top_k', 2000),
        label_hard_negative_perc_margin=getattr(args, 'label_hard_negative_perc_margin', 0.95),
        label_epochs=getattr(args, 'label_epochs', 10),
        label_lr=getattr(args, 'label_lr', 1e-4),
        label_weight_decay=getattr(args, 'label_weight_decay', 0.01),
        label_warmup_ratio=getattr(args, 'label_warmup_ratio', 0.1),
        label_temperature=getattr(args, 'label_temperature', 0.05),
        label_early_stopping=getattr(args, 'label_early_stopping', True),
        label_early_stopping_patience=getattr(args, 'label_early_stopping_patience', 5),
        # Shared params
        seed=getattr(args, 'seed', 42),
        batch_size=getattr(args, 'batch_size', 256),
        checkpoint_interval=getattr(args, 'checkpoint_interval', 50),
        cache_dir=getattr(args, 'cache_dir', EMBEDDING_CACHE_DIR),
        dataset_cache_dir=getattr(args, 'dataset_cache_dir', None),
        output_dir=getattr(args, 'output_dir', 'results/era'),
        pretrained_adapter_path=getattr(args, 'pretrained_adapter_path', None),
    )
