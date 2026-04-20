"""
Identical Text Alignment for query-side adapter initialization.

This module trains a lightweight adapter that maps large-model embeddings
into the small-model embedding space using identical text pairs.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Dict, Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ..cache_config import EMBEDDING_CACHE_DIR, DATASET_CACHE_DIR
from ..evaluation.embedding_cache import EmbeddingCache
from ..evaluation.mair_evaluator import MAIRDataset


@dataclass
class AlignmentConfig:
    """Configuration for identical text alignment."""

    large_model_name: str
    small_model_name: str
    task_names: List[str]  # List of task names to sample from
    samples_per_task: int = 1000  # Number of samples to take from each task
    seed: int = 42
    batch_size: int = 256
    epochs: int = 10
    lr: float = 1e-3
    weight_decay: float = 0.01
    eval_ratio: float = 0.1
    loss_type: str = "cosine"  # "cosine", "mse", or "smooth_l1"
    checkpoint_interval: int = 50  # Save checkpoint every N epochs
    cache_dir: str = EMBEDDING_CACHE_DIR
    dataset_cache_dir: str = DATASET_CACHE_DIR
    output_dir: str = "results/adapter_alignment"
    max_length: Optional[int] = None
    flat_output: bool = False  # When True, use output_dir directly as run_dir (no sub-hierarchy)
    # Early stopping (disabled by default for alignment)
    early_stopping: bool = False
    early_stopping_patience: int = 5
    early_stopping_min_delta: float = 0.0


class LinearAdapter(nn.Module):
    """Simple linear projection adapter with L2 normalization."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.proj(x)
        # L2 normalize so that the output vector has unit length (sum of squares = 1)
        return torch.nn.functional.normalize(output, p=2, dim=-1)



def _safe_name(name: str) -> str:
    return name.replace("/", "__").replace(" ", "_")


def _format_float(value: float) -> str:
    formatted = f"{value:.10f}"
    formatted = formatted.rstrip("0").rstrip(".")
    return "0" if formatted in {"", "-0"} else formatted


def _build_adapter_filename(config: AlignmentConfig, epoch_suffix: str = "") -> str:
    parts = [
        "adapter",
        f"loss{config.loss_type}",
        f"nsamp{config.samples_per_task}",
        f"wd{_format_float(config.weight_decay)}",
        f"lr{_format_float(config.lr)}",
        f"epochs{config.epochs}",
        f"bs{config.batch_size}",
        f"seed{config.seed}",
    ]
    return "__".join(parts) + epoch_suffix + ".pt"


def _sample_corpus_texts(
    task_name: str,
    sample_size: int,
    seed: int,
    dataset_cache_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Sample corpus texts and return texts with their original indices."""
    dataset = MAIRDataset(task_name, cache_dir=dataset_cache_dir)
    dataset.load()
    all_texts = list(dataset.corpus.values())

    rng = random.Random(seed)
    
    if sample_size >= len(all_texts):
        indices = list(range(len(all_texts)))
        rng.shuffle(indices)
        sampled_texts = [all_texts[i] for i in indices]
        return {"texts": sampled_texts, "indices": indices, "all_texts": all_texts, "task_name": task_name}
    
    # Random sample with indices
    indices = rng.sample(range(len(all_texts)), sample_size)
    sampled_texts = [all_texts[i] for i in indices]
    
    return {"texts": sampled_texts, "indices": indices, "all_texts": all_texts, "task_name": task_name}


def _sample_from_multiple_tasks(
    task_names: List[str],
    samples_per_task: int,
    seed: int,
    dataset_cache_dir: Optional[str],
) -> Dict[str, Any]:
    """Sample corpus texts from multiple tasks."""
    all_sampled_texts = []
    all_task_metadata = []  # Store task name, indices, and all_texts for each task
    
    for task_name in task_names:
        sample_result = _sample_corpus_texts(
            task_name=task_name,
            sample_size=samples_per_task,
            seed=seed,
            dataset_cache_dir=dataset_cache_dir,
        )
        all_sampled_texts.extend(sample_result["texts"])
        all_task_metadata.append({
            "task_name": task_name,
            "indices": sample_result["indices"],
            "all_texts": sample_result["all_texts"],
            "num_samples": len(sample_result["texts"])
        })
    
    print(f"Sampled {len(all_sampled_texts)} texts from {len(task_names)} tasks")
    for meta in all_task_metadata:
        print(f"  - {meta['task_name']}: {meta['num_samples']} samples")
    
    return {
        "texts": all_sampled_texts,
        "task_metadata": all_task_metadata,
    }


def _train_eval_split(texts: List[str], eval_ratio: float, seed: int) -> Dict[str, Any]:
    if not 0.0 < eval_ratio < 1.0:
        raise ValueError("eval_ratio must be between 0 and 1.")

    rng = random.Random(seed)
    indices = list(range(len(texts)))
    rng.shuffle(indices)

    eval_size = max(1, int(len(texts) * eval_ratio))
    eval_index_set = set(indices[:eval_size])

    train_indices = [i for i in range(len(texts)) if i not in eval_index_set]
    eval_indices = [i for i in range(len(texts)) if i in eval_index_set]

    train_texts = [texts[i] for i in train_indices]
    eval_texts = [texts[i] for i in eval_indices]

    return {
        "train": train_texts,
        "eval": eval_texts,
        "train_indices": train_indices,
        "eval_indices": eval_indices,
    }


def _load_or_create_split_cache(
    model_name: str,
    all_texts: List[str],
    split_texts: List[str],
    split_indices: List[int],
    cache_dir: str,
    task_name: str,
    split_name: str,
    base_splits: List[Optional[str]],
) -> Optional[np.ndarray]:
    cache = EmbeddingCache(cache_dir=cache_dir, enabled=True, force_recache=False)

    cached = cache.get(model_name, split_texts, task_name=task_name, split=split_name)
    if cached is not None:
        return cached

    for base_split in base_splits:
        base_cached = cache.get(model_name, all_texts, task_name=task_name, split=base_split)
        if base_cached is None:
            continue

        subset = base_cached[split_indices]
        cache.put(model_name, split_texts, subset, task_name=task_name, split=split_name)
        return subset

    return None


def train_identical_text_alignment(config: AlignmentConfig) -> Dict[str, Any]:
    """Train a linear adapter using identical text alignment from multiple tasks."""

    start_time = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Sample from multiple tasks
    sample_result = _sample_from_multiple_tasks(
        task_names=config.task_names,
        samples_per_task=config.samples_per_task,
        seed=config.seed,
        dataset_cache_dir=config.dataset_cache_dir,
    )
    
    texts = sample_result["texts"]
    task_metadata = sample_result["task_metadata"]

    split = _train_eval_split(texts, eval_ratio=config.eval_ratio, seed=config.seed)
    train_texts = split["train"]
    eval_texts = split["eval"]
    train_indices_in_sample = split["train_indices"]
    eval_indices_in_sample = split["eval_indices"]
    
    # Load embeddings from multiple task caches
    train_large_embeddings_list = []
    train_small_embeddings_list = []
    eval_large_embeddings_list = []
    eval_small_embeddings_list = []
    
    text_offset = 0
    for task_meta in task_metadata:
        task_name = task_meta["task_name"]
        task_indices = task_meta["indices"]
        task_all_texts = task_meta["all_texts"]
        num_samples = task_meta["num_samples"]
        mair_task_name = f"MAIR_{task_name}"
        
        # Determine which texts belong to this task
        task_text_range = range(text_offset, text_offset + num_samples)
        
        # Find train/eval texts for this task (use original texts, not train_texts/eval_texts)
        task_train_texts = [texts[i] for i in train_indices_in_sample if i in task_text_range]
        task_eval_texts = [texts[i] for i in eval_indices_in_sample if i in task_text_range]
        
        # Map back to corpus indices
        task_train_indices_in_corpus = [task_indices[i - text_offset] for i in train_indices_in_sample if i in task_text_range]
        task_eval_indices_in_corpus = [task_indices[i - text_offset] for i in eval_indices_in_sample if i in task_text_range]
        
        if task_train_texts:
            train_large = _load_or_create_split_cache(
                config.large_model_name,
                task_all_texts,
                task_train_texts,
                task_train_indices_in_corpus,
                cache_dir=config.cache_dir,
                task_name=mair_task_name,
                split_name="alignment_large_train",
                base_splits=["corpus", None],
            )
            train_small = _load_or_create_split_cache(
                config.small_model_name,
                task_all_texts,
                task_train_texts,
                task_train_indices_in_corpus,
                cache_dir=config.cache_dir,
                task_name=mair_task_name,
                split_name="alignment_small_train",
                base_splits=["corpus", None],
            )
            if train_large is not None and train_small is not None:
                train_large_embeddings_list.append(train_large)
                train_small_embeddings_list.append(train_small)
        
        if task_eval_texts:
            eval_large = _load_or_create_split_cache(
                config.large_model_name,
                task_all_texts,
                task_eval_texts,
                task_eval_indices_in_corpus,
                cache_dir=config.cache_dir,
                task_name=mair_task_name,
                split_name="alignment_large_eval",
                base_splits=["corpus", None],
            )
            eval_small = _load_or_create_split_cache(
                config.small_model_name,
                task_all_texts,
                task_eval_texts,
                task_eval_indices_in_corpus,
                cache_dir=config.cache_dir,
                task_name=mair_task_name,
                split_name="alignment_small_eval",
                base_splits=["corpus", None],
            )
            if eval_large is not None and eval_small is not None:
                eval_large_embeddings_list.append(eval_large)
                eval_small_embeddings_list.append(eval_small)
        
        text_offset += num_samples
    
    # Concatenate all embeddings
    if not train_large_embeddings_list or not train_small_embeddings_list:
        raise RuntimeError("Missing cached train embeddings. Prepare cache before running alignment.")
    if not eval_large_embeddings_list or not eval_small_embeddings_list:
        raise RuntimeError("Missing cached eval embeddings. Prepare cache before running alignment.")
    
    train_large_embeddings = np.concatenate(train_large_embeddings_list, axis=0)
    train_small_embeddings = np.concatenate(train_small_embeddings_list, axis=0)
    eval_large_embeddings = np.concatenate(eval_large_embeddings_list, axis=0)
    eval_small_embeddings = np.concatenate(eval_small_embeddings_list, axis=0)

    if len(train_large_embeddings) != len(train_small_embeddings):
        raise ValueError("Embedding count mismatch between large and small models.")

    if len(eval_large_embeddings) != len(eval_small_embeddings):
        raise ValueError("Eval embedding count mismatch between large and small models.")

    in_dim = int(train_large_embeddings.shape[1])
    out_dim = int(train_small_embeddings.shape[1])

    adapter = LinearAdapter(in_dim, out_dim).to(device)
    print(f"Using Linear adapter: {in_dim} -> {out_dim}")

    x = torch.tensor(train_large_embeddings, dtype=torch.float32)
    y = torch.tensor(train_small_embeddings, dtype=torch.float32)

    x_eval = torch.tensor(eval_large_embeddings, dtype=torch.float32)
    y_eval = torch.tensor(eval_small_embeddings, dtype=torch.float32)

    train_dataset = TensorDataset(x, y)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

    eval_dataset = TensorDataset(x_eval, y_eval)
    eval_loader = DataLoader(eval_dataset, batch_size=config.batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(adapter.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    
    # Select loss function based on config
    if config.loss_type == "cosine":
        def cosine_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            """Cosine similarity loss: 1 - cosine_similarity"""
            return 1 - torch.cosine_similarity(pred, target, dim=-1).mean()
        loss_fn = cosine_loss
        print(f"Using Cosine Similarity Loss")
    elif config.loss_type == "mse":
        loss_fn = nn.MSELoss()
        print(f"Using MSE Loss")
    elif config.loss_type == "smooth_l1":
        loss_fn = nn.SmoothL1Loss()
        print(f"Using Smooth L1 Loss")
    else:
        raise ValueError(f"Unknown loss_type: {config.loss_type}. Choose from 'cosine', 'mse', or 'smooth_l1'.")

    history = []
    
    # For checkpoint saving
    if config.flat_output:
        # Called from unified training: output_dir is already the final run directory
        run_dir = Path(config.output_dir)
    else:
        # Standalone mode: build full hierarchy
        output_base = Path(config.output_dir)
        run_name = f"{_safe_name(config.large_model_name)}__to__{_safe_name(config.small_model_name)}"
        task_str = "_".join(config.task_names) if len(config.task_names) <= 3 else f"multi_{len(config.task_names)}_tasks"
        run_dir = output_base / run_name / f"{task_str}__n{len(texts)}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_interval = config.checkpoint_interval
    checkpoint_paths = []
    
    # Evaluate at epoch 0 (before training)
    print("\n=== Epoch 0 (Before Training) ===")
    adapter.eval()
    initial_train_loss = 0.0
    initial_eval_loss = 0.0
    
    with torch.no_grad():
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            pred = adapter(batch_x)
            loss = loss_fn(pred, batch_y)
            initial_train_loss += loss.item() * batch_x.size(0)
        
        for batch_x, batch_y in eval_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            pred = adapter(batch_x)
            loss = loss_fn(pred, batch_y)
            initial_eval_loss += loss.item() * batch_x.size(0)
    
    initial_train_loss /= len(train_dataset)
    initial_eval_loss /= len(eval_dataset)
    history.append({"epoch": 0, "train_loss": initial_train_loss, "eval_loss": initial_eval_loss})
    print(f"Epoch 0/{config.epochs} - Train Loss: {initial_train_loss:.6f} - Eval Loss: {initial_eval_loss:.6f}")
    
    # Save epoch 0 checkpoint
    epoch0_path = run_dir / _build_adapter_filename(config, epoch_suffix="_epoch0")
    torch.save(adapter.state_dict(), epoch0_path)
    checkpoint_paths.append((0, str(epoch0_path)))
    print(f"  Saved checkpoint: {epoch0_path.name}")
    
    adapter.train()

    # Early stopping state
    best_eval_loss = float("inf")
    best_epoch = 0
    best_state_dict = None
    patience_counter = 0
    stopped_early = False

    for epoch in range(1, config.epochs + 1):
        epoch_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad(set_to_none=True)
            pred = adapter(batch_x)
            loss = loss_fn(pred, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * batch_x.size(0)

        epoch_loss /= len(train_dataset)

        adapter.eval()
        eval_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in eval_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                pred = adapter(batch_x)
                loss = loss_fn(pred, batch_y)
                eval_loss += loss.item() * batch_x.size(0)

        eval_loss /= len(eval_dataset)
        history.append({"epoch": epoch, "train_loss": epoch_loss, "eval_loss": eval_loss})
        print(f"Epoch {epoch}/{config.epochs} - Train Loss: {epoch_loss:.6f} - Eval Loss: {eval_loss:.6f}")
        
        # Early stopping check
        if config.early_stopping:
            if eval_loss < best_eval_loss - config.early_stopping_min_delta:
                best_eval_loss = eval_loss
                best_epoch = epoch
                best_state_dict = {k: v.clone() for k, v in adapter.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= config.early_stopping_patience:
                    print(f"\n*** Early stopping triggered at epoch {epoch} ***")
                    print(f"    Best eval loss: {best_eval_loss:.6f} at epoch {best_epoch}")
                    stopped_early = True
        else:
            # Track best even without early stopping (for metadata)
            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                best_epoch = epoch

        # Save checkpoint at intervals
        if epoch % checkpoint_interval == 0 or epoch == config.epochs:
            checkpoint_path = run_dir / _build_adapter_filename(config, epoch_suffix=f"_epoch{epoch}")
            torch.save(adapter.state_dict(), checkpoint_path)
            checkpoint_paths.append((epoch, str(checkpoint_path)))
            print(f"  Saved checkpoint: {checkpoint_path.name}")
        
        if stopped_early:
            break
        
        adapter.train()

    # Restore best model if early stopping was used and we have a best state
    if config.early_stopping and best_state_dict is not None:
        adapter.load_state_dict(best_state_dict)
        print(f"\nRestored best model from epoch {best_epoch} (eval_loss={best_eval_loss:.6f})")

    adapter.eval()

    # Save final adapter
    adapter_path = run_dir / _build_adapter_filename(config)
    torch.save(adapter.state_dict(), adapter_path)

    meta = {
        "config": asdict(config),
        "in_dim": in_dim,
        "out_dim": out_dim,
        "num_texts": len(texts),
        "num_train_texts": len(train_texts),
        "num_eval_texts": len(eval_texts),
        "train_history": history,
        "checkpoint_paths": checkpoint_paths,
        "stopped_early": stopped_early,
        "best_epoch": best_epoch if best_epoch > 0 else None,
        "best_eval_loss": best_eval_loss if best_eval_loss < float("inf") else None,
        "elapsed_sec": round(time.time() - start_time, 2),
    }

    meta_path = run_dir / "training_meta.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # Clean up GPU memory before returning
    del adapter, optimizer, train_loader, eval_loader, x, y, x_eval, y_eval
    if device == "cuda":
        torch.cuda.empty_cache()
    
    return {
        "adapter_path": str(adapter_path),
        "meta_path": str(meta_path),
        "history": history,
        "checkpoint_paths": checkpoint_paths,
        "stopped_early": stopped_early,
        "best_epoch": best_epoch if best_epoch > 0 else None,
    }

