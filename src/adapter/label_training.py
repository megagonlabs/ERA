"""
Label-based Training for Query-Side Adapter.

This module trains a query adapter using labeled query-document pairs from MAIR.
Supports contrastive loss (InfoNCE).
"""

from __future__ import annotations

import hashlib
import json
import random
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple, cast

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from ..cache_config import EMBEDDING_CACHE_DIR
from ..evaluation.embedding_cache import EmbeddingCache
from ..evaluation.mair_evaluator import MAIRDataset


HARD_NEGATIVE_STRATEGIES = {"naive_topk", "topk_percpos"}
NEGATIVE_MINING_BATCH_SIZE = 4096


@dataclass
class LabelTrainingConfig:
    """Configuration for label-based adapter training."""
    
    # Model configuration
    large_model_name: str
    small_model_name: str
    
    # Task configuration
    task_names: List[str]  # List of MAIR task names for training
    
    # Evaluation mode
    eval_mode: str = "in_domain"  # "in_domain" or "out_of_domain"
    eval_task_names: Optional[List[str]] = None  # Tasks for out-of-domain evaluation
    
    # Train/val/eval split (used for in_domain mode)
    train_ratio: float = 0.5  # Ratio of total queries to use for training (max: 1 - val_ratio - eval_ratio)
    val_ratio: float = 0.1    # Ratio of total queries reserved for validation (fixed)
    eval_ratio: float = 0.5   # Ratio of total queries reserved for evaluation/test (fixed)
    seed: int = 42
    
    # Negative sampling
    num_negatives: int = 5  # Number of negatives per positive
    negative_strategy: str = "topk_percpos"  # "random", "naive_topk", or "topk_percpos"
    hard_negative_top_k: int = 2000
    hard_negative_perc_margin: float = 0.95
    
    # Training hyperparameters
    batch_size: int = 64
    epochs: int = 10
    lr: float = 1e-3
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    
    # Loss configuration
    temperature: float = 0.05  # Temperature for contrastive loss
    
    # Early stopping
    early_stopping: bool = True   # Enable early stopping based on validation loss
    early_stopping_patience: int = 5  # Stop after N epochs without improvement
    early_stopping_min_delta: float = 0.0  # Minimum improvement threshold
    
    # Paths
    cache_dir: str = EMBEDDING_CACHE_DIR
    dataset_cache_dir: Optional[str] = None
    output_dir: str = "results/adapter_label_training"
    
    # Checkpointing
    checkpoint_interval: int = 50
    
    # Pre-trained adapter (for continued training or alignment initialization)
    pretrained_adapter_path: Optional[str] = None
    
    # Output mode
    flat_output: bool = False  # When True, use output_dir directly as run_dir (no sub-hierarchy)


@dataclass 
class QueryDocPair:
    """A single query-document training sample."""
    query_idx: int  # Index in query embedding array
    positive_idx: int  # Index of positive doc in corpus embedding array
    positive_score: float  # Relevance score of the sampled positive document
    negative_indices: List[int]  # Indices of negative docs


class QueryDocDataset(Dataset):
    """Dataset for query-document pairs with cached embeddings."""
    
    def __init__(
        self,
        query_embeddings: np.ndarray,
        doc_embeddings: np.ndarray,
        pairs: List[QueryDocPair],
    ):
        self.query_embeddings = query_embeddings
        self.doc_embeddings = doc_embeddings
        self.pairs = pairs
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        pair = self.pairs[idx]
        
        query_emb = torch.tensor(self.query_embeddings[pair.query_idx], dtype=torch.float32)
        positive_emb = torch.tensor(self.doc_embeddings[pair.positive_idx], dtype=torch.float32)
        positive_score = torch.tensor(pair.positive_score, dtype=torch.float32)
        negative_embs = torch.tensor(
            self.doc_embeddings[pair.negative_indices], dtype=torch.float32
        )
        
        return query_emb, positive_emb, negative_embs, positive_score


class LinearAdapter(nn.Module):
    """Simple linear projection adapter with L2 normalization."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.proj(x)
        return torch.nn.functional.normalize(output, p=2, dim=-1)



def _safe_name(name: str) -> str:
    """Create safe filename from model/task names."""
    return name.replace("/", "__").replace(" ", "_")


def _format_pretrained_adapter_identity(pretrained_adapter_path: Optional[str]) -> Optional[Dict[str, Any]]:
    """Build a cache-safe identity for a pretrained adapter checkpoint."""
    if not pretrained_adapter_path:
        return None

    adapter_path = Path(pretrained_adapter_path)
    identity: Dict[str, Any] = {"path": str(adapter_path)}
    if adapter_path.is_file():
        stat = adapter_path.stat()
        identity.update(
            {
                "resolved_path": str(adapter_path.resolve()),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )
    return identity


def _normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """L2-normalize embeddings for cosine-similarity retrieval."""
    embeddings = np.asarray(embeddings, dtype=np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return embeddings / norms


def _load_mining_adapter(adapter_path: str, device: str) -> nn.Module:
    """Load a trained adapter checkpoint for hard-negative mining."""
    checkpoint = torch.load(adapter_path, map_location=device)

    if "proj.weight" not in checkpoint:
        raise ValueError(f"Expected linear adapter checkpoint (with 'proj.weight'), got keys: {list(checkpoint.keys())}")

    out_dim, in_dim = checkpoint["proj.weight"].shape
    adapter = LinearAdapter(in_dim, out_dim)
    adapter.load_state_dict(checkpoint)
    adapter.to(device)
    adapter.eval()
    return adapter


def _adapt_embeddings_for_mining(
    query_embeddings: np.ndarray,
    adapter: nn.Module,
    device: str,
) -> np.ndarray:
    """Adapt query embeddings into the document space for hard-negative mining."""
    adapted_batches = []
    with torch.no_grad():
        for start in range(0, len(query_embeddings), NEGATIVE_MINING_BATCH_SIZE):
            batch = torch.as_tensor(
                query_embeddings[start:start + NEGATIVE_MINING_BATCH_SIZE],
                dtype=torch.float32,
                device=device,
            )
            adapted = adapter(batch)
            adapted_batches.append(adapted.cpu().numpy())

    return _normalize_embeddings(np.concatenate(adapted_batches, axis=0))


def _prepare_teacher_query_embeddings(
    query_embeddings: np.ndarray,
    doc_embeddings: np.ndarray,
    negative_strategy: str,
    mining_adapter: Optional[nn.Module],
    device: str,
    pretrained_adapter_path: Optional[str],
    fallback_query_embeddings: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Prepare teacher-side query embeddings used for hard-negative retrieval."""
    if mining_adapter is not None:
        return _adapt_embeddings_for_mining(query_embeddings, mining_adapter, device)

    if query_embeddings.shape[1] == doc_embeddings.shape[1]:
        return _normalize_embeddings(query_embeddings)

    if fallback_query_embeddings is not None:
        if fallback_query_embeddings.shape[1] != doc_embeddings.shape[1]:
            raise ValueError(
                "Fallback hard-negative query embeddings must already be in the document space. "
                f"Got fallback dim={fallback_query_embeddings.shape[1]} and doc dim={doc_embeddings.shape[1]}."
            )
        return _normalize_embeddings(fallback_query_embeddings)

    raise ValueError(
        "Hard negative mining strategy "
        f"'{negative_strategy}' requires query/doc embeddings in the same space or "
        "a pretrained adapter checkpoint. For label_only without a pretrained adapter, "
        "populate the small-model query cache so it can be used as the mining teacher. "
        f"Current shapes are {query_embeddings.shape[1]} and {doc_embeddings.shape[1]}. "
        f"Pass --pretrained-adapter-path when using '{negative_strategy}' for asymmetric models. "
        f"Current pretrained_adapter_path={pretrained_adapter_path!r}."
    )


def _rank_topk_candidate_indices(
    similarity_scores: np.ndarray,
    excluded_local_indices: set[int],
    top_k: int,
) -> List[int]:
    """Rank the top-k valid candidate negatives for one query."""
    available_count = len(similarity_scores) - len(excluded_local_indices)
    if available_count <= 0:
        return []

    k = min(top_k, available_count)
    masked_scores = similarity_scores.copy()
    if excluded_local_indices:
        masked_scores[list(excluded_local_indices)] = -np.inf

    top_indices = np.argpartition(masked_scores, -k)[-k:]
    top_indices = top_indices[np.argsort(masked_scores[top_indices])[::-1]]
    return [int(idx) for idx in top_indices if np.isfinite(masked_scores[idx])]


def _select_top_ranked_negatives(
    ranked_local_indices: List[int],
    all_available_local_indices: List[int],
    num_negatives: int,
    rng: random.Random,
) -> List[int]:
    """Select the top-ranked negatives directly and backfill with easy negatives if needed."""
    selected = list(ranked_local_indices[:num_negatives])
    remaining = [idx for idx in all_available_local_indices if idx not in set(selected)]
    needed = num_negatives - len(selected)
    if len(remaining) < needed:
        return []

    if needed > 0:
        selected.extend(rng.sample(remaining, needed))
    return selected


def _sample_topk_percpos_negatives(
    ranked_local_indices: List[int],
    all_available_local_indices: List[int],
    num_negatives: int,
    rng: random.Random,
) -> List[int]:
    """Sample topk-percpos negatives with a fixed hardest example."""
    if num_negatives <= 0:
        return []

    hard_candidates = ranked_local_indices[: 2 * num_negatives]
    if not hard_candidates:
        if len(all_available_local_indices) < num_negatives:
            return []
        return rng.sample(all_available_local_indices, num_negatives)

    selected = [hard_candidates[0]]
    remaining_hard = hard_candidates[1:]
    remaining_needed = num_negatives - len(selected)

    if remaining_needed > 0 and remaining_hard:
        sample_count = min(remaining_needed, len(remaining_hard))
        if sample_count == len(remaining_hard):
            selected.extend(remaining_hard)
        else:
            selected.extend(rng.sample(remaining_hard, sample_count))

    remaining_needed = num_negatives - len(selected)
    if remaining_needed <= 0:
        return selected

    selected_set = set(selected)
    remaining_easy = [idx for idx in all_available_local_indices if idx not in selected_set]
    if len(remaining_easy) < remaining_needed:
        return []

    selected.extend(rng.sample(remaining_easy, remaining_needed))
    return selected


def _compute_train_eval_cache_key(
    config: LabelTrainingConfig,
    effective_train_ratio: float,
    effective_eval_ratio: float = 0.5,
    effective_val_ratio: float = 0.1,
) -> str:
    """
    Compute a deterministic cache key from parameters that affect data preparation.

    The key covers: model names, task names, train_ratio, val_ratio, eval_ratio,
    seed, num_negatives, negative_strategy, eval_mode, and eval_task_names.
    """
    key_parts = {
        "large_model": config.large_model_name,
        "small_model": config.small_model_name,
        "task_names": sorted(config.task_names),
        "effective_train_ratio": effective_train_ratio,
        "effective_val_ratio": effective_val_ratio,
        "effective_eval_ratio": effective_eval_ratio,
        "seed": config.seed,
        "num_negatives": config.num_negatives,
        "negative_strategy": config.negative_strategy,
        "hard_negative_top_k": config.hard_negative_top_k,
        "hard_negative_perc_margin": config.hard_negative_perc_margin,
        "pretrained_adapter": _format_pretrained_adapter_identity(config.pretrained_adapter_path),
        "eval_mode": config.eval_mode,
        "eval_task_names": sorted(config.eval_task_names) if config.eval_task_names else None,
    }
    key_json = json.dumps(key_parts, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(key_json.encode()).hexdigest()[:16]


def _get_train_eval_cache_dir(config: LabelTrainingConfig) -> Path:
    """Return the base directory for train_eval caches."""
    # Place alongside the embedding cache directory
    cache_base = Path(config.cache_dir).parent  # e.g. cache/embeddings -> cache/
    return cache_base / "train_eval"


def _save_train_eval_cache(
    cache_dir: Path,
    cache_key: str,
    query_embeddings: np.ndarray,
    doc_embeddings: np.ndarray,
    all_pairs: List[QueryDocPair],
    eval_query_ids_per_task: Dict[str, List[str]],
    config: LabelTrainingConfig,
    effective_train_ratio: float,
    effective_eval_ratio: float = 0.5,
    effective_val_ratio: float = 0.1,
    val_pairs: Optional[List[QueryDocPair]] = None,
    val_query_ids_per_task: Optional[Dict[str, List[str]]] = None,
) -> Path:
    """
    Save prepared training data to disk so it can be reused.

    Saves:
      - query_embeddings.npy
      - doc_embeddings.npy
      - training_pairs.json  (list of {query_idx, positive_idx, negative_indices})
      - eval_query_ids.json  (task_name -> list of eval query IDs)
      - val_pairs.json       (validation pairs, optional)
      - val_query_ids.json   (task_name -> list of val query IDs, optional)
      - cache_meta.json      (config params for human inspection)
    """
    dest = cache_dir / cache_key
    dest.mkdir(parents=True, exist_ok=True)

    np.save(dest / "query_embeddings.npy", query_embeddings)
    np.save(dest / "doc_embeddings.npy", doc_embeddings)

    pairs_data = [
        {
            "query_idx": p.query_idx,
            "positive_idx": p.positive_idx,
            "positive_score": p.positive_score,
            "negative_indices": p.negative_indices,
        }
        for p in all_pairs
    ]
    with (dest / "training_pairs.json").open("w") as f:
        json.dump(pairs_data, f)

    with (dest / "eval_query_ids.json").open("w") as f:
        json.dump(eval_query_ids_per_task, f, ensure_ascii=False)

    # Save validation data if present
    if val_pairs is not None:
        val_pairs_data = [
            {
                "query_idx": p.query_idx,
                "positive_idx": p.positive_idx,
                "positive_score": p.positive_score,
                "negative_indices": p.negative_indices,
            }
            for p in val_pairs
        ]
        with (dest / "val_pairs.json").open("w") as f:
            json.dump(val_pairs_data, f)

    if val_query_ids_per_task is not None:
        with (dest / "val_query_ids.json").open("w") as f:
            json.dump(val_query_ids_per_task, f, ensure_ascii=False)

    meta = {
        "cache_key": cache_key,
        "large_model": config.large_model_name,
        "small_model": config.small_model_name,
        "task_names": sorted(config.task_names),
        "effective_train_ratio": effective_train_ratio,
        "effective_val_ratio": effective_val_ratio,
        "effective_eval_ratio": effective_eval_ratio,
        "seed": config.seed,
        "num_negatives": config.num_negatives,
        "negative_strategy": config.negative_strategy,
        "hard_negative_top_k": config.hard_negative_top_k,
        "hard_negative_perc_margin": config.hard_negative_perc_margin,
        "pretrained_adapter": _format_pretrained_adapter_identity(config.pretrained_adapter_path),
        "eval_mode": config.eval_mode,
        "eval_task_names": sorted(config.eval_task_names) if config.eval_task_names else None,
        "query_embeddings_shape": list(query_embeddings.shape),
        "doc_embeddings_shape": list(doc_embeddings.shape),
        "num_pairs": len(all_pairs),
        "num_val_pairs": len(val_pairs) if val_pairs else 0,
        "num_eval_tasks": len(eval_query_ids_per_task),
    }
    with (dest / "cache_meta.json").open("w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"  Saved train_eval cache to: {dest}")
    return dest


def _load_train_eval_cache(
    cache_dir: Path,
    cache_key: str,
) -> Optional[Dict[str, Any]]:
    """
    Load cached training data if available.

    Returns:
        Dict with query_embeddings, doc_embeddings, all_pairs,
        eval_query_ids_per_task, or None if cache miss.
    """
    dest = cache_dir / cache_key
    required_files = [
        "query_embeddings.npy",
        "doc_embeddings.npy",
        "training_pairs.json",
        "eval_query_ids.json",
        "cache_meta.json",
    ]
    if not all((dest / f).exists() for f in required_files):
        return None

    try:
        query_embeddings = np.load(dest / "query_embeddings.npy")
        doc_embeddings = np.load(dest / "doc_embeddings.npy")

        with (dest / "training_pairs.json").open() as f:
            pairs_data = json.load(f)
        all_pairs = [
            QueryDocPair(
                query_idx=p["query_idx"],
                positive_idx=p["positive_idx"],
                positive_score=p.get("positive_score", 1.0),
                negative_indices=p["negative_indices"],
            )
            for p in pairs_data
        ]

        with (dest / "eval_query_ids.json").open() as f:
            eval_query_ids_per_task = json.load(f)

        with (dest / "cache_meta.json").open() as f:
            meta = json.load(f)

        # Load validation data if present
        val_pairs = None
        val_query_ids_per_task = None

        if (dest / "val_pairs.json").exists():
            with (dest / "val_pairs.json").open() as f:
                val_pairs_data = json.load(f)
            val_pairs = [
                QueryDocPair(
                    query_idx=p["query_idx"],
                    positive_idx=p["positive_idx"],
                    positive_score=p.get("positive_score", 1.0),
                    negative_indices=p["negative_indices"],
                )
                for p in val_pairs_data
            ]

        if (dest / "val_query_ids.json").exists():
            with (dest / "val_query_ids.json").open() as f:
                val_query_ids_per_task = json.load(f)

        print(f"  Loaded train_eval cache from: {dest}")
        print(f"    Query embeddings: {query_embeddings.shape}")
        print(f"    Doc embeddings:   {doc_embeddings.shape}")
        print(f"    Training pairs:   {len(all_pairs)}")
        if val_pairs is not None:
            print(f"    Validation pairs: {len(val_pairs)}")
        print(f"    Eval tasks:       {len(eval_query_ids_per_task)}")
        return {
            "query_embeddings": query_embeddings,
            "doc_embeddings": doc_embeddings,
            "all_pairs": all_pairs,
            "eval_query_ids_per_task": eval_query_ids_per_task,
            "val_pairs": val_pairs,
            "val_query_ids_per_task": val_query_ids_per_task,
            "meta": meta,
        }
    except Exception as e:
        print(f"  Warning: Failed to load train_eval cache: {e}")
        return None


def _format_float(value: float) -> str:
    """Format float for filename."""
    formatted = f"{value:.10f}"
    formatted = formatted.rstrip("0").rstrip(".")
    return "0" if formatted in {"", "-0"} else formatted


def _build_label_result_tag(
    negative_strategy: str,
    num_negatives: int,
    lr: float,
    weight_decay: float,
    hard_negative_top_k: int,
    hard_negative_perc_margin: float,
) -> Optional[str]:
    """Build a compact directory tag for label-training settings.

    The historical default configuration (contrastive / random / 5 negatives)
    keeps the legacy directory layout so existing results remain addressable.
    """
    if (
        negative_strategy == "random"
        and num_negatives == 5
        and abs(lr - 1e-3) <= 1e-12
        and abs(weight_decay - 0.01) <= 1e-12
    ):
        return None
    tag = (
        f"losscontrastive__neg{negative_strategy}__nneg{num_negatives}"
        f"__wd{_format_float(weight_decay)}__lr{_format_float(lr)}"
    )
    if negative_strategy == "naive_topk":
        tag += f"__topk{hard_negative_top_k}"
    elif negative_strategy == "topk_percpos":
        tag += f"__topk{hard_negative_top_k}__pm{_format_float(hard_negative_perc_margin)}"
    return tag


def _build_adapter_filename(config: LabelTrainingConfig, epoch_suffix: str = "") -> str:
    """Build adapter filename with hyperparameter info."""
    parts = [
        "adapter",
        "losscontrastive",
        f"neg{config.negative_strategy}",
        f"nneg{config.num_negatives}",
        f"train{_format_float(config.train_ratio)}",
    ]
    # Include val_ratio in filename only when it differs from default (0.1)
    if abs(config.val_ratio - 0.1) > 1e-9:
        parts.append(f"val{_format_float(config.val_ratio)}")
    # Include eval_ratio in filename only when it differs from default (0.5)
    if abs(config.eval_ratio - 0.5) > 1e-9:
        parts.append(f"eval{_format_float(config.eval_ratio)}")
    parts += [
        f"wd{_format_float(config.weight_decay)}",
        f"lr{_format_float(config.lr)}",
        f"epochs{config.epochs}",
        f"bs{config.batch_size}",
        f"seed{config.seed}",
    ]
    if config.negative_strategy == "naive_topk":
        parts.append(f"topk{config.hard_negative_top_k}")
    elif config.negative_strategy == "topk_percpos":
        parts.append(f"topk{config.hard_negative_top_k}")
        parts.append(f"pm{_format_float(config.hard_negative_perc_margin)}")
    parts.append(f"temp{_format_float(config.temperature)}")
    return "__".join(parts) + epoch_suffix + ".pt"


def split_queries_by_id(
    qrels: Dict[str, Dict[str, int]],
    train_ratio: float,
    seed: int,
    eval_ratio: float = 0.5,
    val_ratio: float = 0.0,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Split query IDs into train, validation, and eval (test) sets.

    Layout after shuffling (indices as fractions of N):

        [0 .. train_size)  → Train
        [val_start .. eval_start)  → Validation
        [eval_start .. N)  → Eval / Test

    The **eval** set is always fixed as the last ``eval_ratio`` fraction.
    The **validation** set is the ``val_ratio`` fraction immediately before
    eval.  The **train** set is taken from the beginning and its size is
    controlled independently by ``train_ratio``.

    This design keeps the eval (and val) sets constant while varying the
    amount of training data (e.g. train_ratio = 0.1 … 0.4 with
    val_ratio = 0.1, eval_ratio = 0.5).

    Args:
        qrels: Query relevance judgments {qid: {doc_id: score}}
        train_ratio: Fraction of **total** queries to use for training.
                     Must satisfy ``train_ratio + val_ratio + eval_ratio <= 1``.
        seed: Random seed
        eval_ratio: Fraction of total queries reserved for evaluation/test.
                    The eval set is always the last ``eval_ratio`` portion.
        val_ratio: Fraction of total queries reserved for validation.
                   Placed immediately before the eval set.  Set to 0 to
                   disable validation (returns empty list for val_qids).

    Returns:
        Tuple of (train_qids, val_qids, eval_qids)

    Raises:
        ValueError: If ``train_ratio + val_ratio + eval_ratio > 1``.
    """
    if train_ratio + val_ratio + eval_ratio > 1 + 1e-9:
        raise ValueError(
            f"train_ratio ({train_ratio}) + val_ratio ({val_ratio}) + "
            f"eval_ratio ({eval_ratio}) = "
            f"{train_ratio + val_ratio + eval_ratio} > 1."
        )

    all_qids = list(qrels.keys())
    rng = random.Random(seed)
    rng.shuffle(all_qids)

    N = len(all_qids)

    # Eval set: always the last eval_ratio fraction
    eval_start = max(1, int(N * (1 - eval_ratio)))
    eval_qids = all_qids[eval_start:]

    # Validation set: immediately before eval
    val_start = max(1, int(N * (1 - eval_ratio - val_ratio))) if val_ratio > 0 else eval_start
    val_start = min(val_start, eval_start)  # safety: val cannot overlap eval
    val_qids = all_qids[val_start:eval_start]

    # Train set: first train_ratio fraction (must not overlap val/eval)
    train_size = max(1, min(int(N * train_ratio), val_start))
    train_qids = all_qids[:train_size]

    return train_qids, val_qids, eval_qids


def create_training_pairs(
    train_qids: List[str],
    qrels: Dict[str, Dict[str, int]],
    qid_to_idx: Dict[str, int],
    docid_to_idx: Dict[str, int],
    num_negatives: int,
    negative_strategy: str,
    seed: int,
    query_offset: int = 0,
    doc_offset: int = 0,
    teacher_query_embeddings: Optional[np.ndarray] = None,
    teacher_doc_embeddings: Optional[np.ndarray] = None,
    hard_negative_top_k: int = 2000,
    hard_negative_perc_margin: float = 0.95,
) -> List[QueryDocPair]:
    """
    Create query-document training pairs with negative sampling.
    
    Args:
        train_qids: List of training query IDs
        qrels: Query relevance judgments
        qid_to_idx: Mapping from query ID to embedding index
        docid_to_idx: Mapping from document ID to embedding index
        num_negatives: Number of negative samples per positive
        negative_strategy: "random", "naive_topk", or "topk_percpos"
        seed: Random seed
        query_offset: Global query index offset for concatenated embeddings
        doc_offset: Global document index offset for concatenated embeddings
        teacher_query_embeddings: Query embeddings used for hard-negative retrieval
        teacher_doc_embeddings: Document embeddings used for hard-negative retrieval
        hard_negative_top_k: Initial top-k retrieval size for hard negatives
        hard_negative_perc_margin: Maximum negative/positive score ratio for topk-percpos
    
    Returns:
        List of QueryDocPair objects
    """
    rng = random.Random(seed)
    all_doc_indices = list(docid_to_idx.values())
    pairs = []

    normalized_teacher_queries = None
    normalized_teacher_docs = None
    if negative_strategy in HARD_NEGATIVE_STRATEGIES:
        if teacher_query_embeddings is None or teacher_doc_embeddings is None:
            raise ValueError(
                f"Negative strategy '{negative_strategy}' requires teacher query/doc embeddings."
            )
        normalized_teacher_queries = _normalize_embeddings(teacher_query_embeddings)
        normalized_teacher_docs = _normalize_embeddings(teacher_doc_embeddings)
    
    for qid in train_qids:
        if qid not in qid_to_idx:
            continue
        
        local_query_idx = qid_to_idx[qid]
        query_idx = local_query_idx + query_offset
        rel_docs = qrels.get(qid, {})
        
        # Get positive documents (score > 0) that exist in corpus
        positive_doc_ids = [
            doc_id for doc_id, score in rel_docs.items()
            if score > 0 and doc_id in docid_to_idx
        ]
        
        if not positive_doc_ids:
            continue
        
        # Select one positive per query (randomly)
        pos_doc_id = rng.choice(positive_doc_ids)
        local_positive_idx = docid_to_idx[pos_doc_id]
        positive_idx = local_positive_idx + doc_offset
        positive_score = float(rel_docs[pos_doc_id])
        
        if negative_strategy == "random":
            # Random negative sampling
            exclude_set = set(docid_to_idx.get(d, -1) for d in rel_docs.keys())
            exclude_set.add(local_positive_idx)
            
            available_negatives = [i for i in all_doc_indices if i not in exclude_set]
            
            if len(available_negatives) < num_negatives:
                # Not enough negatives, skip this pair
                continue
            
            negative_indices = [idx + doc_offset for idx in rng.sample(available_negatives, num_negatives)]
        elif negative_strategy in HARD_NEGATIVE_STRATEGIES:
            teacher_queries = cast(np.ndarray, normalized_teacher_queries)
            teacher_docs = cast(np.ndarray, normalized_teacher_docs)
            exclude_set = {docid_to_idx[d] for d in rel_docs.keys() if d in docid_to_idx}
            exclude_set.add(local_positive_idx)
            available_negatives = [i for i in all_doc_indices if i not in exclude_set]

            if len(available_negatives) < num_negatives:
                continue

            similarity_scores = teacher_queries[local_query_idx] @ teacher_docs.T
            ranked_candidates = _rank_topk_candidate_indices(
                similarity_scores,
                exclude_set,
                hard_negative_top_k,
            )

            if negative_strategy == "topk_percpos":
                positive_anchor_score = float(similarity_scores[local_positive_idx])
                max_negative_score = positive_anchor_score * hard_negative_perc_margin
                filtered_candidates = []
                candidate_cap = 2 * num_negatives
                for candidate_idx in ranked_candidates:
                    if float(similarity_scores[candidate_idx]) < max_negative_score:
                        filtered_candidates.append(candidate_idx)
                    if len(filtered_candidates) >= candidate_cap:
                        break
                ranked_candidates = filtered_candidates

            if negative_strategy == "topk_percpos":
                sampled_negatives = _sample_topk_percpos_negatives(
                    ranked_local_indices=ranked_candidates,
                    all_available_local_indices=available_negatives,
                    num_negatives=num_negatives,
                    rng=rng,
                )
            else:
                sampled_negatives = _select_top_ranked_negatives(
                    ranked_local_indices=ranked_candidates,
                    all_available_local_indices=available_negatives,
                    num_negatives=num_negatives,
                    rng=rng,
                )
            if len(sampled_negatives) < num_negatives:
                continue

            negative_indices = [idx + doc_offset for idx in sampled_negatives]
        
        pairs.append(QueryDocPair(
            query_idx=query_idx,
            positive_idx=positive_idx,
            positive_score=positive_score,
            negative_indices=negative_indices,
        ))
    
    rng.shuffle(pairs)
    return pairs


def contrastive_loss(
    query_emb: torch.Tensor,
    positive_emb: torch.Tensor,
    negative_embs: torch.Tensor,
    temperature: float = 0.05,
) -> torch.Tensor:
    """
    Contrastive loss (InfoNCE) with in-batch negatives support.
    
    Args:
        query_emb: Query embeddings (B, D) - already passed through adapter
        positive_emb: Positive document embeddings (B, D)
        negative_embs: Negative document embeddings (B, N, D)
        temperature: Temperature scaling factor
    
    Returns:
        Scalar loss value
    """
    batch_size = query_emb.shape[0]
    
    # Compute positive similarities: (B,)
    pos_sim = torch.sum(query_emb * positive_emb, dim=-1) / temperature
    
    # Compute negative similarities: (B, N)
    neg_sim = torch.bmm(negative_embs, query_emb.unsqueeze(-1)).squeeze(-1) / temperature
    
    # InfoNCE loss: -log(exp(pos) / (exp(pos) + sum(exp(neg))))
    logits = torch.cat([pos_sim.unsqueeze(-1), neg_sim], dim=-1)  # (B, 1 + N)
    labels = torch.zeros(batch_size, dtype=torch.long, device=query_emb.device)
    
    loss = nn.functional.cross_entropy(logits, labels)
    return loss


def load_task_embeddings(
    task_name: str,
    config: LabelTrainingConfig,
) -> Optional[Dict[str, Any]]:
    """
    Load cached embeddings for a MAIR task.
    
    Handles both single-instruction and per-query-instruction tasks.
    For per-query instruction tasks (e.g., LeetCode, SWE-Bench-Lite),
    loads embeddings group by group from per-instruction caches and
    reassembles them in original query order.
    
    Returns:
        Dictionary with query/doc embeddings and mappings, or None if not cached.
    """
    from collections import defaultdict
    
    cache = EmbeddingCache(cache_dir=config.cache_dir, enabled=True, force_recache=False)
    
    # Load dataset to get texts
    dataset = MAIRDataset(task_name, cache_dir=config.dataset_cache_dir)
    dataset.load()
    
    # Prepare query texts and mappings
    query_ids = list(dataset.queries.keys())
    query_texts = [dataset.queries[qid] for qid in query_ids]
    qid_to_idx = {qid: idx for idx, qid in enumerate(query_ids)}
    
    # Prepare corpus texts and mappings
    doc_ids = list(dataset.corpus.keys())
    doc_texts = [dataset.corpus[doc_id] for doc_id in doc_ids]
    docid_to_idx = {doc_id: idx for idx, doc_id in enumerate(doc_ids)}
    
    mair_task_name = f"MAIR_{task_name}"
    
    # Determine instruction mode
    instructions = dataset.instructions if dataset.instructions else {}
    unique_instructions = set(instructions.values()) if instructions else set()
    has_per_query_instructions = len(unique_instructions) > 1
    default_instruction = dataset.get_default_instruction()
    
    query_embeddings_large = _load_query_embeddings_for_model(
        cache=cache,
        model_name=config.large_model_name,
        task_name=task_name,
        mair_task_name=mair_task_name,
        query_ids=query_ids,
        query_texts=query_texts,
        instructions=instructions,
        has_per_query_instructions=has_per_query_instructions,
        default_instruction=default_instruction,
        strict_instruction_cache=True,
    )
    
    if query_embeddings_large is None:
        print(f"  Warning: No query embeddings found for {task_name} (large model)")
        return None

    query_embeddings_small = None
    needs_small_query_teacher = (
        config.negative_strategy in HARD_NEGATIVE_STRATEGIES
        and config.pretrained_adapter_path is None
        and config.large_model_name != config.small_model_name
    )
    if needs_small_query_teacher:
        query_embeddings_small = _load_query_embeddings_for_model(
            cache=cache,
            model_name=config.small_model_name,
            task_name=task_name,
            mair_task_name=mair_task_name,
            query_ids=query_ids,
            query_texts=query_texts,
            instructions=instructions,
            has_per_query_instructions=has_per_query_instructions,
            default_instruction=default_instruction,
            strict_instruction_cache=True,
        )

        if query_embeddings_small is None:
            raise RuntimeError(
                f"Hard-negative mining for task '{task_name}' requires small-model query embeddings "
                f"when no pretrained adapter is provided. Missing query cache for model '{config.small_model_name}'."
            )
    
    # Load corpus embeddings (small model)
    doc_embeddings_small = cache.get(
        config.small_model_name,
        doc_texts,
        task_name=mair_task_name,
        split="corpus",
    )
    
    if doc_embeddings_small is None:
        print(f"  Warning: No corpus embeddings found for {task_name} (small model)")
        return None
    
    return {
        "query_embeddings": query_embeddings_large,
        "query_embeddings_small": query_embeddings_small,
        "doc_embeddings": doc_embeddings_small,
        "qid_to_idx": qid_to_idx,
        "docid_to_idx": docid_to_idx,
        "qrels": dataset.qrels,
        "query_ids": query_ids,
        "doc_ids": doc_ids,
        "instruction": default_instruction,
        "num_queries": len(query_ids),
        "num_docs": len(doc_ids),
    }


def _load_perquery_instruction_embeddings(
    cache: EmbeddingCache,
    model_name: str,
    query_ids: List[str],
    query_texts: List[str],
    instructions: Dict[str, str],
    task_name: str,
) -> Optional[np.ndarray]:
    """
    Load embeddings for per-query instruction tasks by reconstructing
    from per-instruction-group caches.
    
    The mair_evaluator caches embeddings grouped by instruction, e.g.:
      queries_perquery_inst_{md5(instruction)[:8]}/
    
    This function groups queries by instruction, loads each group from
    cache, and reassembles them in original query order.
    
    Returns:
        Reconstructed embeddings array, or None if any group is missing.
    """
    from collections import defaultdict
    
    # Group query indices by instruction
    instruction_to_indices = defaultdict(list)
    for i, qid in enumerate(query_ids):
        inst = instructions.get(qid, "")
        instruction_to_indices[inst].append(i)
    
    n_groups = len(instruction_to_indices)
    print(f"  [Per-Query Instructions] {len(query_ids)} queries in {n_groups} instruction groups")
    
    # Try to load each group
    result_parts = {}  # inst -> (indices, embeddings)
    emb_dim = None
    
    for instruction, indices in instruction_to_indices.items():
        group_texts = [query_texts[i] for i in indices]
        inst_hash = hashlib.md5(instruction.encode('utf-8')).hexdigest()[:8]
        
        # Try perquery_inst first (original cache format), then with_instruction
        cached = None
        for split_prefix in [f"queries_perquery_inst_{inst_hash}", f"queries_with_instruction_{inst_hash}"]:
            cached = cache.get(
                model_name,
                group_texts,
                task_name=task_name,
                split=split_prefix,
            )
            if cached is not None:
                break
        
        if cached is None:
            print(f"  [Per-Query Instructions] Cache miss for instruction group (hash={inst_hash}, {len(indices)} queries)")
            return None
        
        if emb_dim is None:
            emb_dim = cached.shape[1]
        
        result_parts[instruction] = (indices, cached)
    
    # Reassemble in original order
    if emb_dim is None:
        return None

    result = np.zeros((len(query_ids), emb_dim), dtype=np.float32)
    for instruction, (indices, embeddings) in result_parts.items():
        for j, idx in enumerate(indices):
            result[idx] = embeddings[j]
    
    print(f"  [Per-Query Instructions] Loaded all {n_groups} groups successfully ({len(query_ids)} queries)")
    return result


def _load_query_embeddings_for_model(
    cache: EmbeddingCache,
    model_name: str,
    task_name: str,
    mair_task_name: str,
    query_ids: List[str],
    query_texts: List[str],
    instructions: Dict[str, str],
    has_per_query_instructions: bool,
    default_instruction: Optional[str],
    strict_instruction_cache: bool,
) -> Optional[np.ndarray]:
    """Load query embeddings for a specific model, preserving instruction semantics."""
    query_embeddings = None

    if has_per_query_instructions:
        query_embeddings = _load_perquery_instruction_embeddings(
            cache=cache,
            model_name=model_name,
            query_ids=query_ids,
            query_texts=query_texts,
            instructions=instructions,
            task_name=mair_task_name,
        )
    else:
        if default_instruction:
            inst_hash = hashlib.md5(default_instruction.encode("utf-8")).hexdigest()[:8]
            for split_prefix in [f"queries_with_instruction_{inst_hash}", f"queries_perquery_inst_{inst_hash}"]:
                query_embeddings = cache.get(
                    model_name,
                    query_texts,
                    task_name=mair_task_name,
                    split=split_prefix,
                )
                if query_embeddings is not None:
                    break

    if query_embeddings is None:
        if strict_instruction_cache and (default_instruction or has_per_query_instructions):
            raise RuntimeError(
                f"No instruction-aware query embeddings found for task '{task_name}'. "
                f"Expected cache in queries_with_instruction_* or queries_perquery_inst_* "
                f"for model '{model_name}'. "
                f"Run the benchmark with instructions enabled first to populate the cache."
            )
        query_embeddings = cache.get(
            model_name,
            query_texts,
            task_name=mair_task_name,
            split="queries",
        )

    return query_embeddings


def train_with_labels(config: LabelTrainingConfig) -> Dict[str, Any]:
    """
    Train adapter using labeled query-document pairs.
    
    Args:
        config: Training configuration
    
    Returns:
        Dictionary with training results and paths
    """
    start_time = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("=" * 60)
    print("Label-based Adapter Training")
    print("=" * 60)
    print(f"Large model: {config.large_model_name}")
    print(f"Small model: {config.small_model_name}")
    print(f"Training tasks: {config.task_names}")
    print(f"Eval mode: {config.eval_mode}")
    if config.eval_mode == "out_of_domain" and config.eval_task_names:
        print(f"Eval tasks (held out): {config.eval_task_names}")
    # Use the same train/val/eval split regardless of eval_mode.
    # In OOD mode, training tasks still reserve eval queries so that
    # in-domain and out-of-domain results are directly comparable.
    effective_train_ratio = config.train_ratio
    effective_val_ratio = config.val_ratio
    effective_eval_ratio = config.eval_ratio
    print(f"Train ratio: {effective_train_ratio}")
    print(f"Val ratio: {effective_val_ratio}")
    print(f"Eval ratio: {effective_eval_ratio}")
    print(f"Early stopping: {config.early_stopping} (patience={config.early_stopping_patience})")
    print(f"Loss type: contrastive")
    print(f"Negative strategy: {config.negative_strategy}")
    print(f"Num negatives: {config.num_negatives}")
    if config.negative_strategy in HARD_NEGATIVE_STRATEGIES:
        print(f"Hard-negative top-k: {config.hard_negative_top_k}")
        if config.negative_strategy == "naive_topk":
            print("Hard-negative selection: take the top num_negatives directly")
        if config.negative_strategy == "topk_percpos":
            print(f"Hard-negative sampling pool: top-{2 * config.num_negatives} after threshold")
            print(f"Hard-negative perc margin: {config.hard_negative_perc_margin}")
    print()
    
    # --- Train/Eval Data Caching ---
    # Compute cache key from data-preparation parameters
    cache_key = _compute_train_eval_cache_key(
        config, effective_train_ratio, effective_eval_ratio, effective_val_ratio
    )
    train_eval_cache_dir = _get_train_eval_cache_dir(config)
    print(f"Train/eval cache key: {cache_key}")
    
    cached = _load_train_eval_cache(train_eval_cache_dir, cache_key)
    
    # This will be populated from cache or built from scratch
    eval_query_ids_per_task: Dict[str, List[str]] = {}
    val_query_ids_per_task: Dict[str, List[str]] = {}
    val_pairs: List[QueryDocPair] = []
    
    if cached is not None:
        # Cache hit: skip expensive loading / pair creation
        query_embeddings = cached["query_embeddings"]
        doc_embeddings = cached["doc_embeddings"]
        all_pairs = cached["all_pairs"]
        eval_query_ids_per_task = cached["eval_query_ids_per_task"]
        val_pairs = cached.get("val_pairs") or []
        val_query_ids_per_task = cached.get("val_query_ids_per_task") or {}
        print(f"\n[Cache HIT] Reusing prepared training data.")
        print(f"Total training pairs: {len(all_pairs)}")
        if val_pairs:
            print(f"Total validation pairs: {len(val_pairs)}")
        print(f"Query embeddings shape: {query_embeddings.shape}")
        print(f"Doc embeddings shape: {doc_embeddings.shape}")
    else:
        print(f"[Cache MISS] Preparing training data from scratch...")
        # Load embeddings from all tasks
        all_query_embeddings = []
        all_doc_embeddings = []
        all_pairs = []
        val_pairs = []

        query_offset = 0
        doc_offset = 0
        mining_adapter = None
        if config.negative_strategy in HARD_NEGATIVE_STRATEGIES and config.pretrained_adapter_path:
            print(f"Loading teacher adapter for hard-negative mining: {config.pretrained_adapter_path}")
            mining_adapter = _load_mining_adapter(config.pretrained_adapter_path, device)

        for task_name in config.task_names:
            print(f"Loading task: {task_name}")
            task_data = load_task_embeddings(task_name, config)

            if task_data is None:
                print(f"  Skipping task {task_name} (missing embeddings)")
                continue

            # Split queries into train/val/eval based on eval_mode
            # For out_of_domain: use all queries for training (effective_train_ratio=1.0)
            # For in_domain: use train_ratio to split, val and eval sets fixed
            train_qids, val_qids, eval_qids = split_queries_by_id(
                task_data["qrels"],
                effective_train_ratio,
                config.seed,
                eval_ratio=effective_eval_ratio,
                val_ratio=effective_val_ratio,
            )

            eval_query_ids_per_task[task_name] = eval_qids
            val_query_ids_per_task[task_name] = val_qids
            print(f"  Train queries: {len(train_qids)}, Val queries: {len(val_qids)}, Eval queries: {len(eval_qids)}")

            teacher_query_embeddings = None
            if config.negative_strategy in HARD_NEGATIVE_STRATEGIES:
                fallback_teacher_query_embeddings = task_data.get("query_embeddings_small")
                if fallback_teacher_query_embeddings is not None and config.pretrained_adapter_path is None:
                    print(
                        f"  Hard-negative teacher: using small-model query cache "
                        f"({config.small_model_name}) against small-model corpus embeddings"
                    )
                teacher_query_embeddings = _prepare_teacher_query_embeddings(
                    task_data["query_embeddings"],
                    task_data["doc_embeddings"],
                    config.negative_strategy,
                    mining_adapter,
                    device,
                    config.pretrained_adapter_path,
                    fallback_query_embeddings=fallback_teacher_query_embeddings,
                )

            # Create training pairs for this task
            task_pairs = create_training_pairs(
                train_qids=train_qids,
                qrels=task_data["qrels"],
                qid_to_idx=task_data["qid_to_idx"],
                docid_to_idx=task_data["docid_to_idx"],
                num_negatives=config.num_negatives,
                negative_strategy=config.negative_strategy,
                seed=config.seed,
                query_offset=query_offset,
                doc_offset=doc_offset,
                teacher_query_embeddings=teacher_query_embeddings,
                teacher_doc_embeddings=task_data["doc_embeddings"],
                hard_negative_top_k=config.hard_negative_top_k,
                hard_negative_perc_margin=config.hard_negative_perc_margin,
            )
            print(f"  Created {len(task_pairs)} training pairs")

            # Create validation pairs (using same negative strategy)
            if val_qids:
                task_val_pairs = create_training_pairs(
                    train_qids=val_qids,
                    qrels=task_data["qrels"],
                    qid_to_idx=task_data["qid_to_idx"],
                    docid_to_idx=task_data["docid_to_idx"],
                    num_negatives=config.num_negatives,
                    negative_strategy=config.negative_strategy,
                    seed=config.seed + 1,  # Different seed for val negatives
                    query_offset=query_offset,
                    doc_offset=doc_offset,
                    teacher_query_embeddings=teacher_query_embeddings,
                    teacher_doc_embeddings=task_data["doc_embeddings"],
                    hard_negative_top_k=config.hard_negative_top_k,
                    hard_negative_perc_margin=config.hard_negative_perc_margin,
                )
                print(f"  Created {len(task_val_pairs)} validation pairs")
                val_pairs.extend(task_val_pairs)

            all_query_embeddings.append(task_data["query_embeddings"])
            all_doc_embeddings.append(task_data["doc_embeddings"])
            all_pairs.extend(task_pairs)

            query_offset += task_data["num_queries"]
            doc_offset += task_data["num_docs"]

        if not all_query_embeddings or not all_pairs:
            raise RuntimeError("No training data available. Check embedding caches.")

        # Concatenate embeddings
        query_embeddings = np.concatenate(all_query_embeddings, axis=0)
        doc_embeddings = np.concatenate(all_doc_embeddings, axis=0)

        print(f"\nTotal training pairs: {len(all_pairs)}")
        if val_pairs:
            print(f"Total validation pairs: {len(val_pairs)}")
        print(f"Query embeddings shape: {query_embeddings.shape}")
        print(f"Doc embeddings shape: {doc_embeddings.shape}")

        # Save to cache for future reuse
        try:
            _save_train_eval_cache(
                cache_dir=train_eval_cache_dir,
                cache_key=cache_key,
                query_embeddings=query_embeddings,
                doc_embeddings=doc_embeddings,
                all_pairs=all_pairs,
                eval_query_ids_per_task=eval_query_ids_per_task,
                config=config,
                effective_train_ratio=effective_train_ratio,
                effective_eval_ratio=effective_eval_ratio,
                effective_val_ratio=effective_val_ratio,
                val_pairs=val_pairs if val_pairs else None,
                val_query_ids_per_task=val_query_ids_per_task if val_query_ids_per_task else None,
            )
        except Exception as e:
            print(f"  Warning: Failed to save train_eval cache: {e}")
    
    in_dim = query_embeddings.shape[1]
    out_dim = doc_embeddings.shape[1]
    
    # Create adapter
    adapter = LinearAdapter(in_dim, out_dim).to(device)
    print(f"Using Linear adapter: {in_dim} -> {out_dim}")
    
    # Load pretrained weights if available
    if config.pretrained_adapter_path:
        print(f"Loading pretrained adapter from: {config.pretrained_adapter_path}")
        checkpoint = torch.load(config.pretrained_adapter_path, map_location=device)
        adapter.load_state_dict(checkpoint)
    
    # Create dataset and dataloader
    if config.num_negatives < 1:
        raise ValueError(
            "label_num_negatives must be >= 1 for contrastive loss."
        )

    dataset = QueryDocDataset(query_embeddings, doc_embeddings, all_pairs)
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True if device == "cuda" else False,
    )
    
    # Create validation dataloader (if validation pairs exist)
    val_dataloader = None
    val_dataset = None
    if val_pairs:
        val_dataset = QueryDocDataset(query_embeddings, doc_embeddings, val_pairs)
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True if device == "cuda" else False,
        )
        print(f"Validation dataloader: {len(val_pairs)} pairs, {len(val_dataloader)} batches")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        adapter.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    
    # Learning rate scheduler with warmup
    total_steps = len(dataloader) * config.epochs
    warmup_steps = int(total_steps * config.warmup_ratio)
    
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Select loss function
    def loss_fn(q, p, n, s):
        return contrastive_loss(q, p, n, config.temperature)
    print(f"Using Contrastive Loss (temperature={config.temperature})")
    
    # Output directory
    if config.flat_output:
        # Called from unified training: output_dir is already the final run directory
        run_dir = Path(config.output_dir)
    else:
        # Standalone mode: build full hierarchy
        output_base = Path(config.output_dir)
        run_name = f"{_safe_name(config.large_model_name)}__to__{_safe_name(config.small_model_name)}"
        task_str = "_".join(config.task_names) if len(config.task_names) <= 3 else f"multi_{len(config.task_names)}_tasks"
        ratio_tag = f"train{_format_float(config.train_ratio)}"
        if abs(config.val_ratio - 0.1) > 1e-9:
            ratio_tag += f"_val{_format_float(config.val_ratio)}"
        if abs(config.eval_ratio - 0.5) > 1e-9:
            ratio_tag += f"_eval{_format_float(config.eval_ratio)}"
        label_tag = _build_label_result_tag(
            config.negative_strategy,
            config.num_negatives,
            config.lr,
            config.weight_decay,
            config.hard_negative_top_k,
            config.hard_negative_perc_margin,
        )
        run_dir_name = f"label__{task_str}__{ratio_tag}"
        if label_tag is not None:
            run_dir_name = f"label__{label_tag}__{task_str}__{ratio_tag}"
        run_dir = output_base / run_name / run_dir_name
    run_dir.mkdir(parents=True, exist_ok=True)
    
    history = []
    checkpoint_paths = []
    
    # Early stopping state
    best_val_loss = float("inf")
    best_epoch = 0
    best_state_dict = None
    patience_counter = 0
    stopped_early = False
    use_early_stopping = config.early_stopping and val_dataloader is not None
    
    if config.early_stopping and val_dataloader is None:
        print("Warning: early_stopping=True but no validation data available. "
              "Early stopping will be disabled for this run.")
    
    # Training loop
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)

    completed_epoch = 0
    
    for epoch in range(1, config.epochs + 1):
        completed_epoch = epoch
        adapter.train()
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_idx, (query_emb, positive_emb, negative_embs, positive_scores) in enumerate(dataloader):
            query_emb = query_emb.to(device)
            positive_emb = positive_emb.to(device)
            negative_embs = negative_embs.to(device)
            positive_scores = positive_scores.to(device)
            
            # Forward pass through adapter
            adapted_query = adapter(query_emb)
            
            # Compute loss
            loss = loss_fn(adapted_query, positive_emb, negative_embs, positive_scores)
            
            # Backward pass
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        current_lr = scheduler.get_last_lr()[0]
        
        # Compute validation loss
        val_loss = None
        if val_dataloader is not None:
            adapter.eval()
            val_epoch_loss = 0.0
            val_num_batches = 0
            with torch.no_grad():
                for query_emb, positive_emb, negative_embs, positive_scores in val_dataloader:
                    query_emb = query_emb.to(device)
                    positive_emb = positive_emb.to(device)
                    negative_embs = negative_embs.to(device)
                    positive_scores = positive_scores.to(device)
                    adapted_query = adapter(query_emb)
                    v_loss = loss_fn(adapted_query, positive_emb, negative_embs, positive_scores)
                    val_epoch_loss += v_loss.item()
                    val_num_batches += 1
            val_loss = val_epoch_loss / max(1, val_num_batches)
        
        epoch_record = {
            "epoch": epoch,
            "train_loss": avg_loss,
            "lr": current_lr,
        }
        if val_loss is not None:
            epoch_record["val_loss"] = val_loss
        history.append(epoch_record)
        
        # Log
        log_msg = f"Epoch {epoch}/{config.epochs} - Loss: {avg_loss:.6f}"
        if val_loss is not None:
            log_msg += f" - Val Loss: {val_loss:.6f}"
        log_msg += f" - LR: {current_lr:.2e}"
        print(log_msg)
        
        # Early stopping check
        if use_early_stopping and val_loss is not None:
            if val_loss < best_val_loss - config.early_stopping_min_delta:
                best_val_loss = val_loss
                best_epoch = epoch
                best_state_dict = {k: v.clone() for k, v in adapter.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= config.early_stopping_patience:
                    print(f"\n*** Early stopping triggered at epoch {epoch} ***")
                    print(f"    Best val loss: {best_val_loss:.6f} at epoch {best_epoch}")
                    stopped_early = True
        elif val_loss is not None:
            # Track best even without early stopping (for metadata)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
        
        # Save checkpoint
        if epoch % config.checkpoint_interval == 0 or epoch == config.epochs:
            checkpoint_path = run_dir / _build_adapter_filename(config, epoch_suffix=f"_epoch{epoch}")
            torch.save(adapter.state_dict(), checkpoint_path)
            checkpoint_paths.append((epoch, str(checkpoint_path)))
            print(f"  Saved checkpoint: {checkpoint_path.name}")
        
        if stopped_early:
            break
    
    # Restore best model if early stopping was used and we have a best state
    if use_early_stopping and best_state_dict is not None:
        adapter.load_state_dict(best_state_dict)
        print(f"\nRestored best model from epoch {best_epoch} (val_loss={best_val_loss:.6f})")
    
    # Save final adapter
    adapter.eval()
    final_path = run_dir / _build_adapter_filename(config)
    torch.save(adapter.state_dict(), final_path)
    
    # Save metadata
    meta = {
        "config": asdict(config),
        "in_dim": in_dim,
        "out_dim": out_dim,
        "num_pairs": len(all_pairs),
        "num_val_pairs": len(val_pairs),
        "num_queries": query_embeddings.shape[0],
        "num_docs": doc_embeddings.shape[0],
        "train_history": history,
        "checkpoint_paths": checkpoint_paths,
        "stopped_early": stopped_early,
        "best_epoch": best_epoch if best_epoch > 0 else None,
        "best_val_loss": best_val_loss if best_val_loss < float("inf") else None,
        "elapsed_sec": round(time.time() - start_time, 2),
    }
    
    meta_path = run_dir / "training_meta.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    
    print("\n" + "=" * 60)
    print("Training completed!")
    if stopped_early:
        print(f"Early stopped at epoch {completed_epoch}, best epoch: {best_epoch}")
    print(f"Adapter saved to: {final_path}")
    print(f"Metadata saved to: {meta_path}")
    print(f"Total time: {meta['elapsed_sec']:.2f}s")
    print("=" * 60)
    
    # Cleanup
    del adapter, optimizer, dataloader, dataset
    if val_dataloader is not None:
        del val_dataloader, val_dataset
    if device == "cuda":
        torch.cuda.empty_cache()
    
    return {
        "adapter_path": str(final_path),
        "meta_path": str(meta_path),
        "history": history,
        "checkpoint_paths": checkpoint_paths,
        "eval_query_ids_per_task": eval_query_ids_per_task,
        "val_query_ids_per_task": val_query_ids_per_task,
        "stopped_early": stopped_early,
        "best_epoch": best_epoch if best_epoch > 0 else None,
    }
