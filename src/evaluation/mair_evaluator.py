"""
MAIR (Massive AI-Related Information Retrieval) Benchmark Evaluator

This module provides evaluation functionality for the MAIR benchmark,
which tests retrieval models across 126 diverse domains including
math, code, law, finance, and more.

Dataset: https://huggingface.co/MAIR-Bench
"""

import json
import math
import numpy as np
import pandas as pd
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
from datasets import load_dataset, concatenate_datasets
from tqdm import tqdm


# =============================================================================
# Model Configuration
# =============================================================================
# Define context length and batch size limits for each model family.
# Format: "model_pattern": {"max_context_length": int, "max_batch_size": int, "model_size": float}
#   - model_pattern: substring to match in model name (lowercase)
#   - max_context_length: maximum context length (tokens) for truncation
#   - max_batch_size: maximum batch size for sorted batching
#   - model_size: estimated model size in billions (for scaling calculations)
MODEL_CONFIGS = {
    # BGE models
    "bge-m3": {"max_context_length": 8192, "max_batch_size": 4096, "model_size": 0.5},
    "bge_m3": {"max_context_length": 8192, "max_batch_size": 4096, "model_size": 0.5},
    
    # Qwen Embedding models (sorted by size for priority matching)
    "0.6b": {"max_context_length": 32768, "max_batch_size": 1024, "model_size": 0.6},
    "0.5b": {"max_context_length": 32768, "max_batch_size": 1024, "model_size": 0.5},
    # "1.5b": {"max_context_length": 32768, "max_batch_size": 512, "model_size": 1.5},
    "4b": {"max_context_length": 32768, "max_batch_size": 1024, "model_size": 4.0},
    "8b": {"max_context_length": 32768, "max_batch_size": 256, "model_size": 8.0},
    
    # OpenAI models (API-based, context length doesn't affect memory)
    "text-embedding": {"max_context_length": 8191, "max_batch_size": 2048, "model_size": 1.0},
}

# Default configuration for unknown models
DEFAULT_MODEL_CONFIG = {"max_context_length": 32768, "max_batch_size": 512, "model_size": 4.0}


def get_model_config(model_name: str) -> Dict[str, Any]:
    """
    Get configuration for a model based on its name.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Dictionary with max_context_length, max_batch_size, and model_size
    """
    m_name_lower = model_name.lower()
    
    # Check each pattern in order (first match wins)
    for pattern, config in MODEL_CONFIGS.items():
        if pattern in m_name_lower:
            return config
    
    return DEFAULT_MODEL_CONFIG


# Available MAIR tasks grouped by category
MAIR_TASKS = {
    "academic": [
        "Competition-Math",
        "ProofWiki_Proof",
        "ProofWiki_Reference",
        "Stacks_Proof",
        "Stacks_Reference",
        "Stein_Proof",
        "Stein_Reference",
        "Trench_Proof",
        "Trench_Reference",
        "TAD",
        "TAS2",
        "StackMathQA",
        "SciDocs",
        "SciFact",
        "FairRanking_2020",
        "LitSearch",
    ],
    "code": [
        "APPS",
        "CodeEditSearch",
        "CodeSearchNet",
        "Conala",
        "HumanEval-X",
        "LeetCode",
        "MBPP",
        "RepoBench",
        "TLDR",
        "SWE-Bench-Lite",
        "FoodAPI",
        "HuggingfaceAPI",
        "PytorchAPI",
        "SpotifyAPI",
        "TMDB",
        "TensorAPI",
        "ToolBench",
        "WeatherAPI",
    ],
    "finance": [
        "Apple",
        "ConvFinQA",
        "FinQA",
        "FinanceBench",
        "HC3Finance",
        "TAT-DQA",
        "Trade-the-event",
        "FiQA",
    ],
    "legal": [
        "AILA2019-Case",
        "AILA2019-Statutes",
        "BSARD",
        "BillSum",
        "CUAD",
        "GerDaLIR",
        "LeCaRDv2",
        "LegalQuAD",
        "REGIR-EU2UK",
        "REGIR-UK2EU",
        "TREC-Legal_2011",
    ],
    "medical": [
        "NFCorpus",
        "Trec-Covid",
        "Monant",
        "CliniDS_2014",
        "CliniDS_2015",
        "CliniDS_2016",
        "ClinicalTrials_2021",
        "ClinicalTrials_2022",
        "ClinicalTrials_2023",
        "Genomics-AdHoc_2004",
        "Genomics-AdHoc_2005",
        "Genomics-AdHoc_2006",
        "Genomics-AdHoc_2007",
        "PrecisionMedicine_2017",
        "PrecisionMedicine_2018",
        "PrecisionMedicine_2019",
        "PrecisionMedicine-Article_2019",
        "PrecisionMedicine-Article_2020",
        "CARE",
    ],
    "web": [
        "AY2",
        "ELI5",
        "Fever",
        "TREx",
        "WnCw",
        "WnWi",
        "WoW",
        "zsRE",
        "ArguAna",
        "CQADupStack",
        "Quora",
        "TopiOCQA",
        "Touche",
        "ACORDAR",
        "CPCD",
        "ChroniclingAmericaQA",
        "NTCIR",
        "PointRec",
        "ProCIS-Dialog",
        "ProCIS-Turn",
        "QuanTemp",
        "WebTableSearch",
        "CAsT_2019",
        "CAsT_2020",
        "CAsT_2021",
        "CAsT_2022",
        "DD_2015",
        "DD_2016",
        "DD_2017",
        "FairRanking_2021",
        "FairRanking_2022",
        "NeuCLIR-Tech_2023",
        "NeuCLIR_2022",
        "NeuCLIR_2023",
        "ProductSearch_2023",
        "ToT_2023",
        "ToT_2024",
        "Core_2017",
        "Microblog_2011",
        "Microblog_2012",
        "Microblog_2013",
        "Microblog_2014",
        "MISeD",
        "SParC",
        "SParC-SQL",
        "Spider",
        "Spider-SQL",
        "ExcluIR",
        "Core17",
        "News21",
        "Robust04",
        "InstructIR",
        "NevIR",
        "IFEval",
    ],
}

# Flatten all tasks
ALL_MAIR_TASKS = []
for category, tasks in MAIR_TASKS.items():
    ALL_MAIR_TASKS.extend(tasks)


@dataclass
class BucketConfig:
    """Configuration for a token-length bucket in sorted batching."""
    indices: List[int]  # Original indices of texts in this bucket
    texts: List[str]  # Texts in this bucket
    max_tokens: int  # Max tokens for this bucket (determines padding)
    batch_size: int  # Optimal batch size for this bucket


def estimate_token_count(text: str, chars_per_token: float = 4.0) -> int:
    """
    Estimate token count from text length.
    Uses character-based estimation as a fast approximation.
    
    Args:
        text: Input text
        chars_per_token: Average characters per token (4.0 is typical for English)
    
    Returns:
        Estimated token count
    """
    return max(1, int(len(text) / chars_per_token))


# Token thresholds for batch size buckets (upper bounds, exclusive)
# Each threshold corresponds to a batch size transition point
# Format: (max_tokens_threshold, batch_size)
# Texts with tokens < threshold go into that bucket
# Note: 32k is the truncation limit, so texts >= 32k go into the last bucket
BATCH_SIZE_THRESHOLDS = [
    (256, 512),    # 0-255 tokens -> bs 512
    (512, 256),    # 256-511 tokens -> bs 256
    (1024, 128),   # 512-1023 tokens -> bs 128
    (2048, 64),    # 1024-2047 tokens -> bs 64
    (4096, 32),    # 2048-4095 tokens -> bs 32
    (8192, 16),    # 4096-8191 tokens -> bs 16
    (16384, 8),    # 8192-16383 tokens -> bs 8
    (float('inf'), 4),  # 16384+ tokens -> bs 4 (truncated at 32k)
]


def create_sorted_buckets(
    texts: List[str],
    model_size_factor: float = 1.0,
    min_batch_size: int = 1,
    max_batch_size: int = 512,
    buffer_ratio: float = 0.3,
    min_buffer: int = 128,
    **kwargs  # Accept but ignore legacy parameters
) -> List[BucketConfig]:
    """
    Sort texts by estimated token length and create buckets based on batch size thresholds.
    
    Instead of splitting into equal-sized buckets, this groups texts by batch size
    transition points, which minimizes padding waste and maximizes efficiency.
    
    Threshold-based buckets for 4B model (model_size_factor=1.0):
        0-255 tokens    -> batch_size 512, max_tokens ~332
        256-511 tokens  -> batch_size 256, max_tokens ~665
        512-1023 tokens -> batch_size 128, max_tokens ~1330
        1024-2047 tokens-> batch_size 64,  max_tokens ~2661
        2048-4095 tokens-> batch_size 32,  max_tokens ~5324
        4096-8191 tokens-> batch_size 16,  max_tokens ~10649
        8192-16383 tokens-> batch_size 8,  max_tokens ~21299
        16384+ tokens   -> batch_size 4,   max_tokens ~32k (truncated)
    
    Args:
        texts: List of texts to encode
        model_size_factor: Scaling factor based on model size (larger model = smaller factor)
        min_batch_size: Minimum batch size per bucket
        max_batch_size: Maximum batch size per bucket
        buffer_ratio: Extra buffer ratio added to max_tokens (0.3 = 30%)
        min_buffer: Minimum buffer tokens to add
    
    Returns:
        List of BucketConfig, each containing texts, indices, max_tokens, and batch_size
    """
    if not texts:
        return []
    
    # Estimate token counts for all texts
    token_counts = [estimate_token_count(t) for t in texts]
    
    # Create (index, token_count, text) tuples
    indexed_texts = [(i, tc, t) for i, (tc, t) in enumerate(zip(token_counts, texts))]
    
    # Group texts by batch size thresholds
    threshold_buckets: Dict[int, List[Tuple[int, int, str]]] = {i: [] for i in range(len(BATCH_SIZE_THRESHOLDS))}
    
    for item in indexed_texts:
        token_count = item[1]
        # Find the appropriate bucket based on token count
        for bucket_idx, (threshold, _) in enumerate(BATCH_SIZE_THRESHOLDS):
            if token_count < threshold:
                threshold_buckets[bucket_idx].append(item)
                break
    
    # Build BucketConfig for non-empty buckets
    buckets = []
    for bucket_idx, items in threshold_buckets.items():
        if not items:
            continue
        
        # Sort items within bucket by token count for better padding efficiency
        items.sort(key=lambda x: x[1])
        
        # Extract indices, texts, and find max tokens in this bucket
        indices = [item[0] for item in items]
        bucket_texts = [item[2] for item in items]
        bucket_max_tokens = max(item[1] for item in items)
        
        # Add buffer to max_tokens to account for tokenizer variance
        buffer = max(min_buffer, int(bucket_max_tokens * buffer_ratio))
        effective_max_tokens = bucket_max_tokens + buffer
        
        # Get base batch size from threshold table
        _, base_batch_size = BATCH_SIZE_THRESHOLDS[bucket_idx]
        
        # Apply model size factor (smaller models can use larger batches)
        scaled_batch_size = int(base_batch_size * model_size_factor)
        
        # Snap to nearest power of 2 (floor for safety)
        if scaled_batch_size >= 1:
            batch_size = 2 ** int(math.log2(scaled_batch_size))
        else:
            batch_size = 1
        
        # Apply limits
        batch_size = max(min_batch_size, min(max_batch_size, batch_size))
        
        buckets.append(BucketConfig(
            indices=indices,
            texts=bucket_texts,
            max_tokens=effective_max_tokens,
            batch_size=batch_size
        ))
    
    # Sort buckets by max_tokens (process shorter texts first for faster initial progress)
    buckets.sort(key=lambda b: b.max_tokens)
    
    return buckets


def reassemble_embeddings(
    buckets: List[BucketConfig],
    bucket_embeddings: List[np.ndarray]
) -> np.ndarray:
    """
    Reassemble embeddings from buckets back to original order.
    
    Args:
        buckets: List of BucketConfig with original indices
        bucket_embeddings: Embeddings for each bucket (in bucket order)
    
    Returns:
        Embeddings in original text order
    """
    if not buckets or not bucket_embeddings:
        return np.array([])
    
    # Get embedding dimension from first non-empty result
    emb_dim = None
    for emb in bucket_embeddings:
        if emb is not None and len(emb) > 0:
            emb_dim = emb.shape[1] if len(emb.shape) > 1 else emb.shape[0]
            break
    
    if emb_dim is None:
        return np.array([])
    
    # Count total texts
    total_texts = sum(len(b.indices) for b in buckets)
    
    # Preallocate output array
    result = np.zeros((total_texts, emb_dim), dtype=np.float32)
    
    # Place embeddings back in original positions
    for bucket, emb in zip(buckets, bucket_embeddings):
        if emb is None or len(emb) == 0:
            continue
        for local_idx, original_idx in enumerate(bucket.indices):
            result[original_idx] = emb[local_idx]
    
    return result


def get_mair_tasks_by_category(category: str) -> List[str]:
    """Get list of MAIR tasks for a specific category."""
    if category not in MAIR_TASKS:
        raise ValueError(f"Unknown category: {category}. Available: {list(MAIR_TASKS.keys())}")
    return MAIR_TASKS[category]


class MAIRDataset:
    """Loader for MAIR benchmark datasets."""
    
    def __init__(self, task_name: str, cache_dir: Optional[str] = None):
        """
        Initialize MAIR dataset loader.
        
        Args:
            task_name: Name of the MAIR task (e.g., "SciFact", "Competition-Math")
            cache_dir: Optional cache directory for datasets
        """
        if task_name not in ALL_MAIR_TASKS:
            raise ValueError(f"Unknown MAIR task: {task_name}. Use list_mair_tasks() to see available tasks.")
        
        self.task_name = task_name
        self.cache_dir = cache_dir
        self._queries = None
        self._docs = None
        self._qrels = None  # Query relevance judgments
    
    def load(self):
        """Load queries and documents from HuggingFace."""
        print(f"Loading MAIR dataset: {self.task_name}")
        
        # Load queries
        try:
            self._queries_ds = load_dataset(
                "MAIR-Bench/MAIR-Queries",
                self.task_name,
                split="queries",
                cache_dir=self.cache_dir
            )
        except (ValueError, Exception):
            # Fallback for datasets with non-standard splits (e.g. CQADupStack)
            ds_dict = load_dataset(
                "MAIR-Bench/MAIR-Queries",
                self.task_name,
                cache_dir=self.cache_dir
            )
            # Concatenate all available splits
            self._queries_ds = concatenate_datasets(list(ds_dict.values()))
        
        # Load documents
        try:
            self._docs_ds = load_dataset(
                "MAIR-Bench/MAIR-Docs",
                self.task_name,
                split="docs",
                cache_dir=self.cache_dir
            )
        except (ValueError, Exception):
             # Fallback for datasets with non-standard splits
            ds_dict = load_dataset(
                "MAIR-Bench/MAIR-Docs",
                self.task_name,
                cache_dir=self.cache_dir
            )
            self._docs_ds = concatenate_datasets(list(ds_dict.values()))
        
        # Process queries
        self._queries = {}
        self._qrels = {}
        self._instructions = {}
        
        for item in self._queries_ds:
            qid = str(item["qid"])
            self._queries[qid] = item["query"]
            self._instructions[qid] = item.get("instruction", "")
            
            # Build qrels from labels
            if "labels" in item and item["labels"]:
                self._qrels[qid] = {
                    str(label["id"]): label["score"]
                    for label in item["labels"]
                }
        
        # Process documents
        self._docs = {}
        for item in self._docs_ds:
            doc_id = str(item["id"])
            self._docs[doc_id] = item["doc"]
        
        print(f"Loaded {len(self._queries)} queries and {len(self._docs)} documents")
        return self
    
    @property
    def queries(self) -> Dict[str, str]:
        """Return queries as {qid: query_text}."""
        if self._queries is None:
            self.load()
        return self._queries
    
    @property
    def corpus(self) -> Dict[str, str]:
        """Return corpus as {doc_id: doc_text}."""
        if self._docs is None:
            self.load()
        return self._docs
    
    @property
    def qrels(self) -> Dict[str, Dict[str, int]]:
        """Return relevance judgments as {qid: {doc_id: score}}."""
        if self._qrels is None:
            self.load()
        return self._qrels
    
    @property
    def instructions(self) -> Dict[str, str]:
        """Return instructions as {qid: instruction}."""
        if self._instructions is None:
            self.load()
        return self._instructions
    
    def get_default_instruction(self) -> str:
        """Get the default instruction for this task (from first query)."""
        if self._instructions is None:
            self.load()
        if self._instructions:
            return next(iter(self._instructions.values()))
        return ""


class MAIREvaluator:
    """Evaluator for MAIR benchmark."""
    
    def __init__(
        self,
        model,
        experiment_name: str = "mair",
        cache_enabled: bool = True,
        cache_dir: str = "cache/embeddings",
        use_instructions: bool = True,
        force_recache: bool = False,
        use_subdirs: bool = True
    ):
        """
        Initialize MAIR evaluator.
        
        Args:
            model: Embedding model (BaseEmbedder or compatible)
            experiment_name: Name for this experiment
            cache_enabled: Whether to enable embedding cache
            cache_dir: Directory for embedding cache
            use_instructions: Whether to use task instructions when encoding queries
            force_recache: If True, ignore existing cache and recompute (but still save to cache)
            use_subdirs: If True, create subdirectories with experiment_name and model_name.
                        If False, use output_folder directly (useful for custom output paths).
        """
        self.model = model
        self.experiment_name = experiment_name
        self.cache_enabled = cache_enabled
        self.cache_dir = cache_dir
        self.use_instructions = use_instructions
        self.force_recache = force_recache
        self.use_subdirs = use_subdirs
        self.model_name = getattr(model, "model_name", "unknown_model")
        
        # Import cache
        from .embedding_cache import EmbeddingCache
        self._cache = EmbeddingCache(cache_dir=cache_dir, enabled=cache_enabled, force_recache=force_recache)

        # Load dataset stats for dynamic context length
        self.dataset_stats = {}
        # Try finding mair_stats.json in cwd or parent directories
        possible_paths = [Path("mair_stats.json"), Path("../mair_stats.json"), Path("retrieval_benchmark/mair_stats.json")]
        for p in possible_paths:
            if p.exists():
                try:
                    with open(p, "r") as f:
                        self.dataset_stats = json.load(f)
                    print(f"Loaded MAIR stats from {p} for {len(self.dataset_stats)} tasks")
                    break
                except Exception as e:
                    print(f"Warning: Failed to load {p}: {e}")
        
        if not self.dataset_stats:
            print("Warning: mair_stats.json not found. Dynamic context length optimization disabled.")

    def _get_adapter(self):
        """Return the underlying adapter module, unwrapping DataParallel if needed."""
        adapter = getattr(self.model, "adapter", None)
        if adapter is None:
            return None
        return getattr(adapter, "module", adapter)

    def _get_adapter_output_dim(self) -> int:
        """Infer adapter output dimensionality without requiring an encode() call."""
        adapter = self._get_adapter()
        if adapter is None:
            raise RuntimeError("Adapter is not configured on the current model")

        if hasattr(adapter, 'proj') and hasattr(adapter.proj, 'weight'):
            return adapter.proj.weight.shape[0]

        raise RuntimeError(f"Unknown adapter type: {type(adapter)}")

    def _apply_query_adapter(self, embeddings: np.ndarray) -> np.ndarray:
        """Apply the model adapter to cached query embeddings."""
        if not (hasattr(self.model, 'adapter') and self.model.adapter is not None):
            return embeddings

        if hasattr(self.model, '_adapt_embeddings'):
            return self.model._adapt_embeddings(embeddings)

        import torch

        with torch.no_grad():
            tensor = torch.as_tensor(embeddings, dtype=torch.float32)
            device = getattr(self.model, 'device', None)
            if device:
                tensor = tensor.to(device)
            adapted = self.model.adapter(tensor)
            adapted = torch.nn.functional.normalize(adapted, p=2, dim=-1)
            return adapted.cpu().numpy()

    def _has_preloaded_cache(self, split: str) -> bool:
        """Return whether the current model has a usable preloaded cache for the split."""
        if split == "queries":
            return bool(getattr(self.model, "_full_query_cache", {}))
        if split == "corpus":
            return bool(getattr(self.model, "_full_corpus_cache", {}))
        return False

    def _filter_eval_subset(
        self,
        queries: Dict[str, str],
        qrels: Dict[str, Dict[str, int]],
        instructions: Dict[str, str],
        eval_query_ids: Optional[List[str]],
    ) -> tuple[Dict[str, str], Dict[str, Dict[str, int]], Dict[str, str]]:
        """Restrict evaluation to an explicit subset of query IDs when requested."""
        if eval_query_ids is None:
            return queries, qrels, instructions

        allowed = {str(qid) for qid in eval_query_ids}
        filtered_queries = {qid: text for qid, text in queries.items() if qid in allowed}
        filtered_qrels = {qid: rels for qid, rels in qrels.items() if qid in filtered_queries}
        filtered_instructions = {
            qid: instructions.get(qid, "") for qid in filtered_queries
        } if instructions else {}

        if not filtered_queries:
            raise ValueError("No eval queries remain after applying eval_query_ids filter")

        print(f"  [Eval filter] Evaluating {len(filtered_queries)}/{len(queries)} queries")
        return filtered_queries, filtered_qrels, filtered_instructions
    
    def run(
        self,
        tasks: List[str],
        output_folder: str = "results",
        batch_size: int = 32,
        top_k: int = 100,
        eval_query_ids_per_task: Optional[Dict[str, List[str]]] = None,
    ) -> Dict[str, Dict[str, float]]:
        """
        Run MAIR evaluation on specified tasks.
        
        Args:
            tasks: List of MAIR task names
            output_folder: Directory to save results
            batch_size: Batch size for encoding
            top_k: Number of top documents to retrieve
            eval_query_ids_per_task: Optional per-task query-id subset used for
                leakage-free evaluation after train/val/eval splitting.
            
        Returns:
            Dictionary of {task_name: {metric: score}}
        """
        import torch
        import gc
        
        # Create output directory
        if self.use_subdirs:
            safe_model_name = self.model_name.replace("/", "__").replace("models__", "")
            output_dir = Path(output_folder) / self.experiment_name / safe_model_name
        else:
            # Use output_folder directly for simpler path structure
            output_dir = Path(output_folder)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"MAIR Results will be saved to: {output_dir}")
        
        all_results = {}
        
        for task_name in tasks:
            print(f"\n{'='*50}")
            print(f"Running MAIR task: {task_name}")
            print(f"{'='*50}")
            
            try:
                task_results = self._evaluate_task(
                    task_name,
                    batch_size,
                    top_k,
                    eval_query_ids=(eval_query_ids_per_task or {}).get(task_name),
                )
                all_results[task_name] = task_results
                
                # Save individual task results
                task_output_dir = output_dir / "tasks" / task_name
                task_output_dir.mkdir(parents=True, exist_ok=True)
                
                with open(task_output_dir / "metrics.json", "w") as f:
                    json.dump(task_results, f, indent=2)
                
                print(f"Task {task_name} completed:")
                for metric, score in task_results.items():
                    print(f"  {metric}: {score:.4f}")

                # Save incremental summary
                self._save_summary({task_name: task_results}, output_dir)
                
            except Exception as e:
                print(f"Error evaluating task {task_name}: {e}")
                import traceback
                traceback.print_exc()
                all_results[task_name] = {"error": str(e)}
            
            # Clear GPU memory between tasks
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        
        # Save summary results
        self._save_summary(all_results, output_dir)
        
        return all_results
    
    def _evaluate_task(
        self,
        task_name: str,
        batch_size: int,
        top_k: int,
        eval_query_ids: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """Evaluate a single MAIR task."""
        import torch
        
        # Load dataset
        dataset = MAIRDataset(task_name)
        dataset.load()
        
        # Get model configuration for context length limits
        model_config = get_model_config(self.model_name)
        model_max_context = model_config["max_context_length"]
        
        # Determine max tokens needed for this task
        # Corpus uses sorted batching which handles this per-bucket
        # For queries, we need to cap at model's max context length
        corpus_max_tokens = None
        query_max_tokens = model_max_context  # Default to model max for queries
        
        if task_name in self.dataset_stats:
            try:
                # Get corpus stats
                corpus_stats = self.dataset_stats[task_name].get("corpus", {})
                if "max_tokens" in corpus_stats:
                    stats_max = int(corpus_stats["max_tokens"])
                    # specific tokenizer variance buffer: +20% but at least +256 tokens
                    # This ensures we don't accidentally truncate due to tokenizer differences
                    buffer = max(256, int(stats_max * 0.2))
                    corpus_max_tokens = stats_max + buffer
                    print(f"  [Config] Corpus max_tokens for {task_name}: {corpus_max_tokens} (Stats: {stats_max} + Buffer: {buffer})")
                
                # Get query stats - cap at model's max context length
                query_stats = self.dataset_stats[task_name].get("queries", {})
                if "max_tokens" in query_stats:
                    q_stats_max = int(query_stats["max_tokens"])
                    q_buffer = max(256, int(q_stats_max * 0.2))
                    q_requested = q_stats_max + q_buffer
                    # Cap at model's context length limit
                    query_max_tokens = min(q_requested, model_max_context)
                    if q_requested > model_max_context:
                        print(f"  [Warning] Query max_tokens ({q_requested}) exceeds model limit ({model_max_context}). Will truncate.")
                    print(f"  [Config] Query max_tokens for {task_name}: {query_max_tokens} (Stats: {q_stats_max})")
            except Exception as e:
                print(f"  [Warning] Failed to parse stats for {task_name}: {e}")

        # Dynamic Batch Size Adjustment for queries
        # Note: For corpus encoding, sorted batching handles this automatically per-bucket
        # Reference: Qwen3-4B @ 32k ctx -> batch_size=64 (A100 80GB)
        # We assume activations dominate memory for large batches/ctx, so B * L * ModelScale ~ Constant
        if "Qwen" in self.model_name:
            effective_ctx = query_max_tokens if query_max_tokens else 32768 # Default if not found
            
            # 1. Base Reference (4B model)
            ref_bs = 16
            ref_ctx = 32768
            ref_size = 4.0 # Billions params
            
            # 2. Estimate Current Model Size
            curr_size = 4.0 # Default fallback
            m_name_lower = self.model_name.lower()
            if "0.6b" in m_name_lower or "0.5b" in m_name_lower:
                curr_size = 0.4
            elif "1.5b" in m_name_lower:
                curr_size = 1.5
            elif "4b" in m_name_lower:
                curr_size = 3.0
            elif "8b" in m_name_lower:
                curr_size = 6.0
            elif "14b" in m_name_lower:
                curr_size = 14.0
            elif "32b" in m_name_lower:
                curr_size = 32.0
            elif "72b" in m_name_lower:
                curr_size = 72.0
                
            # 3. Calculate Scaling Factors
            # Ctx Factor: As ctx decreases, BS increases (inverse)
            ctx_factor = ref_ctx / effective_ctx
            
            # Size Factor: As model size decreases, BS increases (inverse)
            # Use sqrt scaling for size as a safe heuristic (activations typical scale with d_model ~ sqrt(params))
            size_factor = math.sqrt(ref_size / curr_size)
            
            # 4. Compute Target Batch Size
            raw_target_bs = ref_bs * ctx_factor * size_factor
            
            # Snap to nearest power of 2 (conservative floor to avoid OOM)
            if raw_target_bs < 1:
                target_bs = 1
            else:
                target_bs = 2 ** int(math.log2(raw_target_bs))
            
            # 5. Apply Safety Limits
            target_bs = int(max(1, target_bs))
            target_bs = int(min(target_bs, 2048)) # Hard cap to prevent excessive overhead
            
            print(f"  [Config] Dynamic Batch Size (for queries): {target_bs} (Base Ref: {ref_bs}@32k/4B)")
            print(f"    Details: Ctx={effective_ctx} (Factor {ctx_factor:.2f}), Model={curr_size}B (Factor {size_factor:.2f})")
            
            batch_size = target_bs

        queries = dataset.queries
        corpus = dataset.corpus
        qrels = dataset.qrels
        instructions = dataset.instructions if self.use_instructions else {}

        queries, qrels, instructions = self._filter_eval_subset(
            queries,
            qrels,
            instructions,
            eval_query_ids,
        )
        
        print(f"  Queries: {len(queries)}, Corpus: {len(corpus)}")
        print(f"  Instructions: {'enabled' if self.use_instructions else 'disabled'}")
        
        # Check if we have per-query instructions (different instructions per query)
        unique_instructions = set(instructions.values()) if instructions else set()
        has_per_query_instructions = len(unique_instructions) > 1
        
        if self.use_instructions and instructions:
            if has_per_query_instructions:
                print(f"  Per-query instructions detected: {len(unique_instructions)} unique instructions")
            else:
                sample_inst = next(iter(instructions.values()), "")
                print(f"  Single instruction for all queries: {sample_inst[:100]}...")
        
        # Calculate model size factor for sorted batching
        model_size_factor = self._get_model_size_factor()
        
        # Encode queries - use per-query instructions if available
        print("  Encoding queries...")
        query_ids = list(queries.keys())
        query_texts = [queries[qid] for qid in query_ids]
        
        if has_per_query_instructions:
            # Per-query instructions: encode with individual instructions
            query_instructions = [instructions.get(qid, "") for qid in query_ids]
            query_embeddings = self._encode_queries_with_per_query_instructions(
                query_texts, query_instructions, batch_size, task_name,
                model_size_factor=model_size_factor
            )
        else:
            # Single instruction for all queries (or no instructions)
            instruction = next(iter(instructions.values()), None) if instructions else None
            query_embeddings = self._encode_with_sorted_batching(
                query_texts, batch_size, instruction, task_name, "queries",
                model_size_factor=model_size_factor
            )
        
        # Encode corpus with sorted batching for efficiency
        print("  Encoding corpus with sorted batching...")
        doc_ids = list(corpus.keys())
        doc_texts = [corpus[did] for did in doc_ids]
        doc_embeddings = self._encode_with_sorted_batching(
            doc_texts, batch_size, None, task_name, "corpus", 
            model_size_factor=model_size_factor
        )
        
        # Compute similarities and retrieve
        print("  Computing similarities...")
        
        # Convert to torch for efficient computation
        query_emb_tensor = torch.from_numpy(query_embeddings).float()
        doc_emb_tensor = torch.from_numpy(doc_embeddings).float()
        
        # Normalize embeddings
        query_emb_tensor = query_emb_tensor / query_emb_tensor.norm(dim=-1, keepdim=True)
        doc_emb_tensor = doc_emb_tensor / doc_emb_tensor.norm(dim=-1, keepdim=True)
        
        # Compute all similarities
        similarities = torch.mm(query_emb_tensor, doc_emb_tensor.t())
        
        # Get top-k results
        results = {}
        for i, qid in enumerate(query_ids):
            scores, indices = torch.topk(similarities[i], min(top_k, len(doc_ids)))
            results[qid] = {
                doc_ids[idx]: float(score)
                for idx, score in zip(indices.tolist(), scores.tolist())
            }
        
        # Compute metrics
        metrics = self._compute_metrics(results, qrels, [1, 5, 10, 20, 100])
        
        return metrics

    def _encode_with_cache(
        self,
        texts: List[str],
        batch_size: int,
        instruction: Optional[str],
        task_name: str,
        split: str,
        max_tokens: Optional[int] = None
    ) -> np.ndarray:
        """Encode texts with caching support."""
        import torch
        import gc
        
        cache_save_interval = 10000
        all_embeddings = []
        
        total_chunks = (len(texts) + cache_save_interval - 1) // cache_save_interval
        
        # Include instruction info in cache key to differentiate instruction vs no-instruction runs
        # For queries with instruction, include instruction hash in split name to invalidate
        # cache when instruction format changes (e.g., "Instruct: ...\nQuery: ..." vs just prepending)
        if instruction is not None and split == "queries":
            import hashlib
            inst_hash = hashlib.md5(instruction.encode('utf-8')).hexdigest()[:8]
            cache_split = f"queries_with_instruction_{inst_hash}"
        else:
            cache_split = split
        
        for chunk_idx, chunk_start in enumerate(range(0, len(texts), cache_save_interval), 1):
            chunk_end = min(chunk_start + cache_save_interval, len(texts))
            chunk_texts = texts[chunk_start:chunk_end]
            
            print(f"    [Progress {cache_split}] Chunk {chunk_idx}/{total_chunks} ({len(chunk_texts)} texts)")
            
            # Check cache
            cached = self._cache.get(
                self.model_name,
                chunk_texts,
                task_name=f"MAIR_{task_name}",
                split=cache_split
            )
            
            if cached is not None:
                all_embeddings.append(cached)
            else:
                # Encode
                # Pass max_length if provided (for dynamic context length)
                kwargs = {}
                if max_tokens:
                    kwargs['max_length'] = max_tokens
                    
                chunk_emb = self.model.encode(chunk_texts, batch_size=batch_size, instruction=instruction, **kwargs)
                if not isinstance(chunk_emb, np.ndarray):
                    chunk_emb = np.array(chunk_emb)
                
                # Save to cache
                self._cache.put(
                    self.model_name,
                    chunk_texts,
                    chunk_emb,
                    task_name=f"MAIR_{task_name}",
                    split=cache_split
                )
                
                all_embeddings.append(chunk_emb)
                
                # Clear memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
        
        return np.vstack(all_embeddings) if all_embeddings else np.array([])
    
    def _get_model_size_factor(self) -> float:
        """
        Get model size scaling factor for batch size calculations.
        Larger models need smaller factors (smaller batch sizes).
        
        Uses conservative (pessimistic) scaling to prevent OoM errors.
        
        Returns:
            Scaling factor based on estimated model size
        """
        m_name_lower = self.model_name.lower()
        
        # Reference: 4B model = 1.0
        ref_size = 4.0
        
        # Estimate current model size
        curr_size = 4.0  # Default fallback
        if "0.6b" in m_name_lower or "0.5b" in m_name_lower:
            curr_size = 0.6
        elif "1.5b" in m_name_lower:
            curr_size = 1.5
        elif "4b" in m_name_lower:
            curr_size = 4.0
        elif "8b" in m_name_lower:
            curr_size = 8.0
        elif "14b" in m_name_lower:
            curr_size = 14.0
        elif "32b" in m_name_lower:
            curr_size = 32.0
        elif "72b" in m_name_lower:
            curr_size = 72.0
        elif "bge-m3" in m_name_lower:
            curr_size = 0.5  # BGE-M3 is relatively small
        elif "text-embedding" in m_name_lower:
            curr_size = 1.0  # OpenAI models, API-based so doesn't matter much
        
        # Conservative scaling: use cube root instead of sqrt for safety
        # This gives smaller batch sizes for larger models
        # e.g., 4B->4B: 1.0, 0.6B->4B: 1.87 (was 2.58), 8B->4B: 0.79 (was 0.71)
        return (ref_size / curr_size) ** (1/3)
    
    def _encode_queries_with_per_query_instructions(
        self,
        texts: List[str],
        instructions: List[str],
        base_batch_size: int,
        task_name: str,
        model_size_factor: float = 1.0
    ) -> np.ndarray:
        """
        Encode queries where each query has its own instruction.
        
        This handles tasks like Core17, InstructIR, etc. where each query
        has a unique instruction that must be applied individually.
        
        Strategy: Group queries by identical instructions and batch-encode
        each group together for efficiency.
        
        Args:
            texts: List of query texts
            instructions: List of instructions (one per query)
            base_batch_size: Base batch size for encoding
            task_name: Task name for caching
            model_size_factor: Model size scaling factor
            
        Returns:
            Embeddings in original text order
        """
        import torch
        import gc
        from collections import defaultdict
        
        if not texts:
            return np.array([])
        
        assert len(texts) == len(instructions), \
            f"texts ({len(texts)}) and instructions ({len(instructions)}) must have same length"

        if self._has_preloaded_cache("queries"):
            print("    [Per-Query Instructions] Using pre-loaded query cache")
            sample_emb = self.model.encode([texts[0]], batch_size=1, instruction=instructions[0])
            if not isinstance(sample_emb, np.ndarray):
                sample_emb = np.array(sample_emb)
            emb_dim = sample_emb.shape[-1]
            result = np.zeros((len(texts), emb_dim), dtype=np.float32)

            from collections import defaultdict

            instruction_to_indices = defaultdict(list)
            for i, inst in enumerate(instructions):
                instruction_to_indices[inst].append(i)

            for instruction, indices in instruction_to_indices.items():
                group_texts = [texts[i] for i in indices]
                group_embeddings = self.model.encode(
                    group_texts,
                    batch_size=max(1, min(base_batch_size, len(group_texts))),
                    instruction=instruction,
                )
                if not isinstance(group_embeddings, np.ndarray):
                    group_embeddings = np.array(group_embeddings)
                for j, idx in enumerate(indices):
                    result[idx] = group_embeddings[j]

            print(f"    [Per-Query Instructions] Completed encoding {len(texts)} queries from pre-loaded cache")
            return result
        
        # Group queries by instruction for batch efficiency
        instruction_to_indices = defaultdict(list)
        for i, inst in enumerate(instructions):
            instruction_to_indices[inst].append(i)
        
        n_groups = len(instruction_to_indices)
        print(f"    [Per-Query Instructions] Grouping {len(texts)} queries into {n_groups} instruction groups")
        
        # Determine the correct cache model name to use
        # For AdaptedQueryEmbedder, use cache_model_name_for_queries
        cache_model_name = self.model_name
        if hasattr(self.model, 'cache_model_name_for_queries'):
            cache_model_name = self.model.cache_model_name_for_queries
            print(f"    [Per-Query Instructions] Using query cache model: {cache_model_name}")
        
        # Pre-allocate result array
        # Get embedding dimension.
        # For adapter models, derive from adapter output layer to avoid calling
        # model.encode() which requires a pre-loaded in-memory cache that may be
        # empty when evaluate_adapter_on_tasks filters queries to an eval subset.
        if hasattr(self.model, 'adapter') and self.model.adapter is not None:
            emb_dim = self._get_adapter_output_dim()
            print(f"    [Per-Query Instructions] Adapter output dim: {emb_dim}")
        else:
            sample_emb = self.model.encode([texts[0]], batch_size=1, instruction=instructions[0])
            if not isinstance(sample_emb, np.ndarray):
                sample_emb = np.array(sample_emb)
            emb_dim = sample_emb.shape[-1]
        
        result = np.zeros((len(texts), emb_dim), dtype=np.float32)
        
        # Get model configuration
        model_config = get_model_config(self.model_name)
        max_batch_size = model_config["max_batch_size"]
        max_context_length = model_config["max_context_length"]
        
        # Process each instruction group
        processed_groups = 0
        for instruction, indices in instruction_to_indices.items():
            processed_groups += 1
            group_texts = [texts[i] for i in indices]
            
            # Check cache first - use MD5 hash of instruction for consistent cache keys
            import hashlib
            inst_hash = hashlib.md5(instruction.encode('utf-8')).hexdigest()[:8]
            cache_split = f"queries_perquery_inst_{inst_hash}"
            
            cached = self._cache.get(
                cache_model_name,
                group_texts,
                task_name=f"MAIR_{task_name}",
                split=cache_split
            )
            
            if cached is not None:
                # For AdaptedQueryEmbedder, apply adapter to cached large model embeddings
                if hasattr(self.model, 'adapter') and self.model.adapter is not None:
                    cached = self._apply_query_adapter(cached)
                    if processed_groups <= 3 or processed_groups % 10 == 0:
                        print(f"    [Group {processed_groups}/{n_groups}] {len(indices)} queries - cache + adapter")
                else:
                    if processed_groups <= 3 or processed_groups % 10 == 0:
                        print(f"    [Group {processed_groups}/{n_groups}] {len(indices)} queries - using cache")
                
                # Store in result array at original indices
                for j, idx in enumerate(indices):
                    result[idx] = cached[j]
                continue
            
            # Cache miss - for AdaptedQueryEmbedder, this is an error
            # The adapter requires pre-cached large model embeddings
            if hasattr(self.model, 'adapter') and self.model.adapter is not None:
                raise RuntimeError(
                    f"Cache miss for AdaptedQueryEmbedder in _encode_queries_with_per_query_instructions!\n"
                    f"  Task: {task_name}\n"
                    f"  Cache split: {cache_split}\n"
                    f"  Model name: {cache_model_name}\n"
                    f"  Group size: {len(group_texts)}\n"
                    f"  Instruction hash: {inst_hash}\n"
                    f"This indicates the large model embeddings were not pre-cached. "
                    f"Please run the large model evaluation first to populate the cache."
                )
            
            # Encode this group (only for non-adapter models)
            if processed_groups <= 3 or processed_groups % 10 == 0:
                inst_preview = instruction[:50] + "..." if len(instruction) > 50 else instruction
                print(f"    [Group {processed_groups}/{n_groups}] {len(indices)} queries - encoding")
                print(f"      Instruction: {inst_preview}")
            
            # Use sorted batching for this group
            if len(indices) > 1:
                buckets = create_sorted_buckets(
                    group_texts,
                    model_size_factor=model_size_factor,
                    min_batch_size=1,
                    max_batch_size=max_batch_size
                )
                
                bucket_embeddings = []
                for bucket in buckets:
                    effective_max_tokens = min(bucket.max_tokens, max_context_length)
                    kwargs = {'max_length': effective_max_tokens}
                    bucket_emb = self.model.encode(
                        bucket.texts,
                        batch_size=bucket.batch_size,
                        instruction=instruction,
                        **kwargs
                    )
                    if not isinstance(bucket_emb, np.ndarray):
                        bucket_emb = np.array(bucket_emb)
                    bucket_embeddings.append(bucket_emb)
                
                group_embeddings = reassemble_embeddings(buckets, bucket_embeddings)
            else:
                # Single query, encode directly
                group_embeddings = self.model.encode(
                    group_texts,
                    batch_size=1,
                    instruction=instruction
                )
                if not isinstance(group_embeddings, np.ndarray):
                    group_embeddings = np.array(group_embeddings)
            
            # Cache the group embeddings
            self._cache.put(
                cache_model_name,
                group_texts,
                group_embeddings,
                task_name=f"MAIR_{task_name}",
                split=cache_split
            )
            
            # Store in result array at original indices
            for j, idx in enumerate(indices):
                result[idx] = group_embeddings[j]
            
            # Clear memory periodically
            if processed_groups % 20 == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
        
        print(f"    [Per-Query Instructions] Completed encoding {len(texts)} queries")
        return result
    
    def _encode_with_sorted_batching(
        self,
        texts: List[str],
        base_batch_size: int,
        instruction: Optional[str],
        task_name: str,
        split: str,
        model_size_factor: float = 1.0
    ) -> np.ndarray:
        """
        Encode texts using sorted batching for improved efficiency.
        
        Groups texts by batch size threshold boundaries, which minimizes
        padding waste and maximizes throughput. Each bucket uses the optimal
        batch size for its token length range.
        
        Args:
            texts: List of texts to encode
            base_batch_size: Base batch size (used as fallback)
            instruction: Optional instruction to prepend
            task_name: Task name for caching
            split: Split name for caching
            model_size_factor: Model size scaling factor
            num_buckets: Number of buckets to create
        
        Returns:
            Embeddings in original text order
        """
        import torch
        import gc
        import hashlib
        
        if not texts:
            return np.array([])

        if self._has_preloaded_cache(split):
            print(f"    [Sorted Batching] Using pre-loaded {split} cache")
            embeddings = self.model.encode(
                texts,
                batch_size=base_batch_size,
                instruction=instruction,
            )
            if not isinstance(embeddings, np.ndarray):
                embeddings = np.array(embeddings)
            return embeddings
        
        # Include instruction hash in cache key to differentiate different instruction formats
        if instruction is not None and split == "queries":
            inst_hash = hashlib.md5(instruction.encode('utf-8')).hexdigest()[:8]
            cache_split = f"queries_with_instruction_{inst_hash}"
        else:
            cache_split = split
        
        # Determine the correct cache model name to use
        # For AdaptedQueryEmbedder, use the underlying model names
        cache_model_name = self.model_name
        
        if hasattr(self.model, 'cache_model_name_for_corpus') and split == 'corpus':
            cache_model_name = self.model.cache_model_name_for_corpus
            print(f"    [Sorted Batching] Using corpus cache model: {cache_model_name}")
        elif hasattr(self.model, 'cache_model_name_for_queries') and split == 'queries':
            cache_model_name = self.model.cache_model_name_for_queries
            print(f"    [Sorted Batching] Using query cache model: {cache_model_name}")
        
        # Debug: Show what we're looking for
        print(f"    [Sorted Batching] Looking for cache: model={cache_model_name}, task=MAIR_{task_name}, split={cache_split}, texts={len(texts)}, force_recache={self.force_recache}")
        
        # Check if all texts are already cached
        cached = self._cache.get(
            cache_model_name,
            texts,
            task_name=f"MAIR_{task_name}",
            split=cache_split
        )
        if cached is not None:
            print(f"    [Sorted Batching] Using cached embeddings for {len(texts)} texts")
            
            # For AdaptedQueryEmbedder with queries, always apply adapter to cached large model embeddings
            # This is done regardless of force_recache setting
            if hasattr(self.model, 'adapter') and self.model.adapter is not None and split == 'queries':
                print(f"    [Sorted Batching] Applying adapter to {len(cached)} cached query embeddings")
                cached = self._apply_query_adapter(cached)
            
            return cached
        
        # Cache miss - for AdaptedQueryEmbedder, this is an error
        # The adapter requires pre-cached large model embeddings
        if hasattr(self.model, 'adapter') and self.model.adapter is not None:
            raise RuntimeError(
                f"Cache miss for AdaptedQueryEmbedder in _encode_with_sorted_batching!\n"
                f"  Task: {task_name}\n"
                f"  Split: {split}\n"
                f"  Cache split: {cache_split}\n"
                f"  Model name: {cache_model_name}\n"
                f"  Text count: {len(texts)}\n"
                f"This indicates the large model embeddings were not pre-cached. "
                f"Please run the large model evaluation first to populate the cache."
            )
        
        # Get model configuration (only for non-adapter models)
        model_config = get_model_config(self.model_name)
        max_batch_size = model_config["max_batch_size"]
        max_context_length = model_config["max_context_length"]
        
        print(f"    [Model Config] {self.model_name}: max_context={max_context_length}, max_batch={max_batch_size}")
        
        # Estimate instruction token overhead if instruction is provided
        # This is needed because sorted_buckets calculates max_tokens based on original texts,
        # but the actual encoding will prepend "Instruct: {instruction}\nQuery: " to each text
        instruction_overhead = 0
        if instruction is not None:
            # Estimate: "Instruct: " (3 tokens) + instruction + "\nQuery: " (3 tokens)
            # Rough estimate: 1 token per 4 characters for instruction
            instruction_overhead = len(instruction) // 4 + 10
            print(f"    [Sorted Batching] Instruction overhead estimate: ~{instruction_overhead} tokens")
        
        # Create sorted buckets with conservative memory settings
        # Quadratic scaling: bs = 512 * (256 / tokens)^2 for 4B model
        buckets = create_sorted_buckets(
            texts,
            model_size_factor=model_size_factor,
            min_batch_size=1,
            max_batch_size=max_batch_size
        )
        
        # Print bucket statistics
        print(f"    [Sorted Batching] Created {len(buckets)} buckets for {len(texts)} texts:")
        for i, bucket in enumerate(buckets):
            print(f"      Bucket {i+1}: {len(bucket.texts)} texts, "
                  f"max_tokens={bucket.max_tokens}, batch_size={bucket.batch_size}")
        
        # Process each bucket
        bucket_embeddings = []
        for bucket_idx, bucket in enumerate(buckets):
            # Add instruction overhead to max_tokens, then cap to model's context length limit
            effective_max_tokens = min(bucket.max_tokens + instruction_overhead, max_context_length)
            
            print(f"    [Sorted Batching] Processing bucket {bucket_idx + 1}/{len(buckets)} "
                  f"({len(bucket.texts)} texts, bs={bucket.batch_size}, max_tok={effective_max_tokens})")
            
            # Encode bucket with its optimized settings
            kwargs = {'max_length': effective_max_tokens, 'task_name': task_name}
            bucket_emb = self.model.encode(
                bucket.texts, 
                batch_size=bucket.batch_size, 
                instruction=instruction, 
                **kwargs
            )
            if not isinstance(bucket_emb, np.ndarray):
                bucket_emb = np.array(bucket_emb)
            
            bucket_embeddings.append(bucket_emb)
            
            # Clear memory between buckets
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        
        # Reassemble embeddings in original order
        result = reassemble_embeddings(buckets, bucket_embeddings)
        
        # Cache the reassembled result
        if len(result) > 0:
            self._cache.put(
                cache_model_name,
                texts,
                result,
                task_name=f"MAIR_{task_name}",
                split=cache_split
            )
        
        return result
    
    def _compute_metrics(
        self,
        results: Dict[str, Dict[str, float]],
        qrels: Dict[str, Dict[str, int]],
        k_values: List[int]
    ) -> Dict[str, float]:
        """Compute retrieval metrics."""
        metrics = {}
        
        # NDCG@k
        for k in k_values:
            ndcg_scores = []
            for qid in qrels:
                if qid not in results:
                    continue
                
                retrieved = list(results[qid].keys())[:k]
                rel_scores = qrels[qid]
                
                # DCG
                dcg = sum(
                    rel_scores.get(doc_id, 0) / np.log2(rank + 2)
                    for rank, doc_id in enumerate(retrieved)
                )
                
                # Ideal DCG
                ideal_scores = sorted(rel_scores.values(), reverse=True)[:k]
                idcg = sum(
                    score / np.log2(rank + 2)
                    for rank, score in enumerate(ideal_scores)
                )
                
                ndcg_scores.append(dcg / idcg if idcg > 0 else 0)
            
            metrics[f"ndcg_at_{k}"] = np.mean(ndcg_scores) if ndcg_scores else 0
        
        # Recall@k
        for k in k_values:
            recall_scores = []
            for qid in qrels:
                if qid not in results:
                    continue
                
                retrieved = set(list(results[qid].keys())[:k])
                relevant = set(doc_id for doc_id, score in qrels[qid].items() if score > 0)
                
                if relevant:
                    recall_scores.append(len(retrieved & relevant) / len(relevant))
            
            metrics[f"recall_at_{k}"] = np.mean(recall_scores) if recall_scores else 0
        
        # MRR (Mean Reciprocal Rank)
        mrr_scores = []
        for qid in qrels:
            if qid not in results:
                continue
            
            retrieved = list(results[qid].keys())
            relevant = set(doc_id for doc_id, score in qrels[qid].items() if score > 0)
            
            for rank, doc_id in enumerate(retrieved, 1):
                if doc_id in relevant:
                    mrr_scores.append(1 / rank)
                    break
            else:
                mrr_scores.append(0)
        
        metrics["mrr"] = np.mean(mrr_scores) if mrr_scores else 0
        
        # MAP (Mean Average Precision)
        map_scores = []
        for qid in qrels:
            if qid not in results:
                continue
            
            retrieved = list(results[qid].keys())
            relevant = set(doc_id for doc_id, score in qrels[qid].items() if score > 0)
            
            if not relevant:
                continue
            
            precisions = []
            num_relevant_found = 0
            for rank, doc_id in enumerate(retrieved, 1):
                if doc_id in relevant:
                    num_relevant_found += 1
                    precisions.append(num_relevant_found / rank)
            
            map_scores.append(sum(precisions) / len(relevant) if precisions else 0)
        
        metrics["map"] = np.mean(map_scores) if map_scores else 0
        
        return metrics
    
    def _save_summary(self, results: Dict[str, Dict[str, float]], output_dir: Path):
        """Save summary results to CSV."""
        rows = []
        for task_name, metrics in results.items():
            if "error" in metrics:
                continue
            row = {"Model": self.model_name, "Task": task_name}
            row.update(metrics)
            rows.append(row)
        
        if rows:
            df = pd.DataFrame(rows)
            
            # Reorder columns
            cols = ["Model", "Task"]
            metric_cols = sorted([c for c in df.columns if c not in cols])
            df = df[cols + metric_cols]
            
            # Check for existing summary and merge
            summary_path = output_dir / "summary_scores.csv"
            if summary_path.exists():
                try:
                    existing_df = pd.read_csv(summary_path)
                    # Remove duplicates
                    for _, new_row in df.iterrows():
                        mask = (existing_df["Model"] == new_row["Model"]) & (existing_df["Task"] == new_row["Task"])
                        existing_df = existing_df[~mask]
                    df = pd.concat([existing_df, df], ignore_index=True)
                except Exception as e:
                    print(f"Warning: Could not merge with existing CSV: {e}")
            
            df.to_csv(summary_path, index=False)
            print(f"\nSummary saved to: {summary_path}")


def list_mair_tasks() -> List[str]:
    """List all available MAIR tasks."""
    return ALL_MAIR_TASKS


def list_mair_categories() -> Dict[str, List[str]]:
    """List MAIR tasks grouped by category."""
    return MAIR_TASKS
