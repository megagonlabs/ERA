"""
Adapter-based embedder for asymmetric query-side adaptation.

This module provides an embedder that applies a trained adapter to query embeddings
while keeping document embeddings unchanged (from the small model).
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Dict
import sys

import numpy as np
import torch
from torch import nn

from ..cache_config import EMBEDDING_CACHE_DIR

# Ensure src is in path for imports
_src_path = Path(__file__).parent.parent
if str(_src_path) not in sys.path:
    sys.path.insert(0, str(_src_path))

from ..models.base import BaseEmbedder
from ..evaluation.embedding_cache import EmbeddingCache


def _instruction_cache_splits(instruction: str, per_query: bool = False) -> List[str]:
    """Return the only cache splits allowed for instruction-aware queries."""
    import hashlib

    inst_hash = hashlib.md5(instruction.encode("utf-8")).hexdigest()[:8]
    if per_query:
        return [
            f"queries_perquery_inst_{inst_hash}",
            f"queries_with_instruction_{inst_hash}",
        ]
    return [f"queries_with_instruction_{inst_hash}"]


def _instruction_cache_error_prefix(component: str, task_name: str) -> str:
    return (
        f"[{component}] No instruction-aware query cache found for task '{task_name}'. "
        "Instruction-enabled evaluation refuses to read instruction-free query caches."
    )


class LinearAdapter(nn.Module):
    """Simple linear projection adapter with L2 normalization."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.proj(x)
        return torch.nn.functional.normalize(output, p=2, dim=-1)



class AdaptedQueryEmbedder(BaseEmbedder):
    """
    Cache-based embedder that uses large model + adapter for queries and small model for documents.
    
    Architecture:
    - Documents: Small model cache (e.g., 0.6B) - reused directly
    - Queries: Large model cache (e.g., 8B) → Linear adapter → Small model space
    
    This embedder DOES NOT load models. It only reads from existing caches.
    """

    ADAPTER_FORWARD_BATCH_SIZE = 4096

    def __init__(
        self,
        large_model_name: str,
        small_model_name: str,
        adapter_path: str,
        cache_dir: str = EMBEDDING_CACHE_DIR,
        description: str = "",
    ):
        super().__init__()
        
        self.large_model_name = large_model_name
        self.small_model_name = small_model_name
        self.adapter_path = adapter_path
        self.cache_dir = cache_dir
        
        # Initialize cache
        self.cache = EmbeddingCache(cache_dir=cache_dir, enabled=True, force_recache=False)
        
        # Load adapter
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        self.use_data_parallel = self.num_gpus > 1
        self.adapter = self._load_adapter(adapter_path)
        self.adapter.to(self.device)
        if self.use_data_parallel:
            self.adapter = torch.nn.DataParallel(self.adapter)
            print(f"[Adapter] Multi-GPU enabled for retrieval eval: {self.num_gpus} GPUs")
        self.adapter.eval()
        
        # Set model_name for identification
        self.model_name = f"adapted_{large_model_name.replace('/', '_')}_to_{small_model_name.replace('/', '_')}"
        if description:
            self.model_name += f"_{description}"
        
        # Cache model names for retrieving pre-existing embeddings
        # Queries: use large model cache (before adapter)
        # Corpus: use small model cache (no adapter needed)
        self.cache_model_name_for_queries = large_model_name.replace('/', '_')
        self.cache_model_name_for_corpus = small_model_name.replace('/', '_')
        
        self._is_encoding_queries = False
        self._current_task = None
        
        # Cache for full embeddings (to handle bucketed encoding)
        self._full_query_cache = {}
        self._full_corpus_cache = {}

    def _adapt_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Apply the adapter in batches, using all visible GPUs when available."""
        if embeddings is None or len(embeddings) == 0:
            return np.array([])

        adapted_batches = []
        batch_size = self.ADAPTER_FORWARD_BATCH_SIZE * max(1, self.num_gpus)

        with torch.no_grad():
            for start in range(0, len(embeddings), batch_size):
                batch = torch.as_tensor(
                    embeddings[start:start + batch_size],
                    dtype=torch.float32,
                    device=self.device,
                )
                adapted = self.adapter(batch)
                adapted = adapted / adapted.norm(dim=-1, keepdim=True)
                adapted_batches.append(adapted.cpu().numpy())

        return np.concatenate(adapted_batches, axis=0)

    def _load_adapter(self, adapter_path: str) -> nn.Module:
        """Load trained linear adapter from checkpoint."""
        checkpoint = torch.load(adapter_path, map_location=self.device)

        if "proj.weight" not in checkpoint:
            raise ValueError(f"Expected linear adapter checkpoint (with 'proj.weight'), got keys: {list(checkpoint.keys())}")

        weight = checkpoint["proj.weight"]
        out_dim, in_dim = weight.shape

        adapter = LinearAdapter(in_dim, out_dim)
        adapter.load_state_dict(checkpoint)
        print(f"Loaded Linear adapter: {in_dim} -> {out_dim}")
        return adapter

    def _pre_process(self, sentences: List[str], instruction: Optional[str]):
        """Detect if encoding queries or corpus based on instruction presence."""
        # Heuristic: if instruction is provided, we're likely encoding queries
        self._is_encoding_queries = (instruction is not None)
        return sentences, instruction
    
    def load_full_cache(
        self,
        task_name: str,
        all_query_texts: List[str],
        all_corpus_texts: List[str],
        instruction: Optional[str] = None,
        per_query_instructions: Optional[List[Optional[str]]] = None,
    ):
        """Pre-load full caches for queries and corpus before bucketed encoding.
        
        Args:
            task_name: MAIR task name
            all_query_texts: List of all query texts
            all_corpus_texts: List of all corpus texts
            instruction: Single instruction for all queries (if per_query_instructions is None)
            per_query_instructions: List of instructions aligned with all_query_texts
        """
        
        if task_name and not task_name.startswith('MAIR_'):
            task_name = f"MAIR_{task_name}"
        
        # Load query cache with per-query instructions support
        if per_query_instructions:
            print(f"  [Adapter] Loading caches for per-query instructions...")
            normalized_instructions = [inst if inst is not None else "" for inst in per_query_instructions]
            
            # First, check what splits are available
            available_splits = self.cache.list_splits(self.large_model_name, task_name)
            print(f"  [Adapter] Available splits: {available_splits[:10] if len(available_splits) > 10 else available_splits}")
            if len(available_splits) > 10:
                print(f"  [Adapter] ... and {len(available_splits) - 10} more")
            
            # Group queries by instruction (aligned list to avoid collisions on duplicate texts)
            from collections import defaultdict
            instruction_groups = defaultdict(list)
            for query_text, inst in zip(all_query_texts, normalized_instructions):
                instruction_groups[inst].append(query_text)
            
            print(f"  [Adapter] Found {len(instruction_groups)} unique instruction groups")
            
            # Load cache for each instruction group
            self._full_query_cache = {}
            total_loaded = 0
            failed_instructions = []
            
            for inst, query_texts_for_inst in instruction_groups.items():
                # Build candidate split names
                candidate_splits = _instruction_cache_splits(inst, per_query=True)
                
                # Try to find embeddings in one of the candidate splits
                result = self.cache.find_matching_split(
                    self.large_model_name,
                    task_name,
                    query_texts_for_inst,
                    candidate_splits
                )
                
                if result is not None:
                    found_split, query_embeddings = result
                    adapted = self._adapt_embeddings(query_embeddings)
                    
                    # Store with (text, instruction) key to avoid collisions
                    for i, text in enumerate(query_texts_for_inst):
                        self._full_query_cache[(text, inst)] = adapted[i]
                    total_loaded += len(query_texts_for_inst)
                    
                    print(f"    [✓] Loaded {len(query_texts_for_inst)} queries from split '{found_split}'")
                    if inst and len(inst) > 60:
                        print(f"      Instruction: {inst[:60]}...")
                    elif inst:
                        print(f"      Instruction: {inst}")
                else:
                    failed_instructions.append({
                        'instruction': inst,
                        'num_queries': len(query_texts_for_inst),
                        'tried_splits': candidate_splits,
                        'first_query': query_texts_for_inst[0][:80]
                    })
                    print(f"    [✗] Cache not found for {len(query_texts_for_inst)} queries")
                    print(f"      Instruction: {inst[:80] if inst else 'None'}...")
                    print(f"      Tried splits: {candidate_splits}")
                    print(f"      First query: {query_texts_for_inst[0][:80]}...")
            
            print(f"  [Adapter] Loaded and adapted {total_loaded}/{len(all_query_texts)} queries from cache")
            if total_loaded < len(all_query_texts):
                missing = len(all_query_texts) - total_loaded
                raise RuntimeError(
                    f"{_instruction_cache_error_prefix('Adapter', task_name)} "
                    f"Missing {missing} queries across {len(failed_instructions)} instruction groups. "
                    f"Create per-query instruction caches first (expected splits like "
                    f"{failed_instructions[0]['tried_splits'] if failed_instructions else 'queries_perquery_inst_<hash>'})."
                )
        
        elif instruction:
            print(f"  [Adapter] Loading cache for single instruction...")
            
            # Check available splits
            available_splits = self.cache.list_splits(self.large_model_name, task_name)
            print(f"  [Adapter] Available splits: {available_splits[:10] if len(available_splits) > 10 else available_splits}")
            
            # Build candidate split names
            candidate_splits = _instruction_cache_splits(instruction)
            
            # Try to find embeddings
            result = self.cache.find_matching_split(
                self.large_model_name,
                task_name,
                all_query_texts,
                candidate_splits
            )
            
            if result is not None:
                found_split, query_embeddings = result
                adapted = self._adapt_embeddings(query_embeddings)
                
                # Store as mapping from (text, instruction) to embedding
                self._full_query_cache = {}
                for i, text in enumerate(all_query_texts):
                    self._full_query_cache[(text, instruction)] = adapted[i]
                
                print(f"  [Adapter] [✓] Loaded and adapted {len(self._full_query_cache)} queries from split '{found_split}'")
            else:
                raise RuntimeError(
                    f"{_instruction_cache_error_prefix('Adapter', task_name)} "
                    f"Tried splits: {candidate_splits}. "
                    "Run the large-model evaluation with instructions enabled first."
                )
        else:
            print(f"  [Adapter] Loading queries without instructions...")
            
            # Check available splits
            available_splits = self.cache.list_splits(self.large_model_name, task_name)
            print(f"  [Adapter] Available splits: {available_splits[:10] if len(available_splits) > 10 else available_splits}")
            
            split = "queries"
            
            query_embeddings = self.cache.get(
                self.large_model_name,
                all_query_texts,
                task_name=task_name,
                split=split
            )
            
            if query_embeddings is not None:
                adapted = self._adapt_embeddings(query_embeddings)
                
                # Store as mapping from text to embedding
                self._full_query_cache = {text: adapted[i] for i, text in enumerate(all_query_texts)}
                print(f"  [Adapter] [✓] Loaded and adapted {len(self._full_query_cache)} queries from split '{split}'")
            else:
                print(f"  [Adapter] WARNING: Cache not found for queries in split '{split}'")
                print(f"  [Adapter] Available splits: {available_splits}")
                print(f"  [Adapter] Please run the large model evaluation first to create cache")
        
        # Load corpus cache
        print(f"  [Adapter] Loading corpus cache (small model)...")
        available_corpus_splits = self.cache.list_splits(self.small_model_name, task_name)
        print(f"  [Adapter] Available corpus splits: {available_corpus_splits[:10] if len(available_corpus_splits) > 10 else available_corpus_splits}")
        
        corpus_embeddings = self.cache.get(
            self.small_model_name,
            all_corpus_texts,
            task_name=task_name,
            split="corpus"
        )
        
        if corpus_embeddings is not None:
            self._full_corpus_cache = {text: corpus_embeddings[i] for i, text in enumerate(all_corpus_texts)}
            print(f"  [Adapter] [✓] Loaded {len(self._full_corpus_cache)} corpus embeddings from cache")
        else:
            print(f"  [Adapter] WARNING: Corpus cache not found")
            print(f"  [Adapter] Available corpus splits: {available_corpus_splits}")
            print(f"  [Adapter] Please run the small model evaluation first to create corpus cache")
    
    def _encode_implementation(
        self,
        sentences: List[str],
        batch_size: int,
        instruction: Optional[str],
        **kwargs
    ) -> np.ndarray:
        """Encode texts using pre-loaded caches."""
        
        # Smart detection: check which cache contains the first text
        if sentences:
            first_text = sentences[0]
            query_key = (first_text, instruction) if instruction is not None else first_text
            is_query = query_key in self._full_query_cache or first_text in self._full_query_cache
            is_corpus = first_text in self._full_corpus_cache
            
            if is_query:
                # Use query cache
                embeddings = []
                for text in sentences:
                    key = (text, instruction) if instruction is not None else text
                    if key in self._full_query_cache:
                        embeddings.append(self._full_query_cache[key])
                    elif instruction is None and text in self._full_query_cache:
                        embeddings.append(self._full_query_cache[text])
                    else:
                        print(f"ERROR: Text not found in query cache!")
                        print(f"  Text: {text[:100]}")
                        print(f"  Query cache size: {len(self._full_query_cache)}")
                        print(f"  Instruction: {instruction[:80] if instruction else 'None'}")
                        print(f"  Sample cache keys (first 3): {list(self._full_query_cache.keys())[:3]}")
                        raise RuntimeError(f"Text not found in pre-loaded query cache: {text[:100]}")
                return np.array(embeddings)
            elif is_corpus:
                # Use corpus cache
                embeddings = []
                for text in sentences:
                    if text in self._full_corpus_cache:
                        embeddings.append(self._full_corpus_cache[text])
                    else:
                        print(f"ERROR: Text not found in corpus cache!")
                        print(f"  Text: {text[:100]}")
                        print(f"  Corpus cache size: {len(self._full_corpus_cache)}")
                        raise RuntimeError(f"Text not found in pre-loaded corpus cache: {text[:100]}")
                return np.array(embeddings)
            else:
                print(f"ERROR: Text not found in any pre-loaded cache!")
                print(f"  Text: {first_text[:100]}")
                print(f"  Query cache size: {len(self._full_query_cache)}")
                print(f"  Corpus cache size: {len(self._full_corpus_cache)}")
                print(f"  Sample query cache keys (first 3): {list(self._full_query_cache.keys())[:3] if self._full_query_cache else []}")
                print(f"  Sample corpus cache keys (first 3): {list(self._full_corpus_cache.keys())[:3] if self._full_corpus_cache else []}")
                raise RuntimeError(f"Text not found in any pre-loaded cache: {first_text[:100]}")
        
        return np.array([])


class NoAdapterEmbedder(BaseEmbedder):
    """
    Cache-based embedder that uses the small model for BOTH queries and documents.
    No adapter transformation is applied to queries.

    This serves as the "no adapter" baseline: it produces results equivalent to
    running the small model directly on both queries and corpus, but reads from
    pre-computed caches without loading the actual model.

    Note: large_model_name must equal small_model_name (validated at construction).
    """

    def __init__(
        self,
        model_name: str,
        cache_dir: str = EMBEDDING_CACHE_DIR,
        description: str = "",
    ):
        super().__init__()

        self.small_model_name = model_name
        self.large_model_name = model_name  # same model, no adapter
        self.cache_dir = cache_dir

        # Initialize file-based cache
        self.cache = EmbeddingCache(cache_dir=cache_dir, enabled=True, force_recache=False)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # adapter = None signals mair_evaluator to skip transformation
        self.adapter = None

        safe = model_name.replace("/", "_")
        self.model_name = f"no_adapter_{safe}"
        if description:
            self.model_name += f"_{description}"

        # Both queries and corpus use the same small model cache
        self.cache_model_name_for_queries = model_name.replace("/", "_")
        self.cache_model_name_for_corpus = model_name.replace("/", "_")

        self._full_query_cache: Dict[object, np.ndarray] = {}
        self._full_corpus_cache: Dict[str, np.ndarray] = {}

    def load_full_cache(
        self,
        task_name: str,
        all_query_texts: List[str],
        all_corpus_texts: List[str],
        instruction: Optional[str] = None,
        per_query_instructions: Optional[List[Optional[str]]] = None,
    ):
        """Pre-load small model caches for queries and corpus (no adapter applied)."""
        import hashlib
        from collections import defaultdict

        if task_name and not task_name.startswith("MAIR_"):
            task_name = f"MAIR_{task_name}"

        model_key = self.small_model_name.replace("/", "_")

        # ── Query cache ──────────────────────────────────────────
        if per_query_instructions:
            print(f"  [NoAdapter] Loading per-query instruction caches...")
            normalized = [i if i is not None else "" for i in per_query_instructions]
            groups: dict = defaultdict(list)
            for text, inst in zip(all_query_texts, normalized):
                groups[inst].append(text)

            self._full_query_cache = {}
            total_loaded = 0
            for inst, texts_for_inst in groups.items():
                candidates = _instruction_cache_splits(inst, per_query=True)

                result = self.cache.find_matching_split(model_key, task_name, texts_for_inst, candidates)
                if result is not None:
                    found_split, embs = result
                    for i, text in enumerate(texts_for_inst):
                        self._full_query_cache[(text, inst)] = embs[i]
                    total_loaded += len(texts_for_inst)
                    print(f"    [✓] {len(texts_for_inst)} queries from split '{found_split}'")
                else:
                    raise RuntimeError(
                        f"{_instruction_cache_error_prefix('NoAdapter', task_name)} "
                        f"Missing cache for {len(texts_for_inst)} queries. Tried splits: {candidates}."
                    )
            print(f"  [NoAdapter] Loaded {total_loaded}/{len(all_query_texts)} query embeddings")

        elif instruction:
            print(f"  [NoAdapter] Loading cache for single instruction...")
            candidates = _instruction_cache_splits(instruction)

            result = self.cache.find_matching_split(model_key, task_name, all_query_texts, candidates)
            if result is not None:
                found_split, embs = result
                self._full_query_cache = {}
                for i, text in enumerate(all_query_texts):
                    self._full_query_cache[(text, instruction)] = embs[i]
                print(f"  [NoAdapter] [✓] Loaded {len(embs)} query embeddings from '{found_split}'")
            else:
                raise RuntimeError(
                    f"{_instruction_cache_error_prefix('NoAdapter', task_name)} "
                    f"Tried splits: {candidates}."
                )

        else:
            embs = self.cache.get(model_key, all_query_texts, task_name=task_name, split="queries")
            if embs is not None:
                self._full_query_cache = {text: embs[i] for i, text in enumerate(all_query_texts)}
                print(f"  [NoAdapter] [✓] Loaded {len(embs)} query embeddings from 'queries'")
            else:
                print(f"  [NoAdapter] WARNING: Query cache not found for split 'queries'")

        # ── Corpus cache ─────────────────────────────────────────
        print(f"  [NoAdapter] Loading corpus cache...")
        corpus_embs = self.cache.get(model_key, all_corpus_texts, task_name=task_name, split="corpus")
        if corpus_embs is not None:
            self._full_corpus_cache = {text: corpus_embs[i] for i, text in enumerate(all_corpus_texts)}
            print(f"  [NoAdapter] [✓] Loaded {len(corpus_embs)} corpus embeddings")
        else:
            print(f"  [NoAdapter] WARNING: Corpus cache not found")

    def _encode_implementation(
        self,
        sentences: List[str],
        batch_size: int,
        instruction: Optional[str],
        **kwargs,
    ) -> np.ndarray:
        """Return cached small model embeddings without transformation."""
        if not sentences:
            return np.array([])
        first = sentences[0]
        query_key = (first, instruction) if instruction is not None else first
        if query_key in self._full_query_cache or first in self._full_query_cache:
            result = []
            for text in sentences:
                key = (text, instruction) if instruction is not None else text
                if key in self._full_query_cache:
                    result.append(self._full_query_cache[key])
                elif instruction is None and text in self._full_query_cache:
                    result.append(self._full_query_cache[text])
                else:
                    raise RuntimeError(f"Text not found in no-adapter query cache: {text[:100]}")
            return np.array(result)
        elif first in self._full_corpus_cache:
            result = []
            for text in sentences:
                if text not in self._full_corpus_cache:
                    raise RuntimeError(f"Text not found in no-adapter corpus cache: {text[:100]}")
                result.append(self._full_corpus_cache[text])
            return np.array(result)
        else:
            raise RuntimeError(
                f"Text not found in any no-adapter cache: {first[:100]}\n"
                f"Query cache size: {len(self._full_query_cache)}, "
                f"Corpus cache size: {len(self._full_corpus_cache)}"
            )
