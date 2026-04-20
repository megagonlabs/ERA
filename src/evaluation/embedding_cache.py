"""
Embedding cache module for storing and retrieving precomputed embeddings.

This module provides functionality to cache embeddings to avoid redundant API calls
and speed up repeated evaluations with the same model and data.
"""

import hashlib
import json
import os
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np


class EmbeddingCache:
    """Cache for storing and retrieving embeddings."""
    
    def __init__(self, cache_dir: str = "cache/embeddings", enabled: bool = True, force_recache: bool = False):
        """
        Initialize the embedding cache.
        
        Args:
            cache_dir: Directory to store cached embeddings
            enabled: Whether caching is enabled
            force_recache: If True, ignore existing cache and always recompute (but still save to cache)
        """
        self.cache_dir = Path(cache_dir)
        self.enabled = enabled
        self.force_recache = force_recache
        if enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _compute_text_hash(self, texts: List[str]) -> str:
        """
        Compute a hash for a list of texts.
        
        Uses SHA256 hash of concatenated texts with a separator.
        Also includes the count and sample of texts to ensure uniqueness.
        """
        # Create a unique representation
        hash_input = {
            "count": len(texts),
            "first_5": texts[:5] if len(texts) >= 5 else texts,
            "last_5": texts[-5:] if len(texts) >= 5 else [],
            "sample_middle": texts[len(texts)//2:len(texts)//2+3] if len(texts) > 10 else [],
            # Full hash of all texts for exact matching
            "content_hash": hashlib.sha256(
                "\n===SEP===\n".join(texts).encode('utf-8')
            ).hexdigest()
        }
        
        hash_str = json.dumps(hash_input, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(hash_str.encode('utf-8')).hexdigest()[:32]
    
    def _get_cache_path(
        self, 
        model_name: str, 
        text_hash: str,
        task_name: Optional[str] = None,
        split: Optional[str] = None
    ) -> Path:
        """Get the cache file path for given parameters."""
        # Sanitize model name for filesystem
        safe_model_name = model_name.replace("/", "_").replace("\\", "_")
        
        # Build cache path
        cache_subdir = self.cache_dir / safe_model_name
        if task_name:
            cache_subdir = cache_subdir / task_name
        if split:
            cache_subdir = cache_subdir / split
            
        cache_subdir.mkdir(parents=True, exist_ok=True)
        return cache_subdir / f"{text_hash}.npz"
    
    def get(
        self,
        model_name: str,
        texts: List[str],
        task_name: Optional[str] = None,
        split: Optional[str] = None
    ) -> Optional[np.ndarray]:
        """
        Retrieve cached embeddings if available.
        
        Args:
            model_name: Name of the embedding model
            texts: List of texts that were embedded
            task_name: Optional task name for organization
            split: Optional split name (e.g., 'corpus', 'queries')
            
        Returns:
            Cached embeddings as numpy array, or None if not found
        """
        if not self.enabled:
            return None
        
        # If force_recache is enabled, skip loading from cache
        if self.force_recache:
            return None
            
        text_hash = self._compute_text_hash(texts)
        cache_path = self._get_cache_path(model_name, text_hash, task_name, split)
        
        if cache_path.exists():
            try:
                data = np.load(cache_path)
                cached_embeddings = data['embeddings']
                cached_count = int(data['count'])
                
                # Verify the count matches
                if cached_count == len(texts):
                    print(f"  [Cache HIT] Loaded {len(texts)} embeddings from {cache_path}")
                    return cached_embeddings
                else:
                    print(f"  [Cache MISS] Count mismatch: cached {cached_count}, requested {len(texts)}")
                    return None
            except Exception as e:
                print(f"  [Cache ERROR] Failed to load cache: {e}")
                return None
        else:
            # Debug: Show what we looked for
            print(f"  [Cache MISS] File not found: {cache_path}")
            # Show what files exist in that directory
            cache_dir = cache_path.parent
            if cache_dir.exists():
                existing_files = list(cache_dir.glob("*.npz"))
                print(f"  [Cache MISS] Found {len(existing_files)} cache file(s) in {cache_dir}")
                if existing_files:
                    print(f"  [Cache MISS] Existing hash: {existing_files[0].stem}")
                    print(f"  [Cache MISS] Expected hash: {text_hash}")
            else:
                print(f"  [Cache MISS] Directory does not exist: {cache_dir}")
        
        return None
    
    def put(
        self,
        model_name: str,
        texts: List[str],
        embeddings: np.ndarray,
        task_name: Optional[str] = None,
        split: Optional[str] = None
    ) -> bool:
        """
        Store embeddings in cache.
        
        Args:
            model_name: Name of the embedding model
            texts: List of texts that were embedded
            embeddings: Embeddings to cache
            task_name: Optional task name for organization
            split: Optional split name (e.g., 'corpus', 'queries')
            
        Returns:
            True if successfully cached, False otherwise
        """
        if not self.enabled:
            return False
            
        if len(texts) != len(embeddings):
            print(f"  [Cache ERROR] Text count ({len(texts)}) != embedding count ({len(embeddings)})")
            return False
            
        text_hash = self._compute_text_hash(texts)
        cache_path = self._get_cache_path(model_name, text_hash, task_name, split)
        
        try:
            np.savez_compressed(
                cache_path,
                embeddings=embeddings,
                count=len(texts)
            )
            print(f"  [Cache SAVE] Saved {len(texts)} embeddings to {cache_path}")
            return True
        except Exception as e:
            print(f"  [Cache ERROR] Failed to save cache: {e}")
            return False
    
    def clear(self, model_name: Optional[str] = None) -> int:
        """
        Clear cached embeddings.
        
        Args:
            model_name: If provided, only clear cache for this model.
                       If None, clear all cache.
                       
        Returns:
            Number of cache files deleted
        """
        count = 0
        if model_name:
            safe_model_name = model_name.replace("/", "_").replace("\\", "_")
            target_dir = self.cache_dir / safe_model_name
        else:
            target_dir = self.cache_dir
            
        if target_dir.exists():
            for cache_file in target_dir.rglob("*.npz"):
                try:
                    cache_file.unlink()
                    count += 1
                except Exception:
                    pass
                    
        print(f"Cleared {count} cache files")
        return count
    
    def get_cache_info(self) -> dict:
        """Get information about the cache."""
        info = {
            "cache_dir": str(self.cache_dir),
            "enabled": self.enabled,
            "models": {},
            "total_files": 0,
            "total_size_mb": 0.0
        }
        
        if not self.cache_dir.exists():
            return info
            
        for model_dir in self.cache_dir.iterdir():
            if model_dir.is_dir():
                model_files = list(model_dir.rglob("*.npz"))
                model_size = sum(f.stat().st_size for f in model_files) / (1024 * 1024)
                info["models"][model_dir.name] = {
                    "files": len(model_files),
                    "size_mb": round(model_size, 2)
                }
                info["total_files"] += len(model_files)
                info["total_size_mb"] += model_size
                
        info["total_size_mb"] = round(info["total_size_mb"], 2)
        return info
    
    def list_splits(self, model_name: str, task_name: Optional[str] = None) -> List[str]:
        """List available splits for a given model and task.
        
        Args:
            model_name: Name of the embedding model
            task_name: Optional task name
            
        Returns:
            List of available split names
        """
        safe_model_name = model_name.replace("/", "_").replace("\\", "_")
        base_dir = self.cache_dir / safe_model_name
        
        if task_name:
            base_dir = base_dir / task_name
        
        if not base_dir.exists():
            return []
        
        splits = []
        for item in base_dir.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                # Check if this directory contains .npz files
                if any(item.glob("*.npz")):
                    splits.append(item.name)
        
        return sorted(splits)
    
    def find_matching_split(self, model_name: str, task_name: Optional[str], 
                           texts: List[str], preferred_splits: List[str]) -> Optional[Tuple[str, np.ndarray]]:
        """Try to find embeddings in one of the preferred splits.
        
        Args:
            model_name: Name of the embedding model
            task_name: Optional task name
            texts: List of texts to look up
            preferred_splits: List of split names to try, in order of preference
            
        Returns:
            Tuple of (split_name, embeddings) if found, None otherwise
        """
        for split in preferred_splits:
            embeddings = self.get(model_name, texts, task_name=task_name, split=split)
            if embeddings is not None:
                return (split, embeddings)
        return None


# Global cache instance
_global_cache: Optional[EmbeddingCache] = None


def get_cache(cache_dir: str = "cache/embeddings", enabled: bool = True, force_recache: bool = False) -> EmbeddingCache:
    """Get or create the global embedding cache instance."""
    global _global_cache
    if _global_cache is None:
        _global_cache = EmbeddingCache(cache_dir=cache_dir, enabled=enabled, force_recache=force_recache)
    return _global_cache


def set_cache_enabled(enabled: bool):
    """Enable or disable the global cache."""
    global _global_cache
    if _global_cache is not None:
        _global_cache.enabled = enabled
