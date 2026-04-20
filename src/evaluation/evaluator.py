import os
import mteb
import json
import pandas as pd
from pathlib import Path
from typing import List, Optional, Any, Dict
import numpy as np
import torch
from ..models.base import BaseEmbedder
from .embedding_cache import EmbeddingCache, get_cache


class CachingGenericModelWrapper:
    """
    Wrapper to make models compatible with MTEB 2.x and handle caching.
    Implements the EncoderProtocol interface.
    """
    def __init__(
        self, 
        model: BaseEmbedder, 
        is_api_model: bool = False,
        cache_enabled: bool = True,
        cache_dir: str = "cache/embeddings",
        force_recache: bool = False
    ):
        self._model = model
        self._is_api_model = is_api_model
        self.model_name = getattr(model, 'model_name', 'custom_model')
        self.max_seq_length = 8192
        self.truncate_dim = None
        
        # Initialize cache
        self._cache = EmbeddingCache(cache_dir=cache_dir, enabled=cache_enabled, force_recache=force_recache)
        self._current_task = None  # Set by Evaluator before each task
        self._current_split = None  # 'corpus' or 'queries'
    
    def set_task_context(self, task_name: str, split: Optional[str] = None):
        """Set the current task context for cache organization."""
        self._current_task = task_name
        self._current_split = split
    
    def encode(
        self, 
        sentences, 
        *args,
        **kwargs
    ) -> np.ndarray:
        """Encode method that MTEB will call."""
        from torch.utils.data import DataLoader
        
        batch_size = kwargs.get('batch_size', 32)
        # Limit batch size for API models to avoid token limits
        if self._is_api_model:
            batch_size = min(batch_size, 100)
        
        instruction = kwargs.get("instruction", None)
        prompt_type = kwargs.get("prompt_type", None)
        
        # Update split context if provided by MTEB
        task_name = kwargs.get('task_name', None)
        if task_name and task_name != self._current_task:
            self._current_task = task_name
        
        # Check if this is corpus or queries encoding
        # MTEB may pass this info in different ways
        if prompt_type:
            if str(prompt_type).lower() in ['query', 'queries']:
                self._current_split = 'queries'
            elif str(prompt_type).lower() in ['corpus', 'document', 'passage']:
                self._current_split = 'corpus'
        
        if prompt_type and str(prompt_type).lower() == 'query':
            instruction = instruction or 'Represent this query for retrieving relevant documents'
        
        # Handle DataLoader input from MTEB 2.x
        if isinstance(sentences, DataLoader):
            # Collect all texts first
            all_texts = []
            for batch in sentences:
                if isinstance(batch, dict):
                    texts = batch.get('text', batch.get('sentence', []))
                    if not isinstance(texts, list):
                        texts = [texts]
                elif isinstance(batch, (list, tuple)):
                    texts = [str(t) if not isinstance(t, str) else t for t in batch]
                else:
                    texts = [str(batch)]
                all_texts.extend(texts)
            
            if not all_texts:
                return np.array([])
            
            # Encode all texts (with incremental caching handled internally)
            embeddings = self._encode_texts(all_texts, batch_size, instruction)
            
            return embeddings
        
        # Handle list/tuple input
        if hasattr(sentences, '__iter__') and not isinstance(sentences, (str, list, np.ndarray)):
            sentences = list(sentences)
        
        # Ensure it's a list
        if isinstance(sentences, np.ndarray):
            sentences = sentences.tolist()
        if isinstance(sentences, str):
            sentences = [sentences]
        
        # Encode (with incremental caching handled internally)
        embeddings = self._encode_texts(sentences, batch_size, instruction)
        
        return embeddings
    
    def _encode_texts(self, texts: List[str], batch_size: int, instruction: Optional[str]) -> np.ndarray:
        """Internal method to encode texts with incremental caching."""
        import gc
        
        # Process in chunks for incremental caching
        cache_save_interval = 2000  # Save cache every 2000 embeddings
        all_embeddings = []
        
        # Calculate total number of chunks
        total_chunks = (len(texts) + cache_save_interval - 1) // cache_save_interval
        
        # Display split info
        split_info = f" [{self._current_split}]" if self._current_split else ""
        
        for chunk_idx, chunk_start in enumerate(range(0, len(texts), cache_save_interval), 1):
            chunk_end = min(chunk_start + cache_save_interval, len(texts))
            chunk_texts = texts[chunk_start:chunk_end]
            
            print(f"  [Progress{split_info}] Processing chunk {chunk_idx}/{total_chunks} ({len(chunk_texts)} embeddings)")
            
            # Check if this chunk is already cached
            cached_chunk = self._cache.get(
                self.model_name,
                chunk_texts,
                task_name=self._current_task,
                split=self._current_split
            )
            
            if cached_chunk is not None:
                chunk_embeddings = cached_chunk
            else:
                # Encode this chunk
                chunk_embeddings = self._model.encode(chunk_texts, batch_size=batch_size, instruction=instruction)
                if not isinstance(chunk_embeddings, np.ndarray):
                    chunk_embeddings = np.array(chunk_embeddings)
                
                # Save this chunk to cache
                self._cache.put(
                    self.model_name,
                    chunk_texts,
                    chunk_embeddings,
                    task_name=self._current_task,
                    split=self._current_split
                )
                
                # Clear GPU memory after encoding each chunk to prevent OOM
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
            
            all_embeddings.append(chunk_embeddings)
        
        # Combine all chunks
        if all_embeddings:
            return np.vstack(all_embeddings)
        return np.array([])
    
    def similarity(self, embeddings1: np.ndarray, embeddings2: np.ndarray) -> torch.Tensor:
        """Compute cosine similarity between two sets of embeddings."""
        if isinstance(embeddings1, np.ndarray):
            embeddings1 = torch.from_numpy(embeddings1).float()
        if isinstance(embeddings2, np.ndarray):
            embeddings2 = torch.from_numpy(embeddings2).float()
        
        embeddings1 = embeddings1 / embeddings1.norm(dim=-1, keepdim=True)
        embeddings2 = embeddings2 / embeddings2.norm(dim=-1, keepdim=True)
        
        return torch.mm(embeddings1, embeddings2.t())
    
    def similarity_pairwise(self, embeddings1: np.ndarray, embeddings2: np.ndarray) -> torch.Tensor:
        """Compute pairwise cosine similarity."""
        if isinstance(embeddings1, np.ndarray):
            embeddings1 = torch.from_numpy(embeddings1).float()
        if isinstance(embeddings2, np.ndarray):
            embeddings2 = torch.from_numpy(embeddings2).float()
        
        embeddings1 = embeddings1 / embeddings1.norm(dim=-1, keepdim=True)
        embeddings2 = embeddings2 / embeddings2.norm(dim=-1, keepdim=True)
        
        return (embeddings1 * embeddings2).sum(dim=-1)
    
    @property
    def mteb_model_meta(self):
        """Return model metadata."""
        from mteb.models import ModelMeta
        return ModelMeta(
            loader=None,
            name=f"custom/{self.model_name}",
            revision="1.0",
            release_date=None,
            languages=None,
            n_parameters=None,
            memory_usage_mb=None,
            max_tokens=8192,
            embed_dim=None,
            license=None,
            open_weights=False,
            public_training_code=None,
            public_training_data=None,
            framework=[],
            similarity_fn_name="cosine",
            use_instructions=False,
            training_datasets=None,
        )

class Evaluator:
    def __init__(
        self, 
        model: BaseEmbedder, 
        experiment_name: str = "zero-shot",
        cache_enabled: bool = True,
        cache_dir: str = "cache/embeddings",
        force_recache: bool = False
    ):
        self.model = model
        self.experiment_name = experiment_name
        self.cache_enabled = cache_enabled
        self.cache_dir = cache_dir
        self.force_recache = force_recache
        # Try to retrieve model name for better logging structure
        self.model_name = getattr(model, "model_name", "unknown_model")
        # Sanitize model name for filesystem (e.g. "openai/text-embedding" -> "openai__text-embedding")
        # Add experiment name to distinguish different runs
        safe_base_name = self.model_name.replace("/", "__").replace("models__", "")
        self.safe_model_name = f"{safe_base_name}__{experiment_name}"

    def run(self, tasks: List[str], output_folder: str = "results", batch_size: int = 32):
        """
        Run MTEB evaluation.
        """
        # Create experiment and model specific output directory
        # Structure: results/{experiment_name}/{model_name}/
        base_output_dir = Path(output_folder)
        experiment_dir = base_output_dir / self.experiment_name
        model_output_dir = experiment_dir / self.model_name.replace("/", "__").replace("models__", "")
        model_output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Results will be saved to: {model_output_dir}")

        # MTEB automatically handles loading of standard tasks.
        # For custom tasks, we would register them before calling MTEB(tasks=...).
        
        # In MTEB 2.x, we often load tasks via mteb.get_tasks or pass task objects/names differently if using old MTEB class wrapper.
        # But MTEB(tasks=...) is typically still supported or we should use get_tasks.
        # However, the error 'MTEB object has no attribute tasks' suggests internal change in how tasks are stored or accessed.
        
        # Let's use the new way to load tasks explicitly if just strings are passed
        if isinstance(tasks[0], str):
            tasks_objs = mteb.get_tasks(tasks=tasks)
        else:
            tasks_objs = tasks

        # Determine which model representation to pass to MTEB
        from ..models.wrappers import LocalHFEmbedder, OpenAIEmbedder
        
        # Check if the model is an API model (for batch size limits)
        is_api = isinstance(self.model, OpenAIEmbedder)
        
        # Always wrap the model to enable caching and standard MTEB 2.x interface
        wrapped_model = CachingGenericModelWrapper(
            self.model,
            is_api_model=is_api,
            cache_enabled=self.cache_enabled,
            cache_dir=self.cache_dir,
            force_recache=self.force_recache
        )
        model_to_evaluate = wrapped_model
        
        # Save raw MTEB outputs to a subfolder to keep root clean
        mteb_raw_dir = model_output_dir / "mteb_raw"
        
        # Run each task individually to set cache context
        all_results = []
        for task in tasks_objs:
            task_name = task.metadata.name if hasattr(task, 'metadata') else str(task)
            print(f"\n--- Running task: {task_name} ---")
            
            # Set cache context for the wrapped model
            if wrapped_model is not None:
                wrapped_model.set_task_context(task_name)
            
            evaluation = mteb.MTEB(tasks=[task])
            results = evaluation.run(
                model_to_evaluate, 
                output_folder=str(mteb_raw_dir),
                batch_size=batch_size,
                overwrite_results=True,
            )
            all_results.extend(results)

        # Custom result processing
        self._save_custom_results(all_results, model_output_dir)
        
        return all_results

    def _save_custom_results(self, results, output_dir: Path):
        """
        Parse MTEB results and save them in a user-friendly format.
        Structure:
          output_dir/
            summary_scores.csv
            tasks/
              {TaskName}/
                full_results.json
                key_metrics.json
        """
        summary_data = []
        tasks_dir = output_dir / "tasks"
        tasks_dir.mkdir(exist_ok=True)

        for result in results:
            # result is typically an MTEBResult object
            task_name = result.task_name
            
            # Ensure we have a dictionary of scores
            # MTEBResult usually has a 'scores' attribute which is a dict keyed by split (test, evaluation, etc.)
            scores_dict = getattr(result, "scores", {})
            
            # Prepare task directory
            task_folder = tasks_dir / task_name
            task_folder.mkdir(exist_ok=True)
            
            # 1. Save Full Results (JSON)
            # Try to serialize the whole result object or fallback to scores
            full_data = {
                "task_name": task_name,
                "model_name": self.model_name,
                "scores": scores_dict,
                "evaluation_time": getattr(result, "evaluation_time", 0)
            }
            
            with open(task_folder / "full_results.json", "w") as f:
                # Use default=str to handle non-serializable objects like numpy types if any
                json.dump(full_data, f, indent=2, default=str)

            # 2. Extract Key Metrics
            # We prioritize 'test' split, then 'validation', then others
            target_split = "test"
            if target_split not in scores_dict:
                # Find first available split that is not 'train' ideally
                for split in scores_dict.keys():
                    if split != "train":
                        target_split = split
                        break
                else:
                    # Fallback to whatever is there
                    target_split = next(iter(scores_dict.keys())) if scores_dict else None

            if not target_split:
                print(f"Warning: No scores found for task {task_name}")
                continue

            split_scores = scores_dict[target_split]
            
            # Unwrap list if needed (MTEB 2.x often returns a list of scores)
            if isinstance(split_scores, list):
                if len(split_scores) > 0:
                    split_scores = split_scores[0]
                else:
                    split_scores = {}
            
            # Common retrieval metrics to extract
            interest_metrics = [
                "ndcg_at_1", "ndcg_at_3", "ndcg_at_5", "ndcg_at_10", "ndcg_at_20", "ndcg_at_100",
                "map_at_1", "map_at_10", "map_at_100",
                "recall_at_1", "recall_at_10", "recall_at_100",
                "mrr_at_1", "mrr_at_10", "mrr_at_100",
                "precision_at_1", "precision_at_10",
                "main_score" # MTEB often computes a main score
            ]
            
            # Filter metrics
            # Note: split_scores might be a list of results if multiple sub-tasks exist,
            # but usually for retrieval it's a dict of metrics.
            # If it is a list (e.g. clustering), this logic might need adaptation. 
            # Assuming standard retrieval dict structure here.
            
            key_metrics = {}
            if isinstance(split_scores, dict):
                for k, v in split_scores.items():
                    if k in interest_metrics:
                        key_metrics[k] = v
                    # Also include any key that looks like a main indicator
                    if "main" in k: 
                        key_metrics[k] = v
            else:
                # If structure is different, just save raw
                key_metrics = {"raw_score": split_scores}

            with open(task_folder / "key_metrics.json", "w") as f:
                json.dump(key_metrics, f, indent=2, default=str)

            # 3. Add to Summary
            summary_entry = {
                "Model": self.model_name,
                "Task": task_name,
                "Split": target_split
            }
            # Flatten metrics into columns
            # Handle list nesting if necessary (not doing complex flattening here)
            for k, v in key_metrics.items():
                if isinstance(v, (int, float, str)):
                    summary_entry[k] = v
            
            summary_data.append(summary_entry)

        # 4. Save Summary CSV with merging logic
        if summary_data:
            new_df = pd.DataFrame(summary_data)
            summary_csv_path = output_dir / "summary_scores.csv"
            
            # Load existing CSV if it exists
            if summary_csv_path.exists():
                try:
                    existing_df = pd.read_csv(summary_csv_path)
                    
                    # Merge: Remove old entries for the same Model+Task+Split combination
                    # then append new results
                    merge_keys = ["Model", "Task", "Split"]
                    
                    # Create a key to identify duplicates
                    if all(k in existing_df.columns for k in merge_keys):
                        # Remove rows that match the new data's Model+Task+Split
                        for _, new_row in new_df.iterrows():
                            mask = True
                            for key in merge_keys:
                                mask = mask & (existing_df[key] == new_row[key])
                            existing_df = existing_df[~mask]
                        
                        # Combine old and new data
                        df = pd.concat([existing_df, new_df], ignore_index=True)
                    else:
                        # If merge keys don't exist, just append
                        df = pd.concat([existing_df, new_df], ignore_index=True)
                    
                    print(f"Merged new results with existing summary")
                except Exception as e:
                    print(f"Warning: Could not read existing CSV, creating new one. Error: {e}")
                    df = new_df
            else:
                df = new_df
            
            # Reorder columns to put Model, Task, Split first
            first_cols = ["Model", "Task", "Split"]
            other_cols = [c for c in df.columns if c not in first_cols]
            # specific sort for metrics if possible, e.g. ndcg_10
            # simple sort for now
            other_cols.sort()
            
            final_cols = first_cols + other_cols
            # ensure all exist
            final_cols = [c for c in final_cols if c in df.columns]
            
            df = df[final_cols]
            df.to_csv(summary_csv_path, index=False)
            print(f"Summary scores saved to: {summary_csv_path}")

