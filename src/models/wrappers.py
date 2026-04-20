import os
from typing import List, Optional, Any, Union
import numpy as np
import torch
from tenacity import retry, stop_after_attempt, wait_exponential

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    import tiktoken
except ImportError:
    tiktoken = None

from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer

from .base import BaseEmbedder

class OpenAIEmbedder(BaseEmbedder):
    """Wrapper for OpenAI Embeddings API."""
    
    # Maximum tokens per text for each model
    MODEL_MAX_TOKENS = {
        "text-embedding-3-small": 8191,
        "text-embedding-3-large": 8191,
        "text-embedding-ada-002": 8191,
    }
    
    def __init__(self, model_name: str = "text-embedding-3-large", **kwargs):
        super().__init__(**kwargs)
        if OpenAI is None:
            raise ImportError("openai library is not installed.")
        
        self.model_name = model_name
        self.api_key = os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            print("Warning: OPENAI_API_KEY not found in environment variables.")
        self.client = OpenAI(api_key=self.api_key)
        
        # Initialize tokenizer for truncation
        self._tokenizer = None
        self._max_tokens = self.MODEL_MAX_TOKENS.get(model_name, 8191)
    
    def _get_tokenizer(self):
        """Get or initialize the tiktoken tokenizer."""
        if self._tokenizer is None:
            if tiktoken is None:
                print("Warning: tiktoken not installed. Text truncation will use character-based estimation.")
                return None
            try:
                self._tokenizer = tiktoken.encoding_for_model(self.model_name)
            except KeyError:
                # Fall back to cl100k_base which is used by text-embedding-3-* models
                self._tokenizer = tiktoken.get_encoding("cl100k_base")
        return self._tokenizer
    
    def _truncate_text(self, text: str, max_tokens: int = None) -> str:
        """Truncate text to fit within the token limit."""
        if max_tokens is None:
            max_tokens = self._max_tokens
        
        tokenizer = self._get_tokenizer()
        if tokenizer is None:
            # Fallback: estimate ~4 characters per token on average
            max_chars = max_tokens * 4
            if len(text) > max_chars:
                return text[:max_chars]
            return text
        
        tokens = tokenizer.encode(text, disallowed_special=())
        if len(tokens) <= max_tokens:
            return text
        
        # Truncate tokens and decode back to text
        truncated_tokens = tokens[:max_tokens]
        return tokenizer.decode(truncated_tokens)

    def _encode_batch_with_retry(self, batch: List[str]) -> List[List[float]]:
        """Encode a single batch with retry logic."""
        response = self.client.embeddings.create(
            model=self.model_name,
            input=batch,
            encoding_format="float"
        )
        if not response.data:
            return []
        return [data.embedding for data in response.data]

    def _encode_implementation(
        self, sentences: List[str], batch_size: int, instruction: Optional[str], **kwargs
    ) -> np.ndarray:
        # Ensure sentences is a proper list of strings
        if isinstance(sentences, np.ndarray):
            sentences = sentences.tolist()
        if not isinstance(sentences, list):
            sentences = list(sentences)
        # Convert any non-string elements to strings
        sentences = [str(s) if not isinstance(s, str) else s for s in sentences]
        
        # OpenAI models typically handle instructions implicitly via the input text
        # If an instruction is provided, we prepend it.
        processed_sentences = sentences
        if instruction:
            processed_sentences = [f"{instruction}\n{s}" for s in sentences]
        
        # Filter out empty strings (OpenAI doesn't accept them), keeping track of original indices
        non_empty_indices = [i for i, s in enumerate(processed_sentences) if s.strip()]
        non_empty_sentences = [processed_sentences[i] for i in non_empty_indices]
        if not non_empty_sentences:
            return np.array([])
        
        # Truncate texts that exceed the token limit
        non_empty_sentences = [self._truncate_text(s) for s in non_empty_sentences]

        # OpenAI has a token limit of 300,000 per request
        # Use a smaller batch size to stay safe (estimated ~500 tokens/text average)
        # Max safe batch size: ~300000/500 = 600, but we use much smaller to be safe
        max_api_batch = min(batch_size, 100)  # Conservative limit
        
        all_embeddings = []
        for i in range(0, len(non_empty_sentences), max_api_batch):
            batch = non_empty_sentences[i : i + max_api_batch]
            if not batch:
                continue
            
            # Use adaptive batch size - if batch fails, split in half
            try:
                batch_embeddings = self._encode_batch_safe(batch)
                all_embeddings.extend(batch_embeddings)
            except Exception as e:
                print(f"Error encoding batch at index {i}: {e}")
                raise
        
        all_embeddings_array = np.array(all_embeddings)
        
        # If some sentences were empty, restore full-length output with zeros at empty positions
        if len(non_empty_indices) < len(processed_sentences):
            full_result = np.zeros((len(processed_sentences), all_embeddings_array.shape[1]), dtype=np.float32)
            for out_idx, orig_idx in enumerate(non_empty_indices):
                full_result[orig_idx] = all_embeddings_array[out_idx]
            return full_result
        
        return all_embeddings_array
    
    def _is_token_limit_error(self, error_str: str) -> bool:
        """Check if the error is related to token limits."""
        lower = error_str.lower()
        return (
            "max_tokens_per_request" in lower
            or "context_length_exceeded" in lower
            or "maximum context length" in lower
            or "reduce your text" in lower
            or "too many tokens" in lower
        )

    @retry(wait=wait_exponential(multiplier=1, min=2, max=60), stop=stop_after_attempt(5))
    def _encode_batch_safe(self, batch: List[str]) -> List[List[float]]:
        """Encode batch with adaptive splitting if token limit exceeded."""
        try:
            return self._encode_batch_with_retry(batch)
        except Exception as e:
            error_str = str(e)
            if self._is_token_limit_error(error_str):
                if len(batch) > 1:
                    # Split batch in half to isolate the offending text(s)
                    mid = len(batch) // 2
                    left = self._encode_batch_safe(batch[:mid])
                    right = self._encode_batch_safe(batch[mid:])
                    return left + right
                else:
                    # Single item still exceeds limit; re-truncate to half and retry once
                    truncated = self._truncate_text(batch[0], self._max_tokens // 2)
                    return self._encode_batch_with_retry([truncated])
            raise


class PooledModel(torch.nn.Module):
    """Wrapper that performs pooling inside the forward pass to save memory in DataParallel."""
    def __init__(self, model):
        super().__init__()
        self.module = model
        
    def forward(self, **batch_dict):
        outputs = self.module(**batch_dict)
        
        # Perform last token pooling
        last_hidden_states = outputs.last_hidden_state
        attention_mask = batch_dict['attention_mask']
        
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            embeddings = last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            embeddings = last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]
            
        return embeddings

class LocalHFEmbedder(BaseEmbedder):
    """
    Wrapper for local Hugging Face models (BGE-M3, NV-Embed, Qwen).
    Handles device placement and specific formatting.
    Supports multi-GPU via DataParallel when multiple GPUs are available.
    
    Data Parallel Support:
    - Automatically detects multiple GPUs via CUDA_VISIBLE_DEVICES
    - Wraps model with nn.DataParallel for parallel batch processing
    - Effective batch size = batch_size * num_gpus
    - Significantly speeds up large model inference (e.g., 8B models)
    
    Precision:
    - Models >= 8B use bfloat16 to avoid FP16 overflow/NaN issues
    - Smaller models use float16 for efficiency
    """
    def __init__(self, model_name_or_path: str, use_fp16: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name_or_path
        self.use_fp16 = use_fp16
        
        # Determine if model is large (>= 8B) - use bfloat16 to avoid NaN issues
        self.is_large_model = self._is_large_model(model_name_or_path)
        
        # Check for multi-GPU setup
        self.num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        self.use_data_parallel = self.num_gpus > 1
        
        print(f"Loading {model_name_or_path} on {self.device}...")
        if self.use_data_parallel:
            print(f"🚀 Multi-GPU enabled: {self.num_gpus} GPUs detected for Data Parallel")
            print(f"   Effective batch size will be: batch_size × {self.num_gpus}")
        
        # Specific logic for NV-Embed handling as it requires trust_remote_code
        self.is_nv_embed = "NV-Embed" in model_name_or_path
        self.is_qwen = "Qwen" in model_name_or_path
        self.is_bge = "bge" in model_name_or_path.lower()

        if self.is_nv_embed or self.is_qwen:
            attn_implementation = None
            if self.is_qwen:
                # Try to use flash attention if available
                try:
                    import flash_attn
                    attn_implementation = "flash_attention_2"
                    print("✓ Using flash_attention_2 for Qwen model")
                except ImportError:
                    print("Warning: flash_attn not installed. Skipping flash_attention_2.")

            # Determine dtype: use bfloat16 for large models (>= 8B) to avoid NaN
            if use_fp16 and self.device == "cuda":
                if self.is_large_model:
                    model_dtype = torch.bfloat16
                    print(f"✓ Using bfloat16 for large model (>= 8B) to avoid FP16 overflow")
                else:
                    model_dtype = torch.float16
                    print(f"✓ Using float16 for model")
            else:
                model_dtype = torch.float32
            
            # NV-Embed and Qwen often need specific model loading code implementation
            self.model = AutoModel.from_pretrained(
                model_name_or_path, 
                trust_remote_code=True,
                attn_implementation=attn_implementation,
                torch_dtype=model_dtype
            )
            self.model.to(self.device)
            
            # Enable DataParallel for multi-GPU
            if self.use_data_parallel:
                if self.is_qwen:
                    # For Qwen (and large models), wrap with pooling logic to avoid gathering HUGE hidden states
                    self.model = PooledModel(self.model)
                    self.model = torch.nn.DataParallel(self.model)
                    print(f"✓ Model wrapped with PooledModel + DataParallel across {self.num_gpus} GPUs")
                else:
                    self.model = torch.nn.DataParallel(self.model)
                    print(f"✓ Model wrapped with DataParallel across {self.num_gpus} GPUs")
            
            if self.is_qwen:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)

        else:
            # Fallback to SentenceTransformers for BGE and standard models
            self.model = SentenceTransformer(
                model_name_or_path, 
                trust_remote_code=True,
                device=self.device
            )
            if self.is_bge:
                self.model.max_seq_length = 8192
                print(f"  [BGE] Set max_seq_length to {self.model.max_seq_length}")

    def _is_large_model(self, model_name: str) -> bool:
        """
        Check if model is large (>= 8B parameters) based on model name.
        Large models use bfloat16 to avoid FP16 overflow/NaN issues.
        
        Args:
            model_name: Model name or path
            
        Returns:
            True if model is >= 8B parameters
        """
        import re
        name_lower = model_name.lower()
        
        # Extract size patterns like "8b", "8B", "70b", "72b", etc.
        # Match patterns: 8b, 8B, 8-b, 8_b, etc.
        size_patterns = re.findall(r'(\d+\.?\d*)[_-]?b', name_lower)
        
        for size_str in size_patterns:
            try:
                size = float(size_str)
                if size >= 8:
                    return True
            except ValueError:
                continue
        
        return False

    def _encode_implementation(
        self, sentences: List[str], batch_size: int, instruction: Optional[str], **kwargs
    ) -> np.ndarray:
        
        # Adjust batch size for multi-GPU: multiply by number of GPUs
        # DataParallel will automatically split batches across GPUs
        effective_batch_size = batch_size * self.num_gpus if self.use_data_parallel else batch_size
        
        if self.use_data_parallel:
            print(f"  [Data Parallel] Using effective batch_size={effective_batch_size} ({batch_size} × {self.num_gpus} GPUs)")
        
        if self.is_nv_embed:
            return self._encode_nv_embed(sentences, effective_batch_size, instruction, **kwargs)

        if self.is_qwen:
            return self._encode_qwen(sentences, effective_batch_size, instruction, **kwargs)
        
        # Standard SentenceTransformer logic (BGE-M3 etc.)
        input_texts = sentences
        if instruction and self.is_bge:
            # BGE specific formatting: "Instruct: {instruction}\nQuery: {query}"
            input_texts = [f"Instruct: {instruction}\nQuery: {s}" for s in sentences]
        
        # Handle max_length for SentenceTransformer dynamically
        original_max_len = None
        if 'max_length' in kwargs and kwargs['max_length']:
            original_max_len = self.model.max_seq_length
            dataset_max_len = int(kwargs['max_length'])
            # Use original max length as cap if it exists and is reasonable (<1M), otherwise assume 8192 default
            cap = original_max_len if original_max_len and original_max_len < 1_000_000 else 8192
            
            new_len = min(dataset_max_len, cap)
            self.model.max_seq_length = new_len
            print(f"  [ST] Using dynamic max_seq_length={new_len} (Dataset: {dataset_max_len}, Cap: {cap})")

        embeddings = self.model.encode(
            input_texts,
            batch_size=effective_batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        if original_max_len:
            self.model.max_seq_length = original_max_len
            
        return embeddings

    def _encode_nv_embed(
        self, sentences: List[str], batch_size: int, instruction: Optional[str], **kwargs
    ) -> np.ndarray:
        """
        Specific encode logic for NV-Embed-v2 due to instruction formatting requirements.
        """
        # NV-Embed v2 typically expects: output = model.encode(texts, instruction=instruction)
        
        self.model.eval()
        
        with torch.no_grad():
            # If the model has a built-in encode method (most recent NV-Embeds do in their remote code)
            if hasattr(self.model, 'encode'):
                # IMPORTANT: NV-Embed-v2 expects `instruction` arg.
                # If instruction is None, it acts as passage (no prefix).
                embeddings = self.model.encode(
                    sentences, 
                    instruction=instruction if instruction else "", 
                    max_length=1024,
                    batch_size=batch_size
                )
                
                # Check if tensor or numpy
                if isinstance(embeddings, torch.Tensor):
                    embeddings = embeddings.cpu().float().numpy()
                return embeddings
                
            else:
                # Fallback implementation if no encode method (unlikely for NV-Embed official repo)
                # But if so, standard mean pooling etc. would be needed.
                raise NotImplementedError("Loaded NV-Embed model does not expose an 'encode' method.")

    def _last_token_pool(self, last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            # Handle case where sequence_lengths might be -1 if empty, but usually attention mask handles it
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

    def _encode_qwen(self, sentences: List[str], batch_size: int, instruction: Optional[str], **kwargs) -> np.ndarray:
        import gc
        
        self.model.eval()
        all_embeddings = []
        
        # Prepend instruction if provided using Qwen3 recommended format
        processed_sentences = sentences
        if instruction:
            # Qwen3 recommended format: "Instruct: {instruction}\nQuery: {query}"
            processed_sentences = [f"Instruct: {instruction}\nQuery: {s}" for s in sentences]

        # Get base model for config access (unwrap DataParallel and PooledModel if needed)
        # Structure can be: model, DataParallel(model), DataParallel(PooledModel(model))
        base_model = self.model
        if isinstance(base_model, torch.nn.DataParallel):
            base_model = base_model.module
        if isinstance(base_model, PooledModel):
            base_model = base_model.module
        
        # Determine max length for tokenization
        model_cap = self.tokenizer.model_max_length
        # Handle cases where tokenizer max length is not set or excessively large
        if model_cap > 1_000_000:
            if hasattr(base_model, 'config') and hasattr(base_model.config, "max_position_embeddings"):
                model_cap = base_model.config.max_position_embeddings
            else:
                model_cap = 32768  # Fallback to a reasonably large value for Qwen
        
        # Apply dynamic max length if provided
        dataset_max_len = kwargs.get('max_length')
        if dataset_max_len:
            max_len = min(int(dataset_max_len), model_cap)
            print(f"  [Qwen] Using dynamic max_length={max_len} (Dataset: {dataset_max_len}, Model Cap: {model_cap})")
        else:
            max_len = model_cap
            print(f"  [Qwen] Using max_length={max_len} for tokenization")

        for i in range(0, len(processed_sentences), batch_size):
            batch = processed_sentences[i : i + batch_size]
            
            # Tokenize using the model's max length
            batch_dict = self.tokenizer(
                batch,
                max_length=max_len,
                padding=True,
                truncation=True,
                return_tensors='pt'
            ).to(self.device)
            
            outputs = None
            with torch.no_grad():
                # Use DataParallel model if available for automatic parallelization
                if self.use_data_parallel:
                    # Logic is now inside PooledModel wrapper
                    embeddings = self.model(**batch_dict)
                else:
                    outputs = base_model(**batch_dict)
                    embeddings = self._last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
                
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                all_embeddings.append(embeddings.cpu().float().numpy())
            
            # Clear GPU memory after each batch
            del batch_dict, embeddings
            if outputs is not None:
                del outputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
                
        if all_embeddings:
            return np.concatenate(all_embeddings, axis=0)
        return np.array([])

