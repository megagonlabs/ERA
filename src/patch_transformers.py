
import transformers.cache_utils
import torch

def patch_dynamic_cache():
    # Check if DynamicCache needs patching for get_usable_length
    if hasattr(transformers.cache_utils, "DynamicCache"):
        cls = transformers.cache_utils.DynamicCache
        if not hasattr(cls, "get_usable_length"):
            print("Patching transformers.cache_utils.DynamicCache.get_usable_length")
            
            def get_usable_length(self, input_seq_length, layer_idx=None):
                # In older transformers, this returned valid length for attention.
                # For DynamicCache (growing cache), it is effectively the current cache length.
                # But typically this is called with 'input_seq_length' (current chunk).
                # The total length is what matters for position encoding usually.
                # Check implementation of Cache.get_usable_length
                
                # Default implementation acts as if all tokens are usable
                # The total length is cache_length + input_length (if not yet updated)
                # But typical usage in modeling_*.py is:
                # kv_seq_len = past_key_values.get_usable_length(seq_length)
                # where seq_length is input_ids.shape[1]
                
                # If the cache is already updated, get_seq_length returns the full length.
                # If not, we might need to be careful. 
                # However, usually get_usable_length was calling self.get_seq_length(layer_idx)
                
                # Handling layer_idx=None which causes TypeError in recent transformers
                if layer_idx is None:
                    layer_idx = 0
                    
                return self.get_seq_length(layer_idx)

            cls.get_usable_length = get_usable_length

    # Patch MistralAttention to handle missing position_embeddings (compatibility with NV-Embed-v2)
    try:
        from transformers.models.mistral import modeling_mistral
        
        if not hasattr(modeling_mistral.MistralAttention, "_original_forward"):
            modeling_mistral.MistralAttention._original_forward = modeling_mistral.MistralAttention.forward

            def patched_forward(self, hidden_states, position_embeddings=None, *args, **kwargs):
                # If position_embeddings is None, we compute it using position_ids
                if position_embeddings is None:
                    # Initialize rotary_emb if missing
                    if not hasattr(self, "rotary_emb"):
                        self.rotary_emb = modeling_mistral.MistralRotaryEmbedding(self.config)
                        self.rotary_emb.to(hidden_states.device)
                        # Ensure buffers are on correct device
                        if hasattr(self.rotary_emb, "inv_freq"):
                            self.rotary_emb.inv_freq = self.rotary_emb.inv_freq.to(hidden_states.device)

                    # Get position_ids from kwargs (passed by DecoderLayer)
                    position_ids = kwargs.get("position_ids", None)
                    
                    if position_ids is None:
                        # Fallback: create default position_ids
                        seq_len = hidden_states.shape[1]
                        position_ids = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0)
                    
                    # Compute position_embeddings
                    position_embeddings = self.rotary_emb(hidden_states, position_ids)
                
                return self._original_forward(hidden_states, position_embeddings, *args, **kwargs)
            
            print("Patching transformers.models.mistral.modeling_mistral.MistralAttention.forward")
            modeling_mistral.MistralAttention.forward = patched_forward

    except ImportError:
        print("Could not import Mistral models for patching. Usage of NV-Embed-v2 might fail if transformers version is new.")
    except Exception as e:
        print(f"Failed to patch MistralAttention: {e}")


