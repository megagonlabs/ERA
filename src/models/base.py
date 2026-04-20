import abc
from typing import List, Union, Optional, Tuple
import numpy as np

class BaseEmbedder(abc.ABC):
    """
    Base class for all embedding models.
    Designed to support future extensions like Adapters or NUDGE mechanisms.
    """

    def __init__(self, **kwargs):
        pass

    def encode(
        self, 
        sentences: List[str], 
        batch_size: int = 32, 
        instruction: Optional[str] = None,
        **kwargs
    ) -> Union[List[np.ndarray], np.ndarray]:
        """
        Main entry point for encoding texts.
        
        Args:
            sentences: List of utility texts to encode.
            batch_size: Batch size for inference.
            instruction: Instruction string (e.g., "Retrieve relevant documents...").
            
        Returns:
            Embeddings as numpy array or list.
        """
        # Hook for future pre-processing (e.g., modifying queries)
        sentences, instruction = self._pre_process(sentences, instruction)

        # Core embedding logic (Model specific)
        embeddings = self._encode_implementation(sentences, batch_size, instruction, **kwargs)

        # Hook for future post-processing (e.g., Adapter layers, dimensionality reduction)
        embeddings = self._post_process(embeddings)

        return embeddings

    @abc.abstractmethod
    def _encode_implementation(
        self, 
        sentences: List[str], 
        batch_size: int, 
        instruction: Optional[str],
        **kwargs
    ) -> np.ndarray:
        """Implementation of the specific model inference."""
        pass

    def _pre_process(self, sentences: List[str], instruction: Optional[str]) -> Tuple[List[str], Optional[str]]:
        """
        Future hook: Modifying input text or handling instruction formatting before the model.
        """
        return sentences, instruction

    def _post_process(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Future hook: Applying adapters, linear projections, etc.
        """
        return embeddings
