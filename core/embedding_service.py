"""
Embedding Service Module - Local Text Embedding Generation

This module provides text-to-vector conversion using Sentence Transformers
models that run entirely offline after initial download.
"""

import logging
from typing import List, Optional, Union
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer
import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Service for generating text embeddings using Sentence Transformers.
    
    Supports multiple models and operates completely offline after
    the model is downloaded.
    """
    
    # Predefined model configurations
    MODELS = {
        "mini": {
            "name": "all-MiniLM-L6-v2",
            "dimensions": 384,
            "size_mb": 23,
            "description": "Fast, compact model ideal for edge devices",
        },
        "base": {
            "name": "all-MiniLM-L12-v2",
            "dimensions": 384,
            "size_mb": 33,
            "description": "Balanced model with better quality",
        },
        "mpnet": {
            "name": "all-mpnet-base-v2",
            "dimensions": 768,
            "size_mb": 110,
            "description": "High-quality embeddings for desktop",
        },
        "multilingual": {
            "name": "paraphrase-multilingual-MiniLM-L12-v2",
            "dimensions": 384,
            "size_mb": 135,
            "description": "Support for 50+ languages",
        },
    }
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None,
        cache_folder: Optional[str] = None,
        max_seq_length: int = 256,
        normalize_embeddings: bool = True,
    ):
        """
        Initialize the embedding service.
        
        Args:
            model_name: Name of the sentence-transformers model or shorthand
            device: Device to use ('cpu', 'cuda', or None for auto-detect)
            cache_folder: Custom folder for model cache
            max_seq_length: Maximum sequence length for tokenization
            normalize_embeddings: Whether to L2 normalize embeddings
        """
        # Resolve model name
        if model_name in self.MODELS:
            self.model_name = self.MODELS[model_name]["name"]
            self.vector_size = self.MODELS[model_name]["dimensions"]
        else:
            self.model_name = model_name
            self.vector_size = None  # Will be determined after loading
        
        # Set device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        self.cache_folder = cache_folder
        self.max_seq_length = max_seq_length
        self.normalize_embeddings = normalize_embeddings
        
        # Model will be lazily loaded
        self.model = None
        self._model_loaded = False
        
        logger.info(f"Embedding service initialized: model={self.model_name}, device={self.device}")
    
    def _load_model(self):
        """Lazy load the model on first use."""
        if self._model_loaded:
            return
        
        logger.info(f"Loading model: {self.model_name}")
        
        self.model = SentenceTransformer(
            self.model_name,
            device=self.device,
            cache_folder=self.cache_folder,
        )
        
        # Set max sequence length
        self.model.max_seq_length = self.max_seq_length
        
        # Get vector size if not already set
        if self.vector_size is None:
            self.vector_size = self.model.get_sentence_embedding_dimension()
        
        self._model_loaded = True
        logger.info(f"Model loaded: {self.model_name} (dim={self.vector_size})")
    
    def encode(
        self,
        text: Union[str, List[str]],
        batch_size: int = 32,
        show_progress: bool = False,
        convert_to_numpy: bool = True,
    ) -> Union[List[float], np.ndarray, List[List[float]]]:
        """
        Generate embedding(s) for text.
        
        Args:
            text: Single text string or list of texts
            batch_size: Batch size for processing multiple texts
            show_progress: Show progress bar for large batches
            convert_to_numpy: Return numpy array instead of list
        
        Returns:
            Embedding vector(s)
        """
        self._load_model()
        
        # Handle single string
        single_input = isinstance(text, str)
        if single_input:
            text = [text]
        
        # Generate embeddings
        embeddings = self.model.encode(
            text,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
        )
        
        # Return format
        if single_input:
            embedding = embeddings[0]
            return embedding.tolist() if not convert_to_numpy else embedding
        else:
            return embeddings.tolist() if not convert_to_numpy else embeddings
    
    def encode_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> List[List[float]]:
        """
        Generate embeddings for a batch of texts (optimized).
        
        Args:
            texts: List of text strings
            batch_size: Batch size for processing
            show_progress: Show progress bar
        
        Returns:
            List of embedding vectors
        """
        return self.encode(
            texts,
            batch_size=batch_size,
            show_progress=show_progress,
            convert_to_numpy=False,
        )
    
    def similarity(
        self,
        embedding1: Union[List[float], np.ndarray],
        embedding2: Union[List[float], np.ndarray],
    ) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
        
        Returns:
            Similarity score (0-1 for normalized vectors)
        """
        # Convert to numpy if needed
        if isinstance(embedding1, list):
            embedding1 = np.array(embedding1)
        if isinstance(embedding2, list):
            embedding2 = np.array(embedding2)
        
        # Cosine similarity
        similarity = np.dot(embedding1, embedding2)
        
        # If not normalized, divide by magnitudes
        if not self.normalize_embeddings:
            similarity /= (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
        
        return float(similarity)
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings produced by this model.
        
        Returns:
            Embedding dimension
        """
        self._load_model()
        return self.vector_size
    
    def warmup(self, num_samples: int = 5):
        """
        Warm up the model with sample inputs to initialize CUDA kernels.
        
        Args:
            num_samples: Number of warmup samples
        """
        self._load_model()
        logger.info("Warming up model...")
        
        sample_texts = [f"Sample text {i} for model warmup" for i in range(num_samples)]
        self.encode(sample_texts, show_progress=False)
        
        logger.info("Model warmup complete")
    
    @staticmethod
    def list_available_models() -> dict:
        """
        List all predefined models with their characteristics.
        
        Returns:
            Dictionary of model configurations
        """
        return EmbeddingService.MODELS
    
    def get_model_info(self) -> dict:
        """
        Get information about the currently loaded model.
        
        Returns:
            Dictionary with model info
        """
        self._load_model()
        
        return {
            "model_name": self.model_name,
            "embedding_dimension": self.vector_size,
            "max_seq_length": self.max_seq_length,
            "device": self.device,
            "normalize_embeddings": self.normalize_embeddings,
            "model_loaded": self._model_loaded,
        }
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"EmbeddingService("
            f"model={self.model_name}, "
            f"dim={self.vector_size}, "
            f"device={self.device})"
        )


class MultiModelEmbeddingService:
    """
    Service that manages multiple embedding models simultaneously.
    
    Useful for hybrid search or A/B testing different models.
    """
    
    def __init__(self, models_config: List[dict]):
        """
        Initialize with multiple models.
        
        Args:
            models_config: List of model configurations
                Example: [{"name": "mini", "alias": "fast"}, ...]
        """
        self.services = {}
        
        for config in models_config:
            model_name = config.get("name")
            alias = config.get("alias", model_name)
            
            self.services[alias] = EmbeddingService(
                model_name=model_name,
                device=config.get("device"),
                cache_folder=config.get("cache_folder"),
            )
        
        logger.info(f"Multi-model service initialized with {len(self.services)} models")
    
    def encode(self, text: Union[str, List[str]], model_alias: str = None) -> Union[List[float], List[List[float]]]:
        """
        Encode text using specified model.
        
        Args:
            text: Text to encode
            model_alias: Which model to use (uses first if not specified)
        
        Returns:
            Embedding vector(s)
        """
        if model_alias is None:
            model_alias = list(self.services.keys())[0]
        
        if model_alias not in self.services:
            raise ValueError(f"Model alias '{model_alias}' not found")
        
        return self.services[model_alias].encode(text)
    
    def encode_all(self, text: Union[str, List[str]]) -> dict:
        """
        Encode text using all models.
        
        Args:
            text: Text to encode
        
        Returns:
            Dictionary mapping model aliases to embeddings
        """
        results = {}
        for alias, service in self.services.items():
            results[alias] = service.encode(text)
        return results
