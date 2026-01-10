"""
On-Device AI Framework - Core Package

This package contains the core components for the On-Device AI Framework
using Qdrant Embedded Vector Search.
"""

__version__ = "1.0.0"
__author__ = "On-Device AI Framework Team"

from .vector_store import VectorStore
from .embedding_service import EmbeddingService
from .document_processor import DocumentProcessor
from .search_engine import SearchEngine

__all__ = [
    "VectorStore",
    "EmbeddingService",
    "DocumentProcessor",
    "SearchEngine",
]
