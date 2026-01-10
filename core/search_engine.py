"""
Search Engine Module - High-Level Search Interface

This module provides a unified search interface that orchestrates
the document processor, embedding service, and vector store.
"""

import logging
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass

from .vector_store import VectorStore
from .embedding_service import EmbeddingService

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Container for search results."""
    text: str
    score: float
    metadata: Dict[str, Any]
    id: str
    
    def __repr__(self) -> str:
        return f"SearchResult(score={self.score:.4f}, text='{self.text[:50]}...')"


class SearchEngine:
    """
    High-level search engine interface.
    
    Provides semantic search, hybrid search, and filtered search capabilities.
    """
    
    def __init__(
        self,
        vector_store: VectorStore,
        embedding_service: EmbeddingService,
    ):
        """
        Initialize search engine.
        
        Args:
            vector_store: Vector store instance
            embedding_service: Embedding service instance
        """
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        
        logger.info("Search engine initialized")
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        score_threshold: Optional[float] = None,
        return_raw: bool = False,
    ) -> Union[List[SearchResult], List[Dict[str, Any]]]:
        """
        Perform semantic search.
        
        Args:
            query: Search query text
            top_k: Number of results to return
            filters: Optional metadata filters
            score_threshold: Minimum similarity score
            return_raw: Return raw dictionaries instead of SearchResult objects
        
        Returns:
            List of search results, sorted by relevance
        """
        logger.debug(f"Search query: '{query}' (top_k={top_k})")
        
        # Preprocess query
        processed_query = self._preprocess_query(query)
        
        # Generate query embedding
        query_vector = self.embedding_service.encode(processed_query)
        
        # Search in vector store
        raw_results = self.vector_store.search(
            query_vector=query_vector,
            top_k=top_k,
            filters=filters,
            score_threshold=score_threshold,
        )
        
        if return_raw:
            return raw_results
        
        # Convert to SearchResult objects
        results = [
            SearchResult(
                id=r["id"],
                score=r["score"],
                text=r["payload"].get("text", ""),
                metadata=r["payload"],
            )
            for r in raw_results
        ]
        
        logger.debug(f"Search returned {len(results)} results")
        return results
    
    def search_with_rerank(
        self,
        query: str,
        initial_k: int = 50,
        final_k: int = 10,
        diversity_lambda: float = 0.5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """
        Search with diversity re-ranking (Maximal Marginal Relevance).
        
        Args:
            query: Search query text
            initial_k: Number of initial results to fetch
            final_k: Number of final results after re-ranking
            diversity_lambda: Trade-off between relevance and diversity (0-1)
            filters: Optional metadata filters
        
        Returns:
            Re-ranked search results
        """
        # Get initial results
        initial_results = self.search(
            query=query,
            top_k=initial_k,
            filters=filters,
            return_raw=False,
        )
        
        if len(initial_results) <= final_k:
            return initial_results
        
        # Apply MMR re-ranking
        reranked = self._mmr_rerank(
            query=query,
            results=initial_results,
            k=final_k,
            lambda_param=diversity_lambda,
        )
        
        logger.debug(f"Re-ranked {len(initial_results)} to {len(reranked)} results")
        return reranked
    
    def _mmr_rerank(
        self,
        query: str,
        results: List[SearchResult],
        k: int,
        lambda_param: float,
    ) -> List[SearchResult]:
        """
        Maximal Marginal Relevance re-ranking for diversity.
        
        Args:
            query: Original query
            results: Initial results
            k: Number of results to select
            lambda_param: Weight for relevance vs diversity
        
        Returns:
            Re-ranked results
        """
        import numpy as np
        
        # Get query embedding
        query_embedding = np.array(self.embedding_service.encode(query))
        
        # Get result embeddings
        result_texts = [r.text for r in results]
        result_embeddings = np.array([
            self.embedding_service.encode(text)
            for text in result_texts
        ])
        
        # Calculate relevance scores (cosine similarity with query)
        relevance_scores = np.array([
            self.embedding_service.similarity(query_embedding, emb)
            for emb in result_embeddings
        ])
        
        # MMR algorithm
        selected_indices = []
        remaining_indices = list(range(len(results)))
        
        for _ in range(min(k, len(results))):
            if not remaining_indices:
                break
            
            mmr_scores = []
            for idx in remaining_indices:
                # Relevance component
                relevance = relevance_scores[idx]
                
                # Diversity component (max similarity to already selected)
                if selected_indices:
                    selected_embeddings = result_embeddings[selected_indices]
                    max_similarity = max(
                        self.embedding_service.similarity(
                            result_embeddings[idx],
                            selected_emb
                        )
                        for selected_emb in selected_embeddings
                    )
                else:
                    max_similarity = 0
                
                # MMR score
                mmr = lambda_param * relevance - (1 - lambda_param) * max_similarity
                mmr_scores.append((idx, mmr))
            
            # Select best MMR score
            best_idx, best_score = max(mmr_scores, key=lambda x: x[1])
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)
        
        # Return selected results in order
        return [results[idx] for idx in selected_indices]
    
    def hybrid_search(
        self,
        query: str,
        keywords: Optional[List[str]] = None,
        top_k: int = 10,
        alpha: float = 0.7,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """
        Hybrid search combining semantic and keyword matching.
        
        Args:
            query: Search query text
            keywords: Optional explicit keywords for keyword search
            top_k: Number of results to return
            alpha: Weight for semantic search (0=pure keyword, 1=pure semantic)
            filters: Optional metadata filters
        
        Returns:
            Hybrid search results
        """
        # Semantic search
        semantic_results = self.search(
            query=query,
            top_k=top_k * 2,  # Get more candidates
            filters=filters,
            return_raw=False,
        )
        
        # If keywords provided, perform keyword filtering/boosting
        if keywords:
            keyword_boost = self._keyword_boost(semantic_results, keywords)
            
            # Combine scores
            for i, result in enumerate(semantic_results):
                result.score = (
                    alpha * result.score +
                    (1 - alpha) * keyword_boost.get(result.id, 0)
                )
            
            # Re-sort by combined score
            semantic_results.sort(key=lambda x: x.score, reverse=True)
        
        return semantic_results[:top_k]
    
    def _keyword_boost(
        self,
        results: List[SearchResult],
        keywords: List[str],
    ) -> Dict[str, float]:
        """
        Calculate keyword match scores for results.
        
        Args:
            results: Search results
            keywords: Keywords to match
        
        Returns:
            Dictionary mapping result IDs to keyword scores
        """
        scores = {}
        
        for result in results:
            text_lower = result.text.lower()
            keyword_count = sum(
                text_lower.count(keyword.lower())
                for keyword in keywords
            )
            
            # Normalize by text length
            score = keyword_count / (len(result.text) / 100)
            scores[result.id] = min(score, 1.0)  # Cap at 1.0
        
        # Normalize scores to 0-1 range
        if scores:
            max_score = max(scores.values())
            if max_score > 0:
                scores = {k: v / max_score for k, v in scores.items()}
        
        return scores
    
    def search_by_id(self, vector_id: str) -> Optional[SearchResult]:
        """
        Retrieve a specific result by vector ID.
        
        Args:
            vector_id: ID of the vector
        
        Returns:
            SearchResult if found, None otherwise
        """
        # Note: This requires adding a get_by_id method to VectorStore
        logger.warning("search_by_id not fully implemented - requires VectorStore.get_by_id")
        return None
    
    def _preprocess_query(self, query: str) -> str:
        """
        Preprocess query text.
        
        Args:
            query: Raw query text
        
        Returns:
            Preprocessed query
        """
        # Basic preprocessing
        query = query.strip()
        
        # Remove extra whitespace
        query = " ".join(query.split())
        
        return query
    
    def explain_results(
        self,
        results: List[SearchResult],
        query: str,
        top_n: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Explain why results were returned.
        
        Args:
            results: Search results
            query: Original query
            top_n: Number of results to explain
        
        Returns:
            List of explanations
        """
        explanations = []
        
        for result in results[:top_n]:
            # Simple explanation based on keyword matching
            query_words = set(query.lower().split())
            text_words = set(result.text.lower().split())
            
            matching_words = query_words & text_words
            
            explanation = {
                "result_id": result.id,
                "score": result.score,
                "text_preview": result.text[:200] + "..." if len(result.text) > 200 else result.text,
                "matching_keywords": list(matching_words),
                "metadata": result.metadata,
            }
            
            explanations.append(explanation)
        
        return explanations
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get search engine statistics.
        
        Returns:
            Dictionary with statistics
        """
        vector_count = self.vector_store.count()
        model_info = self.embedding_service.get_model_info()
        
        return {
            "total_vectors": vector_count,
            "embedding_model": model_info["model_name"],
            "embedding_dimension": model_info["embedding_dimension"],
            "collection_name": self.vector_store.collection_name,
        }
    
    def __repr__(self) -> str:
        """String representation."""
        stats = self.get_statistics()
        return (
            f"SearchEngine("
            f"vectors={stats['total_vectors']}, "
            f"model={stats['embedding_model']})"
        )


class BatchIndexer:
    """
    Utility class for batch indexing documents.
    """
    
    def __init__(
        self,
        vector_store: VectorStore,
        embedding_service: EmbeddingService,
        batch_size: int = 32,
    ):
        """
        Initialize batch indexer.
        
        Args:
            vector_store: Vector store instance
            embedding_service: Embedding service instance
            batch_size: Batch size for embedding generation
        """
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        self.batch_size = batch_size
        
        logger.info(f"Batch indexer initialized (batch_size={batch_size})")
    
    def index_chunks(
        self,
        chunks: List[Dict[str, Any]],
        show_progress: bool = True,
    ) -> List[str]:
        """
        Index multiple document chunks.
        
        Args:
            chunks: List of chunks from DocumentProcessor
            show_progress: Show progress bar
        
        Returns:
            List of vector IDs
        """
        if not chunks:
            logger.warning("No chunks to index")
            return []
        
        # Extract texts and prepare payloads
        texts = [chunk["text"] for chunk in chunks]
        payloads = [chunk["metadata"] for chunk in chunks]
        
        # Add text to payloads
        for i, payload in enumerate(payloads):
            payload["text"] = texts[i]
        
        # Generate embeddings in batch
        logger.info(f"Generating embeddings for {len(texts)} chunks...")
        embeddings = self.embedding_service.encode_batch(
            texts,
            batch_size=self.batch_size,
            show_progress=show_progress,
        )
        
        # Insert into vector store
        logger.info(f"Inserting {len(embeddings)} vectors...")
        vector_ids = self.vector_store.insert_batch(
            vectors=embeddings,
            payloads=payloads,
        )
        
        logger.info(f"Indexed {len(vector_ids)} chunks successfully")
        return vector_ids
