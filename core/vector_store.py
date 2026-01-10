"""
Vector Store Module - Qdrant Embedded Integration

This module provides a high-level interface for interacting with Qdrant
in embedded mode for vector storage and similarity search.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from uuid import uuid4

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    Range,
    SearchRequest,
    QuantizationConfig,
    ScalarQuantization,
    ScalarType,
    HnswConfigDiff,
)

logger = logging.getLogger(__name__)


class VectorStore:
    """
    Vector Store manager using Qdrant in embedded mode.
    
    Provides methods for:
    - Creating and managing collections
    - Inserting vectors with metadata
    - Searching for similar vectors
    - Updating and deleting vectors
    """
    
    def __init__(
        self,
        storage_path: str = "./qdrant_storage",
        collection_name: str = "documents",
        vector_size: int = 384,
        distance: str = "Cosine",
        enable_quantization: bool = True,
        quantization_type: str = "int8",
        hnsw_m: int = 16,
        hnsw_ef_construct: int = 100,
    ):
        """
        Initialize Vector Store with Qdrant embedded.
        
        Args:
            storage_path: Path for persistent storage
            collection_name: Name of the collection
            vector_size: Dimension of vectors (must match embedding model)
            distance: Distance metric (Cosine, Euclid, or Dot)
            enable_quantization: Enable scalar quantization for memory reduction
            quantization_type: Type of quantization (int8)
            hnsw_m: HNSW parameter - connections per node
            hnsw_ef_construct: HNSW parameter - build-time effort
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.distance = self._parse_distance(distance)
        self.enable_quantization = enable_quantization
        self.quantization_type = quantization_type
        self.hnsw_m = hnsw_m
        self.hnsw_ef_construct = hnsw_ef_construct
        
        # Initialize Qdrant client in embedded mode
        logger.info(f"Initializing Qdrant embedded at {storage_path}")
        self.client = QdrantClient(path=str(self.storage_path))
        
        # Create collection if it doesn't exist
        self._ensure_collection()
        
        logger.info(f"Vector Store initialized: collection={collection_name}")
    
    def _parse_distance(self, distance: str) -> Distance:
        """Parse distance string to Qdrant Distance enum."""
        distance_map = {
            "cosine": Distance.COSINE,
            "euclid": Distance.EUCLID,
            "euclidean": Distance.EUCLID,
            "dot": Distance.DOT,
        }
        return distance_map.get(distance.lower(), Distance.COSINE)
    
    def _ensure_collection(self):
        """Create collection if it doesn't exist."""
        collections = self.client.get_collections().collections
        collection_names = [col.name for col in collections]
        
        if self.collection_name in collection_names:
            logger.info(f"Collection '{self.collection_name}' already exists")
            return
        
        logger.info(f"Creating collection '{self.collection_name}'")
        
        # Configure quantization
        quantization_config = None
        if self.enable_quantization:
            quantization_config = ScalarQuantization(
                scalar=ScalarQuantizationConfig(
                    type=ScalarType.INT8,
                    quantile=0.99,
                    always_ram=True,
                )
            )
        
        # Create collection with configuration
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=self.vector_size,
                distance=self.distance,
            ),
            hnsw_config=HnswConfigDiff(
                m=self.hnsw_m,
                ef_construct=self.hnsw_ef_construct,
            ),
            quantization_config=quantization_config,
        )
        
        logger.info(f"Collection '{self.collection_name}' created successfully")
    
    def insert(
        self,
        vector: List[float],
        payload: Dict[str, Any],
        vector_id: Optional[str] = None,
    ) -> str:
        """
        Insert a single vector with metadata.
        
        Args:
            vector: The embedding vector
            payload: Metadata to store with the vector
            vector_id: Optional custom ID (UUID will be generated if not provided)
        
        Returns:
            The ID of the inserted vector
        """
        if vector_id is None:
            vector_id = str(uuid4())
        
        point = PointStruct(
            id=vector_id,
            vector=vector,
            payload=payload,
        )
        
        self.client.upsert(
            collection_name=self.collection_name,
            points=[point],
        )
        
        logger.debug(f"Inserted vector with ID: {vector_id}")
        return vector_id
    
    def insert_batch(
        self,
        vectors: List[List[float]],
        payloads: List[Dict[str, Any]],
        vector_ids: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Insert multiple vectors in batch.
        
        Args:
            vectors: List of embedding vectors
            payloads: List of metadata dictionaries
            vector_ids: Optional list of custom IDs
        
        Returns:
            List of IDs of inserted vectors
        """
        if len(vectors) != len(payloads):
            raise ValueError("Number of vectors and payloads must match")
        
        if vector_ids is None:
            vector_ids = [str(uuid4()) for _ in range(len(vectors))]
        elif len(vector_ids) != len(vectors):
            raise ValueError("Number of vector IDs must match number of vectors")
        
        points = [
            PointStruct(id=vid, vector=vec, payload=pay)
            for vid, vec, pay in zip(vector_ids, vectors, payloads)
        ]
        
        # Batch upsert
        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )
        
        logger.info(f"Inserted batch of {len(vectors)} vectors")
        return vector_ids
    
    def search(
        self,
        query_vector: List[float],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        score_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors.
        
        Args:
            query_vector: The query embedding vector
            top_k: Number of results to return
            filters: Optional filters on metadata
            score_threshold: Minimum similarity score
        
        Returns:
            List of search results with scores and payloads
        """
        # Build filter if provided
        query_filter = None
        if filters:
            query_filter = self._build_filter(filters)
        
        # Perform search
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k,
            query_filter=query_filter,
            score_threshold=score_threshold,
        )
        
        # Format results
        formatted_results = []
        for result in results:
            formatted_results.append({
                "id": result.id,
                "score": result.score,
                "payload": result.payload,
            })
        
        logger.debug(f"Search returned {len(formatted_results)} results")
        return formatted_results
    
    def _build_filter(self, filters: Dict[str, Any]) -> Filter:
        """
        Build Qdrant filter from simple dictionary.
        
        Supports:
        - Exact match: {"field": "value"}
        - Range: {"field": {"$gte": 10, "$lte": 20}}
        - Contains: {"field": {"$contains": "substring"}}
        """
        conditions = []
        
        for key, value in filters.items():
            if isinstance(value, dict):
                # Handle range or special operators
                if "$gte" in value or "$lte" in value or "$gt" in value or "$lt" in value:
                    conditions.append(
                        FieldCondition(
                            key=key,
                            range=Range(
                                gte=value.get("$gte"),
                                lte=value.get("$lte"),
                                gt=value.get("$gt"),
                                lt=value.get("$lt"),
                            ),
                        )
                    )
                elif "$contains" in value:
                    # For text contains, we use match
                    conditions.append(
                        FieldCondition(
                            key=key,
                            match=MatchValue(value=value["$contains"]),
                        )
                    )
            else:
                # Exact match
                conditions.append(
                    FieldCondition(
                        key=key,
                        match=MatchValue(value=value),
                    )
                )
        
        return Filter(must=conditions)
    
    def delete(self, vector_id: str) -> bool:
        """
        Delete a vector by ID.
        
        Args:
            vector_id: ID of the vector to delete
        
        Returns:
            True if deleted successfully
        """
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=[vector_id],
        )
        logger.debug(f"Deleted vector: {vector_id}")
        return True
    
    def delete_batch(self, vector_ids: List[str]) -> bool:
        """
        Delete multiple vectors by IDs.
        
        Args:
            vector_ids: List of vector IDs to delete
        
        Returns:
            True if deleted successfully
        """
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=vector_ids,
        )
        logger.info(f"Deleted batch of {len(vector_ids)} vectors")
        return True
    
    def update_payload(
        self,
        vector_id: str,
        payload: Dict[str, Any],
    ) -> bool:
        """
        Update the payload of a vector without changing the vector itself.
        
        Args:
            vector_id: ID of the vector
            payload: New payload data
        
        Returns:
            True if updated successfully
        """
        self.client.set_payload(
            collection_name=self.collection_name,
            payload=payload,
            points=[vector_id],
        )
        logger.debug(f"Updated payload for vector: {vector_id}")
        return True
    
    def get_collection_info(self) -> Dict[str, Any]:
        """
        Get information about the collection.
        
        Returns:
            Dictionary with collection stats
        """
        info = self.client.get_collection(collection_name=self.collection_name)
        return {
            "name": self.collection_name,
            "vectors_count": info.points_count,
            "indexed_vectors_count": info.indexed_vectors_count,
            "status": info.status,
            "vector_size": self.vector_size,
            "distance": self.distance,
        }
    
    def count(self) -> int:
        """
        Get the total number of vectors in the collection.
        
        Returns:
            Number of vectors
        """
        info = self.client.get_collection(collection_name=self.collection_name)
        return info.points_count
    
    def delete_collection(self):
        """Delete the entire collection."""
        self.client.delete_collection(collection_name=self.collection_name)
        logger.warning(f"Deleted collection: {self.collection_name}")


# Fix for quantization config
from qdrant_client.models import ScalarQuantizationConfig

# Update the quantization configuration
class VectorStore(VectorStore):
    """Extended with proper quantization config."""
    pass
