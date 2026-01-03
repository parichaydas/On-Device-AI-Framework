"""
Simple in-memory vector index for demo and testing.
Provides add(id, vector, metadata) and search(query_vector, k).
"""
from typing import List, Tuple, Dict, Any
import math


class SimpleVectorIndex:
    def __init__(self):
        self.vectors: Dict[str, List[float]] = {}
        self.meta: Dict[str, Dict[str, Any]] = {}

    def add(self, id: str, vector: List[float], metadata: Dict[str, Any] = None):
        self.vectors[id] = vector
        self.meta[id] = metadata or {}

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(x * x for x in b))
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)

    def search(self, query_vector: List[float], k: int = 5) -> List[Tuple[str, float, Dict[str, Any]]]:
        scores = []
        for id, vec in self.vectors.items():
            sim = self._cosine_similarity(query_vector, vec)
            scores.append((id, sim, self.meta.get(id, {})))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]

    def count(self) -> int:
        return len(self.vectors)
