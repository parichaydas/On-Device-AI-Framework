"""
Lightweight deterministic embedding provider for demo purposes.
This avoids external dependencies and produces small float vectors.
"""
from typing import List


def text_to_embedding(text: str, dim: int = 32) -> List[float]:
    """Convert text into a deterministic pseudo-embedding.

    This is a demo embedding function that hashes tokens to produce
    a stable vector suitable for testing the index.
    """
    vec = [0.0] * dim
    if not text:
        return vec
    tokens = text.split()
    for i, token in enumerate(tokens):
        h = abs(hash(token))
        for j in range(dim):
            vec[j] += ((h >> (j % 24)) & 0xFF) / 255.0 * (1.0 / (i + 1))
    # normalize
    norm = sum(x * x for x in vec) ** 0.5
    if norm == 0:
        return vec
    return [x / norm for x in vec]
