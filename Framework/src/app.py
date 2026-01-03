"""
Small application layer wiring embedding provider and index for demos.
"""
from embedding import text_to_embedding
from index import SimpleVectorIndex
from typing import List, Dict


class FrameworkApp:
    def __init__(self):
        self.index = SimpleVectorIndex()

    def index_documents(self, docs: Dict[str, str]):
        for id, text in docs.items():
            vec = text_to_embedding(text)
            self.index.add(id, vec, {"text": text})

    def query(self, text: str, k: int = 5):
        qv = text_to_embedding(text)
        return self.index.search(qv, k=k)


if __name__ == '__main__':
    app = FrameworkApp()
    sample = {
        "doc1": "On-device AI with embedded vector search",
        "doc2": "Privacy-preserving local inference and retrieval",
        "doc3": "Qdrant Embedded integration example",
    }
    app.index_documents(sample)
    results = app.query("local vector search example", k=3)
    for r in results:
        print(r)
