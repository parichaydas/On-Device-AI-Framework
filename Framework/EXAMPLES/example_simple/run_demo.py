"""
Runnable demo that indexes sample documents and performs a query.
This demo uses the framework `src` modules.
"""
import sys
import os

# Ensure we can import from src by adding Framework/src to sys.path
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
SRC = os.path.join(ROOT, 'src')
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from app import FrameworkApp


def main():
    app = FrameworkApp()
    docs = {
        "doc1": "Fast, offline semantic search on mobile devices.",
        "doc2": "Local vector search using an embedded index.",
        "doc3": "Contextual assistant that runs without cloud.",
    }
    print("Indexing documents...")
    app.index_documents(docs)
    print(f"Indexed {app.index.count()} documents")

    query = "offline semantic search"
    print(f"\nQuery: {query}\nResults:")
    results = app.query(query, k=3)
    for id, score, meta in results:
        print(f"- {id} (score={score:.4f}) -> {meta.get('text')}")


if __name__ == '__main__':
    main()
