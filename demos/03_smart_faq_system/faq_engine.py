"""
Demo 3: Smart FAQ System

An intelligent FAQ system that understands natural language questions
and finds relevant answers from a pre-indexed knowledge base.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import json
from typing import List, Dict, Any
from pathlib import Path

from core.vector_store import VectorStore
from core.embedding_service import EmbeddingService
from core.search_engine import SearchEngine, BatchIndexer
from core.document_processor import DocumentProcessor


class FAQEngine:
    """Engine for FAQ matching and answering."""
    
    def __init__(
        self,
        storage_path: str = "./data/demo3_faq",
        collection_name: str = "faqs",
    ):
        """Initialize FAQ engine."""
        self.vector_store = VectorStore(
            storage_path=storage_path,
            collection_name=collection_name,
            vector_size=384,
        )
        
        self.embedding_service = EmbeddingService(
            model_name="mini",
            device="cpu",
        )
        
        self.search_engine = SearchEngine(
            vector_store=self.vector_store,
            embedding_service=self.embedding_service,
        )
        
        self.doc_processor = DocumentProcessor(
            chunk_size=500,
            chunk_overlap=50,
        )
        
        self.batch_indexer = BatchIndexer(
            vector_store=self.vector_store,
            embedding_service=self.embedding_service,
        )
    
    def load_faqs_from_json(self, json_path: str):
        """
        Load and index FAQs from JSON file.
        
        Expected format:
        [
            {
                "question": "What is...",
                "answer": "It is...",
                "category": "General",
                "keywords": ["word1", "word2"]
            },
            ...
        ]
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            faqs = json.load(f)
        
        print(f"Loading {len(faqs)} FAQs...")
        
        chunks = []
        for idx, faq in enumerate(faqs):
            # Create text combining question and answer
            text = f"Q: {faq['question']}\nA: {faq['answer']}"
            
            metadata = {
                "question": faq["question"],
                "answer": faq["answer"],
                "category": faq.get("category", "General"),
                "keywords": faq.get("keywords", []),
                "faq_id": idx,
                "type": "faq",
            }
            
            # Process as chunks
            processed = self.doc_processor.process_text_directly(
                text=text,
                metadata=metadata,
            )
            
            chunks.extend(processed)
        
        # Index all chunks
        print(f"Indexing {len(chunks)} chunks...")
        vector_ids = self.batch_indexer.index_chunks(chunks, show_progress=True)
        
        print(f"✅ Indexed {len(vector_ids)} FAQ chunks!")
        return len(vector_ids)
    
    def ask(
        self,
        question: str,
        top_k: int = 3,
        category: str = None,
    ) -> List[Dict[str, Any]]:
        """
        Ask a question and get relevant FAQ answers.
        
        Args:
            question: User's question
            top_k: Number of results to return
            category: Optional category filter
        
        Returns:
            List of FAQ results with confidence scores
        """
        # Build filters
        filters = None
        if category:
            filters = {"category": category}
        
        # Search
        results = self.search_engine.search(
            query=question,
            top_k=top_k,
            filters=filters,
        )
        
        # Format results
        faqs = []
        for result in results:
            faqs.append({
                "question": result.metadata.get("question"),
                "answer": result.metadata.get("answer"),
                "category": result.metadata.get("category"),
                "confidence": result.score,
                "keywords": result.metadata.get("keywords", []),
            })
        
        return faqs
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the FAQ system."""
        return self.search_engine.get_statistics()


# Sample FAQ data
SAMPLE_FAQS = [
    {
        "question": "What is the On-Device AI Framework?",
        "answer": "The On-Device AI Framework is a complete solution for building intelligent applications that operate entirely offline using Qdrant embedded vector search and local machine learning models.",
        "category": "General",
        "keywords": ["framework", "offline", "AI", "on-device"]
    },
    {
        "question": "How does semantic search work?",
        "answer": "Semantic search converts text into vector embeddings that capture meaning. When you search, your query is also converted to a vector, and the system finds the most similar vectors using mathematical similarity measures like cosine distance.",
        "category": "Technical",
        "keywords": ["semantic", "search", "vectors", "embeddings"]
    },
    {
        "question": "What file formats are supported?",
        "answer": "The framework supports PDF, DOCX, TXT, MD, and HTML files. The document processor automatically detects the format and extracts text content.",
        "category": "Features",
        "keywords": ["files", "formats", "PDF", "DOCX"]
    },
    {
        "question": "Can this work offline?",
        "answer": "Yes! After the initial setup (downloading the embedding model), the entire framework operates offline. No internet connection is required for indexing or searching documents.",
        "category": "Features",
        "keywords": ["offline", "internet", "connectivity"]
    },
    {
        "question": "What embedding models are available?",
        "answer": "The framework includes several pre-configured models: MiniLM-L6 (fast, 384 dims), MiniLM-L12 (balanced), MPNet (high quality, 768 dims), and multilingual models supporting 50+ languages.",
        "category": "Technical",
        "keywords": ["models", "embeddings", "MiniLM", "MPNet"]
    },
    {
        "question": "How much memory does it use?",
        "answer": "Memory usage depends on the number of documents and model choice. For 100K documents with int8 quantization, expect around 800MB total (425MB index + 90MB model + overhead).",
        "category": "Performance",
        "keywords": ["memory", "RAM", "performance"]
    },
    {
        "question": "How fast is the search?",
        "answer": "Search is very fast! For 100K documents, average search latency is around 21ms on modern hardware. The HNSW index provides sub-100ms search even with millions of documents.",
        "category": "Performance",
        "keywords": ["speed", "latency", "fast", "performance"]
    },
    {
        "question": "Is my data private?",
        "answer": "Absolutely! All data remains on your device. Nothing is transmitted to external servers. This makes the framework ideal for sensitive data like healthcare records, legal documents, or proprietary business information.",
        "category": "Privacy",
        "keywords": ["privacy", "security", "data", "HIPAA"]
    },
    {
        "question": "Can I customize the chunking strategy?",
        "answer": "Yes! The DocumentProcessor supports multiple chunking strategies: fixed-size, sentence-based, and paragraph-based. You can also customize chunk size, overlap, and other parameters.",
        "category": "Customization",
        "keywords": ["chunking", "customization", "configuration"]
    },
    {
        "question": "How do I install the framework?",
        "answer": "Installation is simple: 1) Clone the repository, 2) Install dependencies with 'pip install -r requirements.txt', 3) Run any of the demo applications. The first run will download the embedding model automatically.",
        "category": "Getting Started",
        "keywords": ["install", "setup", "requirements"]
    },
]


def create_sample_faqs():
    """Create sample FAQs JSON file."""
    sample_path = Path("sample_faqs.json")
    with open(sample_path, 'w', encoding='utf-8') as f:
        json.dump(SAMPLE_FAQS, f, indent=2, ensure_ascii=False)
    print(f"✅ Created {sample_path} with {len(SAMPLE_FAQS)} FAQs")
    return str(sample_path)


def main():
    """Main CLI for FAQ system."""
    print("="*60)
    print("  Smart FAQ System - On-Device AI Framework")
    print("="*60)
    print()
    
    # Create sample FAQs if they don't exist
    sample_path = "sample_faqs.json"
    if not Path(sample_path).exists():
        print("Creating sample FAQs...")
        create_sample_faqs()
        print()
    
    # Initialize FAQ engine
    print("Initializing FAQ Engine...")
    engine = FAQEngine()
    
    # Check if FAQs are already indexed
    stats = engine.get_statistics()
    
    if stats["total_vectors"] == 0:
        print("No FAQs indexed yet. Loading sample FAQs...")
        engine.load_faqs_from_json(sample_path)
        print()
    else:
        print(f"✅ FAQ Engine ready with {stats['total_vectors']} indexed entries")
        print()
    
    # Interactive Q&A loop
    print("="*60)
    print("Ask questions! (Type 'quit' to exit)")
    print("="*60)
    print()
    
    while True:
        question = input("❓ Your Question: ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Goodbye!")
            break
        
        if not question:
            continue
        
        # Ask the FAQ engine
        results = engine.ask(question, top_k=3)
        
        print()
        if not results:
            print("❌ No matching FAQs found.")
        else:
            for i, faq in enumerate(results, 1):
                confidence_pct = faq["confidence"] * 100
                
                # Color code based on confidence
                if confidence_pct > 70:
                    conf_symbol = "🟢"
                elif confidence_pct > 50:
                    conf_symbol = "🟡"
                else:
                    conf_symbol = "🔴"
                
                print(f"--- Result #{i} {conf_symbol} Confidence: {confidence_pct:.1f}% ---")
                print(f"Category: {faq['category']}")
                print(f"\nQ: {faq['question']}")
                print(f"\nA: {faq['answer']}")
                print()
        
        print("-"*60)
        print()


if __name__ == "__main__":
    main()
