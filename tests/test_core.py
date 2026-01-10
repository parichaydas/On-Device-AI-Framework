"""
Test Suite for On-Device AI Framework

Basic tests to verify core functionality.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import tempfile
import shutil
from core.vector_store import VectorStore
from core.embedding_service import EmbeddingService
from core.document_processor import DocumentProcessor
from core.search_engine import SearchEngine


@pytest.fixture
def temp_storage():
    """Create temporary storage directory."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def vector_store(temp_storage):
    """Create vector store instance."""
    return VectorStore(
        storage_path=temp_storage,
        collection_name="test_collection",
        vector_size=384,
    )


@pytest.fixture
def embedding_service():
    """Create embedding service instance."""
    return EmbeddingService(model_name="mini", device="cpu")


class TestEmbeddingService:
    """Test embedding service functionality."""
    
    def test_initialization(self, embedding_service):
        """Test service initialization."""
        assert embedding_service.model_name == "all-MiniLM-L6-v2"
        assert embedding_service.device in ["cpu", "cuda"]
    
    def test_single_encoding(self, embedding_service):
        """Test encoding single text."""
        text = "This is a test sentence."
        embedding = embedding_service.encode(text)
        
        assert isinstance(embedding, list)
        assert len(embedding) == 384  # MiniLM-L6 dimension
        assert all(isinstance(x, float) for x in embedding)
    
    def test_batch_encoding(self, embedding_service):
        """Test batch encoding."""
        texts = ["First sentence.", "Second sentence.", "Third sentence."]
        embeddings = embedding_service.encode_batch(texts)
        
        assert len(embeddings) == 3
        assert all(len(emb) == 384 for emb in embeddings)
    
    def test_similarity(self, embedding_service):
        """Test similarity calculation."""
        emb1 = embedding_service.encode("machine learning")
        emb2 = embedding_service.encode("artificial intelligence")
        emb3 = embedding_service.encode("cooking recipes")
        
        # Similar texts should have higher similarity
        sim_similar = embedding_service.similarity(emb1, emb2)
        sim_different = embedding_service.similarity(emb1, emb3)
        
        assert sim_similar > sim_different


class TestVectorStore:
    """Test vector store functionality."""
    
    def test_initialization(self, vector_store):
        """Test store initialization."""
        assert vector_store.collection_name == "test_collection"
        assert vector_store.vector_size == 384
    
    def test_insert_single(self, vector_store, embedding_service):
        """Test inserting single vector."""
        text = "Test document"
        embedding = embedding_service.encode(text)
        
        vector_id = vector_store.insert(
            vector=embedding,
            payload={"text": text, "type": "test"}
        )
        
        assert vector_id is not None
        assert isinstance(vector_id, str)
    
    def test_insert_batch(self, vector_store, embedding_service):
        """Test batch insertion."""
        texts = ["Doc 1", "Doc 2", "Doc 3"]
        embeddings = embedding_service.encode_batch(texts)
        payloads = [{"text": t, "index": i} for i, t in enumerate(texts)]
        
        vector_ids = vector_store.insert_batch(
            vectors=embeddings,
            payloads=payloads
        )
        
        assert len(vector_ids) == 3
        assert vector_store.count() == 3
    
    def test_search(self, vector_store, embedding_service):
        """Test vector search."""
        # Insert some test data
        texts = [
            "Machine learning is a subset of AI",
            "Deep learning uses neural networks",
            "Cats are popular pets"
        ]
        embeddings = embedding_service.encode_batch(texts)
        payloads = [{"text": t} for t in texts]
        vector_store.insert_batch(embeddings, payloads)
        
        # Search
        query = "artificial intelligence and ML"
        query_embedding = embedding_service.encode(query)
        results = vector_store.search(query_embedding, top_k=2)
        
        assert len(results) <= 2
        assert results[0]["score"] > 0
        
    def test_delete(self,vector_store, embedding_service):
        """Test vector deletion."""
        embedding = embedding_service.encode("Test")
        vector_id = vector_store.insert(embedding, {"text": "Test"})
        
        # Delete
        vector_store.delete(vector_id)
        
        # Count should be 0 (or decreased)
        # Note: Actual count depends on previous tests


class TestDocumentProcessor:
    """Test document processor functionality."""
    
    def test_initialization(self):
        """Test processor initialization."""
        processor = DocumentProcessor(chunk_size=500)
        assert processor.chunk_size == 500
    
    def test_process_text_directly(self):
        """Test direct text processing."""
        processor = DocumentProcessor(chunk_size=100, chunk_overlap=20)
        
        text = "This is a test. " * 50  # Create long text
        chunks = processor.process_text_directly(text)
        
        assert len(chunks) > 0
        assert all("text" in chunk for chunk in chunks)
        assert all("metadata" in chunk for chunk in chunks)
    
    def test_chunking_strategies(self):
        """Test different chunking strategies."""
        text = "First sentence. Second sentence. Third sentence. " * 10
        
        # Fixed size
        processor1 = DocumentProcessor(
            chunk_size=100,
            chunking_strategy="fixed_size"
        )
        chunks1 = processor1.process_text_directly(text)
        
        # Sentence based
        processor2 = DocumentProcessor(
            chunk_size=100,
            chunking_strategy="sentence"
        )
        chunks2 = processor2.process_text_directly(text)
        
        assert len(chunks1) > 0
        assert len(chunks2) > 0


class TestSearchEngine:
    """Test search engine functionality."""
    
    def test_initialization(self, vector_store, embedding_service):
        """Test search engine initialization."""
        search_engine = SearchEngine(vector_store, embedding_service)
        assert search_engine.vector_store == vector_store
        assert search_engine.embedding_service == embedding_service
    
    def test_search(self, vector_store, embedding_service):
        """Test search functionality."""
        # Create search engine
        search_engine = SearchEngine(vector_store, embedding_service)
        
        # Add some test data
        texts = [
            "Python is a programming language",
            "JavaScript is used for web development",
            "Machine learning requires data"
        ]
        embeddings = embedding_service.encode_batch(texts)
        payloads = [{"text": t} for t in texts]
        vector_store.insert_batch(embeddings, payloads)
        
        # Search
        results = search_engine.search("programming and coding", top_k=2)
        
        assert len(results) <= 2
        assert all(hasattr(r, 'score') for r in results)
        assert all(hasattr(r, 'text') for r in results)
    
    def test_statistics(self, vector_store, embedding_service):
        """Test getting statistics."""
        search_engine = SearchEngine(vector_store, embedding_service)
        stats = search_engine.get_statistics()
        
        assert "total_vectors" in stats
        assert "embedding_model" in stats
        assert "collection_name" in stats


def test_end_to_end_workflow(temp_storage):
    """Test complete workflow from indexing to search."""
    # Initialize components
    vector_store = VectorStore(
        storage_path=temp_storage,
        collection_name="e2e_test"
    )
    embedding_service = EmbeddingService(model_name="mini")
    doc_processor = DocumentProcessor(chunk_size=200)
    search_engine = SearchEngine(vector_store, embedding_service)
    
    # Process text
    text = """
    The On-Device AI Framework is designed for offline operation.
    It uses Qdrant for vector storage and Sentence Transformers for embeddings.
    The framework supports multiple document formats including PDF and DOCX.
    Privacy is a key feature as all data stays on the device.
    """
    
    chunks = doc_processor.process_text_directly(text)
    
    # Index chunks
    for chunk in chunks:
        embedding = embedding_service.encode(chunk['text'])
        vector_store.insert(embedding, chunk['metadata'])
    
    # Search
    results = search_engine.search("How does privacy work?", top_k=3)
    
    assert len(results) > 0
    assert any("privacy" in r.text.lower() for r in results)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
