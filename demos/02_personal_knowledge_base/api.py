"""
Demo 2: Personal Knowledge Base

A web-based application for managing personal notes and documents
with powerful semantic search capabilities.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
from datetime import datetime
import json

from core.vector_store import VectorStore
from core.embedding_service import EmbeddingService
from core.document_processor import DocumentProcessor
from core.search_engine import SearchEngine, BatchIndexer

# Initialize FastAPI app
app = FastAPI(title="Personal Knowledge Base", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize core components
print("Initializing Knowledge Base components...")

vector_store = VectorStore(
    storage_path="./data/demo2_knowledge",
    collection_name="notes",
    vector_size=384,
)

embedding_service = EmbeddingService(
    model_name="mini",
    device="cpu",
)

doc_processor = DocumentProcessor(
    chunk_size=500,
    chunk_overlap=50,
    chunking_strategy="sentence",
)

search_engine = SearchEngine(
    vector_store=vector_store,
    embedding_service=embedding_service,
)

batch_indexer = BatchIndexer(
    vector_store=vector_store,
    embedding_service=embedding_service,
)

print("✅ Knowledge Base initialized!")

# Pydantic models
class Note(BaseModel):
    title: str
    content: str
    tags: Optional[List[str]] = []

class SearchQuery(BaseModel):
    query: str
    top_k: int = 10
    tags: Optional[List[str]] = []


@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the main HTML page."""
    html_path = Path(__file__).parent / "frontend.html"
    if html_path.exists():
        return html_path.read_text()
    return "<h1>Personal Knowledge Base</h1><p>frontend.html not found</p>"


@app.post("/api/notes")
async def create_note(note: Note):
    """Create a new note."""
    try:
        # Process note content
        chunks = doc_processor.process_text_directly(
            text=note.content,
            metadata={
                "title": note.title,
                "tags": note.tags,
                "created_at": datetime.now().isoformat(),
                "type": "note",
            }
        )
        
        # Index chunks
        vector_ids = batch_indexer.index_chunks(chunks, show_progress=False)
        
        return {
            "success": True,
            "message": f"Note created with {len(vector_ids)} chunks",
            "note_id": vector_ids[0] if vector_ids else None,
            "chunks_count": len(vector_ids),
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/search")
async def search_notes(search_query: SearchQuery):
    """Search notes."""
    try:
        # Build filters
        filters = None
        if search_query.tags:
            filters = {"tags": search_query.tags[0]}  # Simple tag filter
        
        # Perform search
        results = search_engine.search(
            query=search_query.query,
            top_k=search_query.top_k,
            filters=filters,
        )
        
        # Format results
        formatted_results = [
            {
                "text": r.text,
                "score": r.score,
                "title": r.metadata.get("title", "Untitled"),
                "tags": r.metadata.get("tags", []),
                "created_at": r.metadata.get("created_at", ""),
            }
            for r in results
        ]
        
        return {
            "success": True,
            "results": formatted_results,
            "count": len(formatted_results),
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stats")
async def get_stats():
    """Get knowledge base statistics."""
    try:
        stats = search_engine.get_statistics()
        collection_info = vector_store.get_collection_info()
        
        return {
            "total_vectors": stats["total_vectors"],
            "embedding_model": stats["embedding_model"],
            "collection": stats["collection_name"],
            "status": collection_info.get("status", "unknown"),
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload and index a document."""
    try:
        # Save uploaded file temporarily
        temp_path = Path(f"./data/temp_{file.filename}")
        temp_path.parent.mkdir(parents=True, exist_ok=True)
        
        content = await file.read()
        temp_path.write_bytes(content)
        
        # Process document
        chunks = doc_processor.process_file(
            file_path=temp_path,
            metadata={
                "uploaded_at": datetime.now().isoformat(),
                "type": "document",
            }
        )
        
        # Index chunks
        vector_ids = batch_indexer.index_chunks(chunks, show_progress=False)
        
        # Clean up temp file
        temp_path.unlink()
        
        return {
            "success": True,
            "message": f"Document '{file.filename}' indexed",
            "chunks_count": len(vector_ids),
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/notes/{note_id}")
async def delete_note(note_id: str):
    """Delete a note by ID."""
    try:
        vector_store.delete(note_id)
        return {"success": True, "message": "Note deleted"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    print("\n🚀 Starting Personal Knowledge Base...")
    print("📝 Access the application at: http://localhost:8000")
    print("📚 API docs at: http://localhost:8000/docs")
    print("\nPress Ctrl+C to stop the server\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
