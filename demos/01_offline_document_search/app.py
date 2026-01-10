"""
Demo 1: Offline Document Search System

A command-line application for indexing and searching documents
using semantic search - all offline after initial setup.
"""

import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import click
from rich.console import Console
from rich.table import Table
from rich.progress import Progress
from rich.panel import Panel

from core.vector_store import VectorStore
from core.embedding_service import EmbeddingService
from core.document_processor import DocumentProcessor
from core.search_engine import SearchEngine, BatchIndexer

console = Console()


@click.group()
def cli():
    """Offline Document Search System - Search documents using AI, completely offline!"""
    pass


@cli.command()
@click.argument('document_path', type=click.Path(exists=True))
@click.option('--storage', default='./data/demo1_vectors', help='Vector storage path')
@click.option('--collection', default='documents', help='Collection name')
def index(document_path, storage, collection):
    """Index a document or directory of documents."""
    console.print("[bold blue]🔍 Offline Document Search - Indexing[/bold blue]\n")
    
    document_path = Path(document_path)
    
    # Initialize components
    console.print("⚙️  Initializing components...")
    
    vector_store = VectorStore(
        storage_path=storage,
        collection_name=collection,
        vector_size=384,
    )
    
    embedding_service = EmbeddingService(
        model_name="mini",  # Fast, compact model
        device="cpu",
    )
    
    doc_processor = DocumentProcessor(
        chunk_size=500,
        chunk_overlap=50,
        chunking_strategy="sentence",
    )
    
    batch_indexer = BatchIndexer(
        vector_store=vector_store,
        embedding_service=embedding_service,
        batch_size=32,
    )
    
    # Process documents
    all_chunks = []
    
    if document_path.is_file():
        files_to_process = [document_path]
    else:
        # Get all supported files in directory
        files_to_process = []
        for ext in ['.pdf', '.docx', '.txt', '.md', '.html']:
            files_to_process.extend(document_path.glob(f'**/*{ext}'))
    
    console.print(f"📁 Found {len(files_to_process)} file(s) to process\n")
    
    # Process each file
    with Progress() as progress:
        task = progress.add_task("[green]Processing documents...", total=len(files_to_process))
        
        for file_path in files_to_process:
            try:
                console.print(f"📄 Processing: {file_path.name}")
                chunks = doc_processor.process_file(file_path)
                all_chunks.extend(chunks)
                console.print(f"   ✅ Extracted {len(chunks)} chunks")
            except Exception as e:
                console.print(f"   ❌ Error: {e}", style="red")
            
            progress.update(task, advance=1)
    
    # Index all chunks
    if all_chunks:
        console.print(f"\n💾 Indexing {len(all_chunks)} total chunks...")
        vector_ids = batch_indexer.index_chunks(all_chunks, show_progress=True)
        
        console.print(f"\n✨ [bold green]Successfully indexed {len(vector_ids)} chunks![/bold green]")
        
        # Show stats
        stats = vector_store.get_collection_info()
        console.print(f"\n📊 Collection stats:")
        console.print(f"   • Total vectors: {stats['vectors_count']}")
        console.print(f"   • Collection: {stats['name']}")
        console.print(f"   • Status: {stats['status']}")
    else:
        console.print("\n⚠️  No chunks to index", style="yellow")


@cli.command()
@click.option('--storage', default='./data/demo1_vectors', help='Vector storage path')
@click.option('--collection', default='documents', help='Collection name')
@click.option('--top-k', default=5, help='Number of results to return')
def search(storage, collection, top_k):
    """Interactive search interface."""
    console.print("[bold blue]🔍 Offline Document Search - Search Interface[/bold blue]\n")
    
    # Initialize components
    console.print("⚙️  Initializing search engine...")
    
    try:
        vector_store = VectorStore(
            storage_path=storage,
            collection_name=collection,
            vector_size=384,
        )
        
        embedding_service = EmbeddingService(
            model_name="mini",
            device="cpu",
        )
        
        search_engine = SearchEngine(
            vector_store=vector_store,
            embedding_service=embedding_service,
        )
        
        # Show stats
        stats = search_engine.get_statistics()
        console.print(f"✅ Loaded collection with {stats['total_vectors']} vectors\n")
        
        if stats['total_vectors'] == 0:
            console.print("⚠️  No documents indexed yet. Use 'index' command first.", style="yellow")
            return
        
    except Exception as e:
        console.print(f"❌ Error initializing: {e}", style="red")
        return
    
    # Interactive search loop
    console.print("[bold cyan]Enter your search queries (or 'quit' to exit):[/bold cyan]\n")
    
    while True:
        query = console.input("[bold green]Search>[/bold green] ")
        
        if query.lower() in ['quit', 'exit', 'q']:
            console.print("\n👋 Goodbye!")
            break
        
        if not query.strip():
            continue
        
        try:
            # Perform search
            results = search_engine.search(query, top_k=top_k)
            
            if not results:
                console.print("❌ No results found\n", style="yellow")
                continue
            
            # Display results
            console.print(f"\n[bold]Found {len(results)} results:[/bold]\n")
            
            for i, result in enumerate(results, 1):
                # Create result panel
                metadata = result.metadata
                source = metadata.get('filename', 'Unknown')
                chunk_idx = metadata.get('chunk_index', '?')
                
                title = f"Result #{i} - Score: {result.score:.4f}"
                subtitle = f"Source: {source} (Chunk {chunk_idx})"
                
                content = result.text
                if len(content) > 300:
                    content = content[:300] + "..."
                
                panel = Panel(
                    content,
                    title=title,
                    subtitle=subtitle,
                    border_style="blue",
                )
                console.print(panel)
            
            console.print()
            
        except Exception as e:
            console.print(f"❌ Search error: {e}\n", style="red")


@cli.command()
@click.option('--storage', default='./data/demo1_vectors', help='Vector storage path')
@click.option('--collection', default='documents', help='Collection name')
def stats(storage, collection):
    """Show collection statistics."""
    console.print("[bold blue]📊 Collection Statistics[/bold blue]\n")
    
    try:
        vector_store = VectorStore(
            storage_path=storage,
            collection_name=collection,
            vector_size=384,
        )
        
        info = vector_store.get_collection_info()
        
        table = Table(title="Collection Information")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Collection Name", info['name'])
        table.add_row("Total Vectors", str(info['vectors_count']))
        table.add_row("Indexed Vectors", str(info.get('indexed_vectors_count', 'N/A')))
        table.add_row("Vector Dimensions", str(info['vector_size']))
        table.add_row("Distance Metric", str(info['distance']))
        table.add_row("Status", str(info['status']))
        
        console.print(table)
        
    except Exception as e:
        console.print(f"❌ Error: {e}", style="red")


if __name__ == '__main__':
    cli()
