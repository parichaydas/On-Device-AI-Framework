# Demo 1: Offline Document Search System

A powerful command-line application that enables semantic search across your documents - completely offline!

## Features

- ✅ Index PDF, DOCX, TXT, MD, and HTML documents
- ✅ Semantic search (understands meaning, not just keywords)
- ✅ Batch indexing of entire directories
- ✅ Interactive search interface
- ✅ Works 100% offline after initial setup
- ✅ Fast search across thousands of documents

## Quick Start

### 1. Install Dependencies

```bash
pip install -r ../../requirements.txt
```

### 2. Index Documents

Index a single document:
```bash
python app.py index path/to/document.pdf
```

Index an entire directory:
```bash
python app.py index path/to/documents/
```

### 3. Search

Start interactive search:
```bash
python app.py search
```

Then enter your queries:
```
Search> What is machine learning?
Search> deployment best practices
Search> quit
```

### 4. View Statistics

```bash
python app.py stats
```

## Usage Examples

### Index Sample Documents

```bash
# Create sample documents folder
mkdir sample_docs

# Add your PDFs, DOCX, TXT files to sample_docs/

# Index all documents
python app.py index sample_docs/
```

### Advanced Search

```bash
# Return top 10 results instead of 5
python app.py search --top-k 10

# Use custom storage location
python app.py search --storage ./my_vectors --collection my_docs
```

## How It Works

1. **Document Processing**: Extracts text from various formats and splits into chunks
2. **Embedding Generation**: Converts text chunks to vector embeddings using Sentence Transformers
3. **Vector Storage**: Stores embeddings in Qdrant embedded database
4. **Semantic Search**: Finds similar content based on meaning, not just keywords

## Sample Output

```
🔍 Offline Document Search - Search Interface

⚙️  Initializing search engine...
✅ Loaded collection with 1,234 vectors

Enter your search queries (or 'quit' to exit):

Search> machine learning deployment
Search> What is machine learning deployment?

Found 5 results:

╭─ Result #1 - Score: 0.8542 ──────────────────────╮
│ Machine learning deployment involves taking      │
│ trained models and integrating them into        │
│ production systems...                            │
╰─ Source: ml_guide.pdf (Chunk 12) ───────────────╯
```

## Tips

- **Better Results**: Use natural language questions instead of just keywords
- **Chunking**: Smaller chunks (300-500 chars) work better for specific questions
- **Batch Indexing**: Index multiple files at once for efficiency
- **Filters**: Metadata filters can narrow down search scope

## Customization

Edit `app.py` to customize:
- Chunk size and overlap
- Embedding model (mini, base, mpnet)
- Number of results
- Display format
