# Setup Guide - On-Device AI Framework

This guide provides detailed instructions for setting up the On-Device AI Framework on your system.

## System Requirements

### Minimum Requirements
- **CPU:** Dual-core processor (2.0+ GHz)
- **RAM:** 2GB available memory
- **Storage:** 2GB free space (500MB for models, 1.5GB for data)
- **Python:** 3.9 or higher
- **OS:** Windows 10+, macOS 10.14+, Ubuntu 18.04+

### Recommended Requirements
- **CPU:** Quad-core processor with AVX2 support
- **RAM:** 4GB+ available memory
- **Storage:** 5GB+ free space
- **GPU:** NVIDIA GPU with CUDA support (optional, for faster indexing)
- **Python:** 3.10 or 3.11

## Installation Steps

### 1. Python Environment Setup

#### Option A: Using venv (Recommended)

**Windows:**
```powershell
# Create virtual environment
python -m venv venv

# Activate
.\venv\Scripts\activate
```

**Linux/macOS:**
```bash
# Create virtual environment
python3 -m venv venv

# Activate
source venv/bin/activate
```

#### Option B: Using Conda

```bash
# Create conda environment
conda create -n ondevice-ai python=3.10

# Activate
conda activate ondevice-ai
```

### 2. Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install all requirements
pip install -r requirements.txt
```

**Installation will include:**
- Qdrant client (~10MB)
- Sentence Transformers (~50MB)
- PyTorch (~200MB for CPU version)
- Document processing libraries
- Web demo dependencies (optional)

### 3. Verify Installation

```python
# Run verification script
python -c "
from core import VectorStore, EmbeddingService
print('✅ Core imports successful')

# Test embedding service (will download model on first run)
embedding_service = EmbeddingService(model_name='mini')
test_vector = embedding_service.encode('test')
print(f'✅ Embedding service working (vector dim: {len(test_vector)})')

# Test vector store
vector_store = VectorStore(storage_path='./test_data')
print('✅ Vector store initialized')
"
```

### 4. Download Embedding Models

Models are downloaded automatically on first use, but you can pre-download:

```python
from sentence_transformers import SentenceTransformer

# Download default model (all-MiniLM-L6-v2, ~23MB)
model = SentenceTransformer('all-MiniLM-L6-v2')

# Download other models (optional)
# model = SentenceTransformer('all-mpnet-base-v2')  # ~110MB, higher quality
# model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')  # ~135MB
```

## Configuration Options

### Environment Variables

Create a `.env` file in the project root:

```env
# Storage paths
VECTOR_STORE_PATH=./data/vectors
CACHE_PATH=./data/cache

# Model settings
DEFAULT_EMBEDDING_MODEL=all-MiniLM-L6-v2
DEVICE=cpu  # or cuda

# Chunking settings
DEFAULT_CHUNK_SIZE=500
DEFAULT_CHUNK_OVERLAP=50

# Search settings
DEFAULT_TOP_K=10
```

### Configuration File

Create `config.json`:

```json
{
  "vector_store": {
    "storage_path": "./data/vectors",
    "collection_name": "documents",
    "vector_size": 384,
    "distance": "Cosine",
    "enable_quantization": true
  },
  "embedding_service": {
    "model_name": "all-MiniLM-L6-v2",
    "device": "cpu",
    "max_seq_length": 256
  },
  "document_processor": {
    "chunk_size": 500,
    "chunk_overlap": 50,
    "chunking_strategy": "sentence"
  }
}
```

## GPU Support (Optional)

### NVIDIA GPU (CUDA)

1. **Install CUDA Toolkit** (if not already installed)
   - Download from: https://developer.nvidia.com/cuda-downloads
   - Version 11.8 or 12.1 recommended

2. **Install PyTorch with CUDA**

```bash
# Uninstall CPU version
pip uninstall torch

# Install CUDA version
pip install torch==2.1.2+cu118 -f https://download.pytorch.org/whl/torch_stable.html
```

3. **Verify GPU**

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device: {torch.cuda.get_device_name(0)}")
```

4. **Use GPU in framework**

```python
embedding_service = EmbeddingService(
    model_name="mini",
    device="cuda"  # Use GPU
)
```

## Troubleshooting

### Common Issues

#### 1. Module Import Errors

**Problem:** `ModuleNotFoundError: No module named 'qdrant_client'`

**Solution:**
```bash
# Ensure virtual environment is activated
# Re-install requirements
pip install -r requirements.txt
```

#### 2. Model Download Fails

**Problem:** Network error when downloading models

**Solution:**
```bash
# Download manually
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2', cache_folder='./models')
```

#### 3. Out of Memory

**Problem:** `RuntimeError: CUDA out of memory` or system slowdown

**Solution:**
```python
# Use smaller model
embedding_service = EmbeddingService(model_name="mini")

# Enable quantization
vector_store = VectorStore(enable_quantization=True)

# Reduce batch size
batch_indexer = BatchIndexer(batch_size=16)  # Instead of 32
```

#### 4. Slow Indexing

**Problem:** Document indexing is very slow

**Solution:**
```python
# Increase batch size (if memory allows)
batch_indexer = BatchIndexer(batch_size=64)

# Use GPU
embedding_service = EmbeddingService(device="cuda")

# Reduce chunk size for faster processing
doc_processor = DocumentProcessor(chunk_size=300)
```

#### 5. Search Returns No Results

**Problem:** Search always returns empty results

**Solution:**
```python
# Check if documents are indexed
print(f"Vectors in collection: {vector_store.count()}")

# Lower score threshold
results = search_engine.search(query, score_threshold=0.3)

# Check collection name matches
vector_store = VectorStore(collection_name="your_collection")
```

## Platform-Specific Notes

### Windows

- Use PowerShell (not CMD) for better Unicode support
- Antivirus may slow down first-time model downloads
- Path separators: Use raw strings `r"C:\path\to\data"` or forward slashes

### macOS

- Apple Silicon (M1/M2): PyTorch has native support
- May need to install Xcode Command Line Tools:
  ```bash
  xcode-select --install
  ```

### Linux

- Install system dependencies for PDF processing:
  ```bash
  sudo apt-get install python3-dev libpoppler-cpp-dev
  ```

## Running Tests

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=core --cov-report=html

# View coverage report
# Open htmlcov/index.html in browser
```

## Next Steps

1. ✅ **Try the demos** - Run one of the three demo applications
2. ✅ **Read the documentation** - Check out PROJECT_WRITEUP.md and ARCHITECTURE.md
3. ✅ **Build your application** - Use the framework in your own projects
4. ✅ **Customize** - Adjust settings for your specific use case

## Getting Help

- **Documentation:** Check `/docs` folder
- **Examples:** See `/demos` folder
- **Issues:** Open an issue on GitHub
- **Discussions:** GitHub Discussions

---

**Setup complete! 🎉 You're ready to build offline AI applications.**
