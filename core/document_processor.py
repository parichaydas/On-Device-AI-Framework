"""
Document Processor Module - Extract and Chunk Documents

This module handles document format detection, text extraction,
and intelligent chunking for various file formats.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from enum import Enum
import re

# Document format libraries
import PyPDF2
import docx
from bs4 import BeautifulSoup
import chardet

logger = logging.getLogger(__name__)


class ChunkingStrategy(Enum):
    """Supported chunking strategies."""
    FIXED_SIZE = "fixed_size"
    SENTENCE = "sentence"
    PARAGRAPH = "paragraph"
    SEMANTIC = "semantic"


class DocumentProcessor:
    """
    Process documents of various formats and chunk them for embedding.
    
    Supports:
    - PDF files
    - Word documents (DOCX)
    - Text files (TXT, MD)
    - HTML files
    """
    
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        chunking_strategy: Union[str, ChunkingStrategy] = ChunkingStrategy.SENTENCE,
        min_chunk_size: int = 50,
        max_chunk_size: int = 2000,
    ):
        """
        Initialize document processor.
        
        Args:
            chunk_size: Target size for chunks (in characters)
            chunk_overlap: Overlap between chunks (in characters)
            chunking_strategy: Strategy for chunking text
            min_chunk_size: Minimum chunk size to keep
            max_chunk_size: Maximum chunk size allowed
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        if isinstance(chunking_strategy, str):
            self.chunking_strategy = ChunkingStrategy(chunking_strategy)
        else:
            self.chunking_strategy = chunking_strategy
        
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        
        logger.info(
            f"Document processor initialized: "
            f"strategy={self.chunking_strategy.value}, "
            f"chunk_size={chunk_size}"
        )
    
    def process_file(
        self,
        file_path: Union[str, Path],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Process a document file and return chunks with metadata.
        
        Args:
            file_path: Path to the document file
            metadata: Optional additional metadata
        
        Returns:
            List of chunks, each containing:
                - text: The chunk text
                - metadata: Document and chunk metadata
                - chunk_index: Index of this chunk
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        logger.info(f"Processing file: {file_path.name}")
        
        # Detect format and extract text
        file_ext = file_path.suffix.lower()
        
        if file_ext == ".pdf":
            text, doc_metadata = self._extract_pdf(file_path)
        elif file_ext in [".docx", ".doc"]:
            text, doc_metadata = self._extract_docx(file_path)
        elif file_ext in [".txt", ".md", ".text"]:
            text, doc_metadata = self._extract_text(file_path)
        elif file_ext in [".html", ".htm"]:
            text, doc_metadata = self._extract_html(file_path)
        else:
            # Try as plain text
            logger.warning(f"Unknown file type {file_ext}, attempting plain text extraction")
            text, doc_metadata = self._extract_text(file_path)
        
        # Merge metadata
        if metadata:
            doc_metadata.update(metadata)
        
        # Add file info to metadata
        doc_metadata["filename"] = file_path.name
        doc_metadata["file_path"] = str(file_path.absolute())
        doc_metadata["file_size"] = file_path.stat().st_size
        doc_metadata["file_type"] = file_ext
        
        # Chunk the text
        chunks = self._chunk_text(text, doc_metadata)
        
        logger.info(f"Extracted {len(chunks)} chunks from {file_path.name}")
        return chunks
    
    def _extract_pdf(self, file_path: Path) -> tuple:
        """Extract text from PDF file."""
        try:
            with open(file_path, "rb") as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                text_parts = []
                metadata = {
                    "page_count": len(pdf_reader.pages),
                }
                
                # Extract metadata if available
                if pdf_reader.metadata:
                    if pdf_reader.metadata.get("/Title"):
                        metadata["title"] = pdf_reader.metadata.get("/Title")
                    if pdf_reader.metadata.get("/Author"):
                        metadata["author"] = pdf_reader.metadata.get("/Author")
                
                # Extract text from all pages
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text.strip():
                            text_parts.append(page_text)
                    except Exception as e:
                        logger.warning(f"Failed to extract page {page_num}: {e}")
                
                full_text = "\n\n".join(text_parts)
                return full_text, metadata
        
        except Exception as e:
            logger.error(f"Error extracting PDF {file_path}: {e}")
            raise
    
    def _extract_docx(self, file_path: Path) -> tuple:
        """Extract text from DOCX file."""
        try:
            doc = docx.Document(file_path)
            
            metadata = {}
            
            # Extract core properties if available
            if hasattr(doc.core_properties, "title") and doc.core_properties.title:
                metadata["title"] = doc.core_properties.title
            if hasattr(doc.core_properties, "author") and doc.core_properties.author:
                metadata["author"] = doc.core_properties.author
            
            # Extract text from paragraphs
            text_parts = []
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    text_parts.append(paragraph.text)
            
            full_text = "\n\n".join(text_parts)
            return full_text, metadata
        
        except Exception as e:
            logger.error(f"Error extracting DOCX {file_path}: {e}")
            raise
    
    def _extract_text(self, file_path: Path) -> tuple:
        """Extract text from plain text file."""
        try:
            # Detect encoding
            with open(file_path, "rb") as file:
                raw_data = file.read()
                result = chardet.detect(raw_data)
                encoding = result["encoding"] or "utf-8"
            
            # Read with detected encoding
            with open(file_path, "r", encoding=encoding) as file:
                text = file.read()
            
            metadata = {"encoding": encoding}
            return text, metadata
        
        except Exception as e:
            logger.error(f"Error extracting text {file_path}: {e}")
            raise
    
    def _extract_html(self, file_path: Path) -> tuple:
        """Extract text from HTML file."""
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
                html_content = file.read()
            
            soup = BeautifulSoup(html_content, "html.parser")
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Get text
            text = soup.get_text()
            
            # Clean up whitespace
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = "\n".join(chunk for chunk in chunks if chunk)
            
            # Extract metadata from HTML
            metadata = {}
            if soup.title:
                metadata["title"] = soup.title.string
            
            return text, metadata
        
        except Exception as e:
            logger.error(f"Error extracting HTML {file_path}: {e}")
            raise
    
    def _chunk_text(
        self,
        text: str,
        base_metadata: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Chunk text according to the configured strategy."""
        
        if self.chunking_strategy == ChunkingStrategy.FIXED_SIZE:
            chunks = self._chunk_fixed_size(text)
        elif self.chunking_strategy == ChunkingStrategy.SENTENCE:
            chunks = self._chunk_by_sentence(text)
        elif self.chunking_strategy == ChunkingStrategy.PARAGRAPH:
            chunks = self._chunk_by_paragraph(text)
        else:
            # Default to fixed size
            chunks = self._chunk_fixed_size(text)
        
        # Filter and format chunks
        result_chunks = []
        for idx, chunk_text in enumerate(chunks):
            # Skip if too small
            if len(chunk_text) < self.min_chunk_size:
                continue
            
            # Truncate if too large
            if len(chunk_text) > self.max_chunk_size:
                chunk_text = chunk_text[:self.max_chunk_size]
            
            result_chunks.append({
                "text": chunk_text.strip(),
                "metadata": {
                    **base_metadata,
                    "chunk_index": idx,
                    "chunk_size": len(chunk_text),
                },
                "chunk_index": idx,
            })
        
        return result_chunks
    
    def _chunk_fixed_size(self, text: str) -> List[str]:
        """Chunk text into fixed-size chunks with overlap."""
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = start + self.chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start += self.chunk_size - self.chunk_overlap
        
        return chunks
    
    def _chunk_by_sentence(self, text: str) -> List[str]:
        """Chunk text by sentences, respecting chunk size limits."""
        # Simple sentence splitting (can be improved with spaCy/NLTK)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence_size = len(sentence)
            
            # If adding this sentence exceeds chunk size, start new chunk
            if current_size + sentence_size > self.chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))
                
                # Keep overlap
                overlap_text = " ".join(current_chunk)
                if len(overlap_text) > self.chunk_overlap:
                    overlap_sentences = current_chunk[-1:]
                else:
                    overlap_sentences = current_chunk
                
                current_chunk = overlap_sentences
                current_size = sum(len(s) for s in current_chunk)
            
            current_chunk.append(sentence)
            current_size += sentence_size
        
        # Add final chunk
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
    
    def _chunk_by_paragraph(self, text: str) -> List[str]:
        """Chunk text by paragraphs."""
        paragraphs = text.split("\n\n")
        
        chunks = []
        current_chunk = []
        current_size = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            para_size = len(para)
            
            # If paragraph alone is larger than chunk size, split it
            if para_size > self.chunk_size:
                # Add current chunk if exists
                if current_chunk:
                    chunks.append("\n\n".join(current_chunk))
                    current_chunk = []
                    current_size = 0
                
                # Split large paragraph
                sub_chunks = self._chunk_fixed_size(para)
                chunks.extend(sub_chunks)
            else:
                # Check if adding this paragraph exceeds chunk size
                if current_size + para_size > self.chunk_size and current_chunk:
                    chunks.append("\n\n".join(current_chunk))
                    current_chunk = []
                    current_size = 0
                
                current_chunk.append(para)
                current_size += para_size
        
        # Add final chunk
        if current_chunk:
            chunks.append("\n\n".join(current_chunk))
        
        return chunks
    
    def process_text_directly(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Process raw text directly (without file).
        
        Args:
            text: Raw text to process
            metadata: Optional metadata
        
        Returns:
            List of chunks with metadata
        """
        base_metadata = metadata or {}
        base_metadata["source_type"] = "direct_text"
        
        chunks = self._chunk_text(text, base_metadata)
        logger.info(f"Processed direct text into {len(chunks)} chunks")
        return chunks
