"""
Text Chunking Module

Splits documents into chunks suitable for embedding.
Uses LangChain's RecursiveCharacterTextSplitter for better text segmentation.
Preserves metadata across chunks.
"""

import os
from typing import List, Dict
from tqdm import tqdm

# Import RecursiveCharacterTextSplitter from langchain_text_splitters (LangChain 0.1+)
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    raise ImportError(
        "RecursiveCharacterTextSplitter not found. "
        "Please install langchain_text_splitters: pip install langchain_text_splitters"
    )


# Initialize text splitter with default settings
_text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    separators=["\n\n", "\n", ". ", " ", ""]
)


def chunk_documents(documents: List[Dict], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Dict]:
    """
    Chunks documents while preserving metadata.
    
    Uses LangChain's RecursiveCharacterTextSplitter for intelligent text segmentation.
    
    Args:
        documents: List of documents with 'text' and 'metadata' keys
        chunk_size: Maximum characters per chunk (default: 1000)
        chunk_overlap: Overlap between chunks (default: 200)
    
    Returns:
        List of chunked documents with metadata
    """
    if not documents:
        return []
    
    # Reinitialize splitter if custom size/overlap provided
    if chunk_size != 1000 or chunk_overlap != 200:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    else:
        splitter = _text_splitter
    
    chunked = []
    
    print(f"\nChunking {len(documents)} documents...")
    
    for doc in tqdm(documents, desc="Chunking documents"):
        # Extract text and metadata
        text = doc.get("text", "")
        metadata = doc.get("metadata", {})
        
        # Skip empty documents
        if not text or not text.strip():
            continue
        
        # Financial rows should not be further chunked (they are already atomic units)
        if metadata.get("document_type") == "financial_row":
            chunked.append(doc)
            continue
        
        # Use RecursiveCharacterTextSplitter to chunk text
        try:
            # Convert dict to LangChain Document format for splitting
            from langchain_core.documents import Document
            langchain_doc = Document(page_content=text, metadata=metadata)
            
            # Split document
            text_chunks = splitter.split_documents([langchain_doc])
            
            # Convert back to dict format and preserve original metadata
            for i, chunk_doc in enumerate(text_chunks):
                chunk_text_content = chunk_doc.page_content.strip()
                
                # Skip empty or very short chunks
                if not chunk_text_content or len(chunk_text_content) < 20:
                    continue
                
                # Merge chunk metadata with original metadata
                chunk_metadata = metadata.copy()
                chunk_metadata.update(chunk_doc.metadata)
                chunk_metadata["chunk_index"] = i
                chunk_metadata["total_chunks"] = len(text_chunks)
                
                chunked.append({
                    "text": chunk_text_content,
                    "metadata": chunk_metadata
                })
                
        except Exception as e:
            # Fallback: if splitting fails, use original document
            print(f"    [WARN] Failed to chunk document from {metadata.get('source', 'unknown')}: {e}")
            chunked.append(doc)
    
    print(f"\n✓ Created {len(chunked)} chunks from {len(documents)} documents")
    
    return chunked


def chunk_local_documents(directory: str = "documents") -> List[Dict]:
    """
    Loads local TXT documents and returns them as document dictionaries.
    
    Note: This function loads and structures documents but does NOT chunk them.
    Chunking is handled by chunk_documents() to maintain consistency.
    
    Args:
        directory: Path to documents directory
    
    Returns:
        List of document dictionaries with 'text' and 'metadata' keys
        Returns empty list if directory does not exist
    """
    print(f"\nLoading local documents from: {directory}")
    
    documents = []
    
    # Safely check if directory exists
    if not os.path.exists(directory):
        print(f"  [INFO] Directory not found: {directory} (skipping local documents)")
        return documents
    
    if not os.path.isdir(directory):
        print(f"  [WARN] Path exists but is not a directory: {directory}")
        return documents
    
    # List files in directory
    try:
        files = os.listdir(directory)
    except Exception as e:
        print(f"  [ERROR] Failed to list directory: {e}")
        return documents
    
    txt_files = [f for f in files if f.lower().endswith(".txt")]
    
    if not txt_files:
        print(f"  [INFO] No .txt files found in {directory}")
        return documents
    
    print(f"  Found {len(txt_files)} .txt file(s)")
    
    for filename in txt_files:
        filepath = os.path.join(directory, filename)
        print(f"  Loading: {filename}")
        
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
            
            if not content.strip():
                print(f"    [WARN] File is empty, skipping")
                continue
            
            # Create one document per file (chunking happens later)
            documents.append({
                "text": content,
                "metadata": {
                    "source": f"local/{filename}",
                    "filename": filename,
                    "file_path": filepath,
                    "document_type": "text_file"
                }
            })
            
            print(f"    [OK] Loaded {len(content)} characters")
            
        except Exception as e:
            print(f"    [ERROR] Failed to load {filename}: {e}")
            continue
    
    print(f"  ✓ Loaded {len(documents)} local document(s)")
    
    return documents
