#!/usr/bin/env python3
"""
Ingestion Entrypoint

One-time job to:
1. Load documents from Azure Blob Storage
2. Load local TXT documents  
3. Chunk all documents
4. Embed using Azure OpenAI
5. Store in Chroma Cloud

Run: python ingest.py [--fresh]
"""

import argparse
import sys

# Add current directory to path for imports
sys.path.insert(0, ".")

from config.settings import get_config, validate_config
from ingestion.azure_blob_loader import load_azure_documents
from ingestion.chunking import chunk_documents, chunk_local_documents
from ingestion.embed_and_store import embed_and_store_documents


def main(fresh: bool = False, documents_dir: str = "documents"):
    """
    Main ingestion pipeline.
    
    Args:
        fresh: If True, delete existing collection and re-ingest
        documents_dir: Path to local documents directory
    """
    print("=" * 60)
    print("CHROMA CLOUD INGESTION PIPELINE")
    print("=" * 60)
    print()
    
    # Load and validate config
    try:
        config = get_config()
        validate_config(config)
    except ValueError as e:
        print(f"\n[ERROR] Configuration error: {e}")
        print("\nPlease check your .env file contains all required variables:")
        print("  - AZURE_OPENAI_API_KEY")
        print("  - AZURE_OPENAI_ENDPOINT")
        print("  - AZURE_OPENAI_API_VERSION")
        print("  - AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")
        print("  - AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME")
        print("  - AZURE_STORAGE_CONNECTION_STRING")
        print("  - AZURE_BLOB_CONTAINER_NAME")
        print("  - CHROMA_HOST")
        print("  - CHROMA_API_KEY")
        print("  - CHROMA_TENANT")
        print("  - CHROMA_DATABASE")
        return 1
    
    print()
    
    # Step 1: Load Azure Blob documents
    print("[STEP 1/4] Loading Azure Blob documents...")
    azure_docs = load_azure_documents()
    
    print()
    
    # Step 2: Load local documents
    print("[STEP 2/4] Loading local documents...")
    local_docs = chunk_local_documents(documents_dir)
    
    print()
    
    # Step 3: Combine and chunk
    print("[STEP 3/4] Combining and chunking...")
    all_docs = azure_docs + local_docs
    chunked_docs = chunk_documents(all_docs)
    
    print(f"\nTotal documents after chunking: {len(chunked_docs)}")
    print(f"  - From Azure Blob: {len(azure_docs)}")
    print(f"  - From local files: {len(local_docs)}")
    
    print()
    
    # Step 4: Embed and store
    print("[STEP 4/4] Embedding and storing to Chroma Cloud...")
    
    if fresh:
        print("\n[FRESH MODE] Will delete existing collection first.")
    
    stored = embed_and_store_documents(
        documents=chunked_docs,
        fresh=fresh,
        batch_size=50
    )
    
    print()
    print("=" * 60)
    print("INGESTION COMPLETE")
    print("=" * 60)
    print(f"Documents stored: {stored}")
    print()
    print("You can now run the chatbot:")
    print("  python main.py")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ingest documents into Chroma Cloud"
    )
    
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Delete existing collection and re-ingest from scratch"
    )
    
    parser.add_argument(
        "--documents-dir",
        type=str,
        default="documents",
        help="Path to local documents directory (default: documents)"
    )
    
    args = parser.parse_args()
    
    exit_code = main(fresh=args.fresh, documents_dir=args.documents_dir)
    sys.exit(exit_code)
