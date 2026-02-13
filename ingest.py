#!/usr/bin/env python3
"""
Ingestion Entrypoint — STRICT Streaming Pipeline

Architecture:
    1. Load documents from Azure Blob Storage
    2. Load local TXT documents
    3. Chunk all documents
    4. Stream into Chroma Cloud: one source document at a time,
       one chunk at a time — embed → store → verify per chunk.

Run:
    python ingest.py               # resume from last checkpoint
    python ingest.py --fresh       # delete collection + restart

Behaviour:
    - On success: exit 0, count increases for every new chunk.
    - On ANY anomaly: RuntimeError + exit 1.
    - Crash-safe: checkpoint tracks completed source documents;
      re-running resumes from the next unprocessed source.
"""

import argparse
import sys
import traceback

# Add current directory to path for imports
sys.path.insert(0, ".")

from config.settings import get_config, validate_config
from ingestion.azure_blob_loader import load_azure_documents
from ingestion.chunking import chunk_documents, chunk_local_documents
from ingestion.embed_and_store import embed_and_store_documents


def main(
    company_name: str,
    fresh: bool = False,
    documents_dir: str = "documents",
) -> int:
    """Main ingestion pipeline.

    Returns 0 on success, 1 on any failure.
    RuntimeErrors from the storage layer propagate and crash the process.
    """
    print("=" * 60)
    print("CHROMA CLOUD INGESTION PIPELINE (STRICT STREAMING)")
    print("=" * 60)
    print()

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------
    try:
        config = get_config()
        validate_config(config)
    except ValueError as exc:
        print(f"\n[FATAL] Configuration error: {exc}")
        print("\nRequired .env variables:")
        print("  AZURE_OPENAI_API_KEY")
        print("  AZURE_OPENAI_ENDPOINT")
        print("  AZURE_OPENAI_API_VERSION")
        print("  AZURE_OPENAI_LARGE_CHAT_DEPLOYMENT_NAME")
        print("  AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME")
        print("  AZURE_STORAGE_CONNECTION_STRING")
        print("  AZURE_BLOB_CONTAINER_NAME")
        print("  CHROMA_HOST")
        print("  CHROMA_API_KEY")
        print("  CHROMA_TENANT")
        print("  CHROMA_DATABASE")
        return 1

    print()

    # ------------------------------------------------------------------
    # Step 1: Load Azure Blob documents
    # ------------------------------------------------------------------
    print("[STEP 1/4] Loading Azure Blob documents...")
    azure_docs = load_azure_documents()
    print()

    # ------------------------------------------------------------------
    # Step 2: Load local documents
    # ------------------------------------------------------------------
    print("[STEP 2/4] Loading local documents...")
    local_docs = chunk_local_documents(documents_dir)
    print()

    # ------------------------------------------------------------------
    # Step 3: Combine and chunk
    # ------------------------------------------------------------------
    print("[STEP 3/4] Combining and chunking...")
    all_docs = azure_docs + local_docs
    chunked_docs = chunk_documents(all_docs)

    print(f"\nTotal chunks after splitting: {len(chunked_docs)}")
    print(f"  - From Azure Blob: {len(azure_docs)} raw docs")
    print(f"  - From local files: {len(local_docs)} raw docs")
    print()

    # ------------------------------------------------------------------
    # Step 4: Stream-embed and store (one chunk at a time)
    # ------------------------------------------------------------------
    print("[STEP 4/4] Streaming into Chroma Cloud...")

    if fresh:
        print(f"\n[FRESH MODE] Will delete collection for '{company_name}', verify count==0, clear checkpoint.")

    stored = embed_and_store_documents(
        documents=chunked_docs,
        fresh=fresh,
        company_name=company_name,
    )

    print()
    print("=" * 60)
    print("PIPELINE FINISHED SUCCESSFULLY")
    print("=" * 60)
    print(f"  Chunks newly stored: {stored}")
    print()
    print("You can now run the chatbot:")
    print("  python main.py")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ingest documents into Chroma Cloud (STRICT streaming)",
    )

    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Delete existing collection, verify empty, re-ingest from scratch",
    )

    parser.add_argument(
        "--documents-dir",
        type=str,
        default="documents",
        help="Path to local documents directory (default: documents)",
    )
    parser.add_argument(
        "--company",
        type=str,
        required=True,
        help="Company name for dynamic collection isolation",
    )

    args = parser.parse_args()

    try:
        exit_code = main(
            fresh=args.fresh,
            documents_dir=args.documents_dir,
            company_name=args.company,
        )
    except RuntimeError as exc:
        print()
        print("!" * 60)
        print("FATAL ERROR — INGESTION ABORTED")
        print("!" * 60)
        print(f"  {exc}")
        print()
        traceback.print_exc()
        print("!" * 60)
        exit_code = 1
    except Exception as exc:
        print()
        print("!" * 60)
        print("UNEXPECTED ERROR — INGESTION ABORTED")
        print("!" * 60)
        print(f"  {type(exc).__name__}: {exc}")
        print()
        traceback.print_exc()
        print("!" * 60)
        exit_code = 1

    sys.exit(exit_code)
