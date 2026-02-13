"""
Incremental Ingestion Module

Tracks which blobs have been processed to enable incremental ingestion.
Only processes new or updated documents from Azure Blob Storage.
"""

import json
import os
import logging
from typing import List, Dict, Set, Optional
from datetime import datetime

from azure.storage.blob import ContainerClient

from config.settings import get_config
from ingestion.azure_blob_loader import get_container_client, download_blob, parse_pdf_with_context, parse_excel_transposed
from ingestion.chunking import chunk_documents
from ingestion.embed_and_store import embed_and_store_documents

logger = logging.getLogger(__name__)

# Track processed blobs in memory (can be persisted to file or database)
# Format: {blob_name: {"etag": "...", "last_modified": "...", "processed_at": "..."}}
_processed_blobs: Dict[str, Dict] = {}

# Path to store processed blobs metadata (optional persistence)
# Use /tmp in Docker, or local file in development
PROCESSED_BLOBS_FILE = os.getenv(
    "PROCESSED_BLOBS_FILE", 
    "/tmp/processed_blobs.json" if os.path.exists("/.dockerenv") else "./processed_blobs.json"
)


def load_processed_blobs() -> Dict[str, Dict]:
    """Load processed blobs metadata from file (if exists)."""
    if os.path.exists(PROCESSED_BLOBS_FILE):
        try:
            with open(PROCESSED_BLOBS_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load processed blobs file: {e}")
    return {}


def save_processed_blobs(processed_blobs: Dict[str, Dict]):
    """Save processed blobs metadata to file."""
    try:
        # Create directory if it doesn't exist (for non-/tmp paths)
        dir_path = os.path.dirname(PROCESSED_BLOBS_FILE)
        if dir_path and dir_path != "/tmp":
            os.makedirs(dir_path, exist_ok=True)
        elif dir_path == "/tmp":
            # /tmp should already exist, but ensure it's writable
            os.makedirs("/tmp", exist_ok=True)
        
        with open(PROCESSED_BLOBS_FILE, 'w') as f:
            json.dump(processed_blobs, f, indent=2)
        logger.debug(f"Saved processed blobs metadata to {PROCESSED_BLOBS_FILE}")
    except Exception as e:
        logger.warning(f"Failed to save processed blobs file: {e}")
        # Continue execution even if file save fails (metadata is in memory)


def get_new_or_updated_blobs(container_client: ContainerClient, processed_blobs: Dict[str, Dict]) -> List[Dict]:
    """
    Identify new or updated blobs that need processing.
    
    Args:
        container_client: Azure Blob Storage container client
        processed_blobs: Dictionary of previously processed blobs
        
    Returns:
        List of blob metadata dicts that are new or updated
    """
    new_blobs = []
    supported_extensions = ('.pdf', '.xlsx', '.xls', '.txt')
    
    try:
        all_blobs = list(container_client.list_blobs())
        logger.info(f"Found {len(all_blobs)} total blobs in container")
        
        for blob in all_blobs:
            blob_name = blob.name
            blob_ext = blob_name.lower()
            
            # Skip unsupported file types
            if not blob_ext.endswith(supported_extensions):
                continue
            
            # Check if blob is new or updated
            is_new = blob_name not in processed_blobs
            is_updated = False
            
            if not is_new:
                # Check if ETag or last_modified changed
                previous = processed_blobs[blob_name]
                if (previous.get("etag") != blob.etag or 
                    previous.get("last_modified") != blob.last_modified.isoformat()):
                    is_updated = True
            
            if is_new or is_updated:
                new_blobs.append({
                    "name": blob_name,
                    "size": blob.size,
                    "etag": blob.etag,
                    "last_modified": blob.last_modified.isoformat(),
                    "extension": blob_ext
                })
                logger.info(f"  {'NEW' if is_new else 'UPDATED'}: {blob_name} ({blob.size} bytes)")
        
        return new_blobs
    
    except Exception as e:
        logger.error(f"Failed to list blobs: {e}")
        return []


def process_blob_incremental(container_client: ContainerClient, blob_info: Dict) -> List[Dict]:
    """
    Process a single blob and return document chunks.
    
    Args:
        container_client: Azure Blob Storage container client
        blob_info: Blob metadata dict
        
    Returns:
        List of document dicts with text and metadata
    """
    blob_name = blob_info["name"]
    blob_ext = blob_info["extension"]
    
    try:
        # Download blob content
        content = download_blob(container_client, blob_name)
        logger.info(f"Downloaded {blob_name}: {len(content)} bytes")
        
        # Parse based on file type
        docs = []
        config = get_config().azure_blob
        
        if blob_ext.endswith('.pdf'):
            docs = parse_pdf_with_context(content, blob_name)
        elif blob_ext.endswith(('.xlsx', '.xls')):
            docs = parse_excel_transposed(content, blob_name)
        elif blob_ext.endswith('.txt'):
            # Handle TXT files
            try:
                text_content = content.decode('utf-8')
                if text_content.strip():
                    docs = [{
                        "text": text_content,
                        "metadata": {
                            "source": f"azure_blob/{blob_name}",
                            "document_type": "text",
                            "file_type": "txt",
                            "container": config.container_name
                        }
                    }]
            except UnicodeDecodeError:
                logger.warning(f"Failed to decode TXT as UTF-8: {blob_name}")
        
        logger.info(f"Parsed {blob_name}: {len(docs)} document chunks")
        return docs
    
    except Exception as e:
        logger.error(f"Failed to process blob {blob_name}: {e}")
        import traceback
        traceback.print_exc()
        return []


def ingest_incremental(fresh: bool = False) -> Dict:
    """
    Incremental ingestion: Only process new or updated documents from Azure Blob Storage.
    
    Args:
        fresh: If True, ignore processed blobs and re-process everything
        
    Returns:
        Dict with ingestion results
    """
    logger.info("=" * 60)
    logger.info("INCREMENTAL INGESTION PIPELINE")
    logger.info("=" * 60)
    
    # Load processed blobs metadata
    processed_blobs = {} if fresh else load_processed_blobs()
    if fresh:
        logger.info("[FRESH MODE] Will process all blobs")
    else:
        logger.info(f"Loaded {len(processed_blobs)} previously processed blobs")
    
    # Get container client
    try:
        container_client = get_container_client()
        config = get_config().azure_blob
        logger.info(f"Container: {config.container_name}")
    except Exception as e:
        logger.error(f"Failed to connect to Azure Blob: {e}")
        return {
            "success": False,
            "error": str(e),
            "documents_processed": 0,
            "blobs_processed": 0
        }
    
    # Get new or updated blobs
    new_blobs = get_new_or_updated_blobs(container_client, processed_blobs)
    
    if not new_blobs:
        logger.info("No new or updated blobs to process")
        return {
            "success": True,
            "documents_processed": 0,
            "blobs_processed": 0,
            "message": "No new or updated blobs found"
        }
    
    logger.info(f"Found {len(new_blobs)} new or updated blobs to process")
    
    # Process each new blob
    all_documents = []
    successfully_processed = []
    failed_blobs = []
    
    for blob_info in new_blobs:
        blob_name = blob_info["name"]
        try:
            docs = process_blob_incremental(container_client, blob_info)
            if docs:
                all_documents.extend(docs)
                successfully_processed.append(blob_name)
                
                # Update processed blobs metadata
                processed_blobs[blob_name] = {
                    "etag": blob_info["etag"],
                    "last_modified": blob_info["last_modified"],
                    "processed_at": datetime.utcnow().isoformat()
                }
        except Exception as e:
            logger.error(f"Failed to process blob {blob_name}: {e}")
            failed_blobs.append({"blob": blob_name, "error": str(e)})
    
    if not all_documents:
        logger.warning("No documents extracted from new blobs")
        return {
            "success": True,
            "documents_processed": 0,
            "blobs_processed": len(successfully_processed),
            "failed_blobs": failed_blobs,
            "message": "No documents extracted"
        }
    
    # Chunk documents
    logger.info(f"Chunking {len(all_documents)} documents...")
    chunked_docs = chunk_documents(all_documents)
    logger.info(f"Created {len(chunked_docs)} chunks")
    
    # Embed and store
    logger.info("Embedding and storing to Chroma Cloud...")
    stored_count = embed_and_store_documents(
        documents=chunked_docs,
        fresh=False,  # Never delete existing collection in incremental mode
        batch_size=50
    )
    
    # Save processed blobs metadata
    save_processed_blobs(processed_blobs)
    
    logger.info("=" * 60)
    logger.info("INCREMENTAL INGESTION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Blobs processed: {len(successfully_processed)}")
    logger.info(f"Documents stored: {stored_count}")
    if failed_blobs:
        logger.warning(f"Failed blobs: {len(failed_blobs)}")
    
    return {
        "success": True,
        "documents_processed": stored_count,
        "blobs_processed": len(successfully_processed),
        "blobs": successfully_processed,
        "failed_blobs": failed_blobs,
        "chunks_created": len(chunked_docs)
    }
