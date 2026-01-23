"""
Embedding and Storage Module

Stores documents in Chroma Cloud using Azure OpenAI embeddings.
Uses deterministic IDs to prevent duplicates.

Embeds documents using Azure OpenAI before storing in Chroma Cloud.
"""

import hashlib
from typing import List, Dict
from tqdm import tqdm

from langchain_openai import AzureOpenAIEmbeddings
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings

from config.settings import get_config
from vectorstore.chroma_client import get_collection, delete_collection


class AzureOpenAIEmbeddingFunction(EmbeddingFunction):
    """
    Custom embedding function for Chroma using Azure OpenAI.
    
    Wraps LangChain's AzureOpenAIEmbeddings to work with Chroma.
    """
    
    def __init__(self):
        """Initialize Azure OpenAI embeddings."""
        config = get_config().azure_openai
        
        # For text-embedding-3-large, don't pass model parameter (deployment name is sufficient)
        # The deployment name in Azure OpenAI already identifies the model
        embedding_kwargs = {
            "azure_endpoint": config.endpoint,
            "azure_deployment": config.embeddings_deployment,
            "api_key": config.api_key,
            "api_version": config.api_version,
        }
        
        # Only add model parameter for older models if needed
        # text-embedding-3-large doesn't need explicit model parameter
        if "text-embedding-ada-002" in config.embeddings_deployment.lower():
            embedding_kwargs["model"] = "text-embedding-ada-002"
        
        self.embeddings = AzureOpenAIEmbeddings(**embedding_kwargs)
    
    def __call__(self, input: Documents) -> Embeddings:
        """
        Generate embeddings for documents.
        
        Args:
            input: List of document texts
        
        Returns:
            List of embedding vectors
        """
        return self.embeddings.embed_documents(input)


def generate_deterministic_id(text: str, source: str) -> str:
    """
    Generates deterministic ID for deduplication.
    Same content + source = same ID = no duplicate.
    """
    content = f"{source}:{text[:500]}"
    return hashlib.sha256(content.encode()).hexdigest()[:32]


def embed_and_store_documents(
    documents: List[Dict],
    collection_name: str = None,
    fresh: bool = False,
    batch_size: int = 50
) -> int:
    """
    Embeds documents using Azure OpenAI and stores them in Chroma Cloud.
    
    Args:
        documents: List of documents with 'text' and 'metadata'
        collection_name: Target collection (defaults to config)
        fresh: If True, delete existing collection first
        batch_size: Batch size for embedding and upsert (default: 50)
    
    Returns:
        Number of documents stored
    """
    print("=" * 60)
    print("EMBEDDING AND STORING DOCUMENTS TO CHROMA CLOUD")
    print("=" * 60)
    
    if not documents:
        print("[WARN] No documents to store")
        return 0
    
    config = get_config()
    collection_name = collection_name or config.chroma_cloud.collection_name
    
    # Delete existing if fresh ingestion
    if fresh:
        print(f"\n[FRESH MODE] Deleting existing collection '{collection_name}'...")
        try:
            delete_collection(collection_name)
            print(f"  ✓ Collection deleted")
        except Exception as e:
            print(f"  [INFO] Collection may not exist: {e}")
    
    # Initialize embedding function
    print(f"\nInitializing Azure OpenAI embeddings...")
    print(f"  Endpoint: {config.azure_openai.endpoint[:50]}...")
    print(f"  Deployment: {config.azure_openai.embeddings_deployment}")
    
    try:
        embedding_function = AzureOpenAIEmbeddingFunction()
        print("  ✓ Embedding function ready")
    except Exception as e:
        print(f"  [ERROR] Failed to initialize embeddings: {e}")
        import traceback
        traceback.print_exc()
        return 0
    
    # Get collection with custom embedding function
    print(f"\nConnecting to Chroma Cloud collection: {collection_name}")
    try:
        # Import here to avoid circular dependency
        from vectorstore.chroma_client import get_chroma_client
        
        client = get_chroma_client()
        
        # Delete and recreate collection if fresh mode (ensures clean state)
        if fresh:
            print(f"  [FRESH MODE] Deleting existing collection '{collection_name}'...")
            try:
                client.delete_collection(name=collection_name)
                print(f"  ✓ Collection deleted successfully")
            except Exception as e:
                print(f"  [INFO] Collection may not exist or already deleted: {e}")
            
            # Small delay to ensure deletion propagates
            import time
            time.sleep(1)
        
        # Get or create collection WITHOUT embedding_function
        # CRITICAL: For Chroma Cloud with pre-computed embeddings, do NOT pass embedding_function
        # Passing embedding_function causes KeyError '_type' due to JSON schema mismatch
        # We pass embeddings explicitly during upsert(), so embedding_function is not needed here
        # The embedding_function is only needed for query-time when embeddings are computed on-the-fly
        print(f"  Creating/getting collection '{collection_name}'...")
        collection = client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        current_count = collection.count()
        print(f"  ✓ Collection ready (current count: {current_count})")
        
        # Log tenant/database/collection for verification
        import os
        print(f"\n  [VERIFY] Connection details:")
        print(f"    Tenant: {config.chroma_cloud.tenant}")
        print(f"    Database: {config.chroma_cloud.database}")
        print(f"    Collection: {collection.name}")
        print(f"    Collection count: {current_count}")
        
    except Exception as e:
        print(f"  [ERROR] Failed to get collection: {e}")
        import traceback
        traceback.print_exc()
        return 0
    
    print(f"\nProcessing {len(documents)} documents in batches of {batch_size}...")
    print(f"  Embedding model: Azure OpenAI ({config.azure_openai.embeddings_deployment})")
    
    # Diagnostic: Check document structure
    if documents:
        sample_doc = documents[0]
        print(f"\n  [DEBUG] Sample document structure:")
        print(f"    Keys: {list(sample_doc.keys())}")
        if "text" in sample_doc:
            text_len = len(sample_doc.get("text", ""))
            print(f"    Text length: {text_len} chars")
            print(f"    Text preview: {sample_doc.get('text', '')[:100]}...")
        if "metadata" in sample_doc:
            print(f"    Metadata keys: {list(sample_doc.get('metadata', {}).keys())}")
    
    # Process in batches
    total_stored = 0
    total_embedded = 0
    batch_num = 0
    
    for i in tqdm(range(0, len(documents), batch_size), desc="Embedding and storing"):
        batch = documents[i:i + batch_size]
        batch_num += 1
        
        # Extract texts and filter empty
        texts = []
        valid_docs = []
        
        for doc in batch:
            text = doc.get("text", "").strip()
            if text and len(text) >= 10:  # Only include non-empty, meaningful texts
                texts.append(text)
                valid_docs.append(doc)
            else:
                if batch_num == 1:  # Log first batch filtering
                    print(f"    [DEBUG] Skipped document: text length={len(text) if text else 0}")
        
        if not texts:
            print(f"\n  [WARN] Batch {batch_num}: All documents empty or too short, skipping")
            print(f"    [DEBUG] Batch size: {len(batch)}, Valid texts: {len(texts)}")
            continue
        
        # Generate embeddings for this batch using Azure OpenAI
        try:
            print(f"\n  Batch {batch_num}: Generating embeddings for {len(texts)} documents...")
            embeddings = embedding_function(texts)
            
            if not embeddings:
                print(f"  [ERROR] Batch {batch_num}: Embedding function returned empty list")
                continue
            
            if len(embeddings) != len(texts):
                print(f"  [WARN] Batch {batch_num}: Expected {len(texts)} embeddings, got {len(embeddings)}")
            
            total_embedded += len(embeddings)
            print(f"  ✓ Batch {batch_num}: Generated {len(embeddings)} embeddings (dimension: {len(embeddings[0]) if embeddings else 0})")
        except Exception as e:
            print(f"\n  [ERROR] Batch {batch_num}: Failed to generate embeddings: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        # Generate deterministic IDs for valid docs only
        ids = [
            generate_deterministic_id(doc["text"], doc.get("metadata", {}).get("source", ""))
            for doc in valid_docs
        ]
        
        # Extract and clean metadata
        metadatas = []
        for doc in valid_docs:
            meta = doc.get("metadata", {}).copy()
            
            # Remove None values
            keys_to_remove = [k for k, v in meta.items() if v is None]
            for k in keys_to_remove:
                del meta[k]
            
            # Convert float values to strings for Chroma compatibility
            for k, v in list(meta.items()):
                if isinstance(v, float):
                    meta[k] = str(v)
                # Ensure all values are JSON-serializable
                elif not isinstance(v, (str, int, bool)):
                    meta[k] = str(v)
            
            metadatas.append(meta)
        
        # Add documents to Chroma Cloud with pre-computed embeddings
        try:
            # CRITICAL VALIDATION: Ensure all lists match before adding
            assert len(ids) > 0, f"Batch {batch_num}: No IDs generated"
            assert len(ids) == len(texts), f"Batch {batch_num}: IDs ({len(ids)}) and texts ({len(texts)}) length mismatch"
            assert len(ids) == len(embeddings), f"Batch {batch_num}: IDs ({len(ids)}) and embeddings ({len(embeddings)}) length mismatch"
            assert len(ids) == len(metadatas), f"Batch {batch_num}: IDs ({len(ids)}) and metadatas ({len(metadatas)}) length mismatch"
            
            # Instrumentation: Log before adding
            print(f"\n>>> ABOUT TO ADD DOCUMENTS (Batch {batch_num})")
            print(f">>> IDs: {len(ids)}")
            print(f">>> Texts: {len(texts)}")
            print(f">>> Embeddings: {len(embeddings)} (dimension: {len(embeddings[0]) if embeddings else 0})")
            print(f">>> Metadatas: {len(metadatas)}")
            
            # Get collection count BEFORE add
            count_before = collection.count()
            
            # FORCE DIRECT CLOUD STORAGE - use upsert to ensure writes
            collection.upsert(
                ids=ids,
                documents=texts,
                metadatas=metadatas,
                embeddings=embeddings
            )
            
            # Verify immediately after add
            count_after = collection.count()
            added_count = count_after - count_before
            
            print(f">>> ADD FINISHED (Batch {batch_num})")
            print(f">>> COUNT BEFORE: {count_before}")
            print(f">>> COUNT AFTER: {count_after}")
            print(f">>> DOCUMENTS ADDED THIS BATCH: {added_count}")
            
            if added_count != len(ids):
                print(f"  [WARN] Expected {len(ids)} documents added, but count increased by {added_count}")
            
            total_stored += len(valid_docs)
            
            # Log progress every 5 batches
            if batch_num % 5 == 0:
                print(f"    Progress: {total_stored}/{len(documents)} documents stored")
            
            # Log first batch details
            if batch_num == 1:
                print(f"\n  First batch details:")
                print(f"    Documents: {len(valid_docs)}")
                if valid_docs:
                    first_source = valid_docs[0].get("metadata", {}).get("source", "unknown")
                    first_text_len = len(valid_docs[0].get("text", ""))
                    print(f"    Example source: {first_source}")
                    print(f"    Example text length: {first_text_len} chars")
                
        except Exception as e:
            print(f"\n  [ERROR] Batch {batch_num}: Failed to add documents: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Final summary
    final_count = collection.count()
    
    print(f"\n{'=' * 60}")
    print(f"EMBEDDING AND STORAGE COMPLETE")
    print(f"{'=' * 60}")
    print(f"Documents processed: {len(documents)}")
    print(f"Embeddings generated: {total_embedded}")
    print(f"Documents stored: {total_stored}")
    print(f"Collection count: {final_count}")
    print(f"{'=' * 60}")
    
    return total_stored
