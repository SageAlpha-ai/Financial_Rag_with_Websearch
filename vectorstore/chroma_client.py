"""
Chroma Cloud Client

STRICT RULES:
- NEVER use persist_directory
- NEVER use PersistentClient
- ONLY use HttpClient for Chroma Cloud
- ALL embeddings go to Chroma Cloud
- Local disk stores ZERO vectors
"""

from typing import Optional
import chromadb
from chromadb import HttpClient
from chromadb.config import Settings
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection

from config.settings import get_config, ChromaCloudConfig


# Module-level singleton
_client: Optional[ClientAPI] = None


def get_chroma_client() -> ClientAPI:
    """
    Gets Chroma Cloud client (singleton).
    
    NEVER uses local storage.
    ALL vectors stored in Chroma Cloud.
    
    For ChromaDB 0.4.24, uses CloudClient which handles authentication properly.
    Falls back to HttpClient if CloudClient has issues.
    """
    global _client
    
    if _client is not None:
        return _client
    
    import os
    config = get_config().chroma_cloud
    
    # Validate API key is present
    if not config.api_key or not config.api_key.strip():
        raise ValueError(
            "CHROMA_API_KEY environment variable is not set or is empty. "
            "Please set it in your .env file or environment variables. "
            "Get your API key from https://trychroma.com"
        )
    
    print("=" * 60)
    print("CONNECTING TO CHROMA CLOUD")
    print("=" * 60)
    print(f"Host: {config.host}")
    print(f"Tenant: {config.tenant}")
    print(f"Database: {config.database}")
    api_key_display = '*' * (len(config.api_key) - 8) + config.api_key[-8:] if len(config.api_key) > 8 else '***'
    print(f"API Key: {api_key_display}")
    
    # Set environment variables for ChromaDB to read
    # CloudClient and HttpClient both read these in 0.4.24
    os.environ["CHROMA_API_KEY"] = config.api_key
    os.environ["CHROMA_TENANT"] = config.tenant
    os.environ["CHROMA_DATABASE"] = config.database
    
    # FORCE CloudClient usage - it's the only reliable way for Chroma Cloud v2 API
    # CloudClient handles v2 API automatically and is designed specifically for Chroma Cloud
    if not hasattr(chromadb, 'CloudClient'):
        raise ValueError(
            f"CloudClient not available in ChromaDB {chromadb.__version__}. "
            f"Please upgrade: pip install --upgrade 'chromadb>=0.5.0'"
        )
    
    print("Connecting to Chroma Cloud using CloudClient (v2 API)...")
    
    # CloudClient initialization may fail with v1 API error during tenant validation
    # This is a known issue in ChromaDB 0.6.3. We'll catch it and provide upgrade instructions.
    try:
        _client = chromadb.CloudClient(
            api_key=config.api_key,
            tenant=config.tenant,
            database=config.database,
            settings=Settings(
                allow_reset=True,
                anonymized_telemetry=False
            )
        )
        
        # Verify connection
        try:
            heartbeat = _client.heartbeat()
            print(f"✓ Connected to Chroma Cloud (heartbeat: {heartbeat})")
        except Exception as verify_error:
            verify_msg = str(verify_error)
            if "v1 API is deprecated" in verify_msg:
                print(f"⚠ Warning: v1 API detected during verification")
            else:
                print(f"⚠ Connection verification: {verify_msg}")
        
        print("=" * 60)
        return _client
        
    except Exception as cloud_error:
        error_msg = str(cloud_error)
        
        # Provide helpful error message
        if "Permission denied" in error_msg or "401" in error_msg or "Unauthorized" in error_msg:
            raise ValueError(
                f"Authentication failed: Permission denied (401 Unauthorized).\n\n"
                f"This means your CHROMA_API_KEY is either:\n"
                f"  1. Not set correctly in your .env file or environment variables\n"
                f"  2. Invalid or expired\n"
                f"  3. Doesn't have access to tenant '{config.tenant}' and database '{config.database}'\n\n"
                f"To fix this:\n"
                f"  1. Get your API key from https://trychroma.com\n"
                f"  2. Set it in your .env file: CHROMA_API_KEY=your_api_key_here\n"
                f"  3. Verify tenant and database match your Chroma Cloud dashboard\n\n"
                f"Current configuration:\n"
                f"  - Host: {config.host}\n"
                f"  - Tenant: {config.tenant}\n"
                f"  - Database: {config.database}\n"
                f"  - API Key: {api_key_display}\n\n"
                f"Error details: {error_msg}"
            ) from cloud_error
        elif "v1 API is deprecated" in error_msg or "v2 apis" in error_msg:
            # This is a known bug in ChromaDB 0.6.3 - CloudClient calls v1 during tenant validation
            # The workaround is to upgrade to the absolute latest version
            raise ValueError(
                f"ChromaDB v1 API issue detected - CloudClient initialization failed.\n\n"
                f"Current version: {chromadb.__version__}\n"
                f"This is a known bug where CloudClient calls v1 API during tenant validation.\n\n"
                f"SOLUTION: Upgrade to the absolute latest ChromaDB:\n"
                f"  pip uninstall chromadb -y\n"
                f"  pip install --upgrade chromadb\n\n"
                f"Or try a specific newer version:\n"
                f"  pip install chromadb>=0.6.5\n\n"
                f"If upgrade doesn't work, this may require a ChromaDB library fix.\n"
                f"Check: https://github.com/chroma-core/chroma/issues\n\n"
                f"Error: {error_msg}"
            ) from cloud_error
        elif "KeyError" in error_msg and "_type" in error_msg:
            raise ValueError(
                f"ChromaDB version compatibility issue detected (KeyError '_type').\n"
                f"Current version: {chromadb.__version__}\n"
                f"Try: pip install --upgrade 'chromadb>=0.5.0'\n\n"
                f"Error: {error_msg}"
            ) from cloud_error
        else:
            raise ValueError(
                f"Failed to connect to Chroma Cloud.\n\n"
                f"Error: {error_msg}\n\n"
                f"Please verify:\n"
                f"  1. CHROMA_API_KEY is set correctly\n"
                f"  2. Network connection to {config.host}\n"
                f"  3. Tenant and database names are correct\n"
                f"  4. ChromaDB version supports v2 API (run: pip show chromadb)"
            ) from cloud_error
    
    print("=" * 60)
    
    return _client


def get_collection(
    name: Optional[str] = None,
    create_if_missing: bool = True
) -> Collection:
    """
    Gets or creates the main documents collection.
    
    Args:
        name: Collection name (defaults to config value)
        create_if_missing: Whether to create if doesn't exist
    
    Returns:
        Chroma Collection object
    """
    client = get_chroma_client()
    config = get_config().chroma_cloud
    
    collection_name = name or config.collection_name
    
    if create_if_missing:
        collection = client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
    else:
        collection = client.get_collection(name=collection_name)
    
    print(f"Collection: {collection_name} ({collection.count()} documents)")
    
    return collection


def get_chat_history_collection() -> Collection:
    """
    Gets or creates the chat history collection.
    
    Separate collection for storing query/response embeddings.
    """
    client = get_chroma_client()
    
    collection = client.get_or_create_collection(
        name="chat_history",
        metadata={"hnsw:space": "cosine"}
    )
    
    return collection


def delete_collection(name: str) -> None:
    """Deletes a collection (use for fresh re-ingestion)."""
    client = get_chroma_client()
    
    try:
        client.delete_collection(name=name)
        print(f"Deleted collection: {name}")
    except Exception as e:
        print(f"Collection {name} not found or could not be deleted: {e}")
