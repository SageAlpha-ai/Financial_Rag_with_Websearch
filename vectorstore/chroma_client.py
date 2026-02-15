"""
Chroma Cloud Client

STRICT RULES:
- NEVER use persist_directory
- NEVER use PersistentClient
- ONLY use HttpClient for Chroma Cloud
- ALL embeddings go to Chroma Cloud
- Local disk stores ZERO vectors
"""

import logging
from typing import List, Optional
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


# ---------------------------------------------------------------------------
# Multi-collection listing
# ---------------------------------------------------------------------------

_logger = logging.getLogger(__name__)

# Collections that should be excluded from cross-collection search
_SKIP_COLLECTIONS: set = {"chat_history"}


def normalize_collection_name(name: str) -> str:
    """Normalize collection name for deduplication and allowlist/denylist."""
    return name.lower().replace("-", "").replace("_", "").strip()


def list_all_collections(skip_internal: bool = True) -> List[Collection]:
    """List all collections in the current Chroma database.

    Used by the multi-collection retriever to search across every
    company/dataset without relying on company-name-to-collection
    resolution. When collection governance is enabled, normalizes names,
    deduplicates by normalized name, and applies allowlist/denylist.

    Args:
        skip_internal: When ``True`` (default), excludes internal
            collections such as ``chat_history``.

    Returns:
        A list of ``Collection`` objects, each ready to be queried.
        Returns an empty list on failure (never raises).
    """
    try:
        client = get_chroma_client()
        raw_collections = client.list_collections()

        # ChromaDB >= 0.5 returns Collection objects with .name
        # Filter out internal collections
        if skip_internal:
            raw_collections = [
                c for c in raw_collections
                if getattr(c, "name", "") not in _SKIP_COLLECTIONS
            ]

        gov = get_config().collection_governance or {}
        if not gov.get("enabled"):
            _logger.info(
                "[CHROMA] Listed %d searchable collections: %s",
                len(raw_collections),
                [getattr(c, "name", str(c)) for c in raw_collections],
            )
            return raw_collections

        # Governance: pre-normalize allowlist/denylist once
        governance_cfg = gov
        raw_allowlist = governance_cfg.get("allowlist", [])
        raw_denylist = governance_cfg.get("denylist", [])

        normalized_allowlist = {
            normalize_collection_name(name) for name in raw_allowlist
        }
        normalized_denylist = {
            normalize_collection_name(name) for name in raw_denylist
        }

        normalize = governance_cfg.get("normalize", True)

        # Deduplicate by normalized name (keep first occurrence), then apply allowlist/denylist
        seen_norm: set = set()
        deduped: List[Collection] = []
        for c in raw_collections:
            orig_name = getattr(c, "name", str(c))
            norm_name = normalize_collection_name(orig_name) if normalize else orig_name
            if norm_name in seen_norm:
                continue
            seen_norm.add(norm_name)
            if norm_name in normalized_denylist:
                continue
            if normalized_allowlist and norm_name not in normalized_allowlist:
                continue
            deduped.append(c)

        original_count = len(raw_collections)
        final_count = len(deduped)
        duplicates_removed = original_count - final_count
        cleaned_collection_names = [getattr(c, "name", str(c)) for c in deduped]

        _logger.info(
            "collection_governance_applied",
            extra={
                "event_type": "collection_governance",
                "original_count": original_count,
                "final_count": final_count,
                "duplicates_removed": duplicates_removed,
                "collections": cleaned_collection_names,
            },
        )
        return deduped
    except Exception as e:
        _logger.error("[CHROMA] Failed to list collections: %s", e)
        return []


from rag.company_normalizer import normalize_company_name

def get_company_collection(company_name: str) -> Optional[Collection]:
    """
    Gets or creates a collection for a specific company.

    Args:
        company_name: Raw company name from query or planner.

    Returns:
        Chroma ``Collection`` object, or **None** when the company name
        cannot be normalised or the collection cannot be reached.

        The storage layer **never** raises for a missing or un-resolvable
        collection — it reports availability and lets the caller decide
        routing.
    """
    import logging
    logger = logging.getLogger(__name__)

    normalized_name = normalize_company_name(company_name)
    if not normalized_name:
        logger.warning(
            "[CHROMA] collection_missing=%s (normalisation failed)",
            company_name,
        )
        return None

    try:
        client = get_chroma_client()

        logger.info(f"[CHROMA] Dataset: {get_config().chroma_cloud.database}")
        logger.info(f"[CHROMA] Resolved Collection: {normalized_name}")

        collection = client.get_or_create_collection(
            name=normalized_name,
            metadata={"hnsw:space": "cosine"}
        )

        doc_count = collection.count()
        logger.info(f"[CHROMA] Collection '{normalized_name}' has {doc_count} documents")
        return collection

    except Exception as exc:
        logger.warning(
            "[CHROMA] collection_missing=%s error=%s",
            normalized_name,
            exc,
        )
        return None


def get_collection(
    name: Optional[str] = None,
    create_if_missing: bool = True
) -> Optional[Collection]:
    """
    DEPRECATED: Use get_company_collection for per-company isolation.
    Gets or creates a generic collection.

    Returns ``None`` when the collection cannot be accessed (non-fatal).
    """
    import os
    import logging

    logger = logging.getLogger(__name__)

    try:
        client = get_chroma_client()
    except Exception as exc:
        logger.warning("[CHROMA] collection_missing=<default> client_error=%s", exc)
        return None

    config = get_config().chroma_cloud

    collection_name = name or config.collection_name
    if not collection_name:
        logger.warning("[CHROMA] collection_missing=<empty> — no collection name configured")
        return None

    # Check if auto-creation is enabled via environment variable
    auto_create_env = os.getenv("CHROMA_AUTO_CREATE_COLLECTION", "").lower()

    if auto_create_env == "true":
        create_if_missing = True
    elif auto_create_env == "false":
        create_if_missing = False

    logger.info(f"[CHROMA] Getting collection: {collection_name} (Dataset: {config.database})")

    try:
        if create_if_missing:
            collection = client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
        else:
            collection = client.get_collection(name=collection_name)

        doc_count = collection.count()
        logger.info(f"[CHROMA] Collection '{collection_name}' ready with {doc_count} documents")
        return collection

    except Exception as e:
        logger.warning(
            "[CHROMA] collection_missing=%s error=%s", collection_name, e
        )
        return None


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

def verify_and_recreate_collection(name: str) -> Collection:
    """
    Performs a health check on the collection. 
    If it's corrupted or unreachable in a way that suggests a logic error,
    it attempts to delete and recreate it.
    """
    client = get_chroma_client()
    try:
        # Health check: try to fetch 1 item
        collection = client.get_collection(name=name)
        collection.peek(limit=1)
        return collection
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(f"Collection '{name}' health check failed: {e}. Attempting auto-recreation...")
        
        try:
            delete_collection(name)
            return client.create_collection(
                name=name,
                metadata={"hnsw:space": "cosine"}
            )
        except Exception as recreate_err:
            logger.error(f"Auto-recreation failed for '{name}': {recreate_err}")
            raise RuntimeError(f"Critical Chroma failure: Collection '{name}' is corrupt and cannot be recreated.") from recreate_err
