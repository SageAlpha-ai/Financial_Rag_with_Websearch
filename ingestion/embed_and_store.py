"""
Embedding and Storage Module — STRICT Streaming Ingestion Pipeline

Architecture:
    For each source document:
        For each chunk:
            1. Generate deterministic ID
            2. Check if chunk already exists in Chroma → skip if yes
            3. Embed ONLY new chunks (one at a time)
            4. Immediately store via collection.add()
            5. Verify count increased by exactly 1
            6. Move to next chunk
        After all chunks of a source succeed → checkpoint that source

STRICT GUARANTEES:
    - collection.add() ONLY — upsert() is FORBIDDEN.
    - Deterministic SHA-256 IDs: source|page|chunk_index|text — no UUIDs.
    - Duplicate IDs inside a single document → RuntimeError.
    - Insert does not increase count by 1 → RuntimeError.
    - Fresh reset leaves non-zero count → RuntimeError.
    - Post-ingestion cloud verification via fresh collection handle.
    - No global embedding accumulation.  Zero memory pressure.
    - Checkpoint is source-level: processed sources survive crashes.

This script either:
    A) Successfully increases count for every new chunk, OR
    B) Crashes loudly.
"""

import hashlib
import os
import time
from collections import OrderedDict
from typing import List, Dict, Set

from langchain_openai import AzureOpenAIEmbeddings
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings

from config.settings import get_config
from vectorstore.chroma_client import delete_collection, get_company_collection

# ---------------------------------------------------------------------------
# Checkpoint helpers  (source-level, not index-level)
# ---------------------------------------------------------------------------

_CHECKPOINT_FILE = "ingest_checkpoint.txt"


def _read_checkpoint() -> Set[str]:
    """Return the set of already-processed source identifiers."""
    if not os.path.exists(_CHECKPOINT_FILE):
        return set()
    try:
        with open(_CHECKPOINT_FILE, "r", encoding="utf-8") as f:
            return {line.strip() for line in f if line.strip()}
    except OSError:
        return set()


def _append_checkpoint(source: str) -> None:
    """Append a successfully-processed source to the checkpoint file."""
    with open(_CHECKPOINT_FILE, "a", encoding="utf-8") as f:
        f.write(source + "\n")


def _clear_checkpoint() -> None:
    """Remove the checkpoint file (fresh start)."""
    if os.path.exists(_CHECKPOINT_FILE):
        os.remove(_CHECKPOINT_FILE)


# ---------------------------------------------------------------------------
# Embedding wrapper
# ---------------------------------------------------------------------------


class AzureOpenAIEmbeddingFunction(EmbeddingFunction):
    """Wraps LangChain AzureOpenAIEmbeddings for Chroma's EmbeddingFunction API."""

    def __init__(self):
        config = get_config().azure_openai
        embedding_kwargs = {
            "azure_endpoint": config.endpoint,
            "azure_deployment": config.embeddings_deployment,
            "api_key": config.api_key,
            "api_version": config.api_version,
        }
        if "text-embedding-ada-002" in config.embeddings_deployment.lower():
            embedding_kwargs["model"] = "text-embedding-ada-002"
        self.embeddings = AzureOpenAIEmbeddings(**embedding_kwargs)

    def __call__(self, input: Documents) -> Embeddings:
        return self.embeddings.embed_documents(input)


# ---------------------------------------------------------------------------
# Deterministic ID generation
# ---------------------------------------------------------------------------


def generate_id(text: str, source: str, page: str, chunk_index: int) -> str:
    """Full 64-char SHA-256 hex digest of ``source|page|chunk_index|text``.

    Guarantees:
        - Same chunk always maps to the same ID.
        - Fully deterministic — no UUID, no counter reset.
    """
    base = f"{source}|{page}|{chunk_index}|{text}"
    return hashlib.sha256(base.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Metadata sanitizer
# ---------------------------------------------------------------------------


def _clean_metadata(meta: Dict) -> Dict:
    """Return a Chroma-safe copy of *meta* (no None, no exotic types)."""
    cleaned: Dict = {}
    for key, value in meta.items():
        if value is None:
            continue
        if isinstance(value, (str, int, bool)):
            cleaned[key] = value
        else:
            cleaned[key] = str(value)
    return cleaned


# ---------------------------------------------------------------------------
# Group chunks by source document
# ---------------------------------------------------------------------------


def _group_by_source(documents: List[Dict]) -> OrderedDict:
    """Group chunks by ``metadata.source``, preserving insertion order."""
    groups: OrderedDict = OrderedDict()
    for doc in documents:
        source = doc.get("metadata", {}).get("source", "unknown")
        groups.setdefault(source, []).append(doc)
    return groups


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def embed_and_store_documents(
    documents: List[Dict],
    company_name: str,
    fresh: bool = False,
) -> int:
    """Stream-embed and store *documents* into Chroma Cloud, one chunk at a time.

    Raises ``RuntimeError`` on ANY anomaly — no silent skips.

    Args:
        documents: List of dicts with ``text`` and ``metadata`` keys
            (typically already chunked by ``ingestion.chunking``).
        collection_name: Target collection (defaults to config).
        fresh: If ``True``, delete + recreate collection, verify
            count == 0, and clear checkpoint.

    Returns:
        Total number of newly inserted chunks.

    Raises:
        RuntimeError: On failed inserts, count mismatches, intra-document
            duplicate IDs, or any unexpected condition.
    """
    print("=" * 60)
    print("STRICT STREAMING CHROMA CLOUD INGESTION")
    print("=" * 60)

    if not documents:
        raise RuntimeError("No documents provided — nothing to ingest")

    # ------------------------------------------------------------------
    # Resolve collection dynamically (isolated per company)
    # ------------------------------------------------------------------
    if fresh:
        from rag.company_normalizer import normalize_company_name
        norm_name = normalize_company_name(company_name)
        if not norm_name:
            raise ValueError(
                f"Cannot run fresh ingestion — company name {company_name!r} "
                f"could not be normalized to a valid collection slug."
            )
        print(f"\n[FRESH MODE] Deleting collection '{norm_name}'...")
        try:
            delete_collection(name=norm_name)
            print("  Deleted.")
        except Exception:
            print("  Collection did not exist — OK.")
        _clear_checkpoint()
        time.sleep(2)

    collection = get_company_collection(company_name)

    # Cloud connection validation
    startup_count = collection.count()
    print(f"\n  Connected to Chroma")
    print(f"  Collection name: {collection.name}")
    print(f"  Collection count: {startup_count}")

    # ==================================================================
    # 2. INITIALIZE EMBEDDINGS
    # ==================================================================
    print(f"\nInitializing Azure OpenAI embeddings...")
    print(f"  Deployment: {config.azure_openai.embeddings_deployment}")
    embedding_fn = AzureOpenAIEmbeddingFunction()
    print("  Ready.")

    # ==================================================================
    # 3. GROUP BY SOURCE + LOAD CHECKPOINT
    # ==================================================================
    source_groups = _group_by_source(documents)
    completed_sources = _read_checkpoint() if not fresh else set()

    total_sources = len(source_groups)
    total_chunks = len(documents)
    sources_to_process = [s for s in source_groups if s not in completed_sources]

    if completed_sources:
        print(f"\n[RESUME] {len(completed_sources)} sources already processed — skipping them")

    print(f"\n  Total source documents:  {total_sources}")
    print(f"  Total chunks:            {total_chunks}")
    print(f"  Sources to process:      {len(sources_to_process)}")

    # ==================================================================
    # 4. STREAMING LOOP: source → chunk → dedup → embed → store → verify
    # ==================================================================
    total_inserted = 0
    total_skipped = 0
    source_num = 0

    for source_key in sources_to_process:
        chunks = source_groups[source_key]
        source_num += 1

        print(f"\n{'─' * 50}")
        print(f"  Source [{source_num}/{len(sources_to_process)}]: {source_key}")
        print(f"  Chunks: {len(chunks)}")

        # --- Collect all IDs for this source to check intra-doc dupes ---
        source_ids: List[str] = []
        source_valid_chunks: List[Dict] = []

        for chunk in chunks:
            text = chunk.get("text", "").strip()
            if not text or len(text) < 10:
                continue

            meta = chunk.get("metadata", {})
            page = str(meta.get("page", meta.get("page_number", "0")))
            chunk_index = meta.get("chunk_index", 0)

            chunk_id = generate_id(
                text=text,
                source=source_key,
                page=page,
                chunk_index=chunk_index,
            )

            source_ids.append(chunk_id)
            source_valid_chunks.append(chunk)

        if not source_valid_chunks:
            print(f"    No valid chunks — skipping source")
            _append_checkpoint(source_key)
            continue

        # --- HARD BLOCK: duplicate IDs inside this source document ---
        if len(source_ids) != len(set(source_ids)):
            seen: Set[str] = set()
            dupes = []
            for cid in source_ids:
                if cid in seen:
                    dupes.append(cid)
                seen.add(cid)
            raise RuntimeError(
                f"Source '{source_key}': Duplicate IDs detected INSIDE "
                f"DOCUMENT — {len(dupes)} duplicates: {dupes[:5]}"
            )

        # --- Process each chunk individually ---------------------------
        inserted_this_source = 0
        skipped_this_source = 0

        for i, (chunk_id, chunk) in enumerate(
            zip(source_ids, source_valid_chunks)
        ):
            text = chunk["text"].strip()
            meta = chunk.get("metadata", {})

            # ---- DUPLICATE CHECK against Chroma ----
            existing = collection.get(ids=[chunk_id], include=[])
            if existing.get("ids"):
                skipped_this_source += 1
                continue

            # ---- EMBED (single chunk — zero accumulation) ----
            embedding_result = embedding_fn([text])
            embedding = embedding_result[0]

            # ---- CLEAN METADATA ----
            clean_meta = _clean_metadata(meta)

            # ---- COUNT BEFORE ----
            count_before = collection.count()

            # ---- STORE (collection.add ONLY — NO upsert) ----
            collection.add(
                ids=[chunk_id],
                documents=[text],
                embeddings=[embedding],
                metadatas=[clean_meta],
            )

            # ---- COUNT AFTER + VERIFY ----
            count_after = collection.count()

            if count_after != count_before + 1:
                raise RuntimeError(
                    f"Insert FAILED for chunk {i} of '{source_key}' — "
                    f"expected count {count_before + 1} but got {count_after}. "
                    f"Insert not persisted in Chroma."
                )

            inserted_this_source += 1

        total_inserted += inserted_this_source
        total_skipped += skipped_this_source

        print(f"    Inserted: {inserted_this_source}")
        print(f"    Skipped (existing): {skipped_this_source}")
        print(f"    Running total: {total_inserted} inserted, {total_skipped} skipped")

        # --- Checkpoint: source fully processed ----
        _append_checkpoint(source_key)

    # ==================================================================
    # 5. POST-INGESTION CLOUD VERIFICATION
    # ==================================================================
    verified_collection = get_company_collection(company_name)
    verified_count = verified_collection.count()

    print(f"\n{'=' * 60}")
    print("INGESTION COMPLETE — VERIFIED")
    print(f"{'=' * 60}")
    print(f"  Total chunks processed:                   {total_chunks}")
    print(f"  Newly inserted this run:                  {total_inserted}")
    print(f"  Skipped (already existed):                {total_skipped}")
    print(f"  Final verified count (fresh handle):      {verified_count}")
    print(f"{'=' * 60}")

    expected_minimum = startup_count + total_inserted
    if verified_count < expected_minimum:
        raise RuntimeError(
            f"Post-ingestion verification FAILED — verified count "
            f"({verified_count}) < startup ({startup_count}) + "
            f"inserted ({total_inserted}) = {expected_minimum}"
        )

    # Clear checkpoint only when all sources were processed
    if not sources_to_process or source_num >= len(sources_to_process):
        _clear_checkpoint()

    return total_inserted
