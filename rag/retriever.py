"""
Document Retriever

Queries Chroma Cloud for relevant documents.
Uses embedding-based retrieval (compatible with langchain_orchestrator).
Supports year-aware filtering for financial queries.
"""

import hashlib
import logging
import re
from typing import List, Dict, Optional, Tuple

from langchain_openai import AzureOpenAIEmbeddings

from config.settings import get_config
from vectorstore.chroma_client import get_collection, get_company_collection, list_all_collections
from rag.company_extractor import extract_company_name

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_embeddings() -> AzureOpenAIEmbeddings:
    """Build the shared Azure OpenAI embedding client.

    Uses the same deployment and credentials as ``langchain_orchestrator``
    to guarantee embedding-space consistency across the entire pipeline.
    """
    config = get_config()
    embedding_kwargs = {
        "azure_endpoint": config.azure_openai.endpoint,
        "azure_deployment": config.azure_openai.embeddings_deployment,
        "api_key": config.azure_openai.api_key,
        "api_version": config.azure_openai.api_version,
    }
    # Only add explicit model parameter for older ada-002 deployments
    if "text-embedding-ada-002" in config.azure_openai.embeddings_deployment.lower():
        embedding_kwargs["model"] = "text-embedding-ada-002"
    return AzureOpenAIEmbeddings(**embedding_kwargs)


def _deduplicate(
    documents: List[str],
    metadatas: List[Dict],
) -> Tuple[List[str], List[Dict]]:
    """Deduplicate documents by metadata id or content hash.

    Uses ``metadata.id`` when present; otherwise falls back to a SHA-256
    digest of the normalised (stripped, lowercased) text.  This is more
    robust than raw string equality, which breaks on trivial whitespace
    or encoding differences between retrieval sources.
    """
    seen: set = set()
    deduped_docs: List[str] = []
    deduped_metas: List[Dict] = []
    for doc, meta in zip(documents, metadatas):
        # Prefer metadata id if available; otherwise hash the text
        doc_key = meta.get("id") or hashlib.sha256(
            doc.strip().lower().encode("utf-8")
        ).hexdigest()
        if doc_key not in seen:
            seen.add(doc_key)
            deduped_docs.append(doc)
            deduped_metas.append(meta)
    return deduped_docs, deduped_metas


# ---------------------------------------------------------------------------
# Year extraction
# ---------------------------------------------------------------------------


def extract_year_from_query(query: str) -> Optional[str]:
    """
    Extracts fiscal year from query.
    Returns normalized FYxxxx format.
    """
    patterns = [
        r'FY\s*(\d{4})',           # FY2023
        r'fiscal\s+year\s+(\d{4})', # fiscal year 2023
        r'\b(20\d{2})\b',          # 2023
        r'\b(19\d{2})\b',          # 1999
    ]
    
    for pattern in patterns:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            year = match.group(1)
            return f"FY{year}"
    
    return None


# ---------------------------------------------------------------------------
# Public retrieval API
# ---------------------------------------------------------------------------


def retrieve_documents(
    query: str,
    n_results: int = 10,
    company_name: str = None
) -> Tuple[List[str], List[Dict]]:
    """
    Retrieves documents from Chroma Cloud using embedding-based search.

    Uses the same Azure OpenAI embedding model as ingestion and the
    orchestrator to ensure vector-space consistency.  Results are
    deduplicated by content hash before being returned.

    Args:
        query: User query text
        n_results: Number of results to return
        collection_name: Target collection

    Returns:
        Tuple of (documents, metadatas).  Empty on failure.
    """
    try:
        # Resolve company
        company = company_name or extract_company_name(query)
        if not company:
            raise ValueError("Company name could not be resolved.")
        collection = get_company_collection(company)
        embeddings = _get_embeddings()
        query_embedding = embeddings.embed_query(query)

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas"]
        )

        documents = results["documents"][0] if results["documents"] else []
        metadatas = results["metadatas"][0] if results["metadatas"] else []

        return _deduplicate(documents, metadatas)
    except Exception as e:
        logger.error("[RETRIEVER] retrieve_documents failed: %s", e)
        return [], []


def retrieve_with_year_filter(
    query: str,
    n_results: int = 10,
    company_name: str = None
) -> Tuple[List[str], List[Dict], Optional[str]]:
    """
    Retrieves documents with strict year-aware filtering.

    If the query contains a fiscal-year reference:
      - Retrieves ONLY documents whose ``metadata.fiscal_year`` matches.
      - Returns EMPTY when no year-matched documents exist.
      - Year backfilling is intentionally prohibited: returning documents
        from a *different* fiscal year would produce incorrect financial
        figures and violate the answerability contract enforced downstream
        by ``_is_answerable``.

    If no year is detected:
      - Performs standard embedding-based retrieval across all years.

    Empty results are preferable to mismatched evidence because the
    downstream answerability gate treats empty retrieval as "no evidence
    available" — triggering a safe NO_ANSWER or LLM-only disclaimer —
    rather than silently serving wrong-year data as correct.

    Args:
        query: User query text
        n_results: Number of results to return
        collection_name: Target collection

    Returns:
        Tuple of (documents, metadatas, requested_year).  Empty on failure.
    """
    try:
        # Resolve company
        company = company_name or extract_company_name(query)
        if not company:
            raise ValueError("Company name could not be resolved.")
        collection = get_company_collection(company)
        embeddings = _get_embeddings()
        query_embedding = embeddings.embed_query(query)
    except Exception as e:
        logger.error("[RETRIEVER] Failed to initialise retriever: %s", e)
        return [], [], None

    requested_year = extract_year_from_query(query)

    if requested_year:
        # ------------------------------------------------------------ #
        # STRICT year-filtered retrieval.                                #
        # Only documents whose metadata.fiscal_year matches the          #
        # requested year are returned.  If zero match, we return EMPTY   #
        # rather than backfilling with other years — wrong-year data     #
        # would be treated as valid evidence by the LLM and produce      #
        # incorrect financial answers.                                    #
        # ------------------------------------------------------------ #
        try:
            year_results = collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where={"fiscal_year": requested_year},
                include=["documents", "metadatas"]
            )
            documents = (
                year_results["documents"][0]
                if year_results["documents"] else []
            )
            metadatas = (
                year_results["metadatas"][0]
                if year_results["metadatas"] else []
            )
        except Exception as e:
            logger.error(
                "[RETRIEVER] Year-filtered retrieval failed for %s: %s",
                requested_year, e,
            )
            documents = []
            metadatas = []

        if documents:
            documents, metadatas = _deduplicate(documents, metadatas)
            logger.info(
                "[RETRIEVER] Year-filtered retrieval: %d documents for %s",
                len(documents), requested_year,
            )
        else:
            # No documents for the requested year — return empty.
            # Backfilling with other years is intentionally disallowed:
            # mismatched fiscal-year evidence is worse than no evidence
            # because it leads to silently incorrect numeric answers.
            logger.info(
                "[RETRIEVER] Year-filtered retrieval: 0 documents for %s "
                "— returning empty (year backfill prohibited)",
                requested_year,
            )
    else:
        # No year specified — general retrieval across all years.
        try:
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=["documents", "metadatas"]
            )
            documents = results["documents"][0] if results["documents"] else []
            metadatas = results["metadatas"][0] if results["metadatas"] else []
            documents, metadatas = _deduplicate(documents, metadatas)
        except Exception as e:
            logger.error("[RETRIEVER] General retrieval failed: %s", e)
            documents = []
            metadatas = []

    return documents, metadatas, requested_year


# ---------------------------------------------------------------------------
# Multi-collection retrieval
# ---------------------------------------------------------------------------

#: Collections to exclude from cross-collection search
_TOP_K_PER_COLLECTION: int = 5
_FINAL_TOP_K: int = 5


def retrieve_across_collections(
    query: str,
    n_results: int = _FINAL_TOP_K,
) -> Dict:
    """Search ALL collections in the current Chroma database.

    Iterates over every collection, performs cosine similarity search on
    each (``top_k_per_collection = 5``), aggregates results globally,
    sorts by distance, deduplicates, and returns the final top-k.

    This function does NOT rely on company name to resolve a collection.

    Args:
        query: User query text.
        n_results: Final number of documents to return after global sort.

    Returns:
        A dict with the following keys::

            {
                "documents":        List[str],
                "metadatas":        List[Dict],
                "avg_similarity":   float,
                "top_score":        float,
                "collections_used": List[str],
            }

        If all collections return 0 documents, ``documents`` is empty and
        a warning is logged.  The caller should trigger web/LLM fallback.
    """
    empty_result: Dict = {
        "documents": [],
        "metadatas": [],
        "avg_similarity": 0.0,
        "top_score": 0.0,
        "collections_used": [],
    }

    try:
        collections = list_all_collections(skip_internal=True)
    except Exception as e:
        logger.error("[RETRIEVER] Failed to list collections: %s", e)
        return empty_result

    if not collections:
        logger.warning("[RETRIEVER] No searchable collections found")
        return empty_result

    logger.info(
        "[RETRIEVER] Multi-collection search across %d collections",
        len(collections),
    )

    # Generate embedding once
    try:
        embeddings = _get_embeddings()
        query_embedding = embeddings.embed_query(query)
    except Exception as e:
        logger.error("[RETRIEVER] Embedding generation failed: %s", e)
        return empty_result

    requested_year = extract_year_from_query(query)

    # (doc_text, metadata, distance, collection_name)
    all_candidates: list = []
    collections_with_hits: List[str] = []

    for collection in collections:
        col_name = getattr(collection, "name", str(collection))
        try:
            # Year-filtered retrieval
            if requested_year:
                try:
                    yr = collection.query(
                        query_embeddings=[query_embedding],
                        n_results=_TOP_K_PER_COLLECTION,
                        where={"fiscal_year": requested_year},
                        include=["documents", "metadatas", "distances"],
                    )
                    yr_docs = yr.get("documents", [[]])[0] if yr.get("documents") else []
                    yr_metas = yr.get("metadatas", [[]])[0] if yr.get("metadatas") else []
                    yr_dists = yr.get("distances", [[]])[0] if yr.get("distances") else []
                    for d, m, dist in zip(yr_docs, yr_metas, yr_dists):
                        m["_source_collection"] = col_name
                        all_candidates.append((d, m, dist, col_name))
                except Exception as ye:
                    logger.warning("[RETRIEVER] [%s] Year filter failed: %s", col_name, ye)

            # General retrieval
            res = collection.query(
                query_embeddings=[query_embedding],
                n_results=_TOP_K_PER_COLLECTION,
                include=["documents", "metadatas", "distances"],
            )
            g_docs = res.get("documents", [[]])[0] if res.get("documents") else []
            g_metas = res.get("metadatas", [[]])[0] if res.get("metadatas") else []
            g_dists = res.get("distances", [[]])[0] if res.get("distances") else []

            for d, m, dist in zip(g_docs, g_metas, g_dists):
                m["_source_collection"] = col_name
                all_candidates.append((d, m, dist, col_name))

            if g_docs:
                collections_with_hits.append(col_name)

            logger.info(
                "[RETRIEVER] [%s] returned %d docs (best dist: %.4f)",
                col_name,
                len(g_docs),
                min(g_dists) if g_dists else 2.0,
            )
        except Exception as col_err:
            logger.warning("[RETRIEVER] Collection '%s' failed (skipping): %s", col_name, col_err)
            continue

    # Sort globally by distance (ascending = most similar first)
    all_candidates.sort(key=lambda x: x[2])

    # Deduplicate
    seen: set = set()
    deduped: list = []
    for doc, meta, dist, cname in all_candidates:
        key = doc.strip().lower()
        if key not in seen:
            seen.add(key)
            deduped.append((doc, meta, dist, cname))

    # Final top-k
    top = deduped[:n_results]

    documents = [r[0] for r in top]
    metadatas = [r[1] for r in top]
    distances = [r[2] for r in top]

    if not documents:
        logger.warning(
            "[RETRIEVER] ALL collections returned 0 documents — trigger web/LLM fallback"
        )
        return empty_result

    top_score = max(0.0, 1.0 - (distances[0] / 2.0)) if distances else 0.0
    avg_sim = (
        sum(max(0.0, 1.0 - (d / 2.0)) for d in distances) / len(distances)
        if distances
        else 0.0
    )

    result = {
        "documents": documents,
        "metadatas": metadatas,
        "avg_similarity": round(avg_sim, 4),
        "top_score": round(top_score, 4),
        "collections_used": list(set(r[3] for r in top)),
    }

    logger.info(
        "[RETRIEVER] Multi-collection result: %d docs, top_score=%.4f, "
        "avg_similarity=%.4f, collections=%s",
        len(documents),
        result["top_score"],
        result["avg_similarity"],
        result["collections_used"],
    )

    return result
