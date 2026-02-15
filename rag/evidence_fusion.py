"""
Evidence Fusion Module

Merges evidence from multiple sources (RAG, Web Search) with priority rules.
Resolves conflicts and produces unified evidence for LLM summarization.
"""

import logging
import re
from typing import Dict, List, Optional, Tuple
from datetime import datetime

from rag.query_constraints import extract_fiscal_year

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lightweight keyword-based entity extractor for the consistency filter.
# This is intentionally NOT the LLM-backed extract_company_name — the
# consistency filter must run without LLM calls.
# ---------------------------------------------------------------------------

def _extract_entity_from_query(query: str) -> Optional[str]:
    """Extract company/entity from *query* using keyword matching."""
    query_lower = query.lower()
    entity_mappings = {
        "oracle financial services": "Oracle Financial Services Software Ltd",
        "oracle financial": "Oracle Financial Services Software Ltd",
        "ofss": "Oracle Financial Services Software Ltd",
        "microsoft": "Microsoft",
        "apple": "Apple",
        "google": "Google",
        "amazon": "Amazon",
        "meta": "Meta",
        "facebook": "Meta",
        "tesla": "Tesla",
        "nvidia": "NVIDIA",
    }
    for key, value in entity_mappings.items():
        if key in query_lower:
            return value
    return None


class EvidenceFusion:
    """
    Fuses evidence from multiple sources with strict priority rules.
    
    Priority order:
    1. Official audited documents (from web)
    2. Official company documents (from web)
    3. Internal RAG documents (ChromaDB)
    4. Press releases / news (lowest priority)
    """
    
    @staticmethod
    def fuse_evidence(
        rag_documents: List[str],
        rag_metadatas: List[Dict],
        web_documents: List[Dict],
        query: str
    ) -> Dict[str, any]:
        """
        Fuse evidence from RAG and web search.
        
        Args:
            rag_documents: Documents from ChromaDB
            rag_metadatas: Metadata for RAG documents
            web_documents: Documents from web search (with metadata)
            query: Original user query
            
        Returns:
            Dict with:
            - fused_context: str (combined context for LLM)
            - sources: List[str] (source citations)
            - evidence_priority: str (which source was primary)
            - conflicts: List[Dict] (any conflicting data)
        """
        logger.info(f"[FUSION] Fusing evidence: {len(rag_documents)} RAG docs, {len(web_documents)} web docs")
        
        # ---------------------------------------------------------------- #
        # Consistency filter (soft penalty, no hard drops)                  #
        # Documents whose metadata contradicts the query's fiscal-year or   #
        # entity constraints are DEMOTED (priority penalty) but never       #
        # discarded.  Hard drops caused loss of valuable numeric context    #
        # for queries where the only available evidence was from a nearby   #
        # fiscal year.  The Tier-1 guardrail (_is_answerable) remains the   #
        # authoritative answerability gate.                                 #
        # ---------------------------------------------------------------- #
        requested_year = extract_fiscal_year(query)
        requested_entity = _extract_entity_from_query(query)

        # Tag documents with penalty flags — consumed in prioritization.
        if requested_year or requested_entity:
            for meta in rag_metadatas:
                doc_year = meta.get("fiscal_year", "")
                doc_company = meta.get("company", "")

                if (
                    requested_year
                    and doc_year
                    and requested_year.lower() != doc_year.lower()
                ):
                    meta["_year_penalty"] = True
                    logger.info(
                        "[FUSION] Year mismatch penalty applied "
                        "(requested=%s, doc_year=%s, company=%s)",
                        requested_year, doc_year, doc_company,
                    )

                if requested_entity and doc_company:
                    if (
                        requested_entity.lower() not in doc_company.lower()
                        and doc_company.lower() not in requested_entity.lower()
                    ):
                        meta["_entity_penalty"] = True
                        logger.info(
                            "[FUSION] Entity mismatch penalty applied "
                            "(requested=%s, doc_company=%s)",
                            requested_entity, doc_company,
                        )

            for web_doc in web_documents:
                doc_meta = web_doc.get("metadata", {})
                doc_year = doc_meta.get("fiscal_year", "")
                doc_company = doc_meta.get("company", "")

                if (
                    requested_year
                    and doc_year
                    and requested_year.lower() != doc_year.lower()
                ):
                    doc_meta["_year_penalty"] = True
                    logger.info(
                        "[FUSION] Web year mismatch penalty applied "
                        "(requested=%s, doc_year=%s)",
                        requested_year, doc_year,
                    )

                if requested_entity and doc_company:
                    if (
                        requested_entity.lower() not in doc_company.lower()
                        and doc_company.lower() not in requested_entity.lower()
                    ):
                        doc_meta["_entity_penalty"] = True
                        logger.info(
                            "[FUSION] Web entity mismatch penalty applied "
                            "(requested=%s, doc_company=%s)",
                            requested_entity, doc_company,
                        )

            logger.info(
                "[FUSION] After consistency filter: %d RAG docs, %d web docs",
                len(rag_documents), len(web_documents),
            )

        # Prioritize documents by: penalty (False first) → trust_score
        # (descending) → cross_score (descending).
        prioritized_docs = []
        sources = []
        conflicts = []

        # Add web documents.
        for web_doc in web_documents:
            doc_text = web_doc.get("text", "")
            doc_meta = web_doc.get("metadata", {})

            if doc_text:
                has_penalty = bool(
                    doc_meta.get("_year_penalty")
                    or doc_meta.get("_entity_penalty")
                )
                prioritized_docs.append({
                    "text": doc_text,
                    "metadata": doc_meta,
                    "source_type": "web_official",
                    "has_penalty": has_penalty,
                    "trust_score": float(doc_meta.get("trust_score", 0.5)),
                    "cross_score": float(doc_meta.get("cross_score", 0.0)),
                })

                # Add source citation
                source_url = doc_meta.get("url", "")
                source_title = doc_meta.get("title", "Official Document")
                if source_url:
                    sources.append(f"{source_title} - {source_url}")
                else:
                    sources.append(source_title)

        # Add RAG documents.
        for doc, meta in zip(rag_documents, rag_metadatas):
            has_penalty = bool(
                meta.get("_year_penalty") or meta.get("_entity_penalty")
            )
            prioritized_docs.append({
                "text": doc,
                "metadata": meta,
                "source_type": "rag_internal",
                "has_penalty": has_penalty,
                "trust_score": float(meta.get("trust_score", 0.75)),
                "cross_score": float(meta.get("cross_score", 0.0)),
            })

            # Add source citation
            source_info = meta.get("source", meta.get("filename", "Internal Document"))
            if meta.get("page"):
                source_info += f" (page {meta.get('page')})"
            if meta.get("fiscal_year"):
                source_info += f" (FY: {meta.get('fiscal_year')})"
            sources.append(source_info)

        # Sort by: penalty flag (False < True), then trust_score desc,
        # then cross_score desc.
        prioritized_docs.sort(
            key=lambda d: (
                d["has_penalty"],          # False (0) before True (1)
                -d["trust_score"],         # higher trust first
                -d["cross_score"],         # higher relevance first
            )
        )
        logger.info("[FUSION] sorted_by_trust_then_relevance")

        # Check for conflicts (same metric, different values)
        conflicts = EvidenceFusion._detect_conflicts(prioritized_docs, query)

        # Build fused context
        context_parts = []
        for doc_info in prioritized_docs:
            text = doc_info["text"]
            meta = doc_info["metadata"]
            source_type = doc_info["source_type"]
            
            # Add source annotation
            if source_type == "web_official":
                meta_info = "[OFFICIAL SOURCE] "
                if meta.get("url"):
                    meta_info += f"Source: {meta.get('url')}"
                if meta.get("title"):
                    meta_info += f" - {meta.get('title')}"
            else:
                meta_info = "[INTERNAL DOCUMENT] "
                if meta.get("source"):
                    meta_info += f"Source: {meta.get('source')}"
                if meta.get("fiscal_year"):
                    meta_info += f", FY: {meta.get('fiscal_year')}"
            
            context_parts.append(f"{meta_info}\n{text}")
        
        fused_context = "\n\n---\n\n".join(context_parts)
        
        # Determine primary evidence source
        if web_documents:
            evidence_priority = "web_official"
        elif rag_documents:
            evidence_priority = "rag_internal"
        else:
            evidence_priority = "none"
        
        logger.info(f"[FUSION] Evidence fused: priority={evidence_priority}, conflicts={len(conflicts)}")
        
        return {
            "fused_context": fused_context,
            "sources": sources,
            "evidence_priority": evidence_priority,
            "conflicts": conflicts,
            "rag_count": len(rag_documents),
            "web_count": len(web_documents)
        }
    
    @staticmethod
    def _detect_conflicts(
        documents: List[Dict],
        query: str
    ) -> List[Dict]:
        """
        Detect conflicts in financial data.
        
        Looks for same metric with different values.
        """
        conflicts = []
        
        # Extract financial metrics from query
        query_lower = query.lower()
        metrics = []
        if "revenue" in query_lower:
            metrics.append("revenue")
        if "profit" in query_lower or "income" in query_lower:
            metrics.append("profit")
        if "earnings" in query_lower:
            metrics.append("earnings")
        
        # Extract values for each metric from documents
        metric_values = {}
        for doc_info in documents:
            text = doc_info["text"].lower()
            source = doc_info["metadata"].get("source", "unknown")
            
            for metric in metrics:
                # Simple pattern matching for numbers (can be improved)
                import re
                if metric == "revenue":
                    # Look for revenue patterns
                    patterns = [
                        r'revenue[:\s]+[\$₹]?([\d,]+\.?\d*)',
                        r'total\s+revenue[:\s]+[\$₹]?([\d,]+\.?\d*)',
                    ]
                    for pattern in patterns:
                        matches = re.findall(pattern, text)
                        if matches:
                            value = matches[0].replace(',', '')
                            if metric not in metric_values:
                                metric_values[metric] = []
                            metric_values[metric].append({
                                "value": value,
                                "source": source
                            })
        
        # Check for conflicts (same metric, different values)
        for metric, values in metric_values.items():
            unique_values = set(v["value"] for v in values)
            if len(unique_values) > 1:
                conflicts.append({
                    "metric": metric,
                    "values": values,
                    "conflict": True
                })
        
        return conflicts
    
    @staticmethod
    def extract_financial_facts(
        document_text: str,
        metadata: Dict
    ) -> List[Dict]:
        """
        Extract structured financial facts from document.
        
        Returns:
            List of fact dicts with:
            - metric: str
            - value: str
            - year: str
            - currency: str
            - source: str
        """
        facts = []
        
        # This is a simplified extractor
        # In production, use more sophisticated NLP/LLM extraction
        
        text_lower = document_text.lower()
        
        # Extract revenue
        import re
        revenue_patterns = [
            r'revenue[:\s]+[\$₹]?([\d,]+\.?\d*)\s*(crore|million|billion|cr|mn|bn)?',
            r'total\s+revenue[:\s]+[\$₹]?([\d,]+\.?\d*)\s*(crore|million|billion|cr|mn|bn)?',
        ]
        
        for pattern in revenue_patterns:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                value = match.group(1).replace(',', '')
                unit = match.group(2) if match.group(2) else ""
                
                facts.append({
                    "metric": "revenue",
                    "value": value,
                    "unit": unit,
                    "source": metadata.get("source", "unknown"),
                    "year": metadata.get("fiscal_year", ""),
                    "currency": "INR" if "₹" in document_text or "crore" in text_lower else "USD"
                })
        
        return facts
