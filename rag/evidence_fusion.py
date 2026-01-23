"""
Evidence Fusion Module

Merges evidence from multiple sources (RAG, Web Search) with priority rules.
Resolves conflicts and produces unified evidence for LLM summarization.
"""

import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


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
        
        # Prioritize web documents (official sources)
        prioritized_docs = []
        sources = []
        conflicts = []
        
        # Add web documents first (highest priority)
        for web_doc in web_documents:
            doc_text = web_doc.get("text", "")
            doc_meta = web_doc.get("metadata", {})
            
            if doc_text:
                prioritized_docs.append({
                    "text": doc_text,
                    "metadata": doc_meta,
                    "source_type": "web_official",
                    "priority": 1
                })
                
                # Add source citation
                source_url = doc_meta.get("url", "")
                source_title = doc_meta.get("title", "Official Document")
                if source_url:
                    sources.append(f"{source_title} - {source_url}")
                else:
                    sources.append(source_title)
        
        # Add RAG documents (lower priority but still valuable)
        for doc, meta in zip(rag_documents, rag_metadatas):
            prioritized_docs.append({
                "text": doc,
                "metadata": meta,
                "source_type": "rag_internal",
                "priority": 2
            })
            
            # Add source citation
            source_info = meta.get("source", meta.get("filename", "Internal Document"))
            if meta.get("page"):
                source_info += f" (page {meta.get('page')})"
            if meta.get("fiscal_year"):
                source_info += f" (FY: {meta.get('fiscal_year')})"
            sources.append(source_info)
        
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
