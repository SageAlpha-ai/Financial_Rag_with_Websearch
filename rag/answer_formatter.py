"""
Answer Formatter

Formats responses for RAG and LLM fallback modes.
Includes finance-grade validation and disclaimers.
"""

import re
from typing import List, Dict, Any

from openai import AzureOpenAI

from config.settings import get_config


# Finance-grade system prompt
FINANCIAL_SYSTEM_PROMPT = """You are a financial analysis assistant. You MUST follow these rules:

STRICT RULES FOR FINANCIAL QUESTIONS:
1. Only answer using EXPLICITLY STATED values from the retrieved documents.
2. If a value is not present in the documents, say "Not available in the retrieved documents."
3. NEVER guess or calculate values not explicitly provided.
4. If financial data belongs to a SUBSIDIARY, do NOT attribute it to the parent company unless explicitly stated.

REQUIRED FORMAT FOR FINANCIAL ANSWERS:
For any question about revenue, income, or financial metrics, use this format:

Entity: [Exact legal entity name from the documents]
Statement: [Income Statement / Balance Sheet / Cash Flow / Other]
Fiscal Year: [Year from the documents]
Currency: [Currency code]
Unit: [millions/thousands/etc.]

[Metric Name]: [Value]
[Other Metrics as needed]

Source: [Document name from context]

ADDITIONAL RULES:
- Always include the YEAR when citing financial figures
- Always include the CURRENCY and UNIT
- If year or unit is unclear, state: "Year/unit not clearly specified in the source document."

For non-financial questions, answer normally in clear paragraphs.
"""


# LLM fallback system prompt
LLM_FALLBACK_SYSTEM_PROMPT = """You are a helpful assistant with general knowledge about companies and finance.

IMPORTANT: You are answering from your training data, NOT from specific documents.

When providing financial information:
- State that this is based on general knowledge
- Use approximate or estimated language where appropriate
- Mention that figures may be outdated or approximate
- Do NOT cite specific document sources

Keep answers helpful but honest about the limitations of non-grounded responses.
"""


def get_azure_client() -> AzureOpenAI:
    """Gets Azure OpenAI client."""
    config = get_config().azure_openai
    
    return AzureOpenAI(
        api_key=config.api_key,
        api_version=config.api_version,
        azure_endpoint=config.endpoint,
    )


def build_rag_prompt(query: str, documents: List[str], metadatas: List[Dict]) -> List[Dict]:
    """Builds prompt for RAG response."""
    enriched_context = []
    
    for i, (doc, meta) in enumerate(zip(documents, metadatas)):
        meta_str = ""
        if meta.get("company"):
            meta_str += f"Company: {meta['company']}\n"
        if meta.get("fiscal_year"):
            meta_str += f"Fiscal Year: {meta['fiscal_year']}\n"
        if meta.get("currency"):
            meta_str += f"Currency: {meta['currency']}\n"
        if meta.get("source"):
            meta_str += f"Source: {meta['source']}\n"
        
        enriched_context.append(f"--- Document {i+1} ---\n{meta_str}\nContent:\n{doc}\n")
    
    context_text = "\n".join(enriched_context)
    
    return [
        {"role": "system", "content": FINANCIAL_SYSTEM_PROMPT},
        {"role": "user", "content": f"""Question: {query}

Retrieved Documents:
{context_text}

Remember: For financial questions, use the structured format. Include entity, year, currency, and unit."""}
    ]


def build_llm_prompt(query: str) -> List[Dict]:
    """Builds prompt for LLM fallback."""
    return [
        {"role": "system", "content": LLM_FALLBACK_SYSTEM_PROMPT},
        {"role": "user", "content": f"""Question: {query}

NOTE: The requested information was NOT found in the stored documents.
Please answer based on your general training knowledge.
Be clear that this is general knowledge, not from specific documents."""}
    ]


def validate_financial_response(response: str, query: str) -> str:
    """Validates financial answers include year and unit."""
    financial_keywords = ["revenue", "income", "profit", "loss", "earnings", "assets", 
                         "liabilities", "equity", "cash flow", "financial", "fiscal"]
    
    is_financial = any(kw in query.lower() for kw in financial_keywords)
    
    if not is_financial:
        return response
    
    has_year = bool(re.search(r'\b(FY\s*)?20\d{2}\b|\b19\d{2}\b', response))
    
    currency_patterns = [
        r'\b(USD|INR|EUR|GBP|JPY)\b',
        r'\b(million|billion|thousand|crore|lakh)\b',
        r'\$|\u20B9|\u20AC|\u00A3'
    ]
    has_unit = any(re.search(p, response, re.IGNORECASE) for p in currency_patterns)
    
    warnings = []
    if not has_year:
        warnings.append("Year not clearly specified in the response.")
    if not has_unit:
        warnings.append("Currency/unit not clearly specified in the response.")
    
    if warnings:
        response += "\n\n---\nDATA QUALITY NOTES:\n" + "\n".join(f"- {w}" for w in warnings)
    
    return response


def format_rag_response(
    query: str,
    documents: List[str],
    metadatas: List[Dict]
) -> str:
    """
    Generates RAG response using documents.
    
    Args:
        query: User question
        documents: Retrieved documents
        metadatas: Document metadata
    
    Returns:
        Formatted response string
    """
    config = get_config()
    client = get_azure_client()
    
    messages = build_rag_prompt(query, documents, metadatas)
    
    response = client.chat.completions.create(
        model=config.azure_openai.chat_deployment,
        messages=messages,
        temperature=0,
    )
    
    answer = response.choices[0].message.content
    answer = validate_financial_response(answer, query)
    
    return answer


def format_llm_fallback_response(query: str) -> str:
    """
    Generates LLM fallback response (no documents).
    
    Args:
        query: User question
    
    Returns:
        Formatted response with disclaimers
    """
    config = get_config()
    client = get_azure_client()
    
    messages = build_llm_prompt(query)
    
    response = client.chat.completions.create(
        model=config.azure_openai.chat_deployment,
        messages=messages,
        temperature=0.3,
    )
    
    answer = response.choices[0].message.content
    
    # Add clear disclaimer formatting
    financial_keywords = ["revenue", "income", "profit", "loss", "earnings", "assets", 
                         "liabilities", "equity", "cash flow", "financial", "fiscal"]
    is_financial = any(kw in query.lower() for kw in financial_keywords)
    
    formatted = "=" * 60 + "\n"
    formatted += "Answer Type: LLM Pretrained Knowledge (Not Document-Grounded)\n"
    formatted += "=" * 60 + "\n\n"
    formatted += answer
    formatted += "\n\n" + "-" * 60 + "\n"
    formatted += "DISCLAIMER:\n"
    formatted += "This answer is based on the model's general training data\n"
    formatted += "and NOT on documents stored in Azure Blob Storage.\n"
    
    if is_financial:
        formatted += "\nFINANCIAL NOTE:\n"
        formatted += "- Values may be approximate or outdated\n"
        formatted += "- Verify with official financial reports before making decisions\n"
        formatted += "- This is NOT auditable financial data\n"
    
    formatted += "-" * 60
    
    return formatted


def format_sources(metadatas: List[Dict]) -> str:
    """Formats source metadata for display."""
    sources = []
    
    for meta in metadatas:
        source = meta.get('source', meta.get('filename', 'unknown'))
        parts = [source]
        
        if 'line_number' in meta:
            parts.append(f"line {meta['line_number']}")
        if 'page' in meta:
            parts.append(f"page {meta['page']}")
        if 'fiscal_year' in meta and meta['fiscal_year'] != 'Unknown':
            parts.append(f"FY: {meta['fiscal_year']}")
        
        sources.append(": ".join(parts) if len(parts) > 1 else parts[0])
    
    return "\n".join(sources)


def check_entity_confidence(metadatas: List[Dict]) -> tuple[str, int]:
    """Checks entity confidence in retrieved documents."""
    entity_counts = {}
    
    for meta in metadatas:
        company = meta.get("company", "")
        if company:
            entity_counts[company] = entity_counts.get(company, 0) + 1
    
    if not entity_counts:
        return ("Unknown", 0)
    
    top_entity = max(entity_counts, key=entity_counts.get)
    return (top_entity, entity_counts[top_entity])
