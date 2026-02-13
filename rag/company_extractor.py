import re
import logging
from typing import Optional
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config.settings import get_config

logger = logging.getLogger(__name__)

def extract_company_name(query: str) -> str:
    """
    Deterministic company name extractor (Stabilization Phase).

    1. Uses regex-based capitalized phrase detection.
    2. Fallbacks to LLM-based semantic extraction if deterministic fails.
    3. Returns clean company name or empty string.
    """
    # 1. Rule-based: Look for common company terminology or capitalized proper nouns
    # Avoid common starting question words
    clean_query = re.sub(r"^(What|How|Explain|Compare|Is|Why|Where|When|Tell|Give|Show|Analyze|Research)\s+", "", query, flags=re.IGNORECASE)
    
    # Heuristic patterns for company names
    patterns = [
        # Explicitly quoted names: "Tata Motors"
        r"['\"]([^'\"]+)['\"]",
        # Capitalized phrases followed by common suffixes
        r"\b([A-Z][a-zA-Z0-9&]+(?:\s+[A-Z][a-zA-Z0-9&]+)*)\s+(Ltd|Limited|Inc|Corp|Corporation|Plc|SA|AG|Technologies)\b",
        # Multiple capitalized words: Tata Motors
        r"\b([A-Z][a-z0-9&]+(?:\s+[A-Z][a-z0-9&]+)+)\b",
        # Tickers (uppercase 2-6 chars): TCS, RELIANCE, AAPL
        r"\b([A-Z]{2,6})\b"
    ]
    
    for pattern in patterns:
        matches = re.finditer(pattern, clean_query)
        for match in matches:
            candidate = match.group(1 if "(" in pattern else 0).strip()
            
            # Heuristic filter: avoid common words or financial terms
            if candidate.lower() not in [
                "the", "revenue", "ebitda", "profit", "fiscal", "year", "fy2023", "fy2024", 
                "summary", "report", "analysis", "growth", "margin", "income", "balance", "sheet"
            ]:
                if len(candidate) > 1:
                    logger.info(f"[EXTRACTOR] Deterministic match: {candidate}")
                    return candidate

    # 2. Fallback to LLM
    llm_candidate = extract_company_name_llm(query)
    
    if llm_candidate:
        logger.info(f"[EXTRACTOR] LLM fallback match: {llm_candidate}")
        return llm_candidate

    logger.warning(f"[EXTRACTOR] Failed to resolve company for: {query}")
    return ""

def extract_company_name_llm(question: str) -> Optional[str]:
    """Extract company name via Azure OpenAI (semantic fallback)."""
    config = get_config()

    llm = AzureChatOpenAI(
        azure_endpoint=config.azure_openai.endpoint,
        azure_deployment=config.azure_openai.large_chat_deployment,
        api_key=config.azure_openai.api_key,
        api_version=config.azure_openai.api_version,
        temperature=0.0,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a company-name extraction tool.\n"
                "Given a user query, extract ONLY the primary company name.\n"
                "\n"
                "Rules:\n"
                "1. Return the full official company name (e.g. 'Tata Consultancy Services' not 'TCS').\n"
                "2. If the query mentions multiple companies, return only the FIRST one mentioned.\n"
                "3. If no company is mentioned, return exactly: NONE\n"
                "4. Return ONLY the company name — no punctuation, no explanation, no extra text.\n"
                "5. Do NOT wrap the name in quotes.",
            ),
            ("human", "{query}"),
        ]
    )

    chain = prompt | llm | StrOutputParser()

    try:
        raw: str = chain.invoke({"query": question}).strip()
        if not raw or raw.upper() == "NONE":
            return None
        
        clean = raw.strip().strip("\"'.,;:!?")
        return clean if clean else None
    except Exception as exc:
        logger.warning(f"[EXTRACT_COMPANY] LLM call failed: {exc}")
        return None
