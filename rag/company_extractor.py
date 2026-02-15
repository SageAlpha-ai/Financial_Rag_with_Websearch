import re
import logging
from typing import Optional
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config.settings import get_config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Leading verbs / question words that must be stripped before entity
# extraction.  Expanded from original list to prevent "Describe Dixon"
# being returned as an entity.
# ---------------------------------------------------------------------------
_LEADING_VERB_RE = re.compile(
    r"^(What|How|Explain|Compare|Is|Are|Was|Were|Why|Where|When|"
    r"Tell|Give|Show|Analyze|Analyse|Research|Describe|Discuss|"
    r"List|Find|Calculate|Compute|Summarize|Summarise|Define|"
    r"Provide|State|Outline|Elaborate|Detail|Evaluate|Assess|"
    r"Report|Can|Could|Do|Does|Did|Will|Would|Should|Shall|"
    r"Please|Kindly|Help|Get|Fetch)\b[\s,]*",
    re.IGNORECASE,
)

# Single-word strings that are verbs / non-entity tokens even when
# capitalized at sentence boundaries.  Used as a post-extraction filter.
_VERB_BLOCKLIST = {
    "describe", "explain", "compare", "analyze", "analyse", "discuss",
    "list", "find", "calculate", "compute", "summarize", "summarise",
    "define", "provide", "state", "outline", "elaborate", "detail",
    "evaluate", "assess", "report", "show", "tell", "give", "research",
    "help", "get", "fetch", "please", "kindly",
}

# Common financial / query terms that should never be returned as entities.
_NON_ENTITY_TERMS = {
    "the", "revenue", "ebitda", "profit", "fiscal", "year",
    "fy2023", "fy2024", "fy2025", "fy2020", "fy2021", "fy2022",
    "fy2008", "fy2019", "fy2018", "fy2017", "fy2016",
    "summary", "report", "analysis", "growth", "margin", "income",
    "balance", "sheet", "business", "model", "overview", "net",
    "total", "annual", "quarterly", "q1", "q2", "q3", "q4",
    "cash", "flow", "statement", "financial", "financials",
    "what", "how", "why", "where", "when", "which", "who",
}


def extract_company_name(query: str) -> str:
    """Deterministic company name extractor.

    1. Strips leading verbs / question words.
    2. Extracts proper organization entities via regex (longest match wins).
    3. Falls back to LLM-based semantic extraction if deterministic fails.
    4. Returns clean company name or empty string.
    """
    # ------------------------------------------------------------------ #
    # 1. Strip leading verbs to prevent "Describe Dixon" matches          #
    # ------------------------------------------------------------------ #
    clean_query = _LEADING_VERB_RE.sub("", query).strip()
    # Second pass: strip any leftover leading verbs after first removal
    clean_query = _LEADING_VERB_RE.sub("", clean_query).strip()

    if not clean_query:
        clean_query = query  # fallback to original if fully stripped

    # ------------------------------------------------------------------ #
    # 2. Deterministic regex extraction (longest match wins)              #
    # ------------------------------------------------------------------ #
    patterns = [
        # Explicitly quoted names: "Tata Motors"
        r"""['"]([^'"]+)['"]""",
        # Capitalized phrases followed by common suffixes
        (
            r"\b([A-Z][a-zA-Z0-9&]+(?:\s+[A-Z][a-zA-Z0-9&]+)*)"
            r"\s+(?:Ltd|Limited|Inc|Corp|Corporation|Plc|SA|AG|"
            r"Technologies|Software|Services|Bank|Group|Holdings)\b"
        ),
        # Multiple capitalized words (2+ tokens): Dixon Technologies
        r"\b([A-Z][a-zA-Z0-9&]+(?:\s+[A-Z][a-zA-Z0-9&]+)+)\b",
        # Single capitalized proper noun (3+ chars): Dixon, Apple, Google
        r"\b([A-Z][a-z]{2,})\b",
        # Tickers (uppercase 2-6 chars): TCS, RELIANCE, AAPL
        r"\b([A-Z]{2,6})\b",
    ]

    # Collect ALL candidates and pick the longest valid one.
    best_candidate: Optional[str] = None
    best_length = 0

    for pattern in patterns:
        for match in re.finditer(pattern, clean_query):
            candidate = match.group(1).strip()
            if not _is_valid_entity(candidate):
                continue
            if len(candidate) > best_length:
                best_candidate = candidate
                best_length = len(candidate)

    if best_candidate:
        logger.info("[EXTRACTOR] Clean entity detected: %s", best_candidate)
        return best_candidate

    # ------------------------------------------------------------------ #
    # 3. LLM fallback                                                     #
    # ------------------------------------------------------------------ #
    llm_candidate = extract_company_name_llm(query)

    if llm_candidate:
        logger.info("[EXTRACTOR] Clean entity detected: %s (LLM)", llm_candidate)
        return llm_candidate

    logger.info("[EXTRACTOR] Clean entity detected: None")
    return ""


def _is_valid_entity(candidate: str) -> bool:
    """Return True only if *candidate* looks like a real organization name."""
    if not candidate or len(candidate) <= 1:
        return False

    # Reject if the first word is a verb (e.g. "Describe Dixon")
    first_word = candidate.split()[0].lower()
    if first_word in _VERB_BLOCKLIST:
        return False

    # Reject if entire candidate is a non-entity term
    if candidate.lower() in _NON_ENTITY_TERMS:
        return False

    # Reject if ALL words are non-entity terms (e.g. "Net Income")
    words = candidate.lower().split()
    if all(w in _NON_ENTITY_TERMS for w in words):
        return False

    return True


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
        logger.warning("[EXTRACT_COMPANY] LLM call failed: %s", exc)
        return None
