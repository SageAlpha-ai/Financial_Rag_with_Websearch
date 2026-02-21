from typing import List, Optional
from pydantic import BaseModel
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config.settings import get_config
import json
import logging

logger = logging.getLogger(__name__)

class QueryIntent(BaseModel):
    query_type: str  # definition | historical_data | realtime_data | comparison | analysis | report | unknown
    company: Optional[str]
    fiscal_year: Optional[str]
    metrics: List[str]
    requires_realtime: bool
    requires_internal_docs: bool
    requires_web: bool
    format: str  # short | long
    ambiguity: Optional[str] = None

class IntentExtractor:

    def __init__(self):
        config = get_config()

        self.llm = AzureChatOpenAI(
            azure_endpoint=config.azure_openai.endpoint,
            azure_deployment=config.azure_openai.planner_deployment,
            api_key=config.azure_openai.api_key,
            api_version=config.azure_openai.api_version,
            temperature=0.0
        )

        self.prompt = ChatPromptTemplate.from_template("""
You are a strict financial query analyzer.

Extract structured intent from the query.

CRITICAL RULES:
- Do NOT assume or infer a company if not explicitly mentioned.
- If company is missing, return null.
- If fiscal year not mentioned, return null.
- Do NOT guess.
- Return VALID JSON only.
- Prefer FULL LEGAL COMPANY NAMES matching the knowledge base (e.g., "Dixon" -> "Dixon Technologies", "Oracle" -> "Oracle Financial Services Software Ltd").

Allowed query_type values:
definition
historical_data
realtime_data
comparison
analysis
report
unknown

Return JSON in this exact structure:

{{
  "query_type": "...",
  "company": null or "...",
  "fiscal_year": null or "...",
  "metrics": [],
  "requires_realtime": true/false,
  "requires_internal_docs": true/false,
  "requires_web": true/false,
  "format": "short" or "long",
  "ambiguity": null or "company_missing"
}}

Query:
{question}
""")

        self.chain = self.prompt | self.llm | StrOutputParser()

    def extract(self, question: str) -> QueryIntent:
        try:
            response = self.chain.invoke({"question": question})

            # Clean the response to ensure it is valid JSON
            cleaned = response.replace("```json", "").replace("```", "").strip()
            data = json.loads(cleaned)

            return QueryIntent(**data)
        except Exception as e:
            logger.error(f"Intent extraction failed: {e}")
            # Return a default/fallback intent in case of failure
            return QueryIntent(
                query_type="unknown",
                company=None,
                fiscal_year=None,
                metrics=[],
                requires_realtime=False,
                requires_internal_docs=False,
                requires_web=False,
                format="short",
                ambiguity="extraction_failed"
            )
