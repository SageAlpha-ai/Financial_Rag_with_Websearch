#!/usr/bin/env python3
"""
AI RAG Service API

Exposes the RAG query engine as a REST API using FastAPI.
Can be consumed by Node.js or any other service.

Run:
    uvicorn api:app --host 0.0.0.0 --port 8000

Note: 0.0.0.0 is for server binding only. Use http://localhost:8000 in your browser.

Swagger UI:
    http://localhost:8000/docs
"""

import logging
import os
import sys
from typing import List, Optional

# Add current directory to path for imports
sys.path.insert(0, ".")

from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, model_validator

# Import existing logic
from config.settings import get_config, validate_config
# LangChain orchestration replaces manual routing
from rag.langchain_orchestrator import answer_query_simple
# Report generation for long-format reports
from rag.report_generator import is_report_request, generate_report

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ================================
# Input Normalization
# ================================

def normalize_user_input(raw_input: str) -> str:
    """
    Normalize and sanitize raw user input.
    
    Strips JavaScript artifacts, template literals, control characters, and excessive whitespace
    while preserving semantic meaning. Does NOT truncate content.
    
    Args:
        raw_input: Raw user input (may contain code, templates, broken text, control chars)
    
    Returns:
        Normalized string ready for RAG/LLM processing
    """
    import re
    
    if not raw_input:
        return ""
    
    # Start with the input
    normalized = raw_input.strip()
    
    # Strip control characters (except newlines, tabs, carriage returns)
    # Keep: \n, \t, \r (whitespace)
    # Remove: \x00-\x08, \x0B, \x0C, \x0E-\x1F (control chars)
    normalized = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F]', '', normalized)
    
    # Remove JavaScript variable declarations (const/let/var variableName = "value")
    # Only match complete declarations, not partial matches
    normalized = re.sub(r'\b(const|let|var)\s+\w+\s*=\s*["\']?', '', normalized, flags=re.IGNORECASE)
    
    # Remove systemPrompt and similar patterns (only at word boundaries)
    normalized = re.sub(r'\bsystemPrompt\s*=\s*["\']?', '', normalized, flags=re.IGNORECASE)
    normalized = re.sub(r'\bprompt\s*=\s*["\']?', '', normalized, flags=re.IGNORECASE)
    
    # Remove backticks (JavaScript template literals)
    normalized = normalized.replace('`', '')
    
    # Remove template literal placeholders (${...})
    normalized = re.sub(r'\$\{[^}]*\}', '', normalized)
    
    # Remove semicolons at end of lines
    normalized = re.sub(r';\s*\n', '\n', normalized)
    normalized = re.sub(r';\s*$', '', normalized, flags=re.MULTILINE)
    
    # Remove JavaScript keywords that appear as standalone words
    # Be careful - only remove if they're clearly artifacts, not part of natural language
    normalized = re.sub(r'\b(console\.log|console\.error)\s*\([^)]*\)', '', normalized, flags=re.IGNORECASE)
    
    # Collapse multiple newlines into single newline or space
    normalized = re.sub(r'\n\s*\n\s*\n+', '\n\n', normalized)
    
    # Collapse excessive whitespace (3+ spaces) into single space
    normalized = re.sub(r' {3,}', ' ', normalized)
    
    # Normalize tabs to spaces
    normalized = normalized.replace('\t', ' ')
    
    # Remove leading/trailing whitespace from each line
    lines = [line.strip() for line in normalized.split('\n')]
    normalized = '\n'.join(lines)
    
    # Remove empty lines at start and end
    normalized = normalized.strip()
    
    # Final cleanup: collapse any remaining excessive whitespace (but preserve single spaces)
    normalized = re.sub(r'[ \t]+', ' ', normalized)
    normalized = re.sub(r'\n[ \t]+', '\n', normalized)  # Remove trailing spaces on lines
    normalized = re.sub(r'[ \t]+\n', '\n', normalized)  # Remove leading spaces before newlines
    
    # Ensure we have at least some content (after normalization, empty string means invalid)
    if not normalized or len(normalized.strip()) < 1:
        # If normalization removed everything, return original (fallback)
        return raw_input.strip()
    
    return normalized.strip()


# ================================
# Pydantic Models
# ================================

class QueryRequest(BaseModel):
    """Request body for /query endpoint.
    
    Accepts either 'query' or 'question' field (backward compatible).
    At least one field must be provided.
    """
    query: Optional[str] = Field(
        None,
        description="The query text to process (questions, code, templates, or any text)",
        max_length=5000,
        examples=["What is the revenue of Oracle Financial Services for FY2023?"]
    )
    question: Optional[str] = Field(
        None,
        description="[Legacy] The question text (use 'query' for new clients)",
        max_length=5000,
        examples=["What is the revenue of Oracle Financial Services for FY2023?"]
    )
    
    @model_validator(mode='after')
    def validate_at_least_one_field(self):
        """Ensure at least one of 'query' or 'question' is provided."""
        if not self.query and not self.question:
            raise ValueError("At least one of 'query' or 'question' field must be provided")
        return self
    
    def get_input(self) -> str:
        """Get the input text, preferring 'query' over 'question'."""
        return (self.query or self.question or "").strip()


class SourceItem(BaseModel):
    """Source item with structured information."""
    title: str = Field(..., description="Source document title")
    url: Optional[str] = Field(None, description="Official URL to the source")
    publisher: Optional[str] = Field(None, description="Publisher name (Company or Regulator)")


class QueryResponse(BaseModel):
    """Response body for /query endpoint."""
    answer: str = Field(..., description="The generated answer")
    answer_type: str = Field(
        ...,
        description="Answer type: 'sagealpha_rag' (from internal documents), 'sagealpha_ai_search' (from web search), 'sagealpha_hybrid_search' (combined), or 'REPORT' (long-format report)"
    )
    sources: List[SourceItem] = Field(
        default_factory=list,
        description="List of source documents with official URLs (empty list [] for LLM-only answers, never null)"
    )


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    service: str
    chroma_connected: bool
    document_count: int


# ================================
# FastAPI App
# ================================

app = FastAPI(
    title="AI RAG Service",
    description="""
Finance-Grade RAG API powered by:
- **Chroma Cloud** for vector storage
- **Azure OpenAI** for embeddings and chat
- **Hybrid RAG + LLM fallback** for guaranteed answers

## Features

- 📄 Answers from Azure Blob documents when available
- 🤖 Automatic LLM fallback when documents cannot answer
- 🏢 Finance-grade entity and year attribution
- ✅ Never returns "Not available"

## Usage

```javascript
// Node.js example
const response = await fetch("http://localhost:8000/query", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: "What is Oracle's revenue in FY2023?" })
});
const data = await response.json();
console.log(data.answer);
```
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# ================================
# CORS Middleware (for browser/Node.js clients)
# ================================

# CORS: Allow configurable origins via env (default: allow all)
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",") if os.getenv("CORS_ORIGINS") != "*" else ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ================================
# Optional API Key Authentication
# ================================

# Optional: set RAG_API_KEY env var to enable authentication (disabled if not set)
RAG_API_KEY = os.getenv("RAG_API_KEY")

def verify_api_key(x_api_key: Optional[str] = Header(None, alias="x-api-key")):
    """Optional API key authentication. Disabled if RAG_API_KEY env var is not set."""
    if RAG_API_KEY:
        if not x_api_key or x_api_key != RAG_API_KEY:
            raise HTTPException(status_code=401, detail="Invalid API key")
    return True


# ================================
# Startup Event
# ================================

@app.on_event("startup")
async def startup_event():
    """Validate configuration on startup. App continues even if external services are unavailable."""
    try:
        config = get_config()
        validate_config(config)
        logger.info("Configuration loaded successfully")
    except ValueError as e:
        # Log error but don't crash - app can still serve health checks
        logger.error(f"Configuration error: {e}. Some endpoints may not work.")
    
    # Get port from environment (Azure App Service and Render use PORT, local dev uses 8000)
    port = int(os.getenv("PORT", "8000"))
    
    logger.info("=" * 60)
    logger.info("AI RAG SERVICE STARTED")
    logger.info("=" * 60)
    if RAG_API_KEY:
        logger.info("API Key authentication: ENABLED")
    else:
        logger.info("API Key authentication: DISABLED")
    logger.info("=" * 60)
    logger.info(f"Server is running. Open http://localhost:{port}/docs in your browser")
    logger.info("=" * 60)


# ================================
# Endpoints
# ================================

@app.get("/", tags=["Info"])
async def root():
    """Root endpoint with service info."""
    return {
        "service": "AI RAG Service",
        "status": "running",
        "version": "1.0.0",
        "usage": "POST /query with JSON { query: string }",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["Info"])
async def health_check():
    """
    Lightweight health check endpoint for deployment.
    
    Does not block on heavy operations to ensure fast startup and health checks.
    """
    try:
        from vectorstore.chroma_client import get_collection
        collection = get_collection(create_if_missing=False)
        # Lightweight connection check (don't call count() as it can be slow on large collections)
        # Just verify collection exists and is accessible
        _ = collection.name  # Lightweight attribute access
        doc_count = 0  # Don't count documents in health check (too slow)
        
        return HealthResponse(
            status="healthy",
            service="AI RAG Service",
            chroma_connected=True,
            document_count=doc_count
        )
    except Exception as e:
        logger.debug(f"Health check failed: {e}")
        return HealthResponse(
            status="degraded",
            service="AI RAG Service",
            chroma_connected=False,
            document_count=0
        )


@app.get("/query", tags=["Query"])
async def query_help():
    """
    Usage instructions for the /query endpoint.
    
    This endpoint only accepts POST requests.
    Use POST /query with a JSON body containing your question.
    """
    return {
        "error": "Method not allowed",
        "message": "Use POST /query with a JSON body",
        "usage": {
            "method": "POST",
            "url": "/query",
            "headers": {"Content-Type": "application/json"},
            "body": {"query": "Your question here"}
        },
        "example": {
            "query": "What is the revenue of Oracle Financial Services for FY2023?"
        },
        "curl_example": 'curl -X POST http://localhost:8000/query -H "Content-Type: application/json" -d \'{"query": "What is Oracle revenue?"}\''
    }


async def _process_query(req: QueryRequest) -> QueryResponse:
    """
    Internal function to process RAG queries.
    Shared by both /query and /docs/query endpoints.
    """
    try:
        # Get input from request model (supports both 'query' and 'question' for backward compatibility)
        user_input = req.get_input()
        
        if not user_input or not user_input.strip():
            raise HTTPException(
                status_code=400,
                detail="Request body must contain either 'query' or 'question' field with non-empty text"
            )
        
        # Normalize input to handle unstructured text, code, templates, etc.
        normalized_input = normalize_user_input(user_input)
        
        if not normalized_input or not normalized_input.strip():
            logger.warning(f"Input normalization resulted in empty string. Original length: {len(user_input)}")
            raise HTTPException(
                status_code=400,
                detail="Input could not be normalized. Please provide valid text input."
            )
        
        # Route based on intent: report generation vs Q&A
        if is_report_request(normalized_input):
            # Long-format report generation (two-phase: RAG facts + LLM narrative)
            logger.info("Report generation mode detected")
            result = generate_report(normalized_input)
        else:
            # Standard Q&A mode (existing behavior)
            result = answer_query_simple(normalized_input)
        
        # Convert sources to SourceItem format if needed
        # CRITICAL: Ensure sources are never null
        formatted_sources = []
        if result.get("sources"):
            if isinstance(result["sources"], list) and len(result["sources"]) > 0:
                if isinstance(result["sources"][0], dict):
                    # Already in dict format
                    formatted_sources = [
                        SourceItem(**s) if isinstance(s, dict) 
                        else SourceItem(title=str(s), url=None, publisher=None) 
                        for s in result["sources"]
                    ]
                else:
                    # Convert string sources to SourceItem
                    formatted_sources = [
                        SourceItem(title=str(s), url=None, publisher=None) 
                        for s in result["sources"]
                    ]
        
        return QueryResponse(
            answer=result["answer"],
            answer_type=result["answer_type"],
            sources=formatted_sources  # Always a list, never None
        )
    except HTTPException:
        raise  # Re-raise HTTP exceptions as-is
    except Exception as e:
        # Log error server-side but never expose stack traces to clients
        logger.error(f"Query processing failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An error occurred while processing your query. Please try again."
        )


@app.post("/query", response_model=QueryResponse, tags=["Query"])
async def query_rag(
    req: QueryRequest,
    _: bool = Depends(verify_api_key)
):
    """
    Query the RAG system.
    
    Accepts any text input (questions, code, templates, unstructured text).
    Input is normalized and sanitized before processing.
    
    Request body accepts either:
    - **query**: The query text (preferred for new clients)
    - **question**: The question text (legacy field, still supported)
    
    At least one field must be provided.
    
    Returns:
    - **answer**: The generated answer
    - **answer_type**: "RAG" (from documents), "RAG+LLM" (documents + reasoning), or "LLM" (from training data)
    - **sources**: List of document sources (empty list [] for LLM-only answers, never null)
    """
    return await _process_query(req)


@app.post("/docs/query", response_model=QueryResponse, tags=["Query"])
async def query_rag_docs_alias(
    req: QueryRequest,
    _: bool = Depends(verify_api_key)
):
    """
    Alias endpoint for /query (for backward compatibility).
    
    This endpoint forwards to the main /query endpoint.
    Note: /docs is FastAPI's Swagger UI, but this alias is provided for convenience.
    For new integrations, use POST /query instead.
    """
    return await _process_query(req)


# ================================
# Run directly (for development)
# ================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
