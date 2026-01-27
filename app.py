#!/usr/bin/env python3
"""
AI RAG Service API

Exposes the RAG query engine as a REST API using FastAPI.
Can be consumed by Node.js or any other service.

Run:
    uvicorn app:app --host 0.0.0.0 --port 8000

Note: 0.0.0.0 is for server binding only. Use http://localhost:8000 in your browser.

Swagger UI:
    http://localhost:8000/docs
"""

import logging
import os
import sys
from contextlib import asynccontextmanager
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
# Optional API Key Authentication (defined early for lifespan use)
# ================================

# Optional: set RAG_API_KEY env var to enable authentication (disabled if not set)
RAG_API_KEY = os.getenv("RAG_API_KEY")


# ================================
# Lifespan Event Handler (replaces deprecated @app.on_event)
# ================================

def validate_azure_openai_deployments(config) -> None:
    """
    Validates Azure OpenAI deployments exist and are accessible.
    
    Raises ValueError with clear error message if deployments don't exist.
    """
    try:
        from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
        
        logger.info("=" * 60)
        logger.info("VALIDATING AZURE OPENAI DEPLOYMENTS")
        logger.info("=" * 60)
        
        # Validate chat deployment
        logger.info(f"Validating chat deployment: {config.azure_openai.chat_deployment}")
        try:
            test_llm = AzureChatOpenAI(
                azure_endpoint=config.azure_openai.endpoint,
                azure_deployment=config.azure_openai.chat_deployment,
                api_key=config.azure_openai.api_key,
                api_version=config.azure_openai.api_version,
                temperature=0.0,
                max_retries=1,
                timeout=10
            )
            # Make a minimal test call
            test_llm.invoke("test")
            logger.info(f"✓ Chat deployment '{config.azure_openai.chat_deployment}' is valid")
        except Exception as e:
            error_msg = str(e)
            if "NotFound" in error_msg or "404" in error_msg or "deployment" in error_msg.lower():
                raise ValueError(
                    f"Azure OpenAI chat deployment '{config.azure_openai.chat_deployment}' not found.\n\n"
                    f"This means the deployment name in AZURE_OPENAI_CHAT_DEPLOYMENT_NAME does not exist in your Azure OpenAI resource.\n\n"
                    f"To fix:\n"
                    f"  1. Go to Azure Portal → Your OpenAI Resource → Deployments\n"
                    f"  2. Find the exact deployment name (case-sensitive)\n"
                    f"  3. Update AZURE_OPENAI_CHAT_DEPLOYMENT_NAME in your .env file\n\n"
                    f"Current value: {config.azure_openai.chat_deployment}\n"
                    f"Endpoint: {config.azure_openai.endpoint}\n\n"
                    f"Error: {error_msg}"
                ) from e
            else:
                raise ValueError(
                    f"Failed to validate chat deployment '{config.azure_openai.chat_deployment}': {error_msg}"
                ) from e
        
        # Validate embeddings deployment
        logger.info(f"Validating embeddings deployment: {config.azure_openai.embeddings_deployment}")
        try:
            embedding_kwargs = {
                "azure_endpoint": config.azure_openai.endpoint,
                "azure_deployment": config.azure_openai.embeddings_deployment,
                "api_key": config.azure_openai.api_key,
                "api_version": config.azure_openai.api_version,
            }
            if "text-embedding-ada-002" in config.azure_openai.embeddings_deployment.lower():
                embedding_kwargs["model"] = "text-embedding-ada-002"
            
            test_embeddings = AzureOpenAIEmbeddings(**embedding_kwargs)
            # Make a minimal test call
            test_embeddings.embed_query("test")
            logger.info(f"✓ Embeddings deployment '{config.azure_openai.embeddings_deployment}' is valid")
        except Exception as e:
            error_msg = str(e)
            if "NotFound" in error_msg or "404" in error_msg or "deployment" in error_msg.lower():
                raise ValueError(
                    f"Azure OpenAI embeddings deployment '{config.azure_openai.embeddings_deployment}' not found.\n\n"
                    f"This means the deployment name in AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME does not exist in your Azure OpenAI resource.\n\n"
                    f"To fix:\n"
                    f"  1. Go to Azure Portal → Your OpenAI Resource → Deployments\n"
                    f"  2. Find the exact deployment name (case-sensitive)\n"
                    f"  3. Update AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME in your .env file\n\n"
                    f"Current value: {config.azure_openai.embeddings_deployment}\n"
                    f"Endpoint: {config.azure_openai.endpoint}\n\n"
                    f"Error: {error_msg}"
                ) from e
            else:
                raise ValueError(
                    f"Failed to validate embeddings deployment '{config.azure_openai.embeddings_deployment}': {error_msg}"
                ) from e
        
        logger.info("=" * 60)
        logger.info("✓ All Azure OpenAI deployments validated successfully")
        logger.info("=" * 60)
        
    except ValueError:
        raise  # Re-raise our custom errors
    except Exception as e:
        raise ValueError(
            f"Unexpected error during Azure OpenAI validation: {str(e)}"
        ) from e


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    # Startup
    try:
        config = get_config()
        validate_config(config)
        logger.info("Configuration loaded successfully")
        
        # Validate Azure OpenAI deployments (fail fast if misconfigured)
        try:
            validate_azure_openai_deployments(config)
        except ValueError as e:
            logger.error("=" * 60)
            logger.error("AZURE OPENAI DEPLOYMENT VALIDATION FAILED")
            logger.error("=" * 60)
            logger.error(str(e))
            logger.error("=" * 60)
            logger.error("The application will not function correctly until deployments are fixed.")
            # Don't crash - allow health checks, but log clearly
        
        # Validate Chroma collection (non-blocking)
        try:
            from vectorstore.chroma_client import get_collection
            import os
            
            # Auto-create in development mode
            is_production = os.getenv("PYTHON_ENV") == "production" or os.getenv("AZURE_APP_SERVICE")
            auto_create = os.getenv("CHROMA_AUTO_CREATE_COLLECTION", "").lower() == "true" or not is_production
            
            if auto_create:
                logger.info("=" * 60)
                logger.info("CHROMA COLLECTION VALIDATION (AUTO-CREATE MODE)")
                logger.info("=" * 60)
                collection = get_collection(create_if_missing=True)
                doc_count = collection.count()
                logger.info(f"Collection '{collection.name}' ready with {doc_count} documents")
                if doc_count == 0:
                    logger.warning("Collection is empty. Run 'python ingest.py --fresh' to populate it.")
                logger.info("=" * 60)
            else:
                logger.info("=" * 60)
                logger.info("CHROMA COLLECTION VALIDATION (STRICT MODE)")
                logger.info("=" * 60)
                try:
                    collection = get_collection(create_if_missing=False)
                    doc_count = collection.count()
                    logger.info(f"Collection '{collection.name}' ready with {doc_count} documents")
                    if doc_count == 0:
                        logger.warning("Collection is empty. Run 'python ingest.py --fresh' to populate it.")
                except ValueError as e:
                    logger.warning(f"Collection validation: {str(e)}")
                    logger.warning("Queries will fail until collection is created via ingestion.")
                logger.info("=" * 60)
        except Exception as e:
            logger.warning(f"Chroma collection validation skipped: {e}")
            logger.warning("Queries may fail if collection is not available.")
        
    except ValueError as e:
        # Log error but don't crash - app can still serve health checks
        logger.error(f"CRITICAL CONFIGURATION ERROR: {e}")
        logger.error("The application may not function correctly until environment variables are set.")
    
    # Get port from environment (Azure App Service uses PORT, local dev uses 8000)
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
    
    yield  # App runs here
    
    # Shutdown (if needed in future)
    logger.info("Shutting down AI RAG Service...")


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
    redoc_url="/redoc",
    lifespan=lifespan
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
# API Key Authentication Function
# ================================

def verify_api_key(x_api_key: Optional[str] = Header(None, alias="x-api-key")):
    """Optional API key authentication. Disabled if RAG_API_KEY env var is not set."""
    if RAG_API_KEY:
        if not x_api_key or x_api_key != RAG_API_KEY:
            raise HTTPException(status_code=401, detail="Invalid API key")
    return True


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


@app.get("/health", tags=["Info"])
async def health_check():
    """
    Lightweight health check endpoint for Azure App Service.
    """
    return {"status": "ok"}


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
        
        logger.info(f"[API] Processing query: {normalized_input[:100]}...")
        
        # Route based on intent: report generation vs Q&A
        try:
            if is_report_request(normalized_input):
                # Long-format report generation (two-phase: RAG facts + LLM narrative)
                logger.info("[API] Report generation mode detected")
                result = generate_report(normalized_input)
            else:
                # Standard Q&A mode (existing behavior)
                logger.info("[API] Standard Q&A mode - calling answer_query_simple")
                result = answer_query_simple(normalized_input)
            
            # Validate result structure
            if not isinstance(result, dict):
                logger.error(f"[API] Invalid result type: {type(result)}, expected dict")
                raise ValueError("RAG pipeline returned invalid result format")
            
            # Ensure required keys exist
            if "answer" not in result:
                logger.error(f"[API] Result missing 'answer' key. Keys: {result.keys()}")
                result["answer"] = "I apologize, but I encountered an error while processing your query."
            
            if "answer_type" not in result:
                logger.warning("[API] Result missing 'answer_type' key, defaulting to 'sagealpha_rag'")
                result["answer_type"] = "sagealpha_rag"
            
            if "sources" not in result:
                logger.warning("[API] Result missing 'sources' key, defaulting to empty list")
                result["sources"] = []
            
            logger.info(f"[API] Result keys: {list(result.keys())}")
            logger.info(f"[API] Answer type: {result.get('answer_type')}")
            logger.info(f"[API] Sources count: {len(result.get('sources', []))}")
            
        except RuntimeError as e:
            # Handle ChromaDB initialization errors
            error_msg = str(e)
            if "ChromaDB collection is EMPTY" in error_msg or "FATAL ERROR" in error_msg:
                logger.error(f"[API] ChromaDB error: {error_msg}")
                raise HTTPException(
                    status_code=503,
                    detail="The knowledge base is not available. Please ensure documents have been ingested."
                )
            raise
        except ValueError as e:
            # Handle configuration errors
            error_msg = str(e)
            if "CHROMA_API_KEY" in error_msg or "environment variable" in error_msg.lower():
                logger.error(f"[API] Configuration error: {error_msg}")
                raise HTTPException(
                    status_code=503,
                    detail="Service configuration error. Please check environment variables."
                )
            raise
        
        # Convert sources to SourceItem format if needed
        # CRITICAL: Ensure sources are never null
        formatted_sources = []
        sources = result.get("sources", [])
        
        if sources:
            if isinstance(sources, list) and len(sources) > 0:
                try:
                    if isinstance(sources[0], dict):
                        # Already in dict format - validate and convert to SourceItem
                        for s in sources:
                            try:
                                # Try to create SourceItem from dict
                                if "title" in s:
                                    formatted_sources.append(SourceItem(
                                        title=s.get("title", "Unknown"),
                                        url=s.get("url"),
                                        publisher=s.get("publisher")
                                    ))
                                else:
                                    # Fallback: use first key as title
                                    title = str(s.get(list(s.keys())[0] if s else "Unknown", "Unknown"))
                                    formatted_sources.append(SourceItem(
                                        title=title,
                                        url=s.get("url"),
                                        publisher=s.get("publisher")
                                    ))
                            except Exception as source_error:
                                logger.warning(f"[API] Failed to format source item: {source_error}, source: {s}")
                                # Create fallback source
                                formatted_sources.append(SourceItem(
                                    title=str(s) if s else "Unknown",
                                    url=None,
                                    publisher=None
                                ))
                    else:
                        # Convert string sources to SourceItem
                        formatted_sources = [
                            SourceItem(title=str(s), url=None, publisher=None) 
                            for s in sources if s
                        ]
                except Exception as format_error:
                    logger.error(f"[API] Error formatting sources: {format_error}", exc_info=True)
                    # Fallback: create simple sources from strings
                    formatted_sources = [
                        SourceItem(title=str(s), url=None, publisher=None) 
                        for s in sources if s
                    ]
        
        logger.info(f"[API] Formatted {len(formatted_sources)} sources")
        
        # Ensure answer is a string
        answer = str(result.get("answer", ""))
        if not answer:
            answer = "I apologize, but I could not generate an answer for your query."
        
        return QueryResponse(
            answer=answer,
            answer_type=str(result.get("answer_type", "sagealpha_rag")),
            sources=formatted_sources  # Always a list, never None
        )
    except HTTPException:
        raise  # Re-raise HTTP exceptions as-is
    except Exception as e:
        # Log error server-side with full traceback
        logger.error(f"[API] Query processing failed: {str(e)}", exc_info=True)
        logger.error(f"[API] Exception type: {type(e).__name__}")
        
        # Provide more specific error messages for common issues
        error_msg = str(e).lower()
        if "chroma" in error_msg or "collection" in error_msg:
            detail = "Database connection error. Please try again later."
        elif "openai" in error_msg or "azure" in error_msg:
            detail = "AI service error. Please try again later."
        elif "timeout" in error_msg:
            detail = "Request timeout. Please try again with a simpler query."
        else:
            detail = "An error occurred while processing your query. Please try again."
        
        raise HTTPException(
            status_code=500,
            detail=detail
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
    - **answer_type**: "sagealpha_rag" (from documents), "sagealpha_ai_search" (from web search), "sagealpha_hybrid_search" (combined)
    - **sources**: List of document sources (empty list [] for LLM-only answers, never null)
    
    Example request:
    ```json
    {
      "query": "What is the revenue of Oracle Financial Services for FY2023?"
    }
    ```
    """
    return await _process_query(req)


# ================================
# Note: Do NOT run uvicorn here
# ================================
# 
# To run the app, use one of these commands:
# 
# Development (Windows):
#   uvicorn app:app --host 0.0.0.0 --port 8000 --reload
#
# Production (with Gunicorn):
#   gunicorn app:app -c gunicorn_config.py
#
# This prevents double-starting uvicorn and port binding conflicts.
