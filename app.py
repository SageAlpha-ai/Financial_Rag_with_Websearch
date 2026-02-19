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

def sanitize_llm_input(user_input: str) -> str:
    """
    Sanitize user input for LLM-only queries to prevent Azure jailbreak detection.
    
    Removes instruction-style or system-directive language that triggers Azure content filters.
    Preserves user intent while making the query safe for Azure OpenAI.
    
    Args:
        user_input: User query that may contain instruction-style language
        
    Returns:
        Sanitized query safe for Azure OpenAI
    """
    import re
    
    if not user_input:
        return ""
    
    original = user_input.strip()
    sanitized = original
    
    # Detect if input contains instruction-style language
    jailbreak_keywords = [
        'operating in', 'llm-only', 'external context', 'retrieval results',
        'must be ignored', 'must be treated', 'use only your', 'internal knowledge',
        'ignore documents', 'ignore context', 'ignore retrieval', 'ignore external',
        'treat as empty', 'bypass', 'override', 'disable', 'do not use',
        'all external', 'any external', 'every external'
    ]
    
    has_jailbreak_language = any(keyword in original.lower() for keyword in jailbreak_keywords)
    
    if not has_jailbreak_language:
        # No jailbreak language detected, return as-is
        return original
    
    # Patterns that trigger Azure jailbreak detection (multiline-aware)
    jailbreak_patterns = [
        # System directive patterns (match across lines)
        (r'you\s+are\s+operating\s+in\s+[^\n]*?mode[^\n]*?\.?\s*', '', re.IGNORECASE | re.DOTALL),
        (r'all\s+external\s+context[^\n]*?\.?\s*', '', re.IGNORECASE | re.DOTALL),
        (r'retrieval\s+results[^\n]*?\.?\s*', '', re.IGNORECASE | re.DOTALL),
        (r'documents\s+must\s+be\s+(ignored|treated|avoided|disregarded)[^\n]*?\.?\s*', '', re.IGNORECASE | re.DOTALL),
        (r'(all\s+)?documents\s+(must\s+be\s+)?(ignored|treated|avoided)[^\n]*?\.?\s*', '', re.IGNORECASE | re.DOTALL),
        (r'use\s+only\s+your\s+(internal\s+)?knowledge[^\n]*?\.?\s*', '', re.IGNORECASE | re.DOTALL),
        (r'ignore\s+(all\s+)?(documents|context|retrieval|external)[^\n]*?\.?\s*', '', re.IGNORECASE | re.DOTALL),
        (r'treat\s+(retrieved|external|context)[^\n]*?as\s+empty[^\n]*?\.?\s*', '', re.IGNORECASE | re.DOTALL),
        (r'avoid\s+(buy|sell|trading|investment)\s+instructions[^\n]*?\.?\s*', '', re.IGNORECASE | re.DOTALL),
        (r'do\s+not\s+use\s+(documents|context|retrieval)[^\n]*?\.?\s*', '', re.IGNORECASE | re.DOTALL),
        (r'bypass\s+(rag|retrieval|documents)[^\n]*?\.?\s*', '', re.IGNORECASE | re.DOTALL),
        (r'override\s+(system|rag|retrieval)[^\n]*?\.?\s*', '', re.IGNORECASE | re.DOTALL),
        (r'disable\s+(rag|retrieval|documents)[^\n]*?\.?\s*', '', re.IGNORECASE | re.DOTALL),
        (r'operate\s+outside\s+[^\n]*?\.?\s*', '', re.IGNORECASE | re.DOTALL),
        (r'act\s+as\s+if\s+[^\n]*?\.?\s*', '', re.IGNORECASE | re.DOTALL),
    ]
    
    # Remove jailbreak patterns
    for pattern, replacement, flags in jailbreak_patterns:
        sanitized = re.sub(pattern, replacement, sanitized, flags=flags)
    
    # Remove instruction-style sentence starters (line-by-line)
    lines = sanitized.split('\n')
    cleaned_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Remove instruction starters
        line = re.sub(r'^(you\s+are|you\s+must|you\s+should|you\s+need\s+to|you\s+cannot|you\s+shall)\s+', '', line, flags=re.IGNORECASE)
        line = re.sub(r'^(ignore|bypass|override|disable|avoid|treat)\s+', '', line, flags=re.IGNORECASE)
        line = re.sub(r'^(all|any|every)\s+(external|retrieved|document|context)\s+', '', line, flags=re.IGNORECASE)
        
        # Skip lines that are purely instructions
        if line and not any(keyword in line.lower() for keyword in ['ignore', 'bypass', 'override', 'disable', 'operating', 'must be', 'treat as']):
            cleaned_lines.append(line)
    
    sanitized = ' '.join(cleaned_lines)
    
    # Clean up multiple spaces
    sanitized = re.sub(r'\s+', ' ', sanitized)
    sanitized = sanitized.strip()
    
    # If sanitization removed everything meaningful, extract the core question
    if len(sanitized) < 10 or not sanitized:
        # Check if entire input was instruction-style
        instruction_ratio = sum(1 for keyword in jailbreak_keywords if keyword in original.lower()) / max(len(jailbreak_keywords), 1)
        
        if instruction_ratio > 0.3:  # More than 30% of keywords found - likely pure instructions
            # Entire input is instructions - return safe generic query
            sanitized = "Please provide your question or topic of interest."
        else:
            # Try to extract a question from the original input
            question_match = re.search(r'(what|how|why|when|where|who|explain|describe|tell\s+me|can\s+you|please)[^.]*[?.]?', original, re.IGNORECASE)
            if question_match:
                sanitized = question_match.group(0).strip()
            else:
                # Fallback: use first sentence that doesn't contain jailbreak keywords
                sentences = re.split(r'[.!?]\s+', original)
                for sentence in sentences:
                    sentence_clean = sentence.strip()
                    if len(sentence_clean) > 5 and not any(keyword in sentence_clean.lower() for keyword in jailbreak_keywords):
                        sanitized = sentence_clean
                        break
                if not sanitized or len(sanitized) < 5:
                    # Last resort: return a generic query
                    sanitized = "Please provide your question or topic of interest."
    
    # Final cleanup
    sanitized = sanitized.strip()
    if not sanitized or len(sanitized) < 3:
        return "Please provide your question."
    
    return sanitized


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
    strict_json: Optional[bool] = Field(
        False,
        description="If True, strictly enforces JSON output format for Node.js integration",
        examples=[False]
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
    # Detect runtime environment
    is_docker = os.path.exists("/.dockerenv") or os.getenv("DOCKER_CONTAINER") == "true"
    is_production = os.getenv("AZURE_APP_SERVICE") is not None or os.getenv("PYTHON_ENV") == "production"
    fail_fast = is_docker or is_production
    
    # Startup
    try:
        # Load and validate configuration (will raise ValueError if required vars missing)
        config = get_config()
        validate_config(config)
        logger.info("Configuration loaded and validated successfully")
        
        # Validate Azure OpenAI deployments (fail fast in Docker/production)
        try:
            validate_azure_openai_deployments(config)
        except ValueError as e:
            logger.error("=" * 60)
            logger.error("AZURE OPENAI DEPLOYMENT VALIDATION FAILED")
            logger.error("=" * 60)
            logger.error(str(e))
            logger.error("=" * 60)
            if fail_fast:
                logger.error("CRITICAL ERROR: Application cannot start in Docker/production without valid Azure OpenAI configuration.")
                logger.error("=" * 60)
                raise  # Fail fast in Docker/production
            else:
                logger.error("The application will not function correctly until deployments are fixed.")
                # Allow health checks in local dev only
        
        # Validate Chroma collection (non-blocking)
        try:
            from vectorstore.chroma_client import get_collection
            
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
                    logger.warning("Collection is empty. Ensure data is ingested via the external pipeline.")
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
                        logger.warning("Collection is empty. Ensure data is ingested via the external pipeline.")
                except ValueError as e:
                    logger.warning(f"Collection validation: {str(e)}")
                    logger.warning("Queries will fail until collection is created via ingestion.")
                logger.info("=" * 60)
        except Exception as e:
            logger.warning(f"Chroma collection validation skipped: {e}")
            logger.warning("Queries may fail if collection is not available.")
        
    except ValueError as e:
        # Configuration error - fail fast with clear message
        logger.error("=" * 60)
        logger.error("CRITICAL CONFIGURATION ERROR - APPLICATION CANNOT START")
        logger.error("=" * 60)
        logger.error(str(e))
        logger.error("")
        logger.error("Required environment variables are missing.")
        logger.error("")
        logger.error("For Docker, use:")
        logger.error("  docker run --env-file .env <image>")
        logger.error("")
        logger.error("For Azure App Service, set variables in:")
        logger.error("  Configuration → Application settings")
        logger.error("=" * 60)
        # Re-raise to prevent app from starting with invalid config
        raise
    
    logger.info("=" * 60)
    logger.info("AI RAG SERVICE STARTED")
    logger.info("=" * 60)
    if RAG_API_KEY:
        logger.info("API Key authentication: ENABLED")
    else:
        logger.info("API Key authentication: DISABLED")
    logger.info("=" * 60)
    logger.info("Server is running. Access API documentation at /docs")
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
        # Get input from request
        user_input = req.get_input()
        strict_mode = getattr(req, "strict_json", False)
        
        if not user_input or not user_input.strip():
            raise HTTPException(
                status_code=400,
                detail="Request body must contain 'query' or 'question'"
            )
        
        # Normalize input
        normalized_input = normalize_user_input(user_input)
        
        # Apply JSON strict mode instructions if requested
        if strict_mode:
            normalized_input += (
                "\n\nIMPORTANT: "
                "You MUST return ONLY valid JSON. "
                "Do NOT return markdown. "
                "Do NOT return explanation. "
                "Do NOT wrap in code blocks. "
                "Output must be a single JSON object."
            )
        
        logger.info(f"[API] Processing query: {normalized_input[:100]}...")
        
        # Determine intent (just for logging/metrics if needed, but we route EVERYTHING to orchestrator)
        # We removed strict intent gating to ensure hybrid search works for all queries.
        
        # Orchestrator Call
        # This handles RAG, Web Search, and LLM fallback internally.
        logger.info("[API] Delegating to orchestrator")
        try:
            if is_report_request(normalized_input):
                logger.info(f"[API] Generating detailed report for: {normalized_input[:50]}...")
                result = generate_report(normalized_input)
                
                if result.get("rag_used"):
                    result["confidence_level"] = "HIGH"
                else:
                    result["confidence_level"] = "LOW"
            else:
                result = answer_query_simple(normalized_input)
        except Exception as orchestrator_error:
            # Fallback if orchestrator crashes completely
            logger.error(f"[API] Orchestrator crashed: {orchestrator_error}", exc_info=True)
            return QueryResponse(
                answer="I apologize, but I encountered an internal error. Please try again later.",
                answer_type="system_error",
                sources=[]
            )

        # Validate result structure
        if not isinstance(result, dict):
            logger.error(f"[API] Invalid result type: {type(result)}")
            raise ValueError("Orchestrator returned invalid format")
        
        answer = result.get("answer", "No answer generated.")
        answer_type = result.get("answer_type", "sagealpha_rag")
        sources = result.get("sources", [])
        
        # Apply strict JSON validation and repair if requested
        if strict_mode and answer:
            import json
            try:
                # Try simple parse
                _ = json.loads(answer)
            except json.JSONDecodeError:
                # Attempt repair
                try:
                    cleaned = answer.strip()
                    if cleaned.startswith("```json"):
                        cleaned = cleaned[7:]
                    elif cleaned.startswith("```"):
                        cleaned = cleaned[3:]
                    if cleaned.endswith("```"):
                        cleaned = cleaned[:-3]
                    cleaned = cleaned.strip()
                    
                    # Validate again
                    _ = json.loads(cleaned)
                    answer = cleaned
                except Exception:
                    # Final safety fallback
                    logger.warning("[API] JSON Strict Mode: Failed to parse/repair LLM response. Wrapping safely.")
                    answer = json.dumps({
                        "response": answer,
                        "warning": "Model returned non-strict JSON output"
                    })
        
        # Source Formatting
        formatted_sources = []
        if sources:
            for s in sources:
                try:
                    if isinstance(s, dict):
                        formatted_sources.append(SourceItem(
                            title=str(s.get("title", "Unknown")),
                            url=s.get("url"),
                            publisher=s.get("publisher")
                        ))
                    elif hasattr(s, 'title'): # Check if it's already a SourceItem or similar object
                         formatted_sources.append(SourceItem(
                            title=s.title,
                            url=getattr(s, 'url', None),
                            publisher=getattr(s, 'publisher', None)
                        ))
                    else:
                        formatted_sources.append(SourceItem(title=str(s)))
                except Exception as e:
                    logger.warning(f"[API] Failed to format source: {s} - {e}")
                    formatted_sources.append(SourceItem(title="Unknown Source"))
        
        return QueryResponse(
            answer=str(answer),
            answer_type=str(answer_type),
            sources=formatted_sources
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[API] Query processing failed: {e}", exc_info=True)
        return QueryResponse(
            answer="I apologize, but I encountered an unexpected error.",
            answer_type="system_error",
            sources=[]
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
