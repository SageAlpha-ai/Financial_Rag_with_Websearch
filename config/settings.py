"""
Centralized Configuration

Single source of truth for all environment variables.
Loaded once at startup, shared across all modules.
"""

import os
from dataclasses import dataclass, field
from typing import Optional

# Only load .env in local development (NOT in Docker or production)
# In Docker/production, all config MUST come from environment variables
# Docker users MUST use: docker run --env-file .env
is_docker = os.path.exists("/.dockerenv") or os.getenv("DOCKER_CONTAINER") == "true"
is_production = os.getenv("AZURE_APP_SERVICE") is not None or os.getenv("PYTHON_ENV") == "production"

if not is_docker and not is_production:
    # Local development only - load .env file
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass  # dotenv not installed, use environment variables only
elif is_docker:
    # Docker environment - .env must be loaded via --env-file
    import logging
    logger = logging.getLogger(__name__)
    logger.warning("=" * 60)
    logger.warning("DOCKER ENVIRONMENT DETECTED")
    logger.warning("=" * 60)
    logger.warning("Environment variables must be provided via:")
    logger.warning("  docker run --env-file .env <image>")
    logger.warning("  OR")
    logger.warning("  docker run -e VAR1=value1 -e VAR2=value2 <image>")
    logger.warning("=" * 60)


@dataclass(frozen=True)
class AzureOpenAIConfig:
    """Azure OpenAI configuration."""
    api_key: str
    endpoint: str
    api_version: str
    large_chat_deployment: str
    embeddings_deployment: str


@dataclass(frozen=True)
class AzureBlobConfig:
    """Azure Blob Storage configuration."""
    connection_string: str
    container_name: str


@dataclass(frozen=True)
class ChromaCloudConfig:
    """Chroma Cloud configuration."""
    host: str
    api_key: str
    tenant: str
    database: str
    collection_name: str


@dataclass(frozen=True)
class AppConfig:
    """Complete application configuration."""
    azure_openai: AzureOpenAIConfig
    azure_blob: AzureBlobConfig
    chroma_cloud: ChromaCloudConfig
    evidence_adequacy: dict = field(
        default_factory=lambda: EVIDENCE_ADEQUACY_CONFIG
    )
    collection_governance: dict = field(
        default_factory=lambda: COLLECTION_GOVERNANCE
    )
    llm_cost: dict = field(default_factory=lambda: LLM_COST_CONFIG)


# ---------------------------------------------------------------------------
# Web Search Source Priority Configuration
# Used by web_search.py to rank, allow, and block domains in search results.
# ---------------------------------------------------------------------------

# Preferred domains in priority order (highest priority first)
PRIORITY_WEB_DOMAINS: list[str] = [
    "rbi.org.in",
    "nseindia.com",
    "bseindia.com",
    "services.india.gov.in",
]

# Suffix-matched domains allowed as fallback when no priority domain matches
ALLOWED_FALLBACK_DOMAINS: list[str] = [
    ".gov.in",
]

# Domains whose results should always be excluded
BLOCKED_DOMAINS: list[str] = [
    "wikipedia.org",
    "google.com",
    "google.co.in",
]

# ---------------------------------------------------------------------------
# Evidence Adequacy Evaluator configuration
# Weights, thresholds, and distance normalization for the hybrid scoring
# system in rag/evidence_adequacy.py.  Not environment-dependent yet —
# future iterations may override individual keys from env vars.
# ---------------------------------------------------------------------------

EVIDENCE_ADEQUACY_CONFIG: dict = {
    "weights": {
        "llm": 0.5,
        "retrieval": 0.3,
        "doc_type": 0.2,
    },
    "thresholds": {
        "use_rag": 0.65,
        "escalate_web": 0.45,
    },
    "retrieval_distance_max": 1.5,
    "retrieval_weights": {
        "base_similarity": 0.4,
        "gap": 0.2,
        "variance": 0.15,
        "diversity": 0.15,
        "alignment": 0.10,
    },
    "metadata_alignment": {
        "enabled": True,
        "company_boost_weight": 0.08,
        "year_boost_weight": 0.05,
        "max_total_boost": 0.15,
    },
    "retrieval_stability": {
        "enabled": True,
        "dominance_threshold": 0.7,
        "dominance_penalty": 0.02,
        "boost_dampening_threshold": 0.8,
    },
    "retrieval_trust": {
        "enabled": True,
        "distance_gap_threshold": 0.25,
        "min_entropy_normalizer": 1.0,
    },
    "version": "v1_hybrid",
}

# ---------------------------------------------------------------------------
# LLM cost telemetry: per-token USD estimates for observability.
# ---------------------------------------------------------------------------
LLM_COST_CONFIG: dict = {
    "enabled": True,
    "pricing": {
        "gpt-4o-mini": {
            "input": 0.00000015,
            "output": 0.0000006,
        },
        "gpt-5-chat": {
            "input": 0.000003,
            "output": 0.000009,
        },
        "text-embedding-3-large": {
            "input": 0.00000013,
            "output": 0.0,
        },
    },
}

# ---------------------------------------------------------------------------
# Collection governance: dedupe, allowlist/denylist for list_all_collections.
# ---------------------------------------------------------------------------
COLLECTION_GOVERNANCE: dict = {
    "enabled": True,
    "allowlist": [],   # if empty → allow all
    "denylist": [],    # explicit exclusion
    "normalize": True,
}


# ---------------------------------------------------------------------------
# Debug / audit mode
# When True, internal audit metadata (_meta) is included in API responses.
# Can also be activated per-request via the X-Debug: true header.
# ---------------------------------------------------------------------------
DEBUG_MODE: bool = os.getenv("DEBUG_MODE", "false").lower() == "true"


def _get_required_env(var_name: str) -> str:
    """Gets required environment variable or raises error."""
    value = os.getenv(var_name)
    if not value:
        raise ValueError(f"Missing required environment variable: {var_name}")
    return value


def _get_optional_env(var_name: str, default: str = "") -> str:
    """Gets optional environment variable with default."""
    return os.getenv(var_name, default)


def load_config() -> AppConfig:
    """
    Loads and validates all configuration from environment.
    Call once at app startup.
    """
    # Azure OpenAI
    azure_openai = AzureOpenAIConfig(
        api_key=_get_required_env("AZURE_OPENAI_API_KEY"),
        endpoint=_get_required_env("AZURE_OPENAI_ENDPOINT"),
        api_version=_get_required_env("AZURE_OPENAI_API_VERSION"),
        large_chat_deployment=_get_required_env("AZURE_OPENAI_LARGE_CHAT_DEPLOYMENT_NAME"),
        embeddings_deployment=_get_required_env("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME"),
    )
    
    # Azure Blob Storage
    azure_blob = AzureBlobConfig(
        connection_string=_get_required_env("AZURE_STORAGE_CONNECTION_STRING"),
        container_name=_get_required_env("AZURE_BLOB_CONTAINER_NAME"),
    )
    
    # Chroma Cloud
    chroma_cloud = ChromaCloudConfig(
        host=_get_required_env("CHROMA_HOST"),
        api_key=_get_required_env("CHROMA_API_KEY"),
        tenant=_get_required_env("CHROMA_TENANT"),
        database=_get_required_env("CHROMA_DATABASE"),
        collection_name=_get_optional_env("CHROMA_COLLECTION_NAME", "DYNAMIC_COLLECTION_REQUIRED"),
    )
    
    return AppConfig(
        azure_openai=azure_openai,
        azure_blob=azure_blob,
        chroma_cloud=chroma_cloud,
    )


def _mask_secret(value: str, show_chars: int = 4) -> str:
    """Mask secret values for logging (show first N chars + ...)."""
    if not value or len(value) <= show_chars:
        return "***"
    return value[:show_chars] + "..." + ("*" * min(8, len(value) - show_chars - 3))


def _log_env_var_status(var_name: str, is_set: bool, masked_value: str = None) -> None:
    """Log environment variable detection status."""
    import logging
    logger = logging.getLogger(__name__)
    status = "✓ SET" if is_set else "✗ MISSING"
    value_info = f" (value: {masked_value})" if masked_value else ""
    logger.info(f"  {status}: {var_name}{value_info}")


def validate_config(config: AppConfig) -> None:
    """Validates configuration and prints status with masked secrets."""
    import logging
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 60)
    logger.info("CONFIGURATION VALIDATION")
    logger.info("=" * 60)
    
    # Log Azure OpenAI environment variables (masked)
    logger.info("Azure OpenAI Configuration:")
    _log_env_var_status("AZURE_OPENAI_API_KEY", bool(os.getenv("AZURE_OPENAI_API_KEY")), 
                       _mask_secret(os.getenv("AZURE_OPENAI_API_KEY", "")))
    _log_env_var_status("AZURE_OPENAI_ENDPOINT", bool(os.getenv("AZURE_OPENAI_ENDPOINT")),
                       os.getenv("AZURE_OPENAI_ENDPOINT", "not set")[:50] + "..." if os.getenv("AZURE_OPENAI_ENDPOINT") else None)
    _log_env_var_status("AZURE_OPENAI_API_VERSION", bool(os.getenv("AZURE_OPENAI_API_VERSION")),
                       os.getenv("AZURE_OPENAI_API_VERSION", "not set"))
    _log_env_var_status("AZURE_OPENAI_LARGE_CHAT_DEPLOYMENT_NAME", bool(os.getenv("AZURE_OPENAI_LARGE_CHAT_DEPLOYMENT_NAME")),
                       os.getenv("AZURE_OPENAI_LARGE_CHAT_DEPLOYMENT_NAME", "not set"))
    _log_env_var_status("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME", bool(os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME")),
                       os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME", "not set"))
    
    logger.info("")
    logger.info("Loaded Configuration:")
    logger.info(f"  Endpoint: {config.azure_openai.endpoint[:50]}...")
    logger.info(f"  Large Chat Deployment: {config.azure_openai.large_chat_deployment}")
    logger.info(f"  Embeddings Deployment: {config.azure_openai.embeddings_deployment}")
    logger.info(f"  API Version: {config.azure_openai.api_version}")
    
    # Azure Blob
    logger.info("")
    logger.info("Azure Blob Storage:")
    logger.info(f"  Container: {config.azure_blob.container_name}")
    
    # Chroma Cloud
    logger.info("")
    logger.info("Chroma Cloud:")
    logger.info(f"  Host: {config.chroma_cloud.host}")
    logger.info(f"  Tenant: {config.chroma_cloud.tenant}")
    logger.info(f"  Database: {config.chroma_cloud.database}")
    logger.info(f"  Collection: {config.chroma_cloud.collection_name}")
    
    logger.info("=" * 60)


# Singleton config instance
_config: Optional[AppConfig] = None


def get_config() -> AppConfig:
    """Gets the singleton config instance, loading if needed."""
    global _config
    if _config is None:
        _config = load_config()
    return _config
