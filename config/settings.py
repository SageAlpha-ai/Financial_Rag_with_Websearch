"""
Centralized Configuration

Single source of truth for all environment variables.
Loaded once at startup, shared across all modules.
"""

import os
from dataclasses import dataclass
from typing import Optional, List

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
    chat_deployment: str
    planner_deployment: str
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
    supported_companies: List[str]
    enable_query_planner: bool
    enable_web_search: bool


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
        chat_deployment=_get_required_env("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"),
        planner_deployment=_get_optional_env("AZURE_OPENAI_PLANNER_CHAT_DEPLOYMENT_NAME"),
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
        collection_name=_get_optional_env("CHROMA_COLLECTION_NAME", "compliance"),
    )
    
    # Supported Companies
    supported_companies_str = _get_optional_env("SUPPORTED_COMPANIES", "")
    supported_companies = [c.strip() for c in supported_companies_str.split(",") if c.strip()]
    
    # Feature Flags
    enable_query_planner = _get_optional_env("ENABLE_QUERY_PLANNER", "false").lower() == "true"
    enable_web_search = _get_optional_env("ENABLE_WEB_SEARCH", "false").lower() == "true"
    
    return AppConfig(
        azure_openai=azure_openai,
        azure_blob=azure_blob,
        chroma_cloud=chroma_cloud,
        supported_companies=supported_companies,
        enable_query_planner=enable_query_planner,
        enable_web_search=enable_web_search,
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
    _log_env_var_status("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME", bool(os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")),
                       os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME", "not set"))
    _log_env_var_status("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME", bool(os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME")),
                       os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME", "not set"))
    
    logger.info("")
    logger.info("Loaded Configuration:")
    logger.info(f"  Endpoint: {config.azure_openai.endpoint[:50]}...")
    logger.info(f"  Chat Deployment (Response): {config.azure_openai.chat_deployment}")
    logger.info(f"  Chat Deployment (Planner): {config.azure_openai.planner_deployment or 'Not Set'}")
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
    
    # Supported Companies
    logger.info("")
    logger.info("Supported Companies:")
    if config.supported_companies:
        for company in config.supported_companies:
            logger.info(f"  - {company}")
    else:
        logger.warning("  WARNING: No companies supported (SUPPORTED_COMPANIES is empty)")
    
    logger.info("")
    logger.info("Feature Flags:")
    logger.info(f"  Query Planner: {'ENABLED' if config.enable_query_planner else 'DISABLED'}")
    logger.info(f"  Web Search: {'ENABLED' if config.enable_web_search else 'DISABLED'}")
    
    logger.info("=" * 60)


# Singleton config instance
_config: Optional[AppConfig] = None


def get_config() -> AppConfig:
    """Gets the singleton config instance, loading if needed."""
    global _config
    if _config is None:
        _config = load_config()
    return _config
