"""
Centralized Configuration

Single source of truth for all environment variables.
Loaded once at startup, shared across all modules.
"""

import os
from dataclasses import dataclass
from typing import Optional

# Only load .env in development (not in production)
# In production (Render), all config comes from environment variables
if os.getenv("RENDER") is None and os.getenv("PYTHON_ENV") != "production":
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass  # dotenv not installed, use environment variables only


@dataclass(frozen=True)
class AzureOpenAIConfig:
    """Azure OpenAI configuration."""
    api_key: str
    endpoint: str
    api_version: str
    chat_deployment: str
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
    
    return AppConfig(
        azure_openai=azure_openai,
        azure_blob=azure_blob,
        chroma_cloud=chroma_cloud,
    )


def validate_config(config: AppConfig) -> None:
    """Validates configuration and prints status."""
    print("=" * 60)
    print("CONFIGURATION LOADED")
    print("=" * 60)
    
    # Azure OpenAI
    print(f"Azure OpenAI Endpoint: {config.azure_openai.endpoint[:50]}...")
    print(f"Azure OpenAI Chat Deployment: {config.azure_openai.chat_deployment}")
    print(f"Azure OpenAI Embeddings Deployment: {config.azure_openai.embeddings_deployment}")
    
    # Azure Blob
    print(f"Azure Blob Container: {config.azure_blob.container_name}")
    
    # Chroma Cloud
    print(f"Chroma Cloud Host: {config.chroma_cloud.host}")
    print(f"Chroma Cloud Tenant: {config.chroma_cloud.tenant}")
    print(f"Chroma Cloud Database: {config.chroma_cloud.database}")
    print(f"Chroma Cloud Collection: {config.chroma_cloud.collection_name}")
    
    print("=" * 60)


# Singleton config instance
_config: Optional[AppConfig] = None


def get_config() -> AppConfig:
    """Gets the singleton config instance, loading if needed."""
    global _config
    if _config is None:
        _config = load_config()
    return _config
