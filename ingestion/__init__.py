"""Ingestion module - Load, chunk, embed, store."""
from .azure_blob_loader import load_azure_documents
from .chunking import chunk_documents, chunk_local_documents
from .embed_and_store import embed_and_store_documents
