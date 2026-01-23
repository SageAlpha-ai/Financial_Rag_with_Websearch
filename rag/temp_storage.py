"""
Temporary Document Storage Manager

Manages temporary storage for web-retrieved documents.
Documents are automatically cleaned up when:
- Session expires
- Cache TTL expires (default: 24 hours)
- Server restarts
"""

import logging
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class TemporaryStorageManager:
    """
    Manages temporary storage for web-retrieved documents.
    
    Features:
    - Session-based storage
    - Automatic cleanup
    - TTL-based expiration
    - Metadata tracking
    """
    
    def __init__(
        self, 
        base_dir: Optional[Path] = None,
        ttl_hours: int = 24
    ):
        """
        Initialize temporary storage manager.
        
        Args:
            base_dir: Base directory for temp storage (defaults to system temp)
            ttl_hours: Time-to-live in hours (default: 24)
        """
        if base_dir is None:
            base_dir = Path(tempfile.gettempdir()) / "rag_web_docs"
        
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        self.ttl_hours = ttl_hours
        self.sessions: Dict[str, Dict] = {}
        
        logger.info(f"[TEMP_STORAGE] Initialized with base_dir={self.base_dir}, TTL={ttl_hours}h")
        
        # Cleanup old files on startup
        self.cleanup_expired()
    
    def store_document(
        self,
        file_path: Path,
        metadata: Dict,
        session_id: Optional[str] = None
    ) -> Dict:
        """
        Store a document in temporary storage.
        
        Args:
            file_path: Path to document file
            metadata: Document metadata
            session_id: Optional session ID (auto-generated if None)
            
        Returns:
            Storage metadata with file path and session info
        """
        if session_id is None:
            session_id = f"session_{int(time.time())}"
        
        # Create session directory
        session_dir = self.base_dir / session_id
        session_dir.mkdir(exist_ok=True)
        
        # Copy file to session directory
        dest_path = session_dir / file_path.name
        shutil.copy2(file_path, dest_path)
        
        # Create storage metadata
        storage_metadata = {
            **metadata,
            "session_id": session_id,
            "file_path": str(dest_path),
            "stored_at": datetime.utcnow().isoformat(),
            "expires_at": (datetime.utcnow() + timedelta(hours=self.ttl_hours)).isoformat(),
            "ephemeral": True
        }
        
        # Track session
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                "created_at": datetime.utcnow().isoformat(),
                "documents": []
            }
        
        self.sessions[session_id]["documents"].append(storage_metadata)
        
        logger.info(f"[TEMP_STORAGE] Stored document: {dest_path} (session: {session_id})")
        
        return storage_metadata
    
    def get_session_documents(self, session_id: str) -> List[Dict]:
        """Get all documents for a session."""
        if session_id not in self.sessions:
            return []
        
        return self.sessions[session_id].get("documents", [])
    
    def cleanup_session(self, session_id: str) -> int:
        """
        Clean up all documents for a session.
        
        Returns:
            Number of files deleted
        """
        if session_id not in self.sessions:
            return 0
        
        session_dir = self.base_dir / session_id
        deleted_count = 0
        
        if session_dir.exists():
            try:
                # Count files before deletion
                deleted_count = len(list(session_dir.glob("*")))
                
                # Delete directory and all contents
                shutil.rmtree(session_dir)
                
                logger.info(f"[TEMP_STORAGE] Cleaned up session {session_id} ({deleted_count} files)")
            except Exception as e:
                logger.error(f"[TEMP_STORAGE] Failed to cleanup session {session_id}: {e}")
        
        # Remove from sessions dict
        del self.sessions[session_id]
        
        return deleted_count
    
    def cleanup_expired(self) -> int:
        """
        Clean up all expired documents.
        
        Returns:
            Number of files deleted
        """
        deleted_count = 0
        now = datetime.utcnow()
        
        # Check all session directories
        for session_dir in self.base_dir.iterdir():
            if not session_dir.is_dir():
                continue
            
            session_id = session_dir.name
            
            # Check if session has expired documents
            if session_id in self.sessions:
                documents = self.sessions[session_id].get("documents", [])
                expired_docs = [
                    doc for doc in documents
                    if datetime.fromisoformat(doc.get("expires_at", "1970-01-01")) < now
                ]
                
                if expired_docs:
                    # Cleanup entire session if any documents expired
                    deleted_count += self.cleanup_session(session_id)
            else:
                # Session not tracked, check directory age
                try:
                    dir_mtime = datetime.fromtimestamp(session_dir.stat().st_mtime)
                    if (now - dir_mtime) > timedelta(hours=self.ttl_hours):
                        # Delete old directory
                        file_count = len(list(session_dir.glob("*")))
                        shutil.rmtree(session_dir)
                        deleted_count += file_count
                except Exception as e:
                    logger.warning(f"[TEMP_STORAGE] Failed to check/cleanup {session_dir}: {e}")
        
        if deleted_count > 0:
            logger.info(f"[TEMP_STORAGE] Cleaned up {deleted_count} expired files")
        
        return deleted_count
    
    def get_document_path(self, storage_metadata: Dict) -> Optional[Path]:
        """Get file path from storage metadata."""
        file_path = storage_metadata.get("file_path")
        if file_path and Path(file_path).exists():
            return Path(file_path)
        return None
    
    def cleanup_all(self) -> int:
        """
        Clean up all temporary documents (use with caution).
        
        Returns:
            Number of files deleted
        """
        deleted_count = 0
        
        if self.base_dir.exists():
            for session_dir in self.base_dir.iterdir():
                if session_dir.is_dir():
                    file_count = len(list(session_dir.glob("*")))
                    try:
                        shutil.rmtree(session_dir)
                        deleted_count += file_count
                    except Exception as e:
                        logger.error(f"[TEMP_STORAGE] Failed to delete {session_dir}: {e}")
        
        self.sessions.clear()
        
        logger.info(f"[TEMP_STORAGE] Cleaned up all temporary documents ({deleted_count} files)")
        
        return deleted_count


# Global instance
_temp_storage: Optional[TemporaryStorageManager] = None


def get_temp_storage() -> TemporaryStorageManager:
    """Get global temporary storage manager instance."""
    global _temp_storage
    if _temp_storage is None:
        _temp_storage = TemporaryStorageManager()
    return _temp_storage
