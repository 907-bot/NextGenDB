import os
import shutil
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from ..vector.search import BM25Index

logger = logging.getLogger("nextgendb.storage.document")

class DocumentEngine:
    """
    A traditional 'vectorless' document database engine.
    Stores physical files and provides full-text keyword search (BM25).
    """

    def __init__(self, data_dir: Path = Path("data")):
        self.root_dir = data_dir / "documents"
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.index = BM25Index()
        self.metadata: Dict[str, Dict[str, Any]] = {}
        self._load_existing_docs()

    def _load_existing_docs(self):
        """Re-index all documents in the storage directory."""
        logger.info("DocumentEngine: Re-indexing existing files in %s", self.root_dir)
        for file_path in self.root_dir.glob("*.*"):
            doc_id = file_path.name
            try:
                content = file_path.read_text(errors='ignore')
                self.index.add(doc_id, content)
                self.metadata[doc_id] = {
                    "size": file_path.stat().st_size,
                    "path": str(file_path),
                    "type": file_path.suffix.lstrip('.')
                }
            except Exception as e:
                logger.error("Failed to index %s: %s", doc_id, e)

    def store_document(self, doc_id: str, content: str, metadata: Dict[str, Any] = None):
        """Store a file and index its content (Vectorless)."""
        file_path = self.root_dir / doc_id
        file_path.write_text(content)
        
        self.index.add(doc_id, content)
        self.metadata[doc_id] = {
            "size": len(content),
            "path": str(file_path),
            "type": doc_id.split('.')[-1] if '.' in doc_id else "txt",
            **(metadata or {})
        }
        logger.info("Document stored and indexed: %s", doc_id)

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Perform 'vectorless' keyword search using BM25."""
        results = self.index.search(query, top_k=top_k)
        formatted = []
        for doc_id, score in results:
            formatted.append({
                "doc_id": doc_id,
                "score": round(score, 4),
                "metadata": self.metadata.get(doc_id, {}),
                "snippet": self._get_snippet(doc_id, query)
            })
        return formatted

    def _get_snippet(self, doc_id: str, query: str, length: int = 150) -> str:
        """Extract a snippet of text around the first matching keyword."""
        file_path = self.root_dir / doc_id
        if not file_path.exists():
            return ""
        
        content = file_path.read_text(errors='ignore')
        query_terms = query.lower().split()
        
        lower_content = content.lower()
        first_match = -1
        for term in query_terms:
            idx = lower_content.find(term)
            if idx != -1:
                first_match = idx
                break
        
        if first_match == -1:
            return content[:length] + "..."
            
        start = max(0, first_match - length // 2)
        end = min(len(content), start + length)
        return ("..." if start > 0 else "") + content[start:end].replace('\n', ' ') + "..."

    def list_documents(self) -> List[str]:
        return list(self.metadata.keys())
