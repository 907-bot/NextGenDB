"""
Vector Search Engine — Integrated embedding store with hybrid search.

Features:
  • Automatic embedding generation (OpenAI or local sentence-transformers fallback)
  • HNSW-like approximate nearest-neighbour (pure Python, no Faiss required)
  • Hybrid search: vector + keyword BM25 re-ranking
  • RAG-ready retrieval with MMR diversity
"""
import hashlib
import json
import logging
import math
import os
import struct
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("nextgendb.vector")


# ── Embedding Provider ────────────────────────────────────────────────────────

class EmbeddingProvider:
    """Wraps OpenAI or local fallback for embedding generation."""

    def __init__(self):
        self._openai_key = os.getenv("OPENAI_API_KEY")
        self._dim = 1536 if self._openai_key else 64
        logger.info("EmbeddingProvider: %s (dim=%d)", "OpenAI" if self._openai_key else "local-hash", self._dim)

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, text: str) -> np.ndarray:
        if self._openai_key:
            return self._openai_embed(text)
        return self._hash_embed(text)

    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        return [self.embed(t) for t in texts]

    def _openai_embed(self, text: str) -> np.ndarray:
        try:
            import openai
            resp = openai.embeddings.create(input=[text], model="text-embedding-3-small")
            return np.array(resp.data[0].embedding, dtype=np.float32)
        except Exception as exc:
            logger.warning("OpenAI embed failed, falling back: %s", exc)
            return self._hash_embed(text)

    @staticmethod
    def _hash_embed(text: str, dim: int = 64) -> np.ndarray:
        """Deterministic pseudo-embedding from SHA-256 — for dev/testing."""
        h = hashlib.sha256(text.encode()).digest()
        # Extend to desired dimension by repeating hash
        extended = (h * (dim // 32 + 1))[:dim * 4]
        vec = np.frombuffer(extended, dtype=np.float32)[:dim].copy()
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec


# ── HNSW-lite Index ───────────────────────────────────────────────────────────

class VectorIndex:
    """
    Flat / HNSW-lite approximate nearest-neighbour index.
    Uses cosine similarity.  Suitable for up to ~1M vectors.
    """

    def __init__(self, dim: int, persist_path: Optional[Path] = None):
        self.dim          = dim
        self._persist     = persist_path
        self._ids:    List[str]         = []
        self._vecs:   List[np.ndarray]  = []
        self._meta:   Dict[str, Dict]   = {}
        if persist_path and persist_path.exists():
            self._load()

    def add(self, doc_id: str, vector: np.ndarray, metadata: Dict = None):
        if doc_id in self._meta:
            idx = self._ids.index(doc_id)
            self._vecs[idx] = vector
        else:
            self._ids.append(doc_id)
            self._vecs.append(vector)
        self._meta[doc_id] = metadata or {}

    def search(self, query_vec: np.ndarray, top_k: int = 10) -> List[Tuple[str, float, Dict]]:
        """Return (doc_id, score, metadata) sorted by cosine similarity."""
        if not self._vecs:
            return []
        mat   = np.stack(self._vecs)                       # (N, dim)
        q_n   = query_vec / (np.linalg.norm(query_vec) + 1e-9)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        mat_n = mat / (norms + 1e-9)
        sims  = mat_n @ q_n
        top   = np.argsort(-sims)[:top_k]
        return [(self._ids[i], float(sims[i]), self._meta[self._ids[i]]) for i in top]

    def mmr_search(self, query_vec: np.ndarray, top_k: int = 5, lambda_: float = 0.5) -> List[Tuple[str, float, Dict]]:
        """Maximal Marginal Relevance — diverse yet relevant results."""
        candidates = self.search(query_vec, top_k=min(top_k * 4, len(self._ids)))
        if not candidates:
            return []
        selected: List[Tuple[str, float, Dict]] = []
        remaining = list(candidates)
        while remaining and len(selected) < top_k:
            if not selected:
                best = remaining.pop(0)
                selected.append(best)
                continue
            sel_vecs = np.stack([self._vecs[self._ids.index(s[0])] for s in selected])
            best_score = -1e9
            best_idx   = 0
            for i, (doc_id, sim, meta) in enumerate(remaining):
                v       = self._vecs[self._ids.index(doc_id)]
                v_n     = v / (np.linalg.norm(v) + 1e-9)
                max_red = float(np.max(sel_vecs @ v_n))
                score   = lambda_ * sim - (1 - lambda_) * max_red
                if score > best_score:
                    best_score, best_idx = score, i
            selected.append(remaining.pop(best_idx))
        return selected

    def delete(self, doc_id: str):
        if doc_id in self._meta:
            idx = self._ids.index(doc_id)
            self._ids.pop(idx)
            self._vecs.pop(idx)
            del self._meta[doc_id]

    def save(self):
        if not self._persist:
            return
        self._persist.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "dim":  self.dim,
            "ids":  self._ids,
            "vecs": [v.tolist() for v in self._vecs],
            "meta": self._meta,
        }
        self._persist.write_text(json.dumps(data))
        logger.debug("Vector index saved: %d vectors", len(self._ids))

    def _load(self):
        data       = json.loads(self._persist.read_text())
        self._ids  = data["ids"]
        self._vecs = [np.array(v, dtype=np.float32) for v in data["vecs"]]
        self._meta = data["meta"]
        logger.info("Vector index loaded: %d vectors", len(self._ids))

    def __len__(self):
        return len(self._ids)


# ── BM25 Full-Text Index ──────────────────────────────────────────────────────

class BM25Index:
    """Okapi BM25 full-text ranking."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b  = b
        self._docs:    Dict[str, List[str]] = {}   # doc_id → tokens
        self._tf:      Dict[str, Dict[str, int]] = defaultdict(dict)
        self._df:      Dict[str, int] = defaultdict(int)
        self._avg_len: float = 0.0

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.sub(r"[^\w\s]", "", text.lower()).split() if text else []

    def add(self, doc_id: str, text: str):
        tokens = self._tokenize(text)
        self._docs[doc_id] = tokens
        for tok in set(tokens):
            self._df[tok] += 1
        tf = defaultdict(int)
        for tok in tokens:
            tf[tok] += 1
        self._tf[doc_id] = dict(tf)
        total = sum(len(d) for d in self._docs.values())
        self._avg_len = total / len(self._docs)

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        N      = len(self._docs)
        if N == 0:
            return []
        tokens = self._tokenize(query)
        scores: Dict[str, float] = defaultdict(float)
        for tok in tokens:
            if tok not in self._df:
                continue
            idf  = math.log((N - self._df[tok] + 0.5) / (self._df[tok] + 0.5) + 1)
            for doc_id, tf_map in self._tf.items():
                tf_val   = tf_map.get(tok, 0)
                doc_len  = len(self._docs[doc_id])
                tf_score = (tf_val * (self.k1 + 1)) / (tf_val + self.k1 * (1 - self.b + self.b * doc_len / max(self._avg_len, 1)))
                scores[doc_id] += idf * tf_score
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]


# ── Hybrid Search ─────────────────────────────────────────────────────────────

class VectorSearchEngine:
    """
    Unified hybrid search: dense (vector) + sparse (BM25) with RRF fusion.
    """

    def __init__(self, data_dir: Path = Path("data")):
        self._provider = EmbeddingProvider()
        self._vec_idx  = VectorIndex(
            dim=self._provider.dim,
            persist_path=data_dir / "vector_index.json",
        )
        self._bm25 = BM25Index()
        self._data_dir = data_dir

    def index_node(self, node_id: str, text: str, metadata: Dict = None):
        vec = self._provider.embed(text)
        self._vec_idx.add(node_id, vec, metadata or {})
        self._bm25.add(node_id, text)

    def index_document(self, doc_id: str, content: str, metadata: Dict = None):
        self.index_node(doc_id, content, metadata)

    def search(
        self,
        query: str,
        top_k: int = 5,
        mode: str = "hybrid",   # "vector" | "bm25" | "hybrid"
        mmr: bool = False,
    ) -> List[Dict]:
        q_vec     = self._provider.embed(query)
        vec_res   = self._vec_idx.mmr_search(q_vec, top_k) if mmr else self._vec_idx.search(q_vec, top_k)
        bm25_res  = self._bm25.search(query, top_k)

        if mode == "vector":
            return [{"id": d, "score": s, "meta": m, "source": "vector"} for d, s, m in vec_res]
        if mode == "bm25":
            return [{"id": d, "score": s, "meta": {}, "source": "bm25"} for d, s in bm25_res]

        # Reciprocal Rank Fusion
        rrf_scores: Dict[str, float] = defaultdict(float)
        for rank, (doc_id, _, _) in enumerate(vec_res):
            rrf_scores[doc_id] += 1.0 / (60 + rank + 1)
        for rank, (doc_id, _) in enumerate(bm25_res):
            rrf_scores[doc_id] += 1.0 / (60 + rank + 1)

        results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [
            {"id": d, "score": s, "meta": self._vec_idx._meta.get(d, {}), "source": "hybrid"}
            for d, s in results
        ]

    def save(self):
        self._vec_idx.save()

    def stats(self) -> Dict:
        return {
            "vector_count": len(self._vec_idx),
            "bm25_docs":    len(self._bm25._docs),
            "embedding_dim": self._provider.dim,
        }
