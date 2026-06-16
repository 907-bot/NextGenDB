"""Tests for Vector Search Engine"""
import numpy as np
import pytest

from backend.vector.search import VectorIndex, BM25Index, VectorSearchEngine


class TestVectorIndex:
    """Vector similarity search"""

    def test_vector_add_and_search(self, tmp_path):
        """Should add vectors and retrieve by similarity"""
        index = VectorIndex(dim=64, persist_path=tmp_path / "index.json")
        
        v1 = np.random.randn(64).astype(np.float32)
        v1 /= np.linalg.norm(v1)
        
        index.add("doc1", v1, {"title": "Document 1"})
        results = index.search(v1, top_k=1)
        
        assert len(results) == 1
        assert results[0][0] == "doc1"
        assert abs(results[0][1] - 1.0) < 0.01  # Should be ~1.0 (identical)

    def test_vector_mmr_search(self, tmp_path):
        """MMR should return diverse results"""
        index = VectorIndex(dim=32, persist_path=tmp_path / "index.json")
        
        # Add orthogonal vectors
        v1 = np.zeros(32, dtype=np.float32)
        v1[0] = 1.0
        
        v2 = np.zeros(32, dtype=np.float32)
        v2[1] = 1.0
        
        v3 = np.zeros(32, dtype=np.float32)
        v3[2] = 1.0
        
        index.add("doc1", v1)
        index.add("doc2", v2)
        index.add("doc3", v3)
        
        query = np.zeros(32, dtype=np.float32)
        query[0] = 1.0
        
        results = index.mmr_search(query, top_k=2, lambda_=0.5)
        assert len(results) == 2

    def test_vector_persistence(self, tmp_path):
        """Should save and load vectors"""
        persist_path = tmp_path / "vectors.json"
        index1 = VectorIndex(dim=16, persist_path=persist_path)
        
        v = np.ones(16, dtype=np.float32)
        v /= np.linalg.norm(v)
        index1.add("doc1", v, {"text": "hello"})
        index1.save()
        
        # Load in new index
        index2 = VectorIndex(dim=16, persist_path=persist_path)
        assert len(index2) == 1
        assert index2._meta["doc1"]["text"] == "hello"


class TestBM25Index:
    """Full-text search"""

    def test_bm25_ranking(self):
        """Should rank documents by BM25 score"""
        index = BM25Index()
        
        index.add("doc1", "the quick brown fox jumps over the lazy dog")
        index.add("doc2", "the lazy cat sleeps on the mat")
        index.add("doc3", "quick and brown animals run fast")
        
        results = index.search("quick brown", top_k=3)
        
        assert len(results) == 3
        # doc1 and doc3 should rank higher than doc2
        assert results[0][0] in ["doc1", "doc3"]

    def test_bm25_tokenization(self):
        """Should handle punctuation correctly"""
        index = BM25Index()
        
        index.add("doc1", "Hello, World!")
        index.add("doc2", "hello world")
        
        results = index.search("hello world", top_k=2)
        assert len(results) == 2


class TestHybridSearch:
    """Hybrid vector + text search"""

    def test_hybrid_search_rrf_fusion(self, tmp_path):
        """Should fuse vector and BM25 results with RRF"""
        engine = VectorSearchEngine(tmp_path)
        
        engine.index_node("doc1", "machine learning is great")
        engine.index_node("doc2", "deep learning models")
        engine.index_node("doc3", "neural networks research")
        
        results = engine.search("machine learning", mode="hybrid", top_k=3)
        
        assert len(results) == 3
        assert all(r["source"] == "hybrid" for r in results)

    def test_search_modes(self, tmp_path):
        """Should support vector-only and BM25-only modes"""
        engine = VectorSearchEngine(tmp_path)
        
        engine.index_node("doc1", "python programming")
        engine.index_node("doc2", "java programming")
        
        vector_results = engine.search("python", mode="vector", top_k=2)
        assert all(r["source"] == "vector" for r in vector_results)
        
        bm25_results = engine.search("python", mode="bm25", top_k=2)
        assert all(r["source"] == "bm25" for r in bm25_results)
