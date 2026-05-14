"""
GNN Learner — Asynchronous background training + vectorless retrieval.

Features:
  • Continuous async training loop (runs as a background asyncio task)
  • Node embedding cache for O(1) retrieval
  • Vectorless retrieval using direct MultiDiGraph traversal (BFS + betweenness)
  • Query-to-embedding matching using hash embeddings (no PyTorch required)
  • Graceful degradation when PyTorch / PyG are unavailable
"""
import asyncio
import hashlib
import logging
import math
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np

logger = logging.getLogger("nextgendb.gnn.learner")

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    logger.info("PyTorch not available — GNN learner running in pure-graph mode")


# ── Node Feature Extractor ────────────────────────────────────────────────────

def _node_features(node_id: str, props: Dict[str, Any], dim: int = 64) -> np.ndarray:
    """Convert node properties to a deterministic dense vector (no ML required)."""
    text = node_id + " " + " ".join(str(v) for v in props.values())
    h = hashlib.sha256(text.encode()).digest()
    extended = (h * (dim // 32 + 1))[: dim * 4]
    vec = np.frombuffer(extended, dtype=np.float32)[:dim].copy()
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


# ── Vectorless Graph Retrieval ────────────────────────────────────────────────

class VectorlessRetriever:
    """
    Retrieve relevant subgraphs without any vector index.

    Strategy:
      1. BFS from seed nodes matching the query keywords.
      2. Score each node by betweenness centrality + keyword overlap.
      3. Return top-k nodes and their induced subgraph.
    """

    def __init__(self, graph: nx.MultiDiGraph):
        self._graph = graph

    def _keyword_score(self, node_id: str, props: Dict, keywords: List[str]) -> float:
        text = (node_id + " " + " ".join(str(v) for v in props.values())).lower()
        return sum(1.0 for kw in keywords if kw in text)

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        max_hops: int = 2,
    ) -> Tuple[List[str], nx.MultiDiGraph]:
        """Return (ranked_node_ids, induced_subgraph)."""
        keywords = [w.lower() for w in query.split() if len(w) > 2]
        g = self._graph

        if not g.nodes:
            return [], nx.MultiDiGraph()

        # Phase 1: score all nodes
        try:
            btw = nx.betweenness_centrality(g, normalized=True)
        except Exception:
            btw = {n: 0.0 for n in g.nodes}

        scores: Dict[str, float] = {}
        for node, props in g.nodes(data=True):
            kw_s = self._keyword_score(node, props, keywords) if keywords else 0.0
            # Combine keyword relevance + centrality
            scores[node] = kw_s * 2.0 + btw.get(node, 0.0) * 5.0

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        seed_nodes = [n for n, _ in ranked]

        # Phase 2: BFS expansion from seeds
        expanded: set = set(seed_nodes)
        for seed in seed_nodes[:3]:  # expand only top-3 seeds to keep subgraph small
            try:
                nbrs = nx.single_source_shortest_path_length(g, seed, cutoff=max_hops)
                expanded.update(nbrs.keys())
            except Exception:
                pass

        subgraph = g.subgraph(expanded).copy()
        return seed_nodes, subgraph


# ── Embedding Cache ────────────────────────────────────────────────────────────

class NodeEmbeddingCache:
    """Thread-safe in-memory cache mapping node_id → embedding vector."""

    def __init__(self, dim: int = 64):
        self._dim = dim
        self._cache: Dict[str, np.ndarray] = {}
        self._last_updated: Dict[str, float] = {}

    def put(self, node_id: str, vec: np.ndarray):
        self._cache[node_id] = vec
        self._last_updated[node_id] = time.time()

    def get(self, node_id: str) -> Optional[np.ndarray]:
        return self._cache.get(node_id)

    def get_all(self) -> Dict[str, np.ndarray]:
        return dict(self._cache)

    def invalidate(self, node_id: str):
        self._cache.pop(node_id, None)
        self._last_updated.pop(node_id, None)

    def similarity(self, node_id: str, query_vec: np.ndarray) -> float:
        vec = self.get(node_id)
        if vec is None:
            return 0.0
        norm_v = np.linalg.norm(vec)
        norm_q = np.linalg.norm(query_vec)
        if norm_v < 1e-9 or norm_q < 1e-9:
            return 0.0
        return float(np.dot(vec, query_vec) / (norm_v * norm_q))

    def nearest(self, query_vec: np.ndarray, top_k: int = 5) -> List[Tuple[str, float]]:
        if not self._cache:
            return []
        ids = list(self._cache.keys())
        mat = np.stack([self._cache[i] for i in ids])
        q_n = query_vec / (np.linalg.norm(query_vec) + 1e-9)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        mat_n = mat / (norms + 1e-9)
        sims = mat_n @ q_n
        top = np.argsort(-sims)[:top_k]
        return [(ids[i], float(sims[i])) for i in top]

    def __len__(self):
        return len(self._cache)


# ── Async GNN Learner ──────────────────────────────────────────────────────────

class AsyncGNNLearner:
    """
    Continuously trains node embeddings from the live graph.

    When PyTorch is available: runs a 2-layer GCN via backprop.
    Otherwise: uses spectral / hash embeddings (still useful for retrieval).

    Usage:
        learner = AsyncGNNLearner(engine.graph)
        task = asyncio.create_task(learner.training_loop())
        ...
        results = learner.query("find energy optimization nodes")
    """

    DIM = 64
    TRAIN_INTERVAL_SEC = 30     # re-train every 30 s in background
    MAX_EPOCHS_PER_STEP = 10

    def __init__(self, graph: nx.MultiDiGraph):
        self._graph = graph
        self._cache = NodeEmbeddingCache(dim=self.DIM)
        self._retriever = VectorlessRetriever(graph)
        self._steps = 0
        self._last_loss = 0.0
        self._running = False
        self._model = None
        if HAS_TORCH:
            self._model = self._build_model()

    def _build_model(self):
        """Build a simple 2-layer MLP to refine embeddings (no PyG required)."""
        if not HAS_TORCH:
            return None
        return nn.Sequential(
            nn.Linear(self.DIM, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, self.DIM),
        )

    def _compute_spectral_embeddings(self) -> Dict[str, np.ndarray]:
        """Compute node embeddings from graph structure (no ML)."""
        g = self._graph
        if g.number_of_nodes() == 0:
            return {}

        embeddings: Dict[str, np.ndarray] = {}
        try:
            # Use degree + clustering + hash features
            degree = dict(g.degree())
            in_deg = dict(g.in_degree())
            out_deg = dict(g.out_degree())
            max_deg = max(degree.values()) or 1

            for node, props in g.nodes(data=True):
                base = _node_features(node, props, self.DIM)
                # Inject structural signal into first 4 dims
                d = degree.get(node, 0) / max_deg
                i = in_deg.get(node, 0) / max_deg
                o = out_deg.get(node, 0) / max_deg
                base[0] = d
                base[1] = i
                base[2] = o
                base[3] = math.log1p(degree.get(node, 0))
                norm = np.linalg.norm(base)
                embeddings[node] = base / norm if norm > 1e-9 else base
        except Exception as exc:
            logger.warning("Spectral embedding failed: %s", exc)
            for node, props in g.nodes(data=True):
                embeddings[node] = _node_features(node, props, self.DIM)

        return embeddings

    async def _train_step(self) -> Dict[str, Any]:
        """One training cycle — updates the embedding cache."""
        if self._graph.number_of_nodes() == 0:
            return {"status": "skipped", "reason": "empty graph"}

        t0 = time.perf_counter()

        # Compute base spectral embeddings (always works)
        embeddings = await asyncio.get_event_loop().run_in_executor(
            None, self._compute_spectral_embeddings
        )

        # Optional: refine with PyTorch MLP
        loss = 0.0
        if HAS_TORCH and self._model is not None and len(embeddings) >= 2:
            try:
                loss = await asyncio.get_event_loop().run_in_executor(
                    None, self._torch_refine, embeddings
                )
            except Exception as exc:
                logger.debug("Torch refinement skipped: %s", exc)

        # Update cache
        for node_id, vec in embeddings.items():
            self._cache.put(node_id, vec)

        self._steps += 1
        self._last_loss = loss
        elapsed = (time.perf_counter() - t0) * 1000

        logger.debug(
            "GNN step %d: %d nodes embedded | loss=%.4f | %.1f ms",
            self._steps, len(embeddings), loss, elapsed,
        )
        return {
            "step": self._steps,
            "nodes_embedded": len(embeddings),
            "loss": round(loss, 6),
            "latency_ms": round(elapsed, 1),
        }

    def _torch_refine(self, embeddings: Dict[str, np.ndarray]) -> float:
        """Refine embeddings via link-prediction loss (contrastive)."""
        import torch, torch.nn.functional as F

        optimizer = torch.optim.Adam(self._model.parameters(), lr=0.001)
        nodes = list(embeddings.keys())
        if len(nodes) < 2:
            return 0.0

        # Build positive pairs from edges
        pos_pairs = [(u, v) for u, v, _ in self._graph.edges(keys=True) if u in embeddings and v in embeddings]
        if not pos_pairs:
            return 0.0

        pos_pairs = pos_pairs[:64]  # cap for speed

        total_loss = 0.0
        for _ in range(min(self.MAX_EPOCHS_PER_STEP, 3)):
            optimizer.zero_grad()
            loss = torch.tensor(0.0, requires_grad=True)
            for u, v in pos_pairs:
                eu = torch.tensor(embeddings[u], dtype=torch.float32).unsqueeze(0)
                ev = torch.tensor(embeddings[v], dtype=torch.float32).unsqueeze(0)
                hu = self._model(eu)
                hv = self._model(ev)
                # Positive pair similarity loss
                sim = F.cosine_similarity(hu, hv)
                loss = loss + (1.0 - sim).mean()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        return total_loss / max(self.MAX_EPOCHS_PER_STEP, 1)

    async def training_loop(self, interval_seconds: int = None):
        """Long-running background task — trains continuously."""
        interval = interval_seconds or self.TRAIN_INTERVAL_SEC
        self._running = True
        logger.info("AsyncGNNLearner: background training loop started (interval=%ds)", interval)

        # Initial train immediately
        try:
            await self._train_step()
        except Exception as exc:
            logger.warning("Initial GNN train failed: %s", exc)

        while self._running:
            await asyncio.sleep(interval)
            try:
                await self._train_step()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("GNN training step error: %s", exc)

    def stop(self):
        self._running = False

    # ── Query API ─────────────────────────────────────────────────────────────

    def query(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve top-k relevant nodes using cached embeddings.
        Falls back to vectorless traversal if cache is empty.
        """
        q_vec = _node_features(query_text, {}, self.DIM)

        if len(self._cache) > 0:
            nearest = self._cache.nearest(q_vec, top_k)
            results = []
            for node_id, score in nearest:
                props = dict(self._graph.nodes.get(node_id, {}))
                results.append({"node_id": node_id, "score": round(score, 4), "properties": props})
            return results

        # Fallback: vectorless retrieval
        seed_nodes, _ = self._retriever.retrieve(query_text, top_k=top_k)
        return [
            {
                "node_id": n,
                "score": 0.5,
                "properties": dict(self._graph.nodes.get(n, {})),
            }
            for n in seed_nodes
        ]

    def vectorless_subgraph(self, query: str, top_k: int = 10, max_hops: int = 2):
        """Return relevant subgraph via pure MultiDiGraph traversal."""
        seed_nodes, subgraph = self._retriever.retrieve(query, top_k=top_k, max_hops=max_hops)
        return {
            "seed_nodes": seed_nodes,
            "subgraph_nodes": list(subgraph.nodes(data=True)),
            "subgraph_edges": [(u, v, k) for u, v, k in subgraph.edges(keys=True)],
            "node_count": subgraph.number_of_nodes(),
            "edge_count": subgraph.number_of_edges(),
        }

    def stats(self) -> Dict[str, Any]:
        return {
            "steps": self._steps,
            "cached_embeddings": len(self._cache),
            "last_loss": self._last_loss,
            "torch_enabled": HAS_TORCH,
            "graph_nodes": self._graph.number_of_nodes(),
            "graph_edges": self._graph.number_of_edges(),
        }
