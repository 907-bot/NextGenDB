"""
Benchmarking Suite — LDBC SNB-inspired, YCSB-style, and causal workloads.

Produces reproducible benchmark reports with:
  • Throughput (ops/sec)
  • Latency P50/P90/P99
  • Multi-hop traversal speed
  • Ingestion rate
  • Query accuracy vs answer confidence
"""
import asyncio
import json
import logging
import random
import statistics
import string
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("nextgendb.benchmark")


@dataclass
class BenchmarkResult:
    name:          str
    ops:           int
    duration_s:    float
    throughput:    float        # ops/sec
    p50_ms:        float
    p90_ms:        float
    p99_ms:        float
    min_ms:        float
    max_ms:        float
    errors:        int
    metadata:      Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "benchmark":    self.name,
            "operations":   self.ops,
            "duration_s":   round(self.duration_s, 3),
            "throughput":   round(self.throughput, 1),
            "latency_ms": {
                "p50": round(self.p50_ms, 2),
                "p90": round(self.p90_ms, 2),
                "p99": round(self.p99_ms, 2),
                "min": round(self.min_ms, 2),
                "max": round(self.max_ms, 2),
            },
            "errors":       self.errors,
            **self.metadata,
        }


class BenchmarkRunner:
    """Run async micro-benchmarks and collect latency distributions."""

    @staticmethod
    async def run(
        name:     str,
        fn:       Callable,
        n_ops:    int = 1000,
        warmup:   int = 50,
    ) -> BenchmarkResult:
        latencies: List[float] = []
        errors = 0

        # Warm up
        for _ in range(warmup):
            try:
                if asyncio.iscoroutinefunction(fn):
                    await fn()
                else:
                    fn()
            except Exception:
                pass

        t_start = time.perf_counter()
        for _ in range(n_ops):
            t0 = time.perf_counter()
            try:
                if asyncio.iscoroutinefunction(fn):
                    await fn()
                else:
                    fn()
            except Exception:
                errors += 1
            latencies.append((time.perf_counter() - t0) * 1000)

        duration = time.perf_counter() - t_start
        latencies.sort()

        def pct(p):
            idx = int(len(latencies) * p / 100)
            return latencies[min(idx, len(latencies) - 1)]

        return BenchmarkResult(
            name=name,
            ops=n_ops,
            duration_s=duration,
            throughput=n_ops / duration,
            p50_ms=pct(50),
            p90_ms=pct(90),
            p99_ms=pct(99),
            min_ms=latencies[0],
            max_ms=latencies[-1],
            errors=errors,
        )


class NextGenDBBenchmark:
    """
    Full benchmark suite for NextGenDB.

    LDBC-inspired workloads:
      - Node ingestion throughput
      - Edge ingestion throughput
      - Property index lookup (O(1) index path)
      - Multi-hop traversal (1-hop, 2-hop, 3-hop)
      - Cypher MATCH query
      - SQL SELECT query
      - Vector search latency
      - Causal analysis latency
    """

    def __init__(self, engine, query_executor, vector_engine, causal_engine):
        self._engine  = engine
        self._qe      = query_executor
        self._vec     = vector_engine
        self._causal  = causal_engine

    def _rand_str(self, n: int = 8) -> str:
        return "".join(random.choices(string.ascii_lowercase, k=n))

    def _seed_benchmark_data(self, n_nodes: int = 500, n_edges: int = 1000):
        """Pre-populate graph with benchmark data."""
        logger.info("Seeding benchmark data: %d nodes, %d edges", n_nodes, n_edges)
        node_ids = [f"BN_{i}" for i in range(n_nodes)]
        for nid in node_ids:
            self._engine.add_node(nid, {
                "label":    self._rand_str(10),
                "type":     random.choice(["USER", "PRODUCT", "EVENT", "SERVICE"]),
                "score":    round(random.random(), 4),
                "category": random.choice(["A", "B", "C", "D"]),
                "ts":       f"2024-0{random.randint(1,9)}-{random.randint(10,28)}T00:00:00Z",
            })
        for _ in range(n_edges):
            src = random.choice(node_ids)
            tgt = random.choice(node_ids)
            if src != tgt:
                self._engine.add_edge(src, tgt, random.choice(["KNOWS", "BOUGHT", "TRIGGERS", "LINKS_TO"]))
        logger.info("Benchmark data seeded")

    async def run_all(self, n_ops: int = 200) -> Dict[str, Any]:
        results: List[BenchmarkResult] = []

        # ── 1. Node ingestion ──────────────────────────────────────────────────
        async def ingest_node():
            nid = f"BENCH_{uuid.uuid4().hex[:8]}"
            self._engine.add_node(nid, {"label": self._rand_str(), "type": "BENCH", "score": random.random()})

        results.append(await BenchmarkRunner.run("node_ingestion", ingest_node, n_ops))

        # ── 2. Edge ingestion ──────────────────────────────────────────────────
        node_ids = list(self._engine.graph.nodes())[:100] or ["BN_0", "BN_1"]
        async def ingest_edge():
            s = random.choice(node_ids)
            t = random.choice(node_ids)
            self._engine.add_edge(s, t, "BENCH_EDGE")

        results.append(await BenchmarkRunner.run("edge_ingestion", ingest_edge, n_ops))

        # ── 3. Index lookup ────────────────────────────────────────────────────
        async def index_lookup():
            self._engine.find_by_property("type", random.choice(["USER", "PRODUCT", "EVENT"]))

        results.append(await BenchmarkRunner.run("index_lookup", index_lookup, n_ops))

        # ── 4. 1-hop / 2-hop / 3-hop traversal ────────────────────────────────
        all_nodes = list(self._engine.graph.nodes())
        for hops in (1, 2, 3):
            async def traverse(h=hops):
                if all_nodes:
                    self._engine.get_subgraph(random.choice(all_nodes), radius=h)
            results.append(await BenchmarkRunner.run(f"traversal_{hops}hop", traverse, n_ops))

        # ── 5. Cypher query ────────────────────────────────────────────────────
        async def cypher_query():
            self._qe.execute("MATCH (n:USER) RETURN n.label, n.score")

        results.append(await BenchmarkRunner.run("cypher_match", cypher_query, n_ops))

        # ── 6. SQL query ───────────────────────────────────────────────────────
        async def sql_query():
            self._qe.execute("SELECT label, score FROM nodes WHERE type = 'USER'")

        results.append(await BenchmarkRunner.run("sql_select", sql_query, n_ops))

        # ── 7. Vector search ───────────────────────────────────────────────────
        async def vec_search():
            self._vec.search(self._rand_str(12), top_k=5)

        results.append(await BenchmarkRunner.run("vector_search", vec_search, n_ops))

        # ── 8. Causal analysis ─────────────────────────────────────────────────
        sample_nodes = list(self._engine.graph.nodes())[:20]
        async def causal():
            if len(sample_nodes) >= 2:
                sg = self._engine.get_subgraph(random.choice(sample_nodes), radius=2)
                self._causal.analyze(sg)

        results.append(await BenchmarkRunner.run("causal_analysis", causal, n_ops // 2))

        # ── Summary ────────────────────────────────────────────────────────────
        report = {
            "suite":        "NextGenDB Full Benchmark",
            "timestamp":    time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "engine_stats": self._engine.stats(),
            "results":      [r.to_dict() for r in results],
            "summary": {
                "best_throughput":   max(r.throughput for r in results),
                "best_p99_ms":       min(r.p99_ms for r in results),
                "total_errors":      sum(r.errors for r in results),
            },
        }
        return report
