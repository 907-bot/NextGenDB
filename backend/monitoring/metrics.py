"""
Layer 10 — Prometheus Metrics Collector
Exposes a /metrics endpoint compatible with Prometheus scraping.
Tracks: query latency, graph size, GNN training steps, streaming throughput.
Falls back silently if prometheus_client is not installed.
"""
import time
import logging
from functools import wraps
from contextlib import asynccontextmanager
from typing import Callable

logger = logging.getLogger("nextgendb.monitoring.metrics")

# ── Attempt to import prometheus_client ─────────────────────────────────────
try:
    from prometheus_client import (
        Counter, Histogram, Gauge,
        generate_latest, CONTENT_TYPE_LATEST,
    )
    _PROM_AVAILABLE = True
    logger.info("Prometheus metrics: ENABLED")
except ImportError:
    _PROM_AVAILABLE = False
    logger.warning("prometheus_client not installed — metrics will be in-process only.")

try:
    import psutil
    _PSUTIL_AVAILABLE = True
except ImportError:
    _PSUTIL_AVAILABLE = False


# ── Metric definitions ───────────────────────────────────────────────────────
if _PROM_AVAILABLE:
    QUERY_TOTAL = Counter(
        "nextgendb_queries_total",
        "Total number of queries processed",
        ["status"],  # labels: success / error
    )
    QUERY_LATENCY = Histogram(
        "nextgendb_query_latency_seconds",
        "End-to-end query processing latency",
        buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
    )
    GRAPH_NODES = Gauge("nextgendb_graph_nodes", "Current number of nodes in the graph")
    GRAPH_EDGES = Gauge("nextgendb_graph_edges", "Current number of edges in the graph")
    GNN_LOSS    = Gauge("nextgendb_gnn_loss", "Latest GNN training loss")
    GNN_STEPS   = Counter("nextgendb_gnn_steps_total", "Total GNN training steps completed")
    MEMORY_USAGE = Gauge("nextgendb_memory_usage_bytes", "Current memory usage of the NextGenDB process")
    STREAM_EVENTS = Counter(
        "nextgendb_stream_events_total",
        "Total streaming events ingested",
        ["topic"],
    )
    CONFIDENCE_HIST = Histogram(
        "nextgendb_answer_confidence",
        "Distribution of answer confidence scores",
        buckets=[0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99, 1.0],
    )


# ── In-process fallback store ────────────────────────────────────────────────
class _InProcessMetrics:
    def __init__(self):
        self.query_total   = {"success": 0, "error": 0}
        self.latencies     = []
        self.graph_nodes   = 0
        self.graph_edges   = 0
        self.gnn_loss      = 0.0
        self.gnn_steps     = 0
        self.stream_events = {}
        self.confidences   = []

    def snapshot(self) -> dict:
        avg_lat = sum(self.latencies[-100:]) / max(len(self.latencies[-100:]), 1)
        mem_mb = 0.0
        if _PSUTIL_AVAILABLE:
            import os
            process = psutil.Process(os.getpid())
            mem_mb = process.memory_info().rss / (1024 * 1024)
            if _PROM_AVAILABLE:
                MEMORY_USAGE.set(process.memory_info().rss)
                
        return {
            "query_total":     self.query_total,
            "avg_latency_ms":  round(avg_lat * 1000, 2),
            "graph_nodes":     self.graph_nodes,
            "graph_edges":     self.graph_edges,
            "gnn_loss":        self.gnn_loss,
            "gnn_steps":       self.gnn_steps,
            "stream_events":   self.stream_events,
            "avg_confidence":  round(sum(self.confidences[-100:]) / max(len(self.confidences[-100:]), 1), 4),
            "memory_mb":       round(mem_mb, 2),
        }


_inproc = _InProcessMetrics()


# ── Public API ───────────────────────────────────────────────────────────────

def record_query(status: str, latency: float, confidence: float = 0.0):
    _inproc.query_total[status] = _inproc.query_total.get(status, 0) + 1
    _inproc.latencies.append(latency)
    if confidence:
        _inproc.confidences.append(confidence)
    if _PROM_AVAILABLE:
        QUERY_TOTAL.labels(status=status).inc()
        QUERY_LATENCY.observe(latency)
        if confidence:
            CONFIDENCE_HIST.observe(confidence)


def update_graph_metrics(nodes: int, edges: int):
    _inproc.graph_nodes = nodes
    _inproc.graph_edges = edges
    if _PROM_AVAILABLE:
        GRAPH_NODES.set(nodes)
        GRAPH_EDGES.set(edges)


def record_gnn_step(loss: float):
    _inproc.gnn_loss = loss
    _inproc.gnn_steps += 1
    if _PROM_AVAILABLE:
        GNN_LOSS.set(loss)
        GNN_STEPS.inc()


def record_stream_event(topic: str):
    _inproc.stream_events[topic] = _inproc.stream_events.get(topic, 0) + 1
    if _PROM_AVAILABLE:
        STREAM_EVENTS.labels(topic=topic).inc()


def get_metrics_snapshot() -> dict:
    return _inproc.snapshot()


def get_prometheus_output():
    """Returns raw Prometheus text format, or None if unavailable."""
    if _PROM_AVAILABLE:
        if _PSUTIL_AVAILABLE:
            import os
            process = psutil.Process(os.getpid())
            MEMORY_USAGE.set(process.memory_info().rss)
        return generate_latest(), CONTENT_TYPE_LATEST
    return None, None


# ── Timing decorator ─────────────────────────────────────────────────────────
def timed_query(func: Callable):
    """Async decorator that automatically records query latency and status."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        status = "success"
        confidence = 0.0
        try:
            result = await func(*args, **kwargs)
            confidence = getattr(result, "confidence", 0.0)
            return result
        except Exception:
            status = "error"
            raise
        finally:
            latency = time.perf_counter() - t0
            record_query(status, latency, confidence)
    return wrapper
