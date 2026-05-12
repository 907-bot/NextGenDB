"""
Layer 9 — Health & Readiness Controller
Provides /health, /ready, and /live endpoints used by Kubernetes
liveness and readiness probes as well as load-balancers.
"""
import time
import asyncio
import logging
from fastapi import APIRouter
from pydantic import BaseModel
from typing import Dict, Any

logger = logging.getLogger("nextgendb.distributed.health")

router = APIRouter(tags=["Health"])

# Startup timestamp
_start_time = time.time()


class HealthStatus(BaseModel):
    status: str
    uptime_seconds: float
    components: Dict[str, str]
    version: str = "2.0.0"


def _check_graph(graph_model) -> str:
    try:
        n = len(graph_model.graph.nodes)
        return f"OK ({n} nodes)"
    except Exception as e:
        return f"DEGRADED: {e}"


def _check_streaming(producer) -> str:
    if producer is None:
        return "STANDBY (no Kafka)"
    return "OK (in-process queue)" if not producer._use_kafka else "OK (Kafka)"


# These will be injected by main.py at startup
_graph_model_ref = None
_producer_ref    = None


def init_health_refs(graph_model, producer):
    global _graph_model_ref, _producer_ref
    _graph_model_ref = graph_model
    _producer_ref    = producer


@router.get("/health", response_model=HealthStatus)
async def health():
    """Full health check — used by monitoring dashboards."""
    components = {
        "graph_engine":     _check_graph(_graph_model_ref) if _graph_model_ref else "UNKNOWN",
        "streaming_layer":  _check_streaming(_producer_ref),
        "gnn_trainer":      "OK",
        "reasoning_core":   "OK",
    }
    overall = "healthy" if all("OK" in v or "STANDBY" in v for v in components.values()) else "degraded"
    return HealthStatus(
        status=overall,
        uptime_seconds=round(time.time() - _start_time, 2),
        components=components,
    )


@router.get("/live")
async def liveness():
    """Kubernetes liveness probe — just confirm the process is alive."""
    return {"alive": True}


@router.get("/ready")
async def readiness():
    """Kubernetes readiness probe — confirm the system can serve traffic."""
    if _graph_model_ref is None:
        return {"ready": False, "reason": "Graph not initialized"}
    return {"ready": True}
