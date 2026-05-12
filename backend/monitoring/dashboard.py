"""
Layer 10 — Grafana-style Observability Dashboard API
Exposes /metrics (Prometheus) and /dashboard endpoints
for the frontend monitoring panel.
"""
import logging
from fastapi import APIRouter
from fastapi.responses import PlainTextResponse
from typing import Any, Dict

from ..monitoring.metrics import get_metrics_snapshot, get_prometheus_output
from ..distributed.registry import get_registry
from ..distributed.health import router as health_router

logger = logging.getLogger("nextgendb.monitoring.dashboard")

router = APIRouter(tags=["Observability"])


@router.get("/metrics/raw", response_class=PlainTextResponse)
async def prometheus_metrics():
    """Raw Prometheus text format — scraped by Prometheus server."""
    output, content_type = get_prometheus_output()
    if output:
        return PlainTextResponse(output, media_type=content_type)
    return PlainTextResponse("# prometheus_client not installed\n", media_type="text/plain")


@router.get("/metrics/dashboard")
async def dashboard_metrics() -> Dict[str, Any]:
    """
    JSON snapshot of all metrics — consumed by the React
    Monitoring Panel in the frontend dashboard.
    """
    snapshot = get_metrics_snapshot()
    nodes    = get_registry().snapshot()
    return {
        "metrics":      snapshot,
        "cluster_nodes": nodes,
        "layer_status": {
            "layer_1_api":        "ACTIVE",
            "layer_2_agent":      "ACTIVE",
            "layer_3_rag":        "ACTIVE",
            "layer_4_reasoning":  "ACTIVE",
            "layer_5_graph":      "ACTIVE",
            "layer_6_gnn":        "ACTIVE",
            "layer_7_storage":    "ACTIVE (in-memory)",
            "layer_8_streaming":  "ACTIVE (queue)" if snapshot["stream_events"] == {} else "ACTIVE",
            "layer_9_distributed": f"ACTIVE ({len(nodes)} nodes)",
            "layer_10_monitoring": "ACTIVE",
        }
    }


@router.get("/nodes")
async def cluster_nodes():
    """List all registered distributed nodes."""
    return {"nodes": get_registry().snapshot()}
