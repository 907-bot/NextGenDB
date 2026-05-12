"""
Layer 8 — Streaming Ingest REST API
Allows external systems to push events directly via HTTP
(alternative to Kafka for simpler integrations).
"""
import logging
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Any, Dict, Optional

from ..streaming.ingestion import GraphIngestionHandler
from ..monitoring.metrics import record_stream_event

logger = logging.getLogger("nextgendb.api.ingest")
router = APIRouter(tags=["Streaming Ingest"])

# Shared handler — populated at startup via app.state
_handler: Optional[GraphIngestionHandler] = None


def set_handler(h: GraphIngestionHandler):
    global _handler
    _handler = h


class IngestEvent(BaseModel):
    event_id:   Optional[str] = None
    type:       str = "UPSERT_NODE"   # UPSERT_NODE | UPSERT_EDGE | DELETE_NODE
    node_id:    Optional[str] = None  # for node events
    source:     Optional[str] = None  # for edge events
    target:     Optional[str] = None  # for edge events
    edge_type:  Optional[str] = "RELATED"
    label:      Optional[str] = None
    properties: Dict[str, Any] = {}


@router.post("/ingest", status_code=202)
async def ingest_event(event: IngestEvent, background_tasks: BackgroundTasks):
    """
    Push a single graph mutation event.
    Accepted immediately; processed asynchronously.
    """
    if _handler is None:
        raise HTTPException(503, detail="Ingestion handler not initialised yet.")

    record_stream_event("http.ingest")
    background_tasks.add_task(_handler.on_event, event.model_dump())
    return {"accepted": True, "event_type": event.type}


@router.get("/ingest/stats")
async def ingest_stats():
    """Return ingestion processing statistics."""
    if _handler is None:
        return {"status": "not_initialised"}
    return _handler.stats
