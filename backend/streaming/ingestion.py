"""
Layer 8 — Graph Ingestion Handler
Processes raw streaming events and writes them into the live GraphModel.
Supports windowing, deduplication, and event-time processing.
"""
import logging
from typing import Any, Dict
from ..graph.graph_model import GraphModel

logger = logging.getLogger("nextgendb.streaming.ingestion")


class GraphIngestionHandler:
    """
    Receives a decoded Kafka event (or any dict) and applies it to
    the live in-memory graph.  Supports three event types:
      - UPSERT_NODE
      - UPSERT_EDGE
      - DELETE_NODE
    """

    def __init__(self, graph: GraphModel):
        self._graph = graph
        self._processed = 0
        self._errors = 0
        self._seen_ids: set[str] = set()  # simple deduplication window

    async def on_event(self, event: Dict[str, Any]):
        event_id = event.get("event_id")
        if event_id and event_id in self._seen_ids:
            logger.debug("Duplicate event skipped: %s", event_id)
            return
        if event_id:
            self._seen_ids.add(event_id)
            # Keep the dedup window bounded at 50k entries
            if len(self._seen_ids) > 50_000:
                self._seen_ids.clear()

        try:
            await self._dispatch(event)
            self._processed += 1
        except Exception as exc:
            self._errors += 1
            logger.error("Ingestion error for event %s: %s", event_id, exc)

    async def _dispatch(self, event: Dict[str, Any]):
        etype = event.get("type", "UPSERT_NODE").upper()

        if etype == "UPSERT_NODE":
            node_id   = event["node_id"]
            props     = event.get("properties", {})
            props["label"] = event.get("label", node_id)
            self._graph.add_node(node_id, props)
            logger.info("Graph: UPSERT_NODE '%s'", node_id)

        elif etype == "UPSERT_EDGE":
            src       = event["source"]
            tgt       = event["target"]
            edge_type = event.get("edge_type", "RELATED")
            props     = event.get("properties", {})
            # Auto-create missing nodes
            for n in (src, tgt):
                if n not in self._graph.graph:
                    self._graph.add_node(n, {"label": n, "type": "INFERRED"})
            self._graph.add_edge(src, tgt, edge_type, props)
            logger.info("Graph: UPSERT_EDGE '%s' -[%s]-> '%s'", src, edge_type, tgt)

        elif etype == "DELETE_NODE":
            node_id = event["node_id"]
            if node_id in self._graph.graph:
                self._graph.graph.remove_node(node_id)
                logger.info("Graph: DELETE_NODE '%s'", node_id)

        else:
            logger.warning("Unknown event type '%s' — ignored.", etype)

    @property
    def stats(self) -> Dict[str, int]:
        return {
            "processed": self._processed,
            "errors":    self._errors,
            "graph_nodes": len(self._graph.graph.nodes),
            "graph_edges": len(self._graph.graph.edges),
        }
