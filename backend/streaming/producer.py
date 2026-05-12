"""
Layer 8 — Kafka Event Producer
Publishes query events and graph mutations to Kafka topics.
Falls back gracefully to an in-process async queue if Kafka is not available.
"""
import asyncio
import json
import os
import logging
from datetime import datetime, timezone
from typing import Any, Dict

logger = logging.getLogger("nextgendb.streaming.producer")

KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
TOPIC_QUERIES   = "nextgendb.queries"
TOPIC_GRAPH     = "nextgendb.graph_events"
TOPIC_GNN       = "nextgendb.gnn_updates"


class InProcessQueue:
    """Lightweight fallback that mimics the Kafka producer interface."""
    def __init__(self):
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=10_000)

    async def send(self, topic: str, value: Dict[str, Any]):
        await self._queue.put({"topic": topic, "value": value})
        logger.debug("InProcess queue: %s -> %s", topic, value)

    async def flush(self):
        pass

    async def stop(self):
        pass


class KafkaEventProducer:
    """
    Wraps aiokafka AIOKafkaProducer.
    Falls back to InProcessQueue when Kafka is unreachable.
    """

    def __init__(self):
        self._producer = None
        self._fallback: InProcessQueue = InProcessQueue()
        self._use_kafka = False

    async def start(self):
        try:
            from aiokafka import AIOKafkaProducer
            self._producer = AIOKafkaProducer(
                bootstrap_servers=KAFKA_BOOTSTRAP,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                request_timeout_ms=3000,
            )
            await self._producer.start()
            self._use_kafka = True
            logger.info("Kafka producer connected to %s", KAFKA_BOOTSTRAP)
        except Exception as exc:
            logger.warning(
                "Kafka unavailable (%s). Using in-process queue fallback.", exc
            )
            self._use_kafka = False

    async def send(self, topic: str, value: Dict[str, Any]):
        value["_ts"] = datetime.now(timezone.utc).isoformat()
        if self._use_kafka and self._producer:
            try:
                await self._producer.send_and_wait(topic, value)
                return
            except Exception as exc:
                logger.error("Kafka send failed: %s — falling back.", exc)
        await self._fallback.send(topic, value)

    async def send_query_event(self, query: str, plan_steps: int):
        await self.send(TOPIC_QUERIES, {"query": query, "plan_steps": plan_steps})

    async def send_graph_event(self, event_type: str, node_id: str, data: Dict[str, Any]):
        await self.send(TOPIC_GRAPH, {"event": event_type, "node": node_id, "data": data})

    async def send_gnn_update(self, loss: float, accuracy: float):
        await self.send(TOPIC_GNN, {"loss": loss, "accuracy": accuracy})

    async def stop(self):
        if self._use_kafka and self._producer:
            await self._producer.stop()


# Singleton
_producer: KafkaEventProducer | None = None


async def get_producer() -> KafkaEventProducer:
    global _producer
    if _producer is None:
        _producer = KafkaEventProducer()
        await _producer.start()
    return _producer
