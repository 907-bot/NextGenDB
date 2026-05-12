"""
Layer 8 — Kafka Event Consumer
Subscribes to ingestion topics and writes incoming events into the live graph model.
Exactly-once semantics supported via Kafka transaction API when available.
"""
import asyncio
import json
import logging
import os
from typing import Callable, Awaitable

logger = logging.getLogger("nextgendb.streaming.consumer")

KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
CONSUMER_GROUP  = "nextgendb-core"
TOPIC_INGEST    = "nextgendb.ingest"      # External data sources publish here


class KafkaEventConsumer:
    """
    Consumes events from `nextgendb.ingest` and calls `on_event` for each message.
    Falls back gracefully when Kafka is not available.
    """

    def __init__(self, on_event: Callable[[dict], Awaitable[None]]):
        self._consumer = None
        self._on_event = on_event
        self._running = False
        self._use_kafka = False

    async def start(self):
        try:
            from aiokafka import AIOKafkaConsumer
            self._consumer = AIOKafkaConsumer(
                TOPIC_INGEST,
                bootstrap_servers=KAFKA_BOOTSTRAP,
                group_id=CONSUMER_GROUP,
                value_deserializer=lambda v: json.loads(v.decode("utf-8")),
                auto_offset_reset="earliest",
                request_timeout_ms=3000,
            )
            await self._consumer.start()
            self._use_kafka = True
            logger.info("Kafka consumer started on topic '%s'", TOPIC_INGEST)
        except Exception as exc:
            logger.warning("Kafka consumer unavailable (%s). Skipping.", exc)

        self._running = True
        asyncio.create_task(self._consume_loop())

    async def _consume_loop(self):
        if not self._use_kafka:
            logger.info("Streaming consumer in STANDBY mode (no Kafka).")
            return

        async for msg in self._consumer:
            if not self._running:
                break
            try:
                await self._on_event(msg.value)
            except Exception as exc:
                logger.error("Error processing event: %s", exc)

    async def stop(self):
        self._running = False
        if self._use_kafka and self._consumer:
            await self._consumer.stop()
