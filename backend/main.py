"""
NextGenDB — Main Application Entry Point
Wires all 10 layers together on startup.
"""
import asyncio
import logging
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# ── Layer 10: Logging must be configured first ───────────────────────────────
from .monitoring.tracing import setup_logging
setup_logging(
    level=os.getenv("LOG_LEVEL", "INFO"),
    json_output=os.getenv("LOG_FORMAT", "json") == "json",
)
logger = logging.getLogger("nextgendb.main")

# ── Core layers ──────────────────────────────────────────────────────────────
from .api.routes import router as core_router
from .api.ingest_routes import router as ingest_router

# ── Layer 9: Health + Registry ───────────────────────────────────────────────
from .distributed.health import router as health_router, init_health_refs
from .distributed.registry import get_registry, registry_maintenance_loop

# ── Layer 10: Monitoring dashboard ───────────────────────────────────────────
from .monitoring.dashboard import router as monitoring_router

# ── Layer 8: Streaming ───────────────────────────────────────────────────────
from .streaming.producer import get_producer
from .streaming.consumer import KafkaEventConsumer
from .streaming.ingestion import GraphIngestionHandler

# ── Shared graph model (imported here so all layers share one instance) ───────
from .api.routes import graph_model

# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="NextGenDB Neural Intelligence Engine",
    description="10-layer AI graph database engine with streaming, distributed runtime, and full observability.",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Mount routers ────────────────────────────────────────────────────────────
app.include_router(core_router,       prefix="/api/v1")
app.include_router(ingest_router,     prefix="/api/v1")
app.include_router(health_router,     prefix="/api/v1")
app.include_router(monitoring_router, prefix="/api/v1")


# ── Lifespan: startup / shutdown ─────────────────────────────────────────────
@app.on_event("startup")
async def on_startup():
    logger.info("NextGenDB starting up — initialising all 10 layers...")

    # Layer 9: register this node
    registry = get_registry()
    host = os.getenv("POD_IP", "127.0.0.1")
    port = int(os.getenv("PORT", 8000))
    node = await registry.register(host, port, role="api")
    logger.info("Self-registered as node %s", node.node_id)

    # Layer 9: background pruning loop
    asyncio.create_task(registry_maintenance_loop())

    # Layer 8: start Kafka producer
    producer = await get_producer()

    # Layer 8: start Kafka consumer + ingestion handler
    handler  = GraphIngestionHandler(graph_model)
    consumer = KafkaEventConsumer(on_event=handler.on_event)
    await consumer.start()
    app.state.consumer = consumer
    app.state.producer  = producer
    app.state.ingestion = handler

    # Layer 9/10: inject references for health checks
    init_health_refs(graph_model, producer)

    logger.info("All 10 layers initialised. NextGenDB is live.")


@app.on_event("shutdown")
async def on_shutdown():
    logger.info("NextGenDB shutting down...")
    if hasattr(app.state, "consumer"):
        await app.state.consumer.stop()
    if hasattr(app.state, "producer"):
        await app.state.producer.stop()
    logger.info("Shutdown complete.")


@app.get("/")
async def root():
    return {
        "status":  "NextGenDB Neural Engine — All 10 Layers Active",
        "version": "2.0.0",
        "docs":    "/docs",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", 8000)),
        reload=os.getenv("DEV", "false") == "true",
        log_config=None,  # we manage logging ourselves
    )
