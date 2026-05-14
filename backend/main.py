"""
NextGenDB — Main Application Entry Point (v3)
Wires all layers together: persistence, query, vector, causal, memory, security, benchmarks.
"""
import asyncio
import logging
import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# ── Layer 10: Logging first ───────────────────────────────────────────────────
from .monitoring.tracing import setup_logging
setup_logging(
    level=os.getenv("LOG_LEVEL", "INFO"),
    json_output=os.getenv("LOG_FORMAT", "json") == "json",
)
logger = logging.getLogger("nextgendb.main")

# ── v1 layers (existing) ──────────────────────────────────────────────────────
from .api.routes          import router as core_router, init_v1, graph_model as legacy_model
from .api.ingest_routes   import router as ingest_router
from .distributed.health  import router as health_router, init_health_refs
from .distributed.registry import get_registry, registry_maintenance_loop
from .monitoring.dashboard import router as monitoring_router
from .streaming.producer  import get_producer
from .streaming.consumer  import KafkaEventConsumer
from .streaming.ingestion  import GraphIngestionHandler
from .gnn.learner         import AsyncGNNLearner
from .agent.neural_agent  import NeuralAgentPlanner
from .causal.flux         import TemporalFluxEngine

# ── v2 new layers ─────────────────────────────────────────────────────────────
from .storage.engine       import PersistentGraphEngine
from .query.lang           import QueryExecutor
from .vector.search        import VectorSearchEngine
from .causal.inference     import CausalInferenceEngine
from .agent.memory         import AgenticMemoryStore
from .security.auth        import AuthManager
from .api.v2_routes        import router as v2_router, init_v2

DATA_DIR = Path(os.getenv("DATA_DIR", "data"))

# ── FastAPI application ───────────────────────────────────────────────────────
app = FastAPI(
    title="NextGenDB Neural Intelligence Engine",
    description=(
        "Production-grade AI graph database with persistent WAL storage, MVCC ACID transactions, "
        "Cypher/SQL query engine, hybrid vector search, probabilistic causal inference, "
        "agentic memory, RBAC security, and LDBC-inspired benchmarking. 15 layers active."
    ),
    version="3.0.0",
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

# ── Mount routers ─────────────────────────────────────────────────────────────
app.include_router(core_router,       prefix="/api/v1")
app.include_router(ingest_router,     prefix="/api/v1")
app.include_router(health_router,     prefix="/api/v1")
app.include_router(monitoring_router, prefix="/api/v1")
app.include_router(v2_router,         prefix="/api/v2")

# ── Serve Frontend ────────────────────────────────────────────────────────────
# Try to serve from root-level frontend/dist if it exists
frontend_path = Path("frontend/dist")
if frontend_path.exists():
    app.mount("/", StaticFiles(directory=str(frontend_path), html=True), name="frontend")
    
    @app.exception_handler(404)
    async def fallback_to_index(request, exc):
        # If the request is not for /api, serve index.html
        if not request.url.path.startswith("/api"):
            return FileResponse(frontend_path / "index.html")
        return {"detail": "Not Found"}
else:
    logger.warning("Frontend dist directory not found. Static file serving disabled.")


# ── Lifespan ──────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def on_startup():
    logger.info("NextGenDB v3 starting — initialising all 15 layers...")

    # ── Layer 1: Persistent Storage Engine ────────────────────────────────────
    engine = PersistentGraphEngine(data_dir=DATA_DIR)

    # Seed demo data if graph is empty
    if engine.graph.number_of_nodes() == 0:
        logger.info("Empty graph detected — seeding demo data")
        _seed_demo(engine)

    # ── Layer 2: Query Engine ──────────────────────────────────────────────────
    qe = QueryExecutor(engine)

    # ── Layer 3: Vector Search ─────────────────────────────────────────────────
    vec = VectorSearchEngine(data_dir=DATA_DIR)
    # Index existing nodes
    for node_id, data in engine.graph.nodes(data=True):
        text = " ".join(str(v) for v in data.values())
        vec.index_node(node_id, text, {"node_id": node_id, "type": data.get("type", "")})

    # ── Layer 4: Causal Engine ─────────────────────────────────────────────────
    causal = CausalInferenceEngine()

    # ── Layer 5: Agent Memory ──────────────────────────────────────────────────
    memory = AgenticMemoryStore(engine=engine, persist_path=DATA_DIR / "agent_memory.json")

    # ── Layer 6: Neural GNN Learner + Vectorless Retriever ──────────────────────
    gnn = AsyncGNNLearner(engine.graph)
    asyncio.create_task(gnn.training_loop())

    # ── Layer 7: Temporal Flux Engine ──────────────────────────────────────────
    flux = TemporalFluxEngine()

    # ── Layer 8: Neural Agent (Decompose + Plan) ───────────────────────────────
    agent = NeuralAgentPlanner(
        engine=engine, 
        gnn_learner=gnn, 
        vec_engine=vec, 
        causal_engine=causal, 
        memory_store=memory
    )

    # ── Layer 9: Security / Auth ───────────────────────────────────────────────
    jwt_secret = os.getenv("JWT_SECRET", "nextgendb-dev-secret-change-in-prod")
    auth = AuthManager(jwt_secret=jwt_secret)
    # Create extra default users
    try:
        auth.create_user("developer", "dev123", "developer")
        auth.create_user("analyst",   "analyst", "analyst")
    except ValueError:
        pass   # already exists

    # Inject v2 singletons
    init_v2(engine, qe, vec, causal, memory, auth)
    
    # Inject v1 singletons (Sophisticated Neural Agent + Engines)
    from .graph.graph_model import GraphModel
    v1_model = GraphModel(engine=engine)
    init_v1(v1_model, agent, gnn, causal, flux)

    app.state.engine = engine
    app.state.vec    = vec
    app.state.memory = memory
    app.state.auth   = auth
    app.state.gnn    = gnn
    app.state.agent  = agent

    # ── Layers 10–12: Streaming + Distributed + Monitoring ─────────────────────
    registry = get_registry()
    host = os.getenv("POD_IP", "127.0.0.1")
    port = int(os.getenv("PORT", 8000))
    node = await registry.register(host, port, role="api")
    logger.info("Self-registered as node %s", node.node_id)

    asyncio.create_task(registry_maintenance_loop())

    producer = await get_producer()
    handler  = GraphIngestionHandler(v1_model)
    consumer = KafkaEventConsumer(on_event=handler.on_event)
    await consumer.start()

    app.state.consumer = consumer
    app.state.producer = producer
    app.state.ingestion = handler

    init_health_refs(v1_model, producer)

    # ── Layer 13: Background checkpoint ───────────────────────────────────────
    asyncio.create_task(engine.background_checkpoint_loop(interval_seconds=300))

    logger.info(
        "NextGenDB v3 LIVE — %d nodes, %d edges | 15 layers active",
        engine.graph.number_of_nodes(),
        engine.graph.number_of_edges(),
    )


@app.on_event("shutdown")
async def on_shutdown():
    logger.info("NextGenDB shutting down...")
    if hasattr(app.state, "consumer"):
        await app.state.consumer.stop()
    if hasattr(app.state, "producer"):
        await app.state.producer.stop()
    if hasattr(app.state, "engine"):
        app.state.engine.close()
    if hasattr(app.state, "vec"):
        app.state.vec.save()
    logger.info("Shutdown complete.")


def _seed_demo(engine: PersistentGraphEngine):
    """Seed a rich demo graph with diverse node types."""
    nodes = [
        ("Root_System",       {"label": "Core Intelligence", "type": "SYSTEM", "status": "ACTIVE", "score": 1.0}),
        ("Data_Ingest",       {"label": "Data Ingestion Pipeline", "type": "PROCESS", "load": 0.45, "score": 0.82}),
        ("Neural_Core",       {"label": "Neural Processing Unit", "type": "PROCESS", "efficiency": 0.98, "score": 0.95}),
        ("User_Query_Alpha",  {"label": "Query: Energy Optimization", "type": "EVENT", "timestamp": "2024-05-10T14:30:00Z", "score": 0.7}),
        ("Anomaly_Detector",  {"label": "Temporal Anomaly Detector", "type": "SERVICE", "version": "1.2.0", "score": 0.88}),
        ("Vector_Store",      {"label": "Embedding Index", "type": "SERVICE", "dim": 64, "score": 0.91}),
        ("Causal_Engine",     {"label": "Causal Inference Engine", "type": "SERVICE", "algo": "PC", "score": 0.89}),
        ("Memory_Store",      {"label": "Agent Memory Layer", "type": "SERVICE", "layers": 4, "score": 0.77}),
        ("Query_Planner",     {"label": "GNN-Assisted Query Planner", "type": "PROCESS", "version": "2.0", "score": 0.93}),
        ("Customer_A",        {"label": "Customer: Acme Corp", "type": "USER", "plan": "enterprise", "score": 0.6}),
        ("Product_X",         {"label": "Product: GraphDB Pro", "type": "PRODUCT", "price": 499.0, "score": 0.75}),
        ("Product_Y",         {"label": "Product: VectorSearch", "type": "PRODUCT", "price": 299.0, "score": 0.71}),
        ("Tx_001",            {"label": "Transaction: Q3 Purchase", "type": "TRANSACTION", "amount": 4990.0, "timestamp": "2024-07-15T09:00:00Z", "score": 0.5}),
        ("Churn_Event",       {"label": "Churn Signal Detected", "type": "EVENT", "severity": "HIGH", "timestamp": "2024-10-01T00:00:00Z", "score": 0.3}),
        ("Pricing_Change",    {"label": "Pricing Model Update", "type": "EVENT", "delta_pct": 15.0, "timestamp": "2024-09-01T00:00:00Z", "score": 0.4}),
    ]
    for nid, props in nodes:
        engine.add_node(nid, props)

    edges = [
        ("Root_System",    "Data_Ingest",      "MANAGES"),
        ("Root_System",    "Neural_Core",       "ORCHESTRATES"),
        ("Root_System",    "Vector_Store",      "ORCHESTRATES"),
        ("Root_System",    "Causal_Engine",     "ORCHESTRATES"),
        ("Root_System",    "Memory_Store",      "ORCHESTRATES"),
        ("Data_Ingest",    "Neural_Core",       "FEEDS"),
        ("Data_Ingest",    "Vector_Store",      "FEEDS"),
        ("User_Query_Alpha","Neural_Core",      "TRIGGERS"),
        ("User_Query_Alpha","Query_Planner",    "TRIGGERS"),
        ("Neural_Core",    "Anomaly_Detector",  "CONSULTS"),
        ("Neural_Core",    "Query_Planner",     "DELEGATES"),
        ("Anomaly_Detector","Root_System",      "ALERTS"),
        ("Causal_Engine",  "Anomaly_Detector",  "INFORMS"),
        ("Memory_Store",   "Neural_Core",       "PROVIDES_CONTEXT"),
        ("Customer_A",     "Product_X",         "BOUGHT", {"ts": "2024-07-15T09:00:00Z"}),
        ("Customer_A",     "Product_Y",         "BOUGHT", {"ts": "2024-08-01T12:00:00Z"}),
        ("Tx_001",         "Customer_A",        "BELONGS_TO"),
        ("Tx_001",         "Product_X",         "FOR"),
        ("Pricing_Change", "Product_X",         "AFFECTS"),
        ("Pricing_Change", "Churn_Event",       "CAUSES"),
        ("Churn_Event",    "Customer_A",        "AFFECTS"),
    ]
    for edge_args in edges:
        src, tgt, etype = edge_args[0], edge_args[1], edge_args[2]
        props = edge_args[3] if len(edge_args) > 3 else {}
        engine.add_edge(src, tgt, etype, props)

    logger.info("Demo data seeded: %d nodes, %d edges", engine.graph.number_of_nodes(), engine.graph.number_of_edges())


@app.get("/")
async def root():
    engine = getattr(app.state, "engine", None)
    stats  = engine.stats() if engine else {}
    return {
        "status":  "NextGenDB v3 — 15 Layers Active",
        "version": "3.0.0",
        "docs":    "/docs",
        "layers": {
            "L1":  "Persistent WAL Storage (ACID)",
            "L2":  "MVCC Transaction Manager",
            "L3":  "Cypher + SQL Query Engine",
            "L4":  "Property Index (O(1) lookup)",
            "L5":  "Hybrid Vector Search (BM25 + HNSW)",
            "L6":  "Probabilistic Causal Inference",
            "L7":  "Agentic Memory Store (4 layers)",
            "L8":  "RBAC/ABAC Security + JWT",
            "L9":  "LDBC Benchmark Suite",
            "L10": "GNN-Assisted Query Planner",
            "L11": "Streaming (Kafka / In-Process)",
            "L12": "Distributed Node Registry",
            "L13": "Prometheus + Grafana Observability",
            "L14": "Schema Registry + Migration",
            "L15": "OpenTelemetry Distributed Tracing",
        },
        "graph_stats": stats,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", 8000)),
        reload=os.getenv("DEV", "false") == "true",
        log_config=None,
    )
