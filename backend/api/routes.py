import time
import asyncio
import logging
from fastapi import APIRouter, BackgroundTasks, Depends, Request
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

from ..agent.neural_agent import NeuralAgentPlanner
from ..gnn.learner import AsyncGNNLearner
from ..causal.inference import CausalInferenceEngine
from ..causal.flux import TemporalFluxEngine
from ..vector.search import VectorSearchEngine
from ..agent.memory import AgenticMemoryStore
from ..graph.graph_model import GraphModel

# Layer 10: metrics + tracing
from ..monitoring.metrics import (
    record_query, update_graph_metrics, record_gnn_step
)
from ..monitoring.tracing import trace_query_pipeline, trace_gnn_step

logger = logging.getLogger("nextgendb.api.v1")
router = APIRouter()

# ── Singletons (will be injected/initialized) ────────────────────────────────
graph_model   : Optional[GraphModel]           = None
neural_agent  : Optional[NeuralAgentPlanner]  = None
gnn_learner   : Optional[AsyncGNNLearner]      = None
causal_engine : Optional[CausalInferenceEngine] = None
flux_engine   : Optional[TemporalFluxEngine]   = None

def init_v1(model, agent, gnn, causal, flux):
    global graph_model, neural_agent, gnn_learner, causal_engine, flux_engine
    graph_model   = model
    neural_agent  = agent
    gnn_learner   = gnn
    causal_engine = causal
    flux_engine   = flux

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer:         str
    confidence:     float
    steps:          List[Dict[str, Any]]
    graph_snapshot: Dict[str, Any]
    timeline:       List[Dict[str, Any]]
    signals:        Optional[Dict[str, Any]] = None

@router.post("/query", response_model=QueryResponse)
async def handle_query(request: QueryRequest, background_tasks: BackgroundTasks):
    t0 = time.perf_counter()
    status = "success"
    
    if not neural_agent:
        logger.error("Neural Agent not initialized")
        return QueryResponse(
            answer="System error: Neural Agent not initialized",
            confidence=0.0,
            steps=[],
            graph_snapshot={"nodes": [], "links": []},
            timeline=[]
        )

    with trace_query_pipeline(request.query, 6):
        try:
            # 1. Neural Agent Orchestration (Two-stage Decompose + Plan)
            result = await neural_agent.run(request.query)
            
            confidence = result.get("confidence", 0.0)
            answer     = result.get("answer", "No answer found.")
            
            # Extract timeline if available from context
            timeline = []
            for step in result.get("plan_steps", []):
                if step["action"] == "TEMPORAL_ORDER" and step["done"]:
                    # We'd ideally pull the actual result here, but for simplicity:
                    pass
            
            # Use flux engine for active signals
            signals = flux_engine.get_temporal_signals(graph_model.graph) if flux_engine else None

        except Exception as exc:
            logger.error("Query pipeline failed: %s", exc)
            status = "error"
            raise exc
        finally:
            latency = time.perf_counter() - t0
            record_query(status, latency, confidence if status == "success" else 0.0)
            if graph_model:
                update_graph_metrics(
                    len(graph_model.graph.nodes),
                    len(graph_model.graph.edges),
                )

    return QueryResponse(
        answer=answer,
        confidence=confidence,
        steps=result.get("plan_steps", []),
        graph_snapshot=graph_model.to_json() if graph_model else {"nodes": [], "links": []},
        timeline=timeline,
        signals=signals
    )

@router.get("/graph")
async def get_graph():
    return graph_model.to_json() if graph_model else {"nodes": [], "links": []}

