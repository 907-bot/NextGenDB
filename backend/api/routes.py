import time
import asyncio
from fastapi import APIRouter, BackgroundTasks
from pydantic import BaseModel
from typing import List, Dict, Any

from ..agent.planner import Planner
from ..agent.executor import Executor
from ..graph.graph_model import GraphModel
from ..graph.reasoning import CausalEngine, TemporalEngine
from ..gnn.model import GNNTrainer

# Layer 10: metrics + tracing
from ..monitoring.metrics import (
    record_query, update_graph_metrics, record_gnn_step
)
from ..monitoring.tracing import trace_query_pipeline, trace_gnn_step

router = APIRouter()
planner       = Planner()
graph_model   = GraphModel()
graph_model.seed_data()
causal_engine  = CausalEngine()
temporal_engine = TemporalEngine()
gnn_trainer    = GNNTrainer()
executor       = Executor(graph_model, None, (causal_engine, temporal_engine))


class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer:         str
    confidence:     float
    steps:          List[Dict[str, Any]]
    graph_snapshot: Dict[str, Any]
    timeline:       List[Dict[str, Any]]


async def _gnn_background(graph, trainer):
    """Run GNN training step and record metrics — called in background."""
    result = await trainer.train_step_async(graph)
    loss = result.get("loss", 0.0) if isinstance(result, dict) else 0.0
    with trace_gnn_step(len(graph.nodes)):
        record_gnn_step(loss)


@router.post("/query", response_model=QueryResponse)
async def handle_query(request: QueryRequest, background_tasks: BackgroundTasks):
    t0 = time.perf_counter()
    status = "success"

    with trace_query_pipeline(request.query, 4):
        try:
            # 1. Agent Planning
            plan = await planner.generate_plan(request.query)

            # 2. Execution
            execution_context = await executor.run_plan(plan)

            # 3. GNN Update — in background so it never blocks the response
            background_tasks.add_task(_gnn_background, graph_model.graph, gnn_trainer)

            confidence = execution_context.get("SYNTHESIZE", {}).get("confidence", 0.0)
            answer     = execution_context.get("SYNTHESIZE", {}).get("answer", "No answer found.")
            timeline   = execution_context.get("TEMPORAL_REASONING", {}).get("timeline", [])

        except Exception as exc:
            status = "error"
            raise exc
        finally:
            latency = time.perf_counter() - t0
            record_query(status, latency, confidence if status == "success" else 0.0)
            update_graph_metrics(
                len(graph_model.graph.nodes),
                len(graph_model.graph.edges),
            )

    return QueryResponse(
        answer=answer,
        confidence=confidence,
        steps=[step.dict() for step in plan],
        graph_snapshot=graph_model.to_json(),
        timeline=timeline,
    )


@router.get("/graph")
async def get_graph():
    return graph_model.to_json()
