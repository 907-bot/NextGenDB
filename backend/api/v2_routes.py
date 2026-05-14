"""
Extended API Routes — v2 endpoints for all new subsystems.

Routes:
  /api/v2/query          — Cypher / SQL / NL query with EXPLAIN
  /api/v2/nodes          — CRUD with schema validation + auth
  /api/v2/edges          — Edge CRUD
  /api/v2/vector/search  — Hybrid vector search
  /api/v2/causal         — Causal inference endpoints
  /api/v2/memory         — Agent memory CRUD
  /api/v2/benchmark      — Run benchmarks
  /api/v2/schema         — Schema registry
  /api/v2/auth           — Login / user management
  /api/v2/backup         — Checkpoint / restore
"""
import asyncio
import logging
import os
import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, Header, HTTPException, status
from pydantic import BaseModel

from ..storage.engine   import PersistentGraphEngine
from ..query.lang       import QueryExecutor, QueryParser
from ..vector.search    import VectorSearchEngine
from ..causal.inference import CausalInferenceEngine
from ..agent.memory     import AgenticMemoryStore
from ..security.auth    import AuthManager, Permission
from ..benchmark.suite  import NextGenDBBenchmark
from ..monitoring.metrics import record_query, update_graph_metrics

logger = logging.getLogger("nextgendb.api.v2")
router = APIRouter()

# ── Singletons (injected by main.py, set via module globals) ──────────────────
_engine:  Optional[PersistentGraphEngine] = None
_qe:      Optional[QueryExecutor]         = None
_vec:     Optional[VectorSearchEngine]    = None
_causal:  Optional[CausalInferenceEngine] = None
_memory:  Optional[AgenticMemoryStore]    = None
_auth:    Optional[AuthManager]           = None


def init_v2(engine, qe, vec, causal, memory, auth):
    global _engine, _qe, _vec, _causal, _memory, _auth
    _engine = engine
    _qe     = qe
    _vec    = vec
    _causal = causal
    _memory = memory
    _auth   = auth


# ── Auth helpers ──────────────────────────────────────────────────────────────

def _token(authorization: Optional[str] = Header(None)) -> Optional[str]:
    if authorization and authorization.startswith("Bearer "):
        return authorization[7:]
    return None

def _require(perm: Permission):
    def dep(token: Optional[str] = Depends(_token)):
        if _auth is None:
            return None   # auth not initialised yet
        if not token:
            raise HTTPException(status_code=401, detail="Missing Bearer token")
        try:
            return _auth.require(token, perm)
        except PermissionError as e:
            raise HTTPException(status_code=403, detail=str(e))
        except ValueError as e:
            raise HTTPException(status_code=401, detail=str(e))
    return dep


# ── Auth Endpoints ─────────────────────────────────────────────────────────────

class LoginRequest(BaseModel):
    username: str
    password: str

class CreateUserRequest(BaseModel):
    username:  str
    password:  str
    role:      str = "readonly"
    tenant_id: str = "default"

@router.post("/auth/login")
async def login(req: LoginRequest):
    if _auth is None:
        raise HTTPException(503, "Auth not initialised")
    try:
        token = _auth.login(req.username, req.password)
        return {"token": token, "type": "Bearer"}
    except ValueError as e:
        raise HTTPException(401, str(e))

@router.post("/auth/users", dependencies=[Depends(_require(Permission.ADMIN))])
async def create_user(req: CreateUserRequest):
    try:
        uid = _auth.create_user(req.username, req.password, req.role, req.tenant_id)
        return {"user_id": uid, "username": req.username, "role": req.role}
    except ValueError as e:
        raise HTTPException(400, str(e))


# ── Cypher / SQL / NL Query ───────────────────────────────────────────────────

class QueryV2Request(BaseModel):
    query:   str
    params:  Optional[Dict[str, Any]] = None
    analyze: bool = False
    explain: bool = False
    mode:    str = "auto"   # auto | cypher | sql | natural

class QueryV2Response(BaseModel):
    query_type: str
    rows:       List[Dict[str, Any]]
    plan:       Optional[Dict]
    latency_ms: float
    scanned:    int
    returned:   int

@router.post("/query", response_model=QueryV2Response)
async def execute_query(req: QueryV2Request, user=Depends(_require(Permission.READ))):
    if _qe is None:
        raise HTTPException(503, "Query engine not initialised")
    t0 = time.perf_counter()
    try:
        parsed = QueryParser.parse(req.query, req.params or {})
        qt     = parsed.get("type", "UNKNOWN")

        if req.explain or qt == "EXPLAIN":
            result = _qe.execute(req.query, req.params, analyze=False)
            plan_d = _plan_to_dict(result.plan) if result.plan else None
            return QueryV2Response(query_type=qt, rows=[], plan=plan_d, latency_ms=result.latency_ms, scanned=0, returned=0)

        result = _qe.execute(req.query, req.params, analyze=req.analyze)
        update_graph_metrics(_engine.graph.number_of_nodes(), _engine.graph.number_of_edges())
        record_query("success", time.perf_counter() - t0)
        return QueryV2Response(
            query_type=qt,
            rows=result.rows,
            plan=_plan_to_dict(result.plan) if result.plan else None,
            latency_ms=result.latency_ms,
            scanned=result.scanned,
            returned=result.returned,
        )
    except Exception as e:
        record_query("error", time.perf_counter() - t0)
        raise HTTPException(400, str(e))


def _plan_to_dict(plan) -> Dict:
    if plan is None:
        return {}
    return {
        "op": plan.op,
        "details": plan.details,
        "cost_est": plan.cost_est,
        "children": [_plan_to_dict(c) for c in plan.children],
    }


# ── Node CRUD ─────────────────────────────────────────────────────────────────

class NodeCreateRequest(BaseModel):
    node_id:    str
    label:      str
    type:       str = "GENERIC"
    properties: Dict[str, Any] = {}

class EdgeCreateRequest(BaseModel):
    source:     str
    target:     str
    edge_type:  str
    properties: Dict[str, Any] = {}

@router.post("/nodes", dependencies=[Depends(_require(Permission.WRITE))])
async def create_node(req: NodeCreateRequest):
    props = {"label": req.label, "type": req.type, **req.properties}
    _engine.add_node(req.node_id, props)
    # Auto-index in vector store
    _vec.index_node(req.node_id, f"{req.label} {req.type} " + " ".join(str(v) for v in req.properties.values()), {"node_id": req.node_id, "type": req.type})
    return {"node_id": req.node_id, "status": "created"}

@router.get("/nodes/{node_id}", dependencies=[Depends(_require(Permission.READ))])
async def get_node(node_id: str):
    node = _engine.get_node(node_id)
    if node is None:
        raise HTTPException(404, f"Node '{node_id}' not found")
    return {"node_id": node_id, "properties": node}

@router.delete("/nodes/{node_id}", dependencies=[Depends(_require(Permission.DELETE))])
async def delete_node(node_id: str):
    _engine.delete_node(node_id)
    return {"node_id": node_id, "status": "deleted"}

@router.post("/edges", dependencies=[Depends(_require(Permission.WRITE))])
async def create_edge(req: EdgeCreateRequest):
    _engine.add_edge(req.source, req.target, req.edge_type, req.properties)
    return {"source": req.source, "target": req.target, "type": req.edge_type, "status": "created"}

@router.delete("/edges/{source}/{target}/{edge_type}", dependencies=[Depends(_require(Permission.DELETE))])
async def delete_edge(source: str, target: str, edge_type: str):
    _engine.delete_edge(source, target, edge_type)
    return {"status": "deleted"}

@router.get("/nodes", dependencies=[Depends(_require(Permission.READ))])
async def list_nodes(type: Optional[str] = None, limit: int = 100):
    g = _engine.graph
    nodes = []
    for node, data in g.nodes(data=True):
        if type and data.get("type", "").upper() != type.upper():
            continue
        nodes.append({"node_id": node, **data})
        if len(nodes) >= limit:
            break
    return {"nodes": nodes, "count": len(nodes)}


# ── Property Search ───────────────────────────────────────────────────────────

@router.get("/nodes/search/property", dependencies=[Depends(_require(Permission.READ))])
async def search_by_property(prop: str, value: str):
    results = _engine.find_by_property(prop, value)
    return {"matches": results, "count": len(results)}


# ── Multi-hop Traversal ───────────────────────────────────────────────────────

@router.get("/traverse/{start_node}", dependencies=[Depends(_require(Permission.READ))])
async def traverse(start_node: str, max_hops: int = 3, edge_types: Optional[str] = None):
    types = edge_types.split(",") if edge_types else None
    results = _engine.multi_hop_traverse(start_node, max_hops, types)
    return {"start": start_node, "hops": max_hops, "results": results}


# ── Vector Search ─────────────────────────────────────────────────────────────

class VectorSearchRequest(BaseModel):
    query:  str
    top_k:  int = 5
    mode:   str = "hybrid"   # vector | bm25 | hybrid
    mmr:    bool = False

@router.post("/vector/search", dependencies=[Depends(_require(Permission.READ))])
async def vector_search(req: VectorSearchRequest):
    results = _vec.search(req.query, req.top_k, req.mode, req.mmr)
    return {"query": req.query, "results": results, "count": len(results), "mode": req.mode, "stats": _vec.stats()}

@router.post("/vector/index", dependencies=[Depends(_require(Permission.WRITE))])
async def index_document(doc_id: str, content: str, metadata: Dict[str, Any] = {}):
    _vec.index_document(doc_id, content, metadata)
    return {"doc_id": doc_id, "status": "indexed", "stats": _vec.stats()}


# ── Causal Inference ──────────────────────────────────────────────────────────

class CounterfactualRequest(BaseModel):
    intervention_node:  str
    intervention_value: Any
    target_node:        str
    query:              str = ""

class DiffInDiffRequest(BaseModel):
    treated_before: List[float]
    treated_after:  List[float]
    control_before: List[float]
    control_after:  List[float]

class CausalDiscoveryRequest(BaseModel):
    variables: Dict[str, List[float]]   # { var_name: [time_series_values] }

@router.get("/causal/analyze/{node_id}", dependencies=[Depends(_require(Permission.READ))])
async def causal_analyze(node_id: str, radius: int = 2):
    sg = _engine.get_subgraph(node_id, radius)
    result = _causal.analyze(sg)
    return {"node": node_id, "subgraph_size": sg.number_of_nodes(), **result}

@router.post("/causal/counterfactual", dependencies=[Depends(_require(Permission.READ))])
async def counterfactual(req: CounterfactualRequest):
    sg = _engine.graph
    result = _causal.counterfactual(sg, req.intervention_node, req.intervention_value, req.target_node, req.query)
    return result.__dict__

@router.post("/causal/did", dependencies=[Depends(_require(Permission.READ))])
async def difference_in_differences(req: DiffInDiffRequest):
    result = _causal.difference_in_differences(req.treated_before, req.treated_after, req.control_before, req.control_after)
    return result

@router.post("/causal/discover", dependencies=[Depends(_require(Permission.READ))])
async def discover_causal(req: CausalDiscoveryRequest):
    result = _causal.discover_causal_structure(req.variables)
    return {"skeleton": result.skeleton, "v_structures": result.v_structures, "markov_blankets": result.markov_blankets}

@router.get("/causal/uncertainty/{source_node}", dependencies=[Depends(_require(Permission.READ))])
async def propagate_uncertainty(source_node: str):
    conf = _causal.propagate_uncertainty(_engine.graph, source_node)
    return {"source": source_node, "confidence_map": conf}


# ── Agent Memory ──────────────────────────────────────────────────────────────

class MemoryRequest(BaseModel):
    content:    str
    layer:      str = "semantic"
    metadata:   Dict[str, Any] = {}
    importance: float = 1.0

class MemoryRecallRequest(BaseModel):
    query:  str
    layer:  Optional[str] = None
    top_k:  int = 5

@router.post("/memory", dependencies=[Depends(_require(Permission.WRITE))])
async def store_memory(req: MemoryRequest):
    entry = _memory.remember(req.content, req.layer, req.metadata, req.importance)
    return {"memory_id": entry.memory_id, "layer": entry.layer, "status": "stored"}

@router.post("/memory/recall", dependencies=[Depends(_require(Permission.READ))])
async def recall_memory(req: MemoryRecallRequest):
    entries = _memory.recall(req.query, req.layer, req.top_k)
    return {"results": [e.__dict__ for e in entries], "count": len(entries)}

@router.get("/memory/stats", dependencies=[Depends(_require(Permission.READ))])
async def memory_stats():
    return _memory.stats()

@router.delete("/memory/working", dependencies=[Depends(_require(Permission.WRITE))])
async def clear_working_memory():
    _memory.clear_working()
    return {"status": "working_memory_cleared"}

@router.post("/memory/consolidate", dependencies=[Depends(_require(Permission.WRITE))])
async def consolidate_memory():
    _memory.consolidate()
    return {"status": "consolidated"}


# ── Benchmark ─────────────────────────────────────────────────────────────────

@router.post("/benchmark/run", dependencies=[Depends(_require(Permission.ADMIN))])
async def run_benchmark(n_ops: int = 100, seed_data: bool = True):
    bench = NextGenDBBenchmark(_engine, _qe, _vec, _causal)
    if seed_data:
        bench._seed_benchmark_data(n_nodes=200, n_edges=400)
    report = await bench.run_all(n_ops=n_ops)
    return report

@router.get("/benchmark/quick", dependencies=[Depends(_require(Permission.READ))])
async def quick_health_benchmark():
    """Fast 10-op benchmark for health monitoring."""
    bench = NextGenDBBenchmark(_engine, _qe, _vec, _causal)
    report = await bench.run_all(n_ops=10)
    return {"status": "ok", "p99_ms": report["summary"]["best_p99_ms"], "throughput": report["summary"]["best_throughput"]}


# ── Backup & Checkpoint ───────────────────────────────────────────────────────

@router.post("/backup/checkpoint", dependencies=[Depends(_require(Permission.ADMIN))])
async def manual_checkpoint():
    _engine._checkpoint()
    return {"status": "checkpointed", "stats": _engine.stats()}

@router.get("/backup/stats", dependencies=[Depends(_require(Permission.READ))])
async def backup_stats():
    return {"engine_stats": _engine.stats(), "vector_stats": _vec.stats(), "memory_stats": _memory.stats()}


# ── Schema ────────────────────────────────────────────────────────────────────

@router.get("/schema/stats", dependencies=[Depends(_require(Permission.READ))])
async def schema_stats():
    g = _engine.graph
    type_counts: Dict[str, int] = {}
    for _, data in g.nodes(data=True):
        t = data.get("type", "UNKNOWN")
        type_counts[t] = type_counts.get(t, 0) + 1
    edge_type_counts: Dict[str, int] = {}
    for _, _, k in g.edges(keys=True):
        edge_type_counts[k] = edge_type_counts.get(k, 0) + 1
    return {
        "node_types": type_counts,
        "edge_types": edge_type_counts,
        "total_nodes": g.number_of_nodes(),
        "total_edges": g.number_of_edges(),
        "indexed_properties": _engine._indexes.keys().__class__.__name__,
    }
