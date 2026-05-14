"""
Neural Agent — Two-stage query decomposition and planning.

Stage 1 (Decompose): Break a natural language query into atomic sub-goals.
Stage 2 (Plan):     Map sub-goals to graph operations using a priority queue.

Integrates with:
  • AsyncGNNLearner  for embedding-guided retrieval
  • CausalInferenceEngine for causal sub-goal identification
  • AgenticMemoryStore for episodic context injection
  • VectorSearchEngine for semantic grounding
"""
import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("nextgendb.agent.neural")


# ── Action types ──────────────────────────────────────────────────────────────

class ActionType(str, Enum):
    GRAPH_TRAVERSE  = "GRAPH_TRAVERSE"
    VECTOR_SEARCH   = "VECTOR_SEARCH"
    CAUSAL_ANALYZE  = "CAUSAL_ANALYZE"
    TEMPORAL_ORDER  = "TEMPORAL_ORDER"
    MEMORY_RECALL   = "MEMORY_RECALL"
    SYNTHESIZE      = "SYNTHESIZE"
    EXPLAIN         = "EXPLAIN"


# ── Sub-goal ──────────────────────────────────────────────────────────────────

@dataclass
class SubGoal:
    goal_id:     int
    description: str
    action:      ActionType
    args:        Dict[str, Any] = field(default_factory=dict)
    priority:    int = 5           # 1=highest, 10=lowest
    depends_on:  List[int] = field(default_factory=list)
    result:      Optional[Any] = None
    done:        bool = False
    error:       Optional[str] = None


# ── Stage 1 — Query Decomposer ────────────────────────────────────────────────

class QueryDecomposer:
    """
    Heuristic + keyword-based query decomposer.

    In production, this would call an LLM (OpenAI / local) to produce
    a structured plan.  Here we use deterministic rules so the system
    works without any external API key.
    """

    # Keyword → (ActionType, priority)
    _INTENT_MAP: List[Tuple[List[str], ActionType, int]] = [
        (["why", "cause", "reason", "impact", "effect"],  ActionType.CAUSAL_ANALYZE, 2),
        (["when", "timeline", "history", "sequence", "before", "after", "temporal"], ActionType.TEMPORAL_ORDER, 3),
        (["similar", "like", "related", "semantic", "find"],     ActionType.VECTOR_SEARCH, 4),
        (["remember", "recall", "past", "previous", "memory"],   ActionType.MEMORY_RECALL, 3),
        (["path", "hop", "connect", "traverse", "between"],      ActionType.GRAPH_TRAVERSE, 2),
        (["explain", "how", "describe"],                         ActionType.EXPLAIN, 5),
    ]

    def decompose(self, query: str) -> List[SubGoal]:
        q_lower = query.lower()
        goals: List[SubGoal] = []
        gid = 1

        # Always start with context retrieval
        goals.append(SubGoal(
            goal_id=gid,
            description=f"Retrieve graph context for: '{query}'",
            action=ActionType.GRAPH_TRAVERSE,
            args={"query": query, "max_hops": 2},
            priority=1,
        ))
        gid += 1

        seen_actions = {ActionType.GRAPH_TRAVERSE}

        for keywords, action, priority in self._INTENT_MAP:
            if action in seen_actions:
                continue
            if any(kw in q_lower for kw in keywords):
                goals.append(SubGoal(
                    goal_id=gid,
                    description=f"Execute {action.value} for: '{query}'",
                    action=action,
                    args={"query": query},
                    priority=priority,
                    depends_on=[1],
                ))
                seen_actions.add(action)
                gid += 1

        # Always add vector search if not already present
        if ActionType.VECTOR_SEARCH not in seen_actions:
            goals.append(SubGoal(
                goal_id=gid,
                description="Semantic similarity search for supporting evidence",
                action=ActionType.VECTOR_SEARCH,
                args={"query": query, "top_k": 5},
                priority=4,
                depends_on=[1],
            ))
            gid += 1

        # Final synthesis — depends on all prior goals
        synthesis_deps = [g.goal_id for g in goals]
        goals.append(SubGoal(
            goal_id=gid,
            description="Synthesize all evidence into a high-confidence answer",
            action=ActionType.SYNTHESIZE,
            args={"query": query},
            priority=10,
            depends_on=synthesis_deps,
        ))

        # Sort by priority, preserving dependencies
        return sorted(goals, key=lambda g: g.priority)


# ── Stage 2 — Neural Planner ──────────────────────────────────────────────────

class NeuralAgentPlanner:
    """
    Two-stage query decomposition and planning agent.

    Usage:
        agent = NeuralAgentPlanner(engine, gnn_learner, vec_engine, causal, memory)
        result = await agent.run("Why did customer churn increase after the pricing change?")
    """

    def __init__(
        self,
        engine=None,
        gnn_learner=None,
        vec_engine=None,
        causal_engine=None,
        memory_store=None,
    ):
        self._engine  = engine
        self._gnn     = gnn_learner
        self._vec     = vec_engine
        self._causal  = causal_engine
        self._memory  = memory_store
        self._decomposer = QueryDecomposer()

    async def run(self, query: str) -> Dict[str, Any]:
        """Entry point: decompose → plan → execute → synthesize."""
        t0 = time.perf_counter()

        # Stage 1: Decompose
        goals = self._decomposer.decompose(query)
        logger.info("NeuralAgent: decomposed '%s' into %d sub-goals", query[:60], len(goals))

        # Stage 2: Execute in dependency order
        context: Dict[int, Any] = {}
        for goal in goals:
            # Check dependencies are met
            if not all(context.get(dep) is not None for dep in goal.depends_on):
                goal.error = "dependencies not met"
                continue
            try:
                result = await self._execute_goal(goal, query, context)
                goal.result = result
                goal.done = True
                context[goal.goal_id] = result
            except Exception as exc:
                logger.warning("Goal %d (%s) failed: %s", goal.goal_id, goal.action, exc)
                goal.error = str(exc)
                context[goal.goal_id] = {"error": str(exc)}

        # Build final answer
        answer, confidence = self._synthesize(query, context, goals)

        # Store in episodic memory
        if self._memory:
            try:
                self._memory.remember(
                    f"Q: {query} | A: {answer[:120]}",
                    layer="episodic",
                    metadata={"confidence": confidence, "goal_count": len(goals)},
                    importance=confidence,
                )
            except Exception:
                pass

        latency_ms = round((time.perf_counter() - t0) * 1000, 1)

        return {
            "query":       query,
            "answer":      answer,
            "confidence":  round(confidence, 4),
            "plan_steps":  [self._goal_to_dict(g) for g in goals],
            "latency_ms":  latency_ms,
            "context_keys": list(context.keys()),
        }

    async def _execute_goal(
        self, goal: SubGoal, query: str, context: Dict[int, Any]
    ) -> Any:
        """Dispatch to the appropriate sub-system."""

        if goal.action == ActionType.GRAPH_TRAVERSE:
            if self._gnn:
                return self._gnn.vectorless_subgraph(query, top_k=10, max_hops=2)
            if self._engine:
                nodes = list(self._engine.graph.nodes(data=True))[:10]
                return {"nodes": [{n: d} for n, d in nodes], "node_count": len(nodes)}
            return {"nodes": [], "node_count": 0}

        elif goal.action == ActionType.VECTOR_SEARCH:
            if self._vec:
                results = self._vec.search(query, top_k=goal.args.get("top_k", 5))
                return {"hits": results, "count": len(results)}
            if self._gnn:
                return {"hits": self._gnn.query(query, top_k=5), "count": 5}
            return {"hits": [], "count": 0}

        elif goal.action == ActionType.CAUSAL_ANALYZE:
            if self._causal and self._engine:
                subgraph = self._engine.get_subgraph(
                    self._engine.graph.nodes().__iter__().__next__()
                    if self._engine.graph.number_of_nodes() > 0 else "Root_System",
                    radius=2,
                )
                return self._causal.analyze(subgraph)
            return {"root_causes": ["Unknown"], "impact_chain": []}

        elif goal.action == ActionType.TEMPORAL_ORDER:
            if self._engine:
                timed = [
                    {"node": n, "timestamp": d.get("timestamp"), **d}
                    for n, d in self._engine.graph.nodes(data=True)
                    if d.get("timestamp")
                ]
                timed.sort(key=lambda x: x.get("timestamp", ""))
                return {"timeline": timed, "count": len(timed)}
            return {"timeline": [], "count": 0}

        elif goal.action == ActionType.MEMORY_RECALL:
            if self._memory:
                entries = self._memory.recall(query, top_k=5)
                return {"memories": [e.__dict__ for e in entries], "count": len(entries)}
            return {"memories": [], "count": 0}

        elif goal.action == ActionType.EXPLAIN:
            # Return the plan structure itself as explanation
            return {"explanation": f"Processing query via {len(goal.depends_on)} sub-systems"}

        elif goal.action == ActionType.SYNTHESIZE:
            return await asyncio.sleep(0, result=None)  # handled in synthesize()

        return {}

    def _synthesize(
        self, query: str, context: Dict[int, Any], goals: List[SubGoal]
    ) -> Tuple[str, float]:
        """Build a natural language answer from all collected context."""
        parts: List[str] = []
        confidence_signals: List[float] = []

        # Causal findings
        causal_ctx = next(
            (context[g.goal_id] for g in goals if g.action == ActionType.CAUSAL_ANALYZE and g.done), None
        )
        if causal_ctx and "root_causes" in causal_ctx:
            causes = causal_ctx.get("root_causes", [])
            if causes and causes[0] != "Unknown":
                parts.append(f"Causal analysis identifies: {causes[0]}.")
                confidence_signals.append(0.85)

        # Vector search hits
        vec_ctx = next(
            (context[g.goal_id] for g in goals if g.action == ActionType.VECTOR_SEARCH and g.done), None
        )
        if vec_ctx and vec_ctx.get("count", 0) > 0:
            hit_count = vec_ctx["count"]
            parts.append(f"Semantic search found {hit_count} relevant nodes in the graph.")
            confidence_signals.append(0.80)

        # Temporal ordering
        temp_ctx = next(
            (context[g.goal_id] for g in goals if g.action == ActionType.TEMPORAL_ORDER and g.done), None
        )
        if temp_ctx and temp_ctx.get("count", 0) > 0:
            parts.append(f"Timeline contains {temp_ctx['count']} timestamped events.")
            confidence_signals.append(0.75)

        # Graph traversal summary
        graph_ctx = next(
            (context[g.goal_id] for g in goals if g.action == ActionType.GRAPH_TRAVERSE and g.done), None
        )
        if graph_ctx:
            nc = graph_ctx.get("node_count", 0)
            if nc > 0:
                parts.append(f"Graph traversal retrieved a {nc}-node context subgraph.")
                confidence_signals.append(0.90)

        if not parts:
            answer = f"The query '{query}' was processed but no definitive pattern was found in the current graph."
            confidence = 0.50
        else:
            answer = " ".join(parts)
            confidence = sum(confidence_signals) / len(confidence_signals)

        return answer, confidence

    @staticmethod
    def _goal_to_dict(g: SubGoal) -> Dict[str, Any]:
        return {
            "id":          g.goal_id,
            "action":      g.action.value,
            "description": g.description,
            "done":        g.done,
            "error":       g.error,
            "priority":    g.priority,
            "depends_on":  g.depends_on,
        }
