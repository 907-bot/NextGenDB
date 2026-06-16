"""Tests for Neural Agent Planner"""
import pytest

from backend.agent.neural_agent import NeuralAgentPlanner
from backend.storage.engine import PersistentGraphEngine
from backend.vector.search import VectorSearchEngine
from backend.causal.inference import CausalInferenceEngine
from backend.agent.memory import AgenticMemoryStore


class TestNeuralAgentPlanner:
    """Agent decomposition and planning"""

    @pytest.fixture
    def setup(self, tmp_path):
        """Setup agent with all dependencies"""
        engine = PersistentGraphEngine(tmp_path)
        vec_engine = VectorSearchEngine(tmp_path)
        causal_engine = CausalInferenceEngine()
        memory = AgenticMemoryStore(engine, tmp_path / "memory.json")
        
        agent = NeuralAgentPlanner(
            engine=engine,
            gnn_learner=None,
            vec_engine=vec_engine,
            causal_engine=causal_engine,
            memory_store=memory
        )
        
        return engine, agent

    def test_query_decomposition(self, setup):
        """Should decompose complex queries into subgoals"""
        engine, agent = setup
        
        query = "Find all customers who bought Product X and had churn signals"
        plan = agent.decompose(query)
        
        assert plan is not None
        assert len(plan) > 0
        # Should identify intents: search, filter, causality

    def test_execution_planning(self, setup):
        """Should create execution plans"""
        engine, agent = setup
        
        engine.add_node("customer:1", {"status": "active"})
        engine.add_node("product:1", {"name": "Product X"})
        engine.add_edge("customer:1", "product:1", "BOUGHT", {})
        
        subgoals = [
            {"intent": "VECTOR_SEARCH", "query": "Find customers"},
            {"intent": "FILTER", "predicate": "status == active"},
        ]
        
        plan = agent.plan(subgoals)
        
        assert plan is not None
        assert len(plan) > 0

    def test_memory_retrieval_augmentation(self, setup):
        """Should augment reasoning with memory"""
        engine, agent = setup
        
        # Store fact in memory
        agent.memory_store.store_semantic("customers", "Customers who purchased Product X")
        
        # Retrieve context
        context = agent.memory_store.retrieve_semantic("customers")
        
        assert context is not None
        assert "Product X" in context
