"""Tests for Query Language Engine (Cypher + SQL)"""
import pytest

from backend.query.lang import QueryParser, QueryExecutor
from backend.storage.engine import PersistentGraphEngine


class TestQueryParser:
    """Cypher and SQL parsing"""

    def test_parse_simple_match(self):
        """Should parse basic MATCH queries"""
        query = "MATCH (n:User) RETURN n"
        parsed = QueryParser.parse(query)
        
        assert parsed["type"] == "CYPHER"
        assert len(parsed["nodes"]) == 1
        assert parsed["nodes"][0]["label"] == "User"

    def test_parse_match_with_where(self):
        """Should parse MATCH with WHERE clause"""
        query = "MATCH (n:User) WHERE n.age > 30 RETURN n"
        parsed = QueryParser.parse(query)
        
        assert len(parsed["filters"]) == 1
        assert parsed["filters"][0]["prop"] == "age"

    def test_parse_match_with_edges(self):
        """Should parse MATCH with edge patterns"""
        query = "MATCH (n:User)-[e:KNOWS]->(m:User) RETURN n, m"
        parsed = QueryParser.parse(query)
        
        assert len(parsed["nodes"]) == 2
        assert len(parsed["edges"]) == 1
        assert parsed["edges"][0]["type"] == "KNOWS"

    def test_parse_sql_select(self):
        """Should parse SQL SELECT queries"""
        query = "SELECT * FROM nodes WHERE status = 'active'"
        parsed = QueryParser.parse(query)
        
        assert parsed["type"] == "SQL"
        assert parsed["table"] == "nodes"
        assert len(parsed["filters"]) == 1

    def test_parse_parameterized_query(self):
        """Should substitute parameters safely"""
        query = "MATCH (n:User) WHERE n.name = $name RETURN n"
        params = {"name": "Alice"}
        parsed = QueryParser.parse(query, params)
        
        assert parsed["filters"][0]["val"] == "Alice"

    def test_parse_explain_query(self):
        """Should handle EXPLAIN queries"""
        query = "EXPLAIN MATCH (n:User) RETURN n"
        parsed = QueryParser.parse(query)
        
        assert parsed["type"] == "EXPLAIN"
        assert parsed["inner"]["type"] == "CYPHER"


class TestQueryExecutor:
    """Query execution against graph engine"""

    @pytest.fixture
    def engine(self, tmp_path):
        """Setup test graph"""
        engine = PersistentGraphEngine(tmp_path)
        engine.add_node("user:1", {"name": "Alice", "type": "User", "age": 30})
        engine.add_node("user:2", {"name": "Bob", "type": "User", "age": 25})
        engine.add_node("user:3", {"name": "Charlie", "type": "User", "age": 35})
        engine.add_edge("user:1", "user:2", "KNOWS", {})
        engine.add_edge("user:2", "user:3", "KNOWS", {})
        return engine

    def test_execute_match_all(self, engine):
        """Should execute simple MATCH queries"""
        executor = QueryExecutor(engine)
        query = "MATCH (n:User) RETURN n"
        result = executor.execute(query)
        
        assert result.returned == 3
        assert len(result.rows) == 3

    def test_execute_match_with_filter(self, engine):
        """Should execute MATCH with WHERE filters"""
        executor = QueryExecutor(engine)
        query = "MATCH (n:User) WHERE n.name = Alice RETURN n"
        result = executor.execute(query)
        
        assert result.returned == 1
        assert result.rows[0]["name"] == "Alice"

    def test_execute_match_with_edge(self, engine):
        """Should execute MATCH with edge traversal"""
        executor = QueryExecutor(engine)
        query = "MATCH (n:User)-[e:KNOWS]->(m:User) RETURN n, m"
        result = executor.execute(query)
        
        assert result.returned == 2

    def test_execute_sql_select(self, engine):
        """Should execute SQL SELECT queries"""
        executor = QueryExecutor(engine)
        query = "SELECT * FROM nodes"
        result = executor.execute(query)
        
        assert result.returned == 3

    def test_explain_query_plan(self, engine):
        """Should generate query execution plans"""
        executor = QueryExecutor(engine)
        query = "EXPLAIN MATCH (n:User) RETURN n"
        result = executor.execute(query)
        
        assert result.plan is not None
        assert result.plan.op == "CypherMatch"
        plan_str = result.plan.explain()
        assert "NodeScan" in plan_str
