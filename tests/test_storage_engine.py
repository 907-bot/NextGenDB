"""Tests for Persistent Graph Engine (WAL + MVCC + Indexing)"""
import json
import tempfile
from pathlib import Path
import pytest

from backend.storage.engine import PersistentGraphEngine
from backend.storage.wal import WriteAheadLog, WALCorruptError


class TestWriteAheadLog:
    """WAL recovery and corruption detection"""

    def test_wal_append_and_replay(self, tmp_path):
        """WAL should persist records and replay them on recovery"""
        wal_path = tmp_path / "test.wal"
        wal = WriteAheadLog(wal_path)
        
        # Append records
        rec1 = wal.append("ADD_NODE", "node", "node1", {"name": "Alice"}, "tx1")
        rec2 = wal.append("ADD_NODE", "node", "node2", {"name": "Bob"}, "tx1")
        wal.append("COMMIT", "transaction", "tx1", {}, "tx1")
        
        assert rec1.lsn == 1
        assert rec2.lsn == 2
        
        # Replay
        replayed = wal.replay()
        assert len(replayed) == 3
        assert replayed[0].op == "ADD_NODE"
        assert replayed[2].op == "COMMIT"
        
        wal.close()

    def test_wal_checkpoint(self, tmp_path):
        """WAL should truncate after checkpoint"""
        wal_path = tmp_path / "test.wal"
        wal = WriteAheadLog(wal_path)
        
        # Add records
        for i in range(5):
            wal.append("ADD_NODE", "node", f"node{i}", {}, f"tx{i}")
        
        # Checkpoint at LSN 3
        wal.checkpoint(3)
        
        # Reopen and verify only records after LSN 3 remain
        wal.close()
        wal2 = WriteAheadLog(wal_path)
        replayed = wal2.replay()
        assert len(replayed) == 2  # LSN 4 and 5
        wal2.close()


class TestPersistentGraphEngine:
    """Persistent graph engine with MVCC and indexing"""

    def test_add_and_retrieve_node(self, tmp_path):
        """Engine should persist and retrieve nodes"""
        engine = PersistentGraphEngine(tmp_path)
        
        engine.add_node("user:1", {"name": "Alice", "age": 30})
        node = engine.get_node("user:1")
        
        assert node is not None
        assert node["name"] == "Alice"
        assert node["age"] == 30

    def test_add_and_retrieve_edge(self, tmp_path):
        """Engine should persist and retrieve edges"""
        engine = PersistentGraphEngine(tmp_path)
        
        engine.add_node("user:1", {"name": "Alice"})
        engine.add_node("user:2", {"name": "Bob"})
        engine.add_edge("user:1", "user:2", "KNOWS", {"since": 2020})
        
        assert engine.graph.has_edge("user:1", "user:2", key="KNOWS")
        edge_data = engine.graph["user:1"]["user:2"]["KNOWS"]
        assert edge_data["since"] == 2020

    def test_property_indexing(self, tmp_path):
        """Engine should support O(1) property lookups"""
        engine = PersistentGraphEngine(tmp_path)
        
        engine.add_node("user:1", {"status": "active"})
        engine.add_node("user:2", {"status": "active"})
        engine.add_node("user:3", {"status": "inactive"})
        
        active_users = engine.find_by_property("status", "active")
        assert len(active_users) == 2
        assert "user:1" in active_users
        assert "user:2" in active_users

    def test_multi_hop_traversal(self, tmp_path):
        """Engine should support multi-hop graph traversal"""
        engine = PersistentGraphEngine(tmp_path)
        
        # Create chain: A -> B -> C -> D
        for node in ["A", "B", "C", "D"]:
            engine.add_node(node, {"label": node})
        
        engine.add_edge("A", "B", "NEXT", {})
        engine.add_edge("B", "C", "NEXT", {})
        engine.add_edge("C", "D", "NEXT", {})
        
        results = engine.multi_hop_traverse("A", max_hops=3)
        assert len(results) == 4
        assert results[0]["node"] == "A"
        assert results[3]["node"] == "D"

    def test_transaction_rollback(self, tmp_path):
        """Engine should support transaction rollback"""
        engine = PersistentGraphEngine(tmp_path)
        
        tx_id = engine.begin_transaction()
        engine.add_node("user:1", {"name": "Alice"}, tx_id=tx_id)
        engine.rollback(tx_id)
        
        # Node should not exist
        assert engine.get_node("user:1") is None

    def test_checkpoint_and_recovery(self, tmp_path):
        """Engine should recover state from checkpoint + WAL"""
        engine = PersistentGraphEngine(tmp_path)
        engine.add_node("user:1", {"name": "Alice"})
        engine.add_node("user:2", {"name": "Bob"})
        engine._checkpoint()
        
        # Create new engine from same data dir
        engine2 = PersistentGraphEngine(tmp_path)
        assert engine2.get_node("user:1")["name"] == "Alice"
        assert engine2.get_node("user:2")["name"] == "Bob"


class TestMVCC:
    """MVCC transaction manager"""

    def test_concurrent_writes_conflict_detection(self, tmp_path):
        """MVCC should detect write conflicts"""
        engine = PersistentGraphEngine(tmp_path)
        
        # Start two transactions
        tx1 = engine.begin_transaction()
        tx2 = engine.begin_transaction()
        
        # Both try to write to same node
        engine.add_node("node:1", {"value": 1}, tx_id=tx1)
        engine.add_node("node:1", {"value": 2}, tx_id=tx2)
        
        # First should succeed
        engine.commit(tx1)
        
        # Second should fail with conflict
        with pytest.raises(RuntimeError, match="write conflict"):
            engine.commit(tx2)
