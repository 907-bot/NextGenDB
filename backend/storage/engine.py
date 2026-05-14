"""
Persistent Graph Engine — combines WAL + MVCC + JSON snapshot persistence.

Replace NetworkX's in-memory MultiDiGraph with this as the single source of
truth.  The engine:

1. Loads any existing snapshot from disk on startup.
2. Replays uncommitted WAL records on top.
3. Wraps every mutation in a WAL append + MVCC transaction.
4. Periodically checkpoints (serialises full graph → JSON) and truncates WAL.
5. Provides B-tree-like indexing via sorted dict structures for O(log n) lookups.
"""
import asyncio
import json
import logging
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

import networkx as nx

from .wal  import WriteAheadLog
from .txn  import MVCCManager

logger = logging.getLogger("nextgendb.storage.engine")


class IndexEntry:
    __slots__ = ("prop", "value", "node_id")
    def __init__(self, prop: str, value: Any, node_id: str):
        self.prop    = prop
        self.value   = value
        self.node_id = node_id


class PersistentGraphEngine:
    """
    Production-grade persistent graph engine with:
      • WAL-backed crash recovery
      • MVCC ACID transactions
      • Property indexes (inverted index per property key)
      • JSON checkpoint persistence
      • Full NetworkX interoperability (exposes .graph attribute)
    """

    CHECKPOINT_EVERY = 500   # checkpoint every N writes

    def __init__(self, data_dir: str | Path = "data"):
        self._data_dir = Path(data_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)

        self._snapshot_path = self._data_dir / "graph_snapshot.json"
        self._wal           = WriteAheadLog(self._data_dir / "wal.log")
        self._mvcc          = MVCCManager()
        self._lock          = threading.RLock()
        self._write_count   = 0

        # In-memory graph (source of truth after recovery)
        self.graph: nx.MultiDiGraph = nx.MultiDiGraph()

        # Property indexes: { property_key: { value: set(node_ids) } }
        self._indexes: Dict[str, Dict[Any, Set[str]]] = {}

        self._recover()
        logger.info(
            "PersistentGraphEngine ready — %d nodes, %d edges",
            self.graph.number_of_nodes(),
            self.graph.number_of_edges(),
        )

    # ── Recovery ──────────────────────────────────────────────────────────────

    def _recover(self):
        """Load snapshot then replay WAL on top."""
        if self._snapshot_path.exists():
            try:
                data = json.loads(self._snapshot_path.read_text())
                self.graph = nx.node_link_graph(data, multigraph=True, directed=True)
                # Rebuild indexes from snapshot
                for node, props in self.graph.nodes(data=True):
                    self._index_node(node, props)
                logger.info("Loaded snapshot: %d nodes", self.graph.number_of_nodes())
            except Exception as exc:
                logger.error("Snapshot load failed: %s — starting empty", exc)

        # Replay WAL
        tx_records = {}
        committed_txs = set()
        
        for rec in self._wal.replay():
            if rec.op == "COMMIT":
                committed_txs.add(rec.tx_id)
            elif rec.op == "ABORT":
                pass
            else:
                tx_records.setdefault(rec.tx_id, []).append(rec)
                
        replayed = 0
        for tx_id in committed_txs:
            for rec in tx_records.get(tx_id, []):
                if rec.op == "ADD_NODE":
                    self._apply_add_node(rec.entity_id, rec.data)
                elif rec.op == "DEL_NODE":
                    self._apply_del_node(rec.entity_id)
                elif rec.op == "ADD_EDGE":
                    d = rec.data
                    self._apply_add_edge(d["source"], d["target"], d["edge_type"], d.get("props", {}))
                elif rec.op == "DEL_EDGE":
                    d = rec.data
                    self._apply_del_edge(d["source"], d["target"], d.get("edge_type"))
                replayed += 1
                
        if replayed:
            logger.info("WAL replay: applied %d records from %d committed transactions", replayed, len(committed_txs))

    # ── Internal apply (no WAL/MVCC — used during recovery) ───────────────────

    def _apply_add_node(self, node_id: str, props: Dict):
        self.graph.add_node(node_id, **props)
        self._index_node(node_id, props)

    def _apply_del_node(self, node_id: str):
        if node_id in self.graph:
            self._deindex_node(node_id)
            self.graph.remove_node(node_id)

    def _apply_add_edge(self, src: str, tgt: str, edge_type: str, props: Dict):
        self.graph.add_edge(src, tgt, key=edge_type, **(props or {}))

    def _apply_del_edge(self, src: str, tgt: str, edge_type: Optional[str]):
        if self.graph.has_edge(src, tgt, key=edge_type):
            self.graph.remove_edge(src, tgt, key=edge_type)

    # ── Indexing ───────────────────────────────────────────────────────────────

    def _index_node(self, node_id: str, props: Dict):
        for key, value in props.items():
            try:
                hashable = value if not isinstance(value, (list, dict)) else json.dumps(value)
            except TypeError:
                continue
            self._indexes.setdefault(key, {}).setdefault(hashable, set()).add(node_id)

    def _deindex_node(self, node_id: str):
        props = self.graph.nodes.get(node_id, {})
        for key, value in props.items():
            try:
                hashable = value if not isinstance(value, (list, dict)) else json.dumps(value)
            except TypeError:
                continue
            if key in self._indexes and hashable in self._indexes[key]:
                self._indexes[key][hashable].discard(node_id)

    # ── Checkpoint ────────────────────────────────────────────────────────────

    def _maybe_checkpoint(self):
        self._write_count += 1
        if self._write_count >= self.CHECKPOINT_EVERY:
            self._checkpoint()
            self._write_count = 0

    def _checkpoint(self):
        data = nx.node_link_data(self.graph)
        self._snapshot_path.write_text(json.dumps(data, default=str))
        self._wal.checkpoint(self._wal._lsn)
        logger.info("Checkpoint complete — snapshot saved")

    # ── Transaction Management ────────────────────────────────────────────────

    def begin_transaction(self) -> str:
        """Start a new MVCC transaction and return its tx_id."""
        tx = self._mvcc.begin()
        return tx.tx_id

    def commit(self, tx_id: str):
        """Commit an active transaction: WAL append + graph update."""
        with self._lock:
            tx = self._mvcc._active_txs.get(tx_id)
            if not tx:
                raise ValueError(f"Transaction {tx_id} not found or already completed")

            # 1. Finalise MVCC first to check for conflicts
            success = self._mvcc.commit(tx)
            if not success:
                self._wal.append("ABORT", "transaction", tx_id, {}, tx_id)
                raise RuntimeError("Transaction aborted due to write conflict")

            # 2. Log all writes to WAL
            for key, val in tx.writes.items():
                if key.startswith("node:"):
                    nid = key.split(":", 1)[1]
                    self._wal.append("ADD_NODE", "node", nid, val, tx_id)
                elif key.startswith("edge:"):
                    # edge key format: edge:src:tgt:type
                    _, src, tgt, etype = key.split(":", 3)
                    self._wal.append("ADD_EDGE", "edge", f"{src}→{tgt}", {"source": src, "target": tgt, "edge_type": etype, "props": val}, tx_id)
            
            for key in tx.deletes:
                if key.startswith("node:"):
                    nid = key.split(":", 1)[1]
                    self._wal.append("DEL_NODE", "node", nid, {}, tx_id)
                elif key.startswith("edge:"):
                    _, src, tgt, etype = key.split(":", 3)
                    self._wal.append("DEL_EDGE", "edge", f"{src}→{tgt}", {"source": src, "target": tgt, "edge_type": etype}, tx_id)

            # 3. Append COMMIT record
            self._wal.append("COMMIT", "transaction", tx_id, {}, tx_id)

            # 4. Apply to in-memory graph (the single source of truth)
            for key, val in tx.writes.items():
                if key.startswith("node:"):
                    self._apply_add_node(key.split(":", 1)[1], val)
                elif key.startswith("edge:"):
                    _, src, tgt, etype = key.split(":", 3)
                    self._apply_add_edge(src, tgt, etype, val)
            
            for key in tx.deletes:
                if key.startswith("node:"):
                    self._apply_del_node(key.split(":", 1)[1])
                elif key.startswith("edge:"):
                    _, src, tgt, etype = key.split(":", 3)
                    self._apply_del_edge(src, tgt, etype)

            self._maybe_checkpoint()

    def rollback(self, tx_id: str):
        """Abort a transaction."""
        tx = self._mvcc._active_txs.get(tx_id)
        if tx:
            self._wal.append("ABORT", "transaction", tx_id, {}, tx_id)
            self._mvcc.abort(tx)

    # ── Public Write API (Auto-commit helpers) ────────────────────────────────

    def add_node(self, node_id: str, properties: Dict[str, Any], tx_id: Optional[str] = None) -> str:
        """Add a node. If no tx_id provided, it auto-commits."""
        managed = tx_id is None
        if managed:
            tx_id = self.begin_transaction()
        
        tx = self._mvcc._active_txs[tx_id]
        tx.write(f"node:{node_id}", properties)
        
        if managed:
            self.commit(tx_id)
        return node_id

    def add_edge(
        self,
        source: str,
        target: str,
        edge_type: str,
        properties: Dict[str, Any] = None,
        tx_id: Optional[str] = None,
    ) -> Tuple[str, str, str]:
        managed = tx_id is None
        if managed:
            tx_id = self.begin_transaction()
        
        tx = self._mvcc._active_txs[tx_id]
        tx.write(f"edge:{source}:{target}:{edge_type}", properties or {})
        
        if managed:
            self.commit(tx_id)
        return (source, target, edge_type)

    def delete_node(self, node_id: str, tx_id: Optional[str] = None):
        managed = tx_id is None
        if managed:
            tx_id = self.begin_transaction()
        
        tx = self._mvcc._active_txs[tx_id]
        tx.delete(f"node:{node_id}")
        
        if managed:
            self.commit(tx_id)

    def delete_edge(self, source: str, target: str, edge_type: str, tx_id: Optional[str] = None):
        managed = tx_id is None
        if managed:
            tx_id = self.begin_transaction()
        
        tx = self._mvcc._active_txs[tx_id]
        tx.delete(f"edge:{source}:{target}:{edge_type}")
        
        if managed:
            self.commit(tx_id)

    # ── Public Read API ────────────────────────────────────────────────────────

    def get_node(self, node_id: str) -> Optional[Dict]:
        return dict(self.graph.nodes[node_id]) if node_id in self.graph else None

    def find_by_property(self, prop: str, value: Any) -> List[str]:
        """O(1) index lookup for nodes with a matching property value."""
        try:
            hashable = value if not isinstance(value, (list, dict)) else json.dumps(value)
        except TypeError:
            return []
        return list(self._indexes.get(prop, {}).get(hashable, set()))

    def get_subgraph(self, center_node: str, radius: int = 2) -> nx.MultiDiGraph:
        if center_node not in self.graph:
            return nx.MultiDiGraph()
        nodes = nx.single_source_shortest_path_length(self.graph, center_node, cutoff=radius).keys()
        return self.graph.subgraph(nodes)

    def multi_hop_traverse(self, start: str, max_hops: int = 3, edge_types: Optional[List[str]] = None) -> List[Dict]:
        """Multi-hop traversal with optional edge-type filtering."""
        results = []
        visited = set()
        queue   = [(start, 0, [])]
        while queue:
            node, depth, path = queue.pop(0)
            if node in visited or depth > max_hops:
                continue
            visited.add(node)
            node_data = self.get_node(node) or {}
            results.append({"node": node, "depth": depth, "path": path, "properties": node_data})
            for _, nbr, key in self.graph.out_edges(node, keys=True):
                if edge_types and key not in edge_types:
                    continue
                queue.append((nbr, depth + 1, path + [(node, key, nbr)]))
        return results

    def stats(self) -> Dict:
        return {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "indexed_properties": list(self._indexes.keys()),
            "snapshot_path": str(self._snapshot_path),
            "wal_lsn": self._wal._lsn,
        }

    def to_json(self) -> Dict:
        data = nx.node_link_data(self.graph)
        for node in data["nodes"]:
            node["id"] = node.get("id") or node.get("name")
        return data

    def close(self):
        self._checkpoint()
        self._wal.close()
        logger.info("PersistentGraphEngine closed")

    # ── Async checkpoint (background task) ───────────────────────────────────

    async def background_checkpoint_loop(self, interval_seconds: int = 300):
        """Periodically flush graph to disk; run as an asyncio task."""
        while True:
            await asyncio.sleep(interval_seconds)
            with self._lock:
                self._checkpoint()
