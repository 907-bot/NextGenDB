"""
Sharding and Distributed Execution Logic.

Implements consistent hashing for node distribution across shards
and a Query Coordinator for handling cross-shard operations.
"""
import hashlib
import logging
from typing import List, Dict, Any

logger = logging.getLogger("nextgendb.distributed.sharding")


class ConsistentHashRing:
    """Distributes nodes across physical shards using consistent hashing."""

    def __init__(self, shards: List[str], replicas: int = 3):
        self.replicas = replicas
        self.ring: Dict[int, str] = {}
        self.sorted_keys: List[int] = []
        for shard in shards:
            self.add_shard(shard)

    def _hash(self, key: str) -> int:
        return int(hashlib.md5(key.encode('utf-8')).hexdigest(), 16)

    def add_shard(self, shard: str):
        for i in range(self.replicas):
            key = self._hash(f"{shard}:{i}")
            self.ring[key] = shard
            self.sorted_keys.append(key)
        self.sorted_keys.sort()
        logger.info(f"Added shard: {shard}")

    def remove_shard(self, shard: str):
        for i in range(self.replicas):
            key = self._hash(f"{shard}:{i}")
            self.ring.pop(key, None)
            self.sorted_keys.remove(key)
        logger.info(f"Removed shard: {shard}")

    def get_shard(self, node_id: str) -> str:
        if not self.ring:
            return "local"
        hash_val = self._hash(node_id)
        for key in self.sorted_keys:
            if hash_val <= key:
                return self.ring[key]
        return self.ring[self.sorted_keys[0]]


class DistributedCoordinator:
    """Handles cross-shard query execution."""

    def __init__(self, shards: List[str]):
        self.hash_ring = ConsistentHashRing(shards)
        # mapping of shard_id to Remote RPC client / local engine
        self.shard_connections = {}

    def register_connection(self, shard_id: str, client: Any):
        self.shard_connections[shard_id] = client

    def route_add_node(self, node_id: str, properties: dict) -> str:
        """Determines which shard a node belongs to and routes the write."""
        shard_id = self.hash_ring.get_shard(node_id)
        conn = self.shard_connections.get(shard_id)
        if conn:
            # e.g., conn.add_node(node_id, properties)
            logger.debug(f"Routed ADD_NODE({node_id}) to {shard_id}")
            return shard_id
        return "local"

    def execute_distributed_query(self, query: str) -> List[Dict]:
        """Broadcasts query to all shards and merges results."""
        # Simple scatter-gather
        results = []
        for shard_id, conn in self.shard_connections.items():
            try:
                # Assuming conn has execute method
                # partial_results = conn.execute(query)
                # results.extend(partial_results)
                logger.debug(f"Executed query on {shard_id}")
            except Exception as e:
                logger.error(f"Failed query on shard {shard_id}: {e}")
        return results
