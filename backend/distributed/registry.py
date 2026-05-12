"""
Layer 9 — Distributed Node Registry & Load Balancer Simulation
In production this would interface with Kubernetes Service Discovery.
For single-node deployments it tracks worker instances in memory.
"""
import asyncio
import logging
import time
import uuid
from typing import Dict, List

logger = logging.getLogger("nextgendb.distributed.registry")


class NodeInfo:
    def __init__(self, host: str, port: int, role: str = "worker"):
        self.node_id   = str(uuid.uuid4())[:8]
        self.host      = host
        self.port      = port
        self.role      = role           # "api", "graph", "worker"
        self.registered_at = time.time()
        self.last_heartbeat = time.time()
        self.healthy   = True

    def heartbeat(self):
        self.last_heartbeat = time.time()

    def is_stale(self, ttl_seconds: int = 30) -> bool:
        return (time.time() - self.last_heartbeat) > ttl_seconds

    def to_dict(self) -> dict:
        return {
            "node_id":   self.node_id,
            "host":      self.host,
            "port":      self.port,
            "role":      self.role,
            "healthy":   self.healthy,
            "uptime_s":  round(time.time() - self.registered_at, 1),
        }


class NodeRegistry:
    """
    In-memory service registry.
    Production equivalent: etcd, Consul, or K8s Endpoints API.
    """

    def __init__(self):
        self._nodes: Dict[str, NodeInfo] = {}
        self._lock = asyncio.Lock()

    async def register(self, host: str, port: int, role: str = "worker") -> NodeInfo:
        async with self._lock:
            node = NodeInfo(host, port, role)
            self._nodes[node.node_id] = node
            logger.info("Node registered: %s @ %s:%s (%s)", node.node_id, host, port, role)
            return node

    async def heartbeat(self, node_id: str) -> bool:
        async with self._lock:
            if node_id in self._nodes:
                self._nodes[node_id].heartbeat()
                return True
            return False

    async def deregister(self, node_id: str):
        async with self._lock:
            self._nodes.pop(node_id, None)
            logger.info("Node deregistered: %s", node_id)

    async def prune_stale(self):
        """Remove nodes that haven't heartbeated recently."""
        async with self._lock:
            stale = [nid for nid, n in self._nodes.items() if n.is_stale()]
            for nid in stale:
                logger.warning("Pruning stale node: %s", nid)
                del self._nodes[nid]

    def get_healthy(self, role: str = None) -> List[NodeInfo]:
        return [
            n for n in self._nodes.values()
            if n.healthy and (role is None or n.role == role)
        ]

    def snapshot(self) -> List[dict]:
        return [n.to_dict() for n in self._nodes.values()]


# Singleton registry
_registry = NodeRegistry()


def get_registry() -> NodeRegistry:
    return _registry


async def registry_maintenance_loop():
    """Background task: prune stale nodes every 15s."""
    while True:
        await asyncio.sleep(15)
        await _registry.prune_stale()
