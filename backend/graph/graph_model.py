import networkx as nx
from typing import List, Dict, Any, Optional
import datetime

import networkx as nx
from typing import List, Dict, Any, Optional
import datetime
import logging

from ..storage.engine import PersistentGraphEngine

logger = logging.getLogger("nextgendb.graph.model")

class GraphModel:
    """
    Bridge class that maintains backward compatibility with v1 routes
    while using the PersistentGraphEngine for actual storage.
    """
    def __init__(self, engine: Optional[PersistentGraphEngine] = None):
        self.engine = engine or PersistentGraphEngine()
        # Keep a reference to the graph for easy access
        self.graph = self.engine.graph
    
    def add_node(self, node_id: str, properties: Dict[str, Any], tx_id: Optional[str] = None):
        self.engine.add_node(node_id, properties, tx_id=tx_id)
    
    def add_edge(self, source: str, target: str, edge_type: str, properties: Dict[str, Any] = None, tx_id: Optional[str] = None):
        self.engine.add_edge(source, target, edge_type, properties, tx_id=tx_id)
    
    def get_subgraph(self, center_node: str, radius: int = 2) -> nx.MultiDiGraph:
        return self.engine.get_subgraph(center_node, radius)

    def seed_data(self):
        """Seed demo data if the engine is empty."""
        if self.graph.number_of_nodes() == 0:
            logger.info("Seeding initial graph data...")
            # We use the engine's internal _seed_demo logic or implement it here
            from ..main import _seed_demo
            _seed_demo(self.engine)

    def to_json(self):
        return self.engine.to_json()

    def find_by_property(self, prop: str, value: Any) -> List[str]:
        return self.engine.find_by_property(prop, value)

