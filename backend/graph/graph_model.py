import networkx as nx
from typing import List, Dict, Any, Optional
import datetime

class GraphModel:
    def __init__(self):
        self.graph = nx.MultiDiGraph()
    
    def add_node(self, node_id: str, properties: Dict[str, Any]):
        self.graph.add_node(node_id, **properties)
    
    def add_edge(self, source: str, target: str, edge_type: str, properties: Dict[str, Any] = None):
        self.graph.add_edge(source, target, key=edge_type, **(properties or {}))
    
    def get_subgraph(self, center_node: str, radius: int = 2) -> nx.MultiDiGraph:
        if center_node not in self.graph:
            return nx.MultiDiGraph()
        nodes = nx.single_source_shortest_path_length(self.graph, center_node, cutoff=radius).keys()
        return self.graph.subgraph(nodes)

    def seed_data(self):
        """Populate the graph with initial demo data."""
        # Nodes
        self.add_node("Root_System", {"label": "Core Intelligence", "type": "SYSTEM", "status": "ACTIVE"})
        self.add_node("Data_Ingest", {"label": "Data Ingestion Pipeline", "type": "PROCESS", "load": 0.45})
        self.add_node("Neural_Core", {"label": "Neural Processing Unit", "type": "PROCESS", "efficiency": 0.98})
        self.add_node("User_Query_Alpha", {"label": "Query: Energy Optimization", "type": "EVENT", "timestamp": "2024-05-10T14:30:00Z"})
        self.add_node("Anomaly_Detector", {"label": "Temporal Anomaly Detector", "type": "SERVICE", "version": "1.2.0"})
        
        # Edges
        self.add_edge("Root_System", "Data_Ingest", "MANAGES")
        self.add_edge("Root_System", "Neural_Core", "ORCHESTRATES")
        self.add_edge("Data_Ingest", "Neural_Core", "FEEDS")
        self.add_edge("User_Query_Alpha", "Neural_Core", "TRIGGERS")
        self.add_edge("Neural_Core", "Anomaly_Detector", "CONSULTS")
        self.add_edge("Anomaly_Detector", "Root_System", "ALERTS")

    def to_json(self):
        # Convert MultiDiGraph to a format React components can consume easily
        data = nx.node_link_data(self.graph)
        # Ensure nodes have 'id' which is often expected by graph libs
        for node in data['nodes']:
            node['id'] = node.get('id') or node.get('name')
        return data
