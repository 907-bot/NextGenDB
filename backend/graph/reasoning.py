import networkx as nx
from typing import List, Dict, Any

class CausalEngine:
    def analyze(self, graph: nx.MultiDiGraph) -> Dict[str, Any]:
        # Analyze directional edges to find potential "roots"
        if not graph.nodes:
            return {"root_causes": ["None"], "impact_chain": ["Stable State"]}
            
        # Simplistic: Nodes with high out-degree are potential causes
        out_degrees = dict(graph.out_degree())
        potential_causes = sorted(out_degrees.items(), key=lambda x: x[1], reverse=True)
        main_cause = potential_causes[0][0] if potential_causes else "Unknown"
        
        return {
            "root_causes": [f"Influence detected at: {main_cause}"],
            "impact_chain": [f"{main_cause} -> " + " -> ".join(list(graph.successors(main_cause))[:2])],
            "counterfactuals": [f"Disrupting {main_cause} would ripple through {len(list(graph.successors(main_cause)))} downstream nodes"]
        }

class TemporalEngine:
    def detect_sequence(self, graph: nx.MultiDiGraph) -> List[Dict[str, Any]]:
        # Sort nodes by 'timestamp' property if available
        events = []
        for node, data in graph.nodes(data=True):
            if 'timestamp' in data:
                events.append({"event": data.get('label', node), "timestamp": data['timestamp']})
        
        if not events:
            # Fallback mock if no timestamps
            return [{"event": "Genesis", "timestamp": "2024-05-11T00:00:00Z"}]
            
        return sorted(events, key=lambda x: x['timestamp'])
