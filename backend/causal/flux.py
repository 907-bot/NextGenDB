"""
Temporal Flux Tracking Engine — tracks rate of change and temporal signals in the graph.
"""
import logging
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timezone
import networkx as nx

logger = logging.getLogger("nextgendb.causal.flux")

class TemporalFluxEngine:
    """
    Tracks how node properties change over time and identifies "flux" (rate of change).
    This allows identifying leading indicators and lagging effects.
    """

    def __init__(self):
        # history: node_id -> { prop_name: [(timestamp, value)] }
        self._history: Dict[str, Dict[str, List[Tuple[float, Any]]]] = {}

    def record_state(self, node_id: str, properties: Dict[str, Any], timestamp: Optional[float] = None):
        """Record the current state of a node for flux calculation."""
        if timestamp is None:
            timestamp = datetime.now(timezone.utc).timestamp()
            
        if node_id not in self._history:
            self._history[node_id] = {}
            
        for key, value in properties.items():
            if not isinstance(value, (int, float)):
                continue
            if key not in self._history[node_id]:
                self._history[node_id][key] = []
            
            # Keep only last 100 observations to prevent memory leaks
            self._history[node_id][key].append((timestamp, value))
            if len(self._history[node_id][key]) > 100:
                self._history[node_id][key].pop(0)

    def calculate_flux(self, node_id: str, property_key: str) -> float:
        """Calculate the current 'flux' (dv/dt) for a property."""
        history = self._history.get(node_id, {}).get(property_key, [])
        if len(history) < 2:
            return 0.0
            
        # Linear regression / simple delta over last two points
        (t1, v1), (t2, v2) = history[-2], history[-1]
        dt = t2 - t1
        if dt < 1e-6:
            return 0.0
        return (v2 - v1) / dt

    def identify_leading_indicators(self, graph: nx.MultiDiGraph, target_node: str, target_prop: str) -> List[Dict[str, Any]]:
        """Find nodes whose flux precedes changes in the target node's flux."""
        target_flux = self.calculate_flux(target_node, target_prop)
        if abs(target_flux) < 1e-6:
            return []
            
        leading = []
        # Check predecessors in the graph
        for pred in graph.predecessors(target_node):
            for prop in self._history.get(pred, {}).keys():
                flux = self.calculate_flux(pred, prop)
                # If flux has same sign and significant magnitude
                if flux * target_flux > 0 and abs(flux) > 0.1:
                    leading.append({
                        "node": pred,
                        "property": prop,
                        "flux": flux,
                        "correlation": "positive" if flux * target_flux > 0 else "negative"
                    })
        
        return sorted(leading, key=lambda x: abs(x["flux"]), reverse=True)

    def get_temporal_signals(self, graph: nx.MultiDiGraph) -> Dict[str, Any]:
        """Summary of all significant flux signals currently active in the graph."""
        signals = []
        for node in graph.nodes:
            for prop in self._history.get(node, {}).keys():
                flux = self.calculate_flux(node, prop)
                if abs(flux) > 0.5:
                    signals.append({
                        "node": node,
                        "property": prop,
                        "flux": round(flux, 4),
                        "severity": "HIGH" if abs(flux) > 2.0 else "MEDIUM"
                    })
        return {
            "active_signals": signals,
            "signal_count": len(signals),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
