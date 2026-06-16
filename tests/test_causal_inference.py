"""Tests for Causal Inference Engine"""
import pytest
import networkx as nx

from backend.causal.inference import CausalInferenceEngine
from backend.causal.flux import TemporalFluxEngine


class TestCausalInferenceEngine:
    """Causal discovery and inference"""

    def test_pc_algorithm_simple_dag(self):
        """Should discover causal relationships"""
        engine = CausalInferenceEngine()
        
        # Create synthetic data: X -> Y -> Z
        data = {
            "X": [1, 2, 3, 4, 5],
            "Y": [2, 4, 6, 8, 10],  # Y = 2*X
            "Z": [4, 8, 12, 16, 20]  # Z = 2*Y
        }
        
        dag = engine.discover_causal_structure(data)
        
        assert dag is not None
        assert isinstance(dag, nx.DiGraph)
        # Should have edges representing causal relationships
        assert dag.number_of_nodes() > 0

    def test_backdoor_criterion(self):
        """Should identify confounders via backdoor criterion"""
        engine = CausalInferenceEngine()
        
        # Create graph: C -> X -> Y (C is confounder)
        g = nx.DiGraph()
        g.add_edges_from([("C", "X"), ("C", "Y"), ("X", "Y")])
        
        # Control for C to block backdoor path
        controlling = engine.find_confounders(g, "X", "Y")
        assert "C" in controlling or len(controlling) > 0

    def test_treatment_effect_estimation(self):
        """Should estimate treatment effects"""
        engine = CausalInferenceEngine()
        
        # Simple treatment data
        treated = [10, 12, 11, 13, 14]  # Treatment group
        control = [5, 6, 5, 7, 6]       # Control group
        
        ate = engine.estimate_ate(treated, control)
        
        assert ate > 0  # Treatment has positive effect
        assert ate < 10  # Effect is reasonable

    def test_causal_graph_to_json(self):
        """Should serialize causal graphs"""
        engine = CausalInferenceEngine()
        
        g = nx.DiGraph()
        g.add_edges_from([("A", "B"), ("B", "C")])
        
        json_repr = engine.graph_to_json(g)
        
        assert json_repr is not None
        assert "nodes" in json_repr
        assert "edges" in json_repr


class TestTemporalFluxEngine:
    """Temporal rate-of-change analysis"""

    def test_velocity_calculation(self):
        """Should calculate rate of change"""
        engine = TemporalFluxEngine()
        
        # Linear increase: 1, 2, 3, 4, 5
        values = [1, 2, 3, 4, 5]
        timestamps = [1, 2, 3, 4, 5]
        
        flux = engine.calculate_flux(values, timestamps)
        
        assert flux > 0
        assert abs(flux - 1.0) < 0.1  # Slope should be ~1

    def test_anomaly_detection_in_flux(self):
        """Should detect anomalies in flux"""
        engine = TemporalFluxEngine()
        
        # Normal: [1, 2, 3, 4, 100, 6, 7]  -> spike at index 4
        values = [1, 2, 3, 4, 100, 6, 7]
        
        anomalies = engine.detect_flux_anomalies(values)
        
        assert len(anomalies) > 0
        assert 4 in anomalies or any(idx in anomalies for idx in [3, 4, 5])

    def test_leading_indicator_detection(self):
        """Should identify leading indicators"""
        engine = TemporalFluxEngine()
        
        # Signal X leads outcome Y
        x_signal = [1, 2, 3, 4, 5]
        y_outcome = [0, 1, 2, 3, 4]  # Delayed by 1 step
        
        is_leading = engine.is_leading_indicator(x_signal, y_outcome, lag=1)
        
        assert is_leading
