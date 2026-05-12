"""
Layer 10 — Structured Logging & OpenTelemetry Tracing
Provides JSON-structured logs (ELK-compatible) and
OpenTelemetry trace spans for every query pipeline step.
"""
import logging
import json
import sys
import time
from contextlib import contextmanager
from typing import Any, Dict, Optional


# ── JSON Formatter (ELK / Loki compatible) ───────────────────────────────────
class JSONFormatter(logging.Formatter):
    """Outputs log records as single-line JSON — ready for Elasticsearch or Loki."""

    def format(self, record: logging.LogRecord) -> str:
        log_obj = {
            "ts":       self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level":    record.levelname,
            "logger":   record.name,
            "msg":      record.getMessage(),
            "module":   record.module,
            "line":     record.lineno,
        }
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_obj)


def setup_logging(level: str = "INFO", json_output: bool = True):
    """
    Configure root logger.
    Set json_output=False for human-readable output during local development.
    """
    handler = logging.StreamHandler(sys.stdout)
    if json_output:
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("[%(asctime)s] %(levelname)-8s %(name)s — %(message)s")
        )
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    logging.getLogger("uvicorn.access").propagate = False  # prevent double logs


# ── OpenTelemetry Tracing ────────────────────────────────────────────────────
_OTEL_AVAILABLE = False
_tracer = None

try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import (
        BatchSpanProcessor,
        ConsoleSpanExporter,
    )
    provider = TracerProvider()
    # Console exporter — swap for OTLPExporter (Jaeger/Tempo) in production
    provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
    trace.set_tracer_provider(provider)
    _tracer = trace.get_tracer("nextgendb")
    _OTEL_AVAILABLE = True
    logging.getLogger("nextgendb.monitoring.tracing").info("OpenTelemetry tracing: ENABLED")
except ImportError:
    logging.getLogger("nextgendb.monitoring.tracing").warning(
        "opentelemetry-sdk not installed — using no-op tracer."
    )


# ── Public tracing context managers ─────────────────────────────────────────

@contextmanager
def trace_span(name: str, attributes: Optional[Dict[str, Any]] = None):
    """
    Context manager that creates an OTel span if available,
    otherwise acts as a lightweight timer-only span.
    """
    if _OTEL_AVAILABLE and _tracer:
        with _tracer.start_as_current_span(name) as span:
            if attributes:
                for k, v in attributes.items():
                    span.set_attribute(k, str(v))
            yield span
    else:
        # No-op fallback with timing
        _t = time.perf_counter()
        yield None
        elapsed = time.perf_counter() - _t
        logging.getLogger("nextgendb.trace").debug(
            "SPAN [%s] %.3fms", name, elapsed * 1000
        )


def trace_query_pipeline(query: str, plan_steps: int):
    return trace_span("query_pipeline", {
        "query":      query,
        "plan_steps": plan_steps,
    })


def trace_graph_retrieval(center_node: str, radius: int):
    return trace_span("graph_retrieval", {
        "center_node": center_node,
        "radius":      radius,
    })


def trace_gnn_step(nodes: int):
    return trace_span("gnn_training_step", {"graph_nodes": nodes})
