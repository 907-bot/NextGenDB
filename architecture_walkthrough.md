# NextGenDB: Industrialised Architecture Walkthrough

The NextGenDB prototype has been transformed from a static shell into a production-grade, 15-layer neural graph intelligence engine. This document outlines the key subsystems and how they interact.

## 🏗️ The 15-Layer Stack

1.  **L1: Persistent WAL Storage**: Append-only Write-Ahead Log ensures crash-consistency.
2.  **L2: MVCC Transaction Manager**: Multi-Version Concurrency Control with ACID guarantees.
3.  **L3: Cypher + SQL Engine**: Hybrid query language support for graph and relational patterns.
4.  **L4: Property Indexing**: O(1) attribute lookup via the `PersistentGraphEngine`.
5.  **L5: Hybrid Vector Search**: Fusion of dense (embeddings) and sparse (BM25) search.
6.  **L6: Probabilistic Causal Inference**: Structural causal analysis using PC-algorithm signals.
7.  **L7: Agentic Memory Store**: 4-layer memory (episodic, semantic, procedural, working).
8.  **L8: RBAC Security**: Role-Based Access Control and JWT-based authentication.
9.  **L9: LDBC Benchmark Suite**: Reproducible performance testing for graph workloads.
10. **L10: Neural Query Agent**: Two-stage "Decompose + Plan" orchestration pipeline.
11. **L11: Async GNN Learner**: Continuous background training for graph-native embeddings.
12. **L12: Temporal Flux Tracking**: Rate-of-change analysis for predictive leading indicators.
13. **L13: Kafka Streaming**: High-throughput ingestion and mutation event pipeline.
14. **L14: Distributed Registry**: Service discovery for horizontal scaling.
15. **L15: Full Observability**: Prometheus metrics, Grafana dashboards, and OTel tracing.

---

## 🧠 Key Subsystems

### 1. Neural Agent (Decompose + Plan)
Located in `backend/agent/neural_agent.py`. It breaks down natural language queries into a dependency graph of sub-goals.
- **Decomposer**: Identifies intents (Causal, Temporal, Vector, etc.).
- **Planner**: Maps sub-goals to engine operations in a prioritized queue.

### 2. Async GNN Learner
Located in `backend/gnn/learner.py`. It runs a continuous training loop in the background.
- **Vectorless Retrieval**: Uses betweenness centrality and BFS to find relevant subgraphs without needing a pre-built vector index.
- **Async Training**: Periodically refines node embeddings based on graph structure changes.

### 3. Temporal Flux Engine
Located in `backend/causal/flux.py`. It monitors the "velocity" of data changes.
- **Signal Detection**: Identifies nodes that act as leading indicators for downstream effects.
- **Flux Analysis**: Calculates `dv/dt` for numeric node properties during ingestion.

---

## 🚀 Execution Flow

1.  **Ingest**: Kafka events enter via `GraphIngestionHandler`, recording flux signals and writing to WAL.
2.  **Query**: A user query hits the `/api/v1/query` endpoint.
3.  **Plan**: The `NeuralAgentPlanner` decomposes the query into sub-goals.
4.  **Retrieve**: The `VectorSearchEngine` (hybrid) and `GNNLearner` (vectorless) pull relevant context.
5.  **Reason**: The `CausalInferenceEngine` and `TemporalEngine` analyze the retrieved subgraph.
6.  **Synthesize**: The agent combines all evidence into a high-confidence natural language response.
7.  **Train**: In the background, the GNN continues to learn from the new mutations.

---

## 🛠️ Developer Commands

- **Run Backend**: `python -m backend.main`
- **Run Tests**: `pytest tests/`
- **Benchmarking**: `python -m backend.benchmark.suite`
- **Docker Production**: `docker-compose -f docker-compose.prod.yml up --build`
