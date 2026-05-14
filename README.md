# NextGenDB: Neural Graph Intelligence Engine

NextGenDB is a futuristic, end-to-end AI database system that combines graph-based retrieval, causal/temporal reasoning, and GNN pattern learning to provide high-confidence answers to complex queries.

## 🚀 Key Features
- **Project Structure**: Clean separation of Agent, Graph, RAG, and GNN modules.
- **Neural Agent**: Two-stage planning and execution engine for query decomposition.
- **Vectorless Retrieval**: Direct graph traversal (MultiDiGraph) for context gathering.
- **Deep Reasoning**: Causal analysis and temporal flux tracking.
- **GNN Learning**: Asynchronous GCN model that learns graph patterns from query history.
- **Premium UI**: High-end React dashboard with glassmorphism, neon accents, and real-time reasoning logs.

## 🛠️ Tech Stack
- **Backend**: FastAPI, NetworkX, PyTorch, PyTorch Geometric.
- **Frontend**: React (Vite), Framer Motion, Lucide-React, Tailwind CSS.
- **Infrastructure**: Dockerized backend, Google Stitch design system.

## 🚦 Getting Started

### Quick Start (Local)
1. `pip install -r requirements.txt`
2. `python -m backend.main`
3. `cd frontend && npm install && npm run dev`

### Production Deployment (Docker)
```bash
docker-compose up --build
```

## 🧠 Architecture Implementation (All 10 Layers)
1. **User Query (L1/L2)** -> Heuristic-based Neural Planner decomposes intent.
2. **Agent Planning (L2)** -> Generates steps for Retrieval, Causal Analysis, and Temporal Reasoning.
3. **Retrieval (L3/L5)** -> Actual MultiDiGraph traversal via NetworkX.
4. **Reasoning (L4)** -> 
   - **Causal**: Real-time out-degree influence analysis.
   - **Temporal**: Event sequence detection using node timestamps.
5. **Synthesis (L2)** -> Dynamic answer generation with uncertainty estimation.
6. **Learning (L6)** -> Background GNN trainer (PyTorch Geometric) optimizes pattern weights.
7. **Storage (L7)** -> In-memory MultiDiGraph (Extensible to Neo4j/ArangoDB).
8. **Streaming (L8)** -> Kafka-integrated ingestion pipeline with HTTP fallback for real-time mutations.
9. **Distributed (L9)** -> Node registry and health probes for Kubernetes orchestration.
10. **Observability (L10)** -> Prometheus metrics, structured JSON logging, and OpenTelemetry tracing.

## 🛠️ Production Ready Features
- **Dockerized Stack**: Full orchestration including Kafka, Prometheus, and Grafana.
- **Observability Dashboard**: Integrated live monitoring panel for all 10 layers.
- **Self-Healing**: Kubernetes-ready liveness and readiness probes.
- **Neural Dashboard**: Real-time SVG topology visualization of the neural graph.

---
Built with ❤️ by Antigravity
