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

## 🧠 Architecture Implementation
1. **User Query** -> Heuristic-based Neural Planner decomposes intent.
2. **Agent Planning** -> Generates steps for Retrieval, Causal Analysis, and Temporal Reasoning.
3. **Retrieval** -> Actual MultiDiGraph traversal via NetworkX.
4. **Reasoning** -> 
   - **Causal**: Real-time out-degree influence analysis.
   - **Temporal**: Event sequence detection using node timestamps.
5. **Synthesis** -> Dynamic answer generation with uncertainty estimation.
6. **Learning** -> Background GNN trainer (PyTorch Geometric) optimizes pattern weights.

---
Built with ❤️ by Antigravity
