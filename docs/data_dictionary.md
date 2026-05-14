# NextGenDB Data Dictionary

This document defines the core node and edge types utilized in NextGenDB, specifically focusing on the initial seeded demo schema and standard AI-native graph representations.

## 1. Node Types

Nodes represent entities within the graph. Each node has a unique `id`, a `type` for classification, and arbitrary key-value `properties`.

| Node Type | Description | Key Properties | Example Use Case |
| :--- | :--- | :--- | :--- |
| `SYSTEM` | Core infrastructure components or top-level systems. | `status` (String), `score` (Float) | Root intelligence engine tracking. |
| `PROCESS` | Active computational pipelines or tasks. | `load` (Float), `efficiency` (Float) | Tracking data ingestion or query planning load. |
| `SERVICE` | Microservices or modular backend components. | `version` (String), `dim` (Int) | Representing vector indexes or inference engines. |
| `EVENT` | Discrete occurrences in time, such as anomalies or signals. | `timestamp` (ISO8601), `severity` (String) | Tracking churn signals, pricing changes, or errors. |
| `USER` | Human or service accounts interacting with the system. | `plan` (String) | Customer profiles for RBAC or personalization. |
| `PRODUCT` | Items or services available for purchase/interaction. | `price` (Float) | Enterprise DB plans, Vector search modules. |
| `TRANSACTION` | Financial or state-changing operations. | `amount` (Float), `timestamp` (ISO8601) | Purchase logs for causal revenue tracking. |

---

## 2. Edge Types (Relationships)

Edges connect two nodes and represent directional relationships. They can optionally contain properties (like weights or timestamps).

| Edge Type | Source Node | Target Node | Description | Properties |
| :--- | :--- | :--- | :--- | :--- |
| `MANAGES` | `SYSTEM` | `PROCESS` | Indicates ownership of a pipeline. | - |
| `ORCHESTRATES`| `SYSTEM` | `SERVICE` | Top-level direction of microservices. | - |
| `FEEDS` | `PROCESS` | `SERVICE`/`PROCESS` | Data flow from one component to another. | `throughput` (Float) |
| `TRIGGERS` | `EVENT` | `PROCESS` | An event initiates a process. | - |
| `CONSULTS` | `PROCESS` | `SERVICE` | Synchronous request for computation/data. | `latency_ms` (Float) |
| `DELEGATES` | `PROCESS` | `PROCESS` | Task offloading to specialized processes. | - |
| `ALERTS` | `SERVICE` | `SYSTEM` | Warning signals sent upstream. | `level` (String) |
| `INFORMS` | `SERVICE` | `SERVICE` | Knowledge transfer between AI models. | `confidence` (Float)|
| `PROVIDES_CONTEXT`| `SERVICE` | `PROCESS` | RAG/Agentic memory feeding into logic. | - |
| `BOUGHT` | `USER` | `PRODUCT` | E-commerce transaction indicator. | `ts` (ISO8601) |
| `BELONGS_TO` | `TRANSACTION`| `USER` | Reconciles transactions to owners. | - |
| `FOR` | `TRANSACTION`| `PRODUCT` | Associates a transaction to the item. | - |
| `AFFECTS` | `EVENT` | `PRODUCT`/`USER`| Direct consequence of an event. | - |
| `CAUSES` | `EVENT` | `EVENT` | Causal chain link between events. | `probability` (Float)|

---

## 3. System Properties

By default, the core engine manages the following implicit or explicit properties:
*   **`_id`**: A globally unique identifier (string) for nodes.
*   **`vector`**: An optional floating-point array for nodes (used by the hybrid search engine).
*   **`tx_id`**: (Internal) Associated with WAL records during ACID commits.
