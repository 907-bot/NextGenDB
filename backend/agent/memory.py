"""
Agentic Memory Store — Long-term memory for AI agents.

Memory layers:
  • Procedural  — learned skills / tool-use patterns
  • Episodic    — past interaction experiences with timestamps
  • Semantic    — factual knowledge extracted from documents/queries
  • Working     — short-term scratch pad, cleared per session

Compatible with LangGraph / LangChain memory interfaces.
"""
import json
import logging
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("nextgendb.agent.memory")


@dataclass
class MemoryEntry:
    memory_id:   str
    layer:       str          # procedural | episodic | semantic | working
    content:     str
    metadata:    Dict[str, Any]
    created_at:  str
    importance:  float = 1.0  # higher = retained longer
    access_count: int = 0

    def touch(self):
        self.access_count += 1


class AgenticMemoryStore:
    """
    Persistent multi-layer memory store for AI agents.
    Backed by the PersistentGraphEngine as nodes with 'MEMORY_*' labels.
    """

    LAYERS = ("procedural", "episodic", "semantic", "working")

    def __init__(self, engine=None, persist_path: Path = Path("data/agent_memory.json")):
        self._engine = engine
        self._persist = persist_path
        self._store: Dict[str, MemoryEntry] = {}
        self._load()

    # ── CRUD ──────────────────────────────────────────────────────────────────

    def remember(
        self,
        content: str,
        layer: str = "semantic",
        metadata: Dict = None,
        importance: float = 1.0,
    ) -> MemoryEntry:
        if layer not in self.LAYERS:
            raise ValueError(f"Unknown memory layer: {layer}")
        entry = MemoryEntry(
            memory_id=str(uuid.uuid4()),
            layer=layer,
            content=content,
            metadata=metadata or {},
            created_at=datetime.now(timezone.utc).isoformat(),
            importance=importance,
        )
        self._store[entry.memory_id] = entry
        # Persist to graph engine if available
        if self._engine:
            self._engine.add_node(
                f"MEM_{entry.memory_id}",
                {"label": content[:80], "type": f"MEMORY_{layer.upper()}", **entry.metadata},
            )
        self._save()
        logger.debug("Memory stored [%s]: %s…", layer, content[:60])
        return entry

    def recall(
        self,
        query: str,
        layer: Optional[str] = None,
        top_k: int = 5,
    ) -> List[MemoryEntry]:
        """Simple keyword recall — integrate with VectorSearchEngine for semantic recall."""
        results = []
        for entry in self._store.values():
            if layer and entry.layer != layer:
                continue
            if query.lower() in entry.content.lower():
                entry.touch()
                results.append(entry)
        results.sort(key=lambda e: e.importance * (1 + e.access_count * 0.1), reverse=True)
        return results[:top_k]

    def recall_episodic(self, since_iso: Optional[str] = None) -> List[MemoryEntry]:
        entries = [e for e in self._store.values() if e.layer == "episodic"]
        if since_iso:
            entries = [e for e in entries if e.created_at >= since_iso]
        return sorted(entries, key=lambda e: e.created_at, reverse=True)

    def forget(self, memory_id: str):
        self._store.pop(memory_id, None)
        self._save()

    def clear_working(self):
        working = [k for k, v in self._store.items() if v.layer == "working"]
        for k in working:
            del self._store[k]
        self._save()

    def consolidate(self):
        """
        Move high-importance working memories to episodic/semantic.
        Called at end of agent session.
        """
        promoted = 0
        for entry in list(self._store.values()):
            if entry.layer == "working" and entry.importance >= 0.7:
                entry.layer = "episodic"
                promoted += 1
        self.clear_working()
        self._save()
        logger.info("Memory consolidation: %d working → episodic", promoted)

    def stats(self) -> Dict:
        counts = {layer: 0 for layer in self.LAYERS}
        for e in self._store.values():
            counts[e.layer] = counts.get(e.layer, 0) + 1
        return {"total": len(self._store), "by_layer": counts}

    # ── Persistence ───────────────────────────────────────────────────────────

    def _save(self):
        self._persist.parent.mkdir(parents=True, exist_ok=True)
        data = {k: asdict(v) for k, v in self._store.items()}
        self._persist.write_text(json.dumps(data, default=str))

    def _load(self):
        if not self._persist.exists():
            return
        try:
            data = json.loads(self._persist.read_text())
            self._store = {k: MemoryEntry(**v) for k, v in data.items()}
            logger.info("Agent memory loaded: %d entries", len(self._store))
        except Exception as exc:
            logger.warning("Memory load failed: %s", exc)
