"""
ACID Transaction Manager — MVCC-based multi-version concurrency control.

Each transaction gets a monotonically increasing transaction ID (tx_id).
Reads use a snapshot of the committed version map at transaction start.
Writes buffer changes locally until COMMIT; ABORT discards them.
"""
import logging
import threading
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger("nextgendb.storage.txn")


class TxStatus(str, Enum):
    ACTIVE    = "ACTIVE"
    COMMITTED = "COMMITTED"
    ABORTED   = "ABORTED"


@dataclass
class VersionedValue:
    tx_id:   str
    value:   Any
    deleted: bool = False


@dataclass
class Transaction:
    tx_id:      str
    snapshot:   Dict[str, Any]   # committed state at tx start
    writes:     Dict[str, Any] = field(default_factory=dict)
    deletes:    Set[str]        = field(default_factory=set)
    status:     TxStatus        = TxStatus.ACTIVE
    write_set:  List[str]       = field(default_factory=list)  # for conflict detection
    start_idx:  int             = 0                            # index in committed list at start

    def read(self, key: str) -> Any:
        """Read using write-set first (read-your-writes), then snapshot."""
        if key in self.deletes:
            return None
        if key in self.writes:
            return self.writes[key]
        return self.snapshot.get(key)

    def write(self, key: str, value: Any):
        self.writes[key] = value
        self.deletes.discard(key)
        if key not in self.write_set:
            self.write_set.append(key)

    def delete(self, key: str):
        self.deletes.add(key)
        self.writes.pop(key, None)


class MVCCManager:
    """
    Multi-Version Concurrency Control manager.

    Maintains a global committed-state dictionary (the 'store') and issues
    snapshot-isolated transactions.  Conflict detection uses a simple
    first-committer-wins optimistic strategy.
    """

    def __init__(self):
        self._lock        = threading.RLock()
        self._store: Dict[str, Any]     = {}          # committed state
        self._active_txs: Dict[str, Transaction] = {}
        # Stores write sets of committed transactions: list of (tx_id, set_of_keys)
        self._committed_writes: List[Tuple[str, Set[str]]] = []

    # ── Transaction lifecycle ──────────────────────────────────────────────────

    def begin(self) -> Transaction:
        with self._lock:
            tx_id    = str(uuid.uuid4())
            snapshot = dict(self._store)             # copy-on-begin snapshot
            tx = Transaction(tx_id=tx_id, snapshot=snapshot, start_idx=len(self._committed_writes))
            self._active_txs[tx_id] = tx
            logger.debug("TX %s started (snapshot size=%d)", tx_id[:8], len(snapshot))
            return tx

    def commit(self, tx: Transaction) -> bool:
        """
        Commit the transaction.  Returns True on success, False on conflict.
        Uses optimistic conflict detection: abort if any key in write_set
        was modified by a concurrent committed transaction.
        """
        with self._lock:
            if tx.status != TxStatus.ACTIVE:
                raise RuntimeError(f"Cannot commit tx in state {tx.status}")

            # Conflict check: find commits that happened after this tx began
            concurrent_writes: Set[str] = set()
            for committed_tx_id, writes in self._committed_writes[tx.start_idx:]:
                concurrent_writes |= writes

            conflicts = set(tx.write_set) & concurrent_writes
            if conflicts:
                logger.warning("TX %s aborted — write conflict on keys: %s", tx.tx_id[:8], conflicts)
                self._abort_internal(tx)
                return False

            # Apply writes
            self._store.update(tx.writes)
            for key in tx.deletes:
                self._store.pop(key, None)

            tx.status = TxStatus.COMMITTED
            self._committed_writes.append((tx.tx_id, set(tx.write_set)))
            # Clean up active tx map
            self._active_txs.pop(tx.tx_id, None)
            
            # Prune committed list (keep last 1000)
            if len(self._committed_writes) > 1000:
                self._committed_writes = self._committed_writes[-1000:]
            logger.debug("TX %s committed (%d writes, %d deletes)", tx.tx_id[:8], len(tx.writes), len(tx.deletes))
            return True

    def abort(self, tx: Transaction):
        with self._lock:
            self._abort_internal(tx)

    def _abort_internal(self, tx: Transaction):
        tx.status = TxStatus.ABORTED
        self._active_txs.pop(tx.tx_id, None)
        logger.debug("TX %s aborted", tx.tx_id[:8])

    # ── Context manager ────────────────────────────────────────────────────────

    @contextmanager
    def transaction(self):
        """Context manager: auto-commits on exit, rolls back on exception."""
        tx = self.begin()
        try:
            yield tx
            success = self.commit(tx)
            if not success:
                raise RuntimeError("Transaction aborted due to write conflict")
        except Exception:
            self.abort(tx)
            raise

    # ── Read helpers ───────────────────────────────────────────────────────────

    def get(self, key: str) -> Any:
        """Read committed value (no transaction context)."""
        return self._store.get(key)

    def snapshot(self) -> Dict[str, Any]:
        """Return an immutable snapshot of the committed store."""
        with self._lock:
            return dict(self._store)
