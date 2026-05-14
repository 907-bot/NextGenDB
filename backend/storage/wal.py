"""
Write-Ahead Log (WAL) — Crash-safe append-only transaction log.

Every write operation is first committed to this log.  On startup the engine
replays any uncommitted records to restore a consistent state.

File format  (per record):
    [4B CRC32][4B payload_len][payload JSON]
"""
import hashlib
import json
import logging
import os
import struct
import zlib
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Iterator, List, Optional

logger = logging.getLogger("nextgendb.storage.wal")

_HEADER_FMT = "!II"          # big-endian: uint32 crc, uint32 length
_HEADER_SZ  = struct.calcsize(_HEADER_FMT)


class WALCorruptError(RuntimeError):
    """Raised when a WAL record fails its checksum."""


class WALRecord:
    __slots__ = ("lsn", "op", "entity_type", "entity_id", "data", "ts", "tx_id")

    def __init__(
        self,
        lsn: int,
        op: str,
        entity_type: str,
        entity_id: str,
        data: Dict[str, Any],
        tx_id: str,
        ts: Optional[str] = None,
    ):
        self.lsn         = lsn
        self.op          = op           # ADD_NODE | ADD_EDGE | DEL_NODE | DEL_EDGE | COMMIT | ABORT
        self.entity_type = entity_type  # node | edge | transaction
        self.entity_id   = entity_id
        self.data        = data
        self.tx_id       = tx_id
        self.ts          = ts or datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__slots__}


class WriteAheadLog:
    """Thread-safe, append-only Write-Ahead Log."""

    def __init__(self, path: str | Path = "data/wal.log"):
        self._path  = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock  = Lock()
        self._lsn   = 0
        self._fh    = None
        self._open()
        # Fast-forward LSN counter to last known record
        for rec in self._scan():
            self._lsn = rec.lsn
        logger.info("WAL initialised at %s — last LSN=%d", self._path, self._lsn)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _open(self):
        self._fh = open(self._path, "ab+")

    @staticmethod
    def _encode(payload: dict) -> bytes:
        raw   = json.dumps(payload, separators=(",", ":")).encode()
        crc   = zlib.crc32(raw) & 0xFFFFFFFF
        hdr   = struct.pack(_HEADER_FMT, crc, len(raw))
        return hdr + raw

    @staticmethod
    def _decode_one(fh) -> Optional[WALRecord]:
        hdr = fh.read(_HEADER_SZ)
        if not hdr:
            return None
        if len(hdr) < _HEADER_SZ:
            raise WALCorruptError("Truncated WAL header")
        crc_expected, length = struct.unpack(_HEADER_FMT, hdr)
        raw = fh.read(length)
        if len(raw) != length:
            raise WALCorruptError("Truncated WAL payload")
        crc_actual = zlib.crc32(raw) & 0xFFFFFFFF
        if crc_actual != crc_expected:
            raise WALCorruptError(
                f"WAL checksum mismatch: expected={crc_expected} got={crc_actual}"
            )
        payload = json.loads(raw)
        return WALRecord(**payload)

    def _scan(self) -> Iterator[WALRecord]:
        """Read all valid records from the beginning of the log file."""
        try:
            with open(self._path, "rb") as fh:
                while True:
                    rec = self._decode_one(fh)
                    if rec is None:
                        break
                    yield rec
        except WALCorruptError as exc:
            logger.error("WAL corruption detected during scan: %s", exc)

    # ── Public API ────────────────────────────────────────────────────────────

    def append(
        self,
        op: str,
        entity_type: str,
        entity_id: str,
        data: Dict[str, Any],
        tx_id: str,
    ) -> WALRecord:
        with self._lock:
            self._lsn += 1
            rec = WALRecord(
                lsn=self._lsn,
                op=op,
                entity_type=entity_type,
                entity_id=entity_id,
                data=data,
                tx_id=tx_id,
            )
            self._fh.write(self._encode(rec.to_dict()))
            self._fh.flush()
            os.fsync(self._fh.fileno())
            return rec

    def replay(self) -> List[WALRecord]:
        """Return all records — used by the storage engine on startup."""
        return list(self._scan())

    def checkpoint(self, up_to_lsn: int):
        """Truncate the WAL after a successful checkpoint to free disk space."""
        with self._lock:
            records = [r for r in self._scan() if r.lsn > up_to_lsn]
            self._fh.close()
            self._path.write_bytes(b"".join(self._encode(r.to_dict()) for r in records))
            self._open()
            logger.info("WAL checkpointed — retained %d records after LSN=%d", len(records), up_to_lsn)

    def close(self):
        with self._lock:
            if self._fh:
                self._fh.close()
