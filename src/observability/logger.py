"""
Structured pipeline events: printed as JSON log lines and persisted to pipeline_events.

Persistence runs on one background thread. log_event is called from request handlers on the
event loop, and a synchronous insert there would stall every concurrent request for a full
database round trip (§1.2C of the redesign plan). Events are queued and written in batches.
Delivery is best-effort: events still queued when the process is killed are lost; close()
flushes them on a normal shutdown.
"""
import json
import logging
import queue
import threading
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import insert
from sqlalchemy.engine import Engine

from src.registry.schema import pipeline_events

MAX_BATCH_SIZE = 100
_STOP = object()


class PipelineLogger:
    def __init__(self, name: str, engine: Optional[Engine] = None):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(message)s"))
            self.logger.addHandler(handler)

        self._engine = engine
        self._queue: "queue.SimpleQueue" = queue.SimpleQueue()
        self._writer: Optional[threading.Thread] = None
        if engine is not None:
            self._writer = threading.Thread(target=self._write_loop, name="pipeline-events", daemon=True)
            self._writer.start()

    def log_event(self, event: str, **kwargs) -> None:
        timestamp = datetime.now(timezone.utc)
        details = json.loads(json.dumps(kwargs, default=str))
        self.logger.info(json.dumps({"event": event, "timestamp": timestamp.isoformat(), **details}))

        if self._writer is not None:
            self._queue.put(
                {
                    "event": event,
                    "timestamp": timestamp,
                    "tenant_id": details.get("tenant_id"),
                    "query_id": details.get("query_id"),
                    "job_id": details.get("job_id"),
                    "details": details,
                }
            )

    def close(self, timeout: float = 5.0) -> None:
        """Flushes queued events and stops the writer thread."""
        if self._writer is not None:
            self._queue.put(_STOP)
            self._writer.join(timeout)
            self._writer = None

    def _write_loop(self) -> None:
        while True:
            item = self._queue.get()
            if item is _STOP:
                return
            batch = [item]
            stop = False
            while len(batch) < MAX_BATCH_SIZE:
                try:
                    nxt = self._queue.get_nowait()
                except queue.Empty:
                    break
                if nxt is _STOP:
                    stop = True
                    break
                batch.append(nxt)
            try:
                with self._engine.begin() as conn:
                    conn.execute(insert(pipeline_events), batch)
            except Exception as e:
                self.logger.warning(f"PipelineLogger failed to persist {len(batch)} events: {e}")
            if stop:
                return
