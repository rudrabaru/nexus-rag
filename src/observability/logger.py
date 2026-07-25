import logging
import json
from datetime import datetime, timezone
from typing import Optional

class PipelineLogger:
    def __init__(self, name: str, registry=None):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        self.registry = registry
        
        # Add stream handler if not already added to prevent duplicates
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter('%(message)s'))
            self.logger.addHandler(handler)

    def log_event(self, event: str, **kwargs):
        timestamp = datetime.now(timezone.utc).isoformat()
        record = {
            "event": event,
            "timestamp": timestamp,
            **kwargs
        }
        self.logger.info(json.dumps(record))

        if self.registry:
            try:
                with self.registry._get_conn() as conn:
                    tenant_id = kwargs.get("tenant_id")
                    query_id = kwargs.get("query_id")
                    job_id = kwargs.get("job_id")
                    conn.execute(
                        "INSERT INTO pipeline_logs (event, timestamp, tenant_id, query_id, job_id, details) VALUES (?, ?, ?, ?, ?, ?)",
                        (event, timestamp, tenant_id, query_id, job_id, json.dumps(kwargs))
                    )
                    conn.commit()
            except Exception as e:
                self.logger.warning(f"PipelineLogger failed: {e}")

