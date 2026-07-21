from __future__ import annotations

from datetime import datetime, timezone
from threading import Lock

from .logging import JsonlLogger


class GenerationIntegrityStore:
    """Correlates server generation with the browser's final rendered text."""

    def __init__(self, logger: JsonlLogger) -> None:
        self.logger = logger
        self._records: dict[str, dict] = {}
        self._lock = Lock()

    def record_generation(self, debug_id: str, record: dict) -> None:
        # The full diagnostic record belongs in the append-only log. Memory only
        # needs the exact server text for the browser comparison.
        stored = {"server_decoded_text": record["server_decoded_text"]}
        with self._lock:
            self._records[debug_id] = stored
        self.logger.append({
            "event": "generation",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "debug_id": debug_id,
            **record,
        })

    def record_frontend_text(self, debug_id: str, frontend_text: str) -> bool | None:
        with self._lock:
            record = self._records.get(debug_id)
        if record is None:
            return None
        matches = record["server_decoded_text"] == frontend_text
        self.logger.append({
            "event": "frontend_render",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "debug_id": debug_id,
            "frontend_rendered_text": frontend_text,
            "server_frontend_text_match": matches,
        })
        return matches
