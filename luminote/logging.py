from __future__ import annotations

import json
from pathlib import Path
from threading import Lock


class JsonlLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = Lock()
        path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: dict) -> None:
        with self._lock:
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
