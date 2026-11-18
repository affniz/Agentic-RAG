"""Conversations saved to disk, one JSON file each: RAG_DATA/conversations/<id>.json.

A file holds what the UI shows (title, timestamps, turns) and what the agent needs to continue the
chat: its model, effort and the full API `messages`, thinking blocks included. Those must be sent
back unchanged, so the content blocks are stored exactly as the API returned them.

Files are written atomically after each finished turn. Only one server process may use a data
folder at a time (run uvicorn with a single worker), or the processes overwrite each other's files.
"""

from __future__ import annotations

import json
import logging
import re
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from server.feedback import JsonFile

log = logging.getLogger(__name__)
SAFE_ID = re.compile(r"[A-Za-z0-9_-]{1,64}")


def jsonable(value: Any) -> Any:
    """API message content as plain JSON.

    Dicts, lists and strings stay as they are; SDK content blocks (pydantic) are dumped the way the
    API sent them; the fake client's SimpleNamespace blocks become dicts.
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if hasattr(value, "model_dump"):
        return jsonable(value.model_dump(mode="json", exclude_none=True))
    if isinstance(value, SimpleNamespace):
        return jsonable(vars(value))
    raise TypeError(f"can't save {type(value).__name__} in a conversation")


class ConversationStore:
    def __init__(self, directory: Path):
        self.dir = directory
        self.lock = threading.Lock()  # held around save/delete so a delete can't race a save

    def _path(self, cid: str) -> Path:
        if not SAFE_ID.fullmatch(cid):
            raise ValueError(f"bad conversation id {cid!r}")
        return self.dir / f"{cid}.json"

    def save(self, data: dict[str, Any]) -> None:
        """Write one conversation (a dict with an "id"); call with `self.lock` held."""
        JsonFile(self._path(data["id"]), None).write(jsonable(data))

    def delete(self, cid: str) -> None:
        try:
            self._path(cid).unlink(missing_ok=True)
        except ValueError:
            pass

    def load_all(self) -> list[dict[str, Any]]:
        """Every readable conversation file; unreadable ones are skipped with a warning."""
        out = []
        for path in sorted(self.dir.glob("*.json")):
            try:
                data = json.loads(path.read_text())
                if not isinstance(data, dict) or data.get("id") != path.stem or not SAFE_ID.fullmatch(path.stem):
                    raise ValueError("not a conversation file")
                out.append(data)
            except (OSError, ValueError) as e:  # JSONDecodeError and UnicodeDecodeError are ValueErrors
                log.warning("skipping conversation file %s: %s", path, e)
        return out
