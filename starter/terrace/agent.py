from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

from shopping_copilot.application import RealWorldConfig, build_real_world_agent
from starter.agent import Agent


def _load_dotenv(path: str | Path = ".env") -> None:
    dotenv_path = Path(path)
    if not dotenv_path.is_file():
        return
    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key and key not in os.environ:
            os.environ[key] = value.strip().strip("\"'")


class TerraceAgent(Agent):
    """Full Terrace-NUS DeepSeek, typed-memory, and dense-retrieval agent."""

    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl") -> None:
        _load_dotenv()
        api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is required for the Terrace agent")

        self._delegate = build_real_world_agent(
            catalog_path,
            RealWorldConfig(
                api_key=api_key,
                model=os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash"),
                base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
                device=os.environ.get("TERRACE_DEVICE", "cuda"),
            ),
        )

    def reset(self, session_id: str, user_profile: dict) -> None:
        self._delegate.reset(session_id, user_profile)

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        return self._delegate.respond(session_id, user_message, turn, top_k)

    def last_audit(self, session_id: str) -> dict[str, object]:
        method = getattr(self._delegate, "last_audit", None)
        if not callable(method):
            raise LookupError("Terrace agent does not provide turn audits")
        return method(session_id)

    def set_event_sink(
        self,
        session_id: str,
        sink: Callable[[dict[str, object]], None] | None,
    ) -> None:
        method = getattr(self._delegate, "set_event_sink", None)
        if callable(method):
            method(session_id, sink)
