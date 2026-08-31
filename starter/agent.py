from __future__ import annotations

import os
from abc import ABC, abstractmethod
from pathlib import Path


class Agent(ABC):
    """Common evaluator contract implemented by every participant agent."""

    @abstractmethod
    def reset(self, session_id: str, user_profile: dict) -> None:
        raise NotImplementedError

    @abstractmethod
    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        raise NotImplementedError


def build_agent(name: str | None = None, catalog_path: str | Path = "data/catalog.jsonl") -> Agent:
    mode = (name or os.environ.get("TECHJAM_AGENT", "baseline")).strip().lower()
    if mode == "baseline":
        from .baseline import BaselineAgent

        return BaselineAgent(catalog_path)
    if mode in {"v1", "solution"}:
        from .v1 import V1Agent

        return V1Agent(catalog_path)
    if mode in {"terrace", "shopping_copilot", "real_world"}:
        from .terrace import TerraceAgent

        return TerraceAgent(catalog_path)
    raise ValueError(f"unknown agent: {mode}")


__all__ = ["Agent", "build_agent"]
