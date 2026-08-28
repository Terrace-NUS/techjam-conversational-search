from __future__ import annotations

import random
from dataclasses import dataclass

from .query_handler import QueryHandler
from .schema import Item, Modification

MODIFICATION_SESSION_RATE = 0.30


@dataclass(frozen=True)
class Session:
    """Runtime session configuration; the item and ground truth never change."""

    session_id: str
    item: Item
    query_handler: QueryHandler
    modification: Modification | None


def create_session(
    session_id: str,
    item: Item,
    modification: Modification | None = None,
    initial_intent: str = "browsing",
) -> Session:
    """Select exactly four attributes and enable modification for a deterministic 30% of sessions."""
    enabled_modification = None
    if modification is not None:
        rng = random.Random(f"{session_id}:{item.item_id}:modification")
        if rng.random() < MODIFICATION_SESSION_RATE:
            enabled_modification = modification
    preferred = tuple(enabled_modification.fake_attributes) if enabled_modification else ()
    handler = QueryHandler(
        session_id,
        item,
        initial_intent,
        enabled_modification,
        preferred,
    )
    return Session(session_id, item, handler, enabled_modification)