from __future__ import annotations

from dataclasses import dataclass

from .query_handler import QueryHandler
from .schema import Item, Modification

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
    """Create a query handler and enable the supplied modification."""
    enabled_modification = modification
    handler = QueryHandler(
        session_id,
        item,
        initial_intent,
        enabled_modification,
    )
    return Session(session_id, item, handler, enabled_modification)
