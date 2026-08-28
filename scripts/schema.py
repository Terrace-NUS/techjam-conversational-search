from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Item:
    """A preprocessed catalog item: raw fields plus paraphrased per-intent clues."""

    item_id: str
    features: dict
    # {"browsing": {attribute: text}, "buying": {attribute: text}}; true values.
    intent_descriptions: dict[str, dict[str, str]]


@dataclass
class Modification:
    """A same-item attribute switch: 1-2 attributes report a fake value until modify_turn."""

    item_id: str
    # attribute -> {"browsing": text, "buying": text}; fake values, disjoint from Item.intent_descriptions.
    fake_attributes: dict[str, dict[str, str]]
    # attribute -> {"browsing": correction, "buying": correction}; generated correction messages.
    correction_messages: dict[str, dict[str, str]]
    modify_turn: int
    new_intent: str | None = None
