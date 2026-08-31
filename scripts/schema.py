from __future__ import annotations

from dataclasses import dataclass


Clue = str | list[str]


def clue_text(value: Clue | None) -> str:
    """Render stored clue fragments at the customer-message boundary."""
    if isinstance(value, list):
        return "; ".join(str(part).strip() for part in value if str(part).strip())
    return str(value or "").strip()


@dataclass
class Item:
    """A preprocessed catalog item: raw fields plus paraphrased per-intent clues."""

    item_id: str
    features: dict
    # discovery/browsing/buying -> {attribute: [compact clue fragments]}; true values.
    intent_descriptions: dict[str, dict[str, Clue]]


@dataclass
class Modification:
    """Fake candidates for all modifiable fields; the first queried one is activated."""

    item_id: str
    # attribute -> discovery/browsing/buying clue fragments; one is selected per session.
    fake_attributes: dict[str, dict[str, Clue]]
    # attribute -> discovery/browsing/buying correction text.
    correction_messages: dict[str, dict[str, Clue]]
    modify_turn: int
