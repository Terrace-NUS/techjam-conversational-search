from __future__ import annotations

import random
from dataclasses import dataclass, field

from .schema import Item
from .schema import Modification

ACTIVE_ATTRIBUTE_COUNT = 4
INTENTS = ("browsing", "buying")


@dataclass
class QueryHandler:
    """Serve only the session's selected attributes from the current intent."""

    session_id: str
    item: Item
    intent: str = "browsing"
    modification: Modification | None = None
    preferred_attributes: tuple[str, ...] = ()
    active_attributes: tuple[str, ...] = field(init=False)
    disclosed_attributes: set[str] = field(default_factory=set, init=False)
    modification_applied: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        if self.intent not in INTENTS:
            raise ValueError(f"unknown intent: {self.intent}")
        available = sorted(
            set(self.item.intent_descriptions.get("browsing", {}))
            | set(self.item.intent_descriptions.get("buying", {}))
        )
        rng = random.Random(f"{self.session_id}:{self.item.item_id}:active_attributes")
        if len(available) < ACTIVE_ATTRIBUTE_COUNT:
            raise ValueError(
                f"item {self.item.item_id} has {len(available)} attributes; "
                f"exactly {ACTIVE_ATTRIBUTE_COUNT} are required"
            )
        preferred = [name for name in self.preferred_attributes if name in available]
        remaining = [name for name in available if name not in preferred]
        selected = preferred + rng.sample(remaining, k=ACTIVE_ATTRIBUTE_COUNT - len(preferred))
        self.active_attributes = tuple(selected)

    def set_intent(self, intent: str) -> None:
        if intent not in INTENTS:
            raise ValueError(f"unknown intent: {intent}")
        self.intent = intent

    def _apply_modification(self, turn: int) -> str:
        if self.modification_applied or self.modification is None:
            return ""
        if turn < self.modification.modify_turn:
            return ""
        self.modification_applied = True
        corrections = []
        for attribute in self.modification.fake_attributes:
            if attribute not in self.disclosed_attributes:
                continue
            true_description = self.item.intent_descriptions.get(self.intent, {}).get(attribute)
            if true_description:
                corrections.append(f"For {attribute}, I need to correct what I said earlier: {true_description}")
        return " ".join(corrections)

    def answer(self, ask_attribute: object, turn: int = 1) -> str | None:
        """Return the current-intent clue for an active attribute and mark it disclosed.

        A direct request for an inactive attribute is intentionally unanswered. The
        open ``other`` slot reveals at most one still-hidden active attribute.
        """
        correction = self._apply_modification(turn)
        attribute = ask_attribute if isinstance(ask_attribute, str) else None
        if attribute == "other":
            attribute = next(
                (name for name in self.active_attributes if name not in self.disclosed_attributes),
                None,
            )
        description = None
        if attribute in self.active_attributes:
            if (
                not self.modification_applied
                and self.modification is not None
                and attribute in self.modification.fake_attributes
            ):
                description = self.modification.fake_attributes[attribute].get(self.intent)
            else:
                description = self.item.intent_descriptions.get(self.intent, {}).get(attribute)
            if description:
                self.disclosed_attributes.add(attribute)
        if correction and description:
            return f"{description} {correction}"
        return description or correction or None

    def remaining_attributes(self) -> tuple[str, ...]:
        return tuple(
            name for name in self.active_attributes if name not in self.disclosed_attributes
        )
