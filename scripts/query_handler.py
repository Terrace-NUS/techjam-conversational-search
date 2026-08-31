from __future__ import annotations

from dataclasses import dataclass, field

from .schema import Item
from .schema import Modification

INTENTS = ("browsing", "buying")


@dataclass
class QueryHandler:
    """Serve every available attribute from the current intent."""

    session_id: str
    item: Item
    intent: str = "browsing"
    modification: Modification | None = None
    active_attributes: tuple[str, ...] = field(init=False)
    disclosed_attributes: set[str] = field(default_factory=set, init=False)
    modification_applied: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        if self.intent not in INTENTS:
            raise ValueError(f"unknown intent: {self.intent}")
        self.active_attributes = tuple(sorted(
            set(self.item.intent_descriptions.get("browsing", {}))
            | set(self.item.intent_descriptions.get("buying", {}))
        ))

    def _modification_attribute(self) -> str | None:
        if not self.modification or not self.modification.fake_attributes:
            return None
        return next(iter(self.modification.fake_attributes))

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
        attribute = self._modification_attribute()
        if attribute not in self.disclosed_attributes:
            return ""
        return self.modification.correction_messages.get(attribute, {}).get(self.intent, "")

    def answer(self, ask_attribute: object, turn: int = 1) -> str | None:
        """Return the current-intent clue for an available attribute and mark it disclosed.

        An unknown attribute is unanswered. ``other`` is handled like every other
        available attribute.
        """
        correction = self._apply_modification(turn)
        attribute = ask_attribute if isinstance(ask_attribute, str) else None
        description = None
        modification_attribute = self._modification_attribute()
        if attribute in self.active_attributes:
            if (
                not self.modification_applied
                and self.modification is not None
                and attribute == modification_attribute
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
