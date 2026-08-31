from __future__ import annotations

from dataclasses import dataclass, field

from .schema import Item, Modification, clue_text
from .intent_manager import VALID_INTENTS

INTENTS = VALID_INTENTS


@dataclass
class QueryHandler:
    """Serve every available attribute from the current intent."""

    session_id: str
    item: Item
    intent: str = "browsing"
    modification: Modification | None = None
    active_attributes: tuple[str, ...] = field(init=False)
    disclosed_attributes: set[str] = field(default_factory=set, init=False)
    disclosure_history: list[str] = field(default_factory=list, init=False)
    selected_modification_attribute: str | None = field(default=None, init=False)
    modification_applied: bool = field(default=False, init=False)
    intent_transition_pending: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        if self.intent not in INTENTS:
            raise ValueError(f"unknown intent: {self.intent}")
        self.active_attributes = tuple(sorted(set().union(
            *(self.item.intent_descriptions.get(intent, {}) for intent in INTENTS)
        )))

    def set_intent(self, intent: str) -> None:
        if intent not in INTENTS:
            raise ValueError(f"unknown intent: {intent}")
        self.intent_transition_pending = intent != self.intent
        self.intent = intent

    def _apply_modification(self, turn: int) -> str:
        if self.modification_applied or self.modification is None:
            return ""
        if turn < self.modification.modify_turn:
            return ""
        self.modification_applied = True
        attribute = self.selected_modification_attribute
        if attribute is None:
            return ""
        return clue_text(self.modification.correction_messages.get(attribute, {}).get(self.intent))

    def answer(self, ask_attribute: object, turn: int = 1) -> str | None:
        """Return the current-intent clue for an available attribute and mark it disclosed.

        An unknown attribute is unanswered. ``other`` is handled like every other
        available attribute.
        """
        correction = self._apply_modification(turn)
        attribute = ask_attribute if isinstance(ask_attribute, str) else None
        transition_attribute = None
        if self.intent_transition_pending:
            transition_attribute = next(
                (name for name in reversed(self.disclosure_history) if name != attribute),
                None,
            )
            self.intent_transition_pending = False
        description = None
        if attribute in self.active_attributes:
            description = self.item.intent_descriptions.get(self.intent, {}).get(attribute)
            if not self.modification_applied and self.modification is not None:
                selected = self.selected_modification_attribute
                if selected is None and attribute in self.modification.fake_attributes:
                    candidate = self.modification.fake_attributes[attribute].get(self.intent)
                    if clue_text(candidate):
                        self.selected_modification_attribute = attribute
                        description = candidate
                elif attribute == selected:
                    description = self.modification.fake_attributes[attribute].get(self.intent)
            if clue_text(description):
                self.disclosed_attributes.add(attribute)
                self.disclosure_history.append(attribute)
        rendered = clue_text(description)
        transition_description = self.item.intent_descriptions.get(self.intent, {}).get(
            transition_attribute
        )
        if (
            transition_attribute is not None
            and not self.modification_applied
            and self.modification is not None
            and transition_attribute == self.selected_modification_attribute
        ):
            transition_description = self.modification.fake_attributes[transition_attribute].get(
                self.intent
            )
        transition = clue_text(transition_description)
        parts = [part for part in (rendered, correction) if part]
        if transition:
            parts.append(f"Also, for {transition_attribute}: {transition}")
        return " ".join(parts) or None

    def remaining_attributes(self) -> tuple[str, ...]:
        return tuple(
            name for name in self.active_attributes if name not in self.disclosed_attributes
        )
