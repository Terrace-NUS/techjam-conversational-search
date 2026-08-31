from __future__ import annotations

from dataclasses import dataclass

VALID_INTENTS = ("discovery", "browsing", "buying")
NEXT_INTENT = dict(zip(VALID_INTENTS, VALID_INTENTS[1:]))
TRANSITION_THRESHOLDS = {"discovery": 0.3, "browsing": 0.5}


@dataclass
class IntentManager:
    """Per-session intent state machine: discovery -> browsing -> buying, never back.

    `update()` is fed one subscore per non-hit turn (see RewardCalculator). Once the
    intent reaches "buying" it stays there for the rest of the session; a session that
    starts as "buying" never changes.
    """

    intent: str
    threshold: float | None = None

    def __post_init__(self) -> None:
        if self.intent not in VALID_INTENTS:
            raise ValueError(f"unknown intent: {self.intent}")

    def update(self, subscore: float) -> bool:
        """Advance one intent stage when `subscore` clears the threshold.

        Returns True exactly when this call caused an escalation.
        """
        if self.intent in NEXT_INTENT and subscore >= self.current_threshold:
            self.intent = NEXT_INTENT[self.intent]
            return True
        return False

    @property
    def current_threshold(self) -> float:
        return self.threshold if self.threshold is not None else TRANSITION_THRESHOLDS.get(
            self.intent, 0.5
        )
