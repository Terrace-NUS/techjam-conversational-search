from __future__ import annotations

from dataclasses import dataclass

VALID_INTENTS = ("browsing", "buying")


@dataclass
class IntentManager:
    """Per-session intent state machine: browsing escalates to buying, never back.

    `update()` is fed one subscore per non-hit turn (see RewardCalculator). Once the
    intent reaches "buying" it stays there for the rest of the session; a session that
    starts as "buying" never changes.
    """

    intent: str
    threshold: float = 0.8

    def __post_init__(self) -> None:
        if self.intent not in VALID_INTENTS:
            raise ValueError(f"unknown intent: {self.intent}")

    def update(self, subscore: float) -> bool:
        """Escalate browsing -> buying when `subscore` clears the threshold.

        Returns True exactly when this call caused an escalation.
        """
        if self.intent == "browsing" and subscore >= self.threshold:
            self.intent = "buying"
            return True
        return False
