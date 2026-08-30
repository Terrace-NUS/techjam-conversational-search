from __future__ import annotations

from evaluator.reply_model import ReplyModel
from scripts.schema import Item, Modification
from scripts.session import create_session

from .base import Simulator, coarse_category


class V2Simulator(Simulator):
    """Embedded intent-description and modification simulator."""

    @staticmethod
    def _intent(sample: dict) -> str:
        intent = sample.get("intent")
        if intent not in {"buying", "browsing"}:
            raise ValueError(f"invalid v2 intent: {intent!r}")
        return intent

    @staticmethod
    def _item(sample: dict, target: str) -> Item:
        item = Item(
            item_id=str(sample["item_id"]),
            features=dict(sample.get("features") or {}),
            intent_descriptions=dict(sample.get("intent_descriptions") or {}),
        )
        if item.item_id != target:
            raise ValueError(f"v2 item_id {item.item_id!r} does not match target {target!r}")
        return item

    @staticmethod
    def _modification(sample: dict, target: str) -> Modification | None:
        fake_attributes = dict(sample.get("fake_attributes") or {})
        if not fake_attributes:
            return None
        modification = Modification(
            item_id=str(sample["item_id"]),
            fake_attributes=fake_attributes,
            correction_messages=dict(sample.get("correction_messages") or {}),
            modify_turn=int(sample["modify_turn"]),
        )
        if modification.item_id != target:
            raise ValueError(
                f"v2 modification item_id {modification.item_id!r} does not match target {target!r}"
            )
        return modification

    @staticmethod
    def _intent_card(item: Item, intent: str, fallback_category: str) -> dict:
        descriptions = item.intent_descriptions.get(intent, {})
        constraints = [
            str(value)
            for attribute, value in descriptions.items()
            if attribute != "category" and value not in (None, "")
        ]
        return {
            "target_category": str(descriptions.get("category") or fallback_category),
            "hard_constraints": constraints[:2],
            "soft_preferences": constraints[2:4] or constraints[:1],
        }

    def __init__(
        self,
        sample: dict,
        categories: dict[str, list[str]],
        products: dict[str, dict],
        reply_model: ReplyModel,
        session_id: str,
    ) -> None:
        super().__init__(sample, categories, products, reply_model, session_id)
        self.intent = self._intent(sample)
        self.override = bool(sample.get("override"))
        item = self._item(sample, self.target)
        self.modification = self._modification(sample, self.target) if self.override else None
        if self.override and self.modification is None:
            raise ValueError(f"v2 override sample {sample['sample_id']!r} has no modification")
        fallback_category = coarse_category(categories.get(self.target, []))
        self.intent_card = self._intent_card(item, self.intent, fallback_category)
        self.category = str(self.intent_card.get("target_category") or fallback_category)
        self.query_handler = create_session(
            str(sample["sample_id"]), item, self.modification, initial_intent=self.intent
        ).query_handler
        self.override_applied = not self.override

    def initial_message(self) -> str:
        if self.override:
            canonical = f"I'm looking for {self.category}, but I'm still exploring."
        elif self.intent == "buying" and self.intent_card.get("hard_constraints"):
            constraint = str(self.intent_card["hard_constraints"][0])
            canonical = f"I'm looking for {self.category}. A key requirement is: {constraint}."
        else:
            canonical = f"I'm looking for {self.category}, but I'm still exploring."
        return self.reply_model.rewrite_initial_message(canonical)

    @property
    def ready_for_hit(self) -> bool:
        return self.override_applied

    def next_message(self, response: dict, next_turn: int) -> str:
        canonical = self.query_handler.answer(response.get("ask_attribute"), next_turn)
        if not self.override_applied:
            assert self.modification is not None
            if next_turn >= self.modification.modify_turn:
                self.override_applied = True
                canonical = canonical or "I have updated my preferences."
        canonical = canonical or "I don't have an additional preference for that."
        return self.reply_model.rewrite_query_answer(canonical)

    @property
    def scenario_type(self) -> str:
        return self.intent

    def result_metadata(self) -> dict:
        return {"version": "v2", "override": self.override}
