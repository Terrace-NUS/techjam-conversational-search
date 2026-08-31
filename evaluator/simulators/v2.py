from __future__ import annotations

from evaluator.reply_model import ReplyModel
from scripts.intent_manager import VALID_INTENTS
from scripts.query_attribute import extract_attribute
from scripts.query_handler import QueryHandler
from scripts.schema import Item, Modification, clue_text
from scripts.session import create_session

from .base import Simulator, coarse_category


def query_attribute_from_response(
    response: dict, query_handler: QueryHandler
) -> str | None:
    """Prefer the natural-language question, then fall back to ask_attribute."""
    available = query_handler.item.intent_descriptions.get(query_handler.intent, {})
    attribute = extract_attribute(response.get("message"))
    if attribute is not None:
        return attribute if attribute in available else None
    fallback = response.get("ask_attribute")
    return fallback if isinstance(fallback, str) and fallback in available else None


class V2Simulator(Simulator):
    """Embedded intent-description and modification simulator."""

    @staticmethod
    def _intent(sample: dict) -> str:
        intent = sample.get("intent")
        if intent not in VALID_INTENTS:
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
        self.descriptions = item.intent_descriptions.get(self.intent, {})
        self.category = clue_text(self.descriptions.get("category")) or fallback_category
        self.query_handler = create_session(
            str(sample["sample_id"]), item, self.modification, initial_intent=self.intent
        ).query_handler

    def initial_message(self) -> str:
        if self.intent == "buying":
            canonical = f"I'm looking for {self.category}."
        elif self.intent == "browsing":
            canonical = f"I'm looking for {self.category}, but I'm still exploring."
        else:
            canonical = clue_text(
                self.descriptions.get("category")
                or next(iter(self.descriptions.values()), self.category)
            )
        return self.reply_model.rewrite_initial_message(canonical)

    @property
    def ready_for_hit(self) -> bool:
        return True

    def query_attribute(self, response: dict) -> str | None:
        return query_attribute_from_response(response, self.query_handler)

    def next_message(self, response: dict, next_turn: int) -> str:
        attribute = self.query_attribute(response)
        canonical = self.query_handler.answer(attribute, next_turn)
        canonical = canonical or "I don't have an additional preference for that."
        return self.reply_model.rewrite_query_answer(canonical)

    @property
    def scenario_type(self) -> str:
        return self.intent

    def result_metadata(self) -> dict:
        return {"version": "v2", "override": self.override}
