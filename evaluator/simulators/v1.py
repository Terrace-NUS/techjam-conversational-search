from __future__ import annotations

import random
import re

from evaluator.reply_model import ReplyModel

from .base import Simulator, coarse_category

ALLOWED_ATTRIBUTES = {
    "category", "material", "color", "size", "style", "brand",
    "budget", "feature", "use_case", "other",
}
MATERIALS = ("cotton", "polyester", "nylon", "leather", "wool", "spandex", "silk", "rayon", "fabric")
SEARCH_FIELDS = ("title", "features", "details", "description", "categories", "store")
MATERIAL_RE = re.compile(r"\b(cotton|polyester|nylon|leather|wool|spandex|silk|rayon|fabric)\b", re.I)
COLOR_RE = re.compile(r"\b(black|white|blue|red|pink|green|brown|gray|grey|purple|yellow|orange)\b", re.I)


def searchable_text(product: dict) -> str:
    parts: list[str] = []
    for field in SEARCH_FIELDS:
        value = product.get(field)
        if isinstance(value, dict):
            parts.extend(f"{key} {item}" for key, item in value.items())
        elif isinstance(value, list):
            parts.extend(str(item) for item in value)
        elif value is not None:
            parts.append(str(value))
    return " ".join(parts).strip()


def _flatten_values(value: object) -> list[str]:
    if isinstance(value, dict):
        return [f"{key}: {item}" for key, item in value.items() if item not in (None, "", [])]
    if isinstance(value, list):
        return [str(item) for item in value if item not in (None, "")]
    return [str(value)] if value not in (None, "") else []


def _clean_constraint(value: str, limit: int) -> str:
    return re.sub(r"\s+", " ", value).strip(" -;,.\t\n")[:limit].rstrip()


def intent_card(product: dict, limit: int = 180) -> dict:
    title = _clean_constraint(str(product.get("title") or "product"), limit)
    candidates = [*_flatten_values(product.get("features")), *_flatten_values(product.get("details"))]
    corpus = searchable_text(product)
    material = MATERIAL_RE.search(corpus)
    color = COLOR_RE.search(corpus)
    if material:
        candidates.insert(0, material.group(1).lower())
    if color:
        candidates.insert(1, f"color: {color.group(1).lower()}")
    if product.get("price") not in (None, ""):
        candidates.append(f"budget around ${product['price']}")
    cleaned = list(dict.fromkeys(_clean_constraint(item, limit) for item in candidates if _clean_constraint(item, limit)))
    if not cleaned:
        cleaned = [title]
    return {
        "target_category": title,
        "hard_constraints": cleaned[:2],
        "soft_preferences": cleaned[2:4] or cleaned[:1],
    }


def behavior_for(scenario: str, card: dict, rng: random.Random) -> dict:
    behavior: dict = {"scenario_type": scenario}
    if scenario == "intent_override":
        hard = card["hard_constraints"]
        soft = card["soft_preferences"]
        old_value = soft[-1] if soft else "I prefer a different style."
        new_value = hard[0] if hard else "Please prioritize the target requirements."
        behavior["override"] = {
            "turn": rng.choice([3, 4]),
            "old_value": old_value,
            "new_value": new_value,
            "message": f"Actually, ignore my earlier preference. What I need is: {new_value}.",
        }
    return behavior


def classify_constraint(value: str) -> str:
    lowered = value.lower()
    if "budget" in lowered or re.search(r"(?:\$|<=|under)\s*\d", lowered):
        return "budget"
    if any(material in lowered for material in MATERIALS):
        return "material"
    if any(word in lowered for word in ("color", "black", "white", "blue", "red", "pink", "green")):
        return "color"
    if any(word in lowered for word in ("size", "sizing", "width", "wide", "narrow")):
        return "size"
    if any(word in lowered for word in ("department", "style", "fit", "sleeve", "neck")):
        return "style"
    if any(word in lowered for word in ("hiking", "running", "gym", "winter", "outdoor", "work")):
        return "use_case"
    return "feature"


class V1Simulator(Simulator):
    """Legacy scenario_type simulator; missing version fields route here."""

    @staticmethod
    def _materialize_hidden_fields(sample: dict, products: dict[str, dict]) -> tuple[dict, dict]:
        if "intent_card" in sample and "behavior" in sample:
            return sample["intent_card"], sample["behavior"]
        target = str(sample["ground_truth"]["parent_asin"])
        card = intent_card(products[target])
        seed_source = f"{sample.get('sample_id', '')}\0{sample.get('scenario_type', '')}"
        behavior = behavior_for(str(sample["scenario_type"]), card, random.Random(seed_source))
        return card, behavior

    def __init__(
        self,
        sample: dict,
        categories: dict[str, list[str]],
        products: dict[str, dict],
        reply_model: ReplyModel,
        session_id: str,
    ) -> None:
        super().__init__(sample, categories, products, reply_model, session_id)
        intent_card_data, behavior = self._materialize_hidden_fields(sample, products)
        self.effective_sample = {**sample, "intent_card": intent_card_data, "behavior": behavior}
        self.intent_card = intent_card_data
        self.disclosed: set[str] = set()
        self.boundary_used = False
        self.override_applied = sample["scenario_type"] != "intent_override"

    def initial_message(self) -> str:
        category = coarse_category(self.categories.get(self.target, []))
        scenario = self.sample["scenario_type"]
        if scenario == "buying" and self.intent_card.get("hard_constraints"):
            constraint = str(self.intent_card["hard_constraints"][0])
            self.disclosed.add(constraint)
            canonical = f"I'm looking for {category}. A key requirement is: {constraint}."
        elif scenario == "intent_override":
            old_value = str(self.effective_sample["behavior"]["override"]["old_value"])
            canonical = f"I'm looking for {category}. {old_value}"
        else:
            canonical = f"I'm looking for {category}, but I'm still exploring."
        return self.reply_model.rewrite_initial_message(canonical)

    @property
    def ready_for_hit(self) -> bool:
        return self.override_applied

    def next_message(self, response: dict, next_turn: int) -> str:
        override = self.effective_sample.get("behavior", {}).get("override") or {}
        if not self.override_applied and next_turn == int(override.get("turn", 3)):
            self.override_applied = True
            new_value = str(override.get("new_value", ""))
            if new_value:
                self.disclosed.add(new_value)
            return self.reply_model.override_message(override)
        attribute = response.get("ask_attribute")
        attribute = attribute if isinstance(attribute, str) else None
        if self.sample["scenario_type"] == "boundary" and not self.boundary_used and attribute:
            self.boundary_used = True
            canonical = f"I don't have a preference for {attribute}; please use your judgment."
        elif not attribute:
            canonical = "Those options are not quite right yet. Ask me about one specific attribute."
        else:
            if attribute not in ALLOWED_ATTRIBUTES:
                attribute = "other"
            constraints = [
                *[str(value) for value in self.intent_card.get("hard_constraints", [])],
                *[str(value) for value in self.intent_card.get("soft_preferences", [])],
            ]
            matches = [
                value
                for value in constraints
                if value not in self.disclosed
                and (attribute == "other" or classify_constraint(value) == attribute)
            ][:2]
            if matches:
                self.disclosed.update(matches)
                canonical = "For that, what matters is: " + "; ".join(matches) + "."
            else:
                canonical = f"I don't have an additional preference for {attribute}."
        return self.reply_model.rewrite_query_answer(canonical)

    @property
    def scenario_type(self) -> str:
        return str(self.sample["scenario_type"])
