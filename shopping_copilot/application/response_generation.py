"""Deterministic, evidence-bound natural-language response generation."""

from __future__ import annotations

import math
import textwrap
from collections.abc import Mapping
from dataclasses import dataclass

from shopping_copilot.retrieval.deepseek_ranking import QualityRankingHit
from shopping_copilot.session_context import IntentState

from .quality_ranking import RealWorldRankingResult

RESPONSE_SCHEMA = "shopping-copilot/deterministic-response-narrative/v2"
TRANSPARENCY_MOVEMENT_THRESHOLD = 0.12
ATTRIBUTE_QUESTIONS = {
    "category": "What kind of product are you looking for?",
    "feature": "Which product feature matters most to you?",
    "material": "Do you have a preferred material?",
    "budget": "What budget should I stay within?",
    "style": "What style do you prefer?",
    "brand": "Do you have a preferred brand?",
    "other": "What other product requirement or preference matters to you?",
    "use_case": "What will you mainly use it for?",
    "size": "Are there any size or fit requirements?",
    "color": "Do you have a preferred color?",
}
ATTRIBUTE_ORDER = tuple(ATTRIBUTE_QUESTIONS)


@dataclass(frozen=True, slots=True, kw_only=True)
class ProductNarrative:
    """One product explanation copied only from ranking and catalog evidence."""

    parent_asin: str
    title: str
    reason: str
    caveat: str | None


@dataclass(frozen=True, slots=True, kw_only=True)
class ResponseNarrative:
    """Auditable plan and final text for one deterministic assistant response."""

    schema: str
    transparency: float
    previous_transparency: float | None
    presentation_band: str
    movement: str
    category_labels: tuple[str, ...]
    products: tuple[ProductNarrative, ...]
    question_attribute: str
    follow_up: str
    message: str


class DeterministicResponseComposer:
    """Keep ranking evidence for audit while emitting one explicit question."""

    def __init__(self, *, maximum_product_notes: int = 3) -> None:
        if type(maximum_product_notes) is not int or maximum_product_notes <= 0:
            raise ValueError("maximum_product_notes must be positive")
        self._maximum_product_notes = maximum_product_notes

    def compose(
        self,
        *,
        recommendations: tuple[str, ...],
        transparency: float,
        previous_transparency: float | None,
        ranking: RealWorldRankingResult | None,
        intent: IntentState,
        product_metadata: Mapping[str, Mapping[str, object]],
        asked_attributes: tuple[str, ...] = (),
    ) -> ResponseNarrative:
        _validate_transparency(transparency, name="transparency")
        if previous_transparency is not None:
            _validate_transparency(previous_transparency, name="previous_transparency")
        if type(recommendations) is not tuple or any(
            type(parent_asin) is not str or not parent_asin.strip()
            for parent_asin in recommendations
        ):
            raise TypeError("recommendations must contain non-empty product IDs")
        if type(intent) is not IntentState:
            raise TypeError("intent must be an exact IntentState")
        if not isinstance(product_metadata, Mapping):
            raise TypeError("product_metadata must be a mapping")
        if type(asked_attributes) is not tuple or any(
            type(attribute) is not str for attribute in asked_attributes
        ):
            raise TypeError("asked_attributes must be a tuple of strings")

        band = _presentation_band(transparency)
        movement = _movement(transparency, previous_transparency)
        categories = _category_labels(recommendations, product_metadata)
        quality_hits = _quality_hits(ranking)
        products = tuple(
            _product_narrative(
                parent_asin,
                metadata=product_metadata.get(parent_asin, {}),
                quality_hit=quality_hits.get(parent_asin),
            )
            for parent_asin in recommendations[: self._maximum_product_notes]
        )
        question_attribute, follow_up = _next_question(
            intent=intent,
            asked_attributes=asked_attributes,
        )
        return ResponseNarrative(
            schema=RESPONSE_SCHEMA,
            transparency=transparency,
            previous_transparency=previous_transparency,
            presentation_band=band,
            movement=movement,
            category_labels=categories,
            products=products,
            question_attribute=question_attribute,
            follow_up=follow_up,
            message=follow_up,
        )


def _presentation_band(transparency: float) -> str:
    if transparency < 0.35:
        return "broad"
    if transparency < 0.70:
        return "narrowing"
    return "focused"


def _movement(current: float, previous: float | None) -> str:
    if previous is None:
        return "initial"
    delta = current - previous
    if delta >= TRANSPARENCY_MOVEMENT_THRESHOLD:
        return "narrowed"
    if delta <= -TRANSPARENCY_MOVEMENT_THRESHOLD:
        return "broadened"
    return "stable"


def _quality_hits(
    ranking: RealWorldRankingResult | None,
) -> dict[str, QualityRankingHit]:
    if ranking is None or ranking.quality_pipeline is None:
        return {}
    return {hit.parent_asin: hit for hit in ranking.quality_pipeline.quality_ranking.hits}


def _product_narrative(
    parent_asin: str,
    *,
    metadata: Mapping[str, object],
    quality_hit: QualityRankingHit | None,
) -> ProductNarrative:
    raw_title = metadata.get("title")
    title = (
        parent_asin
        if type(raw_title) is not str or not raw_title.strip()
        else _shorten(raw_title, width=100)
    )
    reason = (
        "This is one of the strongest remaining matches from the current search."
        if quality_hit is None or not quality_hit.reason
        else _sentence(_shorten(quality_hit.reason, width=220))
    )
    caveat = None
    if quality_hit is not None and quality_hit.concerns:
        candidate = _sentence(_shorten(quality_hit.concerns[0], width=160))
        if candidate.casefold().rstrip(".") not in reason.casefold().rstrip("."):
            caveat = candidate
    return ProductNarrative(
        parent_asin=parent_asin,
        title=title,
        reason=reason,
        caveat=caveat,
    )


def _next_question(
    *,
    intent: IntentState,
    asked_attributes: tuple[str, ...],
) -> tuple[str, str]:
    active = {item.facet for item in intent.preferences if item.facet is not None}
    unavailable = active | set(intent.dont_care_facets) | set(asked_attributes)
    if intent.goal is not None:
        unavailable.add("category")
    attribute = next(
        (candidate for candidate in ATTRIBUTE_ORDER if candidate not in unavailable),
        "other",
    )
    return attribute, ATTRIBUTE_QUESTIONS[attribute]


def _category_labels(
    recommendations: tuple[str, ...],
    metadata: Mapping[str, Mapping[str, object]],
    *,
    limit: int = 4,
) -> tuple[str, ...]:
    labels: list[str] = []
    for parent_asin in recommendations:
        product = metadata.get(parent_asin, {})
        values = _flatten_strings(product.get("categories"))
        if not values:
            continue
        label = _shorten(values[-1], width=60)
        if label.casefold() in {item.casefold() for item in labels}:
            continue
        labels.append(label)
        if len(labels) == limit:
            break
    return tuple(labels)


def _flatten_strings(value: object) -> list[str]:
    if type(value) is str:
        return [value.strip()] if value.strip() else []
    if isinstance(value, (list, tuple)):
        result: list[str] = []
        for item in value:
            result.extend(_flatten_strings(item))
        return result
    return []


def _shorten(value: str, *, width: int) -> str:
    return textwrap.shorten(" ".join(value.split()), width=width, placeholder="…")


def _sentence(value: str) -> str:
    return value if value.endswith((".", "!", "?")) else f"{value}."


def _validate_transparency(value: float, *, name: str) -> None:
    if type(value) is not float or not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be a finite float in [0, 1]")
