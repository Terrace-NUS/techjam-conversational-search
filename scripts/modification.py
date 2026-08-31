from __future__ import annotations

import random
import re
from pathlib import Path

from starter.v1.catalog import (
    ATTRIBUTES,
    COLORS,
    MATERIALS,
    SIZE_WORDS,
    STYLE_WORDS,
    USE_CASE_WORDS,
)

from .attributes import (
    PRICE_BAND_LABELS,
    band_reference_price,
    browsing_budget_ceiling,
)
from .llm_client import DeepSeekAttributeWriter
from .schema import Modification

# Global vocabulary used to select deterministic fake attribute values.
FAKE_VALUE_VOCAB: dict[str, tuple[str, ...]] = {
    "material": MATERIALS,
    "color": COLORS,
    "size": SIZE_WORDS,
    "style": STYLE_WORDS,
    "use_case": USE_CASE_WORDS,
    "budget": PRICE_BAND_LABELS,
}
assert set(FAKE_VALUE_VOCAB) == set(ATTRIBUTES)

MODIFY_TURN_CHOICES = tuple(range(3, 8))


def _value_tokens(value: str) -> set[str]:
    return {
        token for token in re.findall(r"[a-z0-9]+", value.casefold()) if len(token) > 2
    }


def _conflicts_with_truth(candidate: str, true_value: str) -> bool:
    """Whether a candidate fake would still be satisfied by the item's real value.

    Plain inequality is not enough: a true material of "polyester and mesh" makes the
    candidate "mesh" pass an != check while describing the item perfectly, so the
    modification turn asks the agent to switch to something it already had.
    """
    candidate_tokens, true_tokens = _value_tokens(candidate), _value_tokens(true_value)
    if not candidate_tokens or not true_tokens:
        return candidate.casefold() == true_value.casefold()
    return bool(candidate_tokens & true_tokens)


def _choose_fake_value(
    attribute: str,
    true_value: str,
    rng: random.Random,
) -> str | None:
    if attribute == "budget":
        candidates = sorted(
            value for value in FAKE_VALUE_VOCAB["budget"] if value != true_value
        )
    else:
        candidates = sorted(
            value
            for value in FAKE_VALUE_VOCAB[attribute]
            if not _conflicts_with_truth(value, true_value)
        )
    if not candidates:
        return None
    return rng.choice(sorted(candidates))


def build_modification(
    product: dict,
    item_id: str,
    true_attributes: dict[str, str],
    true_descriptions: dict[str, dict[str, object]],
    writer: DeepSeekAttributeWriter,
    cache_dir: Path,
) -> Modification | None:
    """Build a Modification for one item, or None if it has no fakeable attributes."""
    category = true_attributes.get("category", "clothing item")
    modification_values = dict(true_attributes)
    size_options = modification_values.pop("size_options", None)
    if size_options and "size" not in modification_values:
        modification_values["size"] = f"available options: {size_options}"
    fakeable = sorted(
        attribute for attribute in ATTRIBUTES if attribute in modification_values
    )
    if not fakeable:
        return None

    fake_values: dict[str, str] = {}
    for attribute in fakeable:
        value_rng = random.Random(f"{item_id}:{attribute}:fake_value")
        fake_value = _choose_fake_value(
            attribute, modification_values[attribute], value_rng
        )
        if fake_value is not None:
            fake_values[attribute] = fake_value
    if not fake_values:
        return None

    budget_context = None
    if "budget" in fake_values:
        price_rng = random.Random(f"{item_id}:budget:fake_price")
        exact_price = band_reference_price(fake_values["budget"], price_rng)
        budget_context = {
            "exact_price": exact_price,
            "browsing_ceiling": browsing_budget_ceiling(exact_price),
        }

    generated = writer.describe_modification(
        category,
        fake_values,
        true_descriptions,
        budget_context,
        cache_dir / f"{item_id}.fragment-lists-v3",
    )
    fake_descriptions = generated["fake_descriptions"]
    correction_messages = generated["correction_messages"]
    fake_attributes = {
        attribute: {
            stage: fake_descriptions[stage][attribute]
            for stage in ("discovery", "browsing", "buying")
        }
        for attribute in fake_values
    }

    turn_rng = random.Random(f"{item_id}:modify_turn")
    modify_turn = turn_rng.choice(MODIFY_TURN_CHOICES)
    return Modification(
        item_id=item_id,
        fake_attributes=fake_attributes,
        correction_messages={
            attribute: {
                stage: correction_messages[stage][attribute]
                for stage in ("discovery", "browsing", "buying")
            }
            for attribute in fake_values
        },
        modify_turn=modify_turn,
    )
