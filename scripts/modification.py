from __future__ import annotations

import random
from pathlib import Path

from starter.v1.catalog import ATTRIBUTES, COLORS, MATERIALS, SIZE_WORDS, STYLE_WORDS, USE_CASE_WORDS

from .attributes import (
    PRICE_BAND_LABELS,
    band_reference_price,
    browsing_budget_ceiling,
    extract_attributes,
    item_category,
)
from .llm_client import DeepSeekAttributeWriter, cached_json_call
from .schema import Modification

# Global fallback vocabulary, used only when a category has no catalog-observed
# values for an attribute (category_vocab is preferred: it keeps fakes plausible
# for the item's own category instead of e.g. "wool" for a wristwatch).
FAKE_VALUE_VOCAB: dict[str, tuple[str, ...]] = {
    "material": MATERIALS,
    "color": COLORS,
    "size": SIZE_WORDS,
    "style": STYLE_WORDS,
    "use_case": USE_CASE_WORDS,
    "budget": PRICE_BAND_LABELS,
}
assert set(FAKE_VALUE_VOCAB) == set(ATTRIBUTES)

MODIFY_TURN_CHOICES = (3, 4)
FAKE_ATTRIBUTE_COUNT_WEIGHTS = ((1, 0.7), (2, 0.3))


def _choose_fake_attribute_count(rng: random.Random) -> int:
    counts, weights = zip(*FAKE_ATTRIBUTE_COUNT_WEIGHTS)
    return rng.choices(counts, weights=weights, k=1)[0]


def _choose_fake_value(
    attribute: str,
    true_value: str,
    category: str,
    category_vocab: dict[str, dict[str, list[str]]],
    rng: random.Random,
) -> str | None:
    if attribute == "budget":
        candidates = sorted(value for value in FAKE_VALUE_VOCAB["budget"] if value != true_value)
    else:
        candidates = [
            value
            for value in category_vocab.get(category, {}).get(attribute, [])
            if value != true_value
        ]
        if not candidates:
            candidates = sorted(value for value in FAKE_VALUE_VOCAB[attribute] if value != true_value)
    if not candidates:
        return None
    return rng.choice(sorted(candidates))


def build_modification(
    product: dict,
    item_id: str,
    writer: DeepSeekAttributeWriter,
    cache_dir: Path,
    category_vocab: dict[str, dict[str, list[str]]],
) -> Modification | None:
    """Build a Modification for one item, or None if it has no fakeable attributes."""
    true_attributes = extract_attributes(product)
    category = item_category(product)
    fakeable = sorted(attribute for attribute in ATTRIBUTES if attribute in true_attributes)
    if not fakeable:
        return None

    count_rng = random.Random(f"{item_id}:modify_attrs")
    count = min(_choose_fake_attribute_count(count_rng), len(fakeable))
    chosen = count_rng.sample(fakeable, k=count)

    fake_values: dict[str, str] = {}
    for attribute in chosen:
        value_rng = random.Random(f"{item_id}:{attribute}:fake_value")
        fake_value = _choose_fake_value(attribute, true_attributes[attribute], category, category_vocab, value_rng)
        if fake_value is not None:
            fake_values[attribute] = fake_value
    if not fake_values:
        return None

    budget_context = None
    if "budget" in fake_values:
        price_rng = random.Random(f"{item_id}:budget:fake_price")
        exact_price = band_reference_price(fake_values["budget"], price_rng)
        budget_context = {"exact_price": exact_price, "browsing_ceiling": browsing_budget_ceiling(exact_price)}

    cache_path = cache_dir / f"{item_id}.json"
    fake_descriptions = cached_json_call(
        cache_path, lambda: writer.describe(category, fake_values, budget_context)
    )
    fake_attributes = {
        attribute: {
            "browsing": fake_descriptions["browsing"][attribute],
            "buying": fake_descriptions["buying"][attribute],
        }
        for attribute in fake_values
    }

    turn_rng = random.Random(f"{item_id}:modify_turn")
    modify_turn = turn_rng.choice(MODIFY_TURN_CHOICES)
    return Modification(item_id=item_id, fake_attributes=fake_attributes, modify_turn=modify_turn)

