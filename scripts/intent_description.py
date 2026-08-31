from __future__ import annotations

from pathlib import Path

from .attributes import browsing_budget_ceiling, item_category
from .llm_client import DeepSeekAttributeWriter
from .schema import Item


def build_item(
    product: dict, attributes: dict[str, str], writer: DeepSeekAttributeWriter, cache_dir: Path
) -> Item:
    """Build the true-value Item record with clues for all three intent stages."""
    item_id = str(product["parent_asin"])
    category = item_category(product)

    budget_context = None
    if "budget" in attributes and product.get("price") not in (None, ""):
        exact_price = float(product["price"])
        budget_context = {"exact_price": exact_price, "browsing_ceiling": browsing_budget_ceiling(exact_price)}

    intent_descriptions = writer.describe(
        category,
        attributes,
        budget_context,
        cache_dir / f"{item_id}.fragment-lists-v3",
    )
    return Item(item_id=item_id, features=product, intent_descriptions=intent_descriptions)
