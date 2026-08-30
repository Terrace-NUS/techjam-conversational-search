from __future__ import annotations

from pathlib import Path

from .attributes import browsing_budget_ceiling, item_category
from .llm_client import DeepSeekAttributeWriter, cached_json_call
from .schema import Item


def build_item(
    product: dict, attributes: dict[str, str], writer: DeepSeekAttributeWriter, cache_dir: Path
) -> Item:
    """Build the true-value Item record (paraphrased browsing/buying clues) for one product."""
    item_id = str(product["parent_asin"])
    category = item_category(product)

    budget_context = None
    if "budget" in attributes and product.get("price") not in (None, ""):
        exact_price = float(product["price"])
        budget_context = {"exact_price": exact_price, "browsing_ceiling": browsing_budget_ceiling(exact_price)}

    cache_path = cache_dir / f"{item_id}.json"
    intent_descriptions = cached_json_call(
        cache_path, lambda: writer.describe(category, attributes, budget_context)
    )
    return Item(item_id=item_id, features=product, intent_descriptions=intent_descriptions)
