"""Compact structured product text used by embedding-based reward scoring."""

import re

DESCRIPTION_MAX_CHARS = 500
TITLE_MAX_CHARS = 240
ATTRIBUTE_MAX_CHARS = 160
DETAIL_KEYWORDS = ("material", "fabric", "color", "colour", "size", "dimension", "capacity", "fit", "style", "feature", "function", "waterproof", "compatible", "use")
MARKETING_PHRASES = ("free shipping", "best seller", "best-seller", "limited time", "buy now", "special offer", "great gift", "perfect gift", "sale price")


def _values(value: object) -> list[str]:
    if isinstance(value, dict):
        return [f"{key}: {item}" for key, item in value.items() if item not in (None, "", [])]
    if isinstance(value, list):
        return [str(item) for item in value if item not in (None, "")]
    return [str(value)] if value not in (None, "") else []


def _clean(value: object, limit: int) -> str:
    text = re.sub(r"\s+", " ", " ".join(_values(value))).strip(" -;,.\t\n").casefold()
    for phrase in MARKETING_PHRASES:
        text = text.replace(phrase, " ")
    return re.sub(r"\s+", " ", text).strip(" -;,.\t\n")[:limit].rstrip()


def _unique(value: object, limit: int) -> list[str]:
    result, seen = [], set()
    for item in _values(value):
        text = _clean(item, limit)
        if text and text.casefold() not in seen:
            seen.add(text.casefold())
            result.append(text)
    return result


def structured_product_text(product: dict) -> str:
    parts = []
    if title := _clean(product.get("title"), TITLE_MAX_CHARS):
        parts.append(f"TITLE: {title}")
    if brand := _clean(product.get("store"), ATTRIBUTE_MAX_CHARS):
        parts.append(f"BRAND: {brand}")
    if categories := _unique(product.get("categories"), ATTRIBUTE_MAX_CHARS):
        parts.append(f"CATEGORY: {' | '.join(categories)}")
    if features := _unique(product.get("features"), ATTRIBUTE_MAX_CHARS):
        parts.append(f"FEATURES: {' | '.join(features)}")
    details = product.get("details")
    if isinstance(details, dict):
        attrs = [f"{_clean(k, 80)}: {_clean(v, ATTRIBUTE_MAX_CHARS)}" for k, v in details.items() if any(t in str(k).casefold() for t in DETAIL_KEYWORDS) and _clean(v, ATTRIBUTE_MAX_CHARS)]
        if attrs:
            parts.append("ATTRIBUTES: " + " | ".join(_unique(attrs, ATTRIBUTE_MAX_CHARS)))
    if product.get("price") not in (None, ""):
        parts.append(f"PRICE: {product['price']}")
    if description := _clean(product.get("description"), DESCRIPTION_MAX_CHARS):
        parts.append(f"DESCRIPTION: {description}")
    return "\n".join(parts)
