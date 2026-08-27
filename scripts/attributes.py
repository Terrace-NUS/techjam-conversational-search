from __future__ import annotations

import random
import re

from starter.v1.catalog import (
    COLOR_RE,
    MATERIAL_RE,
    SIZE_RE,
    STYLE_RE,
    USE_CASE_RE,
    coarse_category,
    display_clean,
    field_text,
    flatten_phrases,
    unique_matches,
)

# Fixed bucket edges keep budget grounding stable across catalog price drift.
_PRICE_BAND_BOUNDS: tuple[tuple[float, float | None], ...] = (
    (0.0, 15.0),
    (15.0, 30.0),
    (30.0, 50.0),
    (50.0, 80.0),
    (80.0, None),
)
PRICE_BAND_LABELS = ("under_15", "15_to_30", "30_to_50", "50_to_80", "80_plus")
NEGATION_RE = re.compile(
    r"\b(?:"
    r"do not|don't|does not|doesn't|did not|didn't|"
    r"is not|isn't|are not|aren't|was not|wasn't|were not|weren't|"
    r"cannot|can't|won't|shouldn't|"
    r"no real|not real|not genuine|not made|not intended|not suitable|"
    r"not recommended|not for|free of|free from|instead of|"
    r"avoid|never"
    r")\b",
    re.IGNORECASE,
)

# Hypernyms that pass the material vocabulary but say nothing a shopper could filter on.
MATERIAL_STOPWORDS = frozenset({"fabric"})

# "gold"/"silver" live in both the material and the colour vocabulary, so a single
# span gets claimed by both. Only keep them where the surrounding words say which.
DUAL_SENSE_WORDS = frozenset({"gold", "silver", "rose gold"})
MATERIAL_SENSE_RE = re.compile(
    r"\b(?:material|metal|alloy|made of|made from|plated|plating|filled|sterling|solid|karat|carat|\d{1,2}k)\b",
    re.IGNORECASE,
)
COLOR_SENSE_RE = re.compile(r"\b(?:colou?rs?|tone[ds]?|finish|shade|plated)\b", re.IGNORECASE)
SENSE_WINDOW = 60

# Percentage blends ("96% Nylon, 4% Spandex") and bare delimiter lists
# ("Polyester,Cotton,Spandex") are the authoritative composition bullet; free-running
# marketing copy in the same listing routinely contradicts it.
COMPOSITION_PERCENT_RE = re.compile(r"\d{1,3}\s*%")
LABELLED_VALUE_PATTERNS = (
    r"\b{label}s?\s*[:：]\s*([^.;|\]\}}\n]{{1,80}})",
    r"[【\[]\s*{label}s?\s*[】\]]\s*([^.;|\]\}}\n]{{1,80}})",
)

# Size codes are conventionally upper-case; matching case-sensitively keeps stray
# lower-case letters out. Single letters only count inside a chart entry ('S (28"').
SIZE_CODE_RE = re.compile(r"(?<![A-Za-z0-9])([2-9]XL|XXX?L|X[SL]|[SML])(?![A-Za-z0-9])")
SIZE_RANGE_RE = re.compile(
    r"(?<![A-Za-z0-9])([2-9]XL|XXX?L|X[SL]|[SML])\s*(?:[-–—~/]|\bto\b)\s*"
    r"([2-9]XL|XXX?L|X[SL]|[SML])(?![A-Za-z0-9])"
)
SIZE_CHART_ENTRY_RE = re.compile(r"\s*[(:]\s*\d")
SIZE_CONTEXT_RE = re.compile(
    r"\b(?:size[ds]?|sizing|fit|fits|width|available in|comes in|runs)\b", re.IGNORECASE
)
# These double as ordinary adjectives ("wide straps", "medium control"), so they
# only count as sizes when something nearby is actually talking about sizing.
AMBIGUOUS_SIZE_WORDS = frozenset(
    {"wide", "narrow", "tall", "petite", "large", "medium", "small"}
)
SIZE_CONTEXT_WINDOW = 40

# A listing that names two members of one group is a multi-variant listing, not a
# garment that is both sleeveless and long-sleeved.
EXCLUSIVE_STYLE_GROUPS = (
    frozenset({"sleeveless", "long sleeve", "short sleeve"}),
    frozenset({"slim fit", "loose fit", "regular fit", "fitted", "oversized"}),
)


def price_band(price: object) -> str | None:
    if price in (None, ""):
        return None
    try:
        value = float(price)
    except (TypeError, ValueError):
        return None
    for (_, high), label in zip(_PRICE_BAND_BOUNDS, PRICE_BAND_LABELS):
        if high is None or value < high:
            return label
    return PRICE_BAND_LABELS[-1]


def price_band_bounds(label: str) -> tuple[float, float | None]:
    return _PRICE_BAND_BOUNDS[PRICE_BAND_LABELS.index(label)]


def band_reference_price(label: str, rng: random.Random) -> float:
    """A representative numeric price inside a band, for bands with no real product price (fakes)."""
    low, high = price_band_bounds(label)
    if high is None:
        high = low * 2.0
    return round(rng.uniform(low, high), 2)


def browsing_budget_ceiling(exact_price: float) -> float:
    """A ceiling clearly above the exact price, for a wide 'browsing' budget phrase."""
    ceiling = exact_price * 1.8
    step = 5.0 if ceiling < 50.0 else 10.0
    rounded = round(ceiling / step) * step
    return max(rounded, exact_price + step)


def item_category(product: dict) -> str:
    return coarse_category(product.get("categories") or [])


def _join_values(values: tuple[str, ...]) -> str:
    if len(values) == 1:
        return values[0]
    if len(values) == 2:
        return f"{values[0]} and {values[1]}"
    return f"{', '.join(values[:-1])}, and {values[-1]}"


def searchable_text(product: dict) -> str:
    parts = [
        display_clean(product.get("title") or ""),
        *flatten_phrases(product.get("features")),
        *flatten_phrases(product.get("details")),
        *flatten_phrases(product.get("description")),
        field_text(product.get("categories")),
        field_text(product.get("store")),
    ]
    return " ".join(part for part in parts if part)


def positive_text(text: str) -> str:
    """Remove sentences that describe prohibited or unsupported use cases."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return " ".join(sentence for sentence in sentences if not NEGATION_RE.search(sentence))


def use_case_evidence(text: str) -> str:
    """Keep complete use-case sentences, including their positive or negative polarity."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return " ".join(
        sentence.strip()
        for sentence in sentences
        if sentence.strip() and USE_CASE_RE.search(sentence)
    )


def extract_attributes(product: dict) -> dict[str, str]:
    """Grounded true attribute values for an item, keyed by ask_attribute name.

    Multiple regex matches (e.g. a "cotton/polyester/spandex" blend) are joined
    into one combined value instead of keeping only the first match, so the
    generated description can reflect the full blend rather than a fragment.
    """
    text = searchable_text(product)
    attributes: dict[str, str] = {"category": item_category(product)}

    materials = unique_matches(MATERIAL_RE, text)
    if materials:
        attributes["material"] = _join_values(materials)
    colors = unique_matches(COLOR_RE, text)
    if colors:
        attributes["color"] = _join_values(colors)
    sizes = unique_matches(SIZE_RE, text)
    if sizes:
        attributes["size"] = _join_values(sizes)
    styles = unique_matches(STYLE_RE, text)
    if styles:
        attributes["style"] = _join_values(styles)
    use_case_sentences = use_case_evidence(text)
    if use_case_sentences:
        attributes["use_case"] = use_case_sentences
    store = display_clean(product.get("store") or "")
    if store:
        attributes["brand"] = store
    band = price_band(product.get("price"))
    if band:
        attributes["budget"] = band
    return attributes
