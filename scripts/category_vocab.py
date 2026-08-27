from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from starter.v1.catalog import ATTRIBUTES

from .attributes import extract_attributes, item_category

# budget uses price bands (see modification.FAKE_VALUE_VOCAB), not catalog-observed text values.
FAKE_VOCAB_ATTRIBUTES = tuple(attribute for attribute in ATTRIBUTES if attribute != "budget")


def build_category_vocab(catalog_path: Path) -> dict[str, dict[str, list[str]]]:
    """Scan the catalog once to learn which attribute values actually occur per category."""
    vocab: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    with catalog_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            product = json.loads(line)
            category = item_category(product)
            attributes = extract_attributes(product)
            for attribute in FAKE_VOCAB_ATTRIBUTES:
                value = attributes.get(attribute)
                if value:
                    vocab[category][attribute].add(value)
    return {
        category: {attribute: sorted(values) for attribute, values in per_attribute.items()}
        for category, per_attribute in vocab.items()
    }


def load_or_build_category_vocab(catalog_path: Path, cache_path: Path) -> dict[str, dict[str, list[str]]]:
    if cache_path.is_file():
        return json.loads(cache_path.read_text(encoding="utf-8"))
    vocab = build_category_vocab(catalog_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(vocab, ensure_ascii=False) + "\n", encoding="utf-8")
    return vocab
