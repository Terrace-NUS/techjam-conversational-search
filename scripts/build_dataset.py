from __future__ import annotations

import argparse
import dataclasses
import json
import logging
from pathlib import Path

from evaluator.local_evaluator import load_jsonl

from .category_vocab import load_or_build_category_vocab
from .intent_description import build_item
from .llm_client import DeepSeekAttributeWriter
from .modification import build_modification

logger = logging.getLogger(__name__)


def _load_catalog_products(catalog_path: Path, target_ids: set[str]) -> dict[str, dict]:
    products: dict[str, dict] = {}
    with catalog_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            product = json.loads(line)
            asin = str(product.get("parent_asin"))
            if asin in target_ids:
                products[asin] = product
                if len(products) == len(target_ids):
                    break
    return products


def build_dataset(source_path: Path, catalog_path: Path, out_dir: Path, count: int) -> None:
    samples = load_jsonl(source_path)[:count]
    target_ids = {str(sample["ground_truth"]["parent_asin"]) for sample in samples}
    products = _load_catalog_products(catalog_path, target_ids)

    writer = DeepSeekAttributeWriter()
    out_dir.mkdir(parents=True, exist_ok=True)
    intent_cache_dir = out_dir / "cache" / "intent_desc"
    fake_cache_dir = out_dir / "cache" / "fake_desc"
    category_vocab = load_or_build_category_vocab(catalog_path, out_dir / "cache" / "category_vocab.json")

    items_path = out_dir / "items.jsonl"
    modifications_path = out_dir / "modifications.jsonl"
    with (
        items_path.open("w", encoding="utf-8") as items_file,
        modifications_path.open("w", encoding="utf-8") as modifications_file,
    ):
        for sample in samples:
            item_id = str(sample["ground_truth"]["parent_asin"])
            product = products.get(item_id)
            if product is None:
                logger.warning("skipping %s: not found in catalog", item_id)
                continue

            item = build_item(product, writer, intent_cache_dir)
            items_file.write(json.dumps(dataclasses.asdict(item), ensure_ascii=False) + "\n")

            modification = build_modification(product, item.item_id, writer, fake_cache_dir, category_vocab)
            if modification is None:
                logger.info("skipping modification for %s: no fakeable attributes found", item_id)
                continue
            modifications_file.write(json.dumps(dataclasses.asdict(modification), ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build custom benchmark preprocessing data (intent descriptions + modifications)"
    )
    parser.add_argument("--source", default="data/public_set.jsonl")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--out-dir", default="data/custom")
    parser.add_argument("--count", type=int, default=10)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    build_dataset(Path(args.source), Path(args.catalog), Path(args.out_dir), args.count)


if __name__ == "__main__":
    main()
