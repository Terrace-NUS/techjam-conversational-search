from __future__ import annotations

import argparse
import dataclasses
import json
import logging
from pathlib import Path

from evaluator.local_evaluator import load_jsonl

from .attributes import llm_extract_attributes
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


def _load_ids(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    ids: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                ids.add(str(json.loads(line)["item_id"]))
    return ids


def backfill_modifications(out_dir: Path) -> None:
    """Generate modifications for items already in items.jsonl that don't have one yet."""
    items_path = out_dir / "items.jsonl"
    modifications_path = out_dir / "modifications.jsonl"
    attribute_json_dir = out_dir / "attribute_json"
    fake_cache_dir = out_dir / "cache" / "fake_desc"

    done_ids = _load_ids(modifications_path)
    writer = DeepSeekAttributeWriter()
    with (
        items_path.open(encoding="utf-8") as items_file,
        modifications_path.open("a", encoding="utf-8") as modifications_file,
    ):
        for line in items_file:
            if not line.strip():
                continue
            item = json.loads(line)
            item_id = str(item["item_id"])
            if item_id in done_ids:
                continue
            attributes = json.loads(
                (attribute_json_dir / f"{item_id}.json").read_text(encoding="utf-8")
            )["attributes"]
            try:
                modification = build_modification(
                    item["features"], item_id, attributes, writer, fake_cache_dir
                )
            except ValueError as error:
                logger.warning("skipping modification for %s: %s", item_id, error)
                continue
            if modification is None:
                logger.info("skipping modification for %s: no fakeable attributes found", item_id)
                continue
            modifications_file.write(json.dumps(dataclasses.asdict(modification), ensure_ascii=False) + "\n")


def build_dataset(
    source_path: Path,
    catalog_path: Path,
    out_dir: Path,
    count: int,
    offset: int = 0,
    resume: bool = False,
    skip_modifications: bool = False,
) -> None:
    if count < 1:
        raise ValueError("count must be positive")
    if offset < 0:
        raise ValueError("offset must be non-negative")
    samples = load_jsonl(source_path)[offset : offset + count]

    items_path = out_dir / "items.jsonl"
    modifications_path = out_dir / "modifications.jsonl"
    already_done = 0
    if resume and items_path.is_file():
        with items_path.open(encoding="utf-8") as handle:
            already_done = sum(1 for line in handle if line.strip())
    if already_done:
        logger.info("resuming: skipping %d already-generated items", already_done)
    samples = samples[already_done:]
    file_mode = "a" if already_done else "w"

    target_ids = {str(sample["ground_truth"]["parent_asin"]) for sample in samples}
    products = _load_catalog_products(catalog_path, target_ids)

    writer = DeepSeekAttributeWriter()
    out_dir.mkdir(parents=True, exist_ok=True)
    attribute_cache_dir = out_dir / "cache" / "extracted_attribute"
    attribute_cache_dir.mkdir(parents=True, exist_ok=True)
    attribute_json_dir = out_dir / "attribute_json"
    attribute_json_dir.mkdir(parents=True, exist_ok=True)
    intent_cache_dir = out_dir / "cache" / "intent_desc"
    fake_cache_dir = out_dir / "cache" / "fake_desc"

    with (
        items_path.open(file_mode, encoding="utf-8") as items_file,
        modifications_path.open(file_mode, encoding="utf-8") as modifications_file,
    ):
        for sample in samples:
            item_id = str(sample["ground_truth"]["parent_asin"])
            product = products.get(item_id)
            if product is None:
                logger.warning("skipping %s: not found in catalog", item_id)
                continue

            attributes, rejected = llm_extract_attributes(product, writer, attribute_cache_dir)
            for note in rejected:
                logger.info("%s dropped unverified %s", item_id, note)
            extracted_claims = json.loads(
                (attribute_cache_dir / f"{item_id}.json").read_text(encoding="utf-8")
            )
            (attribute_json_dir / f"{item_id}.json").write_text(
                json.dumps(
                    {
                        "item_id": item_id,
                        "attributes": attributes,
                        "extracted_claims": extracted_claims,
                        "rejected": rejected,
                    },
                    indent=2,
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

            item = build_item(product, attributes, writer, intent_cache_dir)
            items_file.write(json.dumps(dataclasses.asdict(item), ensure_ascii=False) + "\n")

            if skip_modifications:
                continue
            try:
                modification = build_modification(
                    product, item.item_id, attributes, writer, fake_cache_dir
                )
            except ValueError as error:
                logger.warning("skipping modification for %s: %s", item_id, error)
                continue
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
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Zero-based starting row in the source dataset (default: 0).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip items already written to items.jsonl and append instead of overwriting.",
    )
    parser.add_argument(
        "--skip-modifications",
        action="store_true",
        help="Only generate items.jsonl; do not call the LLM for modification data.",
    )
    parser.add_argument(
        "--modifications-only",
        action="store_true",
        help="Backfill modifications.jsonl for items already in items.jsonl that lack one, then exit.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if args.modifications_only:
        backfill_modifications(Path(args.out_dir))
        return
    build_dataset(
        Path(args.source),
        Path(args.catalog),
        Path(args.out_dir),
        args.count,
        args.offset,
        args.resume,
        args.skip_modifications,
    )


if __name__ == "__main__":
    main()
