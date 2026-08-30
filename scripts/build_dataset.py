from __future__ import annotations

import argparse
import dataclasses
import json
import logging
from pathlib import Path

from .attributes import llm_extract_attributes
from .intent_description import build_item
from .llm_client import DeepSeekAttributeWriter
from .modification import build_modification
from .schema import Item, Modification

logger = logging.getLogger(__name__)
DATASET_NAME = "public_set_v2.jsonl"


def _load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _label_v2_samples(samples: list[dict]) -> list[tuple[dict, str, bool]]:
    """Resolve legacy scenarios once so v2 records carry explicit intent state."""
    override_index = 0
    labeled: list[tuple[dict, str, bool]] = []
    for sample in samples:
        scenario = sample.get("scenario_type")
        if scenario == "intent_override":
            intent = "buying" if override_index < 20 else "browsing"
            override_index += 1
            override = True
        elif scenario == "buying":
            intent, override = "buying", False
        else:
            intent, override = "browsing", False
        labeled.append((sample, intent, override))
    return labeled


def _v2_row(
    sample: dict,
    intent: str,
    override: bool,
    item: Item,
    modification: Modification | None,
) -> dict:
    legacy_free_sample = {
        key: value
        for key, value in sample.items()
        if key not in {"version", "scenario_type", "intent", "override"}
    }
    item_fields = dataclasses.asdict(item)
    if modification is None:
        modification_fields = {
            "fake_attributes": {},
            "correction_messages": {},
            "modify_turn": None,
        }
    else:
        modification_fields = dataclasses.asdict(modification)
        modification_fields.pop("item_id")
    return {
        "version": "v2",
        **legacy_free_sample,
        "intent": intent,
        "override": override,
        **item_fields,
        **modification_fields,
    }


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


def build_dataset(
    source_path: Path,
    catalog_path: Path,
    out_dir: Path,
    count: int,
    offset: int = 0,
    resume: bool = False,
) -> None:
    if count < 1:
        raise ValueError("count must be positive")
    if offset < 0:
        raise ValueError("offset must be non-negative")
    samples = _label_v2_samples(_load_jsonl(source_path))[offset : offset + count]

    dataset_path = out_dir / DATASET_NAME
    already_done = 0
    if resume and dataset_path.is_file():
        with dataset_path.open(encoding="utf-8") as handle:
            already_done = sum(1 for line in handle if line.strip())
    if already_done:
        logger.info("resuming: skipping %d already-generated items", already_done)
    samples = samples[already_done:]
    file_mode = "a" if already_done else "w"

    target_ids = {str(sample[0]["ground_truth"]["parent_asin"]) for sample in samples}
    products = _load_catalog_products(catalog_path, target_ids)

    writer = DeepSeekAttributeWriter()
    out_dir.mkdir(parents=True, exist_ok=True)
    attribute_cache_dir = out_dir / "cache" / "extracted_attribute"
    attribute_cache_dir.mkdir(parents=True, exist_ok=True)
    attribute_json_dir = out_dir / "attribute_json"
    attribute_json_dir.mkdir(parents=True, exist_ok=True)
    intent_cache_dir = out_dir / "cache" / "intent_desc"
    fake_cache_dir = out_dir / "cache" / "fake_desc"

    with dataset_path.open(file_mode, encoding="utf-8") as dataset_file:
        for sample, intent, override in samples:
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
            modification = build_modification(
                product, item.item_id, attributes, writer, fake_cache_dir
            )
            if modification is None:
                if override:
                    raise ValueError(f"override sample {sample['sample_id']} has no fakeable attributes")
                logger.info("%s has no fakeable attributes", item_id)
            dataset_file.write(
                json.dumps(
                    _v2_row(sample, intent, override, item, modification),
                    ensure_ascii=False,
                )
                + "\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build custom benchmark preprocessing data (intent descriptions + modifications)"
    )
    parser.add_argument("--source", default="data/public_set.jsonl")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--out-dir", default="data")
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
        help=f"Append after rows already written to {DATASET_NAME}.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    build_dataset(
        Path(args.source),
        Path(args.catalog),
        Path(args.out_dir),
        args.count,
        args.offset,
        args.resume,
    )


if __name__ == "__main__":
    main()
