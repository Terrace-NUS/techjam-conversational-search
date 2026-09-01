from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from scripts.llm_client import DeepSeekAttributeWriter, cached_json_call
from scripts.schema import clue_text


DESCRIPTIVE_ATTRIBUTES = {"feature", "style", "use_case", "other"}


def _regenerate_row(
    row: dict,
    writer: DeepSeekAttributeWriter,
    cache_dir: Path,
) -> dict:
    buying = dict(row.get("intent_descriptions", {}).get("buying") or {})
    if not buying:
        raise ValueError(f"{row.get('sample_id')}: missing buying intent descriptions")
    category = clue_text(buying.get("category")) or "clothing item"
    item_id = str(row["item_id"])
    regenerated: dict[str, list[str]] = {}
    for attribute, current_clues in buying.items():
        source_value = clue_text(current_clues)
        if not source_value:
            continue
        if attribute not in DESCRIPTIVE_ATTRIBUTES:
            regenerated[attribute] = (
                list(current_clues) if isinstance(current_clues, list) else [source_value]
            )
            continue
        cache_path = cache_dir / "attributes" / item_id / f"{attribute}.json"
        try:
            payload = cached_json_call(
                cache_path,
                lambda attribute=attribute, source_value=source_value: {
                    "clues": writer.describe_attribute(
                        category,
                        attribute,
                        source_value,
                        "buying",
                    )
                },
            )
            clues = writer._parse_clue_value(payload)
            writer._validate_stage_clues(
                attribute, source_value, "buying", clues, None
            )
        except Exception as error:
            try:
                clues = writer.describe_attribute(
                    category,
                    attribute,
                    source_value,
                    "buying",
                )
                writer._validate_stage_clues(
                    attribute, source_value, "buying", clues, None
                )
                payload = {"clues": clues}
            except Exception:
                fallback = (
                    row.get("intent_descriptions", {}).get("browsing") or {}
                ).get(attribute)
                fallback_clues = (
                    [str(clue).strip() for clue in fallback if str(clue).strip()]
                    if isinstance(fallback, list)
                    else [clue_text(fallback)]
                )
                fallback_clues = [clue for clue in fallback_clues if clue]
                if not fallback_clues:
                    raise RuntimeError(
                        f"{row.get('sample_id')}.{attribute}: {source_value!r}"
                    ) from error
                try:
                    writer._validate_stage_clues(
                        attribute, source_value, "buying", fallback_clues, None
                    )
                    payload = {"clues": fallback_clues}
                except ValueError:
                    try:
                        fallback_clues = writer.describe_attribute(
                            category,
                            attribute,
                            clue_text(fallback_clues),
                            "buying",
                        )
                        writer._validate_stage_clues(
                            attribute, source_value, "buying", fallback_clues, None
                        )
                        payload = {"clues": fallback_clues}
                    except Exception as fallback_error:
                        discovery = (
                            row.get("intent_descriptions", {}).get("discovery") or {}
                        ).get(attribute)
                        discovery_clues = (
                            [
                                str(clue).strip()
                                for clue in discovery
                                if str(clue).strip()
                            ]
                            if isinstance(discovery, list)
                            else [clue_text(discovery)]
                        )
                        discovery_clues = [clue for clue in discovery_clues if clue]
                        try:
                            if not discovery_clues:
                                raise ValueError("missing discovery fallback")
                            writer._validate_stage_clues(
                                attribute,
                                source_value,
                                "buying",
                                discovery_clues,
                                None,
                            )
                            payload = {"clues": discovery_clues}
                        except ValueError:
                            raise RuntimeError(
                                f"{row.get('sample_id')}.{attribute}: {source_value!r}"
                            ) from fallback_error
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        clues = writer._parse_clue_value(payload)
        regenerated[attribute] = clues

    updated = {
        **row,
        "intent_descriptions": {**row["intent_descriptions"], "buying": regenerated},
    }
    if not row.get("override"):
        return updated

    corrections = {
        attribute: dict(stages)
        for attribute, stages in (row.get("correction_messages") or {}).items()
    }
    for attribute, stages in corrections.items():
        true_clues = regenerated.get(attribute)
        fake_clues = clue_text(
            (row.get("fake_attributes") or {}).get(attribute, {}).get("buying")
        )
        if not true_clues or not fake_clues:
            continue
        clue_hash = hashlib.sha256(clue_text(true_clues).encode("utf-8")).hexdigest()[:12]
        cache_path = (
            cache_dir / "corrections" / item_id / f"{attribute}-{clue_hash}.json"
        )
        payload = cached_json_call(
            cache_path,
            lambda attribute=attribute, fake_clues=fake_clues, true_clues=true_clues: {
                "message": writer.correct_attribute(
                    attribute,
                    "buying",
                    [fake_clues],
                    true_clues,
                    [],
                )
            },
        )
        message = writer._parse_correction_value(payload)
        writer._validate_correction(message, true_clues, [])
        stages["buying"] = [message]
    updated["correction_messages"] = corrections
    return updated


def regenerate_buying(
    dataset_path: Path,
    cache_dir: Path,
    workers: int,
    source_ref: str | None = None,
) -> None:
    if workers < 1:
        raise ValueError("workers must be positive")
    source_text = (
        subprocess.check_output(
            ["git", "show", source_ref],
            text=True,
            encoding="utf-8",
        )
        if source_ref
        else dataset_path.read_text(encoding="utf-8")
    )
    rows = [
        json.loads(line)
        for line in source_text.splitlines()
        if line.strip()
    ]
    writer = DeepSeekAttributeWriter()
    updated: list[dict | None] = [None] * len(rows)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_regenerate_row, row, writer, cache_dir): index
            for index, row in enumerate(rows)
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            updated[futures[future]] = future.result()
            print(
                f"\rRegenerated buying attributes {completed}/{len(rows)}",
                end="",
                file=sys.stderr,
                flush=True,
            )
    if rows:
        print(file=sys.stderr)
    if any(row is None for row in updated):
        raise RuntimeError("buying regeneration did not produce every dataset row")
    temporary_path = dataset_path.with_suffix(dataset_path.suffix + ".tmp")
    temporary_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in updated),
        encoding="utf-8",
    )
    temporary_path.replace(dataset_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate only v2 buying descriptions")
    parser.add_argument("--dataset", type=Path, default=Path("data/public_set_v2.jsonl"))
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/cache/buying_desc_v2"),
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--source-ref",
        required=True,
        help="Immutable source git object, e.g. HEAD:data/public_set_v2.jsonl.",
    )
    args = parser.parse_args()
    regenerate_buying(args.dataset, args.cache_dir, args.workers, args.source_ref)


if __name__ == "__main__":
    main()
