from __future__ import annotations

import argparse
import json
import pickle
import random
from pathlib import Path

from evaluator.local_evaluator import catalog_index, coarse_category, intent_candidates, load_jsonl
from rollout.gpu_simulator import SCENARIO_NAMES
from scripts.generate_rollout_data import POPULARITY_BOUNDS, _target_stratum


SCENARIO_COUNTS = {
    "buying": 320,
    "browsing": 320,
    "intent_override": 120,
    "boundary": 40,
}


def _sample_stratum_counts(weights: list[int], total: int) -> list[int]:
    raw = [total * weight / sum(weights) for weight in weights]
    counts = [int(value) for value in raw]
    for index in sorted(range(len(raw)), key=lambda i: raw[i] - counts[i], reverse=True)[: total - sum(counts)]:
        counts[index] += 1
    return counts


def _capacity_constrained_counts(weights: list[int], capacities: list[int], total: int) -> list[int]:
    if total > sum(capacities):
        raise ValueError(f"only {sum(capacities)} eligible targets available for {total} samples")
    desired = [total * weight / sum(weights) for weight in weights]
    counts = [min(int(value), capacity) for value, capacity in zip(desired, capacities)]
    while sum(counts) < total:
        eligible = [
            index for index, capacity in enumerate(capacities)
            if counts[index] < capacity
        ]
        index = max(eligible, key=lambda item: (desired[item] - counts[item], weights[item]))
        counts[index] += 1
    return counts


def generate_private_records(
    catalog_path: str | Path,
    public_set_path: str | Path,
    *,
    seed: int,
) -> tuple[list[tuple], tuple[str, ...], dict]:
    _, categories, products = catalog_index(catalog_path)
    public_targets = {str(sample["ground_truth"]["parent_asin"]) for sample in load_jsonl(public_set_path)}
    catalog_ids = tuple(products)
    stratum_entries: dict[tuple[int, bool], list[tuple[int, str, tuple[str, ...]]]] = {}
    eligible_count = 0
    for target_id, parent_asin in enumerate(catalog_ids):
        if parent_asin in public_targets:
            continue
        product = products[parent_asin]
        candidates = tuple(intent_candidates(product))
        if len(candidates) < 4:
            continue
        eligible_count += 1
        stratum_entries.setdefault(_target_stratum(product), []).append(
            (target_id, coarse_category(categories[parent_asin]), candidates)
        )

    public_counts: dict[tuple[int, bool], int] = {}
    for sample in load_jsonl(public_set_path):
        product = products[str(sample["ground_truth"]["parent_asin"])]
        stratum = _target_stratum(product)
        public_counts[stratum] = public_counts.get(stratum, 0) + 1
    strata = [stratum for stratum in sorted(public_counts) if stratum_entries.get(stratum)]
    quotas = _capacity_constrained_counts(
        [public_counts[stratum] for stratum in strata],
        [len(stratum_entries[stratum]) for stratum in strata],
        sum(SCENARIO_COUNTS.values()),
    )

    rng = random.Random(seed)
    sampled_entries: list[tuple[int, str, tuple[str, ...]]] = []
    for stratum, quota in zip(strata, quotas):
        pool = stratum_entries[stratum]
        if quota > len(pool):
            raise ValueError(f"not enough private targets in stratum {stratum}: {quota} requested, {len(pool)} available")
        sampled_entries.extend(rng.sample(pool, quota))
    rng.shuffle(sampled_entries)

    scenario_ids = [SCENARIO_NAMES.index(name) for name, count in SCENARIO_COUNTS.items() for _ in range(count)]
    rng.shuffle(scenario_ids)
    override_id = SCENARIO_NAMES.index("intent_override")
    records: list[tuple] = []
    for (target_id, category, candidates), scenario_id in zip(sampled_entries, scenario_ids):
        constraints = tuple(rng.sample(candidates, 4))
        override_turn = rng.choice((3, 4)) if scenario_id == override_id else 0
        records.append((target_id, scenario_id, category, constraints, override_turn))

    stats = {
        "num_samples": len(records),
        "scenario_counts": SCENARIO_COUNTS,
        "public_target_count": len(public_targets),
        "private_unique_target_count": len({record[0] for record in records}),
        "catalog_size": len(catalog_ids),
        "eligible_private_target_count": eligible_count,
        "popularity_bounds": POPULARITY_BOUNDS,
        "strata_quotas": [
            {
                "popularity_bin": stratum[0],
                "price_present": stratum[1],
                "public_count": public_counts[stratum],
                "private_quota": quota,
                "private_pool": len(stratum_entries[stratum]),
            }
            for stratum, quota in zip(strata, quotas)
        ],
    }
    return records, catalog_ids, stats


def write_jsonl(records: list[tuple], catalog_ids: tuple[str, ...], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for index, (target_id, scenario_id, category, constraints, override_turn) in enumerate(records):
            handle.write(json.dumps({
                "sample_id": f"private_{index:04d}",
                "target_parent_asin": catalog_ids[target_id],
                "scenario_type": SCENARIO_NAMES[scenario_id],
                "category": category,
                "hard_constraints": list(constraints[:2]),
                "soft_preferences": list(constraints[2:]),
                "override_turn": override_turn,
            }, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the 800-session private-style test set")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--public-set", default="data/public_set.jsonl")
    parser.add_argument("--output-dir", default="data/generated/private-test-800")
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records, catalog_ids, stats = generate_private_records(args.catalog, args.public_set, seed=args.seed)
    with (output_dir / "test.pkl").open("wb") as handle:
        pickle.dump(records, handle, protocol=pickle.HIGHEST_PROTOCOL)
    write_jsonl(records, catalog_ids, output_dir / "test.jsonl")
    metadata = {
        "format": "techjam-private-style-test-v1",
        "seed": args.seed,
        "record_schema": ["target_id", "scenario_id", "coarse_category", "constraints", "override_turn"],
        "scenario_names": SCENARIO_NAMES,
        "stats": stats,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
