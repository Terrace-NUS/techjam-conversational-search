from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import os
import pickle
import random
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path

from evaluator.local_evaluator import catalog_index, coarse_category, intent_candidates, load_jsonl
from rollout.gpu_simulator import SCENARIO_NAMES


SCENARIO_WEIGHTS = (40, 40, 15, 5)
INTENT_OVERRIDE = SCENARIO_NAMES.index("intent_override")
POPULARITY_BOUNDS = (100, 1_000, 5_000, 10_000, 50_000)
RECORD_SCHEMA = (
    "target_id",
    "scenario_id",
    "coarse_category",
    "constraints",  # first two are hard; last two are soft
    "override_turn",  # zero except intent_override
)


def _number(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _target_stratum(product: dict) -> tuple[int, bool]:
    rating_number = _number(product.get("rating_number")) or 0.0
    price_present = _number(product.get("price")) is not None
    return bisect.bisect_right(POPULARITY_BOUNDS, rating_number), price_present


def build_sampler(
    catalog_path: str | Path,
    public_set_path: str | Path,
) -> tuple[dict, tuple[str, ...], dict]:
    _, categories, products = catalog_index(catalog_path)
    catalog_ids = tuple(products)
    entries: list[tuple[int, str, tuple[str, ...]]] = []
    stratum_entries: dict[tuple[int, bool], list[int]] = defaultdict(list)

    for target_id, parent_asin in enumerate(catalog_ids):
        product = products[parent_asin]
        candidates = tuple(intent_candidates(product))
        if len(candidates) < 4:
            continue
        entry_id = len(entries)
        entries.append((target_id, coarse_category(categories[parent_asin]), candidates))
        stratum_entries[_target_stratum(product)].append(entry_id)

    public_counts: Counter[tuple[int, bool]] = Counter()
    for sample in load_jsonl(public_set_path):
        product = products[str(sample["ground_truth"]["parent_asin"])]
        public_counts[_target_stratum(product)] += 1

    strata = [stratum for stratum in sorted(public_counts) if stratum_entries.get(stratum)]
    if not strata:
        raise ValueError("no catalog products match the public target strata")
    sampler = {
        "entries": tuple(entries),
        "stratum_entries": tuple(tuple(stratum_entries[stratum]) for stratum in strata),
        "stratum_weights": tuple(public_counts[stratum] for stratum in strata),
    }
    stats = {
        "catalog_size": len(catalog_ids),
        "eligible_target_count": len(entries),
        "excluded_targets_with_fewer_than_four_constraints": len(catalog_ids) - len(entries),
        "popularity_bounds": POPULARITY_BOUNDS,
        "strata": [
            {
                "popularity_bin": stratum[0],
                "price_present": stratum[1],
                "public_count": public_counts[stratum],
                "eligible_catalog_count": len(stratum_entries[stratum]),
            }
            for stratum in strata
        ],
    }
    return sampler, catalog_ids, stats


def generate_records(sampler: dict, count: int, seed: int) -> list[tuple]:
    rng = random.Random(seed)
    entries = sampler["entries"]
    stratum_entries = sampler["stratum_entries"]
    stratum_ids = rng.choices(
        range(len(stratum_entries)),
        weights=sampler["stratum_weights"],
        k=count,
    )
    scenario_ids = rng.choices(range(len(SCENARIO_NAMES)), weights=SCENARIO_WEIGHTS, k=count)
    records: list[tuple] = []
    for stratum_id, scenario_id in zip(stratum_ids, scenario_ids):
        entry_id = rng.choice(stratum_entries[stratum_id])
        target_id, category, candidates = entries[entry_id]
        constraints = tuple(rng.sample(candidates, 4))
        override_turn = rng.choice((3, 4)) if scenario_id == INTENT_OVERRIDE else 0
        records.append((target_id, scenario_id, category, constraints, override_turn))
    return records


def _part_path(output_dir: str | Path, part_id: int) -> Path:
    return Path(output_dir) / f"part-{part_id:06d}.pkl"


def _load_pickle_count(path: Path) -> int:
    with path.open("rb") as handle:
        records = pickle.load(handle)
    if not isinstance(records, list):
        raise ValueError(f"{path} does not contain a record list")
    return len(records)


def _generate_part(
    sampler: dict,
    output_dir: str,
    part_id: int,
    count: int,
    seed: int,
    resume: bool,
) -> dict:
    output = _part_path(output_dir, part_id)
    if resume and output.exists():
        existing_count = _load_pickle_count(output)
        if existing_count != count:
            raise ValueError(f"{output} has {existing_count} records, expected {count}")
        return {"part_id": part_id, "count": count, "bytes": output.stat().st_size, "resumed": True}

    records = generate_records(sampler, count, seed + part_id)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile("wb", dir=output.parent, delete=False) as handle:
            temporary = Path(handle.name)
            pickle.dump(records, handle, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return {"part_id": part_id, "count": count, "bytes": output.stat().st_size, "resumed": False}


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def generate_pickles(args: argparse.Namespace) -> dict:
    try:
        import ray
    except ModuleNotFoundError as exc:
        raise RuntimeError("install project dependencies with `uv sync` to use Ray generation") from exc

    output_dir = Path(args.pkl_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    existing_parts = list(output_dir.glob("part-*.pkl"))
    if existing_parts and not args.resume:
        raise FileExistsError(f"{output_dir} already contains pickle parts; pass --resume to verify and continue")

    started = time.perf_counter()
    sampler, catalog_ids, sampling_stats = build_sampler(args.catalog, args.public_set)
    part_count = (args.num_samples + args.chunk_size - 1) // args.chunk_size
    metadata = {
        "format": "techjam-rollout-pickle-v1",
        "record_schema": RECORD_SCHEMA,
        "scenario_names": SCENARIO_NAMES,
        "scenario_weights": SCENARIO_WEIGHTS,
        "num_samples": args.num_samples,
        "chunk_size": args.chunk_size,
        "part_count": part_count,
        "seed": args.seed,
        "catalog_sha256": _sha256(args.catalog),
        "sampling": sampling_stats,
    }
    with (output_dir / "metadata.pkl").open("wb") as handle:
        pickle.dump({**metadata, "catalog_ids": catalog_ids}, handle, protocol=pickle.HIGHEST_PROTOCOL)

    init_options = {"include_dashboard": False, "ignore_reinit_error": True}
    if args.ray_address:
        init_options["address"] = args.ray_address
    elif args.workers:
        init_options["num_cpus"] = args.workers
    ray.init(**init_options)
    remote_generate = ray.remote(_generate_part)
    sampler_ref = ray.put(sampler)
    pending = []
    for part_id in range(part_count):
        count = min(args.chunk_size, args.num_samples - part_id * args.chunk_size)
        pending.append(remote_generate.remote(
            sampler_ref,
            str(output_dir),
            part_id,
            count,
            args.seed,
            args.resume,
        ))

    total_count = total_bytes = completed = 0
    try:
        while pending:
            ready, pending = ray.wait(pending, num_returns=min(32, len(pending)))
            for result in ray.get(ready):
                total_count += result["count"]
                total_bytes += result["bytes"]
                completed += 1
            print(
                f"generated {completed}/{part_count} parts, "
                f"{total_count:,}/{args.num_samples:,} samples, {total_bytes / 2**30:.2f} GiB",
                flush=True,
            )
    finally:
        ray.shutdown()

    if total_count != args.num_samples:
        raise RuntimeError(f"generated {total_count} records, expected {args.num_samples}")
    manifest = {
        **metadata,
        "pickle_bytes": total_bytes,
        "generation_seconds": round(time.perf_counter() - started, 3),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate synthetic TechJam rollout data")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--public-set", default="data/public_set.jsonl")
    parser.add_argument("--pkl-dir", default="data/generated/rollout-10m/pkl")
    parser.add_argument("--num-samples", type=int, default=10_000_000)
    parser.add_argument("--chunk-size", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--ray-address", default=None)
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.num_samples <= 0 or args.chunk_size <= 0:
        raise ValueError("num_samples and chunk_size must be positive")
    generate_pickles(args)


if __name__ == "__main__":
    main()
