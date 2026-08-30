from __future__ import annotations

import argparse
import copy
import heapq
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluator.local_evaluator import classify_constraint
from starter.agent import COLOR_RE, MATERIAL_RE, _card_constraints, _constraint_key, _searchable_text


def load_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def numeric(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def rating_bucket(value: object) -> str:
    number = numeric(value)
    if number is None:
        return "missing"
    if number < 3.5:
        return "below_3.5"
    if number < 4.0:
        return "3.5_to_3.99"
    if number < 4.5:
        return "4.0_to_4.49"
    return "4.5_plus"


def review_bucket(value: object) -> str:
    number = numeric(value)
    if number is None:
        return "missing"
    if number < 10:
        return "under_10"
    if number < 50:
        return "10_to_49"
    if number < 200:
        return "50_to_199"
    if number < 1000:
        return "200_to_999"
    return "1000_plus"


def price_bucket(value: object) -> str:
    number = numeric(value)
    if number is None:
        return "missing_or_invalid"
    if number <= 10:
        return "up_to_10"
    if number <= 25:
        return "10_to_25"
    if number <= 50:
        return "25_to_50"
    if number <= 100:
        return "50_to_100"
    return "over_100"


def fine_category(product: dict) -> str:
    values = product.get("categories") or []
    flattened: list[str] = []
    for value in values:
        flattened.extend(part.strip().casefold() for part in str(value).split(",") if part.strip())
    excluded = {"clothing", "clothing shoes & jewelry", "clothing, shoes & jewelry"}
    cleaned = [value for value in flattened if value not in excluded]
    return cleaned[-1] if cleaned else "unknown"


def product_features(product: dict) -> dict[str, str]:
    corpus = _searchable_text(product)
    material = MATERIAL_RE.search(corpus)
    color = COLOR_RE.search(corpus)
    constraints = _card_constraints(product)
    card_types = "-".join(classify_constraint(value) for value in constraints)
    return {
        "rating": rating_bucket(product.get("average_rating")),
        "reviews": review_bucket(product.get("rating_number")),
        "price": price_bucket(product.get("price")),
        "material": material.group(1).casefold() if material else "none",
        "color": color.group(1).casefold() if color else "none",
        "category": fine_category(product),
        "card_types": card_types or "none",
        "card_length": str(len(constraints)),
        "has_features": str(bool(product.get("features"))).lower(),
        "has_details": str(bool(product.get("details"))).lower(),
        "has_description": str(bool(product.get("description"))).lower(),
    }


def distributions(rows: list[dict], features: dict[str, dict[str, str]]) -> dict[str, Counter[str]]:
    result: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        values = features[str(row["parent_asin"])]
        for group, value in values.items():
            result[group][value] += 1
    return dict(result)


def learned_log_lifts(
    catalog_asins: list[str],
    target_asins: list[str],
    features: dict[str, dict[str, str]],
) -> dict[str, dict[str, float]]:
    catalog_rows = [{"parent_asin": asin} for asin in catalog_asins]
    target_rows = [{"parent_asin": asin} for asin in target_asins]
    catalog_dist = distributions(catalog_rows, features)
    target_dist = distributions(target_rows, features)
    result: dict[str, dict[str, float]] = {}
    for group, catalog_counts in catalog_dist.items():
        values = set(catalog_counts) | set(target_dist[group])
        weights: dict[str, float] = {}
        for value in values:
            target_count = target_dist[group][value]
            if target_count < 3:
                continue
            target_rate = (target_count + 1.0) / (len(target_asins) + len(values))
            catalog_rate = (catalog_counts[value] + 1.0) / (len(catalog_asins) + len(values))
            raw = math.log(target_rate / catalog_rate)
            shrinkage = target_count / (target_count + 8.0)
            weights[value] = round(max(-3.0, min(3.0, raw * shrinkage)), 6)
        result[group] = dict(sorted(weights.items()))
    return result


def prior_score(values: dict[str, str], weights: dict[str, dict[str, float]]) -> float:
    return sum(weights.get(group, {}).get(value, 0.0) for group, value in values.items())


def similarity(
    source_asin: str,
    candidate_asin: str,
    features: dict[str, dict[str, str]],
    constraint_sets: dict[str, set[str]],
    priors: dict[str, float],
) -> float:
    source = features[source_asin]
    candidate = features[candidate_asin]
    group_weights = {
        "category": 8.0,
        "material": 4.0,
        "color": 2.0,
        "rating": 2.0,
        "reviews": 2.0,
        "price": 1.5,
        "card_types": 3.0,
        "card_length": 1.0,
    }
    score = sum(weight for group, weight in group_weights.items() if source[group] == candidate[group])
    left, right = constraint_sets[source_asin], constraint_sets[candidate_asin]
    union = left | right
    if union:
        score += 6.0 * len(left & right) / len(union)
    score += 0.75 * priors[candidate_asin]
    return score


def select_similar_targets(
    public_samples: list[dict],
    catalog_asins: list[str],
    features: dict[str, dict[str, str]],
    constraint_sets: dict[str, set[str]],
    prior_scores: dict[str, float],
    neighbors_per_target: int = 9,
) -> dict[str, list[str]]:
    public_targets = {str(sample["ground_truth"]["parent_asin"]) for sample in public_samples}
    selected = set(public_targets)
    pools: dict[str, list[str]] = {}
    for target in public_targets:
        candidates = (asin for asin in catalog_asins if asin not in public_targets)
        best = heapq.nlargest(
            250,
            candidates,
            key=lambda asin: (similarity(target, asin, features, constraint_sets, prior_scores), asin),
        )
        pools[target] = best

    assignments: dict[str, list[str]] = {target: [] for target in public_targets}
    for _ in range(neighbors_per_target):
        for sample in public_samples:
            target = str(sample["ground_truth"]["parent_asin"])
            while pools[target] and pools[target][0] in selected:
                pools[target].pop(0)
            if not pools[target]:
                fallback = max(
                    (asin for asin in catalog_asins if asin not in selected),
                    key=lambda asin: (prior_scores[asin], asin),
                )
                chosen = fallback
            else:
                chosen = pools[target].pop(0)
            assignments[target].append(chosen)
            selected.add(chosen)
    return assignments


def percentage_rows(counts: Counter[str], total: int) -> dict[str, float]:
    return {key: round(value / total, 6) for key, value in sorted(counts.items())}


def top_shifts(
    catalog_dist: dict[str, Counter[str]],
    target_dist: dict[str, Counter[str]],
    catalog_total: int,
    target_total: int,
    limit: int = 12,
) -> list[dict]:
    rows: list[dict] = []
    for group in sorted(catalog_dist):
        for value in set(catalog_dist[group]) | set(target_dist[group]):
            catalog_rate = catalog_dist[group][value] / catalog_total
            target_rate = target_dist[group][value] / target_total
            rows.append({
                "group": group,
                "value": value,
                "catalog_rate": round(catalog_rate, 6),
                "public_rate": round(target_rate, 6),
                "delta": round(target_rate - catalog_rate, 6),
                "lift": None if catalog_rate == 0 else round(target_rate / catalog_rate, 3),
                "public_count": target_dist[group][value],
            })
    return sorted(rows, key=lambda row: abs(row["delta"]), reverse=True)[:limit]


def write_report(path: Path, summary: dict) -> None:
    lines = [
        "# Public targets vs. 50k catalog distribution",
        "",
        f"- Catalog products: {summary['catalog_count']:,}",
        f"- Public targets: {summary['public_count']:,}",
        f"- Expanded targets: {summary['expanded_count']:,}",
        f"- Unique expanded targets: {summary['expanded_unique_targets']:,}",
        "",
        "## Largest public-to-catalog distribution shifts",
        "",
        "| Group | Value | Catalog | Public | Delta | Lift | Public n |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary["top_shifts"]:
        lift = "n/a" if row["lift"] is None else f"{row['lift']:.3f}x"
        lines.append(
            f"| {row['group']} | {row['value']} | {row['catalog_rate']:.2%} | "
            f"{row['public_rate']:.2%} | {row['delta']:+.2%} | {lift} | {row['public_count']} |"
        )
    lines.extend([
        "",
        "## Full normalized distributions",
        "",
        "```json",
        json.dumps(summary["distributions"], indent=2, ensure_ascii=False),
        "```",
        "",
        "The expanded set keeps every public sample and adds nine unique catalog-nearest",
        "targets per public target. Scenario, difficulty, and profile mixtures are therefore",
        "replicated exactly at 10x scale. It is a public-like stress set, not organizer ground truth.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--public", default="data/public_set.jsonl")
    parser.add_argument(
        "--expanded",
        default="analysis/generated/expanded_public_like_2000.jsonl",
    )
    parser.add_argument("--priors", default="starter/target_priors.json")
    parser.add_argument("--summary", default="analysis/reports/public_catalog_distribution.json")
    parser.add_argument("--report", default="analysis/reports/public_catalog_distribution.md")
    args = parser.parse_args()

    catalog = load_jsonl(args.catalog)
    public_samples = load_jsonl(args.public)
    products = {str(product["parent_asin"]): product for product in catalog}
    catalog_asins = list(products)
    public_targets = [str(sample["ground_truth"]["parent_asin"]) for sample in public_samples]
    features = {asin: product_features(product) for asin, product in products.items()}
    constraint_sets = {
        asin: {_constraint_key(value) for value in _card_constraints(product)}
        for asin, product in products.items()
    }

    weights = learned_log_lifts(catalog_asins, public_targets, features)
    prior_scores = {asin: prior_score(features[asin], weights) for asin in catalog_asins}
    assignments = select_similar_targets(
        public_samples, catalog_asins, features, constraint_sets, prior_scores
    )

    expanded: list[dict] = [copy.deepcopy(sample) for sample in public_samples]
    for sample in public_samples:
        source_target = str(sample["ground_truth"]["parent_asin"])
        for index, asin in enumerate(assignments[source_target], start=1):
            clone = copy.deepcopy(sample)
            clone["sample_id"] = f"expanded_{sample['sample_id']}_{index:02d}"
            clone["ground_truth"] = {"parent_asin": asin}
            clone["synthetic_source_target"] = source_target
            expanded.append(clone)

    Path(args.expanded).write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in expanded),
        encoding="utf-8",
    )
    Path(args.priors).write_text(
        json.dumps({
            "version": 1,
            "source": "public_set distribution vs frozen catalog",
            "public_sample_count": len(public_samples),
            "weights": weights,
        }, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    catalog_dist = distributions([{"parent_asin": asin} for asin in catalog_asins], features)
    public_dist = distributions([{"parent_asin": asin} for asin in public_targets], features)
    expanded_targets = [str(row["ground_truth"]["parent_asin"]) for row in expanded]
    expanded_dist = distributions([{"parent_asin": asin} for asin in expanded_targets], features)
    summary = {
        "catalog_count": len(catalog_asins),
        "public_count": len(public_targets),
        "expanded_count": len(expanded),
        "expanded_unique_targets": len(set(expanded_targets)),
        "top_shifts": top_shifts(
            catalog_dist, public_dist, len(catalog_asins), len(public_targets)
        ),
        "distributions": {
            group: {
                "catalog": percentage_rows(catalog_dist[group], len(catalog_asins)),
                "public": percentage_rows(public_dist[group], len(public_targets)),
                "expanded": percentage_rows(expanded_dist[group], len(expanded_targets)),
            }
            for group in sorted(catalog_dist)
        },
    }
    Path(args.summary).write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    write_report(Path(args.report), summary)
    print(json.dumps({key: value for key, value in summary.items() if key != "distributions"}, indent=2))


if __name__ == "__main__":
    main()
