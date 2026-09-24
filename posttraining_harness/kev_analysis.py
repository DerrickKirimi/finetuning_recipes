#!/usr/bin/env python3
"""Independently verify and analyze raw output from ``kev_judge.py``."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import statistics

from posttraining_harness.kev_judge import (
    SERVE_MAX_PACKED,
    SERVE_MAX_STATE,
    geometric_average,
    pair_digest,
    read_jsonl,
    valid_distribution,
)


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        raise ValueError("percentile of an empty sample")
    ordered = sorted(values)
    at = (len(ordered) - 1) * quantile
    lower, upper = math.floor(at), math.ceil(at)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - at) + ordered[upper] * (at - lower)


def bootstrap_mean_interval(
    values: list[float], rng: random.Random, resamples: int = 10_000
) -> list[float]:
    if not values or resamples <= 0:
        raise ValueError("bootstrap needs values and a positive resample count")
    count = len(values)
    means = [statistics.mean(values[rng.randrange(count)] for _ in range(count)) for _ in range(resamples)]
    return [percentile(means, 0.025), percentile(means, 0.975)]


def ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranked = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + end - 1) / 2
        for index in order[start:end]:
            ranked[index] = rank
        start = end
    return ranked


def correlation(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        raise ValueError("correlation requires equal samples with at least two values")
    left_mean, right_mean = statistics.mean(left), statistics.mean(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right))
    denominator = math.sqrt(
        sum((x - left_mean) ** 2 for x in left) * sum((y - right_mean) ** 2 for y in right)
    )
    if not denominator:
        raise ValueError("correlation is undefined for a constant sample")
    return numerator / denominator


def close(left: float, right: float, tolerance: float = 2e-7) -> bool:
    return abs(left - right) <= tolerance


def verify_row(row: dict, pair: dict) -> dict:
    """Recompute a row from its six raw probability distributions."""
    a, b = pair["candidate_a"], pair["candidate_b"]
    if row.get("pair_id") != pair["pair_id"] or row.get("pair_sha256") != pair_digest(pair):
        raise ValueError(f"row identity mismatch for {pair['pair_id']}")
    if row.get("candidate_a") != a or row.get("candidate_b") != b:
        raise ValueError(f"candidate mapping mismatch for {pair['pair_id']}")
    placements = row.get("placements")
    if not isinstance(placements, list) or len(placements) != 2:
        raise ValueError(f"expected two placements for {pair['pair_id']}")
    expected_placements = [(a, b), (b, a)]
    placement_credits = []
    placement_argmaxes = []
    rotation_ranges = []
    for placement, expected in zip(placements, expected_placements):
        if (placement.get("left"), placement.get("right")) != expected:
            raise ValueError(f"placement order mismatch for {pair['pair_id']}")
        rotations = placement.get("rotations")
        if not isinstance(rotations, list) or len(rotations) != 3:
            raise ValueError(f"expected three rotations for {pair['pair_id']}")
        token_counts = placement.get("token_counts")
        if not isinstance(token_counts, list) or len(token_counts) != 3:
            raise ValueError(f"expected three token-count records for {pair['pair_id']}")
        for counts in token_counts:
            if (
                not isinstance(counts.get("packed"), int)
                or not isinstance(counts.get("state"), int)
                or not 0 < counts["packed"] <= SERVE_MAX_PACKED
                or not 0 < counts["state"] <= SERVE_MAX_STATE
            ):
                raise ValueError(f"invalid token counts for {pair['pair_id']}: {counts}")
        for distribution in rotations:
            valid_distribution(distribution, {a, b, "tie"})
        recomputed = geometric_average(rotations)
        stored = placement.get("probabilities")
        valid_distribution(stored, {a, b, "tie"})
        if not all(close(recomputed[label], stored[label]) for label in recomputed):
            raise ValueError(f"stored geometric average differs for {pair['pair_id']}")
        argmax = max(recomputed, key=recomputed.get)
        if placement.get("argmax") != argmax:
            raise ValueError(f"stored placement argmax differs for {pair['pair_id']}")
        placement_argmaxes.append(argmax)
        placement_credits.append(recomputed[a] + 0.5 * recomputed["tie"])
        credits = [distribution[a] + 0.5 * distribution["tie"] for distribution in rotations]
        rotation_ranges.append(max(credits) - min(credits))

    continuous = statistics.mean(placement_credits)
    predicted = placement_argmaxes[0] if placement_argmaxes[0] == placement_argmaxes[1] else "tie"
    discrete = 1.0 if predicted == a else 0.5 if predicted == "tie" else 0.0
    consistent = placement_argmaxes[0] == placement_argmaxes[1]
    expected_fields = {
        "continuous_credit_a": continuous,
        "discrete_credit_a": discrete,
        "placement_consistent": consistent,
        "predicted_preference": predicted,
    }
    for name, expected in expected_fields.items():
        stored = row.get(name)
        equal = stored == expected if isinstance(expected, (bool, str)) else close(float(stored), expected)
        if not equal:
            raise ValueError(f"stored {name} differs for {pair['pair_id']}: {stored!r} != {expected!r}")
    return {
        "continuous": continuous,
        "discrete": discrete,
        "consistent": consistent,
        "placement_difference": abs(placement_credits[0] - placement_credits[1]),
        "rotation_ranges": rotation_ranges,
    }


def comparison_credits(path: Path) -> dict[str, float]:
    document = json.loads(path.read_text(encoding="utf-8"))
    rows = document.get("per_pair")
    if not isinstance(rows, list):
        raise ValueError("comparison JSON must contain a per_pair list")
    result = {}
    for row in rows:
        pair_id = row.get("pair_id")
        if not isinstance(pair_id, str) or pair_id in result:
            raise ValueError("comparison pair IDs must be unique strings")
        result[pair_id] = float(row["credit_model_a"])
    return result


def analyze(
    pairs: list[dict],
    rows: list[dict],
    rejected: list[dict],
    *,
    seed: int,
    resamples: int,
    max_rejections: int,
    comparison: dict[str, float] | None = None,
) -> dict:
    if len(rejected) > max_rejections:
        raise ValueError(f"{len(rejected)} rejections exceeds the declared maximum {max_rejections}")
    pair_by_id = {pair["pair_id"]: pair for pair in pairs}
    if len(pair_by_id) != len(pairs):
        raise ValueError("input pair IDs are not unique")
    candidate_pairs = {(pair["candidate_a"], pair["candidate_b"]) for pair in pairs}
    if len(candidate_pairs) != 1:
        raise ValueError("all rows in one aggregate must compare the same candidate names")
    rejected_ids = [item.get("pair_id") for item in rejected]
    if len(set(rejected_ids)) != len(rejected_ids) or not set(rejected_ids) <= set(pair_by_id):
        raise ValueError("rejected pair IDs are duplicate or unknown")
    expected_kept = [pair for pair in pairs if pair["pair_id"] not in set(rejected_ids)]
    if [row.get("pair_id") for row in rows] != [pair["pair_id"] for pair in expected_kept]:
        raise ValueError("scored rows are not exactly the non-rejected input rows in order")

    verified = [verify_row(row, pair) for row, pair in zip(rows, expected_kept)]
    if not verified:
        raise ValueError("no scored rows")
    continuous = [float(item["continuous"]) for item in verified]
    discrete = [float(item["discrete"]) for item in verified]
    placement_differences = [float(item["placement_difference"]) for item in verified]
    # Preserve placement as the unit of this sensitivity statistic. Collapsing
    # the two placements within each pair leaves the mean unchanged but changes
    # the median, which must be taken over all 2N placement-level ranges.
    rotation_ranges = [float(value) for item in verified for value in item["rotation_ranges"]]
    rng = random.Random(seed)
    continuous_interval = bootstrap_mean_interval(continuous, rng, resamples)
    discrete_interval = bootstrap_mean_interval(discrete, rng, resamples)
    result = {
        "schema_version": 1,
        "pairs_frozen": len(pairs),
        "pairs_scored": len(rows),
        "rejected_pair_ids": rejected_ids,
        "max_rejections": max_rejections,
        "candidate_a": expected_kept[0]["candidate_a"],
        "candidate_b": expected_kept[0]["candidate_b"],
        "bootstrap_seed": seed,
        "bootstrap_resamples": resamples,
        "continuous_credit_a": statistics.mean(continuous),
        "continuous_credit_a_ci95": continuous_interval,
        "discrete_credit_a": statistics.mean(discrete),
        "discrete_credit_a_ci95": discrete_interval,
        "placement_argmax_consistency": statistics.mean(float(item["consistent"]) for item in verified),
        "mean_absolute_placement_credit_difference": statistics.mean(placement_differences),
        "median_absolute_placement_credit_difference": statistics.median(placement_differences),
        "mean_option_rotation_credit_range": statistics.mean(rotation_ranges),
        "median_option_rotation_credit_range": statistics.median(rotation_ranges),
        "directional_verdict": (
            "candidate_a" if continuous_interval[0] > 0.5
            else "candidate_b" if continuous_interval[1] < 0.5
            else "inconclusive"
        ),
    }
    if comparison is not None:
        ids = [pair["pair_id"] for pair in expected_kept]
        missing = set(ids) - set(comparison)
        if missing:
            raise ValueError(f"comparison is missing {len(missing)} scored pair IDs")
        aligned = [comparison[pair_id] for pair_id in ids]
        result.update(
            comparison_credit_a=statistics.mean(aligned),
            comparison_pearson=correlation(continuous, aligned),
            comparison_spearman=correlation(ranks(continuous), ranks(aligned)),
        )
    return result


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--pairs", required=True, type=Path)
    result.add_argument("--rows", required=True, type=Path)
    result.add_argument("--summary", required=True, type=Path)
    result.add_argument("--comparison", type=Path, help="optional pairwise judge summary with per_pair credits")
    result.add_argument("--seed", type=int, default=20260923)
    result.add_argument("--resamples", type=int, default=10_000)
    result.add_argument("--max-rejections", type=int, default=0)
    result.add_argument("--output", required=True, type=Path)
    return result


def main() -> None:
    args = parser().parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    if not summary.get("pass") or summary.get("stage") != "complete":
        raise SystemExit("scorer summary is not complete and passing")
    pairs = read_jsonl(args.pairs)
    rows = read_jsonl(args.rows)
    rejected = list(summary.get("rejected") or [])
    result = analyze(
        pairs,
        rows,
        rejected,
        seed=args.seed,
        resamples=args.resamples,
        max_rejections=args.max_rejections,
        comparison=comparison_credits(args.comparison) if args.comparison else None,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
