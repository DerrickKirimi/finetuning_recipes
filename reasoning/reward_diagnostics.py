"""Score grouped rollouts and evaluate reward models in GRPO advantage space."""

from __future__ import annotations

from collections import defaultdict
import math
from typing import Iterable

import numpy as np

from reasoning import env


ZERO_TOLERANCE = 1e-6


def enrich_structural_rewards(rows: list[dict]) -> list[dict]:
    """Copy rollout rows and add deterministic GRPO reward components."""
    enriched = []
    for row in rows:
        completion = str(row["completion"])
        reference = str(row["reference"])
        item = dict(row)
        item["structural_rewards"] = {
            "think_format_reward": env.score_think_format(completion),
            "output_datatype_reward": env.score_output_datatype(completion, reference),
            "output_schema_reward": env.score_output_schema(completion, reference),
            "doom_loop_reward": env.score_doom_loop(completion),
            "length_penalty_reward": env.score_length_penalty_batch(
                [completion], [reference]
            )[0],
        }
        enriched.append(item)
    return enriched


def add_semantic_scores(rows: list[dict], label: str, scorer, batch_size: int = 128) -> None:
    """Add one candidate's exact GRPO semantic component in place.

    Malformed completions receive zero, matching ``score_neuraltxt_batch``. Valid
    completions send only the text after ``</think>`` to the learned scorer.
    """
    valid_indices = []
    references = []
    responses = []
    for index, row in enumerate(rows):
        completion = str(row["completion"])
        if not env._has_expected_format(completion):
            continue
        _, response, _ = env._extract_think_content(completion)
        if not response:
            continue
        valid_indices.append(index)
        references.append(str(row["reference"]))
        responses.append(response)

    scores = scorer.score_batch(references, responses, batch_size=batch_size)
    if len(scores) != len(valid_indices):
        raise ValueError("reward scorer returned the wrong number of scores")
    for row in rows:
        row.setdefault("semantic_rewards", {})[label] = 0.0
    for index, score in zip(valid_indices, scores):
        value = float(score)
        if not math.isfinite(value):
            raise ValueError(f"non-finite reward score from {label}")
        rows[index]["semantic_rewards"][label] = value


def _midranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return ranks


def spearman(x: Iterable[float], y: Iterable[float]) -> float | None:
    left = np.asarray(list(x), dtype=np.float64)
    right = np.asarray(list(y), dtype=np.float64)
    if len(left) < 2 or len(left) != len(right):
        return None
    left_rank = _midranks(left)
    right_rank = _midranks(right)
    if np.std(left_rank) == 0 or np.std(right_rank) == 0:
        return None
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def roc_auc(scores: Iterable[float], labels: Iterable[int]) -> float | None:
    values = np.asarray(list(scores), dtype=np.float64)
    binary = np.asarray(list(labels), dtype=np.int8)
    positive = binary == 1
    n_pos = int(positive.sum())
    n_neg = int((~positive).sum())
    if len(values) != len(binary) or n_pos == 0 or n_neg == 0:
        return None
    ranks = _midranks(values)
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def group_records(rows: list[dict], label: str) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["group_id"]].append(row)
    records = []
    for group_id in sorted(grouped):
        group = sorted(grouped[group_id], key=lambda row: row["rollout_index"])
        if [row["rollout_index"] for row in group] != list(range(8)):
            raise ValueError(f"group {group_id} does not contain rollout indices 0..7")
        candidate = np.asarray(
            [row["semantic_rewards"][label] for row in group], dtype=np.float64
        )
        reference = np.asarray([row["reference_score"] for row in group], dtype=np.float64)
        if not np.isfinite(candidate).all() or not np.isfinite(reference).all():
            raise ValueError(f"group {group_id} contains non-finite scores")
        if np.any((reference < 0) | (reference > 1)):
            raise ValueError(f"group {group_id} reference scores must be in [0,1]")
        candidate_advantage = candidate - candidate.mean()
        reference_advantage = reference - reference.mean()
        absolute_errors = np.abs(candidate_advantage - reference_advantage)
        baseline_errors = np.abs(reference_advantage)
        records.append(
            {
                "group_id": group_id,
                "mae": float(absolute_errors.mean()),
                "baseline_mae": float(baseline_errors.mean()),
                "improvement": float((baseline_errors - absolute_errors).mean()),
                "candidate": candidate,
                "reference": reference,
            }
        )
    return records


def _quantile_interval(values: list[float]) -> list[float]:
    return [
        float(np.quantile(values, 0.025, method="linear")),
        float(np.quantile(values, 0.975, method="linear")),
    ]


def analyze_split(
    rows: list[dict],
    label: str,
    *,
    bootstrap_samples: int = 2000,
    seed: int = 3407,
) -> dict:
    records = group_records(rows, label)
    if not records or bootstrap_samples < 1:
        raise ValueError("analysis requires groups and at least one bootstrap sample")
    rng = np.random.default_rng(seed)
    mae_samples = []
    improvement_samples = []
    auc_samples = []
    spearman_samples = []
    for _ in range(bootstrap_samples):
        sampled = [records[index] for index in rng.integers(0, len(records), len(records))]
        mae_samples.append(float(np.mean([record["mae"] for record in sampled])))
        improvement_samples.append(
            float(np.mean([record["improvement"] for record in sampled]))
        )
        candidate = np.concatenate([record["candidate"] for record in sampled])
        reference = np.concatenate([record["reference"] for record in sampled])
        auc = roc_auc(candidate, reference >= 0.5)
        rho = spearman(candidate, reference)
        if auc is not None:
            auc_samples.append(auc)
        if rho is not None:
            spearman_samples.append(rho)

    candidate = np.concatenate([record["candidate"] for record in records])
    reference = np.concatenate([record["reference"] for record in records])
    auc = roc_auc(candidate, reference >= 0.5)
    rho = spearman(candidate, reference)
    return {
        "groups": len(records),
        "rollouts": len(candidate),
        "advantage_mae": float(np.mean([record["mae"] for record in records])),
        "advantage_mae_ci95": _quantile_interval(mae_samples),
        "constant_advantage_mae": float(
            np.mean([record["baseline_mae"] for record in records])
        ),
        "paired_mae_improvement": float(
            np.mean([record["improvement"] for record in records])
        ),
        "paired_mae_improvement_ci95": _quantile_interval(improvement_samples),
        "roc_auc_reference_ge_0_5": auc,
        "roc_auc_ci95": _quantile_interval(auc_samples) if auc_samples else None,
        "spearman": rho,
        "spearman_ci95": _quantile_interval(spearman_samples)
        if spearman_samples
        else None,
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": seed,
    }


def component_variance(rows: list[dict], selected_label: str) -> dict:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["group_id"]].append(row)
    component_names = sorted(next(iter(rows))["structural_rewards"])
    component_names.append("neuraltxt_reward")
    result = {}
    for component in component_names:
        variances = []
        ranges = []
        for group_id in sorted(grouped):
            group = grouped[group_id]
            if component == "neuraltxt_reward":
                values = [row["semantic_rewards"][selected_label] for row in group]
            else:
                values = [row["structural_rewards"][component] for row in group]
            variances.append(float(np.var(values)))
            ranges.append(float(np.max(values) - np.min(values)))
        active = [value > ZERO_TOLERANCE for value in ranges]
        result[component] = {
            "groups": len(variances),
            "active_groups": int(sum(active)),
            "active_fraction": float(np.mean(active)),
            "variance_median": float(np.median(variances)),
            "variance_mean": float(np.mean(variances)),
            "range_median": float(np.median(ranges)),
            "ranges": ranges,
            "variances": variances,
        }
    semantic = result["neuraltxt_reward"]
    structural_total_ranges = []
    aggregate_ranges = []
    for group_id in sorted(grouped):
        group = grouped[group_id]
        structural = np.asarray(
            [
                sum(
                    value * env.DEFAULT_REWARD_WEIGHTS.get(name, 1.0)
                    for name, value in row["structural_rewards"].items()
                )
                for row in group
            ]
        )
        semantic_values = np.asarray(
            [row["semantic_rewards"][selected_label] for row in group]
        )
        structural_total_ranges.append(float(np.ptp(structural)))
        aggregate_ranges.append(float(np.ptp(structural + semantic_values)))
    structural_active = np.asarray(structural_total_ranges) > ZERO_TOLERANCE
    aggregate_active = np.asarray(aggregate_ranges) > ZERO_TOLERANCE
    return {
        "components": result,
        "structural_total_active_fraction": float(structural_active.mean()),
        "aggregate_active_fraction": float(aggregate_active.mean()),
        "semantic_gate_pass": semantic["active_fraction"] >= 0.5,
        "aggregate_gate_pass": float(aggregate_active.mean()) >= 0.5,
        "zero_tolerance": ZERO_TOLERANCE,
    }
