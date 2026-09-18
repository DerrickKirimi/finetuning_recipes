"""Select a reward ablation in advantage space and report its variance gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from reasoning.reward_diagnostics import analyze_split, component_variance


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def atomic_json(path: Path, value) -> None:
    staging = path.with_suffix(path.suffix + ".partial")
    staging.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(staging, path)


def label_sort_key(label: str):
    if label.startswith("k") and label[1:].isdigit():
        return (0, int(label[1:]))
    return (1, label)


def attach_reference_scores(rows: list[dict], labels: list[dict]) -> None:
    label_map = {}
    for label in labels:
        identity = (label.get("group_id"), label.get("rollout_index"))
        if identity in label_map:
            raise ValueError(f"duplicate adjudication identity {identity}")
        label_map[identity] = label.get("reference_score")
    row_ids = {(row.get("group_id"), row.get("rollout_index")) for row in rows}
    if set(label_map) != row_ids:
        raise ValueError("adjudication identities do not exactly match scored rollouts")
    for row in rows:
        row["reference_score"] = label_map[(row["group_id"], row["rollout_index"])]


def analyze(rows: list[dict], labels: list[dict], *, bootstrap_samples: int, seed: int) -> dict:
    if not rows or len(rows) != 1600:
        raise ValueError("expected the complete 1,600-row scored corpus")
    attach_reference_scores(rows, labels)
    variants = sorted(rows[0].get("semantic_rewards", {}), key=label_sort_key)
    if not variants or any(set(row.get("semantic_rewards", {})) != set(variants) for row in rows):
        raise ValueError("every row must carry the same reward variants")
    validation = [row for row in rows if row.get("split") == "validation"]
    test = [row for row in rows if row.get("split") == "test"]
    if len(validation) != 800 or len(test) != 800:
        raise ValueError("expected 800 validation and 800 test rollouts")

    validation_metrics = {
        label: analyze_split(
            validation,
            label,
            bootstrap_samples=bootstrap_samples,
            seed=seed,
        )
        for label in variants
    }
    selected = min(
        variants,
        key=lambda label: (validation_metrics[label]["advantage_mae"], label_sort_key(label)),
    )
    test_metrics = analyze_split(
        test,
        selected,
        bootstrap_samples=bootstrap_samples,
        seed=seed + 1,
    )
    variance = {
        "validation": component_variance(validation, selected),
        "test": component_variance(test, selected),
    }
    item9_pass = (
        test_metrics["advantage_mae_ci95"][1] <= 0.10
        and test_metrics["paired_mae_improvement_ci95"][0] > 0
    )
    item95_pass = (
        variance["validation"]["semantic_gate_pass"]
        and variance["validation"]["aggregate_gate_pass"]
    )
    return {
        "schema_version": 1,
        "pass": item9_pass and item95_pass,
        "item9_pass": item9_pass,
        "item9_5_pass": item95_pass,
        "selected_variant": selected,
        "selection_rule": "minimum validation advantage MAE; exact ties choose smallest numeric k",
        "validation": validation_metrics,
        "test_selected_only": test_metrics,
        "component_variance": variance,
        "bootstrap_samples": bootstrap_samples,
        "validation_seed": seed,
        "test_seed": seed + 1,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()

    rows = read_jsonl(args.scores)
    labels = read_jsonl(args.labels)
    report = analyze(
        rows,
        labels,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    report.update({
        "scores_sha256": sha256(args.scores),
        "labels_sha256": sha256(args.labels),
    })
    atomic_json(args.output, report)
    print(
        json.dumps(
            {
                "pass": report["pass"],
                "item9_pass": report["item9_pass"],
                "item9_5_pass": report["item9_5_pass"],
                "selected_variant": report["selected_variant"],
            },
            indent=2,
        )
    )
    return 0 if report["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
