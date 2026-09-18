"""Prepare and verify a gated grouped-scalar adjudication run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from posttraining_harness import judge, scalar_judge


PILOT_GROUPS = 10
EXACT_AGREEMENT_MIN = 0.75
ORDER_MAE_MAX = 0.125
ORDER_MAX_ERROR = 0.5
ANCHOR_EXACT_MIN = 0.95
ANCHOR_SEPARATION_MIN = 0.75
EXPECTED_MODEL = "gemini-3.5-flash-lite"
EXPECTED_PRICING = {"input_per_million": 0.30, "output_per_million": 2.50}
EXPECTED_GENERATION = scalar_judge.ScalarConfig(
    model=EXPECTED_MODEL,
    cap_usd=1.0,
    price_input_per_million=0.30,
    price_output_per_million=2.50,
).generation_config()


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(groups: list[dict], pilot_groups: int = PILOT_GROUPS) -> tuple[list[dict], list[dict]]:
    scalar_judge.validate_groups(groups)
    validation = [group for group in groups if group.get("split") == "validation"]
    if len(validation) != 100 or pilot_groups < 1 or pilot_groups > len(validation):
        raise ValueError("expected 100 validation groups and a valid pilot size")
    pilot = validation[:pilot_groups]
    calibration = []
    for group in pilot:
        candidates = []
        for index in range(8):
            expected = 4 if index < 4 else 0
            candidates.append(
                {
                    "candidate_id": f"{group['group_id']}:anchor-{index}",
                    "rollout_index": index,
                    "answer": group["reference"] if expected == 4 else "",
                    "anchor_expected_level": expected,
                }
            )
        calibration.append(
            {
                "group_id": f"anchor-{group['group_id']}",
                "split": "calibration",
                "source_index": group["source_index"],
                "task": group["task"],
                "reference": group["reference"],
                "candidates": candidates,
            }
        )
    scalar_judge.validate_groups(pilot)
    scalar_judge.validate_groups(calibration)
    return pilot, calibration


def latest_successes(groups: list[dict], log_path: Path) -> tuple[dict, list[dict]]:
    expected_groups = {group["group_id"]: group for group in groups}
    records = judge.read_log(log_path)
    successful = {}
    for record in records:
        group = expected_groups.get(record.get("group_id"))
        generation = record.get("generation_config", {})
        seed = generation.get("seed")
        order = record.get("order")
        prompt_matches = (
            group is not None
            and order in {"forward", "reverse"}
            and isinstance(seed, int)
            and record.get("prompt_version") == scalar_judge.PROMPT_VERSION
            and record.get("prompt_sha256")
            == scalar_judge.fingerprint(scalar_judge.build_prompt(group, order, seed)[0])
        )
        if (
            record.get("outcome") == "ok"
            and prompt_matches
        ):
            successful[(record["group_id"], order)] = record
    return successful, records


def configuration_identity(records: list[dict]) -> set[tuple]:
    return {
        (
            record.get("model"),
            json.dumps(record.get("generation_config"), sort_keys=True),
            json.dumps(record.get("pricing"), sort_keys=True),
            record.get("prompt_version"),
        )
        for record in records
        if record.get("outcome") == "ok"
    }


def analyze_pilot(
    pilot_groups: list[dict],
    pilot_log: Path,
    calibration_groups: list[dict],
    calibration_log: Path,
) -> dict:
    scalar_judge.validate_groups(pilot_groups)
    scalar_judge.validate_groups(calibration_groups)
    pilot_success, pilot_records = latest_successes(pilot_groups, pilot_log)
    anchor_success, anchor_records = latest_successes(calibration_groups, calibration_log)
    expected_pilot = {
        (group["group_id"], order)
        for group in pilot_groups
        for order in ("forward", "reverse")
    }
    expected_anchor = {
        (group["group_id"], order)
        for group in calibration_groups
        for order in ("forward", "reverse")
    }

    exact = 0
    errors = []
    for group in pilot_groups:
        forward = {
            row["candidate_id"]: row["score_level"]
            for row in pilot_success.get((group["group_id"], "forward"), {}).get("ratings", [])
        }
        reverse = {
            row["candidate_id"]: row["score_level"]
            for row in pilot_success.get((group["group_id"], "reverse"), {}).get("ratings", [])
        }
        for candidate in group["candidates"]:
            candidate_id = candidate["candidate_id"]
            if candidate_id in forward and candidate_id in reverse:
                difference = abs(forward[candidate_id] - reverse[candidate_id]) / 4
                errors.append(difference)
                exact += forward[candidate_id] == reverse[candidate_id]

    anchor_total = anchor_exact = 0
    separations = []
    calibration_by_id = {group["group_id"]: group for group in calibration_groups}
    for key, record in anchor_success.items():
        expected = {
            row["candidate_id"]: row["anchor_expected_level"]
            for row in calibration_by_id[key[0]]["candidates"]
        }
        positives = []
        negatives = []
        for rating in record.get("ratings", []):
            target = expected.get(rating["candidate_id"])
            if target is None:
                continue
            anchor_total += 1
            anchor_exact += rating["score_level"] == target
            (positives if target == 4 else negatives).append(rating["reference_score"])
        if len(positives) == len(negatives) == 4:
            separations.append(sum(positives) / 4 - sum(negatives) / 4)

    records = pilot_records + anchor_records
    identities = configuration_identity(records)
    model_versions = {
        record.get("model_version")
        for record in records
        if record.get("outcome") == "ok" and record.get("model_version")
    }
    forbidden = [
        record
        for record in records
        if record.get("outcome") in {"reservation_underflow", "blocked", "parse_failure"}
        or (record.get("outcome") == "http_error" and record.get("http_status") not in judge.RETRY_STATUSES)
    ]
    successful_records = [record for record in records if record.get("outcome") == "ok"]
    metrics = {
        "pilot_successful_calls": len(pilot_success),
        "pilot_ratings_compared": len(errors),
        "order_exact_agreement": exact / len(errors) if errors else None,
        "order_normalized_mae": sum(errors) / len(errors) if errors else None,
        "order_max_normalized_error": max(errors) if errors else None,
        "anchor_successful_calls": len(anchor_success),
        "anchor_ratings": anchor_total,
        "anchor_exact_accuracy": anchor_exact / anchor_total if anchor_total else None,
        "minimum_anchor_separation": min(separations) if separations else None,
        "spent_usd": judge.spent_usd(records),
    }
    checks = {
        "all_pilot_orders_complete": set(pilot_success) == expected_pilot,
        "all_anchor_orders_complete": set(anchor_success) == expected_anchor,
        "all_80_pilot_candidate_pairs_compared": len(errors) == 80,
        "order_exact_agreement": metrics["order_exact_agreement"] is not None
        and metrics["order_exact_agreement"] >= EXACT_AGREEMENT_MIN,
        "order_normalized_mae": metrics["order_normalized_mae"] is not None
        and metrics["order_normalized_mae"] <= ORDER_MAE_MAX,
        "order_max_error": metrics["order_max_normalized_error"] is not None
        and metrics["order_max_normalized_error"] <= ORDER_MAX_ERROR,
        "all_160_anchor_ratings_present": anchor_total == 160,
        "anchor_exact_accuracy": metrics["anchor_exact_accuracy"] is not None
        and metrics["anchor_exact_accuracy"] >= ANCHOR_EXACT_MIN,
        "anchor_separation_each_call": len(separations) == 20
        and metrics["minimum_anchor_separation"] >= ANCHOR_SEPARATION_MIN,
        "one_shared_configuration": len(identities) == 1,
        "exact_registered_model": {record.get("model") for record in successful_records}
        == {EXPECTED_MODEL},
        "exact_registered_generation": {
            json.dumps(record.get("generation_config"), sort_keys=True)
            for record in successful_records
        }
        == {json.dumps(EXPECTED_GENERATION, sort_keys=True)},
        "exact_registered_pricing": {
            json.dumps(record.get("pricing"), sort_keys=True)
            for record in successful_records
        }
        == {json.dumps(EXPECTED_PRICING, sort_keys=True)},
        "one_resolved_model_version": len(model_versions) == 1,
        "no_forbidden_failures": not forbidden,
        "pilot_spend_at_or_below_0_40": metrics["spent_usd"] <= 0.40,
    }
    return {
        "schema_version": 1,
        "pass": all(checks.values()),
        "checks": checks,
        "failed": sorted(name for name, ok in checks.items() if not ok),
        "thresholds": {
            "exact_agreement_min": EXACT_AGREEMENT_MIN,
            "order_mae_max": ORDER_MAE_MAX,
            "order_max_error": ORDER_MAX_ERROR,
            "anchor_exact_min": ANCHOR_EXACT_MIN,
            "anchor_separation_min": ANCHOR_SEPARATION_MIN,
        },
        "metrics": metrics,
        "resolved_model_versions": sorted(model_versions),
        "configuration_count": len(identities),
        "forbidden_failures": len(forbidden),
    }


def verify_production(groups: list[dict], log_path: Path, summary_path: Path, pilot: dict) -> dict:
    scalar_judge.validate_groups(groups)
    successes, records = latest_successes(groups, log_path)
    expected = {(group["group_id"], "forward") for group in groups}
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    labels_path = summary_path.with_name(summary["labels_path"])
    labels = [json.loads(line) for line in labels_path.read_text().splitlines() if line]
    identities = configuration_identity(records)
    versions = {
        record.get("model_version")
        for record in records
        if record.get("outcome") == "ok" and record.get("model_version")
    }
    forbidden = [
        record
        for record in records
        if record.get("outcome") in {"reservation_underflow", "blocked", "parse_failure"}
        or (record.get("outcome") == "http_error" and record.get("http_status") not in judge.RETRY_STATUSES)
    ]
    successful_records = [record for record in records if record.get("outcome") == "ok"]
    label_pairs = {(row.get("group_id"), row.get("rollout_index")) for row in labels}
    expected_pairs = {
        (group["group_id"], candidate["rollout_index"])
        for group in groups
        for candidate in group["candidates"]
    }
    checks = {
        "pilot_passed": pilot.get("pass") is True,
        "all_200_forward_calls_complete": set(successes) == expected,
        "no_reverse_calls": not any(key[1] == "reverse" for key in successes),
        "one_configuration": len(identities) == 1,
        "exact_registered_model": {record.get("model") for record in successful_records}
        == {EXPECTED_MODEL},
        "exact_registered_generation": {
            json.dumps(record.get("generation_config"), sort_keys=True)
            for record in successful_records
        }
        == {json.dumps(EXPECTED_GENERATION, sort_keys=True)},
        "exact_registered_pricing": {
            json.dumps(record.get("pricing"), sort_keys=True)
            for record in successful_records
        }
        == {json.dumps(EXPECTED_PRICING, sort_keys=True)},
        "same_resolved_model_as_pilot": versions == set(pilot.get("resolved_model_versions", [])),
        "no_forbidden_failures": not forbidden,
        "summary_counts": summary.get("groups_with_scores") == 200 and len(labels) == 1600,
        "one_score_per_candidate": all(row.get("orders") == 1 for row in labels),
        "label_identity": label_pairs == expected_pairs,
        "labels_hash": summary.get("labels_sha256") == judge.sha256_text(labels_path.read_text()),
        "production_spend_at_or_below_1_60": float(summary.get("spent_usd", 99)) <= 1.60,
    }
    return {
        "schema_version": 1,
        "pass": all(checks.values()),
        "checks": checks,
        "failed": sorted(name for name, ok in checks.items() if not ok),
        "groups": len(groups),
        "labels": len(labels),
        "spent_usd": summary.get("spent_usd"),
        "resolved_model_versions": sorted(versions),
        "forbidden_failures": len(forbidden),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--groups", type=Path, required=True)
    prep.add_argument("--pilot", type=Path, required=True)
    prep.add_argument("--calibration", type=Path, required=True)
    analyze = sub.add_parser("analyze-pilot")
    analyze.add_argument("--pilot", type=Path, required=True)
    analyze.add_argument("--pilot-log", type=Path, required=True)
    analyze.add_argument("--calibration", type=Path, required=True)
    analyze.add_argument("--calibration-log", type=Path, required=True)
    analyze.add_argument("--out", type=Path, required=True)
    verify = sub.add_parser("verify-production")
    verify.add_argument("--groups", type=Path, required=True)
    verify.add_argument("--log", type=Path, required=True)
    verify.add_argument("--summary", type=Path, required=True)
    verify.add_argument("--pilot-result", type=Path, required=True)
    verify.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.command == "prepare":
        groups = scalar_judge.read_groups(args.groups)
        pilot, calibration = prepare(groups)
        write_jsonl(args.pilot, pilot)
        write_jsonl(args.calibration, calibration)
        result = {
            "schema_version": 1,
            "groups_sha256": file_sha256(args.groups),
            "pilot_groups": len(pilot),
            "pilot_sha256": file_sha256(args.pilot),
            "calibration_groups": len(calibration),
            "calibration_sha256": file_sha256(args.calibration),
        }
    elif args.command == "analyze-pilot":
        result = analyze_pilot(
            scalar_judge.read_groups(args.pilot), args.pilot_log,
            scalar_judge.read_groups(args.calibration), args.calibration_log,
        )
    else:
        result = verify_production(
            scalar_judge.read_groups(args.groups), args.log, args.summary,
            json.loads(args.pilot_result.read_text()),
        )
    if hasattr(args, "out"):
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    else:
        manifest = args.pilot.with_name("pilot-input-manifest.json")
        manifest.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("pass", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
