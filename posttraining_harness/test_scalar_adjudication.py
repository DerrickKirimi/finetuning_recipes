import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from posttraining_harness import scalar_adjudication, scalar_judge


def groups():
    result = []
    for split in ("validation", "test"):
        for index in range(100):
            result.append(
                {
                    "group_id": f"{split}-{index:03d}",
                    "split": split,
                    "source_index": index + (0 if split == "validation" else 100),
                    "task": "question",
                    "reference": "correct",
                    "candidates": [
                        {
                            "candidate_id": f"{split}-{index:03d}:{rollout}",
                            "rollout_index": rollout,
                            "answer": f"answer {rollout}",
                        }
                        for rollout in range(8)
                    ],
                }
            )
    return result


def record(group, order, *, anchors=False):
    ratings = []
    for candidate in group["candidates"]:
        level = candidate["anchor_expected_level"] if anchors else candidate["rollout_index"] % 5
        ratings.append(
            {
                "candidate_id": candidate["candidate_id"],
                "score_level": level,
                "reference_score": level / 4,
                "rationale": "reason",
            }
        )
    return {
        "group_id": group["group_id"],
        "order": order,
        "outcome": "ok",
        "ratings": ratings,
        "model": scalar_adjudication.EXPECTED_MODEL,
        "model_version": "gemini-test-001",
        "prompt_version": scalar_judge.PROMPT_VERSION,
        "generation_config": scalar_adjudication.EXPECTED_GENERATION,
        "pricing": scalar_adjudication.EXPECTED_PRICING,
        "prompt_sha256": scalar_judge.fingerprint(
            scalar_judge.build_prompt(group, order, 20260915)[0]
        ),
        "charged_usd": 0.001,
    }


class ScalarAdjudicationTests(unittest.TestCase):
    def test_prepare_uses_first_ten_validation_groups_and_endpoint_anchors(self):
        pilot, calibration = scalar_adjudication.prepare(groups())
        self.assertEqual([row["group_id"] for row in pilot], [f"validation-{i:03d}" for i in range(10)])
        self.assertEqual(len(calibration), 10)
        self.assertEqual(
            [row["anchor_expected_level"] for row in calibration[0]["candidates"]],
            [4, 4, 4, 4, 0, 0, 0, 0],
        )

    def test_perfect_stable_pilot_passes_and_changed_order_fails(self):
        pilot, calibration = scalar_adjudication.prepare(groups())
        with TemporaryDirectory() as scratch:
            root = Path(scratch)
            pilot_log = root / "pilot.jsonl"
            anchor_log = root / "anchors.jsonl"
            pilot_records = [record(group, order) for group in pilot for order in ("forward", "reverse")]
            anchor_records = [record(group, order, anchors=True) for group in calibration for order in ("forward", "reverse")]
            scalar_adjudication.write_jsonl(pilot_log, pilot_records)
            scalar_adjudication.write_jsonl(anchor_log, anchor_records)
            passed = scalar_adjudication.analyze_pilot(pilot, pilot_log, calibration, anchor_log)
            self.assertTrue(passed["pass"])
            pilot_records[1]["ratings"][0]["score_level"] = 4
            pilot_records[1]["ratings"][0]["reference_score"] = 1.0
            scalar_adjudication.write_jsonl(pilot_log, pilot_records)
            failed = scalar_adjudication.analyze_pilot(pilot, pilot_log, calibration, anchor_log)
        self.assertFalse(failed["pass"])

    def test_production_requires_every_forward_identity_and_matching_version(self):
        all_groups = groups()
        records = [record(group, "forward") for group in all_groups]
        labels = [
            {
                "group_id": group["group_id"],
                "rollout_index": candidate["rollout_index"],
                "reference_score": candidate["rollout_index"] % 5 / 4,
                "orders": 1,
            }
            for group in all_groups
            for candidate in group["candidates"]
        ]
        with TemporaryDirectory() as scratch:
            root = Path(scratch)
            log = root / "production.jsonl"
            labels_path = root / "summary-labels.jsonl"
            summary_path = root / "summary.json"
            scalar_adjudication.write_jsonl(log, records)
            scalar_adjudication.write_jsonl(labels_path, labels)
            summary_path.write_text(
                json.dumps(
                    {
                        "groups_with_scores": 200,
                        "spent_usd": 0.2,
                        "labels_path": labels_path.name,
                        "labels_sha256": scalar_judge.judge.sha256_text(labels_path.read_text()),
                    }
                )
            )
            report = scalar_adjudication.verify_production(
                all_groups,
                log,
                summary_path,
                {"pass": True, "resolved_model_versions": ["gemini-test-001"]},
            )
        self.assertTrue(report["pass"])


if __name__ == "__main__":
    unittest.main()
