"""CPU-only protocol, resume, and independent-analysis tests for the Kev judge."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from posttraining_harness import kev_analysis, kev_judge


def pair(pair_id: str = "p1", a_text: str = "correct") -> dict:
    return {
        "pair_id": pair_id,
        "task": "Answer the question.",
        "reference": "The reference answer.",
        "answers": {"model_a": a_text, "model_b": "incorrect"},
        "candidate_a": "model_a",
        "candidate_b": "model_b",
    }


class FakePredictor:
    class ContextOverflow(Exception):
        pass

    def __init__(self) -> None:
        self.calls = []

    def __call__(self, request: dict, prefix=None) -> kev_judge.Prediction:
        self.calls.append((request, prefix))
        # Prefer the semantically correct answer regardless of its placement. The
        # dict insertion order still rotates, exercising the label-order protocol.
        if request["state"]["answer_a"] == "correct":
            distribution = {"answer_a": 0.8, "answer_b": 0.1, "tie": 0.1}
        else:
            distribution = {"answer_a": 0.1, "answer_b": 0.8, "tie": 0.1}
        return kev_judge.Prediction(distribution, object(), 100, 40)


class ProtocolTests(unittest.TestCase):
    def test_request_rotates_criteria_without_changing_mapping(self) -> None:
        original, mapping = kev_judge.request_for(pair(), ("model_a", "model_b"), 0)
        rotated, rotated_mapping = kev_judge.request_for(pair(), ("model_a", "model_b"), 1)
        self.assertEqual(list(original["questions"]["preference"]["criteria"]), ["answer_a", "answer_b", "tie"])
        self.assertEqual(list(rotated["questions"]["preference"]["criteria"]), ["answer_b", "tie", "answer_a"])
        self.assertEqual(mapping, rotated_mapping)
        self.assertEqual(rotated["state"]["answer_a"], "correct")

    def test_geometric_average_is_normalized_and_not_arithmetic(self) -> None:
        result = kev_judge.geometric_average(
            [{"a": 0.9, "b": 0.1}, {"a": 0.5, "b": 0.5}]
        )
        self.assertAlmostEqual(sum(result.values()), 1.0)
        self.assertGreater(result["a"], result["b"])
        self.assertNotAlmostEqual(result["a"], 0.7)

    def test_pair_is_scored_in_two_placements_and_three_rotations(self) -> None:
        predictor = FakePredictor()
        result = kev_judge.score_pair(pair(), predictor)
        self.assertEqual(len(predictor.calls), 6)
        self.assertEqual([(p["left"], p["right"]) for p in result["placements"]],
                         [("model_a", "model_b"), ("model_b", "model_a")])
        self.assertTrue(result["placement_consistent"])
        self.assertEqual(result["predicted_preference"], "model_a")
        self.assertAlmostEqual(result["continuous_credit_a"], 0.85)
        self.assertEqual(result["discrete_credit_a"], 1.0)

    def test_jsonl_reader_does_not_split_unicode_line_separators(self) -> None:
        record = pair(a_text="before\u2028after")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pairs.jsonl"
            path.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
            self.assertEqual(kev_judge.read_jsonl(path), [record])


class ResumeTests(unittest.TestCase):
    def test_resume_requires_an_in_order_prefix_bound_to_input_bytes(self) -> None:
        pairs = [pair("p1"), pair("p2")]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.jsonl"
            row = kev_judge.score_pair(pairs[0], FakePredictor())
            kev_judge.append_jsonl(path, row)
            self.assertEqual(list(kev_judge.read_completed(path, pairs)), ["p1"])

            changed = [pair("p1", a_text="changed"), pair("p2")]
            with self.assertRaisesRegex(ValueError, "different input bytes"):
                kev_judge.read_completed(path, changed)

    def test_resume_rejects_duplicates_and_out_of_order_rows(self) -> None:
        pairs = [pair("p1"), pair("p2")]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.jsonl"
            second = kev_judge.score_pair(pairs[1], FakePredictor())
            kev_judge.append_jsonl(path, second)
            with self.assertRaisesRegex(ValueError, "in-order prefix"):
                kev_judge.read_completed(path, pairs)
            kev_judge.append_jsonl(path, second)
            with self.assertRaisesRegex(ValueError, "duplicate"):
                kev_judge.read_completed(path, pairs)


class PinTests(unittest.TestCase):
    def test_checkpoint_files_are_hash_verified(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "adapter.bin").write_bytes(b"exact bytes")
            digest = kev_judge.sha256_file(root / "adapter.bin")
            with mock.patch.object(kev_judge, "KEV_CHECKPOINT_FILES_SHA256", {"adapter.bin": digest}):
                self.assertEqual(kev_judge.verify_checkpoint_files(root), {"adapter.bin": digest})
                (root / "adapter.bin").write_bytes(b"changed")
                with self.assertRaisesRegex(RuntimeError, "digest differs"):
                    kev_judge.verify_checkpoint_files(root)

    def test_source_commit_requires_a_clean_checkout(self) -> None:
        with mock.patch.object(
            kev_judge.subprocess,
            "check_output",
            side_effect=[kev_judge.KEV_SOURCE_COMMIT + "\n", "?? generated.txt\n"],
        ):
            with self.assertRaisesRegex(RuntimeError, "modifications"):
                kev_judge.verify_source_commit(Path("/source"))


class AnalysisTests(unittest.TestCase):
    def setUp(self) -> None:
        self.pairs = [pair("p1"), pair("p2")]
        self.rows = [kev_judge.score_pair(item, FakePredictor()) for item in self.pairs]

    def test_analyzer_recomputes_fields_and_is_deterministic(self) -> None:
        first = kev_analysis.analyze(
            self.pairs, self.rows, [], seed=7, resamples=200, max_rejections=0,
        )
        second = kev_analysis.analyze(
            self.pairs, self.rows, [], seed=7, resamples=200, max_rejections=0,
        )
        self.assertEqual(first, second)
        self.assertAlmostEqual(first["continuous_credit_a"], 0.85)
        self.assertEqual(first["directional_verdict"], "candidate_a")
        self.assertTrue(first["placement_argmax_consistency"])

    def test_analyzer_rejects_tampered_derived_field(self) -> None:
        self.rows[0]["continuous_credit_a"] = 0.0
        with self.assertRaisesRegex(ValueError, "continuous_credit_a"):
            kev_analysis.analyze(self.pairs, self.rows, [], seed=7, resamples=20, max_rejections=0)

    def test_analyzer_rejects_tampered_raw_distribution(self) -> None:
        self.rows[0]["placements"][0]["rotations"][0]["model_a"] = 0.7
        with self.assertRaises(ValueError):
            kev_analysis.analyze(self.pairs, self.rows, [], seed=7, resamples=20, max_rejections=0)

    def test_analyzer_aligns_an_optional_comparison_by_pair_id(self) -> None:
        second = pair("p2")
        second["answers"] = {"model_a": "incorrect", "model_b": "correct"}
        pairs = [pair("p1"), second]
        rows = [kev_judge.score_pair(item, FakePredictor()) for item in pairs]
        result = kev_analysis.analyze(
            pairs,
            rows,
            [],
            seed=7,
            resamples=20,
            max_rejections=0,
            comparison={"p2": 0.0, "p1": 1.0},
        )
        self.assertAlmostEqual(result["comparison_pearson"], 1.0)
        self.assertAlmostEqual(result["comparison_spearman"], 1.0)

    def test_analyzer_checks_serving_token_limits(self) -> None:
        self.rows[0]["placements"][0]["token_counts"][0]["packed"] = kev_judge.SERVE_MAX_PACKED + 1
        with self.assertRaisesRegex(ValueError, "token counts"):
            kev_analysis.analyze(self.pairs, self.rows, [], seed=7, resamples=20, max_rejections=0)

    def test_declared_rejection_policy_is_enforced(self) -> None:
        with self.assertRaisesRegex(ValueError, "exceeds"):
            kev_analysis.analyze(
                self.pairs[:1], [], [{"pair_id": "p1"}], seed=7, resamples=20, max_rejections=0
            )


if __name__ == "__main__":
    unittest.main()
