import json
import unittest

from reasoning.reward_diagnostics import (
    add_semantic_scores,
    analyze_split,
    component_variance,
    enrich_structural_rewards,
    roc_auc,
    spearman,
)
from reasoning.analyze_reward_diagnostics import analyze, attach_reference_scores, label_sort_key
from reasoning.score_reward_diagnostics import validate_rollouts
from reasoning.build_scalar_judge_groups import build_groups, final_answer


def rows(groups=2):
    result = []
    for group in range(groups):
        for rollout in range(8):
            result.append(
                {
                    "group_id": f"validation-{group:03d}",
                    "split": "validation",
                    "source_index": group,
                    "rollout_index": rollout,
                    "reference": "answer",
                    "completion": f"<think>x</think> answer {rollout}",
                    "reference_score": rollout / 7,
                }
            )
    return result


class FakeScorer:
    def score_batch(self, references, responses, batch_size=128):
        del references, batch_size
        return [int(response.rsplit(" ", 1)[-1]) / 7 for response in responses]


class RewardDiagnosticTests(unittest.TestCase):
    def test_perfect_candidate_has_zero_advantage_error(self):
        data = enrich_structural_rewards(rows())
        add_semantic_scores(data, "k0", FakeScorer())
        report = analyze_split(data, "k0", bootstrap_samples=50)
        self.assertAlmostEqual(report["advantage_mae"], 0.0)
        self.assertEqual(report["roc_auc_reference_ge_0_5"], 1.0)
        self.assertAlmostEqual(report["spearman"], 1.0)
        self.assertGreater(report["paired_mae_improvement_ci95"][0], 0)

    def test_constant_candidate_matches_zero_advantage_baseline(self):
        data = enrich_structural_rewards(rows())
        for row in data:
            row["semantic_rewards"] = {"k0": 0.4}
        report = analyze_split(data, "k0", bootstrap_samples=20)
        self.assertAlmostEqual(report["advantage_mae"], report["constant_advantage_mae"])
        self.assertAlmostEqual(report["paired_mae_improvement"], 0.0)

    def test_malformed_completion_gets_zero_semantic_score(self):
        data = enrich_structural_rewards(rows(groups=1))
        data[0]["completion"] = "answer without think tags"
        add_semantic_scores(data, "k0", FakeScorer())
        self.assertEqual(data[0]["semantic_rewards"]["k0"], 0.0)

    def test_component_variance_gate_detects_active_semantic_groups(self):
        data = enrich_structural_rewards(rows())
        add_semantic_scores(data, "k0", FakeScorer())
        report = component_variance(data, "k0")
        self.assertTrue(report["semantic_gate_pass"])
        self.assertTrue(report["aggregate_gate_pass"])

    def test_rank_metrics_handle_ties_and_degenerate_labels(self):
        self.assertEqual(roc_auc([0, 0, 1, 1], [0, 0, 1, 1]), 1.0)
        self.assertIsNone(roc_auc([0, 1], [1, 1]))
        self.assertIsNone(spearman([1, 1], [0, 1]))

    def test_reference_labels_require_exact_rollout_identity(self):
        data = rows(groups=1)
        labels = [
            {
                "group_id": row["group_id"],
                "rollout_index": row["rollout_index"],
                "reference_score": row["reference_score"],
            }
            for row in data
        ]
        for row in data:
            del row["reference_score"]
        attach_reference_scores(data, labels)
        self.assertEqual(data[-1]["reference_score"], 1.0)
        with self.assertRaises(ValueError):
            attach_reference_scores(data, labels[:-1])

    def test_numeric_k_tie_break_order(self):
        self.assertEqual(sorted(["k5", "k0", "k3", "other"], key=label_sort_key), ["k0", "k3", "k5", "other"])

    def test_full_rollout_validator_rejects_incomplete_corpus(self):
        with self.assertRaises(ValueError):
            validate_rollouts(rows(groups=1))

    def test_end_to_end_selection_uses_validation_and_passes_perfect_candidate(self):
        data = []
        labels = []
        for split in ("validation", "test"):
            for group in range(100):
                group_id = f"{split}-{group:03d}"
                for rollout in range(8):
                    score = rollout / 7
                    data.append(
                        {
                            "group_id": group_id,
                            "split": split,
                            "source_index": group,
                            "rollout_index": rollout,
                            "reference": "answer",
                            "completion": f"<think>x</think> answer {rollout}",
                            "structural_rewards": {
                                "think_format_reward": 1.0,
                                "output_datatype_reward": 0.5,
                                "output_schema_reward": 0.5,
                                "doom_loop_reward": 0.0,
                                "length_penalty_reward": 0.0,
                            },
                            "semantic_rewards": {"k0": score, "k2": 0.5},
                        }
                    )
                    labels.append(
                        {
                            "group_id": group_id,
                            "rollout_index": rollout,
                            "reference_score": score,
                        }
                    )
        report = analyze(data, labels, bootstrap_samples=20, seed=3407)
        self.assertEqual(report["selected_variant"], "k0")
        self.assertTrue(report["item9_pass"])
        self.assertTrue(report["item9_5_pass"])
        self.assertTrue(report["pass"])

    def test_scalar_groups_join_frozen_identity_and_strip_valid_reasoning(self):
        prompt_rows = [
            {
                "group_id": f"{'validation' if group < 100 else 'test'}-{group % 100:03d}",
                "split": "validation" if group < 100 else "test",
                "source_index": group,
                "instruction": "answer",
                "input": "passage",
                "reference": "reference",
            }
            for group in range(200)
        ]
        rollout_rows = []
        for prompt in prompt_rows:
            for rollout in range(8):
                rollout_rows.append(
                    {
                        "group_id": prompt["group_id"],
                        "split": prompt["split"],
                        "source_index": prompt["source_index"],
                        "rollout_index": rollout,
                        "reference": prompt["reference"],
                        "completion": "<think>private chain</think> final answer",
                    }
                )
        result = build_groups(prompt_rows, rollout_rows)
        self.assertEqual(len(result), 200)
        self.assertEqual(result[0]["task"], "answer\n\npassage")
        self.assertEqual(result[0]["candidates"][0]["answer"], "final answer")
        self.assertNotIn("private chain", json.dumps(result[0]))
        rollout_rows.pop()
        with self.assertRaises(ValueError):
            build_groups(prompt_rows, rollout_rows)

    def test_scalar_final_answer_preserves_malformed_output(self):
        self.assertEqual(final_answer("plain answer"), "plain answer")
        self.assertEqual(
            final_answer("<think>private chain without a close"),
            "[malformed reasoning output omitted]",
        )
        self.assertEqual(final_answer("junk </think> public answer"), "public answer")


if __name__ == "__main__":
    unittest.main()
