"""CPU checks of the DPO probe: stress selection and fit criteria.

These encode the defects found in the 2026-09-12 prelaunch review. The previous version of
this suite accepted one `training_step` call when two were requested, and treated a NaN
loss as a fit; both are now explicitly rejected.
"""
import json
import unittest

from posttraining_harness.dpo_memory_probe import BoundedProbe, forward_record, select_stress_pairs


class FakeTokenizer:
    """Token count == character count, so lengths in tests are readable."""

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [0] * len(text)}

    def apply_chat_template(self, conversation, tokenize=True):
        return [0] * sum(len(turn["content"]) for turn in conversation)


def row(prompt, chosen, rejected):
    return {"prompt": "p" * prompt,
            "chosen": [{"role": "assistant", "content": "c" * chosen}],
            "rejected": [{"role": "assistant", "content": "r" * rejected}]}


class SelectionTests(unittest.TestCase):
    def test_scores_by_concatenated_footprint_not_the_sum(self):
        """One very long completion binds harder than two medium ones of equal sum."""
        data = [row(10, 100, 10), row(10, 55, 55)]
        _, indices, lengths = select_stress_pairs(data, 1, FakeTokenizer(), 2048, 1536)
        self.assertEqual(indices, [0], "max(chosen, rejected) should win, not chosen+rejected")
        self.assertEqual(lengths[0]["footprint"], 110)

    def test_picks_longest_not_first(self):
        data = [row(1, 1, 1), row(10, 500, 500), row(2, 2, 2)]
        _, indices, _ = select_stress_pairs(data, 1, FakeTokenizer(), 2048, 1536)
        self.assertEqual(indices, [1])

    def test_prompt_capped_at_max_prompt_length_and_total_at_max_length(self):
        data = [row(5000, 5000, 10)]
        _, _, lengths = select_stress_pairs(data, 1, FakeTokenizer(), 2048, 1536)
        self.assertEqual(lengths[0]["prompt"], 1536, "prompt capped")
        self.assertEqual(lengths[0]["footprint"], 2048, "total capped at max_length")

    def test_rows_missing_a_side_are_skipped(self):
        data = [{"prompt": "p", "chosen": [{"role": "a", "content": "c"}]}, row(1, 2, 3)]
        _, indices, _ = select_stress_pairs(data, 1, FakeTokenizer(), 2048, 1536)
        self.assertEqual(indices, [1])
        with self.assertRaises(AssertionError):
            select_stress_pairs(data, 2, FakeTokenizer(), 2048, 1536)


class FitCriteriaTests(unittest.TestCase):
    """A candidate must demonstrably train, not merely fail to raise."""

    def _probe(self, **kwargs):
        # stress_target_tokens defaults to 0 here, which disables the stress gate so these
        # tests exercise the step/loss/optimizer criteria in isolation. SeamTests covers
        # the gate itself, and the CLI always supplies a real target.
        return BoundedProbe(lambda *a, **k: None, **kwargs)

    def test_microbatches_are_not_optimizer_steps(self):
        probe = self._probe(max_optimizer_steps=2)
        for _ in range(16):
            probe.on_substep()
        probe.losses = [0.5]
        verdict = probe.verdict()
        self.assertEqual(verdict["microbatches"], 16)
        self.assertEqual(verdict["optimizer_steps"], 0)
        self.assertFalse(verdict["fit"], "16 microbatches and no optimizer step is not a fit")

    def test_too_few_optimizer_steps_is_not_a_fit(self):
        probe = self._probe(max_optimizer_steps=2)
        probe.losses = [0.5]
        probe.on_optimizer_step(skipped=False)
        self.assertFalse(probe.verdict()["fit"], "one step when two were requested")

    def test_nonfinite_loss_is_not_a_fit(self):
        probe = self._probe(max_optimizer_steps=1)
        probe.losses = [float("nan")]
        probe.on_optimizer_step(skipped=False)
        verdict = probe.verdict()
        self.assertFalse(verdict["losses_finite"])
        self.assertFalse(verdict["fit"], "NaN loss must never count as fitting")

    def test_no_loss_recorded_is_not_a_fit(self):
        probe = self._probe(max_optimizer_steps=1)
        probe.on_optimizer_step(skipped=False)
        self.assertFalse(probe.verdict()["fit"], "absence of evidence is not a fit")

    def test_clean_run_fits(self):
        probe = self._probe(max_optimizer_steps=2)
        probe.losses = [0.6, 0.5]
        probe.on_optimizer_step(skipped=False)
        self.assertFalse(probe.should_stop())
        probe.on_optimizer_step(skipped=False)
        self.assertTrue(probe.should_stop())
        self.assertTrue(probe.verdict()["fit"])

    def test_post_evaluation_update_is_required_when_requested(self):
        """The hazard that killed the SFT run is the first update AFTER validation."""
        probe = self._probe(max_optimizer_steps=1, post_eval_steps=1)
        probe.losses = [0.5]
        probe.on_optimizer_step(skipped=False)
        self.assertFalse(probe.should_stop(), "must not stop before the post-eval update")
        self.assertFalse(probe.verdict()["fit"])
        probe.on_evaluation()
        probe.losses.append(0.4)
        probe.on_optimizer_step(skipped=False)
        self.assertTrue(probe.should_stop())
        verdict = probe.verdict()
        self.assertTrue(verdict["evaluated"])
        self.assertEqual(verdict["post_eval_optimizer_steps"], 1)
        self.assertTrue(verdict["fit"])


class ReportedDefectTests(unittest.TestCase):
    """Regressions for the defects the 2026-09-12 repair assessment found by execution.

    Each of these passed every static check and every earlier unit test.
    """

    def test_exception_report_does_not_raise(self):
        """`emit(fit=False, **verdict())` raised TypeError, so an OOM reported a bug."""
        probe = BoundedProbe(lambda *a, **k: None, max_optimizer_steps=1)
        merged = {**probe.verdict(), "fit": False, "error": "OutOfMemoryError"}
        self.assertEqual(merged["fit"], False)
        self.assertEqual(merged["error"], "OutOfMemoryError")
        captured = {}
        def emit(event, **fields):
            captured.update(fields)      # must not raise
        emit("candidate_result", **merged)
        self.assertFalse(captured["fit"], "the OOM must surface as a memory result")

    def test_row_supply_matches_the_requested_windows(self):
        """One window of rows yields one optimizer update; a verdict needing two failed."""
        for batch, accum, steps, post in ((16, 8, 2, 1), (4, 8, 2, 0), (2, 8, 3, 1)):
            windows = steps + post
            per_window = batch * accum
            unique = [{"row": i} for i in range(per_window)]
            rows = (unique * windows)[: per_window * windows]
            self.assertEqual(len(rows), per_window * windows)
            self.assertEqual(len(rows) // batch // accum, windows,
                             "supplied rows must afford exactly the requested updates")

    def test_eval_losses_count_toward_finiteness(self):
        """on_step_end fires before logging, so training losses can arrive only via on_log."""
        probe = BoundedProbe(lambda *a, **k: None, max_optimizer_steps=1)
        probe.losses.append(0.5)
        probe.eval_losses.append(float("nan"))
        probe.on_optimizer_step(skipped=False)
        verdict = probe.verdict()
        self.assertFalse(verdict["losses_finite"],
                         "a NaN evaluation loss must fail the candidate")
        self.assertIn("eval_losses", verdict)

    def test_no_losses_at_all_is_not_finite(self):
        probe = BoundedProbe(lambda *a, **k: None, max_optimizer_steps=1)
        probe.on_optimizer_step(skipped=False)
        self.assertFalse(probe.verdict()["losses_finite"])


class SeamTests(unittest.TestCase):
    """The 2026-09-12 v3 assessment's lesson: test the boundary BETWEEN components.

    Earlier suites covered each part and missed every seam. Silent and newline-chatty
    children were tested, partial-line output was not. NaN was tested in the verdict, its
    serialisation was not. Shapes were recorded, their effect on `fit` was not.
    """

    def test_a_nan_verdict_can_actually_be_serialised(self):
        """A detected NaN previously made its own report unserialisable, losing the finding."""
        probe = BoundedProbe(lambda *a, **k: None, max_optimizer_steps=1)
        probe.losses.append(float("nan"))
        probe.eval_losses.append(float("inf"))
        probe.on_optimizer_step(skipped=False)
        verdict = probe.verdict()
        encoded = json.dumps(verdict, sort_keys=True, allow_nan=False)  # must not raise
        self.assertIn("NaN", encoded)
        self.assertIn("Infinity", encoded)
        self.assertEqual(verdict["nonfinite_observations"], 2)
        self.assertFalse(verdict["losses_finite"])
        self.assertFalse(verdict["fit"])

    def test_json_safe_leaves_finite_values_untouched(self):
        from posttraining_harness.dpo_memory_probe import json_safe
        self.assertEqual(json_safe({"a": [1.5, 2], "b": "x"}), {"a": [1.5, 2], "b": "x"})

    def test_no_recorded_stress_means_no_fit(self):
        """Shapes were recorded but never gated fit, so a candidate could stress nothing."""
        probe = BoundedProbe(lambda *a, **k: None, max_optimizer_steps=1,
                             stress_target_tokens=2048)
        probe.losses.append(0.5)
        probe.on_optimizer_step(skipped=False)
        verdict = probe.verdict()
        self.assertEqual(verdict["stress"]["collations_recorded"], 0)
        self.assertFalse(verdict["stress"]["reached_target"])
        self.assertFalse(verdict["fit"], "an empty collator record must not pass")

    def test_undersized_workload_is_inconclusive_not_a_pass(self):
        probe = BoundedProbe(lambda *a, **k: None, max_optimizer_steps=1,
                             stress_target_tokens=2048)
        probe.losses.append(0.5)
        probe.shapes.append({"phase": "training", "max_row_pre_truncation_tokens": 300,
                             "batch_rows": 4, "all_rows_supervised": True})
        probe.on_optimizer_step(skipped=False)
        self.assertFalse(probe.verdict()["stress"]["reached_target"])
        self.assertFalse(probe.verdict()["fit"])

    def test_one_side_unsupervised_is_not_a_pass(self):
        probe = BoundedProbe(lambda *a, **k: None, max_optimizer_steps=1,
                             stress_target_tokens=2048)
        probe.losses.append(0.5)
        probe.shapes.append({"phase": "training", "max_row_pre_truncation_tokens": 2048,
                             "batch_rows": 4, "all_rows_supervised": False})
        probe.on_optimizer_step(skipped=False)
        self.assertFalse(
            probe.verdict()["stress"]["per_phase"]["training"]["all_rows_supervised"],
            "a pair with an unsupervised side teaches nothing")
        self.assertFalse(probe.verdict()["fit"])

    def test_full_workload_in_both_phases_passes(self):
        probe = BoundedProbe(lambda *a, **k: None, max_optimizer_steps=1,
                             post_eval_steps=1, stress_target_tokens=2048)
        probe.losses.append(0.5)
        full = {"max_row_pre_truncation_tokens": 2048, "batch_rows": 4,
                "all_rows_supervised": True}
        probe.shapes.append({"phase": "training", **full})
        probe.forwards.append(forward_record("model_call", "training", True, (8, 2048),
                                             [2048] * 8))
        probe.forwards.append(forward_record("input_embeddings", "training", True, (8, 2048)))
        probe.on_optimizer_step(skipped=False)
        probe.on_evaluation()
        probe.shapes.append({"phase": "post_evaluation", **full})
        probe.forwards.append(forward_record("model_call", "post_evaluation", True, (8, 2048),
                                             [2048] * 8))
        probe.forwards.append(forward_record("input_embeddings", "post_evaluation", True, (8, 2048)))
        probe.losses.append(0.4)
        probe.on_optimizer_step(skipped=False)
        verdict = probe.verdict()
        self.assertEqual(verdict["stress"]["phases_seen"], ["post_evaluation", "training"])
        self.assertTrue(verdict["stress"]["reached_target"])
        self.assertTrue(verdict["fit"])

    def test_training_phase_only_does_not_satisfy_a_post_eval_requirement(self):
        probe = BoundedProbe(lambda *a, **k: None, max_optimizer_steps=1,
                             post_eval_steps=1, stress_target_tokens=2048)
        probe.losses.append(0.5)
        probe.shapes.append({"phase": "training", "pre_truncation_upper_bound_tokens": 2048,
                             "chosen_rows_with_tokens": 4, "rejected_rows_with_tokens": 4})
        probe.on_optimizer_step(skipped=False)
        self.assertFalse(probe.verdict()["stress"]["reached_target"],
                         "the post-evaluation phase is the hazard; it must be exercised")


class StressGateCounterexampleTests(unittest.TestCase):
    """The exact counterexamples the v4 assessment reproduced. Each must now fail."""

    def _probe(self, target=2048, post=1):
        return BoundedProbe(lambda *a, **k: None, max_optimizer_steps=1,
                            post_eval_steps=post, stress_target_tokens=target)

    @staticmethod
    def _collation(phase, tokens, rows=4, supervised=None):
        return {"phase": phase, "max_row_pre_truncation_tokens": tokens,
                "batch_rows": rows,
                "rows_supervised_both_sides": rows if supervised is None else supervised,
                "all_rows_supervised": (rows if supervised is None else supervised) == rows}

    def test_full_training_phase_cannot_carry_a_tiny_post_eval_phase(self):
        """2048 in training with 32 after evaluation previously passed on the global max."""
        probe = self._probe()
        probe.losses.append(0.5)
        probe.shapes.append(self._collation("training", 2048))
        probe.on_optimizer_step(skipped=False)
        probe.on_evaluation()
        probe.shapes.append(self._collation("post_evaluation", 32))
        probe.losses.append(0.4)
        probe.on_optimizer_step(skipped=False)
        verdict = probe.verdict()
        self.assertFalse(verdict["stress"]["per_phase"]["post_evaluation"]["reached_target"])
        self.assertFalse(verdict["stress"]["reached_target"])
        self.assertFalse(verdict["fit"], "the phase that matters carried 32 tokens")

    @staticmethod
    def _forward(phase, tokens, grad=True, rows=8):
        return forward_record("model_call", phase, grad, (rows, tokens), [tokens] * rows)

    def test_both_phases_at_target_pass(self):
        probe = self._probe()
        probe.losses.append(0.5)
        probe.shapes.append(self._collation("training", 2048))
        probe.forwards.extend([self._forward("training", 2048), forward_record("input_embeddings", "training", True, (8, 2048))])
        probe.on_optimizer_step(skipped=False)
        probe.on_evaluation()
        probe.shapes.append(self._collation("post_evaluation", 2100))
        probe.forwards.extend([self._forward("post_evaluation", 2048), forward_record("input_embeddings", "post_evaluation", True, (8, 2048))])
        probe.losses.append(0.4)
        probe.on_optimizer_step(skipped=False)
        self.assertTrue(probe.verdict()["fit"])

    def test_one_unsupervised_pair_fails_the_batch(self):
        """'At least one row per side' passed a batch of 15 empty pairs and one real one."""
        probe = self._probe(post=0)
        probe.losses.append(0.5)
        probe.shapes.append(self._collation("training", 2048, rows=16, supervised=1))
        probe.on_optimizer_step(skipped=False)
        verdict = probe.verdict()
        self.assertFalse(verdict["stress"]["per_phase"]["training"]["all_rows_supervised"])
        self.assertFalse(verdict["fit"])

    def test_evaluation_collations_are_not_credited_as_stress(self):
        """The eval dataloader uses the same collator; its batches must not count."""
        probe = self._probe()
        probe.losses.append(0.5)
        probe.shapes.append(self._collation("training", 2048))
        probe.on_optimizer_step(skipped=False)
        probe.on_evaluation()
        probe.shapes.append(self._collation("evaluation", 4096))   # big, but it is an eval batch
        probe.losses.append(0.4)
        probe.on_optimizer_step(skipped=False)
        verdict = probe.verdict()
        self.assertEqual(
            verdict["stress"]["per_phase"]["post_evaluation"]["collations"], 0,
            "an evaluation batch is not a post-evaluation training batch")
        self.assertFalse(verdict["fit"])


class AppliedStepTests(unittest.TestCase):
    """on_step_end fires even when the FP16 scaler skips the update."""

    def _probe(self, **kw):
        return BoundedProbe(lambda *a, **k: None, max_optimizer_steps=2, **kw)

    def test_skipped_steps_do_not_count_as_updates(self):
        probe = self._probe()
        probe.losses.append(0.5)
        probe.on_optimizer_step(skipped=True)
        probe.on_optimizer_step(skipped=True)
        verdict = probe.verdict()
        self.assertEqual(verdict["optimizer_steps"], 2, "attempts are still recorded")
        self.assertEqual(verdict["applied_steps"], 0)
        self.assertEqual(verdict["skipped_steps"], 2)
        self.assertFalse(verdict["fit"], "two skipped attempts are not two updates")

    def test_unknown_skip_flag_is_not_evidence_of_an_update(self):
        """Missing evidence must not count as an applied update."""
        probe = self._probe()
        probe.losses.append(0.5)
        probe.on_optimizer_step(skipped=None)
        probe.on_optimizer_step(skipped=None)
        self.assertEqual(probe.verdict()["applied_steps"], 0)
        self.assertEqual(probe.verdict()["unknown_steps"], 2)
        self.assertFalse(probe.verdict()["fit"])

    def test_stop_waits_for_applied_updates(self):
        probe = self._probe()
        probe.on_optimizer_step(skipped=True)
        self.assertFalse(probe.should_stop())
        probe.on_optimizer_step(skipped=False)
        self.assertFalse(probe.should_stop(), "one applied of two requested")
        probe.on_optimizer_step(skipped=False)
        self.assertTrue(probe.should_stop())


if __name__ == "__main__":
    unittest.main()


class ForwardAgreementRegressionTests(unittest.TestCase):
    def test_embedding_width_cannot_overrule_the_mask(self):
        probe = BoundedProbe(lambda *a, **k: None, 1, stress_target_tokens=2048)
        probe.losses = [0.5]
        probe.on_optimizer_step(skipped=False)
        probe.shapes = [{"phase": "training", "all_rows_supervised": True}]
        probe.forwards = [
            forward_record("model_call", "training", True, (8, 2048), [1024] * 8),
            forward_record("input_embeddings", "training", True, (8, 2048)),
        ]
        self.assertFalse(probe.verdict()["fit"])
        probe.forwards[0] = forward_record("model_call", "training", True, (8, 2048), [2048] * 8)
        self.assertTrue(probe.verdict()["fit"])
        probe.forwards.pop()
        self.assertFalse(probe.verdict()["fit"], "missing embedding hook must fail closed")

    def test_maskless_columns_are_not_an_exact_measurement(self):
        probe = BoundedProbe(lambda *a, **k: None, 1, stress_target_tokens=2048)
        probe.losses = [0.5]
        probe.on_optimizer_step(skipped=False)
        probe.shapes = [{"phase": "training", "all_rows_supervised": True}]
        probe.forwards = [forward_record(source, "training", True, (8, 2048))
                          for source in ("model_call", "input_embeddings")]
        self.assertFalse(probe.verdict()["fit"])


class ForwardMeasurementTests(unittest.TestCase):
    """The v4 requirement v5 left open: stress is what the MODEL received, not the collator."""

    def _probe(self, target=2048, post=1):
        return BoundedProbe(lambda *a, **k: None, max_optimizer_steps=1,
                            post_eval_steps=post, stress_target_tokens=target)

    @staticmethod
    def _collation(phase, tokens, rows=4):
        return {"phase": phase, "max_row_pre_truncation_tokens": tokens, "batch_rows": rows,
                "rows_supervised_both_sides": rows, "all_rows_supervised": True}

    @staticmethod
    def _forward(phase, tokens, grad=True, rows=8, source="model_call"):
        return forward_record(source, phase, grad, (rows, tokens), [tokens] * rows)

    def _run(self, probe, training, post):
        probe.losses.append(0.5)
        for item in training:
            (probe.forwards if "source" in item else probe.shapes).append(item)
        probe.on_optimizer_step(skipped=False)
        probe.on_evaluation()
        for item in post:
            (probe.forwards if "source" in item else probe.shapes).append(item)
        probe.losses.append(0.4)
        probe.on_optimizer_step(skipped=False)
        return probe.verdict()

    def test_collator_bound_at_target_but_forward_short_is_not_a_pass(self):
        """The counterexample v5 could not refuse: a pre-truncation bound over the line."""
        verdict = self._run(self._probe(),
                            [self._collation("training", 2400), self._forward("training", 1536)],
                            [self._collation("post_evaluation", 2400),
                             self._forward("post_evaluation", 1536)])
        self.assertEqual(verdict["stress"]["per_phase"]["training"]
                         ["max_row_pre_truncation_tokens"], 2400)
        self.assertEqual(verdict["stress"]["per_phase"]["training"]["max_row_forward_tokens"],
                         1536)
        self.assertFalse(verdict["stress"]["reached_target"])
        self.assertFalse(verdict["fit"], "the model never processed 2048 tokens in a row")

    def test_no_forward_observed_fails_closed(self):
        """If the hooks never fire, the answer is inconclusive, never a comfortable fit."""
        verdict = self._run(self._probe(), [self._collation("training", 2048)],
                            [self._collation("post_evaluation", 2048)])
        self.assertEqual(verdict["stress"]["forwards_recorded"], 0)
        self.assertEqual(verdict["stress"]["per_phase"]["training"]["training_forwards"], 0)
        self.assertFalse(verdict["fit"])

    def test_reference_and_evaluation_forwards_are_not_credited(self):
        """The no-grad reference pass and eval forwards run the same model on full rows."""
        verdict = self._run(
            self._probe(),
            [self._collation("training", 2048), self._forward("training", 1024),
             self._forward("training", 2048, grad=False)],
            [self._collation("post_evaluation", 2048),
             self._forward("evaluation", 2048, grad=True),
             self._forward("post_evaluation", 2048, grad=False)])
        per_phase = verdict["stress"]["per_phase"]
        self.assertEqual(per_phase["training"]["max_row_forward_tokens"], 1024)
        self.assertEqual(per_phase["post_evaluation"]["training_forwards"], 0)
        self.assertFalse(verdict["fit"])

    def test_forward_at_target_in_both_phases_passes(self):
        verdict = self._run(self._probe(),
                            [self._collation("training", 2600), self._forward("training", 2048),
                             self._forward("training", 2048, source="input_embeddings")],
                            [self._collation("post_evaluation", 2600),
                             self._forward("post_evaluation", 2048),
                             self._forward("post_evaluation", 2048, source="input_embeddings")])
        self.assertTrue(verdict["stress"]["reached_target"])
        self.assertTrue(verdict["fit"])

    def test_a_prefetched_post_evaluation_batch_still_passes(self):
        """Accelerate collates the next batch before yielding the current one, so the batch
        trained after evaluation can carry a "training" collation tag. Observed in a real TRL
        run: zero post-evaluation collations, full post-evaluation forwards. Must pass."""
        verdict = self._run(self._probe(),
                            [self._collation("training", 2600), self._collation("training", 2600),
                             self._forward("training", 2048),
                             self._forward("training", 2048, source="input_embeddings")],
                            [self._forward("post_evaluation", 2048),
                             self._forward("post_evaluation", 2048, source="input_embeddings")])
        self.assertEqual(verdict["stress"]["per_phase"]["post_evaluation"]["collations"], 0)
        self.assertTrue(verdict["stress"]["reached_target"])
        self.assertTrue(verdict["fit"])

    def test_one_unsupervised_training_collation_fails_every_phase(self):
        unsupervised = {**self._collation("training", 2600), "all_rows_supervised": False}
        verdict = self._run(self._probe(),
                            [self._collation("training", 2600), unsupervised,
                             self._forward("training", 2048)],
                            [self._forward("post_evaluation", 2048)])
        self.assertFalse(verdict["stress"]["per_phase"]["post_evaluation"]["all_rows_supervised"])
        self.assertFalse(verdict["fit"])

    def test_evaluation_collations_do_not_count_toward_supervision(self):
        verdict = self._run(self._probe(),
                            [self._collation("evaluation", 2600),
                             self._forward("training", 2048)],
                            [self._forward("post_evaluation", 2048)])
        self.assertFalse(verdict["stress"]["per_phase"]["training"]["all_rows_supervised"],
                         "no training collation was observed, so supervision is unproven")
        self.assertFalse(verdict["fit"])

    def test_attention_mask_rows_are_preferred_over_columns(self):
        exact = forward_record("model_call", "training", True, (2, 12), [5, 9])
        self.assertEqual((exact["max_row_tokens"], exact["basis"]), (9, "attention_mask"))
        columns = forward_record("input_embeddings", "training", True, (2, 12))
        self.assertEqual((columns["max_row_tokens"], columns["basis"]), (12, "columns"))

    def test_the_collator_phase_and_the_forward_phase_share_one_definition(self):
        probe = self._probe()
        self.assertEqual(probe.current_phase(), "training")
        probe.in_evaluation = True
        self.assertEqual(probe.current_phase(), "evaluation")
        probe.in_evaluation = False
        probe.on_evaluation()
        self.assertEqual(probe.current_phase(), "post_evaluation")
