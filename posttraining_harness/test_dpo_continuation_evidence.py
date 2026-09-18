"""Real TRL DPOTrainer on CPU: the evidence a multi-segment continuation needs.

* check_data_position across epoch boundaries, including an epoch whose last update has fewer than grad_accum
  microbatches, for an uninterrupted run and for a stopped-then-resumed run;
* the in-training reference guard: probes at train begin, first update, every N updates and train end compare
  bitwise with the train-begin probe; a mutated frozen weight stops training with state saved and a recorded reason.
"""
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

REQUIRED = ("torch", "transformers", "trl", "peft", "tokenizers", "datasets")
MISSING = [name for name in REQUIRED if importlib.util.find_spec(name) is None]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "preference_optimization")]


@unittest.skipIf(MISSING, f"needs {MISSING}; run under the fork's main .venv")
class MultiEpochDataPositionTests(unittest.TestCase):
    # 26 rows / batch 2 = 13 microbatches; accumulation 2 -> 7 updates per epoch, the 7th with a single microbatch.
    ROWS, HORIZON, STOP = 26, 10, 5

    @classmethod
    def setUpClass(cls):
        import dpo_setup as setup
        from test_dpo_segments_real_trainer import make_rows, make_trainer
        from training_controls import make_stop_callback

        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        common = ["--batch_size", "2", "--grad_accum", "2", "--max_steps", str(cls.HORIZON), "--save_steps", "5",
                  "--eval_steps", "5", "--logging_steps", "1", "--dataloader_num_workers", "0"]
        train, val = make_rows(cls.ROWS), make_rows(4, offset=30)

        def run(output, fingerprints, stop=None, resume=None):
            _, trainer = make_trainer(output, train, val, common + (["--stop_after_steps", str(stop)] if stop else []))
            if stop:
                trainer.add_callback(make_stop_callback(stop, {}))
            trainer.data_collator = setup.FingerprintingCollator(trainer.data_collator, fingerprints)
            trainer.train(resume_from_checkpoint=resume)
            return trainer

        cls.paths = {k: root / f"{k}.jsonl" for k in ("full", "s1", "s2")}
        cls.full = run(root / "full", cls.paths["full"])
        cls.s1 = run(root / "seg", cls.paths["s1"], stop=cls.STOP)
        cls.s2 = run(root / "seg", cls.paths["s2"], resume=str(root / "seg" / f"checkpoint-{cls.STOP}"))
        cls.seed = cls.full.args.seed

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def check(self, trainer, path, first, last):
        import dpo_setup as setup
        return setup.check_data_position(trainer.train_dataset, trainer.eval_dataset, path, seed=self.seed,
                                         first_step=first, last_step=last, batch_size=2, grad_accum=2)

    def test_window_model(self):
        import dpo_setup as setup

        windows = setup.update_microbatch_windows(26, 2, 2, 0, 10)
        self.assertEqual(windows[:7], [(0, 0, 2), (0, 2, 4), (0, 4, 6), (0, 6, 8), (0, 8, 10), (0, 10, 12), (0, 12, 13)])
        self.assertEqual(windows[7:], [(1, 0, 2), (1, 2, 4), (1, 4, 6)])
        self.assertEqual(setup.update_microbatch_windows(63_184, 4, 32, 493, 495),
                         [(0, 15_776, 15_796), (1, 0, 32)])

    def test_uninterrupted_run_crossing_an_epoch_with_a_short_window(self):
        result = self.check(self.full, self.paths["full"], 0, self.HORIZON)
        self.assertTrue(result["trained_match"] and result["lookahead_ok"], result)
        self.assertEqual(result["epochs_spanned"], [0, 1])
        self.assertEqual(result["trained_rows_expected"], 2 * 2 * 6 + 2 + 2 * 2 * 3)

    def test_stopped_and_resumed_segments_cross_the_boundary_faithfully(self):
        first = self.check(self.s1, self.paths["s1"], 0, self.STOP)
        second = self.check(self.s2, self.paths["s2"], self.STOP, self.HORIZON)
        self.assertTrue(first["trained_match"] and first["lookahead_ok"], first)
        self.assertTrue(second["trained_match"] and second["lookahead_ok"], second)
        self.assertEqual(second["epochs_spanned"], [0, 1])
        self.assertEqual(self.s2.state.global_step, self.HORIZON)

    def test_corruption_and_wrong_epoch_seed_are_detected(self):
        import dpo_setup as setup

        records = [json.loads(line) for line in self.paths["s2"].read_text().splitlines()]
        held_out = {setup.row_fingerprint(row) for row in self.s2.eval_dataset}
        # Evaluation batches share the collator and are filtered out by content, so corrupt a training microbatch.
        first_train = next(i for i, r in enumerate(records) if not set(r["rows"]) & held_out and len(set(r["rows"])) == 2)
        records[first_train]["rows"] = list(reversed(records[first_train]["rows"]))
        corrupted = Path(self.tmp.name) / "corrupted.jsonl"
        corrupted.write_text("".join(json.dumps(r) + "\n" for r in records))
        self.assertFalse(self.check(self.s2, corrupted, self.STOP, self.HORIZON)["trained_match"])
        import dpo_setup as setup
        wrong_seed = setup.check_data_position(self.s2.train_dataset, self.s2.eval_dataset, self.paths["s2"],
                                               seed=self.seed + 1, first_step=self.STOP, last_step=self.HORIZON,
                                               batch_size=2, grad_accum=2)
        self.assertFalse(wrong_seed["trained_match"])


@unittest.skipIf(MISSING, f"needs {MISSING}; run under the fork's main .venv")
class ReferenceGuardTests(unittest.TestCase):
    def run_guard(self, mutate_at=None):
        import dpo_setup as setup
        import torch
        from test_dpo_segments_real_trainer import make_rows, make_trainer
        from training_controls import terminal_record
        from transformers import TrainerCallback

        with tempfile.TemporaryDirectory() as d:
            argv = ["--batch_size", "2", "--grad_accum", "1", "--max_steps", "6", "--save_steps", "3", "--eval_steps", "3",
                    "--dataloader_num_workers", "0", "--reference_probe_rows", "2", "--reference_guard_every", "2"]
            args, trainer = make_trainer(d, make_rows(12), make_rows(2, offset=30), argv)
            guard, stop = {}, {}
            if mutate_at is not None:
                class MutateFrozenWeight(TrainerCallback):
                    def on_step_begin(self, args, state, control, model=None, **kwargs):
                        if state.global_step == mutate_at:
                            with torch.no_grad():
                                name, param = next((n, p) for n, p in model.named_parameters()
                                                   if "lora_" not in n and p.dim() == 2)
                                param[0, 0] += 0.5
                trainer.add_callback(MutateFrozenWeight())
            trainer.add_callback(setup.make_reference_guard_callback(trainer, args.reference_probe_rows,
                                                                     args.reference_guard_every, guard, stop))
            trainer.train()
            json.dumps(guard)
            checkpoints = sorted(p.name for p in Path(d).glob("checkpoint-*") if p.is_dir())
            return guard, terminal_record(trainer.state, stop, None), trainer.state.global_step, checkpoints

    def test_clean_run_probes_every_point_and_all_match(self):
        guard, stop, step, _ = self.run_guard()
        self.assertEqual([p["label"] for p in guard["probes"]],
                         ["train_begin", "first_update", "step_2", "step_4", "step_6", "train_end"])
        self.assertTrue(all(c["ok"] for c in guard["comparisons"]))
        self.assertTrue(all(all(p["context"]["side_effects"].values()) for p in guard["probes"]))
        self.assertFalse(guard["failed"])
        self.assertEqual((stop["stop_reason"], step), ("horizon", 6))
        begin, end = guard["probes"][0], guard["probes"][-1]
        self.assertNotEqual(begin["adapter_sha256"], end["adapter_sha256"], "the policy must move for the guard to mean anything")

    def test_a_mutated_frozen_weight_stops_training_with_the_reason_recorded(self):
        guard, stop, step, checkpoints = self.run_guard(mutate_at=2)
        self.assertTrue(guard["failed"])
        self.assertEqual(guard["first_failure"], "step_4")
        failure = next(c for c in guard["comparisons"] if not c["ok"])
        self.assertIn("parameters", failure["frozen_state_components_differing"])
        self.assertEqual(stop["stop_reason"], "reference_guard")
        self.assertEqual(step, 4)
        self.assertIn("checkpoint-4", checkpoints)

    def test_guard_argument_rules(self):
        import dpo_setup as setup

        def parse(extra):
            args = setup.build_parser().parse_args(["--max_steps", "10"] + extra)
            setup.check_args(args)

        parse(["--reference_probe_rows", "4", "--reference_guard_every", "50"])
        for extra in (["--reference_guard_every", "50"], ["--reference_probe_rows", "4", "--reference_guard_every", "0"],
                      ["--reference_probe_rows", "4", "--reference_guard_every", "5", "--method", "orpo"]):
            with self.subTest(extra=extra), self.assertRaises(SystemExit):
                parse(extra)


if __name__ == "__main__":
    unittest.main()
