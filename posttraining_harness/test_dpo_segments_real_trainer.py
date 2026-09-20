"""Real TRL DPOTrainer, real PEFT, a tiny random Llama, CPU only.

Drives the code production uses (dpo_setup's parser and config kwargs, the segment stop callback, telemetry, the
fingerprinting collator, the reference probe) through TRL's own training loop to prove:

* loss normalization: per-device batch b with accumulation k updates exactly like batch b*k with no accumulation,
  including a final window with fewer than k microbatches, on a small fixture and at the pilot's 4 x 32 vs 128;
* resumable segments: stop mid-epoch, resume in a fresh trainer, and reach the same weights, losses and data order as
  an uninterrupted run, with optimizer, scheduler, RNG, adapter and early-stopping state restored;
* the reference (adapter disabled) does not move while the policy does;
* a nan training loss stays visible in the logs under the production config;
* the epoch-derived horizon used to freeze --max_steps matches the installed Trainer.

It does not use Unsloth, fp16 or bitsandbytes, so it says nothing about Unsloth's patched path or GradScaler
restoration; the hardware run records the same evidence and the pilot driver checks it.
"""
import importlib.util
import json
import math
import random
import sys
import tempfile
import unittest
from pathlib import Path

REQUIRED = ("torch", "transformers", "trl", "peft", "tokenizers", "datasets")
MISSING = [name for name in REQUIRED if importlib.util.find_spec(name) is None]
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "preference_optimization"))

WORDS = [f"w{i}" for i in range(40)]


def make_tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    vocab = {"<pad>": 0, "<eos>": 1, "<unk>": 2}
    vocab.update({w: i + 3 for i, w in enumerate(WORDS)})
    backend = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="<pad>", eos_token="<eos>", unk_token="<unk>")


def make_model():
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    base = LlamaForCausalLM(LlamaConfig(vocab_size=len(WORDS) + 3, hidden_size=16, intermediate_size=32,
                                        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
                                        max_position_embeddings=128, pad_token_id=0, bos_token_id=1, eos_token_id=1))
    return get_peft_model(base, LoraConfig(r=2, lora_alpha=2, task_type="CAUSAL_LM", target_modules=["q_proj", "v_proj"]))


def make_rows(n, offset=0):
    def text(k, start):
        return " ".join(WORDS[(start + i) % len(WORDS)] for i in range(k))
    return [{"prompt": text(3 + i % 5, i + offset), "chosen": text(2 + i % 3, i + 7), "rejected": text(1 + (i + 1) % 4, i + 13)}
            for i in range(n)]


def make_trainer(output_dir, train_rows, val_rows, argv, **overrides):
    import dpo_setup as setup
    from datasets import Dataset
    from trl import DPOConfig, DPOTrainer

    args = setup.build_parser().parse_args(argv)
    setup.check_args(args)
    cpu_overrides = dict(optim="adamw_torch", max_length=32, max_prompt_length=16)   # no bitsandbytes on CPU
    cpu_overrides.update(overrides)
    kwargs = setup.preference_config_kwargs(args, {"fp16": False, "bf16": False}, str(output_dir), **cpu_overrides)
    trainer = DPOTrainer(model=make_model(), ref_model=None, args=DPOConfig(**kwargs),
                         train_dataset=Dataset.from_list(train_rows), eval_dataset=Dataset.from_list(val_rows),
                         processing_class=make_tokenizer())
    return args, trainer


def lora_state(model):
    return {n: p.detach().clone() for n, p in model.named_parameters() if "lora_" in n}


def events(path, name):
    return [e for e in map(json.loads, Path(path).read_text().splitlines()) if e["event"] == name]


def collated_rows(path):
    return [row for record in map(json.loads, Path(path).read_text().splitlines()) for row in record["rows"]]


@unittest.skipIf(MISSING, f"needs {MISSING}; run under the fork's main .venv")
class AccumulationNormalizationTests(unittest.TestCase):
    def train(self, rows, batch, accum):
        from transformers import TrainerCallback

        with tempfile.TemporaryDirectory() as d:
            argv = ["--batch_size", str(batch), "--grad_accum", str(accum), "--max_steps", "2",
                    "--dataloader_num_workers", "0"]
            _, trainer = make_trainer(d, rows, rows[:2], argv, optim="sgd", learning_rate=0.5,
                                      warmup_steps=0, warmup_ratio=0.0, lr_scheduler_type="constant", weight_decay=0.0,
                                      max_grad_norm=0.0, save_strategy="no", eval_strategy="no",
                                      load_best_model_at_end=False)
            self.assertIs(trainer.model_accepts_loss_kwargs, False)
            initial = lora_state(trainer.model)
            snapshots = []

            class Snapshot(TrainerCallback):
                def on_step_end(self, args, state, control, model=None, **kwargs):
                    snapshots.append(lora_state(model))
            trainer.add_callback(Snapshot())
            trainer.train()
            return initial, snapshots

    def assert_same_updates(self, rows, accumulated, single):
        import torch

        start_a, updates_a = self.train(rows, *accumulated)
        start_b, updates_b = self.train(rows, *single)
        for name in start_a:
            torch.testing.assert_close(start_a[name], start_b[name])
        self.assertEqual((len(updates_a), len(updates_b)), (2, 2))
        for step in range(2):
            deltas = []
            for name in start_a:
                delta_a = updates_a[step][name] - (start_a[name] if step == 0 else updates_a[step - 1][name])
                delta_b = updates_b[step][name] - (start_b[name] if step == 0 else updates_b[step - 1][name])
                torch.testing.assert_close(delta_a, delta_b, rtol=1e-4, atol=1e-6, msg=f"update {step + 1} {name}")
                deltas.append(float(delta_b.abs().max()))
            self.assertGreater(max(deltas), 0.0, "an update that changes nothing proves nothing")

    def test_small_fixture_including_the_partial_window(self):
        # 12 rows: batch 2 x accumulation 4 gives windows of 4 and 2 microbatches; batch 8 gives batches of 8 and 4.
        self.assert_same_updates(make_rows(12), accumulated=(2, 4), single=(8, 1))

    def test_pilot_geometry_4_x_32_equals_128_including_the_partial_window(self):
        # 148 rows: 4 x 32 gives windows of 32 and 5 microbatches (128 and 20 pairs); batch 128 gives 128 and 20.
        self.assert_same_updates(make_rows(148), accumulated=(4, 32), single=(128, 1))


@unittest.skipIf(MISSING, f"needs {MISSING}; run under the fork's main .venv")
class SegmentResumeTests(unittest.TestCase):
    TRAIN, VAL = make_rows(24), make_rows(4, offset=20)
    HORIZON, STOP = 8, 5        # 24 rows / batch 2 / accumulation 2 = 6 updates per epoch, so the resume crosses epochs
    PROBE_ROWS = 2

    @classmethod
    def setUpClass(cls):
        import dpo_setup as setup
        from training_controls import make_stop_callback, terminal_record, validate_resume_checkpoint
        from transformers import EarlyStoppingCallback

        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        common = ["--batch_size", "2", "--grad_accum", "2", "--max_steps", str(cls.HORIZON), "--save_steps", "2",
                  "--eval_steps", "2", "--logging_steps", "1", "--dataloader_num_workers", "0"]
        cls.paths = {name: root / name for name in ("full-tel.jsonl", "full-fp.jsonl", "s1-tel.jsonl", "s1-fp.jsonl",
                                                     "s2-tel.jsonl", "s2-fp.jsonl")}

        def run(output, telemetry, fingerprints, stop=None, resume=None):
            argv = common + (["--stop_after_steps", str(stop)] if stop else [])
            _, trainer = make_trainer(output, cls.TRAIN, cls.VAL, argv)
            record = {}
            # A threshold no evaluation can beat makes the patience counter advance, so restoring it is observable.
            trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=10, early_stopping_threshold=1e9))
            if stop:
                trainer.add_callback(make_stop_callback(stop, record))
            trainer.add_callback(setup.make_telemetry_callback(telemetry, record, trainer))
            trainer.data_collator = setup.FingerprintingCollator(trainer.data_collator, fingerprints)
            before = setup.reference_probe(trainer, cls.PROBE_ROWS)
            trainer.train(resume_from_checkpoint=resume)
            after = setup.reference_probe(trainer, cls.PROBE_ROWS)
            return trainer, terminal_record(trainer.state, record, stop), before, after

        cls.full, cls.full_stop, _, _ = run(root / "full", cls.paths["full-tel.jsonl"], cls.paths["full-fp.jsonl"])
        cls.s1, cls.s1_stop, cls.probe_start, _ = run(root / "seg", cls.paths["s1-tel.jsonl"], cls.paths["s1-fp.jsonl"],
                                                      stop=cls.STOP)
        # Disturb every global generator: a resume that silently fails to restore RNG must not be rescued by chance.
        import numpy as np
        import torch
        random.seed(999)
        np.random.seed(999)
        torch.manual_seed(999)
        cls.checkpoint = root / "seg" / f"checkpoint-{cls.STOP}"
        cls.resume_info = validate_resume_checkpoint(cls.checkpoint, expected_max_steps=cls.HORIZON)
        cls.s2, cls.s2_stop, _, cls.probe_end = run(root / "seg", cls.paths["s2-tel.jsonl"], cls.paths["s2-fp.jsonl"],
                                                    resume=str(cls.checkpoint))
        cls.root = root

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_stop_records_and_resume_validation(self):
        from training_controls import ContinuationError, validate_resume_checkpoint

        self.assertEqual((self.s1_stop["stop_reason"], self.s1_stop["global_step"]), ("target_step", self.STOP))
        self.assertEqual((self.s2_stop["stop_reason"], self.s2_stop["global_step"]), ("horizon", self.HORIZON))
        self.assertEqual(self.full_stop["stop_reason"], "horizon")
        self.assertEqual(self.resume_info["global_step"], self.STOP)
        with self.assertRaises(ContinuationError):
            validate_resume_checkpoint(self.checkpoint, expected_max_steps=self.HORIZON + 1)
        with self.assertRaises(ContinuationError):      # CPU training writes no scaler; an fp16 resume must demand one
            validate_resume_checkpoint(self.checkpoint, expected_max_steps=self.HORIZON, require_scaler=True)

    def test_production_config_restores_callbacks_and_does_not_filter_losses(self):
        self.assertIs(self.s2.args.logging_nan_inf_filter, False)
        self.assertIs(self.s2.args.restore_callback_states_from_checkpoint, True)

    def test_resumed_segments_reach_the_uninterrupted_weights_and_losses(self):
        import torch
        from safetensors.torch import load_file

        full = load_file(self.root / "full" / f"checkpoint-{self.HORIZON}" / "adapter_model.safetensors")
        resumed = load_file(self.root / "seg" / f"checkpoint-{self.HORIZON}" / "adapter_model.safetensors")
        self.assertEqual(full.keys(), resumed.keys())
        for name in full:
            torch.testing.assert_close(resumed[name], full[name], rtol=1e-5, atol=1e-6, msg=name)

        def losses(path):
            return {e["global_step"]: e["logs"]["loss"] for e in events(path, "log") if "loss" in e["logs"]}
        full_losses = losses(self.paths["full-tel.jsonl"])
        seg_losses = {**losses(self.paths["s1-tel.jsonl"]), **losses(self.paths["s2-tel.jsonl"])}
        self.assertEqual(sorted(seg_losses), list(range(1, self.HORIZON + 1)))
        for step, loss in full_losses.items():
            self.assertAlmostEqual(seg_losses[step], loss, places=5, msg=f"step {step}")

    def test_data_order_resumes_at_the_true_position_and_matches_the_sampler(self):
        import dpo_setup as setup

        train_fps = [setup.row_fingerprint(r) for r in self.full.train_dataset]
        members = set(train_fps)
        self.assertEqual(len(members), len(train_fps), "fixture rows must be distinguishable")

        def train_only(path):
            return [fp for fp in collated_rows(path) if fp in members]
        full = train_only(self.paths["full-fp.jsonl"])
        seg1, seg2 = train_only(self.paths["s1-fp.jsonl"]), train_only(self.paths["s2-fp.jsonl"])
        rows_per_update = 2 * 2
        # Accelerate's dataloader fetches one batch ahead, so a stopped segment collates at most one microbatch it
        # never trains on. That lookahead must be exactly where the resumed segment starts.
        trained, lookahead = seg1[:self.STOP * rows_per_update], seg1[self.STOP * rows_per_update:]
        self.assertLessEqual(len(lookahead), 2)
        self.assertEqual(seg2[:len(lookahead)], lookahead)
        self.assertEqual(trained + seg2, full)
        seed = events(self.paths["s1-tel.jsonl"], "train_begin")[0]["torch_initial_seed"]
        self.assertEqual(seed, self.s1.args.seed)
        expected = [train_fps[i] for epoch in range(2) for i in setup.expected_sampler_order(len(train_fps), seed, epoch)]
        self.assertEqual(full, expected[:len(full)])

        # The production check, on the stopped segment and on a corrupted log.
        check = setup.check_data_position(self.s1.train_dataset, self.s1.eval_dataset, self.paths["s1-fp.jsonl"],
                                          seed=seed, first_step=0, last_step=self.STOP, batch_size=2, grad_accum=2)
        self.assertTrue(check["trained_match"] and check["lookahead_ok"], check)
        records = [json.loads(line) for line in self.paths["s1-fp.jsonl"].read_text().splitlines()]
        records[0]["rows"] = list(reversed(records[0]["rows"]))
        swapped = self.root / "swapped-fp.jsonl"
        swapped.write_text("".join(json.dumps(r) + "\n" for r in records))
        check = setup.check_data_position(self.s1.train_dataset, self.s1.eval_dataset, swapped,
                                          seed=seed, first_step=0, last_step=self.STOP, batch_size=2, grad_accum=2)
        self.assertFalse(check["trained_match"])
        # Windows beyond the first epoch are now checked too (see test_dpo_continuation_evidence for the full proof).
        self.assertTrue(setup.check_data_position(self.s1.train_dataset, self.s1.eval_dataset, swapped, seed=seed,
                                                  first_step=3, last_step=6, batch_size=2, grad_accum=2)["checked"])

    def test_optimizer_scheduler_rng_adapter_and_callbacks_are_restored(self):
        saved = [e for e in events(self.paths["s1-tel.jsonl"], "save") if e["global_step"] == self.STOP]
        self.assertEqual(len(saved), 1, "the segment stop must force a save at a non-multiple of save_steps")
        saved = saved[0]
        begin = events(self.paths["s2-tel.jsonl"], "train_begin")[0]
        first = events(self.paths["s2-tel.jsonl"], "first_step_begin")[0]
        self.assertEqual(begin["global_step"], self.STOP)
        self.assertEqual(begin["max_steps"], self.HORIZON)
        self.assertEqual(begin["optimizer"], saved["optimizer"])
        self.assertEqual(begin["optimizer"]["step_values"], [self.STOP])
        self.assertEqual(begin["optimizer"]["param_states"], begin["optimizer"]["params"])
        self.assertGreater(begin["optimizer"]["params"], 0)
        self.assertEqual(begin["scheduler_last_epoch"], saved["scheduler_last_epoch"])
        self.assertEqual(begin["learning_rate"], saved["learning_rate"])
        self.assertEqual(begin["adapter_sha256"], saved["adapter_sha256"])
        self.assertEqual(first["rng"], saved["rng"])
        self.assertIsNone(begin["scaler"])          # no GradScaler on CPU; the GPU run compares real scaler state
        self.assertEqual(begin["scaler"], saved["scaler"])

        stored = json.loads((self.checkpoint / "trainer_state.json").read_text())["stateful_callbacks"]
        counter = stored["EarlyStoppingCallback"]["attributes"]["early_stopping_patience_counter"]
        self.assertGreater(counter, 0, "the fixture must advance the counter for restoration to be observable")
        self.assertEqual(saved["callback_states"], {"EarlyStoppingCallback": counter})
        self.assertEqual(begin["callback_states"], {"EarlyStoppingCallback": counter})

    def test_reference_is_fixed_while_the_policy_moves(self):
        for key in ("reference_chosen_logps", "reference_rejected_logps"):
            for a, b in zip(self.probe_start[key], self.probe_end[key]):
                self.assertTrue(math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-6), f"{key}: {a} != {b}")
        self.assertEqual(self.probe_start["base_sha256"], self.probe_end["base_sha256"])
        self.assertNotEqual(self.probe_start["adapter_sha256"], self.probe_end["adapter_sha256"])
        moved = max(abs(a - b) for a, b in zip(self.probe_start["policy_chosen_logps"], self.probe_end["policy_chosen_logps"]))
        self.assertGreater(moved, 1e-6)
        trainable = [n for n, p in self.s2.model.named_parameters() if p.requires_grad and "lora_" not in n]
        self.assertEqual(trainable, [])
        self.assertIsNone(self.s2.ref_model)
        self.assertTrue(self.s2.is_peft_model)


@unittest.skipIf(MISSING, f"needs {MISSING}; run under the fork's main .venv")
class NonFiniteLossTests(unittest.TestCase):
    def logged_losses(self, **overrides):
        with tempfile.TemporaryDirectory() as d:
            argv = ["--batch_size", "2", "--grad_accum", "1", "--max_steps", "2", "--logging_steps", "1",
                    "--dataloader_num_workers", "0"]
            _, trainer = make_trainer(d, make_rows(8), make_rows(2), argv, save_strategy="no", eval_strategy="no",
                                      load_best_model_at_end=False, **overrides)
            original = trainer.compute_loss

            def nan_loss(model, inputs, return_outputs=False, num_items_in_batch=None):
                return original(model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch) * float("nan")
            trainer.compute_loss = nan_loss
            trainer.train()
            return [h["loss"] for h in trainer.state.log_history if "loss" in h]

    def test_production_config_logs_a_nan_loss_that_the_default_filter_hides(self):
        self.assertTrue(all(math.isnan(v) for v in self.logged_losses()))
        hidden = self.logged_losses(logging_nan_inf_filter=True)
        self.assertTrue(all(math.isfinite(v) for v in hidden), "the default filter is what made finite logs meaningless")


@unittest.skipIf(MISSING, f"needs {MISSING}; run under the fork's main .venv")
class LossVariantTests(unittest.TestCase):
    """--rpo_alpha and --ld_alpha reach TRL's loss through the production config path, not just its config object."""

    def history(self, **overrides):
        with tempfile.TemporaryDirectory() as d:
            argv = ["--batch_size", "2", "--grad_accum", "1", "--max_steps", "2", "--logging_steps", "1",
                    "--dataloader_num_workers", "0"]
            _, trainer = make_trainer(d, make_rows(8), make_rows(2), argv, save_strategy="no", eval_strategy="no",
                                      load_best_model_at_end=False, **overrides)
            trainer.train()
            return [h for h in trainer.state.log_history if "loss" in h]

    def test_rpo_adds_the_nll_term_and_changes_the_loss(self):
        plain, rpo = self.history(), self.history(rpo_alpha=1.0)
        self.assertNotIn("nll_loss", plain[0])
        self.assertIn("nll_loss", rpo[0])
        self.assertNotAlmostEqual(plain[0]["loss"], rpo[0]["loss"], places=6)

    def test_ld_changes_the_loss_when_answer_lengths_differ(self):
        plain, ld = self.history(), self.history(ld_alpha=0.5)
        self.assertNotAlmostEqual(plain[0]["loss"], ld[0]["loss"], places=6)


@unittest.skipIf(MISSING, f"needs {MISSING}; run under the fork's main .venv")
class DigestTests(unittest.TestCase):
    def test_optimizer_digest_covers_every_state_and_group_setting(self):
        from types import SimpleNamespace

        import dpo_setup as setup
        import torch

        params = [torch.nn.Parameter(torch.zeros(1)) for _ in range(9)]
        state = {p: {"step": torch.tensor(51.0), "exp_avg": torch.tensor([float(i)])} for i, p in enumerate(params)}
        optimizer = SimpleNamespace(param_groups=[{"params": params, "lr": 2e-4, "betas": (0.9, 0.999)}], state=state)
        before = setup.optimizer_digest(optimizer)
        self.assertEqual((before["params"], before["param_states"], before["step_values"]), (9, 9, [51]))
        state[params[8]]["exp_avg"].add_(100)
        self.assertNotEqual(setup.optimizer_digest(optimizer)["sha256"], before["sha256"])
        state[params[8]]["exp_avg"].sub_(100)
        self.assertEqual(setup.optimizer_digest(optimizer)["sha256"], before["sha256"])
        optimizer.param_groups[0]["lr"] = 1e-4
        self.assertNotEqual(setup.optimizer_digest(optimizer)["sha256"], before["sha256"])

    def test_bitsandbytes_nested_layout_hashes_like_the_flat_one_and_sees_deep_changes(self):
        from types import SimpleNamespace

        import dpo_setup as setup
        import torch

        param = torch.nn.Parameter(torch.zeros(32, 576))
        buffers = {"state1": torch.zeros(32, 576, dtype=torch.uint8), "qmap1": torch.linspace(-1, 1, 256),
                   "absmax1": torch.ones(72)}
        flat = SimpleNamespace(param_groups=[{"params": [param], "lr": 2e-4}],
                               state={param: {"step": 51, **{k: v.clone() for k, v in buffers.items()}}})
        nested = SimpleNamespace(param_groups=[{"params": [param], "lr": 2e-4}],
                                 state={param: {"step": 51, "__bnb_optimizer_quant_state__":
                                                {k: v.clone() for k, v in buffers.items()}}})
        self.assertEqual(setup.optimizer_digest(flat), setup.optimizer_digest(nested))
        before = setup.optimizer_digest(nested)["sha256"]
        # A change deep inside an 18,432-element buffer: its repr is truncated, its bytes are not.
        nested.state[param]["__bnb_optimizer_quant_state__"]["state1"][16, 300] = 7
        self.assertNotEqual(setup.optimizer_digest(nested)["sha256"], before)
        nested.state[param]["__bnb_optimizer_quant_state__"]["opaque"] = object()
        with self.assertRaises(TypeError):
            setup.optimizer_digest(nested)


@unittest.skipIf(MISSING, f"needs {MISSING}; run under the fork's main .venv")
class PreflightTests(unittest.TestCase):
    """The fail-fast checks accept a real TRL trainer's record and reject each unsafe fact before any update."""

    @classmethod
    def setUpClass(cls):
        import dpo_setup as setup

        cls.tmp = tempfile.TemporaryDirectory()
        argv = ["--batch_size", "2", "--grad_accum", "1", "--max_steps", "4", "--dataloader_num_workers", "0",
                "--reference_probe_rows", "2", "--fail_fast_checks"]
        cls.args, trainer = make_trainer(cls.tmp.name, make_rows(8), make_rows(2), argv)
        cls.record = {"trainer": setup.trainer_record(trainer), "probe_before": setup.reference_probe(trainer, 2)}

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_real_trainer_record_passes(self):
        import dpo_setup as setup

        self.assertEqual(setup.preflight_problems(self.record, self.args), [])

    def test_each_unsafe_fact_is_rejected(self):
        import copy

        import dpo_setup as setup

        mutations = {
            "ref_model": lambda r: r["trainer"].update(ref_model_is_none=False),
            "peft": lambda r: r["trainer"].update(is_peft_model=None),
            "reference_free": lambda r: r["trainer"].update(reference_free=True),
            "precompute": lambda r: r["trainer"].update(precompute_ref_log_probs=True),
            "filter": lambda r: r["trainer"].update(logging_nan_inf_filter=True),
            "callbacks": lambda r: r["trainer"].update(restore_callback_states_from_checkpoint=False),
            "loss_kwargs": lambda r: r["trainer"].update(model_accepts_loss_kwargs=True),
            "accelerate_accumulation": lambda r: r["trainer"].update(accelerator_gradient_accumulation_steps=32),
            "trainer_accumulation": lambda r: r["trainer"].update(gradient_accumulation_steps=32),
            "missing_attribute": lambda r: r["trainer"].pop("model_accepts_loss_kwargs"),
            "non_adapter": lambda r: r["trainer"].update(trainable_non_adapter=["lm_head.weight"]),
            "probe_nan": lambda r: r["probe_before"].update(reference_rejected_logps=[float("nan")] * 2),
            "probe_empty": lambda r: r["probe_before"].update(policy_chosen_logps=[]),
            "probe_missing": lambda r: r.pop("probe_before"),
            "probe_hash": lambda r: r["probe_before"].update(base_sha256="base"),
        }
        for name, mutate in mutations.items():
            record = copy.deepcopy(self.record)
            mutate(record)
            with self.subTest(mutation=name):
                self.assertNotEqual(setup.preflight_problems(record, self.args), [])


@unittest.skipIf(MISSING or importlib.util.find_spec("unsloth") is None, "needs the fork's main .venv with Unsloth installed")
class UnslothTrainingStepPatchTests(unittest.TestCase):
    """Unsloth cannot be imported without a GPU, so its global Trainer patches are checked against its installed source.

    patch_gradient_accumulation_fix rewrites Trainer.training_step by string replacement and wraps Trainer.__init__ to
    force accelerator.gradient_accumulation_steps to 1. On the pinned versions every rewrite target is absent, and the
    accelerator is already at 1, so the patched step normalizes a DPO loss exactly as the CPU equivalence tests prove.
    If a version change makes any target match, this fails and the equivalence must be re-proven.
    """

    def test_rewrite_targets_do_not_match_the_installed_training_step(self):
        import inspect
        import re

        from transformers import Trainer

        unsloth_source = (Path(importlib.util.find_spec("unsloth").submodule_search_locations[0]) / "models/_utils.py").read_text()
        start = unsloth_source.index("def patch_gradient_accumulation_fix(Trainer):")
        patch_source = unsloth_source[start:unsloth_source.index("\ndef ", start + 10)]
        literal_targets = ["loss *= self.args.gradient_accumulation_steps", "if self.model_accepts_loss_kwargs:"]
        for target in literal_targets:
            self.assertIn(repr(target)[1:-1], patch_source.replace('"', "'"), f"Unsloth no longer rewrites {target!r}")
        self.assertIn("accelerator.gradient_accumulation_steps = 1", patch_source)

        training_step = inspect.getsource(Trainer.training_step)
        for target in literal_targets:
            self.assertNotIn(target, training_step)
        regex = (r"else:\n([\s]{4,})self\.accelerator\.backward\(loss, \*\*kwargs\)\n(.+?)if num_items_in_batch is None\:\n"
                 r"(.+?)return loss\.detach\(\) \/ self\.args\.gradient_accumulation_steps")
        self.assertIsNone(re.search(regex, training_step))
        self.assertIn("loss = loss / self.current_gradient_accumulation_steps", training_step)

    def test_accelerator_accumulation_is_already_one_and_dpo_batches_carry_no_labels(self):
        import dpo_setup as setup

        with tempfile.TemporaryDirectory() as d:
            argv = ["--batch_size", "4", "--grad_accum", "32", "--max_steps", "2", "--dataloader_num_workers", "0"]
            _, trainer = make_trainer(d, make_rows(8), make_rows(2), argv)
            self.assertEqual(trainer.args.gradient_accumulation_steps, 32)
            self.assertEqual(setup.trainer_record(trainer)["gradient_accumulation_steps"], 32)
            self.assertEqual(setup.trainer_record(trainer)["accelerator_gradient_accumulation_steps"], 1)
            batch = trainer.data_collator([trainer.train_dataset[0], trainer.train_dataset[1]])
            self.assertNotIn("labels", batch, "without labels num_items_in_batch is None and the loss is divided by the window")


@unittest.skipIf(MISSING, f"needs {MISSING}; run under the fork's main .venv")
class HorizonTests(unittest.TestCase):
    def test_epoch_equivalent_matches_installed_trainer(self):
        import dpo_setup as setup

        class Loader:
            def __init__(self, rows, batch):
                self.dataset, self.batches = range(rows), -(-rows // batch)

            def __len__(self):
                return self.batches

        with tempfile.TemporaryDirectory() as d:
            argv = ["--batch_size", "2", "--grad_accum", "2", "--dataloader_num_workers", "0"]
            _, trainer = make_trainer(d, make_rows(8), make_rows(2), argv)
            for rows, batch, accum, epochs in ((63_184, 4, 32, 3), (12, 2, 4, 3), (13, 2, 4, 1), (24, 2, 2, 3)):
                args = trainer.args
                args.per_device_train_batch_size, args.gradient_accumulation_steps = batch, accum
                args.num_train_epochs, args.max_steps = epochs, -1
                values = trainer.set_initial_training_values(args, Loader(rows, batch), batch * accum)
                with self.subTest(rows=rows, batch=batch, accum=accum):
                    self.assertEqual(values[-1], setup.epoch_equivalent_max_steps(rows, batch, accum, epochs))
        self.assertEqual(setup.epoch_equivalent_max_steps(63_184, 4, 32, 3), 1482)


if __name__ == "__main__":
    unittest.main()
