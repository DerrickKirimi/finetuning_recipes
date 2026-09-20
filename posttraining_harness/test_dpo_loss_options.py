"""The DPO-only loss options (RPO's NLL term, LD-DPO's length weight) parse, default off, and reach TRL's DPOConfig."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "preference_optimization"))
import dpo_setup  # noqa: E402


def _history(patch=False, **overrides):
    """Two real training steps on the tiny fixture, returning the logged step records."""
    import tempfile

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import test_dpo_segments_real_trainer as fixture
    from trl_compat import patch_ld_dpo_mask, unpatch_ld_dpo_mask

    with tempfile.TemporaryDirectory() as d:
        argv = ["--batch_size", "2", "--grad_accum", "1", "--max_steps", "2", "--logging_steps", "1",
                "--dataloader_num_workers", "0"]
        _, trainer = fixture.make_trainer(d, fixture.make_rows(8), fixture.make_rows(2), argv, save_strategy="no",
                                          eval_strategy="no", load_best_model_at_end=False, **overrides)
        if patch:
            patch_ld_dpo_mask(trainer)
        try:
            trainer.train()
        finally:
            if patch:  # the patch rebinds a class attribute; restore it for the other tests in this process
                unpatch_ld_dpo_mask(type(trainer))
        return [h for h in trainer.state.log_history if "loss" in h]


class DpoLossOptions(unittest.TestCase):
    def test_default_off(self):
        args = dpo_setup.build_parser().parse_args([])
        self.assertIsNone(args.rpo_alpha)
        self.assertIsNone(args.ld_alpha)

    def test_parse_values(self):
        args = dpo_setup.build_parser().parse_args(["--rpo_alpha", "1.0", "--ld_alpha", "0.5"])
        self.assertEqual((args.rpo_alpha, args.ld_alpha), (1.0, 0.5))

    def test_trl_config_accepts_them(self):
        from trl import DPOConfig
        config = DPOConfig(output_dir="/tmp/unused-dpo-options", rpo_alpha=1.0, ld_alpha=0.5, report_to="none",
                           bf16=False, fp16=False)
        self.assertEqual((config.rpo_alpha, config.ld_alpha), (1.0, 0.5))

    def test_trainer_passes_them_only_for_dpo(self):
        source = (Path(__file__).resolve().parents[1] / "preference_optimization/train_preference.py").read_text()
        self.assertIn("args=DPOConfig(**config_kwargs, **dpo_only)", source)
        self.assertIn('if dpo_only and args.method != "dpo":', source)

    def test_trainer_record_reads_them_from_the_trainer(self):
        from types import SimpleNamespace
        import torch
        model = torch.nn.Linear(2, 2)
        args = SimpleNamespace(logging_nan_inf_filter=False, restore_callback_states_from_checkpoint=True,
                               gradient_accumulation_steps=32, loss_type=["sigmoid"], rpo_alpha=1.0, ld_alpha=None)
        record = dpo_setup.trainer_record(SimpleNamespace(model=model, args=args))
        self.assertEqual((record["rpo_alpha"], record["ld_alpha"], record["loss_type"]), (1.0, None, ["sigmoid"]))

    def test_ld_alpha_without_the_patch_kills_the_gradient(self):
        """The defect this project found in TRL 0.24: LD-DPO masks by absolute position, so every log-prob sums to
        zero and the loss stops depending on the policy. Kept as a regression witness for the patch below."""
        plain, ld = _history(), _history(ld_alpha=0.5)
        self.assertGreater(plain[0]["grad_norm"], 0.0)
        self.assertEqual(ld[0]["grad_norm"], 0.0)
        self.assertEqual(ld[0]["logps/chosen"], 0.0)

    def test_patched_ld_alpha_one_equals_plain_dpo(self):
        """LD-DPO with alpha = 1 weights the whole response equally: by definition it IS standard DPO."""
        plain = _history()
        patched = _history(ld_alpha=1.0, patch=True)
        self.assertAlmostEqual(plain[0]["loss"], patched[0]["loss"], places=6)
        self.assertAlmostEqual(plain[0]["logps/chosen"], patched[0]["logps/chosen"], places=4)

    def test_patched_ld_alpha_half_trains_and_differs(self):
        plain = _history()
        patched = _history(ld_alpha=0.5, patch=True)
        self.assertGreater(patched[0]["grad_norm"], 0.0)
        self.assertLess(patched[0]["logps/chosen"], 0.0)
        self.assertNotAlmostEqual(plain[0]["loss"], patched[0]["loss"], places=6)


if __name__ == "__main__":
    unittest.main()
