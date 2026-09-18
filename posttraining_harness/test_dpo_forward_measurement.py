"""Real TRL DPOTrainer, real PEFT, a tiny random Llama, CPU only.

The unit tests prove the verdict uses forward measurements. This proves the measurements
exist: that both hooks fire through a PEFT-wrapped model inside TRL's own training loop,
that the reference and evaluation passes are recorded but not credited, and - the question
the v4 review raised - that TRL really does deliver a shorter input than the collator's
pre-truncation bound when rows exceed `max_length`.

Needs torch, transformers, trl, peft, tokenizers and datasets. It does not use Unsloth, so it
says nothing about Unsloth's patched path; the hooks are designed to fire there too, and the
hardware run records `forwards_recorded` so a silent failure is visible.
"""
import importlib.util
import tempfile
import unittest

REQUIRED = ("torch", "transformers", "trl", "peft", "tokenizers", "datasets")
MISSING = [name for name in REQUIRED if importlib.util.find_spec(name) is None]


@unittest.skipIf(MISSING, f"needs {MISSING}; run under the fork's main .venv, not posttraining_harness/.venv")
class RealTrainerForwardMeasurementTests(unittest.TestCase):
    MAX_LENGTH = 24
    MAX_PROMPT = 16

    @classmethod
    def setUpClass(cls):
        import torch
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model
        from tokenizers import Tokenizer, models, pre_tokenizers
        from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
        from trl import DPOConfig, DPOTrainer

        import posttraining_harness.dpo_memory_probe as probe_module
        from posttraining_harness.dpo_memory_probe import BoundedProbe, attach

        words = [f"w{i}" for i in range(40)]
        vocab = {"<pad>": 0, "<eos>": 1, "<unk>": 2}
        vocab.update({w: i + 3 for i, w in enumerate(words)})
        backend = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
        backend.pre_tokenizer = pre_tokenizers.Whitespace()
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="<pad>",
                                            eos_token="<eos>", unk_token="<unk>")

        torch.manual_seed(0)
        model = LlamaForCausalLM(LlamaConfig(
            vocab_size=len(vocab), hidden_size=16, intermediate_size=32, num_hidden_layers=1,
            num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=128,
            pad_token_id=0, bos_token_id=1, eos_token_id=1))
        model = get_peft_model(model, LoraConfig(r=2, lora_alpha=2, task_type="CAUSAL_LM",
                                                 target_modules=["q_proj", "v_proj"]))

        def text(n, offset):
            return " ".join(words[(offset + i) % len(words)] for i in range(n))

        # Prompt 20 words -> 16 after max_prompt_length; chosen 15 + eos = 16. Pre-truncation
        # longest row 32, above MAX_LENGTH 24, so TRL must truncate at forward.
        rows = [{"prompt": text(20, i), "chosen": text(15, i + 1), "rejected": text(9, i + 2)}
                for i in range(4)]
        cls.tmp = tempfile.TemporaryDirectory()
        trainer = DPOTrainer(
            model=model, ref_model=None, processing_class=tokenizer,
            train_dataset=Dataset.from_list(rows), eval_dataset=Dataset.from_list(rows[:2]),
            args=DPOConfig(
                output_dir=cls.tmp.name, per_device_train_batch_size=2,
                per_device_eval_batch_size=2, gradient_accumulation_steps=1, max_steps=2,
                learning_rate=1e-3, logging_steps=1, report_to="none", save_strategy="no",
                eval_strategy="steps", eval_steps=1, max_length=cls.MAX_LENGTH,
                max_prompt_length=cls.MAX_PROMPT, use_cpu=True, bf16=False, fp16=False,
                dataloader_num_workers=0, seed=0))

        cls.original_snapshot = probe_module.memory_snapshot
        probe_module.memory_snapshot = lambda: {}          # CUDA-only; not under test here
        cls.probe = BoundedProbe(lambda *a, **k: None, max_optimizer_steps=1,
                                 post_eval_steps=1, stress_target_tokens=cls.MAX_LENGTH)
        attach(trainer, cls.probe)
        trainer.train()
        cls.verdict = cls.probe.verdict()

    @classmethod
    def tearDownClass(cls):
        import posttraining_harness.dpo_memory_probe as probe_module
        probe_module.memory_snapshot = cls.original_snapshot
        cls.tmp.cleanup()

    def _forwards(self, phase=None, grad=None, source=None):
        return [f for f in self.probe.forwards
                if (phase is None or f["phase"] == phase)
                and (grad is None or f["grad_enabled"] is grad)
                and (source is None or f["source"] == source)]

    def test_both_hooks_fire_through_peft_inside_trl(self):
        self.assertTrue(self._forwards(source="model_call"), "top-level call hook never fired")
        self.assertTrue(self._forwards(source="input_embeddings"),
                        "embedding hook never fired")

    def test_trl_truncates_below_the_collator_bound(self):
        """The reason the verdict moved: the collator saw 32, the model received 24."""
        collated = max(r["max_row_pre_truncation_tokens"] for r in self.probe.shapes
                       if r["phase"] == "training")
        delivered = max(f["max_row_tokens"] for f in self._forwards("training", grad=True))
        self.assertEqual(collated, 32)
        self.assertEqual(delivered, self.MAX_LENGTH)
        self.assertLess(delivered, collated)

    def test_both_measurement_points_agree(self):
        for phase in ("training", "post_evaluation"):
            by_source = {s: max(f["max_row_tokens"] for f in self._forwards(phase, True, s))
                         for s in ("model_call", "input_embeddings")}
            self.assertEqual(by_source["model_call"], by_source["input_embeddings"], phase)

    def test_reference_and_evaluation_passes_are_recorded_without_grad(self):
        self.assertTrue(self._forwards("training", grad=False),
                        "the adapter-disabled reference pass was not observed")
        evaluation = self._forwards("evaluation")
        self.assertTrue(evaluation, "evaluation forwards were not observed")
        self.assertTrue(all(not f["grad_enabled"] for f in evaluation))

    def test_collation_phase_lags_the_forward_phase_by_one_batch(self):
        """Why supervision is no longer judged per phase. Accelerate prefetches the next batch,
        so the one batch trained after evaluation here was collated before it."""
        post_collations = [r for r in self.probe.shapes if r["phase"] == "post_evaluation"]
        post_forwards = self._forwards("post_evaluation", grad=True)
        self.assertEqual(post_collations, [])
        self.assertTrue(post_forwards)
        self.assertTrue(self.verdict["stress"]["per_phase"]["post_evaluation"]["reached_target"])

    def test_a_target_above_what_the_model_received_is_refused(self):
        """Exactly the false pass v5 allowed: the collator's 32 would clear a 30 target."""
        self.assertTrue(self.verdict["stress"]["reached_target"])
        self.probe.stress_target_tokens = 30
        try:
            refused = self.probe.verdict()
        finally:
            self.probe.stress_target_tokens = self.MAX_LENGTH
        self.assertFalse(refused["stress"]["reached_target"])
        self.assertFalse(refused["fit"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
