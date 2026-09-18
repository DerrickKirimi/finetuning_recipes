import json
import signal
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from transformers import BertConfig, BertModel, BertTokenizerFast

from reward_models.reward_training import (
    RewardRegressor,
    RewardRecord,
    TrainingConfig,
    latest_checkpoint,
    load_records,
    run_training,
    tokenize_records,
)
from reward_models.reward_model import load_reward_model


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


class RewardTrainingTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.model_dir = self.root / "tiny-model"
        self.model_dir.mkdir()
        vocabulary = [
            "[PAD]",
            "[UNK]",
            "[CLS]",
            "[SEP]",
            "[MASK]",
            "alpha",
            "beta",
            "gamma",
            "delta",
            "answer",
            "wrong",
            "right",
        ]
        (self.model_dir / "vocab.txt").write_text("\n".join(vocabulary) + "\n")
        tokenizer = BertTokenizerFast(vocab_file=str(self.model_dir / "vocab.txt"))
        tokenizer.save_pretrained(self.model_dir)
        torch.manual_seed(123)
        BertModel(
            BertConfig(
                vocab_size=len(tokenizer),
                hidden_size=16,
                num_hidden_layers=2,
                num_attention_heads=2,
                intermediate_size=24,
                max_position_embeddings=64,
            )
        ).save_pretrained(self.model_dir)
        self.tokenizer = tokenizer
        self.train_file = self.root / "train.jsonl"
        self.validation_file = self.root / "validation.jsonl"
        rows = [
            {
                "reference": f"alpha beta gamma {i}",
                "response": "right answer" if i % 2 else "wrong answer",
                "score": 5.0 if i % 2 else 1.0,
            }
            for i in range(8)
        ]
        write_jsonl(self.train_file, rows)
        write_jsonl(self.validation_file, rows[:4])

    def tearDown(self):
        self.temporary.cleanup()

    def config(self, output_dir: Path, **overrides) -> TrainingConfig:
        values = dict(
            train_file=str(self.train_file),
            validation_file=str(self.validation_file),
            output_dir=str(output_dir),
            model=str(self.model_dir),
            local_files_only=True,
            max_length=16,
            batch_size=2,
            epochs=3,
            patience=10,
            unfreeze_layers=1,
            validation_every=2,
            checkpoint_every=1,
            keep_checkpoints=4,
            seed=17,
            device="cpu",
        )
        values.update(overrides)
        return TrainingConfig(**values)

    def test_balanced_pair_keeps_response_tokens_under_truncation(self):
        record = RewardRecord(" ".join(["alpha"] * 30), "right answer", 5.0)
        encoded = tokenize_records(
            self.tokenizer, [record], contract="balanced_pair", max_length=12
        )
        tokens = self.tokenizer.convert_ids_to_tokens(encoded["input_ids"][0])
        self.assertIn("right", tokens)
        self.assertIn("answer", tokens)
        self.assertEqual(set(encoded), {"input_ids", "attention_mask", "labels"})

    def test_resume_matches_uninterrupted_model_and_optimizer(self):
        uninterrupted_dir = self.root / "uninterrupted"
        resumed_dir = self.root / "resumed"
        full = run_training(self.config(uninterrupted_dir, max_updates=5))
        paused = run_training(self.config(resumed_dir, stop_after_updates=1, max_updates=5))
        self.assertEqual(paused["status"], "paused")
        self.assertIsNone(paused["best_mse"])
        self.assertEqual(paused["patience_counter"], 0)
        resumed = run_training(self.config(resumed_dir, resume_latest=True, max_updates=5))
        self.assertEqual(full["final_parameter_sha256"], resumed["final_parameter_sha256"])
        full_state = torch.load(latest_checkpoint(uninterrupted_dir), weights_only=False)
        resumed_state = torch.load(latest_checkpoint(resumed_dir), weights_only=False)
        self.assertEqual(full_state["global_step"], resumed_state["global_step"])
        self.assertEqual(full_state["epoch"], resumed_state["epoch"])
        self.assertEqual(full_state["next_batch_index"], resumed_state["next_batch_index"])
        self.assertEqual(full_state["best_mse"], resumed_state["best_mse"])
        self.assertEqual(full_state["patience_counter"], resumed_state["patience_counter"])
        self.assertEqual(full_state["optimizer"].keys(), resumed_state["optimizer"].keys())
        for name, tensor in full_state["model"].items():
            torch.testing.assert_close(tensor, resumed_state["model"][name], rtol=0, atol=0)
        self.assert_nested_equal(full_state["optimizer"], resumed_state["optimizer"])

    def test_balanced_export_loads_and_scores_with_its_contract(self):
        output_dir = self.root / "export"
        run_training(self.config(output_dir, max_updates=2))
        metadata = json.loads(
            (output_dir / "current/reward_model_metadata.json").read_text(encoding="utf-8")
        )
        self.assertEqual(metadata["tokenization_contract"], "balanced_pair")
        scorer, _ = load_reward_model(output_dir / "current")
        self.assertEqual(scorer.tokenization_contract, "balanced_pair")
        self.assertEqual(scorer.pooling, "meanmax")
        self.assertEqual(scorer.max_length, 16)
        score = scorer.score(" ".join(["alpha"] * 30), "right answer")
        self.assertTrue(torch.isfinite(torch.tensor(score)))
        responses = ["right answer", "wrong answer"]
        references = ["alpha beta", "gamma delta"]
        batched = scorer.batch_score(responses, references, batch_size=2)
        scalar = [
            scorer.score(reference=reference, response=response)
            for response, reference in zip(responses, references)
        ]
        np.testing.assert_allclose(batched, scalar, rtol=1e-6, atol=1e-6)

    def test_resume_rejects_semantic_contract_change(self):
        output_dir = self.root / "mismatch"
        run_training(self.config(output_dir, stop_after_updates=1))
        previous_handler = signal.getsignal(signal.SIGTERM)
        with self.assertRaisesRegex(ValueError, "Resume contract mismatch"):
            run_training(self.config(output_dir, resume_latest=True, batch_size=4))
        self.assertIs(signal.getsignal(signal.SIGTERM), previous_handler)

    def test_gradient_checkpointing_is_opt_in_and_in_resume_contract(self):
        plain = RewardRegressor(
            str(self.model_dir),
            model_revision=None,
            dropout=0.2,
            unfreeze_layers=2,
            pooling="meanmax",
            attention_implementation="eager",
            local_files_only=True,
        )
        checkpointed = RewardRegressor(
            str(self.model_dir),
            model_revision=None,
            dropout=0.2,
            unfreeze_layers=2,
            pooling="meanmax",
            attention_implementation="eager",
            local_files_only=True,
            gradient_checkpointing=True,
        )
        self.assertFalse(plain.encoder.is_gradient_checkpointing)
        self.assertTrue(checkpointed.encoder.is_gradient_checkpointing)
        encoded = self.tokenizer(
            "alpha beta", "right answer", return_tensors="pt", padding=True
        )
        checkpointed(
            encoded["input_ids"], encoded["attention_mask"]
        ).sum().backward()
        self.assertTrue(
            all(
                any(parameter.grad is not None for parameter in layer.parameters())
                for layer in checkpointed.encoder.encoder.layer
            )
        )

        output_dir = self.root / "gradient-checkpointing"
        run_training(
            self.config(
                output_dir,
                unfreeze_layers=2,
                gradient_checkpointing=True,
                stop_after_updates=1,
            )
        )
        manifest = json.loads((output_dir / "run-manifest.json").read_text())
        self.assertTrue(manifest["semantic_contract"]["gradient_checkpointing"])
        with self.assertRaisesRegex(ValueError, "Resume contract mismatch"):
            run_training(
                self.config(
                    output_dir,
                    unfreeze_layers=2,
                    resume_latest=True,
                    gradient_checkpointing=False,
                )
            )

    def test_gradient_checkpointing_resume_matches_uninterrupted(self):
        uninterrupted_dir = self.root / "checkpointed-uninterrupted"
        resumed_dir = self.root / "checkpointed-resumed"
        options = {"unfreeze_layers": 2, "gradient_checkpointing": True, "max_updates": 3}
        full = run_training(self.config(uninterrupted_dir, **options))
        paused = run_training(
            self.config(resumed_dir, stop_after_updates=1, **options)
        )
        self.assertEqual(paused["status"], "paused")
        resumed = run_training(
            self.config(resumed_dir, resume_latest=True, **options)
        )
        self.assertEqual(full["final_parameter_sha256"], resumed["final_parameter_sha256"])
        full_state = torch.load(latest_checkpoint(uninterrupted_dir), weights_only=False)
        resumed_state = torch.load(latest_checkpoint(resumed_dir), weights_only=False)
        for name, tensor in full_state["model"].items():
            torch.testing.assert_close(tensor, resumed_state["model"][name], rtol=0, atol=0)
        self.assert_nested_equal(full_state["optimizer"], resumed_state["optimizer"])

    def test_gradient_checkpointing_preserves_one_update_exactly(self):
        plain_dir = self.root / "plain-one-update"
        checkpointed_dir = self.root / "checkpointed-one-update"
        options = {"unfreeze_layers": 2, "max_updates": 1}
        plain = run_training(self.config(plain_dir, gradient_checkpointing=False, **options))
        checkpointed = run_training(
            self.config(checkpointed_dir, gradient_checkpointing=True, **options)
        )

        self.assertEqual(
            plain["final_parameter_sha256"], checkpointed["final_parameter_sha256"]
        )
        plain_state = torch.load(latest_checkpoint(plain_dir), weights_only=False)
        checkpointed_state = torch.load(
            latest_checkpoint(checkpointed_dir), weights_only=False
        )
        for name, tensor in plain_state["model"].items():
            torch.testing.assert_close(
                tensor, checkpointed_state["model"][name], rtol=0, atol=0
            )
        self.assert_nested_equal(
            plain_state["optimizer"], checkpointed_state["optimizer"]
        )
        self.assert_nested_equal(
            plain_state["rng_state"], checkpointed_state["rng_state"]
        )

    def test_resume_rejects_a_terminal_checkpoint(self):
        output_dir = self.root / "terminal"
        run_training(self.config(output_dir, max_updates=1))
        with self.assertRaisesRegex(ValueError, "terminal run"):
            run_training(self.config(output_dir, resume_latest=True, max_updates=2))

    def test_latest_checkpoint_rejects_corruption(self):
        output_dir = self.root / "corrupt"
        run_training(self.config(output_dir, stop_after_updates=1))
        checkpoint = latest_checkpoint(output_dir)
        with checkpoint.open("ab") as handle:
            handle.write(b"corrupt")
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            latest_checkpoint(output_dir)

    def test_head_only_training_has_no_empty_encoder_optimizer_group(self):
        output_dir = self.root / "head-only"
        run_training(self.config(output_dir, unfreeze_layers=0, stop_after_updates=1))
        checkpoint = torch.load(latest_checkpoint(output_dir), weights_only=False)
        self.assertEqual(len(checkpoint["optimizer"]["param_groups"]), 1)
        self.assertEqual(checkpoint["optimizer"]["param_groups"][0]["name"], "head")

    def test_loader_rejects_the_reference_scripts_pre_normalization_schema(self):
        path = self.root / "wrong-schema.jsonl"
        write_jsonl(
            path,
            [
                {
                    "orig_reference_answer": "alpha",
                    "orig_response": "beta",
                    "orig_score": 3.0,
                }
            ],
        )
        with self.assertRaisesRegex(ValueError, "Invalid reward row"):
            load_records(path)

    def assert_nested_equal(self, left, right):
        if isinstance(left, torch.Tensor):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        elif isinstance(left, np.ndarray):
            np.testing.assert_array_equal(left, right)
        elif isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys())
            for key in left:
                self.assert_nested_equal(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            self.assertEqual(len(left), len(right))
            for left_item, right_item in zip(left, right):
                self.assert_nested_equal(left_item, right_item)
        else:
            self.assertEqual(left, right)


if __name__ == "__main__":
    unittest.main()
