import random
import json
from pathlib import Path
from types import SimpleNamespace
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import torch

from reasoning.grpo.repro_runtime import (
    BoundaryState,
    capture_rng_state,
    contract_sha256,
    seed_everything,
)
from reasoning.grpo.rollout import temporarily_merged_adapter
from reasoning.grpo.train_t4 import (
    _directory_hashes,
    _isolated_rng,
    _rng_states_equal,
    _verify_selection_artifacts,
    _write_selection_artifacts,
    T4GRPOTrainer,
    build_contract,
    validate_contract,
)


def config():
    return {
        "model": {
            "name": "local-model",
            "revision": "a" * 40,
            "max_new_tokens": 640,
            "min_new_tokens": 0,
            "max_prompt_tokens": 1408,
            "torch_dtype": "float16",
            "attn_implementation": "sdpa",
        },
        "reward": {"path": "local-reward", "archive_sha256": "b" * 64},
        "data": {
            "train_path": "train.jsonl",
            "train_sha256": "c" * 64,
            "dataset_seed": None,
            "train_data_size": 8,
            "num_epochs": 3,
        },
        "training": {
            "rollout_batch_size": 1,
            "n_rollouts": 4,
            "temperature": 0.5,
            "top_p": 0.95,
            "batch_size": 1,
            "gradient_accumulation_steps": 4,
            "learning_rate": 5e-6,
            "num_repeats": 1,
            "buffer_size": 4,
        },
        "loss": {
            "kld_weight": 0.02,
            "entropy_weight": 0.0,
            "loss_implementation": "dr_grpo",
            "max_tokens": 256,
        },
        "runtime": {"seed": 3407, "max_train_events": 1, "max_wall_seconds": 3600},
    }


class GrpoT4ContractTests(unittest.TestCase):
    def test_merged_rollout_restores_float16_frozen_state_exactly(self):
        class LossyMergeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.base = torch.nn.Parameter(
                    torch.tensor([1000.0], dtype=torch.float16), requires_grad=False
                )
                self.adapter = torch.nn.Parameter(
                    torch.tensor([0.3], dtype=torch.float16), requires_grad=True
                )
                self.register_buffer("scale", torch.tensor([2.0], dtype=torch.float16))

            def merge_adapter(self):
                with torch.no_grad():
                    self.base.add_(self.adapter)

            def unmerge_adapter(self):
                with torch.no_grad():
                    self.base.sub_(self.adapter)

        model = LossyMergeModel()
        base_before = model.base.detach().clone()
        buffer_before = model.scale.detach().clone()
        adapter_before = model.adapter.detach().clone()
        with temporarily_merged_adapter(model):
            self.assertFalse(torch.equal(model.base, base_before))
        self.assertTrue(torch.equal(model.base, base_before))
        self.assertTrue(torch.equal(model.scale, buffer_before))
        self.assertTrue(torch.equal(model.adapter, adapter_before))

    def test_merged_rollout_restores_state_after_generation_error(self):
        class FailingMergeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.base = torch.nn.Parameter(
                    torch.tensor([1000.0], dtype=torch.float16), requires_grad=False
                )
                self.adapter = torch.nn.Parameter(
                    torch.tensor([0.3], dtype=torch.float16), requires_grad=True
                )

            def merge_adapter(self):
                with torch.no_grad():
                    self.base.add_(self.adapter)

            def unmerge_adapter(self):
                with torch.no_grad():
                    self.base.sub_(self.adapter)

        model = FailingMergeModel()
        before = model.base.detach().clone()
        with self.assertRaisesRegex(RuntimeError, "decode failed"):
            with temporarily_merged_adapter(model):
                raise RuntimeError("decode failed")
        self.assertTrue(torch.equal(model.base, before))

    def test_runtime_caps_do_not_change_resume_contract(self):
        first = config()
        second = config()
        second["runtime"]["max_train_events"] = 99
        second["runtime"]["max_wall_seconds"] = 12
        self.assertEqual(build_contract(first), build_contract(second))

    def test_reference_geometry_closes_accumulation_window(self):
        contract = build_contract(config())
        validate_contract(contract)

    def test_buffer_must_hold_whole_rollout_groups(self):
        value = config()
        value["training"]["buffer_size"] = 5
        with self.assertRaisesRegex(ValueError, "rollout_batch_size"):
            validate_contract(build_contract(value))

    def test_train_event_must_close_accumulation(self):
        value = config()
        value["training"]["gradient_accumulation_steps"] = 3
        with self.assertRaisesRegex(ValueError, "accumulation"):
            validate_contract(build_contract(value))

    def test_dtype_and_attention_are_t4_specific(self):
        value = config()
        value["model"]["torch_dtype"] = "bfloat16"
        with self.assertRaisesRegex(ValueError, "float16"):
            validate_contract(build_contract(value))

    def test_minimum_generation_cannot_exceed_maximum(self):
        value = config()
        value["model"]["min_new_tokens"] = 641
        with self.assertRaisesRegex(ValueError, "min_new_tokens"):
            validate_contract(build_contract(value))

    def test_evaluation_contract_is_semantic_and_validated(self):
        value = config()
        value["evaluation"] = {
            "path": "eval.jsonl",
            "sha256": "d" * 64,
            "every_train_events": 5,
            "batch_size": 4,
            "max_new_tokens": 640,
            "temperature": 0.2,
            "top_p": 0.95,
            "seed": 16,
        }
        contract = build_contract(value)
        validate_contract(contract)
        changed = config()
        changed["evaluation"] = dict(value["evaluation"], seed=17)
        self.assertNotEqual(contract, build_contract(changed))

        value["evaluation"]["every_train_events"] = 0
        with self.assertRaisesRegex(ValueError, "every_train_events"):
            validate_contract(build_contract(value))

    def test_evaluation_rng_is_exactly_observational(self):
        shuffle = seed_everything(3407)
        before = capture_rng_state(shuffle)
        with _isolated_rng(16, shuffle):
            random.random()
            np.random.random()
            torch.rand(3)
            shuffle.random()
        after = capture_rng_state(shuffle)
        self.assertTrue(_rng_states_equal(before, after))

    def test_evaluation_writes_rows_and_restores_policy_and_rng(self):
        class Model:
            def eval(self):
                return self

        class Tokenizer:
            eos_token_id = 0

            @staticmethod
            def batch_decode(_values, skip_special_tokens):
                self.assertTrue(skip_special_tokens)
                return ["<think>reason</think> answer"]

        with TemporaryDirectory() as directory:
            trainer = T4GRPOTrainer.__new__(T4GRPOTrainer)
            trainer.contract = {
                "model": {"max_prompt_tokens": 1408},
                "evaluation": {
                    "seed": 16,
                    "max_new_tokens": 2,
                    "top_p": 0.95,
                    "temperature": 0.2,
                },
            }
            trainer.eval_loader = [{
                "input_ids": torch.tensor([[7, 8]]),
                "attention_mask": torch.tensor([[1, 1]]),
                "answer": ["answer"],
                "source": [{"eval_id": "eval-000", "_source_index": 42}],
                "item": [{"prompt": [{"role": "user", "content": "question"}]}],
            }]
            trainer.model = Model()
            trainer.tokenizer = Tokenizer()
            trainer.reward_model = object()
            trainer.state = SimpleNamespace(train_events=5)
            trainer.output_dir = Path(directory)
            trainer.events_path = Path(directory) / "events.jsonl"
            trainer.device = torch.device("cpu")
            trainer.shuffle_rng = seed_everything(3407)
            before = capture_rng_state(trainer.shuffle_rng)
            rewards = SimpleNamespace(
                total=np.array([2.5]),
                components={"semantic": np.array([0.75])},
            )
            with patch("reasoning.grpo.train_t4._trainable_sha256", return_value="policy"), patch(
                "reasoning.grpo.train_t4._frozen_policy_sha256", return_value="base"
            ), patch(
                "reasoning.grpo.train_t4.generate_responses",
                return_value=torch.tensor([[7, 8, 9, 0]]),
            ), patch("reasoning.grpo.train_t4.score_completions", return_value=rewards):
                result = trainer.evaluate()

            self.assertTrue(_rng_states_equal(before, capture_rng_state(trainer.shuffle_rng)))
            self.assertEqual(result["rows"], 1)
            self.assertEqual(result["mean_total_reward"], 2.5)
            row = json.loads((Path(directory) / "eval-event-000005.jsonl").read_text())
            self.assertEqual(row["eval_id"], "eval-000")
            self.assertEqual(row["source_index"], 42)
            self.assertEqual(row["completion_tokens"], 1)

    def test_checkpoint_retention_keeps_only_latest_and_best(self):
        with TemporaryDirectory() as directory:
            trainer = T4GRPOTrainer.__new__(T4GRPOTrainer)
            trainer.output_dir = Path(directory)
            trainer.contract = {"run": "pinned"}
            trainer.state = BoundaryState(train_events=1)
            trainer.model = object()
            trainer.optimizer = object()
            trainer.shuffle_rng = random.Random(1)
            trainer.scaler = None
            trainer.buffer = []
            trainer.pending_microbatches = 0

            def fake_save(root, **kwargs):
                destination = Path(root) / f"event-{kwargs['state'].train_events:06d}"
                destination.mkdir(parents=True)
                (destination / "manifest.json").write_text("{}\n")
                return destination

            with patch(
                "reasoning.grpo.train_t4.save_boundary_checkpoint", side_effect=fake_save
            ), patch("reasoning.grpo.train_t4._write_selection_artifacts"):
                trainer.checkpoint()
                trainer.state.train_events = 2
                trainer.state.best_train_event = 1
                trainer.checkpoint()
                self.assertEqual(
                    {path.name for path in (Path(directory) / "checkpoints").glob("event-*")},
                    {"event-000001", "event-000002"},
                )
                trainer.state.train_events = 3
                trainer.state.best_train_event = 3
                trainer.checkpoint()

            self.assertEqual(
                {path.name for path in (Path(directory) / "checkpoints").glob("event-*")},
                {"event-000003"},
            )

    def test_selection_artifacts_survive_segment_boundary_and_reject_damage(self):
        with TemporaryDirectory() as directory:
            output = Path(directory)
            contract = {"run": "pinned"}
            state = BoundaryState(
                train_events=5, best_train_event=3, best_eval_reward=2.75
            )
            checkpoint = output / "checkpoints" / "event-000003"
            checkpoint.mkdir(parents=True)
            (checkpoint / "state.json").write_text(
                json.dumps(
                    {
                        "contract_sha256": contract_sha256(contract),
                        "state": {"train_events": 3},
                    }
                )
            )
            (checkpoint / "adapter_state.pt").write_bytes(b"adapter")
            model = output / "model-best-event-000003"
            model.mkdir()
            (model / "selection.json").write_text(
                json.dumps({"train_event": 3, "mean_total_reward": 2.75})
            )
            (model / "adapter_model.safetensors").write_bytes(b"model")

            _write_selection_artifacts(output, state, contract)
            _verify_selection_artifacts(output, state, contract)
            manifest = json.loads(
                (output / "selection-artifacts.json").read_text()
            )
            self.assertEqual(
                manifest["checkpoint_files"], _directory_hashes(checkpoint)
            )

            (model / "adapter_model.safetensors").write_bytes(b"damaged")
            with self.assertRaisesRegex(RuntimeError, "best model files differ"):
                _verify_selection_artifacts(output, state, contract)


if __name__ == "__main__":
    unittest.main()
