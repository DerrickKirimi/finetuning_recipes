import random
import unittest
from tempfile import TemporaryDirectory

import torch

from reasoning.grpo.repro_runtime import (
    BoundaryState,
    ResumeContractError,
    contract_sha256,
    load_boundary_checkpoint,
    save_boundary_checkpoint,
    seed_everything,
)


class TinyPeftLike(torch.nn.Module):
    """The real PEFT state helpers are patched for this tiny state test."""

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([1.0]))


class GrpoResumeRuntimeTests(unittest.TestCase):
    def test_contract_hash_is_order_independent(self):
        self.assertEqual(contract_sha256({"a": 1, "b": 2}), contract_sha256({"b": 2, "a": 1}))

    def test_checkpoint_requires_clean_boundary(self):
        with TemporaryDirectory() as directory:
            model = TinyPeftLike()
            optimizer = torch.optim.Adam(model.parameters())
            with self.assertRaisesRegex(ResumeContractError, "empty rollout"):
                save_boundary_checkpoint(
                    directory,
                    contract={"run": 1},
                    state=BoundaryState(train_events=1),
                    model=model,
                    optimizer=optimizer,
                    shuffle_rng=random.Random(1),
                    buffer_size=1,
                    pending_microbatches=0,
                )

    def test_save_restore_state_optimizer_adapter_and_rng(self):
        from unittest.mock import patch

        contract = {"model": "pinned", "geometry": {"batch": 1}}
        with TemporaryDirectory() as directory:
            model = TinyPeftLike()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            shuffle_rng = seed_everything(3407)
            loss = model.weight.square().sum()
            loss.backward()
            optimizer.step()
            state = BoundaryState(
                epoch=1,
                next_batch=7,
                train_events=2,
                optimizer_updates=3,
                rollout_batches=7,
                experiences=28,
            )

            with patch(
                "reasoning.grpo.repro_runtime.get_peft_model_state_dict",
                side_effect=lambda value: {"weight": value.weight.detach().clone()},
            ), patch(
                "reasoning.grpo.repro_runtime.set_peft_model_state_dict",
                side_effect=lambda value, saved: value.load_state_dict(saved),
            ):
                checkpoint = save_boundary_checkpoint(
                    directory,
                    contract=contract,
                    state=state,
                    model=model,
                    optimizer=optimizer,
                    shuffle_rng=shuffle_rng,
                    buffer_size=0,
                    pending_microbatches=0,
                )
                expected_weight = model.weight.detach().clone()
                expected_random = shuffle_rng.random()

                model.weight.data.fill_(99)
                restored_rng = random.Random(9)
                restored = load_boundary_checkpoint(
                    checkpoint,
                    contract=contract,
                    model=model,
                    optimizer=optimizer,
                    shuffle_rng=restored_rng,
                )

            self.assertEqual(restored, state)
            self.assertTrue(torch.equal(model.weight, expected_weight))
            self.assertEqual(restored_rng.random(), expected_random)

    def test_changed_contract_fails_before_restore(self):
        from unittest.mock import patch

        with TemporaryDirectory() as directory:
            model = TinyPeftLike()
            optimizer = torch.optim.Adam(model.parameters())
            with patch(
                "reasoning.grpo.repro_runtime.get_peft_model_state_dict",
                side_effect=lambda value: {"weight": value.weight.detach().clone()},
            ):
                checkpoint = save_boundary_checkpoint(
                    directory,
                    contract={"seed": 1},
                    state=BoundaryState(train_events=1),
                    model=model,
                    optimizer=optimizer,
                    shuffle_rng=random.Random(1),
                    buffer_size=0,
                    pending_microbatches=0,
                )
            with self.assertRaisesRegex(ResumeContractError, "contract"):
                load_boundary_checkpoint(
                    checkpoint,
                    contract={"seed": 2},
                    model=model,
                    optimizer=optimizer,
                    shuffle_rng=random.Random(1),
                )


if __name__ == "__main__":
    unittest.main()
