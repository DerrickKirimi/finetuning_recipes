import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from reasoning.generate_grouped_rollouts import (
    batch_path,
    batch_plan,
    completed_batches,
    flatten_batches,
    read_jsonl,
    validate_batch,
    write_batch,
)


def prompt(group_id: str, source_index: int) -> dict:
    return {
        "group_id": group_id,
        "split": "validation",
        "source_index": source_index,
        "instruction": "question",
        "input": "passage",
        "reference": "answer",
    }


def generated(prompt_row: dict, batch_index: int, count: int = 2) -> list[dict]:
    return [
        {
            "group_id": prompt_row["group_id"],
            "split": prompt_row["split"],
            "source_index": prompt_row["source_index"],
            "rollout_index": index,
            "reference": prompt_row["reference"],
            "completion": f"completion {index}",
            "batch_index": batch_index,
        }
        for index in range(count)
    ]


class GroupedRolloutCheckpointTests(unittest.TestCase):
    def test_plan_rejects_duplicate_group_ids(self):
        with self.assertRaisesRegex(ValueError, "unique"):
            batch_plan([prompt("same", 1), prompt("same", 2)], 1)

    def test_resume_accepts_only_contiguous_valid_batches(self):
        prompts = [prompt(f"g{index}", index) for index in range(5)]
        plan = batch_plan(prompts, 2)
        with TemporaryDirectory() as scratch:
            output = Path(scratch)
            (output / "batches").mkdir()
            write_batch(batch_path(output, 0), generated(prompts[0], 0) + generated(prompts[1], 0))
            self.assertEqual(completed_batches(output, plan, 2), 1)
            write_batch(batch_path(output, 2), generated(prompts[4], 2))
            with self.assertRaisesRegex(ValueError, "non-contiguous"):
                completed_batches(output, plan, 2)

    def test_batch_validation_rejects_missing_rollout(self):
        row = prompt("g0", 0)
        with self.assertRaisesRegex(ValueError, "group counts"):
            validate_batch(generated(row, 0, count=1), [row], 2, 0)

    def test_atomic_batches_flatten_in_order(self):
        prompts = [prompt("g0", 0), prompt("g1", 1)]
        with TemporaryDirectory() as scratch:
            output = Path(scratch)
            (output / "batches").mkdir()
            first = generated(prompts[0], 0)
            second = generated(prompts[1], 1)
            write_batch(batch_path(output, 0), first)
            write_batch(batch_path(output, 1), second)
            flatten_batches(output, 2, output / "partial-rollouts.jsonl")
            self.assertEqual(read_jsonl(output / "partial-rollouts.jsonl"), first + second)
            self.assertFalse(any(output.rglob("*.partial")))


if __name__ == "__main__":
    unittest.main()
