"""CPU checks of diagnostic batch routing, evidence and stop behavior."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from posttraining_harness.sft_memory_probe import install_probe, select_full_length_rows


class MemoryProbeTests(unittest.TestCase):
    def test_selection_rejects_short_padded_and_fully_masked_rows(self):
        rows = [
            {"input_ids": [1], "labels": [1]},
            {"input_ids": [1, 2], "labels": [-100, -100]},
            {"input_ids": [1, 2], "labels": [-100, 2], "attention_mask": [1, 0]},
            {"input_ids": [1, 2], "labels": [-100, 2]},
        ]
        selected, indices = select_full_length_rows(rows, 1, 2)
        self.assertEqual(indices, [3])
        self.assertEqual(selected[0]["labels"], [-100, 2])
        with self.assertRaises(AssertionError):
            select_full_length_rows(rows, 2, 2)

    def test_routes_two_stress_updates_and_records_completion(self):
        import torch
        args = SimpleNamespace(per_device_train_batch_size=16,
                               gradient_accumulation_steps=8, max_steps=5826,
                               eval_steps=50, save_steps=50, device="cpu")
        row = {"input_ids": [1] * 2048, "labels": [-100] * 1024 + [1] * 1024}
        callbacks, calls = [], []

        def collate(rows):
            return {key: torch.tensor([r[key] for r in rows]) for key in rows[0]}

        def original(model, inputs, num_items_in_batch=None):
            calls.append((inputs["input_ids"].shape, num_items_in_batch))
            return torch.tensor(0.5)

        trainer = SimpleNamespace(args=args, state=SimpleNamespace(global_step=0),
                                  train_dataset=[row] * 16, data_collator=collate,
                                  training_step=original, add_callback=callbacks.append)
        with tempfile.TemporaryDirectory() as tmp, patch.multiple(
            torch.cuda, device_count=lambda: 1, synchronize=lambda: None,
            reset_peak_memory_stats=lambda: None, mem_get_info=lambda: (8, 16),
            memory_allocated=lambda: 4, memory_reserved=lambda: 8,
            max_memory_allocated=lambda: 6, max_memory_reserved=lambda: 8,
        ):
            report = Path(tmp) / "probe.jsonl"
            install_probe(trainer, str(report))
            normal = {"input_ids": torch.tensor([[2, 3]])}
            for _ in range(8):
                trainer.training_step(None, normal, num_items_in_batch=99)
            trainer.state.global_step = 1
            trainer.training_step(None, normal, num_items_in_batch=99)
            self.assertEqual(calls[-1], (torch.Size([1, 2]), 99))
            trainer.state.global_step = 50
            callbacks[0].on_evaluate(args, trainer.state, None, metrics={"eval_loss": 1.0})
            for _ in range(8):
                trainer.training_step(None, normal, 99)
            self.assertEqual(calls[-1][0], torch.Size([16, 2048]))
            self.assertEqual(calls[-1][1].item(), 16 * 1024 * 8)
            trainer.state.global_step = 60
            control = SimpleNamespace(should_save=False, should_training_stop=False)
            callbacks[0].on_step_end(args, trainer.state, control)
            self.assertTrue(control.should_save and control.should_training_stop)
            callbacks[0].on_train_end(args, trainer.state, control)
            records = [json.loads(line) for line in report.read_text().splitlines()]
            self.assertTrue(records[-1]["passed"])
            self.assertEqual(records[-1]["stress_microbatches"], {"0": 8, "50": 8})
            self.assertEqual(sum(r["event"] == "microbatch" for r in records), 17)


if __name__ == "__main__":
    unittest.main()
