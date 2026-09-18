"""Opt-in diagnostic instrumentation for an existing SFT trainer.

This deliberately replaces two updates with full-length examples. Its checkpoints
are diagnostic artifacts, not an unchanged-data training continuation.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
from pathlib import Path


def select_full_length_rows(dataset, count: int, length: int):
    selected, indices = [], []
    for index, row in enumerate(dataset):
        ids, labels = row["input_ids"], row["labels"]
        attention = row.get("attention_mask", [1] * len(ids))
        if (len(ids) == length and len(labels) == length
                and all(attention) and any(label != -100 for label in labels)):
            selected.append({"input_ids": ids, "attention_mask": attention, "labels": labels})
            indices.append(index)
            if len(selected) == count:
                return selected, indices
    raise AssertionError(f"need {count} response-bearing rows of length {length}; found {len(selected)}")


def install_probe(trainer, output: str, stop_step: int = 60):
    import torch
    from transformers import TrainerCallback

    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    assert not path.exists(), f"refuse to overwrite probe evidence: {path}"

    def emit(event, **fields):
        record = {"event": event, "monotonic_seconds": time.monotonic(), **fields}
        with path.open("a") as handle:
            handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        if event != "microbatch":
            print("MEMORY_PROBE " + json.dumps(record, sort_keys=True), flush=True)

    def memory():
        free, total = torch.cuda.mem_get_info()
        return {"allocated": torch.cuda.memory_allocated(),
                "reserved": torch.cuda.memory_reserved(),
                "peak_allocated": torch.cuda.max_memory_allocated(),
                "peak_reserved": torch.cuda.max_memory_reserved(),
                "device_free": free, "device_total": total}

    batch_size = trainer.args.per_device_train_batch_size
    accumulation = trainer.args.gradient_accumulation_steps
    assert batch_size == 16 and accumulation == 8
    assert trainer.args.max_steps == 5826
    assert trainer.args.eval_steps == 50 and trainer.args.save_steps == 50
    assert torch.cuda.device_count() == 1
    rows, indices = select_full_length_rows(trainer.train_dataset, batch_size, 2048)
    encoded = json.dumps(rows, sort_keys=True).encode()
    supervised = sum(sum(x != -100 for x in row["labels"][1:]) for row in rows)
    assert supervised > 0
    stress_before_steps = (0, 50)
    counters = {str(step): 0 for step in stress_before_steps}
    completed_evaluations = []
    emit("configuration", diagnostic_only=True, batch_size=batch_size,
         grad_accum=accumulation, max_steps=trainer.args.max_steps,
         stress_before_steps=list(stress_before_steps),
         selected_train_indices=indices, selected_rows_sha256=hashlib.sha256(encoded).hexdigest(),
         supervised_tokens_per_stress_microbatch=supervised,
         allocator=os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
         model_accepts_loss_kwargs=getattr(trainer, "model_accepts_loss_kwargs", None))

    original_step = trainer.training_step

    def measured_step(model, inputs, *args, **kwargs):
        step = int(trainer.state.global_step)
        stress = step in stress_before_steps
        if stress:
            inputs = trainer.data_collator(rows)
            # Trainer normalizes over the whole accumulation group when it accepts
            # num_items_in_batch. The replacement group repeats these rows 8 times.
            items = torch.tensor(supervised * accumulation, device=trainer.args.device)
            if args:
                args = (items, *args[1:])
            elif "num_items_in_batch" in kwargs:
                kwargs["num_items_in_batch"] = items
            counters[str(step)] += 1
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        before = memory()
        started = time.monotonic()
        try:
            loss = original_step(model, inputs, *args, **kwargs)
            torch.cuda.synchronize()
            value = float(loss.detach().float().cpu())
            assert math.isfinite(value), f"nonfinite loss at step {step}"
            emit("microbatch", before_step=step, stress=stress,
                 input_shape=list(inputs["input_ids"].shape), loss=value,
                 elapsed_seconds=time.monotonic() - started, before=before, after=memory())
            return loss
        except Exception as exc:
            emit("microbatch_failure", before_step=step, stress=stress,
                 error_type=type(exc).__name__, error=str(exc), memory=memory())
            raise

    trainer.training_step = measured_step

    class MemoryCallback(TrainerCallback):
        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            torch.cuda.synchronize()
            completed_evaluations.append(int(state.global_step))
            emit("evaluation_complete", step=int(state.global_step), metrics=metrics, memory=memory())

        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step >= stop_step:
                control.should_save = True
                control.should_training_stop = True
            return control

        def on_train_end(self, args, state, control, **kwargs):
            passed = (state.global_step == stop_step and 50 in completed_evaluations
                      and all(count == accumulation for count in counters.values()))
            emit("probe_complete", passed=passed, global_step=int(state.global_step),
                 stress_microbatches=counters, evaluations=completed_evaluations,
                 memory=memory())
            assert passed, "memory probe did not exercise both full-length updates and evaluation"

    trainer.add_callback(MemoryCallback())
