"""Opt-in controls for bounded, resumable preference-training segments. Importable without Unsloth or a GPU.

A segment is part of one full training schedule. The learning-rate horizon (``max_steps``) belongs to the whole schedule;
``stop_after_steps`` ends this segment early without shortening it. Stopping must leave a checkpoint that can resume the
same schedule, and resuming must fail loudly rather than silently start a fresh run.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Written by transformers' Trainer._save_checkpoint for a PEFT model with an optimizer and LR scheduler.
REQUIRED_CHECKPOINT_FILES = ("trainer_state.json", "optimizer.pt", "scheduler.pt", "rng_state.pth", "training_args.bin",
                             "adapter_model.safetensors", "adapter_config.json")


class ContinuationError(RuntimeError):
    """A segment or resume request is inconsistent with the schedule or the checkpoint on disk."""


def validate_segment_args(*, max_steps: int, stop_after_steps: int | None, save_steps: int, eval_steps: int) -> None:
    if max_steps <= 0:
        raise ContinuationError("max_steps must be a positive full-schedule horizon for a resumable run")
    if save_steps <= 0 or eval_steps <= 0:
        raise ContinuationError("save_steps and eval_steps must be positive")
    if stop_after_steps is not None and not 0 < stop_after_steps < max_steps:
        raise ContinuationError(f"stop_after_steps must satisfy 0 < stop_after_steps < max_steps ({max_steps})")


def make_stop_callback(stop_after_steps: int, record: dict[str, Any]):
    """End the segment at `stop_after_steps` applied updates and force a save there.

    Installed transformers saves only when global_step is a multiple of save_steps or reaches max_steps; a segment stop
    is usually neither, so without forcing `should_save` the terminal optimizer state would be lost.
    """
    from transformers import TrainerCallback

    class StopAfterStepsCallback(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step >= stop_after_steps:
                control.should_training_stop = True
                control.should_save = True
                record["stop_after_steps_fired"] = True
                record["stopped_at_global_step"] = int(state.global_step)
            return control

    return StopAfterStepsCallback()


def validate_resume_checkpoint(path: str | Path, *, expected_max_steps: int, require_scaler: bool = False) -> dict:
    """Refuse a checkpoint that cannot faithfully continue this schedule. Returns its recorded state."""
    checkpoint = Path(path)
    if not checkpoint.is_dir():
        raise ContinuationError(f"resume checkpoint {checkpoint} is not a directory")
    name = checkpoint.name
    if not name.startswith("checkpoint-") or not name.removeprefix("checkpoint-").isdigit():
        raise ContinuationError(f"resume checkpoint directory must be named checkpoint-<step>, not {name}")
    required = REQUIRED_CHECKPOINT_FILES + (("scaler.pt",) if require_scaler else ())
    missing = [f for f in required if not (checkpoint / f).is_file()]
    empty = [f for f in required if (checkpoint / f).is_file() and (checkpoint / f).stat().st_size == 0]
    if missing or empty:
        raise ContinuationError(f"resume checkpoint {checkpoint} is incomplete: missing {missing}, empty {empty}")
    try:
        state = json.loads((checkpoint / "trainer_state.json").read_text())
    except json.JSONDecodeError as exc:
        raise ContinuationError(f"trainer_state.json in {checkpoint} is not valid JSON: {exc}") from exc
    step = int(name.removeprefix("checkpoint-"))
    if state.get("global_step") != step:
        raise ContinuationError(f"trainer_state global_step {state.get('global_step')} != directory step {step}")
    if state.get("max_steps") != expected_max_steps:
        raise ContinuationError(f"checkpoint schedule max_steps {state.get('max_steps')} != requested {expected_max_steps}; "
                                "resuming would change the learning-rate horizon")
    if not 0 < step < expected_max_steps:
        raise ContinuationError(f"checkpoint step {step} is not inside the schedule 0 < step < {expected_max_steps}")
    return {"global_step": step, "max_steps": state["max_steps"], "best_model_checkpoint": state.get("best_model_checkpoint"),
            "best_metric": state.get("best_metric"), "files": sorted(p.name for p in checkpoint.iterdir())}


def select_precision(torch_module) -> dict:
    """fp16 compute unless the GPU supports bfloat16 natively. Emulated bfloat16 (as on a T4) does not count."""
    cuda = bool(torch_module.cuda.is_available())
    native_bf16 = cuda and bool(torch_module.cuda.is_bf16_supported(including_emulation=False))
    return {"cuda": cuda, "native_bf16": native_bf16, "bf16": native_bf16, "fp16": cuda and not native_bf16}


def terminal_record(state, stop_record: dict[str, Any], stop_after_steps: int | None) -> dict:
    """Why the segment ended, recorded explicitly so nobody infers it from which checkpoint is newest."""
    record = {
        "global_step": int(state.global_step),
        "max_steps": int(state.max_steps),
        "stop_after_steps": stop_after_steps,
        "stop_after_steps_fired": bool(stop_record.get("stop_after_steps_fired")),
        "best_model_checkpoint": state.best_model_checkpoint,
        "best_metric": state.best_metric,
    }
    record["deadline_fired"] = bool(stop_record.get("deadline_fired"))
    record["reference_guard_fired"] = bool(stop_record.get("reference_guard_fired"))
    record["stop_reason"] = ("reference_guard" if record["reference_guard_fired"]
                             else "target_step" if record["stop_after_steps_fired"]
                             else "deadline" if record["deadline_fired"]
                             else "horizon" if record["global_step"] >= record["max_steps"]
                             else "early_stopping_or_other")
    return record
