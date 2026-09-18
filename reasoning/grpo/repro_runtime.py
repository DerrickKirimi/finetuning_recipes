"""Deterministic, boundary-only checkpoint controls for the T4 GRPO port.

The reference trainer keeps rollout experiences only in memory.  This module
deliberately permits a checkpoint only after a complete train event, when the
rollout buffer is empty and no gradient-accumulation window is open.  That
turns resume into a small, auditable state restoration problem instead of
serializing stochastic rollout tensors and partially accumulated gradients.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from peft import get_peft_model_state_dict, set_peft_model_state_dict


SCHEMA_VERSION = 1


class ResumeContractError(ValueError):
    """Raised when a checkpoint cannot continue the requested run contract."""


@dataclass
class BoundaryState:
    epoch: int = 0
    next_batch: int = 0
    train_events: int = 0
    optimizer_updates: int = 0
    rollout_batches: int = 0
    experiences: int = 0
    best_eval_reward: float | None = None
    best_train_event: int | None = None


def canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def contract_sha256(contract: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(contract).encode("utf-8")).hexdigest()


def seed_everything(seed: int) -> random.Random:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return random.Random(seed)


def capture_rng_state(shuffle_rng: random.Random) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "shuffle": shuffle_rng.getstate(),
    }


def restore_rng_state(state: Mapping[str, Any], shuffle_rng: random.Random) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("cuda"):
        torch.cuda.set_rng_state_all(state["cuda"])
    shuffle_rng.setstate(state["shuffle"])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_boundary_checkpoint(
    checkpoint_root: str | Path,
    *,
    contract: Mapping[str, Any],
    state: BoundaryState,
    model,
    optimizer: torch.optim.Optimizer,
    shuffle_rng: random.Random,
    scaler=None,
    buffer_size: int,
    pending_microbatches: int,
) -> Path:
    """Atomically save an exact resume point at an empty-buffer boundary."""
    if buffer_size != 0:
        raise ResumeContractError("checkpoint requires an empty rollout buffer")
    if pending_microbatches != 0:
        raise ResumeContractError("checkpoint requires a closed accumulation window")

    root = Path(checkpoint_root)
    root.mkdir(parents=True, exist_ok=True)
    final = root / f"event-{state.train_events:06d}"
    if final.exists():
        raise FileExistsError(f"checkpoint already exists: {final}")

    temp = Path(tempfile.mkdtemp(prefix=f".{final.name}-", dir=root))
    try:
        torch.save(get_peft_model_state_dict(model), temp / "adapter_state.pt")
        torch.save(optimizer.state_dict(), temp / "optimizer_state.pt")
        if scaler is not None:
            torch.save(scaler.state_dict(), temp / "scaler_state.pt")
        torch.save(capture_rng_state(shuffle_rng), temp / "rng_state.pt")
        state_record = {
            "schema_version": SCHEMA_VERSION,
            "contract_sha256": contract_sha256(contract),
            "state": asdict(state),
            "boundary": {"buffer_size": 0, "pending_microbatches": 0},
        }
        (temp / "state.json").write_text(
            json.dumps(state_record, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        hashes = {
            path.name: _sha256(path)
            for path in sorted(temp.iterdir())
            if path.is_file()
        }
        (temp / "manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": SCHEMA_VERSION,
                    "contract_sha256": contract_sha256(contract),
                    "files": hashes,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(temp, final)
    except BaseException:
        shutil.rmtree(temp, ignore_errors=True)
        raise
    return final


def load_boundary_checkpoint(
    checkpoint_dir: str | Path,
    *,
    contract: Mapping[str, Any],
    model,
    optimizer: torch.optim.Optimizer,
    shuffle_rng: random.Random,
    scaler=None,
) -> BoundaryState:
    """Verify and restore a boundary checkpoint into freshly built objects."""
    path = Path(checkpoint_dir)
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    expected_contract = contract_sha256(contract)
    if manifest.get("contract_sha256") != expected_contract:
        raise ResumeContractError("checkpoint run contract differs from requested run")
    for name, expected in manifest.get("files", {}).items():
        if _sha256(path / name) != expected:
            raise ResumeContractError(f"checkpoint file hash mismatch: {name}")

    record = json.loads((path / "state.json").read_text(encoding="utf-8"))
    if record.get("boundary") != {"buffer_size": 0, "pending_microbatches": 0}:
        raise ResumeContractError("checkpoint is not an empty-buffer boundary")
    if record.get("contract_sha256") != expected_contract:
        raise ResumeContractError("state run contract differs from requested run")

    adapter_state = torch.load(
        path / "adapter_state.pt", map_location="cpu", weights_only=True
    )
    result = set_peft_model_state_dict(model, adapter_state)
    # Adapter-only loading is intentionally non-strict, so every frozen base
    # weight appears in missing_keys.  Missing LoRA tensors or any unexpected
    # tensor are the actual restore failures.
    missing_adapter = [
        key for key in getattr(result, "missing_keys", []) if "lora_" in key
    ]
    if missing_adapter or getattr(result, "unexpected_keys", None):
        raise ResumeContractError(
            "adapter restore mismatch: "
            f"missing_adapter={missing_adapter}, unexpected={result.unexpected_keys}"
        )
    optimizer.load_state_dict(
        torch.load(path / "optimizer_state.pt", map_location="cpu", weights_only=True)
    )
    scaler_path = path / "scaler_state.pt"
    if scaler is not None:
        if not scaler_path.is_file():
            raise ResumeContractError("checkpoint is missing GradScaler state")
        scaler.load_state_dict(
            torch.load(scaler_path, map_location="cpu", weights_only=True)
        )
    elif scaler_path.is_file():
        raise ResumeContractError("checkpoint has GradScaler state but runtime does not")
    rng_state = torch.load(
        path / "rng_state.pt", map_location="cpu", weights_only=False
    )
    restore_rng_state(rng_state, shuffle_rng)
    return BoundaryState(**record["state"])
