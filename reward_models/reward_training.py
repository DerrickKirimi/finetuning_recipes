"""Restartable reward-model training with explicit data and tokenization contracts.

This module is a reproducible alternative to ``train_reward_model.py``.  The
reference script remains unchanged so its original behavior stays inspectable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import signal
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr
from transformers import AutoModel, AutoTokenizer


REFERENCE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOKENIZATION_CONTRACTS = ("balanced_pair", "reference_tail")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_directory(path: Path) -> str:
    """Hash a local model snapshot by relative name and file contents."""
    digest = hashlib.sha256()
    files = sorted(candidate for candidate in path.rglob("*") if candidate.is_file())
    if not files:
        raise ValueError(f"Model directory contains no files: {path}")
    for candidate in files:
        relative = candidate.relative_to(path).as_posix()
        digest.update(relative.encode("utf-8") + b"\0")
        digest.update(sha256_file(candidate).encode("ascii") + b"\0")
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_torch_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        torch.save(value, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(canonical_json(value) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


@dataclass(frozen=True)
class RewardRecord:
    reference: str
    response: str
    score: float


@dataclass(frozen=True)
class TrainingConfig:
    train_file: str
    validation_file: str
    output_dir: str
    model: str = REFERENCE_MODEL
    model_revision: str | None = None
    tokenization_contract: str = "balanced_pair"
    max_length: int = 512
    batch_size: int = 64
    learning_rate_head: float = 5e-4
    learning_rate_encoder: float = 5e-5
    dropout: float = 0.2
    epochs: int = 50
    patience: int = 5
    unfreeze_layers: int = 3
    pooling: str = "meanmax"
    attention_implementation: str = "eager"
    validation_every: int = 250
    checkpoint_every: int = 250
    keep_checkpoints: int = 2
    logging_every: int = 10
    seed: int = 42
    device: str = "auto"
    deterministic_algorithms: bool = True
    gradient_checkpointing: bool = False
    local_files_only: bool = False
    resume_from: str | None = None
    resume_latest: bool = False
    max_updates: int | None = None
    stop_after_updates: int | None = None
    max_runtime_seconds: float | None = None
    measure_step_time: bool = False


def load_records(path: Path) -> list[RewardRecord]:
    records: list[RewardRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            raw = json.loads(line)
            try:
                reference = raw["reference"]
                response = raw["response"]
                score = float(raw["score"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid reward row at {path}:{line_number}: {exc}") from exc
            if not isinstance(reference, str) or not isinstance(response, str):
                raise ValueError(f"Non-string text at {path}:{line_number}")
            if not math.isfinite(score) or not 1.0 <= score <= 5.0:
                raise ValueError(f"Score outside finite [1, 5] at {path}:{line_number}: {score}")
            records.append(RewardRecord(reference, response, score))
    if not records:
        raise ValueError(f"No records found in {path}")
    return records


def get_encoder_layers(encoder: nn.Module) -> Sequence[nn.Module]:
    if hasattr(encoder, "encoder") and hasattr(encoder.encoder, "layer"):
        return encoder.encoder.layer
    if hasattr(encoder, "transformer") and hasattr(encoder.transformer, "layer"):
        return encoder.transformer.layer
    raise AttributeError(f"Unsupported encoder layout for {encoder.__class__.__name__}")


class RewardRegressor(nn.Module):
    def __init__(
        self,
        model_name_or_path: str,
        *,
        model_revision: str | None,
        dropout: float,
        unfreeze_layers: int,
        pooling: str,
        attention_implementation: str,
        local_files_only: bool,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        if pooling not in {"mean", "max", "meanmax"}:
            raise ValueError(f"Unsupported pooling: {pooling}")
        self.pooling = pooling
        self.encoder = AutoModel.from_pretrained(
            model_name_or_path,
            revision=model_revision,
            local_files_only=local_files_only,
            attn_implementation=attention_implementation,
        )
        if gradient_checkpointing:
            if not self.encoder.supports_gradient_checkpointing:
                raise ValueError(
                    f"{self.encoder.__class__.__name__} does not support gradient checkpointing"
                )
            # Non-reentrant checkpointing does not require an input tensor with
            # requires_grad=True. That matters for token-ID inputs and for later
            # layer-only variants whose embedding/early-layer parameters stay frozen.
            self.encoder.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        layers = get_encoder_layers(self.encoder)
        if not 0 <= unfreeze_layers <= len(layers):
            raise ValueError(
                f"unfreeze_layers must be in [0, {len(layers)}], got {unfreeze_layers}"
            )
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False
        for layer in layers[-unfreeze_layers:] if unfreeze_layers else []:
            for parameter in layer.parameters():
                parameter.requires_grad = True
        pool_dim = self.encoder.config.hidden_size * (2 if pooling == "meanmax" else 1)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(pool_dim, 1))

    def pool(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        mean = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        if self.pooling == "mean":
            return mean
        maximum = hidden.masked_fill(mask == 0, float("-inf")).max(1).values
        if self.pooling == "max":
            return maximum
        return torch.cat([mean, maximum], dim=-1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # token_type_ids are deliberately omitted for parity with the reference encoder call.
        encoded = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.pool(encoded.last_hidden_state, attention_mask)
        return self.head(pooled).squeeze(-1)


def tokenize_records(
    tokenizer: Any,
    records: Sequence[RewardRecord],
    *,
    contract: str,
    max_length: int,
) -> dict[str, torch.Tensor]:
    if contract == "balanced_pair":
        encoded = tokenizer(
            [record.reference for record in records],
            [record.response for record in records],
            padding=True,
            truncation="longest_first",
            max_length=max_length,
            return_tensors="pt",
        )
    elif contract == "reference_tail":
        encoded = tokenizer(
            [f"{record.reference} [SEP] {record.response}" for record in records],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
    else:
        raise ValueError(f"Unknown tokenization contract: {contract}")
    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
        "labels": torch.tensor([(record.score - 1.0) / 4.0 for record in records]),
    }


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int, deterministic_algorithms: bool) -> None:
    if deterministic_algorithms:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic_algorithms)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = deterministic_algorithms


def epoch_order(length: int, seed: int, epoch: int) -> list[int]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + epoch)
    return torch.randperm(length, generator=generator).tolist()


def batches_for_epoch(length: int, batch_size: int, seed: int, epoch: int) -> list[list[int]]:
    order = epoch_order(length, seed, epoch)
    return [order[start : start + batch_size] for start in range(0, length, batch_size)]


def optimizer_state_to_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: optimizer_state_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [optimizer_state_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(optimizer_state_to_cpu(item) for item in value)
    return value


def state_dict_to_cpu(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu() for key, value in model.state_dict().items()}


def tensor_digest(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(tensor.dtype).encode("ascii") + b"\0")
        digest.update(str(tuple(tensor.shape)).encode("ascii") + b"\0")
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def model_identity(model: str, revision: str | None) -> dict[str, Any]:
    local_path = Path(model).expanduser()
    if local_path.is_dir():
        return {"kind": "local_directory", "sha256": sha256_directory(local_path.resolve())}
    return {"kind": "hub", "model": model, "revision": revision}


def runtime_identity(device: torch.device) -> dict[str, Any]:
    identity: dict[str, Any] = {
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "device_type": device.type,
        "cuda_build": torch.version.cuda,
    }
    if device.type == "cuda":
        identity["cuda_device_name"] = torch.cuda.get_device_name(device)
        identity["cuda_capability"] = list(torch.cuda.get_device_capability(device))
    return identity


def semantic_contract(
    config: TrainingConfig,
    train_sha: str,
    validation_sha: str,
    *,
    model_source: dict[str, Any],
    runtime: dict[str, Any],
) -> dict[str, Any]:
    # Runtime limits, checkpoint cadence and output paths may change across resumable segments.
    return {
        "schema_version": 1,
        "model_source": model_source,
        "runtime": runtime,
        "train_sha256": train_sha,
        "validation_sha256": validation_sha,
        "tokenization_contract": config.tokenization_contract,
        "max_length": config.max_length,
        "batch_size": config.batch_size,
        "learning_rate_head": config.learning_rate_head,
        "learning_rate_encoder": config.learning_rate_encoder,
        "dropout": config.dropout,
        "epochs": config.epochs,
        "patience": config.patience,
        "unfreeze_layers": config.unfreeze_layers,
        "pooling": config.pooling,
        "attention_implementation": config.attention_implementation,
        "validation_every": config.validation_every,
        "seed": config.seed,
        "deterministic_algorithms": config.deterministic_algorithms,
        "gradient_checkpointing": config.gradient_checkpointing,
    }


def rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if state["torch_cuda"] is not None:
        if not torch.cuda.is_available():
            raise RuntimeError("Checkpoint contains CUDA RNG state but CUDA is unavailable")
        torch.cuda.set_rng_state_all(state["torch_cuda"])


class TerminationSignal:
    """Turn SIGTERM into a stop request handled at the next batch boundary."""

    def __init__(self) -> None:
        self.requested: int | None = None
        self.previous = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGTERM, self._request)

    def _request(self, signum: int, _frame: Any) -> None:
        self.requested = signum

    def restore(self) -> None:
        signal.signal(signal.SIGTERM, self.previous)


def latest_checkpoint(output_dir: Path) -> Path | None:
    pointer = output_dir / "latest-checkpoint.json"
    if pointer.is_file():
        metadata = json.loads(pointer.read_text(encoding="utf-8"))
        checkpoint = output_dir / metadata["path"]
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Latest checkpoint pointer names a missing file: {checkpoint}")
        actual_sha = sha256_file(checkpoint)
        if actual_sha != metadata["sha256"]:
            raise ValueError(
                "Latest checkpoint checksum mismatch: "
                f"recorded={metadata['sha256']} actual={actual_sha}"
            )
        return checkpoint
    checkpoints = sorted(output_dir.glob("checkpoint-step-*.pt"))
    return checkpoints[-1] if checkpoints else None


def prune_checkpoints(output_dir: Path, keep: int) -> None:
    checkpoints = sorted(output_dir.glob("checkpoint-step-*.pt"))
    for path in checkpoints[: max(0, len(checkpoints) - keep)]:
        path.unlink()


def save_export(
    directory: Path,
    model: RewardRegressor,
    tokenizer: Any,
    metadata: dict[str, Any],
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    model.encoder.save_pretrained(directory, safe_serialization=True)
    tokenizer.save_pretrained(directory)
    atomic_torch_save(
        directory / "head_weights.pt",
        {key.removeprefix("head."): value for key, value in state_dict_to_cpu(model).items() if key.startswith("head.")},
    )
    atomic_json(directory / "reward_model_metadata.json", metadata)


@torch.no_grad()
def evaluate(
    model: RewardRegressor,
    tokenizer: Any,
    records: Sequence[RewardRecord],
    *,
    batch_size: int,
    contract: str,
    max_length: int,
    device: torch.device,
) -> dict[str, float | int | None]:
    was_training = model.training
    model.eval()
    squared_error_sum = 0.0
    labels_all: list[float] = []
    predictions_all: list[float] = []
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        tokenized = tokenize_records(tokenizer, batch, contract=contract, max_length=max_length)
        labels = tokenized["labels"].to(device)
        predictions = model(
            tokenized["input_ids"].to(device), tokenized["attention_mask"].to(device)
        )
        if not torch.isfinite(predictions).all():
            raise FloatingPointError("Non-finite validation prediction")
        squared_error_sum += F.mse_loss(predictions, labels, reduction="sum").item()
        labels_all.extend(labels.detach().cpu().tolist())
        predictions_all.extend(predictions.detach().cpu().tolist())
    correlation_value = float(spearmanr(labels_all, predictions_all).statistic)
    correlation = correlation_value if math.isfinite(correlation_value) else None
    metrics: dict[str, float | int | None] = {
        "mse": squared_error_sum / len(records),
        "spearman": correlation,
        "rows": len(records),
    }
    if not math.isfinite(float(metrics["mse"])):
        raise FloatingPointError("Non-finite validation MSE")
    # Spearman is undefined for a constant label or prediction vector. Preserve that fact.
    if was_training:
        model.train()
    return metrics


def build_optimizer(model: RewardRegressor, config: TrainingConfig) -> torch.optim.Optimizer:
    head_parameters = list(model.head.parameters())
    encoder_parameters = [p for p in model.encoder.parameters() if p.requires_grad]
    groups: list[dict[str, Any]] = [
        {"params": head_parameters, "lr": config.learning_rate_head, "name": "head"}
    ]
    if encoder_parameters:
        groups.append(
            {
                "params": encoder_parameters,
                "lr": config.learning_rate_encoder,
                "name": "encoder",
            }
        )
    return torch.optim.AdamW(groups)


def _run_training(config: TrainingConfig, termination: TerminationSignal) -> dict[str, Any]:
    if config.tokenization_contract not in TOKENIZATION_CONTRACTS:
        raise ValueError(f"Unsupported tokenization contract: {config.tokenization_contract}")
    if config.batch_size <= 0 or config.max_length <= 0 or config.epochs <= 0:
        raise ValueError("batch_size, max_length and epochs must be positive")
    if config.validation_every <= 0 or config.checkpoint_every <= 0:
        raise ValueError("validation_every and checkpoint_every must be positive")
    if config.keep_checkpoints <= 0 or config.logging_every <= 0:
        raise ValueError("keep_checkpoints and logging_every must be positive")
    if config.resume_from and config.resume_latest:
        raise ValueError("Choose only one of resume_from and resume_latest")

    output_dir = Path(config.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = Path(config.train_file).resolve()
    validation_path = Path(config.validation_file).resolve()
    train_sha = sha256_file(train_path)
    validation_sha = sha256_file(validation_path)
    set_seed(config.seed, config.deterministic_algorithms)
    device = choose_device(config.device)
    source_identity = model_identity(config.model, config.model_revision)
    runtime = runtime_identity(device)
    contract = semantic_contract(
        config,
        train_sha,
        validation_sha,
        model_source=source_identity,
        runtime=runtime,
    )
    contract_sha = hashlib.sha256(canonical_json(contract).encode("utf-8")).hexdigest()
    tokenizer = AutoTokenizer.from_pretrained(
        config.model,
        revision=config.model_revision,
        local_files_only=config.local_files_only,
    )
    model = RewardRegressor(
        config.model,
        model_revision=config.model_revision,
        dropout=config.dropout,
        unfreeze_layers=config.unfreeze_layers,
        pooling=config.pooling,
        attention_implementation=config.attention_implementation,
        local_files_only=config.local_files_only,
        gradient_checkpointing=config.gradient_checkpointing,
    ).to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    optimizer = build_optimizer(model, config)
    train_records = load_records(train_path)
    validation_records = load_records(validation_path)
    batches_per_epoch = math.ceil(len(train_records) / config.batch_size)

    initial_digest = tensor_digest(state_dict_to_cpu(model))
    epoch = 0
    next_batch_index = 0
    global_step = 0
    best_mse = float("inf")
    best_step: int | None = None
    patience_counter = 0
    stopped_early = False

    checkpoint_path: Path | None = None
    if config.resume_from:
        checkpoint_path = Path(config.resume_from).resolve()
    elif config.resume_latest:
        checkpoint_path = latest_checkpoint(output_dir)
        if checkpoint_path is None:
            raise FileNotFoundError(f"No checkpoint found under {output_dir}")
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint.get("contract_sha256") != contract_sha:
            raise ValueError(
                "Resume contract mismatch: "
                f"checkpoint={checkpoint.get('contract_sha256')} current={contract_sha}; "
                f"checkpoint_contract={canonical_json(checkpoint.get('semantic_contract'))}; "
                f"current_contract={canonical_json(contract)}"
            )
        if checkpoint.get("continuable") is not True:
            raise ValueError("Checkpoint records a terminal run and cannot be resumed")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        epoch = int(checkpoint["epoch"])
        next_batch_index = int(checkpoint["next_batch_index"])
        global_step = int(checkpoint["global_step"])
        best_mse = float(checkpoint["best_mse"])
        best_step = checkpoint["best_step"]
        patience_counter = int(checkpoint["patience_counter"])
        restore_rng_state(checkpoint["rng_state"])

    manifest = {
        "schema_version": 1,
        "config": asdict(config),
        "semantic_contract": contract,
        "contract_sha256": contract_sha,
        "device": str(device),
        "model_argument": config.model,
        "model_revision_argument": config.model_revision,
        "train_rows": len(train_records),
        "validation_rows": len(validation_records),
        "batches_per_epoch": batches_per_epoch,
        "initial_parameter_sha256": initial_digest,
        "resumed_from": str(checkpoint_path) if checkpoint_path else None,
        "deviations_from_reference": [
            "local frozen JSONL inputs replace runtime Hub loading and augmentation",
            "balanced_pair is the default; reference_tail preserves the original right-truncating contract",
            "validation MSE is row-weighted rather than an unweighted mean of batch means",
            "explicit seeded epoch permutations replace DataLoader shuffle",
            "eager attention plus deterministic kernels make same-runtime resume testable",
            "atomic full-state checkpoints and runtime limits make training resumable",
            "current and best exports are retained separately",
            "activation checkpointing is explicit, opt-in and part of the resume contract",
        ],
    }
    atomic_json(output_dir / "run-manifest.json", manifest)
    events_path = output_dir / "events.jsonl"
    append_jsonl(
        events_path,
        {
            "event": "resume" if checkpoint_path else "start",
            "global_step": global_step,
            "epoch": epoch,
            "next_batch_index": next_batch_index,
            "time_unix": time.time(),
        },
    )

    started = time.monotonic()
    invocation_updates = 0
    stop_reason: str | None = None
    latest_validation: dict[str, Any] | None = None

    def checkpoint_state(*, continuable: bool) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "contract_sha256": contract_sha,
            "semantic_contract": contract,
            "model": state_dict_to_cpu(model),
            "optimizer": optimizer_state_to_cpu(optimizer.state_dict()),
            "epoch": epoch,
            "next_batch_index": next_batch_index,
            "global_step": global_step,
            "best_mse": best_mse,
            "best_step": best_step,
            "patience_counter": patience_counter,
            "rng_state": rng_state(),
            "initial_parameter_sha256": initial_digest,
            "continuable": continuable,
        }

    def save_checkpoint(*, continuable: bool = True) -> Path:
        path = output_dir / f"checkpoint-step-{global_step:08d}.pt"
        atomic_torch_save(path, checkpoint_state(continuable=continuable))
        prune_checkpoints(output_dir, config.keep_checkpoints)
        atomic_json(
            output_dir / "latest-checkpoint.json",
            {"path": path.name, "global_step": global_step, "sha256": sha256_file(path)},
        )
        return path

    def validate(*, changes_training_state: bool, reason: str) -> dict[str, Any]:
        nonlocal best_mse, best_step, patience_counter, stopped_early
        validation_started = time.monotonic()
        metrics = evaluate(
            model,
            tokenizer,
            validation_records,
            batch_size=config.batch_size,
            contract=config.tokenization_contract,
            max_length=config.max_length,
            device=device,
        )
        metrics["seconds"] = time.monotonic() - validation_started
        record: dict[str, Any] = {
            "event": "validation",
            "global_step": global_step,
            "epoch": epoch,
            "next_batch_index": next_batch_index,
            "reason": reason,
            "changes_training_state": changes_training_state,
            **metrics,
        }
        if changes_training_state:
            improved = float(metrics["mse"]) < best_mse
            record["improved"] = improved
            if improved:
                best_mse = float(metrics["mse"])
                best_step = global_step
                patience_counter = 0
                save_export(
                    output_dir / "best",
                    model,
                    tokenizer,
                    {
                        "kind": "best",
                        "global_step": global_step,
                        "validation": metrics,
                        "contract_sha256": contract_sha,
                        "tokenization_contract": config.tokenization_contract,
                        "max_length": config.max_length,
                        "pooling": config.pooling,
                        "dropout": config.dropout,
                    },
                )
            else:
                patience_counter += 1
            stopped_early = patience_counter >= config.patience
            record["patience_counter"] = patience_counter
        append_jsonl(events_path, record)
        return record

    model.train()
    while epoch < config.epochs and not stopped_early:
        epoch_batches = batches_for_epoch(len(train_records), config.batch_size, config.seed, epoch)
        while next_batch_index < len(epoch_batches):
            if termination.requested is not None:
                stop_reason = f"signal_{termination.requested}"
                break
            if config.max_updates is not None and global_step >= config.max_updates:
                stop_reason = "max_updates"
                break
            if config.stop_after_updates is not None and invocation_updates >= config.stop_after_updates:
                stop_reason = "stop_after_updates"
                break
            if config.max_runtime_seconds is not None and time.monotonic() - started >= config.max_runtime_seconds:
                stop_reason = "max_runtime_seconds"
                break

            batch_indices = epoch_batches[next_batch_index]
            batch = [train_records[index] for index in batch_indices]
            if config.measure_step_time and device.type == "cuda":
                torch.cuda.synchronize(device)
            update_started = time.monotonic()
            tokenized = tokenize_records(
                tokenizer,
                batch,
                contract=config.tokenization_contract,
                max_length=config.max_length,
            )
            optimizer.zero_grad(set_to_none=True)
            labels = tokenized["labels"].to(device)
            predictions = model(
                tokenized["input_ids"].to(device), tokenized["attention_mask"].to(device)
            )
            loss = F.mse_loss(predictions, labels)
            if not torch.isfinite(loss):
                save_checkpoint()
                raise FloatingPointError(f"Non-finite training loss at update {global_step + 1}")
            loss.backward()
            for name, parameter in model.named_parameters():
                if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                    save_checkpoint()
                    raise FloatingPointError(
                        f"Non-finite gradient for {name} at update {global_step + 1}"
                    )
            optimizer.step()
            if config.measure_step_time and device.type == "cuda":
                torch.cuda.synchronize(device)
            update_seconds = time.monotonic() - update_started if config.measure_step_time else None

            global_step += 1
            invocation_updates += 1
            next_batch_index += 1
            if next_batch_index == len(epoch_batches):
                epoch += 1
                next_batch_index = 0
            append_jsonl(
                events_path,
                {
                    "event": "train_update",
                    "global_step": global_step,
                    "epoch": epoch,
                    "next_batch_index": next_batch_index,
                    "loss": float(loss.detach().cpu()),
                    "seconds": update_seconds,
                },
            )
            if global_step == 1 or global_step % config.logging_every == 0:
                print(
                    canonical_json(
                        {
                            "event": "train_update",
                            "global_step": global_step,
                            "epoch": epoch,
                            "next_batch_index": next_batch_index,
                            "loss": float(loss.detach().cpu()),
                            "seconds": update_seconds,
                        }
                    ),
                    flush=True,
                )

            if global_step % config.validation_every == 0:
                latest_validation = validate(changes_training_state=True, reason="scheduled")
            if global_step % config.checkpoint_every == 0:
                save_checkpoint()
            model.train()
            if stopped_early:
                stop_reason = "early_stopping"
                break
        if stop_reason is not None:
            break

    completed_training = epoch >= config.epochs
    terminal_stop = completed_training or stop_reason in {"max_updates", "early_stopping"}
    if terminal_stop and (latest_validation is None or latest_validation["global_step"] != global_step):
        latest_validation = validate(changes_training_state=True, reason="terminal")
    elif not terminal_stop:
        # Preserve resumable state before a potentially long diagnostic validation.
        save_checkpoint()
        # Useful evidence for a bounded pilot without changing the state resumed next time.
        latest_validation = validate(changes_training_state=False, reason="operational_pause")

    checkpoint = save_checkpoint(continuable=not terminal_stop)
    final_digest = tensor_digest(state_dict_to_cpu(model))
    memory = None
    if device.type == "cuda":
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        memory = {
            "allocated_bytes": torch.cuda.memory_allocated(device),
            "reserved_bytes": torch.cuda.memory_reserved(device),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
            "device_free_bytes": free_bytes,
            "device_total_bytes": total_bytes,
        }
    final_metadata = {
        "kind": "current",
        "global_step": global_step,
        "epoch": epoch,
        "next_batch_index": next_batch_index,
        "contract_sha256": contract_sha,
        "cuda_memory": memory,
        "parameter_sha256": final_digest,
        "tokenization_contract": config.tokenization_contract,
        "max_length": config.max_length,
        "pooling": config.pooling,
        "dropout": config.dropout,
    }
    save_export(output_dir / "current", model, tokenizer, final_metadata)
    result = {
        "schema_version": 1,
        "status": "complete" if terminal_stop else "paused",
        "stop_reason": stop_reason or ("epochs_complete" if completed_training else "unknown"),
        "global_step": global_step,
        "epoch": epoch,
        "next_batch_index": next_batch_index,
        "invocation_updates": invocation_updates,
        "best_mse": best_mse if math.isfinite(best_mse) else None,
        "best_step": best_step,
        "patience_counter": patience_counter,
        "latest_validation": latest_validation,
        "checkpoint": checkpoint.name,
        "checkpoint_sha256": sha256_file(checkpoint),
        "initial_parameter_sha256": initial_digest,
        "final_parameter_sha256": final_digest,
        "parameters_changed": initial_digest != final_digest,
        "elapsed_seconds": time.monotonic() - started,
        "contract_sha256": contract_sha,
    }
    atomic_json(output_dir / "result.json", result)
    return result


def run_training(config: TrainingConfig) -> dict[str, Any]:
    termination = TerminationSignal()
    try:
        return _run_training(config, termination)
    finally:
        termination.restore()


def parse_args(argv: Sequence[str] | None = None) -> TrainingConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--validation-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default=REFERENCE_MODEL)
    parser.add_argument("--model-revision")
    parser.add_argument("--tokenization-contract", choices=TOKENIZATION_CONTRACTS, default="balanced_pair")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate-head", type=float, default=5e-4)
    parser.add_argument("--learning-rate-encoder", type=float, default=5e-5)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--unfreeze-layers", type=int, default=3)
    parser.add_argument("--pooling", choices=("mean", "max", "meanmax"), default="meanmax")
    parser.add_argument("--attention-implementation", default="eager")
    parser.add_argument("--validation-every", type=int, default=250)
    parser.add_argument("--checkpoint-every", type=int, default=250)
    parser.add_argument("--keep-checkpoints", type=int, default=2)
    parser.add_argument("--logging-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--deterministic-algorithms", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--resume-from")
    parser.add_argument("--resume-latest", action="store_true")
    parser.add_argument("--max-updates", type=int)
    parser.add_argument("--stop-after-updates", type=int)
    parser.add_argument("--max-runtime-seconds", type=float)
    parser.add_argument("--measure-step-time", action="store_true")
    return TrainingConfig(**vars(parser.parse_args(argv)))


def main(argv: Sequence[str] | None = None) -> int:
    result = run_training(parse_args(argv))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
