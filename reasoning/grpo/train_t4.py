"""Restartable single-T4 port of the repository's reference GRPO trainer.

This keeps the reference rollout, reward, PPO-clipped policy objective and
full-distribution KL objective.  It changes systems behavior explicitly:
float16 + SDPA, one visible CUDA device, a required local learned-reward
archive, prompt-length validation, and checkpoints only at empty-buffer train
event boundaries.  A bounded execution/resume gate must pass before a long
run is staged.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import json
import os
import random
import shutil
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from peft import LoraConfig, get_peft_model
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from reasoning.env import score_completions
from reward_models.reward_model import load_reward_model

from .grpo_utils import (
    calculate_entropy,
    calculate_grpo_loss,
    calculate_kld_loss,
    generate_responses,
)
from .paper_dataset import PaperInstructionDataset, collate_fn
from .repro_runtime import (
    BoundaryState,
    capture_rng_state,
    contract_sha256,
    load_boundary_checkpoint,
    restore_rng_state,
    save_boundary_checkpoint,
    seed_everything,
)
from .rollout import calculate_log_probs, collect_rollouts


SCHEMA_VERSION = 1


def _json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _required(mapping: dict, key: str, section: str):
    if key not in mapping:
        raise ValueError(f"missing {section}.{key}")
    return mapping[key]


def build_contract(config: dict) -> dict:
    """Return the immutable fields that must match across resume processes."""
    model = config["model"]
    training = config["training"]
    data = config["data"]
    loss = config["loss"]
    reward = config["reward"]
    runtime = config["runtime"]
    contract = {
        "schema_version": SCHEMA_VERSION,
        "model": {
            "name": _required(model, "name", "model"),
            "revision": _required(model, "revision", "model"),
            "max_new_tokens": int(_required(model, "max_new_tokens", "model")),
            "min_new_tokens": int(model.get("min_new_tokens", 0)),
            "max_prompt_tokens": int(_required(model, "max_prompt_tokens", "model")),
            "torch_dtype": _required(model, "torch_dtype", "model"),
            "attn_implementation": _required(model, "attn_implementation", "model"),
            "lora": model.get(
                "lora",
                {
                    "r": 32,
                    "alpha": 64,
                    "targets": [
                        "q_proj", "v_proj", "k_proj", "o_proj",
                        "up_proj", "down_proj", "gate_proj",
                    ],
                },
            ),
        },
        "reward": {
            "path": _required(reward, "path", "reward"),
            "archive_sha256": _required(reward, "archive_sha256", "reward"),
        },
        "data": {
            "train_path": _required(data, "train_path", "data"),
            "train_sha256": _required(data, "train_sha256", "data"),
            "dataset_seed": data.get("dataset_seed"),
            "train_data_size": data.get("train_data_size"),
            "num_epochs": int(_required(data, "num_epochs", "data")),
        },
        "training": {
            key: training[key]
            for key in (
                "rollout_batch_size", "n_rollouts", "temperature", "top_p",
                "batch_size", "gradient_accumulation_steps", "learning_rate",
                "num_repeats", "buffer_size",
            )
        },
        "loss": {
            key: loss[key]
            for key in (
                "kld_weight", "entropy_weight", "loss_implementation", "max_tokens",
            )
        },
        "seed": int(_required(runtime, "seed", "runtime")),
    }
    evaluation = config.get("evaluation")
    if evaluation is not None:
        contract["evaluation"] = {
            "path": _required(evaluation, "path", "evaluation"),
            "sha256": _required(evaluation, "sha256", "evaluation"),
            "every_train_events": int(_required(evaluation, "every_train_events", "evaluation")),
            "batch_size": int(_required(evaluation, "batch_size", "evaluation")),
            "max_new_tokens": int(_required(evaluation, "max_new_tokens", "evaluation")),
            "temperature": float(_required(evaluation, "temperature", "evaluation")),
            "top_p": float(_required(evaluation, "top_p", "evaluation")),
            "seed": int(_required(evaluation, "seed", "evaluation")),
        }
    return contract


def validate_contract(contract: dict) -> None:
    model = contract["model"]
    training = contract["training"]
    if model["torch_dtype"] != "float16":
        raise ValueError("the T4 port requires model.torch_dtype=float16")
    if model["attn_implementation"] != "sdpa":
        raise ValueError("the T4 port requires model.attn_implementation=sdpa")
    if not 0 <= int(model["min_new_tokens"]) <= int(model["max_new_tokens"]):
        raise ValueError("min_new_tokens must be between zero and max_new_tokens")
    if int(training["n_rollouts"]) < 2:
        raise ValueError("GRPO requires at least two rollouts per prompt")
    experiences_per_rollout = int(training["rollout_batch_size"]) * int(training["n_rollouts"])
    if int(training["buffer_size"]) % experiences_per_rollout:
        raise ValueError("buffer_size must be divisible by rollout_batch_size * n_rollouts")
    microbatches = (
        int(training["buffer_size"]) // int(training["batch_size"])
    ) * int(training["num_repeats"])
    if int(training["buffer_size"]) % int(training["batch_size"]):
        raise ValueError("buffer_size must be divisible by batch_size")
    if microbatches % int(training["gradient_accumulation_steps"]):
        raise ValueError("one train event must close its gradient-accumulation window")
    evaluation = contract.get("evaluation")
    if evaluation is not None:
        if int(evaluation["every_train_events"]) < 1:
            raise ValueError("evaluation.every_train_events must be positive")
        if int(evaluation["batch_size"]) < 1:
            raise ValueError("evaluation.batch_size must be positive")
        if not 0 < int(evaluation["max_new_tokens"]) <= int(model["max_new_tokens"]):
            raise ValueError("evaluation.max_new_tokens must fit the training generation contract")
        if not 0 < float(evaluation["temperature"]):
            raise ValueError("evaluation.temperature must be positive for sampled source evaluation")
        if not 0 < float(evaluation["top_p"]) <= 1:
            raise ValueError("evaluation.top_p must be in (0, 1]")


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _directory_hashes(path: str | Path) -> dict[str, str]:
    """Hash every regular file below *path* using stable relative names."""
    root = Path(path)
    if not root.is_dir():
        raise FileNotFoundError(f"artifact directory does not exist: {root}")
    return {
        file.relative_to(root).as_posix(): _file_sha256(file)
        for file in sorted(root.rglob("*"))
        if file.is_file()
    }


def _selection_artifact_path(output_dir: Path, train_event: int) -> Path:
    return output_dir / f"model-best-event-{train_event:06d}"


def _write_selection_artifacts(
    output_dir: Path, state: BoundaryState, contract: dict[str, Any]
) -> None:
    """Bind the selected model and its resumable checkpoint to the run state."""
    if state.best_train_event is None and state.best_eval_reward is None:
        return
    if state.best_train_event is None or state.best_eval_reward is None:
        raise RuntimeError("checkpoint state has an incomplete selection record")
    event = int(state.best_train_event)
    checkpoint = output_dir / "checkpoints" / f"event-{event:06d}"
    model = _selection_artifact_path(output_dir, event)
    checkpoint_state = json.loads(
        (checkpoint / "state.json").read_text(encoding="utf-8")
    )
    selection = json.loads((model / "selection.json").read_text(encoding="utf-8"))
    expected_contract = contract_sha256(contract)
    if checkpoint_state.get("contract_sha256") != expected_contract:
        raise RuntimeError("best checkpoint has a different run contract")
    recorded_state = checkpoint_state.get("state", {})
    if int(recorded_state.get("train_events", -1)) != event:
        raise RuntimeError("best checkpoint event differs from selection state")
    if int(selection.get("train_event", -1)) != event:
        raise RuntimeError("best model event differs from selection state")
    if float(selection.get("mean_total_reward")) != float(state.best_eval_reward):
        raise RuntimeError("best model reward differs from selection state")
    _json_dump(
        output_dir / "selection-artifacts.json",
        {
            "schema_version": 1,
            "contract_sha256": expected_contract,
            "best_train_event": event,
            "best_eval_reward": float(state.best_eval_reward),
            "checkpoint_path": checkpoint.relative_to(output_dir).as_posix(),
            "checkpoint_files": _directory_hashes(checkpoint),
            "model_path": model.relative_to(output_dir).as_posix(),
            "model_files": _directory_hashes(model),
        },
    )


def _verify_selection_artifacts(
    output_dir: Path, state: BoundaryState, contract: dict[str, Any]
) -> None:
    """Fail closed when a resumed segment did not carry its prior best bytes."""
    if state.best_train_event is None and state.best_eval_reward is None:
        return
    if state.best_train_event is None or state.best_eval_reward is None:
        raise RuntimeError("resume state has an incomplete checkpoint-selection record")
    manifest_path = output_dir / "selection-artifacts.json"
    if not manifest_path.is_file():
        raise RuntimeError("resume is missing selection-artifacts.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    event = int(state.best_train_event)
    expected = {
        "contract_sha256": contract_sha256(contract),
        "best_train_event": event,
        "best_eval_reward": float(state.best_eval_reward),
        "checkpoint_path": f"checkpoints/event-{event:06d}",
        "model_path": f"model-best-event-{event:06d}",
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise RuntimeError(f"resume selection artifact mismatch: {key}")
    checkpoint = output_dir / manifest["checkpoint_path"]
    model = output_dir / manifest["model_path"]
    if _directory_hashes(checkpoint) != manifest.get("checkpoint_files"):
        raise RuntimeError("resumed best checkpoint files differ from their manifest")
    if _directory_hashes(model) != manifest.get("model_files"):
        raise RuntimeError("resumed best model files differ from their manifest")
    checkpoint_state = json.loads(
        (checkpoint / "state.json").read_text(encoding="utf-8")
    )
    selection = json.loads((model / "selection.json").read_text(encoding="utf-8"))
    if int(checkpoint_state.get("state", {}).get("train_events", -1)) != event:
        raise RuntimeError("resumed best checkpoint contains the wrong event")
    if int(selection.get("train_event", -1)) != event:
        raise RuntimeError("resumed best model contains the wrong event")
    if float(selection.get("mean_total_reward")) != float(state.best_eval_reward):
        raise RuntimeError("resumed best model contains the wrong reward")


def _rng_states_equal(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return (
        left["python"] == right["python"]
        and left["numpy"][0] == right["numpy"][0]
        and np.array_equal(left["numpy"][1], right["numpy"][1])
        and left["numpy"][2:] == right["numpy"][2:]
        and torch.equal(left["torch"], right["torch"])
        and len(left["cuda"]) == len(right["cuda"])
        and all(torch.equal(a, b) for a, b in zip(left["cuda"], right["cuda"]))
        and left["shuffle"] == right["shuffle"]
    )


@contextlib.contextmanager
def _isolated_rng(seed: int, shuffle_rng: random.Random):
    """Give evaluation a fixed seed, then exactly restore training RNG streams."""
    before = capture_rng_state(shuffle_rng)
    seed_everything(seed)
    try:
        yield
    finally:
        restore_rng_state(before, shuffle_rng)
        after = capture_rng_state(shuffle_rng)
        if not _rng_states_equal(before, after):
            raise RuntimeError("evaluation changed training RNG state")


def _close_unterminated_think(response: str) -> str:
    if "<think>" in response and "</think>" not in response:
        return response + "</think>"
    return response


def _tensor_sha256(model) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().contiguous().cpu()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _trainable_sha256(model) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.named_parameters()):
        if value.requires_grad:
            tensor = value.detach().contiguous().cpu()
            digest.update(name.encode("utf-8"))
            digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _frozen_policy_sha256(model) -> str:
    """Hash frozen policy tensors to detect lossy LoRA merge/unmerge drift."""
    digest = hashlib.sha256()
    for name, value in sorted(model.named_parameters()):
        if not value.requires_grad:
            tensor = value.detach().contiguous().cpu()
            digest.update(name.encode("utf-8"))
            digest.update(tensor.view(torch.uint8).numpy().tobytes())
    for name, value in sorted(model.named_buffers()):
        tensor = value.detach().contiguous().cpu()
        digest.update(name.encode("utf-8"))
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _cuda_memory() -> dict[str, int | None]:
    if not torch.cuda.is_available():
        return {"allocated": None, "reserved": None, "peak_allocated": None, "peak_reserved": None, "free": None, "total": None}
    free, total = torch.cuda.mem_get_info()
    return {
        "allocated": int(torch.cuda.memory_allocated()),
        "reserved": int(torch.cuda.memory_reserved()),
        "peak_allocated": int(torch.cuda.max_memory_allocated()),
        "peak_reserved": int(torch.cuda.max_memory_reserved()),
        "free": int(free),
        "total": int(total),
    }


@contextlib.contextmanager
def _autocast(device: torch.device):
    if device.type == "cuda":
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            yield
    else:
        yield


class T4GRPOTrainer:
    def __init__(self, config: dict, output_dir: Path, resume: Path | None = None, allow_cpu: bool = False):
        self.config = config
        self.contract = build_contract(config)
        validate_contract(self.contract)
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = output_dir / "events.jsonl"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type != "cuda" and not allow_cpu:
            raise RuntimeError("CUDA is required; --allow-cpu is only for bounded tests")
        if self.device.type == "cuda" and torch.cuda.device_count() != 1:
            raise RuntimeError("the T4 port requires exactly one visible CUDA device")

        runtime = config["runtime"]
        self.shuffle_rng = seed_everything(int(runtime["seed"]))
        self.started = time.monotonic()
        self.max_train_events = int(runtime["max_train_events"])
        self.max_wall_seconds = float(runtime["max_wall_seconds"])
        self.reference_guard_every = int(runtime.get("reference_guard_every", 1))
        self.policy_base_guard_every = int(runtime.get("policy_base_guard_every", 1))
        self.checkpoint_every_train_events = int(runtime.get("checkpoint_every_train_events", 1))
        if self.checkpoint_every_train_events < 1:
            raise ValueError("runtime.checkpoint_every_train_events must be positive")

        model_config = self.contract["model"]
        source = model_config["name"]
        revision = model_config["revision"]
        load_kwargs = {
            "revision": revision,
            "torch_dtype": torch.float16 if self.device.type == "cuda" else torch.float32,
            "attn_implementation": model_config["attn_implementation"],
        }
        base = AutoModelForCausalLM.from_pretrained(source, **load_kwargs)
        if int(base.config.max_position_embeddings) < (
            int(model_config["max_prompt_tokens"]) + int(model_config["max_new_tokens"])
        ):
            raise ValueError("max_prompt_tokens + max_new_tokens exceeds model context")
        lora = model_config["lora"]
        self.model = get_peft_model(
            base,
            LoraConfig(
                task_type="CAUSAL_LM",
                r=int(lora["r"]),
                lora_alpha=int(lora["alpha"]),
                target_modules=list(lora["targets"]),
            ),
        ).to(self.device)
        self.reference = None
        if float(self.contract["loss"]["kld_weight"]) > 0:
            self.reference = AutoModelForCausalLM.from_pretrained(source, **load_kwargs).to(self.device)
            self.reference.eval()
            self.reference.requires_grad_(False)
            self.reference_sha256 = _tensor_sha256(self.reference)
        else:
            self.reference_sha256 = None

        self.tokenizer = AutoTokenizer.from_pretrained(source, revision=revision)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.reward_model, _ = load_reward_model(self.contract["reward"]["path"])

        training = self.contract["training"]
        self.optimizer = torch.optim.Adam(
            (parameter for parameter in self.model.parameters() if parameter.requires_grad),
            lr=float(training["learning_rate"]),
        )
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.device.type == "cuda")
        self.state = BoundaryState()
        self.resume_path = resume
        if resume is not None:
            self.state = load_boundary_checkpoint(
                resume,
                contract=self.contract,
                model=self.model,
                optimizer=self.optimizer,
                shuffle_rng=self.shuffle_rng,
                scaler=self.scaler,
            )
            _verify_selection_artifacts(self.output_dir, self.state, self.contract)

        data = self.contract["data"]
        self.dataset = PaperInstructionDataset(
            data["train_path"],
            split="train",
            tokenizer=self.tokenizer,
            data_size=data["train_data_size"],
            seed=data["dataset_seed"],
        )
        self.loader = DataLoader(
            self.dataset,
            batch_size=int(training["rollout_batch_size"]),
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            collate_fn=lambda rows: collate_fn(rows, self.tokenizer.pad_token_id),
        )
        self.eval_loader = None
        evaluation = self.contract.get("evaluation")
        if evaluation is not None:
            eval_path = Path(evaluation["path"])
            if not eval_path.is_file():
                raise FileNotFoundError(f"evaluation split does not exist: {eval_path}")
            if _file_sha256(eval_path) != evaluation["sha256"]:
                raise ValueError("evaluation split hash differs from the run contract")
            eval_dataset = PaperInstructionDataset(
                eval_path,
                split="train",
                tokenizer=self.tokenizer,
                data_size=None,
                seed=None,
            )
            self.eval_loader = DataLoader(
                eval_dataset,
                batch_size=int(evaluation["batch_size"]),
                shuffle=False,
                num_workers=0,
                pin_memory=False,
                collate_fn=lambda rows: collate_fn(rows, self.tokenizer.pad_token_id),
            )
        self.buffer = []
        self.pending_microbatches = 0
        self.initial_policy_sha256 = _trainable_sha256(self.model)
        self.frozen_policy_sha256 = _frozen_policy_sha256(self.model)
        if self.resume_path is not None:
            _append_jsonl(
                self.events_path,
                {
                    "event": "resume_loaded",
                    "checkpoint": str(self.resume_path),
                    "state": asdict(self.state),
                    "policy_sha256": self.initial_policy_sha256,
                    "reference_sha256": self.reference_sha256,
                    "frozen_policy_sha256": self.frozen_policy_sha256,
                },
            )
        _json_dump(
            self.output_dir / "run-contract.json",
            {"contract": self.contract, "contract_sha256": contract_sha256(self.contract)},
        )

    def evaluate(self) -> dict[str, Any]:
        """Evaluate on the frozen split without advancing any training RNG stream."""
        evaluation = self.contract.get("evaluation")
        if evaluation is None or self.eval_loader is None:
            raise RuntimeError("evaluation is not configured")
        event = self.state.train_events
        rows = []
        totals = []
        policy_before = _trainable_sha256(self.model)
        frozen_before = _frozen_policy_sha256(self.model)
        self.model.eval()
        with _isolated_rng(int(evaluation["seed"]), self.shuffle_rng):
            # Keep the adapter unmerged. In fp16, merge/unmerge can round frozen
            # base weights; evaluation is allowed to cost more, but it may not
            # mutate the policy that the next training event will continue.
            with torch.inference_mode():
                for batch in self.eval_loader:
                    self._validate_prompt_batch(batch)
                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    outputs = generate_responses(
                        self.model,
                        {"input_ids": input_ids, "attention_mask": attention_mask},
                        max_new_tokens=int(evaluation["max_new_tokens"]),
                        n_rollouts=1,
                        top_p=float(evaluation["top_p"]),
                        temperature=float(evaluation["temperature"]),
                        do_sample=True,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )
                    input_length = input_ids.shape[1]
                    completions = outputs[:, input_length:]
                    texts = [
                        _close_unterminated_think(value)
                        for value in self.tokenizer.batch_decode(
                            completions, skip_special_tokens=True
                        )
                    ]
                    references = list(batch["answer"])
                    rewards = score_completions(
                        texts,
                        references,
                        reward_model=self.reward_model,
                    )
                    token_counts = (
                        (completions != self.tokenizer.eos_token_id)
                        .to(torch.int32)
                        .sum(axis=-1)
                        .cpu()
                        .tolist()
                    )
                    for index, (text, reference, source) in enumerate(
                        zip(texts, references, batch["source"])
                    ):
                        total = float(rewards.total[index])
                        totals.append(total)
                        rows.append(
                            {
                                "eval_id": source.get("eval_id"),
                                "source_index": source.get("_source_index"),
                                "prompt_sha256": hashlib.sha256(
                                    json.dumps(
                                        batch["item"][index]["prompt"], sort_keys=True
                                    ).encode("utf-8")
                                ).hexdigest(),
                                "completion": text,
                                "reference": reference,
                                "completion_tokens": int(token_counts[index]),
                                "total_reward": total,
                                "components": {
                                    name: float(values[index])
                                    for name, values in rewards.components.items()
                                },
                            }
                        )
                    del outputs, completions, input_ids, attention_mask
        if _trainable_sha256(self.model) != policy_before:
            raise RuntimeError("evaluation changed trainable policy tensors")
        if _frozen_policy_sha256(self.model) != frozen_before:
            raise RuntimeError("evaluation changed frozen policy tensors")
        if not totals or not all(np.isfinite(value) for value in totals):
            raise FloatingPointError("evaluation produced no finite total rewards")

        destination = self.output_dir / f"eval-event-{event:06d}.jsonl"
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        result = {
            "event": "evaluation",
            "train_event": event,
            "rows": len(rows),
            "mean_total_reward": float(np.mean(totals)),
            "std_total_reward": float(np.std(totals, ddof=1)) if len(totals) > 1 else 0.0,
            "minimum_total_reward": float(np.min(totals)),
            "maximum_total_reward": float(np.max(totals)),
            "rows_path": str(destination),
            "rows_sha256": _file_sha256(destination),
            "policy_sha256": policy_before,
            "frozen_policy_sha256": frozen_before,
        }
        _append_jsonl(self.events_path, result)
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        return result

    def _validate_prompt_batch(self, batch: dict) -> None:
        lengths = batch["attention_mask"].sum(dim=1).tolist()
        maximum = int(self.contract["model"]["max_prompt_tokens"])
        if any(int(length) > maximum for length in lengths):
            raise ValueError(f"prompt length exceeds frozen cap {maximum}: {lengths}")

    def collect(self, batch: dict) -> None:
        self._validate_prompt_batch(batch)
        batch["input_ids"] = batch["input_ids"].to(self.device)
        batch["attention_mask"] = batch["attention_mask"].to(self.device)
        self.model.eval()
        with torch.inference_mode():
            rollouts = collect_rollouts(
                self.model,
                self.tokenizer,
                batch,
                max_new_tokens=int(self.contract["model"]["max_new_tokens"]),
                n_rollouts=int(self.contract["training"]["n_rollouts"]),
                top_p=float(self.contract["training"]["top_p"]),
                temperature=float(self.contract["training"]["temperature"]),
                logprob_chunk_size=int(self.contract["training"]["batch_size"]),
                min_new_tokens=int(self.contract["model"]["min_new_tokens"]),
            )
        rewards = score_completions(
            rollouts.completion_texts,
            rollouts.references,
            reward_model=self.reward_model,
        )
        grouped = rewards.total.reshape(rollouts.num_prompts, rollouts.group_size)
        advantages = grouped - grouped.mean(axis=1, keepdims=True)
        advantages = torch.tensor(advantages.reshape(-1, 1), dtype=torch.float32)
        self.buffer.extend(rollouts.to_experiences(advantages))
        self.state.rollout_batches += 1
        self.state.experiences += len(advantages)
        if (
            self.policy_base_guard_every > 0
            and self.state.rollout_batches % self.policy_base_guard_every == 0
        ):
            observed_base = _frozen_policy_sha256(self.model)
            if observed_base != self.frozen_policy_sha256:
                raise RuntimeError("frozen policy tensors changed during rollout merge/unmerge")
        prompt_hashes = [
            hashlib.sha256(
                json.dumps(item["prompt"], sort_keys=True).encode("utf-8")
            ).hexdigest()
            for item in batch["item"]
        ]
        _append_jsonl(
            self.events_path,
            {
                "event": "rollout_batch",
                "epoch": self.state.epoch,
                "batch": self.state.next_batch,
                "prompt_sha256": prompt_hashes,
                "prompt_tokens": [int(value) for value in batch["attention_mask"].sum(dim=1).tolist()],
                "completion_tokens": {
                    "minimum": int(rollouts.token_counts.min()),
                    "maximum": int(rollouts.token_counts.max()),
                    "mean": float(rollouts.token_counts.mean()),
                },
                "reward": {
                    "mean": float(grouped.mean()),
                    "std": float(grouped.std()),
                    "minimum": float(grouped.min()),
                    "maximum": float(grouped.max()),
                },
            },
        )
        del rollouts
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def _loss(self, rows) -> tuple[torch.Tensor, dict[str, float]]:
        pad_id = self.tokenizer.pad_token_id
        input_ids = pad_sequence([row[0] for row in rows], batch_first=True, padding_side="left", padding_value=pad_id).to(self.device)
        attention = pad_sequence([torch.ones_like(row[0]) for row in rows], batch_first=True, padding_side="left", padding_value=0).to(self.device)
        response_mask = pad_sequence([row[1] for row in rows], batch_first=True, padding_side="left", padding_value=0).to(self.device)
        old_log_probs = pad_sequence([row[2] for row in rows], batch_first=True, padding_side="left", padding_value=0).to(self.device)
        advantages = torch.cat([row[3] for row in rows], dim=0).unsqueeze(-1).to(self.device)
        loss_config = self.contract["loss"]
        with _autocast(self.device):
            current, full = calculate_log_probs(self.model, input_ids, attention)
            policy = calculate_grpo_loss(
                current,
                old_log_probs,
                advantages,
                response_mask,
                loss_implementation=loss_config["loss_implementation"],
                max_tokens=int(loss_config["max_tokens"]),
            )
            total = policy
            kl = torch.zeros((), device=self.device)
            if self.reference is not None:
                with torch.no_grad():
                    _, reference_full = calculate_log_probs(self.reference, input_ids, attention)
                kl = calculate_kld_loss(
                    full,
                    reference_full,
                    response_mask,
                    loss_implementation=loss_config["loss_implementation"],
                    max_tokens=int(loss_config["max_tokens"]),
                )
                total = total + float(loss_config["kld_weight"]) * kl
            entropy = torch.zeros((), device=self.device)
            if float(loss_config["entropy_weight"]) > 0:
                entropy = calculate_entropy(
                    full,
                    response_mask,
                    loss_implementation=loss_config["loss_implementation"],
                    max_tokens=int(loss_config["max_tokens"]),
                )
                total = total - float(loss_config["entropy_weight"]) * entropy
        if not torch.isfinite(total):
            raise FloatingPointError(f"non-finite GRPO loss: {total.item()}")
        return total, {"policy_loss": float(policy.detach()), "kl_loss": float(kl.detach()), "entropy": float(entropy.detach()), "total_loss": float(total.detach())}

    def train_event(self) -> dict[str, Any]:
        training = self.contract["training"]
        if len(self.buffer) != int(training["buffer_size"]):
            raise RuntimeError("train event requires exactly buffer_size experiences")
        self.shuffle_rng.shuffle(self.buffer)
        rows = self.buffer
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        accumulation = int(training["gradient_accumulation_steps"])
        batch_size = int(training["batch_size"])
        losses = []
        for _ in range(int(training["num_repeats"])):
            for start in range(0, len(rows), batch_size):
                loss, metrics = self._loss(rows[start : start + batch_size])
                self.pending_microbatches += 1
                self.scaler.scale(loss / accumulation).backward()
                losses.append(metrics)
                if self.pending_microbatches == accumulation:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.pending_microbatches = 0
                    self.state.optimizer_updates += 1
        if self.pending_microbatches:
            raise RuntimeError("train event left an open accumulation window")
        self.state.train_events += 1
        self.buffer = []
        self.model.eval()
        if self.reference is not None and self.state.train_events % self.reference_guard_every == 0:
            observed = _tensor_sha256(self.reference)
            if observed != self.reference_sha256:
                raise RuntimeError("frozen reference model changed")
        result = {
            "event": "train_event",
            "state": asdict(self.state),
            "loss": {
                name: float(np.mean([row[name] for row in losses]))
                for name in losses[0]
            },
            "policy_sha256": _trainable_sha256(self.model),
            "reference_sha256": self.reference_sha256,
            "frozen_policy_sha256": self.frozen_policy_sha256,
            "cuda_memory": _cuda_memory(),
            "elapsed_seconds": time.monotonic() - self.started,
        }
        _append_jsonl(self.events_path, result)
        return result

    def checkpoint(self) -> Path:
        checkpoint = save_boundary_checkpoint(
            self.output_dir / "checkpoints",
            contract=self.contract,
            state=self.state,
            model=self.model,
            optimizer=self.optimizer,
            shuffle_rng=self.shuffle_rng,
            scaler=self.scaler,
            buffer_size=len(self.buffer),
            pending_microbatches=self.pending_microbatches,
        )
        _json_dump(
            self.output_dir / "checkpoints" / "latest.json",
            {
                "train_event": self.state.train_events,
                "path": checkpoint.name,
                "manifest_sha256": _file_sha256(checkpoint / "manifest.json"),
            },
        )
        keep = {checkpoint.name}
        if self.state.best_train_event is not None:
            keep.add(f"event-{self.state.best_train_event:06d}")
        for previous in (self.output_dir / "checkpoints").glob("event-*"):
            if previous.is_dir() and previous.name not in keep:
                shutil.rmtree(previous)
        _write_selection_artifacts(self.output_dir, self.state, self.contract)
        if self.state.best_train_event is not None:
            selected = _selection_artifact_path(
                self.output_dir, self.state.best_train_event
            )
            for previous in self.output_dir.glob("model-best-event-*"):
                if previous.is_dir() and previous != selected:
                    shutil.rmtree(previous)
        return checkpoint

    def ensure_current_checkpoint(self) -> Path:
        latest = self.output_dir / "checkpoints" / "latest.json"
        if latest.is_file():
            record = json.loads(latest.read_text(encoding="utf-8"))
            candidate = latest.parent / record["path"]
            if (
                int(record["train_event"]) == self.state.train_events
                and candidate.is_dir()
                and _file_sha256(candidate / "manifest.json")
                == record["manifest_sha256"]
            ):
                return candidate
        return self.checkpoint()

    def _save_best_model(self, evaluation: dict[str, Any]) -> None:
        event = self.state.train_events
        destination = _selection_artifact_path(self.output_dir, event)
        if destination.exists():
            raise FileExistsError(f"best-model artifact already exists: {destination}")
        temporary = self.output_dir / f".{destination.name}.tmp"
        if temporary.exists():
            shutil.rmtree(temporary)
        self.model.save_pretrained(temporary)
        self.tokenizer.save_pretrained(temporary)
        _json_dump(
            temporary / "selection.json",
            {
                "train_event": self.state.train_events,
                "mean_total_reward": evaluation["mean_total_reward"],
                "eval_rows_sha256": evaluation["rows_sha256"],
                "policy_sha256": evaluation["policy_sha256"],
            },
        )
        os.replace(temporary, destination)

    def run(self) -> dict[str, Any]:
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        stop_reason = "epochs_complete"
        for epoch in range(self.state.epoch, int(self.contract["data"]["num_epochs"])):
            self.state.epoch = epoch
            start_batch = self.state.next_batch
            for batch_index, batch in enumerate(self.loader):
                if batch_index < start_batch:
                    continue
                self.state.next_batch = batch_index
                self.collect(batch)
                self.state.next_batch = batch_index + 1
                if len(self.buffer) == int(self.contract["training"]["buffer_size"]):
                    self.train_event()
                    evaluation = None
                    improved = False
                    evaluation_contract = self.contract.get("evaluation")
                    if (
                        evaluation_contract is not None
                        and self.state.train_events
                        % int(evaluation_contract["every_train_events"])
                        == 0
                    ):
                        evaluation = self.evaluate()
                        observed = float(evaluation["mean_total_reward"])
                        improved = (
                            self.state.best_eval_reward is None
                            or observed > self.state.best_eval_reward
                        )
                        if improved:
                            self.state.best_eval_reward = observed
                            self.state.best_train_event = self.state.train_events
                            self._save_best_model(evaluation)
                        _append_jsonl(
                            self.events_path,
                            {
                                "event": "checkpoint_selection",
                                "train_event": self.state.train_events,
                                "improved": improved,
                                "mean_total_reward": observed,
                                "best_eval_reward": self.state.best_eval_reward,
                                "best_train_event": self.state.best_train_event,
                            },
                        )
                    reached_event_cap = self.state.train_events >= self.max_train_events
                    reached_time_cap = time.monotonic() - self.started >= self.max_wall_seconds
                    if (
                        improved
                        or reached_event_cap
                        or reached_time_cap
                        or self.state.train_events % self.checkpoint_every_train_events == 0
                    ):
                        self.checkpoint()
                    if reached_event_cap:
                        stop_reason = "max_train_events"
                        return self.finish(stop_reason)
                    if reached_time_cap:
                        stop_reason = "runtime_boundary"
                        return self.finish(stop_reason)
                elif len(self.buffer) > int(self.contract["training"]["buffer_size"]):
                    raise RuntimeError("rollout buffer overshot buffer_size")
            self.state.epoch = epoch + 1
            self.state.next_batch = 0
        if self.buffer:
            _append_jsonl(self.events_path, {"event": "dropped_partial_buffer", "count": len(self.buffer)})
            self.buffer = []
        self.ensure_current_checkpoint()
        return self.finish(stop_reason)

    def finish(self, stop_reason: str) -> dict[str, Any]:
        policy_sha = _trainable_sha256(self.model)
        if self.reference is not None and _tensor_sha256(self.reference) != self.reference_sha256:
            raise RuntimeError("frozen reference model changed at terminal boundary")
        if _frozen_policy_sha256(self.model) != self.frozen_policy_sha256:
            raise RuntimeError("frozen policy tensors changed at terminal boundary")
        export_dir = self.output_dir / "model-current"
        self.model.save_pretrained(export_dir)
        self.tokenizer.save_pretrained(export_dir)
        selection_manifest = self.output_dir / "selection-artifacts.json"
        best_model_export = None
        if selection_manifest.is_file():
            selected = json.loads(selection_manifest.read_text(encoding="utf-8"))
            best_model_export = str(self.output_dir / selected["model_path"])
        result = {
            "schema_version": SCHEMA_VERSION,
            "status": "complete" if stop_reason == "epochs_complete" else "paused",
            "stop_reason": stop_reason,
            "contract_sha256": contract_sha256(self.contract),
            "state": asdict(self.state),
            "policy_changed": policy_sha != self.initial_policy_sha256,
            "initial_policy_sha256": self.initial_policy_sha256,
            "final_policy_sha256": policy_sha,
            "reference_sha256": self.reference_sha256,
            "frozen_policy_sha256": self.frozen_policy_sha256,
            "model_export": str(export_dir),
            "best_model_export": best_model_export,
            "cuda_memory": _cuda_memory(),
            "elapsed_seconds": time.monotonic() - self.started,
        }
        _json_dump(self.output_dir / "result.json", result)
        return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    trainer = T4GRPOTrainer(
        config,
        Path(args.output_dir),
        resume=Path(args.resume) if args.resume else None,
        allow_cpu=args.allow_cpu,
    )
    print(json.dumps(trainer.run(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
