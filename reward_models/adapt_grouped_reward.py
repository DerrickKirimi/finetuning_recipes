"""Adapt an exported reward model to grouped reference-judge scores.

The reference reward-training entry point remains unchanged.  This module is an
opt-in distillation path for data where every prompt group contains several candidate
responses and only within-group reward differences drive the downstream optimizer.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import re
import time
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from reward_models.reward_training import (
    RewardRegressor,
    atomic_json,
    canonical_json,
    save_export,
    set_seed,
    sha256_directory,
    sha256_file,
    tensor_digest,
)


@dataclass(frozen=True)
class GroupedRewardRecord:
    group_id: str
    rollout_index: int
    fold: int
    reference: str
    response: str
    score: float
    scoreable: bool


@dataclass(frozen=True)
class Candidate:
    name: str
    epochs: int
    learning_rate_head: float
    learning_rate_encoder: float
    advantage_beta: float
    raw_score_weight: float


@dataclass(frozen=True)
class AdaptationConfig:
    data_file: str
    parent_model_dir: str
    candidates_file: str
    output_dir: str
    folds: int = 5
    group_size: int = 8
    groups_per_batch: int = 4
    max_length: int = 512
    dropout: float = 0.2
    pooling: str = "meanmax"
    attention_implementation: str = "eager"
    unfreeze_layers: int = 6
    seed: int = 3417
    device: str = "auto"
    deterministic_algorithms: bool = True
    gradient_checkpointing: bool = False
    max_gradient_norm: float = 1.0
    development_mae_gate: float = 0.10
    minimum_active_fraction: float = 0.50


def load_grouped_records(
    path: Path, *, folds: int, group_size: int
) -> list[list[GroupedRewardRecord]]:
    grouped: dict[str, list[GroupedRewardRecord]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            raw = json.loads(line)
            try:
                row = GroupedRewardRecord(
                    group_id=str(raw["group_id"]),
                    rollout_index=int(raw["rollout_index"]),
                    fold=int(raw["fold"]),
                    reference=raw["reference"],
                    response=raw["response"],
                    score=float(raw["score"]),
                    scoreable=raw["scoreable"],
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid grouped row at {path}:{line_number}: {exc}") from exc
            if not row.group_id:
                raise ValueError(f"Empty group_id at {path}:{line_number}")
            if not isinstance(row.reference, str) or not isinstance(row.response, str):
                raise ValueError(f"Non-string text at {path}:{line_number}")
            if not isinstance(row.scoreable, bool):
                raise ValueError(f"scoreable must be a bool at {path}:{line_number}")
            if not math.isfinite(row.score) or not 0.0 <= row.score <= 1.0:
                raise ValueError(f"Score outside finite [0, 1] at {path}:{line_number}")
            if not 0 <= row.fold < folds:
                raise ValueError(f"Fold outside [0, {folds}) at {path}:{line_number}")
            grouped[row.group_id].append(row)
    if not grouped:
        raise ValueError(f"No grouped records found in {path}")

    result = []
    fold_counts = [0] * folds
    for group_id in sorted(grouped):
        rows = sorted(grouped[group_id], key=lambda row: row.rollout_index)
        if len(rows) != group_size:
            raise ValueError(f"Group {group_id} has {len(rows)} rows, expected {group_size}")
        if [row.rollout_index for row in rows] != list(range(group_size)):
            raise ValueError(f"Group {group_id} rollout indices are not 0..{group_size - 1}")
        if len({row.fold for row in rows}) != 1:
            raise ValueError(f"Group {group_id} spans multiple folds")
        if len({row.reference for row in rows}) != 1:
            raise ValueError(f"Group {group_id} has multiple references")
        if not any(row.scoreable for row in rows):
            raise ValueError(f"Group {group_id} has no scoreable response")
        fold_counts[rows[0].fold] += 1
        result.append(rows)
    if any(count == 0 for count in fold_counts):
        raise ValueError(f"Every fold must contain a group; counts={fold_counts}")
    return result


def load_candidates(path: Path) -> list[Candidate]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not raw:
        raise ValueError("Candidates file must contain a nonempty JSON list")
    candidates = []
    for index, item in enumerate(raw):
        try:
            candidate = Candidate(
                name=item["name"],
                epochs=int(item["epochs"]),
                learning_rate_head=float(item["learning_rate_head"]),
                learning_rate_encoder=float(item["learning_rate_encoder"]),
                advantage_beta=float(item["advantage_beta"]),
                raw_score_weight=float(item["raw_score_weight"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid candidate {index}: {exc}") from exc
        if not isinstance(candidate.name, str) or not re.fullmatch(
            r"[a-z0-9][a-z0-9._-]*", candidate.name
        ):
            raise ValueError(f"Candidate {index} has an invalid name")
        if candidate.epochs <= 0:
            raise ValueError(f"Candidate {candidate.name} epochs must be positive")
        if candidate.learning_rate_head <= 0 or candidate.learning_rate_encoder <= 0:
            raise ValueError(f"Candidate {candidate.name} learning rates must be positive")
        if candidate.advantage_beta <= 0:
            raise ValueError(f"Candidate {candidate.name} advantage_beta must be positive")
        if candidate.raw_score_weight < 0:
            raise ValueError(f"Candidate {candidate.name} raw_score_weight must be nonnegative")
        candidates.append(candidate)
    names = [candidate.name for candidate in candidates]
    if len(set(names)) != len(names):
        raise ValueError("Candidate names must be unique")
    return candidates


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_parent_model(config: AdaptationConfig, device: torch.device) -> tuple[Any, Any]:
    parent = Path(config.parent_model_dir)
    required = ("config.json", "head_weights.pt", "reward_model_metadata.json")
    missing = [name for name in required if not (parent / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Parent reward export is incomplete: {missing}")
    metadata = json.loads((parent / "reward_model_metadata.json").read_text(encoding="utf-8"))
    expected = {
        "pooling": config.pooling,
        "max_length": config.max_length,
        "tokenization_contract": "balanced_pair",
    }
    actual = {key: metadata.get(key) for key in expected}
    if actual != expected:
        raise ValueError(f"Parent reward contract mismatch: expected={expected} actual={actual}")
    tokenizer = AutoTokenizer.from_pretrained(parent, local_files_only=True)
    model = RewardRegressor(
        str(parent),
        model_revision=None,
        dropout=config.dropout,
        unfreeze_layers=config.unfreeze_layers,
        pooling=config.pooling,
        attention_implementation=config.attention_implementation,
        local_files_only=True,
        gradient_checkpointing=config.gradient_checkpointing,
    )
    head_state = torch.load(parent / "head_weights.pt", map_location="cpu", weights_only=True)
    model.head.load_state_dict(head_state)
    return model.to(device), tokenizer


def group_batches(
    groups: Sequence[list[GroupedRewardRecord]], groups_per_batch: int, seed: int, epoch: int
) -> list[list[list[GroupedRewardRecord]]]:
    order = list(range(len(groups)))
    random.Random(seed + epoch).shuffle(order)
    return [
        [groups[index] for index in order[start : start + groups_per_batch]]
        for start in range(0, len(order), groups_per_batch)
    ]


def score_group_batch(
    model: Any,
    tokenizer: Any,
    groups: Sequence[list[GroupedRewardRecord]],
    *,
    max_length: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows = [row for group in groups for row in group]
    valid_indices = [index for index, row in enumerate(rows) if row.scoreable]
    predictions = torch.zeros(len(rows), dtype=torch.float32, device=device)
    if valid_indices:
        encoded = tokenizer(
            [rows[index].reference for index in valid_indices],
            [rows[index].response for index in valid_indices],
            padding=True,
            truncation="longest_first",
            max_length=max_length,
            return_tensors="pt",
        )
        scored = model(
            encoded["input_ids"].to(device), encoded["attention_mask"].to(device)
        ).float()
        index_tensor = torch.tensor(valid_indices, dtype=torch.long, device=device)
        predictions = predictions.index_copy(0, index_tensor, scored)
    targets = torch.tensor([row.score for row in rows], dtype=torch.float32, device=device)
    group_size = len(groups[0])
    return predictions.reshape(len(groups), group_size), targets.reshape(len(groups), group_size)


def grouped_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    *,
    advantage_beta: float,
    raw_score_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    predicted_advantage = predictions - predictions.mean(dim=1, keepdim=True)
    target_advantage = targets - targets.mean(dim=1, keepdim=True)
    advantage = F.smooth_l1_loss(
        predicted_advantage, target_advantage, beta=advantage_beta
    )
    raw = F.mse_loss(predictions, targets)
    total = advantage + raw_score_weight * raw
    return total, {
        "loss": float(total.detach().cpu()),
        "advantage_loss": float(advantage.detach().cpu()),
        "raw_score_loss": float(raw.detach().cpu()),
    }


def build_optimizer(model: Any, candidate: Candidate) -> torch.optim.Optimizer:
    groups = [
        {
            "params": list(model.head.parameters()),
            "lr": candidate.learning_rate_head,
            "name": "head",
        }
    ]
    encoder = [parameter for parameter in model.encoder.parameters() if parameter.requires_grad]
    if encoder:
        groups.append(
            {
                "params": encoder,
                "lr": candidate.learning_rate_encoder,
                "name": "encoder",
            }
        )
    return torch.optim.AdamW(groups)


def train_model(
    model: Any,
    tokenizer: Any,
    groups: Sequence[list[GroupedRewardRecord]],
    candidate: Candidate,
    config: AdaptationConfig,
    device: torch.device,
    *,
    run_seed: int,
) -> list[dict[str, float | int]]:
    optimizer = build_optimizer(model, candidate)
    history = []
    for epoch in range(candidate.epochs):
        epoch_started = time.monotonic()
        model.train()
        totals = defaultdict(float)
        batches = group_batches(groups, config.groups_per_batch, run_seed, epoch)
        for batch in batches:
            optimizer.zero_grad(set_to_none=True)
            predictions, targets = score_group_batch(
                model, tokenizer, batch, max_length=config.max_length, device=device
            )
            loss, metrics = grouped_loss(
                predictions,
                targets,
                advantage_beta=candidate.advantage_beta,
                raw_score_weight=candidate.raw_score_weight,
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite grouped adaptation loss")
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.max_gradient_norm
            )
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError("Non-finite grouped adaptation gradient norm")
            optimizer.step()
            for name, value in metrics.items():
                totals[name] += value
        history.append(
            {
                "epoch": epoch + 1,
                "batches": len(batches),
                "seconds": time.monotonic() - epoch_started,
                **{name: value / len(batches) for name, value in totals.items()},
            }
        )
    return history


@torch.no_grad()
def evaluate_groups(
    model: Any,
    tokenizer: Any,
    groups: Sequence[list[GroupedRewardRecord]],
    config: AdaptationConfig,
    device: torch.device,
) -> tuple[dict[str, float | int], list[dict[str, Any]]]:
    model.eval()
    output = []
    group_errors = []
    pairwise = []
    active = []
    for start in range(0, len(groups), config.groups_per_batch):
        batch = groups[start : start + config.groups_per_batch]
        predictions, targets = score_group_batch(
            model, tokenizer, batch, max_length=config.max_length, device=device
        )
        predictions_np = predictions.cpu().numpy()
        targets_np = targets.cpu().numpy()
        for group, predicted, target in zip(batch, predictions_np, targets_np):
            predicted_advantage = predicted - predicted.mean()
            target_advantage = target - target.mean()
            group_errors.append(float(np.abs(predicted_advantage - target_advantage).mean()))
            active.append(float(np.ptp(predicted) > 1e-6))
            correct = 0.0
            compared = 0
            for left in range(len(group)):
                for right in range(left + 1, len(group)):
                    target_sign = np.sign(target[left] - target[right])
                    if target_sign == 0:
                        continue
                    prediction_sign = np.sign(predicted[left] - predicted[right])
                    compared += 1
                    correct += float(prediction_sign == target_sign)
                    correct += 0.5 * float(prediction_sign == 0)
            pairwise.append(correct / compared if compared else 0.5)
            for row, prediction in zip(group, predicted):
                output.append(
                    {
                        "group_id": row.group_id,
                        "rollout_index": row.rollout_index,
                        "fold": row.fold,
                        "prediction": float(prediction),
                        "score": row.score,
                        "scoreable": row.scoreable,
                    }
                )
    return (
        {
            "groups": len(groups),
            "rows": len(output),
            "advantage_mae": float(np.mean(group_errors)),
            "pairwise_concordance": float(np.mean(pairwise)),
            "active_fraction": float(np.mean(active)),
        },
        output,
    )


def validate_fold_result(
    result: dict[str, Any],
    *,
    path: Path,
    contract_sha: str,
    candidate: Candidate,
    fold: int,
    expected_groups: Sequence[list[GroupedRewardRecord]],
) -> None:
    if result.get("contract_sha256") != contract_sha:
        raise ValueError(f"Cached fold result has a different contract: {path}")
    if result.get("candidate") != asdict(candidate) or result.get("fold") != fold:
        raise ValueError(f"Cached fold result has the wrong candidate or fold: {path}")
    expected = {
        (row.group_id, row.rollout_index, row.fold)
        for group in expected_groups
        for row in group
    }
    predictions = result.get("predictions")
    if not isinstance(predictions, list):
        raise ValueError(f"Cached fold result has no prediction list: {path}")
    actual = {
        (row.get("group_id"), row.get("rollout_index"), row.get("fold"))
        for row in predictions
    }
    if len(predictions) != len(expected) or actual != expected:
        raise ValueError(f"Cached fold result prediction identities are incomplete: {path}")
    validation = result.get("validation", {})
    if validation.get("groups") != len(expected_groups) or validation.get("rows") != len(expected):
        raise ValueError(f"Cached fold result counts are inconsistent: {path}")
    for name in ("advantage_mae", "pairwise_concordance", "active_fraction"):
        value = validation.get(name)
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"Cached fold result has invalid {name}: {path}")
    if any(
        not isinstance(row.get("prediction"), (int, float))
        or not math.isfinite(row["prediction"])
        for row in predictions
    ):
        raise ValueError(f"Cached fold result has a non-finite prediction: {path}")


def run_adaptation(config: AdaptationConfig) -> dict[str, Any]:
    if config.folds < 2:
        raise ValueError("folds must be at least 2")
    if config.group_size < 2:
        raise ValueError("group_size must be at least 2")
    if config.groups_per_batch <= 0:
        raise ValueError("groups_per_batch must be positive")
    if config.max_length <= 0:
        raise ValueError("max_length must be positive")
    if config.max_gradient_norm <= 0:
        raise ValueError("max_gradient_norm must be positive")
    if config.development_mae_gate < 0:
        raise ValueError("development_mae_gate must be nonnegative")
    if not 0.0 <= config.minimum_active_fraction <= 1.0:
        raise ValueError("minimum_active_fraction must be in [0, 1]")
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = Path(config.data_file)
    parent_path = Path(config.parent_model_dir)
    candidates_path = Path(config.candidates_file)
    groups = load_grouped_records(data_path, folds=config.folds, group_size=config.group_size)
    candidates = load_candidates(candidates_path)
    device = choose_device(config.device)
    set_seed(config.seed, config.deterministic_algorithms)

    semantic_config = asdict(config)
    for location in ("data_file", "parent_model_dir", "candidates_file", "output_dir", "device"):
        semantic_config.pop(location)
    identity = {
        "schema_version": 1,
        "semantic_config": semantic_config,
        "data_sha256": sha256_file(data_path),
        "candidates_sha256": sha256_file(candidates_path),
        "parent_sha256": sha256_directory(parent_path),
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "device_type": device.type,
    }
    if device.type == "cuda":
        identity["cuda_device_name"] = torch.cuda.get_device_name(device)
        identity["cuda_capability"] = list(torch.cuda.get_device_capability(device))
    contract_sha = hashlib.sha256(canonical_json(identity).encode("utf-8")).hexdigest()
    atomic_json(output_dir / "adaptation-contract.json", {**identity, "sha256": contract_sha})

    result_path = output_dir / "result.json"
    if result_path.is_file():
        completed = json.loads(result_path.read_text(encoding="utf-8"))
        if completed.get("contract_sha256") != contract_sha:
            raise ValueError("Existing terminal result has a different adaptation contract")
        selected_model = completed.get("selected_model")
        if selected_model is not None:
            export_dir = output_dir / selected_model["path"]
            if sha256_directory(export_dir) != selected_model["sha256"]:
                raise ValueError("Existing selected-model export checksum mismatch")
        return completed

    set_seed(config.seed, config.deterministic_algorithms)
    parent_started = time.monotonic()
    parent_model, parent_tokenizer = load_parent_model(config, device)
    parent_metrics, _ = evaluate_groups(
        parent_model, parent_tokenizer, groups, config, device
    )
    parent_metrics["seconds"] = time.monotonic() - parent_started
    del parent_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    candidate_results = []
    for candidate_index, candidate in enumerate(candidates):
        fold_results = []
        for fold in range(config.folds):
            path = output_dir / "folds" / candidate.name / f"fold-{fold}.json"
            validation_groups = [group for group in groups if group[0].fold == fold]
            if path.is_file():
                cached = json.loads(path.read_text(encoding="utf-8"))
                validate_fold_result(
                    cached,
                    path=path,
                    contract_sha=contract_sha,
                    candidate=candidate,
                    fold=fold,
                    expected_groups=validation_groups,
                )
                fold_results.append(cached)
                continue
            run_seed = config.seed + candidate_index * 100 + fold
            fold_started = time.monotonic()
            set_seed(run_seed, config.deterministic_algorithms)
            model, tokenizer = load_parent_model(config, device)
            train_groups = [group for group in groups if group[0].fold != fold]
            history = train_model(
                model,
                tokenizer,
                train_groups,
                candidate,
                config,
                device,
                run_seed=run_seed,
            )
            metrics, predictions = evaluate_groups(
                model, tokenizer, validation_groups, config, device
            )
            fold_result = {
                "schema_version": 1,
                "contract_sha256": contract_sha,
                "candidate": asdict(candidate),
                "fold": fold,
                "seed": run_seed,
                "train_groups": len(train_groups),
                "validation": metrics,
                "history": history,
                "predictions": predictions,
                "seconds": time.monotonic() - fold_started,
                "final_parameter_sha256": tensor_digest(model.state_dict()),
            }
            atomic_json(path, fold_result)
            fold_results.append(fold_result)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        weighted_mae = sum(
            result["validation"]["advantage_mae"] * result["validation"]["groups"]
            for result in fold_results
        ) / sum(result["validation"]["groups"] for result in fold_results)
        weighted_pairwise = sum(
            result["validation"]["pairwise_concordance"] * result["validation"]["groups"]
            for result in fold_results
        ) / sum(result["validation"]["groups"] for result in fold_results)
        weighted_active = sum(
            result["validation"]["active_fraction"] * result["validation"]["groups"]
            for result in fold_results
        ) / sum(result["validation"]["groups"] for result in fold_results)
        candidate_results.append(
            {
                "candidate": asdict(candidate),
                "advantage_mae": weighted_mae,
                "pairwise_concordance": weighted_pairwise,
                "active_fraction": weighted_active,
                "folds": [result["validation"] for result in fold_results],
            }
        )

    selected = min(
        candidate_results,
        key=lambda result: (result["advantage_mae"], result["candidate"]["name"]),
    )
    selected_candidate = next(
        candidate for candidate in candidates if candidate.name == selected["candidate"]["name"]
    )
    development_pass = (
        selected["advantage_mae"] <= config.development_mae_gate
        and selected["active_fraction"] >= config.minimum_active_fraction
    )
    result = {
        "schema_version": 1,
        "status": "development_pass" if development_pass else "development_fail",
        "contract_sha256": contract_sha,
        "selection_rule": (
            "minimum group-weighted out-of-fold advantage MAE; exact ties choose name"
        ),
        "development_mae_gate": config.development_mae_gate,
        "minimum_active_fraction": config.minimum_active_fraction,
        "development_pass": development_pass,
        "parent_development": parent_metrics,
        "selected_candidate": selected_candidate.name,
        "candidates": candidate_results,
        "warning": (
            "Development results do not qualify the reward model; a frozen new holdout is required."
        ),
    }
    if development_pass:
        final_seed = config.seed + 10_000
        final_started = time.monotonic()
        set_seed(final_seed, config.deterministic_algorithms)
        model, tokenizer = load_parent_model(config, device)
        final_history = train_model(
            model,
            tokenizer,
            groups,
            selected_candidate,
            config,
            device,
            run_seed=final_seed,
        )
        export_dir = output_dir / "selected-model"
        metadata = {
            "schema_version": 1,
            "kind": "grouped_judge_adaptation",
            "contract_sha256": contract_sha,
            "parent_sha256": identity["parent_sha256"],
            "data_sha256": identity["data_sha256"],
            "candidate": asdict(selected_candidate),
            "development_advantage_mae": selected["advantage_mae"],
            "development_pairwise_concordance": selected["pairwise_concordance"],
            "tokenization_contract": "balanced_pair",
            "pooling": config.pooling,
            "max_length": config.max_length,
            "unfreeze_layers": config.unfreeze_layers,
            "seed": final_seed,
            "history": final_history,
            "seconds": time.monotonic() - final_started,
        }
        save_export(export_dir, model, tokenizer, metadata)
        result["selected_model"] = {
            "path": export_dir.name,
            "sha256": sha256_directory(export_dir),
            "parameter_sha256": tensor_digest(model.state_dict()),
        }
    atomic_json(result_path, result)
    return result


def parse_args() -> AdaptationConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-file", required=True)
    parser.add_argument("--parent-model-dir", required=True)
    parser.add_argument("--candidates-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--groups-per-batch", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--pooling", choices=("mean", "max", "meanmax"), default="meanmax")
    parser.add_argument("--attention-implementation", default="eager")
    parser.add_argument("--unfreeze-layers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=3417)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-deterministic-algorithms", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--max-gradient-norm", type=float, default=1.0)
    parser.add_argument("--development-mae-gate", type=float, default=0.10)
    parser.add_argument("--minimum-active-fraction", type=float, default=0.50)
    args = parser.parse_args()
    return AdaptationConfig(
        data_file=args.data_file,
        parent_model_dir=args.parent_model_dir,
        candidates_file=args.candidates_file,
        output_dir=args.output_dir,
        folds=args.folds,
        group_size=args.group_size,
        groups_per_batch=args.groups_per_batch,
        max_length=args.max_length,
        dropout=args.dropout,
        pooling=args.pooling,
        attention_implementation=args.attention_implementation,
        unfreeze_layers=args.unfreeze_layers,
        seed=args.seed,
        device=args.device,
        deterministic_algorithms=not args.no_deterministic_algorithms,
        gradient_checkpointing=args.gradient_checkpointing,
        max_gradient_norm=args.max_gradient_norm,
        development_mae_gate=args.development_mae_gate,
        minimum_active_fraction=args.minimum_active_fraction,
    )


if __name__ == "__main__":
    print(json.dumps(run_adaptation(parse_args()), indent=2, sort_keys=True))
