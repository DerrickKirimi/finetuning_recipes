"""Restartable grouped rollout generation for reward diagnostics and GRPO gates."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random
import time
from typing import Any


SCHEMA_VERSION = 1


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    staging = path.with_suffix(path.suffix + ".partial")
    with staging.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    staging.replace(path)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def batch_plan(rows: list[dict], batch_size: int) -> list[list[dict]]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    required = {"group_id", "split", "source_index", "instruction", "input", "reference"}
    if any(not required.issubset(row) for row in rows):
        raise ValueError(f"every prompt row must contain {sorted(required)}")
    group_ids = [row["group_id"] for row in rows]
    if len(group_ids) != len(set(group_ids)):
        raise ValueError("group_id values must be unique")
    return [rows[start : start + batch_size] for start in range(0, len(rows), batch_size)]


def batch_path(output_dir: Path, index: int) -> Path:
    return output_dir / "batches" / f"batch-{index:05d}.jsonl"


def validate_batch(rows: list[dict], prompts: list[dict], rollouts_per_group: int, batch_index: int) -> None:
    expected_ids = {row["group_id"] for row in prompts}
    counts = Counter(row.get("group_id") for row in rows)
    if set(counts) != expected_ids or any(count != rollouts_per_group for count in counts.values()):
        raise ValueError(f"batch {batch_index} has incorrect group counts: {dict(counts)}")
    for prompt in prompts:
        group = [row for row in rows if row["group_id"] == prompt["group_id"]]
        if sorted(row.get("rollout_index") for row in group) != list(range(rollouts_per_group)):
            raise ValueError(f"batch {batch_index} has bad rollout indices for {prompt['group_id']}")
        if any(
            row.get("source_index") != prompt["source_index"]
            or row.get("split") != prompt["split"]
            or row.get("reference") != prompt["reference"]
            for row in group
        ):
            raise ValueError(f"batch {batch_index} changed frozen prompt identity")


def completed_batches(output_dir: Path, plan: list[list[dict]], rollouts_per_group: int) -> int:
    completed = 0
    for index, prompts in enumerate(plan):
        path = batch_path(output_dir, index)
        if not path.exists():
            break
        validate_batch(read_jsonl(path), prompts, rollouts_per_group, index)
        completed += 1
    later = [path for path in (output_dir / "batches").glob("batch-*.jsonl") if int(path.stem.split("-")[-1]) >= completed]
    if later:
        raise ValueError(f"non-contiguous or unexpected batch files: {[path.name for path in later]}")
    return completed


def write_batch(path: Path, rows: list[dict]) -> None:
    staging = path.with_suffix(path.suffix + ".partial")
    with staging.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    staging.replace(path)


def flatten_batches(output_dir: Path, count: int, destination: Path) -> None:
    staging = destination.with_suffix(destination.suffix + ".partial")
    with staging.open("wb") as output:
        for index in range(count):
            output.write(batch_path(output_dir, index).read_bytes())
        output.flush()
        os.fsync(output.fileno())
    staging.replace(destination)


def build_prompt(row: dict, system_prompt: str) -> list[dict[str, str]]:
    question = str(row["instruction"])
    if row.get("input"):
        question += f"\n\n{row['input']}"
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]


def generate_batch(model, tokenizer, torch, prompts: list[dict], *, settings: dict, batch_index: int, system_prompt: str) -> list[dict]:
    rendered = [
        tokenizer.apply_chat_template(
            build_prompt(row, system_prompt), tokenize=False, add_generation_prompt=True
        )
        for row in prompts
    ]
    encoded = tokenizer(rendered, return_tensors="pt", padding=True)
    encoded = {name: value.to(model.device) for name, value in encoded.items()}
    seed = int(settings["seed"]) + batch_index
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    with torch.inference_mode():
        generated = model.generate(
            **encoded,
            max_new_tokens=int(settings["max_new_tokens"]),
            do_sample=True,
            temperature=float(settings["temperature"]),
            top_p=float(settings["top_p"]),
            num_return_sequences=int(settings["rollouts_per_group"]),
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
    completion_ids = generated[:, encoded["input_ids"].shape[1] :].detach().cpu()
    texts = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
    result = []
    cursor = 0
    for prompt in prompts:
        for rollout_index in range(int(settings["rollouts_per_group"])):
            ids = completion_ids[cursor].tolist()
            try:
                eos_index = ids.index(tokenizer.eos_token_id)
                token_count = eos_index
                ended_with_eos = True
            except ValueError:
                token_count = len(ids)
                ended_with_eos = False
            result.append(
                {
                    "group_id": prompt["group_id"],
                    "split": prompt["split"],
                    "source_index": prompt["source_index"],
                    "rollout_index": rollout_index,
                    "reference": prompt["reference"],
                    "completion": texts[cursor],
                    "completion_tokens": token_count,
                    "ended_with_eos": ended_with_eos,
                    "hit_max_new_tokens": not ended_with_eos and token_count == int(settings["max_new_tokens"]),
                    "generation_seed": seed,
                    "batch_index": batch_index,
                }
            )
            cursor += 1
    validate_batch(result, prompts, int(settings["rollouts_per_group"]), batch_index)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--rollouts-per-group", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=640)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--runtime-seconds", type=float, default=21600)
    args = parser.parse_args()

    if args.rollouts_per_group < 2 or args.max_new_tokens < 1 or args.runtime_seconds <= 0:
        parser.error("rollouts must be >=2 and token/runtime limits must be positive")
    prompts = read_jsonl(args.prompts)
    plan = batch_plan(prompts, args.batch_size)
    settings = {
        "schema_version": SCHEMA_VERSION,
        "model_id": args.model_id,
        "model_revision": args.model_revision,
        "model_weights_sha256": sha256(args.model / "model.safetensors"),
        "prompts_sha256": sha256(args.prompts),
        "batch_size": args.batch_size,
        "rollouts_per_group": args.rollouts_per_group,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "seed": args.seed,
        "seed_derivation": "seed + zero-based batch index",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "batches").mkdir(exist_ok=True)
    settings_path = args.output / "generation-contract.json"
    if settings_path.exists() and json.loads(settings_path.read_text()) != settings:
        raise ValueError("existing output was created under a different generation contract")
    atomic_json(settings_path, settings)
    completed = completed_batches(args.output, plan, args.rollouts_per_group)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from reasoning.env import SYSTEM_PROMPT

    if not torch.cuda.is_available():
        raise RuntimeError("this full rollout path requires CUDA")
    if torch.cuda.is_bf16_supported(including_emulation=False):
        precision = "float16-on-native-bf16-capable-device"
    else:
        precision = "float16"
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        attn_implementation="sdpa",
        local_files_only=True,
    ).to("cuda")
    model.eval()
    started = time.monotonic()
    stopped_for_runtime = False
    for index in range(completed, len(plan)):
        if index > completed and time.monotonic() - started >= args.runtime_seconds:
            stopped_for_runtime = True
            break
        rows = generate_batch(
            model, tokenizer, torch, plan[index], settings=settings,
            batch_index=index, system_prompt=SYSTEM_PROMPT,
        )
        write_batch(batch_path(args.output, index), rows)
        completed = index + 1
        atomic_json(
            args.output / "progress.json",
            {
                "completed_batches": completed,
                "completed_groups": sum(len(batch) for batch in plan[:completed]),
                "completed_rollouts": sum(len(batch) for batch in plan[:completed]) * args.rollouts_per_group,
                "total_batches": len(plan),
                "elapsed_seconds_this_process": time.monotonic() - started,
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                "free_bytes": torch.cuda.mem_get_info()[0],
                "precision": precision,
            },
        )

    flatten_batches(args.output, completed, args.output / "partial-rollouts.jsonl")
    complete = completed == len(plan)
    if complete:
        flatten_batches(args.output, completed, args.output / "rollouts.jsonl")
    status = {
        "complete": complete,
        "continuable": not complete,
        "stop_reason": "complete" if complete else "runtime" if stopped_for_runtime else "interrupted",
        "completed_batches": completed,
        "completed_groups": sum(len(batch) for batch in plan[:completed]),
        "completed_rollouts": sum(len(batch) for batch in plan[:completed]) * args.rollouts_per_group,
        "total_batches": len(plan),
        "total_groups": len(prompts),
        "expected_rollouts": len(prompts) * args.rollouts_per_group,
        "elapsed_seconds_this_process": time.monotonic() - started,
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "free_bytes": torch.cuda.mem_get_info()[0],
        "precision": precision,
    }
    atomic_json(args.output / "result.json", status)
    print(json.dumps(status, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
