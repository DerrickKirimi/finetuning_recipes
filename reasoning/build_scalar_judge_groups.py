"""Join frozen prompts and grouped rollouts into scalar-judge requests."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path

from reasoning import env


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def final_answer(completion: str) -> str:
    if env._has_expected_format(completion):
        return env._extract_think_content(completion)[1]
    text = completion.strip()
    if "</think>" in text:
        return text.rsplit("</think>", 1)[1].strip() or "[empty final answer]"
    if "<think>" in text:
        return "[malformed reasoning output omitted]"
    return text


def build_groups(prompts: list[dict], rollouts: list[dict]) -> list[dict]:
    if len(prompts) != 200 or len({row.get("group_id") for row in prompts}) != 200:
        raise ValueError("expected 200 unique frozen prompt groups")
    split_counts = Counter(row.get("split") for row in prompts)
    if split_counts != {"validation": 100, "test": 100}:
        raise ValueError("expected exactly 100 validation and 100 test groups")
    if len({row.get("source_index") for row in prompts}) != 200:
        raise ValueError("expected 200 unique source indices")
    for split in ("validation", "test"):
        expected = {f"{split}-{index:03d}" for index in range(100)}
        actual = {row["group_id"] for row in prompts if row.get("split") == split}
        if actual != expected:
            raise ValueError(f"{split} group ids must be contiguous 000..099")
    by_group: dict[str, list[dict]] = defaultdict(list)
    for row in rollouts:
        by_group[row.get("group_id")].append(row)
    if set(by_group) != {row["group_id"] for row in prompts}:
        raise ValueError("rollout and prompt group identities differ")
    result = []
    for prompt in prompts:
        group_id = prompt["group_id"]
        rows = sorted(by_group[group_id], key=lambda row: row.get("rollout_index", -1))
        if [row.get("rollout_index") for row in rows] != list(range(8)):
            raise ValueError(f"{group_id} must contain rollout indices 0..7")
        if any(
            row.get("split") != prompt["split"]
            or row.get("source_index") != prompt["source_index"]
            or row.get("reference") != prompt["reference"]
            for row in rows
        ):
            raise ValueError(f"{group_id} changed frozen prompt identity")
        task = str(prompt["instruction"])
        if prompt.get("input"):
            task += f"\n\n{prompt['input']}"
        candidates = []
        for row in rows:
            completion = str(row["completion"])
            candidates.append(
                {
                    "candidate_id": f"{group_id}:{row['rollout_index']}",
                    "rollout_index": row["rollout_index"],
                    "answer": final_answer(completion),
                    "format_valid": env._has_expected_format(completion),
                    "completion_sha256": hashlib.sha256(completion.encode("utf-8")).hexdigest(),
                }
            )
        result.append(
            {
                "group_id": group_id,
                "split": prompt["split"],
                "source_index": prompt["source_index"],
                "task": task,
                "reference": prompt["reference"],
                "candidates": candidates,
            }
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    groups = build_groups(read_jsonl(args.prompts), read_jsonl(args.rollouts))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    staging = args.output.with_suffix(args.output.suffix + ".partial")
    with staging.open("w", encoding="utf-8") as handle:
        for group in groups:
            handle.write(json.dumps(group, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(staging, args.output)
    report = {
        "schema_version": 1,
        "hidden_reasoning_policy": "send final answer only; omit malformed think content",
        "groups": len(groups),
        "candidates": sum(len(group["candidates"]) for group in groups),
        "prompts_sha256": sha256(args.prompts),
        "rollouts_sha256": sha256(args.rollouts),
        "output_sha256": sha256(args.output),
    }
    report_path = args.output.with_name(args.output.stem + "-manifest.json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
