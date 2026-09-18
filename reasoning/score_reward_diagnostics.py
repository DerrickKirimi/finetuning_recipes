"""Score a frozen grouped-rollout corpus with one or more reward-model exports."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import tempfile
import zipfile

from reward_models.reward_model import load_reward_model
from reasoning.reward_diagnostics import add_semantic_scores, enrich_structural_rewards


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def atomic_json(path: Path, value) -> None:
    staging = path.with_suffix(path.suffix + ".partial")
    staging.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(staging, path)


def atomic_jsonl(path: Path, rows: list[dict]) -> None:
    staging = path.with_suffix(path.suffix + ".partial")
    with staging.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(staging, path)


def safe_extract(archive_path: Path, destination: Path) -> None:
    with zipfile.ZipFile(archive_path) as archive:
        if archive.testzip() is not None:
            raise ValueError(f"corrupt reward archive: {archive_path}")
        for member in archive.namelist():
            target = (destination / member).resolve()
            if not str(target).startswith(str(destination.resolve()) + os.sep):
                raise ValueError(f"unsafe archive member: {member}")
        archive.extractall(destination)


def validate_rollouts(rows: list[dict]) -> None:
    required = {"group_id", "split", "source_index", "rollout_index", "reference", "completion"}
    if len(rows) != 1600 or any(not required.issubset(row) for row in rows):
        raise ValueError("expected 1,600 complete rollout rows with the frozen identity fields")
    identities = {(row["group_id"], row["rollout_index"]) for row in rows}
    if len(identities) != len(rows):
        raise ValueError("duplicate group/rollout identity")
    groups = {}
    for row in rows:
        groups.setdefault(row["group_id"], []).append(row)
    if len(groups) != 200:
        raise ValueError("expected exactly 200 prompt groups")
    if any(
        sorted(row["rollout_index"] for row in group) != list(range(8))
        for group in groups.values()
    ):
        raise ValueError("every prompt group must contain rollout indices 0..7")
    split_counts = {
        split: len({row["group_id"] for row in rows if row.get("split") == split})
        for split in ("validation", "test")
    }
    if split_counts != {"validation": 100, "test": 100}:
        raise ValueError("expected 100 validation and 100 test groups")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True, help="JSON manifest of label/path/hash entries")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch-size must be positive")

    rows = read_jsonl(args.rollouts)
    validate_rollouts(rows)
    model_manifest = json.loads(args.models.read_text(encoding="utf-8"))
    specs = model_manifest.get("models", [])
    labels = [spec.get("label") for spec in specs]
    if not specs or any(not isinstance(label, str) or not label for label in labels):
        raise ValueError("model manifest must contain labelled models")
    if len(labels) != len(set(labels)):
        raise ValueError("model labels must be unique")

    scored = enrich_structural_rewards(rows)
    model_records = []
    with tempfile.TemporaryDirectory() as scratch:
        scratch_root = Path(scratch)
        for index, spec in enumerate(specs):
            source = Path(spec["path"]).expanduser().resolve()
            if not source.is_file():
                raise FileNotFoundError(source)
            source_hash = sha256(source)
            if not isinstance(spec.get("sha256"), str) or source_hash != spec["sha256"]:
                raise ValueError(f"archive hash mismatch for {spec['label']}")
            model_dir = scratch_root / f"model-{index}"
            model_dir.mkdir()
            safe_extract(source, model_dir)
            metadata = json.loads(
                (model_dir / "reward_model_metadata.json").read_text(encoding="utf-8")
            )
            for name, expected in spec.get("metadata", {}).items():
                if metadata.get(name) != expected:
                    raise ValueError(f"metadata mismatch for {spec['label']}: {name}")
            scorer, _ = load_reward_model(model_dir)
            add_semantic_scores(scored, spec["label"], scorer, batch_size=args.batch_size)
            model_records.append(
                {
                    "label": spec["label"],
                    "archive": source.name,
                    "archive_sha256": source_hash,
                    "metadata": metadata,
                }
            )
            del scorer
            gc.collect()

    args.output.mkdir(parents=True, exist_ok=False)
    scores_path = args.output / "reward-scores.jsonl"
    atomic_jsonl(scores_path, scored)
    result = {
        "schema_version": 1,
        "status": "complete",
        "rows": len(scored),
        "groups": len({row["group_id"] for row in scored}),
        "rollouts_sha256": sha256(args.rollouts),
        "models_manifest_sha256": sha256(args.models),
        "models": model_records,
        "scores_path": scores_path.name,
        "scores_sha256": sha256(scores_path),
        "batch_size": args.batch_size,
    }
    atomic_json(args.output / "scoring-result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
