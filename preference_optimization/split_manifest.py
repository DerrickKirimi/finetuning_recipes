"""Select DPO training and validation rows from a grouped split manifest instead of a random row split.

A grouped split is only meaningful if it is applied to exactly the data it was built from. So, before any model is
loaded, this module verifies:

* the dataset file's sha256 equals the one the manifest records (content identity, not a Hub name or row count);
* the manifest's and spot-check's own sha256 equal the values supplied by the caller;
* the spot-check was written for this manifest;
* the indices are plain integers, unique, in range, non-empty and match the recorded counts, with no overlap;
* every recorded split check passed (a Boolean property exactly True, an overlap count exactly 0);
* sampled (index, prompt key) pairs match the loaded rows.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


class SplitError(ValueError):
    """The manifest, spot-check or dataset do not describe the same split."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def prompt_key(row: dict[str, Any]) -> str:
    """sha256 of the whitespace-normalized non-system prompt messages, as the split builder computes it."""
    prompt = row["prompt"]
    if isinstance(prompt, str):
        text = prompt
    else:
        text = "\n".join(m["content"] for m in prompt if m.get("role") != "system")
    return hashlib.sha256(" ".join(text.split()).encode("utf-8")).hexdigest()


def _check_indices(name: str, indices: Any, rows: int, expected: int) -> None:
    if not isinstance(indices, list) or not indices:
        raise SplitError(f"{name} must be a non-empty list")
    if any(isinstance(i, bool) or not isinstance(i, int) for i in indices):
        raise SplitError(f"{name} must contain plain integers")
    if len(set(indices)) != len(indices):
        raise SplitError(f"{name} contains duplicate indices")
    if min(indices) < 0 or max(indices) >= rows:
        raise SplitError(f"{name} has indices outside 0..{rows - 1}")
    if len(indices) != expected:
        raise SplitError(f"{name} has {len(indices)} rows; the manifest records {expected}")


def verify_split_files(manifest_path: str | Path, spotcheck_path: str | Path, *, manifest_sha256: str,
                       spotcheck_sha256: str, dataset_file: str | Path) -> tuple[dict, list[dict]]:
    """Verify everything that does not need the rows loaded. Cheap; run it before allocating a model."""
    actual = sha256_file(manifest_path)
    if actual != manifest_sha256:
        raise SplitError(f"split manifest sha256 {actual[:12]} != expected {manifest_sha256[:12]}")
    actual = sha256_file(spotcheck_path)
    if actual != spotcheck_sha256:
        raise SplitError(f"spot-check sha256 {actual[:12]} != expected {spotcheck_sha256[:12]}")
    manifest = json.loads(Path(manifest_path).read_text())
    spotcheck = json.loads(Path(spotcheck_path).read_text())
    if spotcheck.get("manifest_sha256") != manifest_sha256:
        raise SplitError("spot-check was written for a different manifest")
    source = manifest["sources"]["dpo_train"]
    actual = sha256_file(dataset_file)
    if actual != source["sha256"]:
        raise SplitError(f"dataset file sha256 {actual[:12]} != the split's source {source['sha256'][:12]}")
    checks = manifest.get("checks") or {}
    failed = {k: v for k, v in checks.items()
              if (isinstance(v, bool) and v is not True) or (not isinstance(v, bool) and v != 0)}
    if not checks or failed or manifest.get("all_checks_pass") is not True:
        raise SplitError(f"split manifest records failed or missing checks: {failed or 'none recorded'}")
    rows = source["rows"]
    _check_indices("train_indices", manifest.get("train_indices"), rows, manifest["train_rows"])
    _check_indices("validation_indices", manifest.get("validation_indices"), rows, manifest["validation_rows"])
    if set(manifest["train_indices"]) & set(manifest["validation_indices"]):
        raise SplitError("split manifest train and validation indices overlap")
    items = spotcheck.get("rows") or []
    if not items:
        raise SplitError("spot-check list is empty")
    return manifest, items


def select_split(dataset, manifest: dict, spotcheck: list[dict]):
    """Return (train_dataset, eval_dataset) from an already-verified manifest, checking sampled rows on the loaded data."""
    expected_rows = manifest["sources"]["dpo_train"]["rows"]
    if len(dataset) != expected_rows:
        raise SplitError(f"dataset has {len(dataset)} rows; the split manifest was built for {expected_rows}")
    for item in spotcheck:
        actual = prompt_key(dataset[item["index"]])
        if actual != item["prompt_key"]:
            raise SplitError(f"row {item['index']} prompt key {actual[:12]} does not match the split's {item['prompt_key'][:12]}")
    return dataset.select(manifest["train_indices"]), dataset.select(manifest["validation_indices"])
