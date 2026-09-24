#!/usr/bin/env python3
"""Run a pinned Kev-4B pairwise decision-model evaluation.

The scorer evaluates every pair in both answer placements and rotates Kev's three
choice labels three times per placement.  It keeps the six raw distributions so a
separate program can recompute every aggregate.  Heavy dependencies are imported
only by :class:`KevPredictor`; the protocol can therefore be tested on CPU with a
fake predictor.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
from typing import Callable, NamedTuple


KEV_MODEL_ID = "jaredpalmer/kev-4b"
KEV_MODEL_REVISION = "485ace8703592fcf405488b262449990824cfed1"
KEV_SOURCE_COMMIT = "557598fced1dada75dfbf36ed144dce309ac6ceb"
BASE_MODEL_ID = "Qwen/Qwen3.5-4B-Base"
BASE_MODEL_REVISION = "1001bb4d826a52d1f399e183466143f4da7b741b"
SERVE_MAX_PACKED = 16_384
SERVE_MAX_STATE = 8_192
BASE_SHARDS = (
    "model.safetensors-00001-of-00002.safetensors",
    "model.safetensors-00002-of-00002.safetensors",
)
BASE_SHARDS_SHA256 = {
    "model.safetensors-00001-of-00002.safetensors": "df547074dce70532a0493e5433152bd17a65efb89088cfabc2e7e2371a93d712",
    "model.safetensors-00002-of-00002.safetensors": "590fbaac095dd31db886c322d9d2f7df47777966391acf306ddddc3e4e3a15ef",
}
KEV_CHECKPOINT_FILES_SHA256 = {
    "adapter_config.json": "8a05dfd6c5e7f61a62e6db5d8de8093a8dbbd105aa7ee7b5e7b73f6249f45617",
    "adapter_model.safetensors": "9797de69a42188e411b17b7b4fcb66a23374dcebc21d71a7a66f836b5d34df2b",
    "head.pt": "d8f796da36ff7bd7c0fb9496b452139bb7851af4fc82b07b500b682d3f721d6a",
    "merges.txt": "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5",
    "special_tokens_map.json": "6676f091c8bc4d1b50146427cfde92073402866b87b6e39223227931b70083e9",
    "tokenizer.json": "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523",
    "tokenizer_config.json": "8671bed7c852ce9e661be94f179a7b4ffd091c2a65aea0363e5501c20318ee45",
    "vocab.json": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
}
VALIDATED_PACKAGE_VERSIONS = {
    "transformers": "5.17.0",
    "peft": "0.21.0",
    "accelerate": "1.15.0",
    "flash-linear-attention": "0.5.2",
    "pydantic": "2.12.3",
}

RUBRIC = (
    "Choose the answer that better satisfies the task using the supplied reference as a guide. "
    "Prioritize factual correctness, relevance, completeness, and clarity. Do not reward length by itself. "
    "Choose tie when the answers are materially equivalent or similarly wrong."
)


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def pair_digest(pair: dict) -> str:
    encoded = json.dumps(pair, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    # JSON strings may contain Unicode line separators. JSONL is delimited by U+000A.
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]


def validate_pair(pair: dict) -> None:
    required = {"pair_id", "task", "reference", "answers", "candidate_a", "candidate_b"}
    missing = required - set(pair)
    if missing:
        raise ValueError(f"pair is missing fields: {sorted(missing)}")
    a, b = pair["candidate_a"], pair["candidate_b"]
    if not isinstance(a, str) or not isinstance(b, str) or not a or not b or a == b:
        raise ValueError("candidate_a and candidate_b must be distinct non-empty strings")
    if set(pair["answers"]) != {a, b}:
        raise ValueError(f"answers must contain exactly {a!r} and {b!r}")
    for key in ("pair_id", "task", "reference"):
        if not isinstance(pair[key], str) or not pair[key]:
            raise ValueError(f"{key} must be a non-empty string")
    if not all(isinstance(value, str) for value in pair["answers"].values()):
        raise ValueError("answer values must be strings")


def rotate(mapping: dict, offset: int) -> dict:
    keys = list(mapping)
    offset %= len(keys)
    return {key: mapping[key] for key in keys[offset:] + keys[:offset]}


def request_for(pair: dict, placement: tuple[str, str], rotation: int) -> tuple[dict, dict[str, str]]:
    """Build one System One request and the option-to-candidate map."""
    left, right = placement
    if {left, right} != {pair["candidate_a"], pair["candidate_b"]}:
        raise ValueError("placement does not contain both candidates")
    option_to_candidate = {"answer_a": left, "answer_b": right, "tie": "tie"}
    criteria = {
        "answer_a": "Answer A is better under the rubric.",
        "answer_b": "Answer B is better under the rubric.",
        "tie": "The answers are materially equivalent or similarly wrong.",
    }
    request = {
        "state": {
            "task": pair["task"],
            "reference_answer": pair["reference"],
            "answer_a": pair["answers"][left],
            "answer_b": pair["answers"][right],
        },
        "questions": {
            "preference": {
                "type": "choice",
                "instructions": RUBRIC,
                "criteria": rotate(criteria, rotation),
            }
        },
    }
    return request, option_to_candidate


def valid_distribution(distribution: dict[str, float], labels: set[str], tolerance: float = 1e-5) -> None:
    if set(distribution) != labels:
        raise ValueError(f"distribution labels {sorted(distribution)} do not equal {sorted(labels)}")
    if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in distribution.values()):
        raise ValueError(f"invalid probability in {distribution}")
    if abs(sum(distribution.values()) - 1.0) >= tolerance:
        raise ValueError(f"probabilities do not sum to one: {distribution}")


def map_distribution(distribution: dict[str, float], mapping: dict[str, str]) -> dict[str, float]:
    return {mapping[key]: float(value) for key, value in distribution.items()}


def geometric_average(distributions: list[dict[str, float]]) -> dict[str, float]:
    if not distributions:
        raise ValueError("at least one distribution is required")
    keys = list(distributions[0])
    if any(set(distribution) != set(keys) for distribution in distributions):
        raise ValueError("distribution labels differ")
    logs = {
        key: sum(math.log(max(distribution[key], 1e-30)) for distribution in distributions) / len(distributions)
        for key in keys
    }
    top = max(logs.values())
    exponentials = {key: math.exp(value - top) for key, value in logs.items()}
    total = sum(exponentials.values())
    return {key: value / total for key, value in exponentials.items()}


class Prediction(NamedTuple):
    distribution: dict[str, float]
    prefix: object | None
    packed_tokens: int
    state_tokens: int


Predict = Callable[[dict, object | None], Prediction]


def score_pair(pair: dict, predict: Predict) -> dict:
    """Score one pair with two placements and three label rotations."""
    validate_pair(pair)
    a, b = pair["candidate_a"], pair["candidate_b"]
    placements = []
    for left, right in ((a, b), (b, a)):
        rotations: list[dict[str, float]] = []
        token_counts = []
        prefix = None
        for rotation in range(3):
            request, mapping = request_for(pair, (left, right), rotation)
            prediction = predict(request, prefix)
            valid_distribution(prediction.distribution, set(mapping))
            prefix = prediction.prefix
            rotations.append(map_distribution(prediction.distribution, mapping))
            token_counts.append({"packed": prediction.packed_tokens, "state": prediction.state_tokens})
        probabilities = geometric_average(rotations)
        valid_distribution(probabilities, {a, b, "tie"})
        placements.append(
            {
                "left": left,
                "right": right,
                "rotations": rotations,
                "probabilities": probabilities,
                "token_counts": token_counts,
                "argmax": max(probabilities, key=probabilities.get),
            }
        )

    placement_credits = [p["probabilities"][a] + 0.5 * p["probabilities"]["tie"] for p in placements]
    argmaxes = [p["argmax"] for p in placements]
    predicted = argmaxes[0] if argmaxes[0] == argmaxes[1] else "tie"
    result = {
        "pair_id": pair["pair_id"],
        "pair_sha256": pair_digest(pair),
        "candidate_a": a,
        "candidate_b": b,
        "placements": placements,
        "continuous_credit_a": statistics.mean(placement_credits),
        "discrete_credit_a": 1.0 if predicted == a else 0.5 if predicted == "tie" else 0.0,
        "placement_consistent": argmaxes[0] == argmaxes[1],
        "predicted_preference": predicted,
    }
    if "user_preference" in pair:
        if pair["user_preference"] not in {a, b, "tie"}:
            raise ValueError("user_preference must name candidate_a, candidate_b, or tie")
        result["user_preference"] = pair["user_preference"]
        result["agrees_with_user"] = predicted == pair["user_preference"]
    return result


def read_completed(path: Path, pairs: list[dict], rejected_ids: set[str] | None = None) -> dict[str, dict]:
    """Read resumable rows and bind each row to the exact current input pair."""
    by_id = {pair["pair_id"]: pair for pair in pairs}
    if len(by_id) != len(pairs):
        raise ValueError("input pair_id values are not unique")
    completed: dict[str, dict] = {}
    if not path.exists():
        return completed
    for row in read_jsonl(path):
        pair_id = row.get("pair_id")
        if pair_id not in by_id:
            raise ValueError(f"output contains unknown pair_id {pair_id!r}")
        if pair_id in completed:
            raise ValueError(f"output contains duplicate pair_id {pair_id!r}")
        if row.get("pair_sha256") != pair_digest(by_id[pair_id]):
            raise ValueError(f"output row {pair_id!r} belongs to different input bytes")
        completed[pair_id] = row
    rejected_ids = rejected_ids or set()
    if not rejected_ids <= set(by_id) or rejected_ids & set(completed):
        raise ValueError("rejected pair IDs are unknown or also present in scored output")
    processed = set(completed) | rejected_ids
    expected_prefix = [pair["pair_id"] for pair in pairs[: len(processed)]]
    if processed != set(expected_prefix):
        raise ValueError("scored and rejected rows are not an in-order prefix of the input")
    expected_scored = [pair_id for pair_id in expected_prefix if pair_id not in rejected_ids]
    if list(completed) != expected_scored:
        raise ValueError("resumable scored rows are not in input order")
    return completed


def append_jsonl(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)


def verify_source_commit(source: Path, expected: str = KEV_SOURCE_COMMIT) -> None:
    actual = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if actual != expected:
        raise RuntimeError(f"Kev source is {actual}, expected {expected}")
    status = subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain", "--untracked-files=all"], text=True
    )
    if status:
        raise RuntimeError("Kev source checkout has tracked or untracked modifications")


def verify_checkpoint_files(checkpoint: Path) -> dict[str, str]:
    actual = {}
    for name, expected in KEV_CHECKPOINT_FILES_SHA256.items():
        path = checkpoint / name
        if not path.is_file():
            raise RuntimeError(f"Kev checkpoint is missing {name}")
        actual[name] = sha256_file(path)
        if actual[name] != expected:
            raise RuntimeError(f"Kev checkpoint digest differs for {name}")
    return actual


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def verify_runtime_packages(allow_drift: bool = False) -> dict[str, str | None]:
    actual = {name: package_version(name) for name in VALIDATED_PACKAGE_VERSIONS}
    mismatches = {
        name: {"expected": expected, "actual": actual[name]}
        for name, expected in VALIDATED_PACKAGE_VERSIONS.items()
        if actual[name] != expected
    }
    if mismatches and not allow_drift:
        raise RuntimeError(f"serving package versions differ from the validated runtime: {mismatches}")
    return actual


def find_base_shards(cache: Path) -> dict[str, str]:
    found = {}
    for name in BASE_SHARDS:
        matches = [path for path in cache.rglob(name) if path.is_file() and path.stat().st_size]
        if not matches:
            raise RuntimeError(f"did not find non-empty cached {name}")
        digests = {sha256_file(path) for path in matches}
        if len(digests) != 1:
            raise RuntimeError(f"cached copies of {name} do not have one digest")
        found[name] = digests.pop()
        if found[name] != BASE_SHARDS_SHA256[name]:
            raise RuntimeError(f"cached base-model digest differs for {name}")
    return found


class KevPredictor:
    """Pinned, unmerged FP16 Kev runtime used by the published T4 experiment."""

    def __init__(self, source: Path, checkpoint: Path, device: str = "cuda") -> None:
        verify_source_commit(source)
        sys.path.insert(0, str(source))
        import torch
        from kev.api import SystemOneRequest, question_keys, to_record
        from kev.checkpoint import Checkpoint, LoadOptions
        from kev.model import SERVE_MAX_BRANCH, SERVE_MAX_PACKED, SERVE_MAX_STATE, ContextOverflow

        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        self.torch = torch
        self.SystemOneRequest = SystemOneRequest
        self.question_keys = question_keys
        self.to_record = to_record
        self.max_packed = SERVE_MAX_PACKED
        self.max_state = SERVE_MAX_STATE
        self.max_branch = SERVE_MAX_BRANCH
        self.ContextOverflow = ContextOverflow
        loaded = Checkpoint(str(checkpoint))
        if loaded.meta.base != BASE_MODEL_ID or loaded.meta.base_revision != BASE_MODEL_REVISION:
            raise RuntimeError(
                f"checkpoint base is {loaded.meta.base}@{loaded.meta.base_revision}; "
                f"expected {BASE_MODEL_ID}@{BASE_MODEL_REVISION}"
            )
        options = LoadOptions(dtype=torch.float16, merge=False, attn="sdpa", backend="torch")
        self.tokenizer, self.model = loaded.load(device, options)

    def __call__(self, request: dict, prefix: object | None = None) -> Prediction:
        typed = self.SystemOneRequest.model_validate(request)
        internal, _ = self.to_record(typed)
        encoded = self.model.encode(
            self.tokenizer,
            internal,
            max_state=self.max_state,
            max_branch=self.max_branch,
            strict=True,
        )
        if len(encoded["ids"]) > self.max_packed:
            raise self.ContextOverflow(f"packed request is {len(encoded['ids'])} tokens, limit {self.max_packed}")
        if prefix is None:
            probabilities, prefix = self.model.probs_and_prefix(encoded)
        else:
            probabilities = self.model.probs_with_prefix(encoded, prefix)
        keys = self.question_keys("choice", request["questions"]["preference"]["criteria"])
        distribution = dict(zip(keys, [float(value) for value in probabilities[0].tolist()]))
        return Prediction(distribution, prefix, len(encoded["ids"]), encoded["seg"].count(0))


def snapshot_checkpoint(cache: Path) -> Path:
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id=KEV_MODEL_ID,
            revision=KEV_MODEL_REVISION,
            cache_dir=cache,
        )
    )


def run(args: argparse.Namespace) -> dict:
    cache = Path(args.cache).resolve()
    cache.mkdir(parents=True, exist_ok=True)
    # Hugging Face fixes cache locations at import time. Set these before importing
    # Transformers, PEFT, Kev, or huggingface_hub.
    os.environ.update(HF_HOME=str(cache), HF_HUB_DISABLE_XET="1", TOKENIZERS_PARALLELISM="false")

    pairs = read_jsonl(Path(args.pairs))
    if not pairs:
        raise ValueError("pairs file is empty")
    for pair in pairs:
        validate_pair(pair)
    if len({pair["pair_id"] for pair in pairs}) != len(pairs):
        raise ValueError("pair_id values are not unique")

    output = Path(args.output_dir)
    rows_path = output / "kev-rows.jsonl"
    summary_path = output / "kev-summary.json"
    environment_path = output / "kev-environment.json"
    previous_summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    rejected = list(previous_summary.get("rejected") or [])
    for item in rejected:
        pair = next((pair for pair in pairs if pair["pair_id"] == item.get("pair_id")), None)
        if pair is None or item.get("pair_sha256") != pair_digest(pair):
            raise ValueError("saved rejection belongs to different input bytes")
    completed = read_completed(rows_path, pairs, {item["pair_id"] for item in rejected})
    if previous_summary.get("pass") and previous_summary.get("stage") == "complete":
        if len(completed) + len(rejected) != len(pairs):
            raise ValueError("complete summary does not account for every input pair")
        return previous_summary
    started = time.monotonic()
    started_utc = now()
    runtime_packages = verify_runtime_packages(args.allow_runtime_drift)
    checkpoint = Path(args.checkpoint).resolve() if args.checkpoint else snapshot_checkpoint(cache)
    checkpoint_files = verify_checkpoint_files(checkpoint)
    torchao = package_version("torchao")
    if torchao is not None:
        from packaging.version import Version
    if torchao is not None and Version(torchao) < Version("0.16"):
        raise RuntimeError(
            f"installed torchao {torchao} is incompatible with PEFT 0.21; remove unused torchao or install >=0.16"
        )
    predictor = KevPredictor(Path(args.kev_source).resolve(), checkpoint, args.device)

    import torch

    if args.require_device_substring and args.require_device_substring.lower() not in torch.cuda.get_device_name(0).lower():
        raise RuntimeError(
            f"device {torch.cuda.get_device_name(0)!r} does not contain {args.require_device_substring!r}"
        )
    torch.cuda.synchronize()
    setup_seconds = time.monotonic() - started
    torch.cuda.reset_peak_memory_stats()
    durations = []
    for index, pair in enumerate(pairs):
        if pair["pair_id"] in completed:
            continue
        if args.wall_cap_seconds and time.monotonic() - started > args.wall_cap_seconds - args.output_reserve_seconds:
            raise TimeoutError("wall cap reached with the configured output reserve")
        pair_started = time.monotonic()
        try:
            result = score_pair(pair, predictor)
        except predictor.ContextOverflow as error:
            rejected.append({"pair_id": pair["pair_id"], "pair_sha256": pair_digest(pair), "error": str(error)})
            atomic_json(
                summary_path,
                {
                    "schema_version": 1,
                    "pass": False,
                    "stage": "scoring",
                    "started_utc": previous_summary.get("started_utc", started_utc),
                    "completed_rows": len(completed),
                    "input_rows": len(pairs),
                    "rejected": rejected,
                },
            )
            continue
        result["elapsed_seconds"] = time.monotonic() - pair_started
        durations.append(result["elapsed_seconds"])
        append_jsonl(rows_path, result)
        completed[pair["pair_id"]] = result
        atomic_json(
            summary_path,
            {
                "schema_version": 1,
                "pass": False,
                "stage": "scoring",
                "started_utc": started_utc,
                "completed_rows": len(completed),
                "input_rows": len(pairs),
                "last_pair_id": pair["pair_id"],
                "rejected": rejected,
                "mean_pair_seconds_this_process": statistics.mean(durations),
            },
        )
        print(f"[{index + 1}/{len(pairs)}] {pair['pair_id']} {result['elapsed_seconds']:.2f}s", flush=True)

    if len(rejected) > args.max_rejections:
        atomic_json(
            summary_path,
            {
                "schema_version": 1,
                "pass": False,
                "stage": "gate_failed",
                "started_utc": previous_summary.get("started_utc", started_utc),
                "completed_rows": len(completed),
                "input_rows": len(pairs),
                "rejected": rejected,
                "max_rejections": args.max_rejections,
            },
        )
        raise RuntimeError(f"{len(rejected)} rows exceeded Kev's context limit; maximum is {args.max_rejections}")
    if len(completed) + len(rejected) != len(pairs):
        raise RuntimeError("not every input pair was scored or explicitly rejected")
    base_files = find_base_shards(cache)
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else args.device
    environment = {
        "schema_version": 1,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": device_name,
        "capability": list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None,
        "native_bf16": bool(torch.cuda.is_bf16_supported(including_emulation=False)) if torch.cuda.is_available() else None,
        "packages": {
            name: package_version(name)
            for name in ("transformers", "peft", "accelerate", "flash-linear-attention", "torchao", "triton", "pydantic")
        },
    }
    atomic_json(environment_path, environment)
    summary = {
        "schema_version": 1,
        "pass": True,
        "stage": "complete",
        "started_utc": started_utc,
        "finished_utc": now(),
        "input_sha256": sha256_file(Path(args.pairs)),
        "input_rows": len(pairs),
        "completed_rows": len(completed),
        "rejected": rejected,
        "model_id": KEV_MODEL_ID,
        "model_revision": KEV_MODEL_REVISION,
        "base_id": BASE_MODEL_ID,
        "base_revision": BASE_MODEL_REVISION,
        "kev_source_commit": KEV_SOURCE_COMMIT,
        "precision": "float16",
        "merge_adapter": False,
        "setup_seconds": setup_seconds,
        "scoring_seconds_this_process": sum(durations),
        "mean_pair_seconds_this_process": statistics.mean(durations) if durations else None,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
        "peak_reserved_bytes": torch.cuda.max_memory_reserved() if torch.cuda.is_available() else None,
        "resolved_base_files": base_files,
        "kev_checkpoint_files": checkpoint_files,
        "validated_runtime_packages": runtime_packages,
        "runtime_drift_allowed": args.allow_runtime_drift,
    }
    atomic_json(summary_path, summary)
    return summary


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--pairs", required=True, type=Path)
    result.add_argument("--kev-source", required=True, type=Path, help="Kev Git checkout at the pinned commit")
    result.add_argument("--checkpoint", type=Path, help="local pinned snapshot; otherwise download from the Hub")
    result.add_argument("--cache", required=True, type=Path, help="cache outside the source working tree")
    result.add_argument("--output-dir", required=True, type=Path)
    result.add_argument("--device", default="cuda")
    result.add_argument("--require-device-substring", default="")
    result.add_argument("--wall-cap-seconds", type=int, default=0, help="zero disables the wall cap")
    result.add_argument("--output-reserve-seconds", type=int, default=300)
    result.add_argument("--max-rejections", type=int, default=0)
    result.add_argument(
        "--allow-runtime-drift",
        action="store_true",
        help="run with different serving package versions and record that deviation",
    )
    return result


def main() -> None:
    arguments = parser().parse_args()
    if arguments.wall_cap_seconds and arguments.wall_cap_seconds <= arguments.output_reserve_seconds:
        raise SystemExit("wall cap must be greater than output reserve")
    print(json.dumps(run(arguments), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
