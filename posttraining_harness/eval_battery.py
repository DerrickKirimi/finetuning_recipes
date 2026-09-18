"""The fixed cross-stage evaluation battery: base / CPT / SFT / DPO / GRPO, one harness.

**Once run against a checkpoint this must not change** — any later edit invalidates every
comparison made before it. A discovered mistake becomes a new, separately named battery, not an
amendment.

Two tracks, reported separately
-------------------------------
Base and CPT are not instruction-tuned and cannot meaningfully answer a chat-formatted
prompt, while SFT onward are trained for exactly that. Scoring only one framing would
either handicap the instruction models or flatter them.

- **Track A, continuation.** The passage prefix (`input`) alone, no template, no system
  prompt; the reference is `output`. This is what CPT was evaluated on and the only track
  on which base and CPT are meaningfully comparable to SFT.
- **Track B, instruction.** The exact ChatML rendering used in training - system prompt,
  then `"{instruction}\\n\\n{input}"` as the user turn - with `add_generation_prompt=True`.
  Base and CPT are still scored here and are expected to do badly; that gap is the finding.

A single blended number across both would hide which effect is which.

Prompt set
----------
Drawn from `fixed_test_battery` in the SFT split manifest: rows from a **separate test
source**, with zero overlap against the full training source, one row per seeded clean
component. This is deliberately *not* the internal validation split, which was used for
checkpoint selection and is therefore not an untouched test set.

The manifest's own `prompt_sha256` values are re-verified on load, so a silently changed
source is caught rather than quietly evaluated.
"""
from __future__ import annotations

import ast
import hashlib
import json
import math
import shutil
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

BATTERY_VERSION = "1"
DEFAULT_PROMPTS = 500
MAX_NEW_TOKENS = 256
REPETITION_PENALTY = 1.2
GENERATION_BATCH_SIZE = 4
BERTSCORE_MODEL = "distilbert-base-uncased"
BERTSCORE_MODEL_REVISION = "12040accade4e8a0f71eabdb258fecc2e7e948be"
BERTSCORE_NUM_LAYERS = 5
BERTSCORE_MODEL_FILES_SHA256 = {
    "config.json": "69c94b0222d5d1f4b0ad027ca7416cdafb98378cbbb8305d0bf47c9365c60c83",
    "model.safetensors": "5e3f1108e3cb34ee048634875d8482665b65ac713291a7e32396fb18f6ff0063",
    "tokenizer.json": "ce64fce797c24f68df90b40a3f74f579b336a493db14bd583fd520ea0d8c9a98",
    "tokenizer_config.json": "a025160ef0431f1a392f6f050c1310f4c5d9fb6f275932dbccba73c4d214bf10",
    "vocab.txt": "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3",
}


def merged_prompt(instruction: str, passage: str) -> str:
    """The prompt exactly as TRAINING built it (sft.py: instruction, blank line, input).

    Raw, not whitespace-collapsed: this is the string the model is actually shown.
    """
    return f"{instruction}\n\n{passage}"


def identity_prompt(instruction: str, passage: str) -> str:
    """The prompt IDENTITY used by the split manifest, which is a DIFFERENT string.

    posttraining_harness/data_boundaries.py collapses whitespace before hashing, because identity exists
    for deduplication and overlap detection, not for prompting. Conflating the two fails
    verification on every row whose whitespace is not already normalised - which is exactly
    how the first version of this module rejected the real 500-row battery.
    """
    # posttraining_harness/data_boundaries.py imports its siblings as bare modules, so the package path
    # alone cannot import it. This is the repo's existing convention - see
    # posttraining_harness/dpo_tokens.py:17-20 - and importing the real function is worth the awkwardness:
    # a local copy of `normalized` would drift from the one that built the manifest.
    import sys
    directory = str(Path(__file__).resolve().parent)
    if directory not in sys.path:
        sys.path.insert(0, directory)
    from data_boundaries import normalized
    return normalized(instruction) + "\n\n" + normalized(passage)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def extract_system_prompt(sft_path: Path) -> str:
    """Read the system prompt out of `sft.py` rather than restating it here.

    A copy would drift, and a drifted system prompt silently changes track B into a
    different task while every number still looks comparable.
    """
    tree = ast.parse(Path(sft_path).read_text())
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign)
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == "SYSTEM_PROMPT"
                and isinstance(node.value, ast.Constant)):
            return node.value.value
    raise AssertionError(f"no SYSTEM_PROMPT literal found in {sft_path}")


def build_manifest(split_manifest_path, test_source_path, sft_path, count=DEFAULT_PROMPTS):
    """Pin the battery: which rows, in which order, and what they contained.

    Takes the first `count` indices of `fixed_test_battery` in its recorded order, which is
    already the result of a seeded shuffle over clean components. Re-verifies each row's
    recorded `prompt_sha256` against the source before accepting it.
    """
    split = json.loads(Path(split_manifest_path).read_text())
    battery = split["fixed_test_battery"]
    indices = list(battery["indices"])[:count]
    expected = dict(zip(battery["indices"], battery["prompt_sha256"]))

    wanted = set(indices)
    rows = {}
    with open(test_source_path) as handle:
        for position, line in enumerate(handle):
            if position in wanted:
                rows[position] = json.loads(line)
    missing = wanted - set(rows)
    assert not missing, f"test source is missing battery rows: {sorted(missing)[:5]}"

    selected, drifted = [], []
    for index in indices:
        row = rows[index]
        identity = sha256_text(identity_prompt(row["instruction"], row.get("input", "")))
        if expected.get(index) not in (None, identity):
            drifted.append({"index": index, "expected": expected[index], "actual": identity})
        selected.append({"index": index, "prompt_sha256": identity,
                         "merged_prompt_sha256": sha256_text(
                             merged_prompt(row["instruction"], row.get("input", ""))),
                         "output_sha256": sha256_text(row["output"])})
    assert not drifted, f"test source drifted from the split manifest: {drifted[:3]}"

    system_prompt = extract_system_prompt(sft_path)
    # Which rows each track can actually cover. Recorded, never discovered at run time:
    # the tracks cover different row sets by necessity, and that must be auditable.
    continuation_indices = [row["index"] for row in selected
                            if continuation_split(rows[row["index"]].get("input", "")) is not None]
    content = hashlib.sha256(
        json.dumps([r["prompt_sha256"] for r in selected], sort_keys=True).encode()).hexdigest()
    return {
        "battery_version": BATTERY_VERSION,
        "prompt_count": len(selected),
        "content_sha256": content,
        "rows": selected,
        "source": {
            "dataset": split["dataset"], "revision": split["dataset_revision"],
            "test_source_sha256": split["test_source"]["sha256"],
            "selection": battery["selection"],
            "overlap_with_full_training_source": battery["overlap_with_full_training_source"],
        },
        "generation": {
            "max_new_tokens": MAX_NEW_TOKENS, "do_sample": False,
            "repetition_penalty": REPETITION_PENALTY, "batch_size": GENERATION_BATCH_SIZE,
        },
        "tracks": {
            "continuation": {
                "description": "prefix/suffix split of the passage; output is not used",
                "min_words": CONTINUATION_MIN_WORDS,
                "prefix_fraction": CONTINUATION_PREFIX_FRACTION,
                "indices": continuation_indices,
                "row_count": len(continuation_indices),
            },
            "instruction": {
                "description": "ChatML, system + {instruction}\\n\\n{input}, generation prompt",
                "template_sha256": sha256_text(CHATML_TEMPLATE),
                "indices": [row["index"] for row in selected],
                "row_count": len(selected),
            },
        },
        "system_prompt_sha256": sha256_text(system_prompt),
        "bertscore_model": BERTSCORE_MODEL,
        "bertscore_model_revision": BERTSCORE_MODEL_REVISION,
        "bertscore_num_layers": BERTSCORE_NUM_LAYERS,
        "bertscore_model_files_sha256": BERTSCORE_MODEL_FILES_SHA256,
        "limits": [
            "The prompt set is a held-out TEST battery, not the internal validation split "
            "used for checkpoint selection.",
            "Component grouping establishes prompt/passage disjointness, not source-document "
            "disjointness: published rows carry no document id.",
            "One seed, one run per checkpoint: the direction of an effect is established, "
            "its magnitude is not.",
            "Greedy decoding only; says nothing about sampled-generation quality.",
            "The two tracks cover DIFFERENT row sets; compare across checkpoints within a "
            "track, never across tracks.",
            "Instruction-track numbers are NOT comparable to the earlier CPT metric values, "
            "which were continuation-only under different extraction.",
        ],
    }


def load_rows(manifest, test_source_path):
    """Re-read the pinned rows and re-verify their hashes. Never trust the manifest alone."""
    wanted = {row["index"]: row for row in manifest["rows"]}
    loaded = {}
    with open(test_source_path) as handle:
        for position, line in enumerate(handle):
            if position in wanted:
                loaded[position] = json.loads(line)
    assert set(loaded) == set(wanted), "pinned rows absent from the test source"
    out = []
    for row in manifest["rows"]:
        source = loaded[row["index"]]
        identity = identity_prompt(source["instruction"], source.get("input", ""))
        assert sha256_text(identity) == row["prompt_sha256"], f"row {row['index']} identity drifted"
        prompt = merged_prompt(source["instruction"], source.get("input", ""))
        assert sha256_text(prompt) == row["merged_prompt_sha256"], f"row {row['index']} prompt drifted"
        assert sha256_text(source["output"]) == row["output_sha256"], f"row {row['index']} output drifted"
        out.append({"index": row["index"], "instruction": source["instruction"],
                    "input": source.get("input", ""), "output": source["output"],
                    "merged_prompt": prompt})
    return out


def verify_bertscore_snapshot(snapshot_path, manifest):
    """Require the exact scorer weights frozen by the battery manifest.

    A Hub model name follows its branch. BERTScore accepts a local Transformers directory,
    so the battery resolves the named model once, verifies every required byte here, and
    scores from that directory. This makes the actual scorer weights portable and makes a
    revision mismatch fail before checkpoint loading or generation.
    """
    path = Path(snapshot_path)
    assert path.is_dir(), f"BERTScore snapshot is not a directory: {path}"
    expected_revision = manifest["bertscore_model_revision"]
    assert path.name == expected_revision, (
        f"BERTScore snapshot directory {path.name!r} is not pinned revision "
        f"{expected_revision!r}")
    expected = manifest["bertscore_model_files_sha256"]
    missing = sorted(name for name in expected if not (path / name).is_file())
    assert not missing, f"BERTScore snapshot is missing required files: {missing}"
    actual = {name: hashlib.sha256((path / name).read_bytes()).hexdigest()
              for name in sorted(expected)}
    mismatched = {name: {"expected": expected[name], "actual": actual[name]}
                  for name in expected if actual[name] != expected[name]}
    assert not mismatched, f"BERTScore snapshot file mismatch: {mismatched}"
    return {
        "model": manifest["bertscore_model"],
        "revision": expected_revision,
        "num_layers": manifest["bertscore_num_layers"],
        "path": str(path),
        "files_sha256": actual,
    }


CONTINUATION_MIN_WORDS = 60
CONTINUATION_PREFIX_FRACTION = 0.6


def continuation_split(passage, min_words=CONTINUATION_MIN_WORDS,
                       fraction=CONTINUATION_PREFIX_FRACTION):
    """Split a passage into (prefix, continuation), or return None if it is too short.

    Splitting on whitespace rather than tokens keeps the split independent of any
    tokenizer, so every checkpoint sees byte-identical prompts regardless of its
    vocabulary. Returning None rather than a degenerate split is deliberate: a two-word
    reference scores as noise and would quietly dilute the metric.
    """
    words = passage.split()
    if len(words) < min_words:
        return None
    cut = max(1, int(len(words) * fraction))
    if len(words) - cut < 1:
        return None
    return " ".join(words[:cut]), " ".join(words[cut:])


def render_continuation(row):
    """Prefix of the passage in, rest of the passage out. No template, no system prompt."""
    split = continuation_split(row["input"])
    assert split is not None, f"row {row['index']} is too short for the continuation track"
    return split


CHATML_TEMPLATE = (
    "<|im_start|>system\n{system}<|im_end|>\n"
    "<|im_start|>user\n{user}<|im_end|>\n"
    "<|im_start|>assistant\n"
)


def render_chatml(system_prompt, user_prompt):
    """Render ChatML explicitly, never via the tokenizer's own template.

    `tokenizer.apply_chat_template` uses whatever template that tokenizer carries. The base
    and CPT tokenizers may carry none, or a different one, so the same row would reach
    different checkpoints as different bytes and the comparison would measure the template.
    The literal here is the ChatML shape `get_chat_template(..., "chatml")` installs, with
    a trailing generation prompt because this is inference and training had none.
    """
    return CHATML_TEMPLATE.format(system=system_prompt, user=user_prompt)


def render_instruction(row, system_prompt):
    """The instruction track: training's rendering, plus a generation prompt."""
    return render_chatml(system_prompt, row["merged_prompt"]), row["output"]


def validate_manifest(manifest):
    """Refuse to run unless the module's constants still match what the manifest pinned.

    A manifest that only describes settings is decoration: the run can quietly use
    different ones. These are the settings that change the numbers.
    """
    generation = manifest["generation"]
    mismatches = []
    for key, actual in (("max_new_tokens", MAX_NEW_TOKENS), ("do_sample", False),
                        ("repetition_penalty", REPETITION_PENALTY),
                        ("batch_size", GENERATION_BATCH_SIZE)):
        if generation.get(key) != actual:
            mismatches.append({"setting": key, "manifest": generation.get(key), "module": actual})
    instruction = manifest["tracks"]["instruction"]
    if instruction.get("template_sha256") != sha256_text(CHATML_TEMPLATE):
        mismatches.append({"setting": "chatml_template", "manifest": instruction.get("template_sha256"),
                           "module": sha256_text(CHATML_TEMPLATE)})
    continuation = manifest["tracks"]["continuation"]
    for key, actual in (("min_words", CONTINUATION_MIN_WORDS),
                        ("prefix_fraction", CONTINUATION_PREFIX_FRACTION)):
        if continuation.get(key) != actual:
            mismatches.append({"setting": f"continuation.{key}",
                               "manifest": continuation.get(key), "module": actual})
    for key, actual in (
            ("bertscore_model", BERTSCORE_MODEL),
            ("bertscore_model_revision", BERTSCORE_MODEL_REVISION),
            ("bertscore_num_layers", BERTSCORE_NUM_LAYERS),
            ("bertscore_model_files_sha256", BERTSCORE_MODEL_FILES_SHA256)):
        if manifest.get(key) != actual:
            mismatches.append({"setting": key, "manifest": manifest.get(key),
                               "module": actual})
    assert not mismatches, f"module settings differ from the pinned battery: {mismatches}"
    return True


def run_track(model, tokenizer, rows, track, system_prompt, generate_batch,
              compute_perplexity, max_new_tokens=MAX_NEW_TOKENS,
              batch_size=GENERATION_BATCH_SIZE):
    """Generate and score one track.

    `generate_batch` and `compute_perplexity` are injected from `cpt/inference.py`, which
    already handles left padding for decoder-only batch generation. Injection keeps this
    module testable on CPU without importing torch.
    """
    assert track in ("continuation", "instruction"), f"unknown track {track!r}"
    prompts, references, used = [], [], []
    for row in rows:
        if track == "continuation":
            split = continuation_split(row["input"])
            if split is None:
                continue                      # recorded in the manifest, not silently dropped
            prompt, reference = split
        else:
            prompt, reference = render_instruction(row, system_prompt)
        prompts.append(prompt)
        references.append(reference)
        used.append(row["index"])

    assert prompts, f"track {track}: no usable rows"
    # Wall time per phase, so the first full run calibrates the ones after it. Generation
    # dominates and scales with the tokens actually produced, not with the row count, which
    # is why a bounded smoke could not forecast a full track.
    began = time.monotonic()
    predictions = generate_batch(model, tokenizer, prompts,
                                 max_new_tokens=max_new_tokens, batch_size=batch_size,
                                 repetition_penalty=REPETITION_PENALTY)
    generation_seconds = round(time.monotonic() - began, 3)
    assert len(predictions) == len(prompts), (
        f"track {track}: {len(predictions)} generations for {len(prompts)} prompts")
    # A dict: the number, how many rows produced it, and which could not be scored.
    # `used` carries the row ids, so a skipped row is reported by identity rather than by
    # a bare count that nobody can chase back to a row.
    began = time.monotonic()
    perplexity = compute_perplexity(model, tokenizer, prompts, references, indices=used)
    perplexity_seconds = round(time.monotonic() - began, 3)
    assert isinstance(perplexity, dict) and "perplexity" in perplexity, (
        "compute_perplexity must return the detail dict, not a bare float: rows that "
        "cannot be scored have to be visible rather than absorbed into an aggregate")
    return {
        "track": track,
        "indices": used,
        "timing": {"generation_seconds": generation_seconds,
                   "perplexity_seconds": perplexity_seconds,
                   "rows": len(prompts),
                   "generated_words": sum(len(p.split()) for p in predictions)},
        "prompts": prompts,
        "prompt_sha256": [sha256_text(p) for p in prompts],
        "predictions": predictions,
        "references": references,
        "perplexity": perplexity,
    }


def score_track(result, calculate_metrics):
    """Aggregate metrics only.

    Per-sample scores are written to `generations.jsonl` for inspection but are never the
    headline: a BLEU figure was once misreported here by reading a per-sample table as the
    aggregate.
    """
    empty = sum(1 for p in result["predictions"] if not p.strip())
    metrics = {k: float(v) for k, v in
               calculate_metrics(result["predictions"], result["references"]).items()}
    detail = result["perplexity"]
    metrics["perplexity"] = float(detail["perplexity"])
    # Perplexity and the generation metrics can cover different row counts, because a row
    # may generate fine and still be unscoreable for perplexity. Reporting one count would
    # misdescribe one of them.
    metrics["perplexity_scored_rows"] = detail["scored_rows"]
    metrics["perplexity_skipped_rows"] = len(detail["skipped"])
    metrics["perplexity_skipped"] = detail["skipped"]        # ids and reasons, not just a count
    metrics["supervised_tokens"] = detail["supervised_tokens"]
    metrics["empty_generations"] = empty
    metrics["generation_scored_examples"] = len(result["predictions"])
    # A NaN or an infinity is not a low score, it is the absence of one. Report it by name
    # rather than raising: the generations cost the run and are already in hand, so an
    # aggregate that cannot be computed must not also destroy the text that produced it.
    # Consumers gate on this list; `write_outputs` refuses to publish a nonempty one.
    metrics.update({f"{k}": v for k, v in result.get("timing", {}).items()
                    if k.endswith("_seconds")})
    metrics["nonfinite_metrics"] = sorted(
        name for name, value in metrics.items()
        if isinstance(value, float) and not math.isfinite(value))
    return metrics


def write_outputs(directory, manifest, provenance, results, metrics, overwrite=False):
    """Write one checkpoint's evaluation, with provenance that identifies what was run.

    `provenance` must identify the artefact immutably, not label it. A directory named
    "sft" tells a later reader nothing about which adapter, which parent, or which
    revision produced the numbers.

    Refuses to overwrite a completed evaluation: re-running silently over a finished result
    is how a battery stops being fixed.
    """
    required = {"checkpoint_label", "model_path", "weight_sha256", "parent", "adapter_sha256"}
    missing = required - set(provenance)
    assert not missing, f"provenance is missing {sorted(missing)}"

    # A published result must not carry a NaN or an infinity. The per-track partials were
    # already written, so refusing here costs the consolidation and keeps the generations:
    # the run is recoverable, and nothing unusable acquires a manifest and a SHA256SUMS.
    unusable = {track: values["nonfinite_metrics"] for track, values in metrics.items()
                if values.get("nonfinite_metrics")}
    assert not unusable, (
        f"refusing to publish nonfinite metrics: {unusable}. A checksum proves the bytes, "
        "not the arithmetic. The generations survive in the per-track partials; fix the "
        "scorer or the rows and rescore from them rather than regenerating.")

    directory = Path(directory)
    completed = directory / "metrics.json"
    if completed.exists() and not overwrite:
        raise AssertionError(
            f"{completed} already exists; refusing to overwrite a completed evaluation. "
            "Write to a new directory, or pass overwrite=True deliberately.")
    directory.mkdir(parents=True, exist_ok=True)

    with (directory / "generations.jsonl").open("w") as handle:
        for track, result in results.items():
            for position, index in enumerate(result["indices"]):
                handle.write(json.dumps({
                    "checkpoint": provenance["checkpoint_label"], "track": track,
                    "index": index,
                    "prompt_sha256": result["prompt_sha256"][position],
                    "prediction": result["predictions"][position],
                    "reference": result["references"][position],
                }, sort_keys=True) + "\n")

    (directory / "metrics.json").write_text(json.dumps({
        "provenance": provenance,
        "battery_version": manifest["battery_version"],
        "battery_content_sha256": manifest["content_sha256"],
        "generation": manifest["generation"],
        "tracks": metrics,
        "rows_per_track": {track: len(result["indices"]) for track, result in results.items()},
    }, indent=2, sort_keys=True) + "\n")
    (directory / "battery-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    digests = []
    for path in sorted(directory.iterdir()):
        # Partials are deliberately removed after the consolidated write succeeds. They
        # must never enter the checksum file for the completed result: doing so leaves a
        # SHA256SUMS that fails immediately after its own writer returns.
        if (path.name == "SHA256SUMS" or path.name.startswith("partial-")
                or not path.is_file()):
            continue
        digests.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}")
    (directory / "SHA256SUMS").write_text("\n".join(digests) + "\n")
    return directory


CHATML_MARKERS = ("<|im_start|>", "<|im_end|>", "<|endoftext|>")


def verify_tokenizer_markers(tokenizer, checkpoint_dir):
    """The loaded tokenizer must give the ChatML markers the ids this checkpoint trained on.

    The SFT checkpoint does not merely add tokens: it swaps two. Its saved tokenizer maps
    <|im_end|> to id 0 and displaces <|endoftext|> to 2, the reverse of the base, because
    ChatML's turn end is bound onto the pretrained EOS embedding. Both files then list an
    identical set of 49,152 token strings, so a vocabulary-membership check sees no
    difference at all -- and loading the base tokenizer alongside the SFT adapter would
    feed id 2 everywhere training used id 0, in every prompt of the instruction track,
    producing numbers that look like a weak SFT rather than a misconfigured one.

    Compared against the ids recorded on disk rather than against a hardcoded table, so a
    checkpoint that legitimately arranges them differently is respected, not overruled.
    """
    report = {"checkpoint_dir": str(checkpoint_dir)}
    tokenizer_file = Path(checkpoint_dir) / "tokenizer.json"
    if not tokenizer_file.exists():
        report["skipped"] = f"{tokenizer_file} absent; nothing to compare against"
        return report
    saved = {entry["content"]: entry["id"]
             for entry in json.loads(tokenizer_file.read_text()).get("added_tokens", [])}
    loaded, disagreements = {}, {}
    for marker in CHATML_MARKERS:
        if marker not in saved:
            continue
        actual = tokenizer.convert_tokens_to_ids(marker)
        loaded[marker] = actual
        if actual != saved[marker]:
            disagreements[marker] = {"on_disk": saved[marker], "loaded": actual}
    report["marker_ids"] = loaded
    report["disagreements"] = disagreements
    if disagreements:
        raise AssertionError(
            f"the loaded tokenizer disagrees with {tokenizer_file} on {sorted(disagreements)}: "
            f"{disagreements}. Every prompt would be tokenized differently from training; "
            f"refusing to evaluate. Load the tokenizer saved beside the checkpoint.")
    return report


def _partial_run_identity(manifest, provenance):
    """Bind reusable partials to immutable inputs without binding them to a temp path."""
    runtime = provenance.get("runtime") or {}
    identity = {
        "manifest_sha256": hashlib.sha256(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "checkpoint_label": provenance.get("checkpoint_label"),
        "weight_sha256": provenance.get("weight_sha256"),
        "adapter_sha256": provenance.get("adapter_sha256"),
        "parent_check": provenance.get("parent_check"),
        "metric_runtime": provenance.get("metric_runtime"),
        "limit": provenance.get("limit"),
        "runtime": {
            key: runtime.get(key) for key in (
                "dtype", "device", "quantization", "tokenizer_serialisation_sha256",
                "special_token_ids", "padding_side")
        },
        "evaluator_source_sha256": {
            "posttraining_harness/eval_battery.py": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "cpt/inference.py": hashlib.sha256(
                (Path(__file__).resolve().parents[1] / "cpt/inference.py").read_bytes()
            ).hexdigest(),
            "cpt/evals.py": hashlib.sha256(
                (Path(__file__).resolve().parents[1] / "cpt/evals.py").read_bytes()
            ).hexdigest(),
        },
    }
    return hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def write_track_partial(directory, track, manifest, provenance, result, metrics):
    """Persist one finished track before starting the next.

    The battery writes its real outputs only once both tracks are scored, so an
    interruption after an hour of generation used to leave nothing on disk -- and the
    expensive half is generation, which is exactly the half worth not repeating. A
    partial is not a result: it carries no manifest and no SHA256SUMS, is named so it
    cannot be mistaken for one, and is removed once the consolidated write succeeds.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"partial-{track}.json"
    path.write_text(json.dumps({
        "warning": "incomplete run; not a published result",
        "track": track,
        "checkpoint_label": provenance.get("checkpoint_label"),
        "run_identity_sha256": _partial_run_identity(manifest, provenance),
        "written": datetime.now(timezone.utc).isoformat(),
        "metrics": metrics,
        "rows": [
            {"index": index,
             "prompt_sha256": result["prompt_sha256"][position],
             "prediction": result["predictions"][position],
             "reference": result["references"][position]}
            for position, index in enumerate(result["indices"])
        ],
    }, indent=2, sort_keys=True, default=str) + "\n")
    return path


def load_track_partial(directory, track, manifest, provenance, expected_rows=None,
                       system_prompt=None):
    """Load a completed track only when it belongs to this exact evaluation run."""
    path = Path(directory) / f"partial-{track}.json"
    if not path.exists():
        return None
    saved = json.loads(path.read_text())
    assert saved.get("warning") == "incomplete run; not a published result", (
        f"{path} is not a battery partial")
    assert saved.get("track") == track, f"{path} claims track {saved.get('track')!r}"
    expected = _partial_run_identity(manifest, provenance)
    assert saved.get("run_identity_sha256") == expected, (
        f"{path} belongs to a different manifest, checkpoint, runtime, or limit; "
        "refusing to mix evaluation runs")
    saved_rows = saved.get("rows") or []
    assert saved_rows, f"{path} contains no completed rows"
    required = {"index", "prompt_sha256", "prediction", "reference"}
    assert all(required <= set(row) for row in saved_rows), f"{path} has an incomplete row"
    if expected_rows is not None:
        expected = []
        for row in expected_rows:
            if track == "continuation":
                split = continuation_split(row["input"])
                if split is None:
                    continue
                prompt, reference = split
            else:
                assert track == "instruction", f"unknown track {track}"
                prompt, reference = render_instruction(row, system_prompt)
            expected.append({"index": row["index"], "prompt_sha256": sha256_text(prompt),
                             "reference": reference})
        observed = [{key: row[key] for key in ("index", "prompt_sha256", "reference")}
                    for row in saved_rows]
        assert observed == expected, (
            f"{path} rows do not match the currently rendered track; refusing stale, "
            "reordered, or corrupted partial output")
    return {
        "result": {
            "track": track,
            "indices": [row["index"] for row in saved_rows],
            "prompt_sha256": [row["prompt_sha256"] for row in saved_rows],
            "predictions": [row["prediction"] for row in saved_rows],
            "references": [row["reference"] for row in saved_rows],
        },
        "metrics": saved["metrics"],
        "path": path,
    }


def teacher_forced_perplexity(model, tokenizer, prompts, references, max_length=2048,
                              indices=None):
    """Perplexity of the reference given the prompt, with the reference protected.

    The harness in `cpt/inference.py` builds `prefix + " " + gt`, tokenizes the prefix
    separately to find its length, and truncates the joined text from the RIGHT at 2048.
    Three problems follow, and all three are silent:

    1. Tokenizing the prefix alone can give a different length than the same text inside
       the joined string, because merges cross the boundary. The label mask is then off by
       a token or two and scores the wrong positions.
    2. Right-truncation discards the END of the text - which is the reference. A long
       prompt can leave the reference partly or wholly absent while still producing a
       number.
    3. If the prompt alone exceeds `max_length`, every label is masked, `num_gt_tokens` is
       zero, and the aggregate divides by zero or silently absorbs the row.

    So: tokenize the two pieces separately and concatenate ids, truncate the PROMPT from
    the left so the reference always survives whole, and refuse rather than score a row
    with no supervised token. Rows that cannot be scored are returned, not dropped.
    """
    import torch

    total_loss, total_tokens, scored, skipped = 0.0, 0, 0, []
    for position, (prompt, reference) in enumerate(zip(prompts, references)):
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        reference_ids = tokenizer(reference, add_special_tokens=False)["input_ids"]
        if not reference_ids:
            skipped.append({"position": position, "index": indices[position] if indices else None,
                            "reason": "reference tokenizes to nothing"})
            continue
        if len(reference_ids) >= max_length:
            # Clipping here would contradict the whole-reference contract this function
            # exists to provide, and would do it silently. Refuse the row instead.
            skipped.append({"position": position, "index": indices[position] if indices else None,
                            "reason": f"reference of {len(reference_ids)} tokens does not fit "
                                      f"within max_length {max_length}",
                            "reference_tokens": len(reference_ids)})
            continue
        room = max_length - len(reference_ids)
        kept_prompt = prompt_ids[-room:] if room > 0 else []
        if not kept_prompt:
            skipped.append({"position": position, "index": indices[position] if indices else None,
                            "reason": f"reference of {len(reference_ids)} tokens leaves no room "
                                      f"for a prompt within {max_length}"})
            continue

        input_ids = torch.tensor([kept_prompt + reference_ids], device=model.device)
        labels = input_ids.clone()
        labels[0, : len(kept_prompt)] = -100
        supervised = int((labels != -100).sum())
        assert supervised == len(reference_ids), "label mask does not cover exactly the reference"

        with torch.no_grad():
            outputs = model(input_ids=input_ids, labels=labels)
        total_loss += float(outputs.loss) * supervised
        total_tokens += supervised
        scored += 1

    assert total_tokens > 0, f"no row produced a supervised token; skipped={skipped[:3]}"
    import math as _math
    return {
        "perplexity": _math.exp(total_loss / total_tokens),
        "scored_rows": scored,
        "supervised_tokens": total_tokens,
        "skipped": skipped,
    }


def describe_checkpoint(model_path, parent=None, adapter_path=None):
    """Immutable provenance for one evaluated artefact.

    `write_outputs` refuses a bare label, because a directory named "sft" tells a later
    reader nothing about which adapter, which parent, or which revision produced the
    numbers. Hashes are over the weight files actually loaded.
    """
    model_path = Path(model_path)
    weights = sorted(
        [p for p in model_path.rglob("*.safetensors")] or
        [p for p in model_path.rglob("*.bin")])
    assert weights, f"no weight files under {model_path}"
    # Per-file CONVENTIONAL sha256, so an expected digest can be a plain
    # `sha256sum model.safetensors` - which is what every other record in this project
    # holds, and what anyone would supply. An earlier version compared only against an
    # aggregate that folded filenames in with the bytes, so the project's own recorded
    # constant 22e6df5b... was rejected by the very check meant to confirm it.
    per_file = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in weights}
    aggregate = hashlib.sha256()
    for path in weights:
        aggregate.update(path.name.encode())
        aggregate.update(path.read_bytes())
    provenance = {
        "checkpoint_label": model_path.name,
        "model_path": str(model_path),
        "weight_sha256": per_file,
        # Kept for multi-file models, and labelled so nobody mistakes it for a file digest.
        "model_sha256_aggregate": aggregate.hexdigest(),
        "model_sha256_aggregate_algorithm": "sha256 over (filename bytes || file bytes) "
                                            "for each weight file, sorted by name",
        "weight_files": [p.name for p in weights],
        "parent": parent or "unspecified",
        "adapter_sha256": None,
    }
    if adapter_path is not None:
        adapter = Path(adapter_path)
        files = sorted(adapter.rglob("adapter_model.safetensors"))
        assert files, f"no adapter weights under {adapter}"
        provenance["adapter_sha256"] = hashlib.sha256(files[0].read_bytes()).hexdigest()
        provenance["adapter_path"] = str(adapter)
        config = adapter / "adapter_config.json"
        if config.exists():
            # The adapter records the base it was trained against. Carrying it forward is
            # what makes "which SFT is this" answerable later, and a bare adapter is not a
            # merged base: whatever consumes this must merge against THIS parent.
            provenance["adapter_declared_base"] = json.loads(
                config.read_text()).get("base_model_name_or_path")
    return provenance


def _serialise_config(config):
    """Render a quantization config whatever shape the loader left it in.

    Installed Unsloth writes `model.config.quantization_config` as a **dict** for 4-bit
    loading (`unsloth/models/loader.py`), while transformers uses a config object. An
    earlier version called `vars(config)`, which raises `TypeError: vars() argument must
    have __dict__ attribute` on a dict - and it runs after the model has loaded and before
    generation, so a GPU run would have crashed having already paid for the load. The test
    that missed it supplied a SimpleNamespace: the shape it was written against rather than
    the shape it would meet.
    """
    if config is None:
        return None

    def scalar(value):
        return value if isinstance(value, (str, int, float, bool, type(None))) else str(value)

    if isinstance(config, dict):
        return {str(k): scalar(v) for k, v in config.items()}
    for method in ("to_dict", "to_diff_dict"):
        converter = getattr(config, method, None)
        if callable(converter):
            try:
                return {str(k): scalar(v) for k, v in converter().items()}
            except Exception:
                pass
    try:
        return {k: scalar(v) for k, v in vars(config).items()}
    except TypeError:
        return {"repr": str(config)}


def describe_runtime(model, tokenizer):
    """What the weights were actually run AS. Hashes alone do not identify that.

    The GPU loader defaults to 4-bit while the CPU path runs float32, so the same weight
    file evaluated on each produces a different model in every way that matters to a
    metric. Recording the digest and calling the artefact identified would be false.
    """
    runtime = {}
    try:
        parameters = list(model.parameters())
        runtime["dtype"] = str(parameters[0].dtype) if parameters else None
        runtime["device"] = str(parameters[0].device) if parameters else None
    except Exception as exc:
        runtime["dtype_error"] = f"{type(exc).__name__}: {exc}"
    config = getattr(getattr(model, "config", None), "quantization_config", None)
    runtime["quantization"] = _serialise_config(config)
    for attribute in ("name_or_path", "torch_dtype"):
        value = getattr(getattr(model, "config", None), attribute, None)
        if value is not None:
            runtime[f"config_{attribute}"] = str(value)
    try:
        vocabulary = tokenizer.get_vocab()
        runtime["tokenizer_vocab_size"] = len(vocabulary)
        runtime["tokenizer_vocab_sha256"] = hashlib.sha256(
            json.dumps(sorted(vocabulary.items()), sort_keys=True).encode()).hexdigest()
        runtime["tokenizer_class"] = type(tokenizer).__name__
        runtime["padding_side"] = getattr(tokenizer, "padding_side", None)
    except Exception as exc:
        runtime["tokenizer_error"] = f"{type(exc).__name__}: {exc}"
    try:
        runtime["special_token_ids"] = {
            "bos": getattr(tokenizer, "bos_token_id", None),
            "eos": getattr(tokenizer, "eos_token_id", None),
            "pad": getattr(tokenizer, "pad_token_id", None),
            **{marker: tokenizer.convert_tokens_to_ids(marker) for marker in CHATML_MARKERS},
        }
    except Exception as exc:
        runtime["special_token_ids_error"] = f"{type(exc).__name__}: {exc}"
    runtime.update(loader_identity(model))
    runtime.update(tokenizer_identity(tokenizer))
    return runtime


def loader_identity(model):
    """Name the loader that produced this model, rather than inferring it downstream.

    The CUDA route goes through Unsloth and the CPU route through plain Transformers, and
    a bounded GPU check exists to prove the first one was taken. Device and 4-bit metadata
    only imply it; a class, a module and an installed version say it. Recording this costs
    nothing and makes the archived output answer the question without the kernel log.
    """
    import sys
    identity = {
        "model_class": type(model).__name__,
        "model_module": type(model).__module__,
        "unsloth_imported": "unsloth" in sys.modules,
    }
    try:
        import importlib.metadata
        identity["unsloth_version"] = importlib.metadata.version("unsloth")
    except Exception as exc:
        identity["unsloth_version"] = None
        identity["unsloth_version_error"] = f"{type(exc).__name__}: {exc}"
    return identity


def tokenizer_identity(tokenizer):
    """Hash what the tokenizer IS, not just the strings it knows.

    Two tokenizers can share a vocabulary and still segment text differently: merges
    decide how words split, the normalizer decides what reaches the merges, and the
    special-token map decides whether the assistant marker survives as one token. A
    vocabulary digest matching across checkpoints is therefore evidence, not proof, and
    the comparison this battery exists to make is only meaningful if the tokenizer is
    held fixed. Serialising the tokenizer and hashing every file it writes captures all
    of that, and does it without assuming the source directory is still on disk -- the
    GPU path is handed a tokenizer that Unsloth may have rebuilt in memory.
    """
    identity = {}
    scratch = tempfile.mkdtemp(prefix="battery-tokenizer-")
    try:
        tokenizer.save_pretrained(scratch)
        files = {}
        for path in sorted(Path(scratch).rglob("*")):
            if path.is_file():
                files[str(path.relative_to(scratch))] = hashlib.sha256(
                    path.read_bytes()).hexdigest()
        identity["tokenizer_files_sha256"] = files
        identity["tokenizer_serialisation_sha256"] = hashlib.sha256(
            json.dumps(files, sort_keys=True).encode()).hexdigest()
    except Exception as exc:
        identity["tokenizer_serialisation_error"] = f"{type(exc).__name__}: {exc}"
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    return identity


def reserve_output_dir(directory, overwrite=False):
    """Refuse a completed evaluation BEFORE generating, not after.

    Refusing at write time means a repeat run spends the entire battery and is turned away
    at the end, and an interruption loses everything unsaved. Claim the directory first.
    """
    directory = Path(directory)
    completed = directory / "metrics.json"
    if completed.exists() and not overwrite:
        raise AssertionError(
            f"{completed} already exists; refusing before spending the run. Write to a new "
            "directory, or pass --overwrite deliberately.")
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def verify_parent(provenance, expected_parent_sha256=None):
    """Compare the loaded parent's bytes against a trusted digest, or say it was not checked.

    The previous version returned `checked: True` having compared nothing - it asserted the
    declared base string was non-empty and reported success. A mismatched digest passed. A
    check that cannot fail is not a check, and recording it as one is worse than recording
    no check at all, because a later reader will believe it.

    `expected_parent_sha256` must come from an independent record - the driver constant, a
    manifest, a published hash - not from the artefact being verified.
    """
    declared = provenance.get("adapter_declared_base")
    per_file = provenance.get("weight_sha256") or {}
    aggregate = provenance.get("model_sha256_aggregate")
    if expected_parent_sha256 is None:
        return {"checked": False, "reason": "no expected parent digest supplied",
                "declared_base": declared, "weight_sha256": per_file}
    if not per_file:
        return {"checked": False, "reason": "no loaded-parent digests to compare",
                "declared_base": declared}

    # A conventional per-file sha256 is what every record in this project holds and what a
    # human would compute. The aggregate is accepted too, but named, so a match says which
    # it was rather than leaving the reader to guess.
    matched_files = [name for name, digest in per_file.items()
                     if digest == expected_parent_sha256]
    matched_aggregate = (aggregate == expected_parent_sha256)
    result = {"checked": True,
              "matched": bool(matched_files or matched_aggregate),
              "matched_via": ("weight_file" if matched_files
                              else "aggregate" if matched_aggregate else None),
              "matched_files": matched_files,
              "declared_base": declared,
              "weight_sha256": per_file,
              "expected_parent_sha256": expected_parent_sha256}
    assert result["matched"], (
        f"expected parent {expected_parent_sha256} matches neither any weight file "
        f"{per_file} nor the aggregate {aggregate}; refusing to evaluate an adapter "
        "against the wrong base")
    return result


def main(argv=None):
    """Run the pinned battery against one checkpoint. Both tracks, one directory out."""
    import argparse

    parser = argparse.ArgumentParser(description="Run the fixed evaluation battery.")
    parser.add_argument("--manifest", required=True, help="pinned battery-manifest.json")
    parser.add_argument("--test-source", required=True, help="test.jsonl the manifest pins")
    parser.add_argument("--sft-entrypoint", required=True,
                        help="instruction_tuning/sft.py, read for SYSTEM_PROMPT")
    parser.add_argument("--model-path", required=True,
                        help="Model directory. May be an adapter IF --base-model-id is given.")
    parser.add_argument("--base-model-id", default=None,
                        help="Base for an adapter. REQUIRED when --model-path is an adapter: "
                             "cpt/inference.py otherwise defaults to HuggingFaceTB/SmolLM-135M, "
                             "the original base, and would silently evaluate the wrong model.")
    parser.add_argument("--adapter-path", default=None,
                        help="optional adapter, recorded and hashed for provenance")
    parser.add_argument("--parent", default=None, help="what this model was built from")
    parser.add_argument("--expected-parent-sha256", default=None,
                        help="trusted digest of the parent weights, from an independent "
                             "record - a plain `sha256sum <weight file>` is what this "
                             "expects. Supplied, the run refuses a mismatched base; "
                             "omitted, provenance records that no automated check ran.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bertscore-model-path", required=True,
                        help="local snapshot of the manifest-pinned BERTScore model; its "
                             "revision directory name and required file hashes are verified "
                             "before checkpoint loading")
    parser.add_argument("--overwrite", action="store_true",
                        help="replace a completed evaluation in --output-dir")
    parser.add_argument("--resume-partials", action="store_true",
                        help="reuse completed partial-<track>.json files only after their "
                             "manifest, checkpoint, runtime and limit identity verifies")
    parser.add_argument("--tracks", default="continuation,instruction")
    parser.add_argument("--limit", type=int, default=0,
                        help="take only the first N rows of the SOURCE battery, before "
                             "per-track filtering. The continuation track may therefore "
                             "yield fewer than N rows, because short passages are excluded. "
                             "For a bounded smoke run before the full battery.")
    args = parser.parse_args(argv)

    manifest = json.loads(Path(args.manifest).read_text())
    validate_manifest(manifest)          # the pinned settings govern; refuse drift
    bertscore_snapshot = verify_bertscore_snapshot(args.bertscore_model_path, manifest)
    rows = load_rows(manifest, args.test_source)
    system_prompt = extract_system_prompt(Path(args.sft_entrypoint))
    assert sha256_text(system_prompt) == manifest["system_prompt_sha256"], (
        "the system prompt differs from the one the battery was pinned with; track "
        "'instruction' would be a different task")

    # An adapter without its true base is the wrong model, and the wrong model still
    # produces plausible numbers. Refuse rather than default.
    is_adapter = (Path(args.model_path) / "adapter_config.json").exists()
    if is_adapter:
        assert args.base_model_id, (
            f"{args.model_path} is an adapter; pass --base-model-id explicitly. The loader "
            "defaults to the original base, which is not this adapter's parent.")
        declared = json.loads(
            (Path(args.model_path) / "adapter_config.json").read_text()
        ).get("base_model_name_or_path")
        print(f"adapter declares base: {declared!r}; loading against {args.base_model_id!r}",
              flush=True)

    reserve_output_dir(args.output_dir, args.overwrite)   # before any generation

    provenance = describe_checkpoint(
        args.base_model_id if is_adapter else args.model_path,
        args.parent, args.model_path if is_adapter else args.adapter_path)
    provenance["checkpoint_label"] = Path(args.model_path).name
    provenance["is_adapter"] = is_adapter
    provenance["base_model_id"] = args.base_model_id
    provenance["parent_check"] = verify_parent(provenance, args.expected_parent_sha256)
    provenance["limit"] = args.limit or None

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cpt"))
    from inference import load_model, generate_batch            # left padding handled there
    from evals import calculate_metrics, metric_runtime_identity

    model, tokenizer = (load_model(args.model_path, base_model_id=args.base_model_id)
                        if args.base_model_id else load_model(args.model_path))
    provenance["runtime"] = describe_runtime(model, tokenizer)
    provenance["metric_runtime"] = metric_runtime_identity(
        args.bertscore_model_path, manifest["bertscore_num_layers"])
    provenance["bertscore_snapshot"] = bertscore_snapshot
    provenance["tokenizer_check"] = verify_tokenizer_markers(tokenizer, args.model_path)
    print(f"runtime: {json.dumps(provenance['runtime'], sort_keys=True, default=str)}", flush=True)
    results, metrics = {}, {}
    for track in [t.strip() for t in args.tracks.split(",") if t.strip()]:
        selected = rows[: args.limit] if args.limit else rows
        resumed = (load_track_partial(args.output_dir, track, manifest, provenance,
                                      selected, system_prompt)
                   if args.resume_partials else None)
        if resumed is not None:
            results[track] = resumed["result"]
            metrics[track] = resumed["metrics"]
            print(f"{track}: resumed verified partial {resumed['path']}", flush=True)
            continue
        result = run_track(model, tokenizer, selected, track, system_prompt,
                           generate_batch,
                           teacher_forced_perplexity)
        results[track] = result
        metrics[track] = score_track(
            result,
            lambda predictions, references: calculate_metrics(
                predictions, references,
                bertscore_model=args.bertscore_model_path,
                bertscore_num_layers=manifest["bertscore_num_layers"]))
        partial = write_track_partial(args.output_dir, track, manifest, provenance, result,
                                      metrics[track])
        print(f"{track}: {json.dumps(metrics[track], sort_keys=True)}", flush=True)
        print(f"  partial saved: {partial}", flush=True)

    directory = write_outputs(args.output_dir, manifest, provenance, results, metrics,
                              overwrite=True)   # the reservation above already decided this
    for track in results:                       # consolidated write succeeded; partials served their purpose
        (Path(args.output_dir) / f"partial-{track}.json").unlink(missing_ok=True)
    print(f"written: {directory}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
