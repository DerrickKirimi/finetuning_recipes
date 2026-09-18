"""The parts of preference training that need neither Unsloth nor a GPU.

`train_preference.py` must import Unsloth first and cannot run on a CPU, so everything that decides what is trained, on
which rows, for how long, and what gets recorded lives here. CPU tests drive a real TRL DPOTrainer through these same
functions, so what they prove is what production runs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import time
from pathlib import Path
from typing import Any

try:
    from preference_optimization.chat_formatting import normalize_explicit_preference_example
    from preference_optimization.split_manifest import SplitError, select_split, sha256_file, verify_split_files
    from preference_optimization.training_controls import validate_segment_args
except ModuleNotFoundError:
    from chat_formatting import normalize_explicit_preference_example
    from split_manifest import SplitError, select_split, sha256_file, verify_split_files
    from training_controls import validate_segment_args

SEED = 3407
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Preference tuning (DPO / ORPO) with Unsloth.")
    p.add_argument("--base_model_id", type=str, default="paperbd/smollm_135M_neuraltxt_v1")
    p.add_argument("--output_model_id", "-o", type=str, default="preference_tuned")
    p.add_argument("--dataset", "-d", type=str, default="paperbd/paper_preference_150K-v1")
    p.add_argument("--method", type=str, choices=["dpo", "orpo"], default="dpo", help="Preference optimization method.")
    p.add_argument("--max_seq_length", type=int, default=2048)
    p.add_argument("--batch_size", "-bs", type=int, default=32)
    p.add_argument("--eval_batch_size", type=int, default=None, help="Defaults to --batch_size.")
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--epochs", "-e", type=int, default=3)
    p.add_argument("--lora_r", type=int, default=32)
    p.add_argument("--learning_rate", "-lr", type=float, default=2e-4)
    p.add_argument("--beta", type=float, default=0.1, help="DPO/ORPO beta hyperparameter.")
    p.add_argument("--max_prompt_length", type=int, default=1536)
    p.add_argument("--dataset_revision", type=str, default=None, help="Pin the preference dataset to this revision.")
    p.add_argument("--dataloader_num_workers", type=int, default=8)
    p.add_argument("--dataset_num_proc", type=int, default=None)

    data = p.add_argument_group("exact data identity")
    data.add_argument("--dataset_file", type=str, default=None,
                      help="Local JSONL used instead of --dataset. Bound by sha256: the split manifest's source hash, or "
                           "--dataset_sha256 without a manifest.")
    data.add_argument("--dataset_sha256", type=str, default=None)
    data.add_argument("--split_manifest", type=str, default=None,
                      help="Grouped split manifest (train/validation row indices); replaces the random 2%% row split.")
    data.add_argument("--split_manifest_sha256", type=str, default=None)
    data.add_argument("--split_spotcheck", type=str, default=None,
                      help="(index, prompt key) pairs verifying the manifest matches the loaded dataset.")
    data.add_argument("--split_spotcheck_sha256", type=str, default=None)
    data.add_argument("--train_rows_limit", type=int, default=None,
                      help="Keep only the first N selected training rows (diagnostics and smoke runs, never production).")
    data.add_argument("--validation_rows_limit", type=int, default=None,
                      help="Keep only the first N selected validation rows (diagnostics and smoke runs).")

    seg = p.add_argument_group("bounded, resumable segments (opt-in)")
    seg.add_argument("--output_dir", type=str, default=None, help="Defaults to models/<output_model_id>.")
    seg.add_argument("--max_steps", type=int, default=-1,
                     help="Full-schedule horizon in optimizer updates. Required for segments; fixes the LR schedule.")
    seg.add_argument("--stop_after_steps", type=int, default=None,
                     help="End this segment after this many applied updates (global step), saving full state there.")
    seg.add_argument("--resume_from_checkpoint", type=str, default=None,
                     help="checkpoint-<step> directory to continue. Missing or incomplete state fails; never starts fresh.")
    seg.add_argument("--stop_by_unix", type=float, default=None,
                     help="Wall-clock deadline: stop and save after the first update that ends past it.")
    seg.add_argument("--save_steps", type=int, default=50)
    seg.add_argument("--eval_steps", type=int, default=50)
    seg.add_argument("--logging_steps", type=int, default=10)
    seg.add_argument("--save_total_limit", type=int, default=3)
    seg.add_argument("--early_stopping_patience", type=int, default=3)

    rec = p.add_argument_group("evidence (opt-in)")
    rec.add_argument("--run_record", type=str, default=None, help="JSON: inputs, precision, horizon, stop reason, probes.")
    rec.add_argument("--telemetry", type=str, default=None, help="JSONL: step timings, memory, logs, evals, saves.")
    rec.add_argument("--batch_fingerprints", type=str, default=None, help="JSONL: sha256 of every collated row, in order.")
    rec.add_argument("--reference_probe_rows", type=int, default=0,
                     help="Validation rows whose reference and policy log-probabilities are recorded before and after.")
    rec.add_argument("--expected_marker_ids", type=str, default=None,
                     help='JSON map, e.g. {"<|im_end|>": 0}; checked after the chat template is applied.')
    rec.add_argument("--require_template_unchanged", action="store_true",
                     help="Fail if get_chat_template changes how a probe conversation tokenizes.")
    rec.add_argument("--fail_fast_checks", action="store_true",
                     help="DPO: before training, fail if the reference topology, logging/callback settings or the "
                          "reference probe are not what a resumable pilot requires, instead of finding out afterwards.")
    rec.add_argument("--reference_lifecycle_diagnostic", choices=["train_update", "wrapper_only"], default=None,
                     help="DPO: instead of training, probe the adapter-disabled reference across lifecycle transitions "
                          "(train_update: --stop_after_steps real updates, at most 5, via trainer.train(); wrapper_only: "
                          "Accelerate's model preparation with no update) and write the probes to --diagnostic_record.")
    rec.add_argument("--diagnostic_record", type=str, default=None)
    rec.add_argument("--reference_guard_every", type=int, default=None,
                     help="DPO: probe the adapter-disabled reference inside training (train begin, after the first update, "
                          "every N updates, train end) and stop with state saved if any probe differs bitwise from the "
                          "train-begin probe or the frozen-state digest changes.")
    return p


def check_args(args: argparse.Namespace) -> None:
    """Reject inconsistent requests before anything expensive happens."""
    if args.split_manifest:
        missing = [name for name in ("split_spotcheck", "split_manifest_sha256", "split_spotcheck_sha256", "dataset_file")
                   if not getattr(args, name)]
        if missing:
            raise SystemExit(f"--split_manifest requires {', '.join('--' + m for m in missing)}")
    elif args.split_spotcheck:
        raise SystemExit("--split_spotcheck without --split_manifest")
    if args.dataset_file and not args.split_manifest and not args.dataset_sha256:
        raise SystemExit("--dataset_file without a split manifest requires --dataset_sha256")
    if args.max_steps > 0 or args.stop_after_steps is not None or args.resume_from_checkpoint:
        validate_segment_args(max_steps=args.max_steps, stop_after_steps=args.stop_after_steps,
                              save_steps=args.save_steps, eval_steps=args.eval_steps)
    if args.save_steps % args.eval_steps:
        raise SystemExit("load_best_model_at_end needs save_steps to be a multiple of eval_steps")
    for name in ("train_rows_limit", "validation_rows_limit"):
        if getattr(args, name) is not None and getattr(args, name) <= 0:
            raise SystemExit(f"--{name} must be positive")
    if args.reference_lifecycle_diagnostic:
        if args.method != "dpo" or not args.reference_probe_rows or not args.diagnostic_record:
            raise SystemExit("--reference_lifecycle_diagnostic needs --method dpo, --reference_probe_rows and --diagnostic_record")
        if args.resume_from_checkpoint:
            raise SystemExit("--reference_lifecycle_diagnostic starts from a fresh adapter; do not resume")
        if args.reference_lifecycle_diagnostic == "train_update" and not (args.stop_after_steps and args.stop_after_steps <= 5):
            # A diagnostic, not training: a few updates. With linear warmup the first update runs at learning rate 0.
            raise SystemExit("--reference_lifecycle_diagnostic train_update needs --stop_after_steps between 1 and 5")
    elif args.diagnostic_record:
        raise SystemExit("--diagnostic_record without --reference_lifecycle_diagnostic")
    if args.reference_guard_every is not None:
        if args.method != "dpo" or not args.reference_probe_rows or args.reference_guard_every <= 0:
            raise SystemExit("--reference_guard_every needs --method dpo, --reference_probe_rows and a positive interval")


def load_preference_datasets(args: argparse.Namespace):
    """Return (train, validation, record). With a split manifest, every identity check runs before rows are selected."""
    from datasets import load_dataset

    record: dict[str, Any] = {}
    verified = None
    if args.split_manifest:
        verified = verify_split_files(args.split_manifest, args.split_spotcheck,
                                      manifest_sha256=args.split_manifest_sha256,
                                      spotcheck_sha256=args.split_spotcheck_sha256, dataset_file=args.dataset_file)
        record["split"] = {"manifest_sha256": args.split_manifest_sha256, "spotcheck_sha256": args.split_spotcheck_sha256,
                           "spotcheck_rows": len(verified[1])}
    if args.dataset_file:
        if not args.split_manifest:
            actual = sha256_file(args.dataset_file)
            if actual != args.dataset_sha256:
                raise SplitError(f"dataset file sha256 {actual[:12]} != expected {args.dataset_sha256[:12]}")
        dataset = load_dataset("json", data_files=args.dataset_file, split="train")
        record["dataset"] = {"file": str(args.dataset_file), "sha256": args.dataset_sha256 or
                             verified[0]["sources"]["dpo_train"]["sha256"], "rows": len(dataset)}
    else:
        dataset = load_dataset(args.dataset, split="train", revision=args.dataset_revision)
        record["dataset"] = {"hub_id": args.dataset, "revision": args.dataset_revision, "rows": len(dataset)}
    dataset = dataset.map(normalize_explicit_preference_example,
                          remove_columns=[c for c in dataset.column_names if c not in {"prompt", "chosen", "rejected"}])
    if verified:
        train_dataset, val_dataset = select_split(dataset, *verified)
    else:
        split = dataset.train_test_split(test_size=0.02, seed=SEED)
        train_dataset, val_dataset = split["train"], split["test"]
    if args.train_rows_limit or args.validation_rows_limit:
        record["rows_before_limits"] = {"train": len(train_dataset), "validation": len(val_dataset)}
        if args.train_rows_limit:
            train_dataset = train_dataset.select(range(min(args.train_rows_limit, len(train_dataset))))
        if args.validation_rows_limit:
            val_dataset = val_dataset.select(range(min(args.validation_rows_limit, len(val_dataset))))
    record["rows"] = {"train": len(train_dataset), "validation": len(val_dataset)}
    return train_dataset, val_dataset, record


def epoch_equivalent_max_steps(train_rows: int, batch_size: int, grad_accum: int, epochs: float) -> int:
    """The horizon the installed Trainer derives from num_train_epochs (dataloader keeps the last partial batch)."""
    batches = math.ceil(train_rows / batch_size)
    per_epoch = max(batches // grad_accum + int(batches % grad_accum > 0), 1)
    return math.ceil(epochs * per_epoch)


def preference_config_kwargs(args: argparse.Namespace, precision: dict, output_dir: str, **overrides) -> dict:
    kwargs = dict(
        output_dir=output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size or args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        warmup_ratio=0.03,
        warmup_steps=5,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        dataloader_num_workers=args.dataloader_num_workers,
        optim="adamw_8bit",
        weight_decay=0.001,
        lr_scheduler_type="linear",
        report_to="none",
        seed=SEED,
        fp16=precision["fp16"],
        bf16=precision["bf16"],
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        beta=args.beta,
        max_length=args.max_seq_length,
        max_prompt_length=args.max_prompt_length,
        dataset_num_proc=args.dataset_num_proc,
        # Transformers' default replaces a nan/inf step loss with the running mean before logging, so a finite logged
        # loss would not prove a finite raw loss.
        logging_nan_inf_filter=False,
        # Default False: a resumed segment would restart early-stopping patience and other callback state.
        restore_callback_states_from_checkpoint=True,
    )
    kwargs.update(overrides)
    return kwargs


def marker_report(tokenizer, expected: dict[str, int] | None) -> dict:
    ids = {token: tokenizer.convert_tokens_to_ids(token) for token in ("<|endoftext|>", "<|im_start|>", "<|im_end|>")}
    report = {"marker_ids": ids, "eos_token": tokenizer.eos_token, "eos_token_id": tokenizer.eos_token_id,
              "pad_token": tokenizer.pad_token, "pad_token_id": tokenizer.pad_token_id}
    if expected is not None:
        actual = {k: tokenizer.convert_tokens_to_ids(k) for k in expected}
        wrong = {k: (actual[k], v) for k, v in expected.items() if actual[k] != v}
        if wrong:
            raise SystemExit(f"chat marker ids differ from expected (actual, expected): {wrong}")
        report["expected_marker_ids_match"] = True
    return report


PROBE_CONVERSATION = [{"role": "system", "content": "You are a careful assistant."},
                      {"role": "user", "content": "Summarise the passage.\n\nA short passage."},
                      {"role": "assistant", "content": "A summary."}]


def template_fingerprint(tokenizer) -> dict:
    ids = tokenizer.apply_chat_template(PROBE_CONVERSATION, tokenize=True)
    prompt = tokenizer.apply_chat_template(PROBE_CONVERSATION[:2], tokenize=False, add_generation_prompt=True)
    return {"ids_sha256": hashlib.sha256(json.dumps(list(ids)).encode()).hexdigest(), "tokens": len(ids),
            "generation_prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()}


def row_fingerprint(example: dict) -> str:
    payload = [list(example["prompt_input_ids"]), list(example["chosen_input_ids"]), list(example["rejected_input_ids"])]
    return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode()).hexdigest()


class FingerprintingCollator:
    """Records a fingerprint of every row the trainer collates, in order, then defers to the real collator.

    Training and evaluation share one collator; their rows are told apart afterwards by content, since the splits share
    no row. Batches skipped on resume are never collated, so a resumed run's first record is its true data position.
    """

    def __init__(self, inner, path: str | Path):
        self.inner, self.path, self.calls = inner, Path(path), 0

    def __call__(self, examples):
        with self.path.open("a") as handle:
            handle.write(json.dumps({"call": self.calls, "rows": [row_fingerprint(e) for e in examples]}) + "\n")
        self.calls += 1
        return self.inner(examples)


def expected_sampler_order(rows: int, seed: int, epoch: int = 0) -> list[int]:
    """Row order of Accelerate's seedable random sampler for one epoch (checked against the real trainer in tests)."""
    import torch

    generator = torch.Generator()
    generator.manual_seed(seed + epoch)
    return torch.randperm(rows, generator=generator).tolist()


def collated_fingerprints(path: str | Path) -> list[str]:
    return [row for line in Path(path).read_text().splitlines() if line.strip() for row in json.loads(line)["rows"]]


def update_microbatch_windows(train_rows: int, batch_size: int, grad_accum: int, first_step: int,
                             last_step: int) -> list[tuple[int, int, int]]:
    """(epoch, first microbatch, end microbatch) consumed by each update first_step+1..last_step.

    Mirrors the installed Trainer: an epoch has ceil(rows / batch_size) microbatches and ceil(microbatches / grad_accum)
    updates, the last of which takes only the leftover microbatches. Resuming skips step % updates_per_epoch whole
    windows inside the current epoch. Verified against a real trainer across an epoch boundary with a short window.
    """
    batches = math.ceil(train_rows / batch_size)
    per_epoch = max(batches // grad_accum + int(batches % grad_accum > 0), 1)
    windows = []
    for update in range(first_step + 1, last_step + 1):
        epoch, index = divmod(update - 1, per_epoch)
        windows.append((epoch, index * grad_accum, min((index + 1) * grad_accum, batches)))
    return windows


def check_data_position(train_dataset, eval_dataset, fingerprints_path: str | Path, *, seed: int, first_step: int,
                        last_step: int, batch_size: int, grad_accum: int) -> dict:
    """Did this segment train exactly the rows the sampler puts at updates first_step+1..last_step, in any epoch?

    Each epoch's order is Accelerate's seeded permutation for that epoch. At most one extra microbatch may be collated
    but not trained (the dataloader fetches one batch ahead inside an epoch); it must be the next expected rows.
    """
    rows = len(train_dataset)
    windows = update_microbatch_windows(rows, batch_size, grad_accum, first_step, last_step)
    orders, cache = {}, {}

    def fingerprint(index):
        if index not in cache:
            cache[index] = row_fingerprint(train_dataset[index])
        return cache[index]

    def microbatch(epoch, number):
        if epoch not in orders:
            orders[epoch] = expected_sampler_order(rows, seed, epoch)
        return [fingerprint(i) for i in orders[epoch][number * batch_size:(number + 1) * batch_size]]

    expected = [fp for epoch, begin, end in windows for number in range(begin, end) for fp in microbatch(epoch, number)]
    batches = math.ceil(rows / batch_size)
    lookahead_expected = []
    if windows:
        epoch, _, end = windows[-1]
        if end < batches:
            lookahead_expected = microbatch(epoch, end)
    held_out = {row_fingerprint(row) for row in eval_dataset}
    observed = [fp for fp in collated_fingerprints(fingerprints_path) if fp not in held_out]
    trained, lookahead = observed[:len(expected)], observed[len(expected):]
    mismatch = next((i for i, (a, b) in enumerate(zip(trained, expected)) if a != b), None)
    return {
        "checked": True, "seed": seed, "first_step": first_step, "last_step": last_step,
        "epochs_spanned": sorted({w[0] for w in windows}), "trained_rows_expected": len(expected),
        "train_rows_collated": len(observed), "first_mismatch": mismatch,
        "trained_match": len(trained) == len(expected) and mismatch is None,
        "lookahead_rows": len(lookahead),
        "lookahead_ok": len(lookahead) <= batch_size and lookahead == lookahead_expected[:len(lookahead)],
        "expected_first_row": expected[0] if expected else None,
    }


def _rss() -> dict:
    fields = {}
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith(("VmRSS:", "VmHWM:")):
                name, value = line.split(":", 1)
                fields[name] = int(value.split()[0]) * 1024
    except OSError:
        pass
    return fields


def _digest_tensors(tensors) -> str:
    digest = hashlib.sha256()
    for name, tensor in tensors:
        digest.update(name.encode())
        digest.update(tensor.detach().float().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def adapter_digest(model) -> str:
    return _digest_tensors((n, p) for n, p in sorted(model.named_parameters()) if "lora_" in n)


def base_digest(model) -> str:
    """Frozen (non-adapter) weights. Quantized storage is hashed as stored."""
    import torch

    digest = hashlib.sha256()
    for name, param in sorted(model.named_parameters()):
        if "lora_" in name:
            continue
        tensor = param.detach().cpu().contiguous()
        if tensor.dtype == torch.bfloat16:       # numpy has no bfloat16
            tensor = tensor.float()
        digest.update(name.encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _update_value(digest, value) -> None:
    """Hash tensors by dtype, shape and every byte; recurse into containers. repr() is used only for scalars, because
    the repr of a large tensor is truncated and would hide a change in its middle."""
    import torch

    if isinstance(value, torch.dtype | torch.device):
        digest.update(f"{type(value).__name__}:{value}".encode())
    elif isinstance(value, torch.Size):
        _update_value(digest, tuple(value))
    elif isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        digest.update(f"T{tensor.dtype}{tuple(tensor.shape)}".encode())
        if tensor.dtype == torch.bfloat16:      # numpy has no bfloat16
            tensor = tensor.float()
        digest.update(tensor.numpy().tobytes())
    elif isinstance(value, dict):
        digest.update(f"D{len(value)}".encode())
        for key in sorted(value, key=repr):
            digest.update(f"{key!r}=".encode())
            _update_value(digest, value[key])
    elif isinstance(value, (list, tuple)):
        digest.update(f"L{len(value)}".encode())
        for item in value:
            _update_value(digest, item)
    elif value is None or isinstance(value, (bool, int, float, str)):
        digest.update(repr(value).encode())
    else:
        raise TypeError(f"optimizer state holds an unhashable {type(value).__name__}; extend the digest explicitly")


QUANT_STATE_FIELDS = (
    "absmax", "shape", "code", "dtype", "blocksize", "quant_type", "offset",
    # bitsandbytes branches on both fields in the 4-bit forward/dequantization path.  `state2` normally implies
    # `nested`, but hashing only state2 would miss an inconsistent or mutated flag.  CPU packing also changes whether
    # Linear4bit passes the stored weight or its transpose to matmul_4bit.
    "nested", "packing_format_for_cpu",
)
MODULE_QUANT_ATTRIBUTES = ("compute_dtype", "compute_type_is_set", "quant_type", "compress_statistics", "quant_storage",
                           "blocksize")
ADAPTER_TOPOLOGY_ATTRIBUTES = ("scaling", "merged_adapters", "_disable_adapters", "active_adapter", "use_dora")


def _hash_quant_state(digest, quant_state, depth: int = 0) -> None:
    """Every field a bitsandbytes 4-bit QuantState dequantizes with, including the nested (double-quant) state."""
    if quant_state is None:
        digest.update(b"quant_state=None")
        return
    if depth > 4:
        raise ValueError("quant_state nesting deeper than bitsandbytes produces")
    for field in QUANT_STATE_FIELDS:
        digest.update(f"{depth}.{field}=".encode())
        _update_value(digest, getattr(quant_state, field, None))
    digest.update(f"{depth}.state2=".encode())
    _hash_quant_state(digest, getattr(quant_state, "state2", None), depth + 1)


def frozen_state_digest(model) -> dict:
    """What the adapter-disabled reference forward reads, by component, so a claimed-fixed reference names its evidence.

    base_digest covers named parameters only. A 4-bit layer also dequantizes with QuantState tensors that are not
    parameters, rotary caches are non-persistent buffers, the 4-bit compute dtype is a module attribute set on the first
    forward, and LoRA scaling/merge state decides whether disabling the adapter really yields the base model.
    """
    components, counts = {}, {}

    def finish(name, digest, count):
        components[name], counts[name] = digest.hexdigest(), count

    digest, count = hashlib.sha256(), 0
    for name, param in sorted(model.named_parameters()):
        if "lora_" in name:
            continue
        digest.update(name.encode())
        _update_value(digest, param.data)
        count += 1
    finish("parameters", digest, count)

    digest, count = hashlib.sha256(), 0
    for name, buffer in sorted(model.named_buffers()):
        if "lora_" in name:
            continue
        digest.update(name.encode())
        _update_value(digest, buffer)
        count += 1
    finish("buffers", digest, count)

    digest, count = hashlib.sha256(), 0
    for name, param in sorted(model.named_parameters()):
        quant_state = getattr(param, "quant_state", None)
        if quant_state is not None:
            digest.update(name.encode())
            _hash_quant_state(digest, quant_state)
            count += 1
    finish("quant_state", digest, count)

    quant_digest, adapter_digest_, quant_count, adapter_count = hashlib.sha256(), hashlib.sha256(), 0, 0
    for name, module in sorted(model.named_modules()):
        for attribute in MODULE_QUANT_ATTRIBUTES:
            if attribute in vars(module):
                quant_digest.update(f"{name}.{attribute}=".encode())
                _update_value(quant_digest, vars(module)[attribute])
                quant_count += 1
        for attribute in ADAPTER_TOPOLOGY_ATTRIBUTES:
            if attribute in vars(module):
                value = vars(module)[attribute]
                adapter_digest_.update(f"{name}.{attribute}=".encode())
                _update_value(adapter_digest_, sorted(value) if isinstance(value, set) else value)
                adapter_count += 1
    finish("module_attributes", quant_digest, quant_count)
    finish("adapter_topology", adapter_digest_, adapter_count)
    combined = hashlib.sha256(json.dumps(components, sort_keys=True).encode()).hexdigest()
    return {"sha256": combined, "components": components, "counts": counts}


def _state_view(state: dict) -> dict:
    """bitsandbytes' saved state nests its 8-bit buffers under __bnb_optimizer_quant_state__ while the live optimizer holds
    them flat; hash one layout so a restored optimizer compares equal to the one that was saved."""
    view = {k: v for k, v in state.items() if k != "__bnb_optimizer_quant_state__"}
    nested = state.get("__bnb_optimizer_quant_state__")
    if isinstance(nested, dict):
        for key, value in nested.items():
            if key in view:
                raise ValueError(f"optimizer state has {key} both flat and nested")
            view[key] = value
    return view


def optimizer_digest(optimizer) -> dict:
    """Every state entry of every parameter, plus each group's settings, in parameter-group order.

    Parameter-group order is the one ordering shared by a live optimizer and one rebuilt by load_state_dict, so equal
    digests mean the whole state was restored, not a sample of it.
    """
    inner = getattr(optimizer, "optimizer", optimizer)
    digest = hashlib.sha256()
    steps, param_states, params = set(), 0, 0
    for group_index, group in enumerate(inner.param_groups):
        for key in sorted(k for k in group if k != "params"):
            digest.update(f"group{group_index}.{key}=".encode())
            _update_value(digest, group[key])
        for param_index, param in enumerate(group["params"]):
            params += 1
            state = inner.state.get(param)
            if not state:
                digest.update(f"param{group_index}.{param_index}:none".encode())
                continue
            param_states += 1
            view = _state_view(state)
            # Hash the normalized view's shape: bitsandbytes serializes its flat live buffers under one nested key,
            # and the digest must compare equal before and after load_state_dict.
            digest.update(f"param{group_index}.{param_index}:{len(view)}".encode())
            for key in sorted(view):
                digest.update(f"{key}=".encode())
                _update_value(digest, view[key])
            if "step" in state:
                steps.add(int(float(state["step"])))
    return {"params": params, "param_states": param_states, "step_values": sorted(steps), "sha256": digest.hexdigest()}


def rng_digest() -> dict:
    import random

    import numpy as np
    import torch

    record = {"python": hashlib.sha256(repr(random.getstate()).encode()).hexdigest(),
              "numpy": hashlib.sha256(repr(np.random.get_state()).encode()).hexdigest(),
              "torch_cpu": hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()}
    if torch.cuda.is_available():
        record["torch_cuda"] = hashlib.sha256(b"".join(s.numpy().tobytes() for s in torch.cuda.get_rng_state_all())).hexdigest()
    return record


def scaler_state(trainer) -> dict | None:
    """The fp16 GradScaler state (scale, growth tracker, ...), or None when no scaler is active."""
    scaler = getattr(getattr(trainer, "accelerator", None), "scaler", None) if trainer is not None else None
    state = scaler.state_dict() if scaler is not None else {}
    return {k: float(v) if hasattr(v, "item") else v for k, v in state.items()} or None


def callback_states(trainer) -> dict:
    """Restorable callback counters that decide when a run stops."""
    if trainer is None:
        return {}
    return {type(c).__name__: c.early_stopping_patience_counter for c in trainer.callback_handler.callbacks
            if hasattr(c, "early_stopping_patience_counter")}


def make_telemetry_callback(path: str | Path, stop_record: dict, trainer=None):
    """JSONL evidence: timings, memory, logs, evaluations, saves, and the state needed to prove a resume was faithful.

    `trainer` gives access to what callbacks are not handed: the GradScaler and the other callbacks' state.
    """
    import torch
    from transformers import TrainerCallback

    path = Path(path)

    def emit(event: str, **fields):
        fields.update(event=event, unix=time.time())
        with path.open("a") as handle:
            handle.write(json.dumps(fields, default=str) + "\n")

    def memory() -> dict:
        record = _rss()
        if torch.cuda.is_available():
            record.update(cuda_max_allocated=torch.cuda.max_memory_allocated(),
                          cuda_max_reserved=torch.cuda.max_memory_reserved())
            torch.cuda.reset_peak_memory_stats()
        return record

    class TelemetryCallback(TrainerCallback):
        last = None
        first_step_seen = False

        def on_train_begin(self, args, state, control, model=None, optimizer=None, lr_scheduler=None, **kwargs):
            self.last = time.monotonic()
            emit("train_begin", global_step=state.global_step, max_steps=state.max_steps,
                 torch_initial_seed=torch.random.initial_seed(),
                 scheduler_last_epoch=getattr(lr_scheduler, "last_epoch", None),
                 learning_rate=lr_scheduler.get_last_lr() if lr_scheduler is not None else None,
                 optimizer=optimizer_digest(optimizer) if optimizer is not None else None,
                 scaler=scaler_state(trainer), callback_states=callback_states(trainer),
                 adapter_sha256=adapter_digest(model) if model is not None else None, memory=memory())

        def on_step_begin(self, args, state, control, **kwargs):
            if not self.first_step_seen:
                self.first_step_seen = True
                emit("first_step_begin", global_step=state.global_step, rng=rng_digest())

        def on_step_end(self, args, state, control, optimizer=None, **kwargs):
            now = time.monotonic()
            # global_step advances even when a fp16 overflow skips the optimizer step; only this says it was applied.
            skipped = getattr(optimizer, "step_was_skipped", None) if optimizer is not None else None
            emit("step_end", global_step=state.global_step, seconds=now - self.last, memory=memory(),
                 optimizer_step_skipped=None if skipped is None else bool(skipped))
            self.last = now

        def on_log(self, args, state, control, logs=None, **kwargs):
            emit("log", global_step=state.global_step, logs=logs)

        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            emit("evaluate", global_step=state.global_step, metrics=metrics, memory=memory())
            self.last = time.monotonic()      # evaluation time is reported by eval_runtime, not charged to a step

        def on_save(self, args, state, control, model=None, optimizer=None, lr_scheduler=None, **kwargs):
            checkpoint = Path(args.output_dir) / f"checkpoint-{state.global_step}"
            size = sum(p.stat().st_size for p in checkpoint.rglob("*") if p.is_file()) if checkpoint.is_dir() else None
            emit("save", global_step=state.global_step, checkpoint=str(checkpoint), bytes=size,
                 scheduler_last_epoch=getattr(lr_scheduler, "last_epoch", None),
                 learning_rate=lr_scheduler.get_last_lr() if lr_scheduler is not None else None,
                 optimizer=optimizer_digest(optimizer) if optimizer is not None else None,
                 scaler=scaler_state(trainer), callback_states=callback_states(trainer),
                 adapter_sha256=adapter_digest(model) if model is not None else None, rng=rng_digest())
            self.last = time.monotonic()

        def on_train_end(self, args, state, control, **kwargs):
            emit("train_end", global_step=state.global_step, stop=dict(stop_record), memory=memory())

    return TelemetryCallback()


def make_deadline_callback(stop_by_unix: float, record: dict):
    from transformers import TrainerCallback

    class DeadlineCallback(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if time.time() >= stop_by_unix:
                control.should_training_stop = True
                control.should_save = True
                record["deadline_fired"] = True
                record["deadline_global_step"] = int(state.global_step)
            return control

    return DeadlineCallback()


def _model_chain(model):
    chain, m = [model], model
    while hasattr(m, "model") and m.model is not m and len(chain) < 16:
        m = m.model
        chain.append(m)
    return chain


def _training_mode_digest(model) -> dict:
    digest = hashlib.sha256()
    count = 0
    for name, module in model.named_modules():
        digest.update(f"{name}={bool(module.training)}".encode())
        count += 1
    return {"modules": count, "sha256": digest.hexdigest()}


def _gradient_state_digest(model) -> dict:
    digest = hashlib.sha256()
    params = present = 0
    for name, param in sorted(model.named_parameters()):
        digest.update(name.encode())
        params += 1
        if param.grad is None:
            digest.update(b"=None")
        else:
            digest.update(b"=gradient")
            _update_value(digest, param.grad)
            present += 1
    return {"params": params, "gradients_present": present, "sha256": digest.hexdigest()}


def reference_probe(trainer, rows: int, label: str | None = None) -> dict:
    """Policy and reference (adapter-disabled) log-probabilities for the first validation rows, no gradient.

    Besides the values it records what decides whether two probes are comparable: the dtype of every returned tensor
    and of the model's logits, the completion token counts, whether Accelerate has wrapped the forward (its fp16
    preparation converts outputs to float32), the Unsloth generation flag, and the full frozen-state digest.
    """
    import torch

    collator = getattr(trainer.data_collator, "inner", trainer.data_collator)
    examples = [trainer.eval_dataset[i] for i in range(rows)]
    batch = trainer._prepare_inputs(collator(examples))
    model = trainer.model
    training_modes = [(module, bool(module.training)) for module in model.modules()]
    rng_before = rng_digest()
    gradients_before = _gradient_state_digest(model)
    training_before = _training_mode_digest(model)
    logits_dtypes = []

    def record_logits(module, inputs, output):
        logits = getattr(output, "logits", None)
        if logits is None and isinstance(output, (tuple, list)) and output:
            logits = output[0]
        logits_dtypes.append(str(getattr(logits, "dtype", None)))

    context = {"label": label, "model_training_before": bool(model.training),
               "forward_wrapped_by_accelerate": hasattr(model, "_original_forward"),
               "unsloth_generation_flag": any(hasattr(m, "_flag_for_generation") for m in _model_chain(model)),
               "accelerator_mixed_precision": str(getattr(getattr(trainer, "accelerator", None), "mixed_precision", None))}
    handle = model.register_forward_hook(record_logits)
    model.eval()
    try:
        with torch.no_grad(), trainer.compute_loss_context_manager():
            policy = trainer.concatenated_forward(model, batch)
            ref_chosen, ref_rejected = trainer.compute_ref_log_probs(batch)
    finally:
        handle.remove()
        # `model.train(previous_top_level_value)` would overwrite intentionally heterogeneous submodule modes.
        model.train(training_modes[0][1])
        for module, was_training in training_modes:
            module.training = was_training
    context["forward_logits_dtypes"] = logits_dtypes
    rng_after = rng_digest()
    gradients_after = _gradient_state_digest(model)
    training_after = _training_mode_digest(model)
    context["side_effects"] = {
        "rng_unchanged": rng_before == rng_after,
        "gradients_unchanged": gradients_before == gradients_after,
        "training_modes_unchanged": training_before == training_after,
    }
    as_list = lambda t: [float(x) for x in t.detach().float().cpu()]  # noqa: E731
    tokens = lambda key: [int(x) for x in batch[key].sum(-1).cpu()]  # noqa: E731
    return {"rows": rows, "policy_chosen_logps": as_list(policy["chosen_logps"]),
            "policy_rejected_logps": as_list(policy["rejected_logps"]),
            "reference_chosen_logps": as_list(ref_chosen), "reference_rejected_logps": as_list(ref_rejected),
            "output_dtypes": {"policy_chosen": str(policy["chosen_logps"].dtype),
                              "policy_rejected": str(policy["rejected_logps"].dtype),
                              "reference_chosen": str(ref_chosen.dtype), "reference_rejected": str(ref_rejected.dtype)},
            "completion_tokens": {"chosen": tokens("chosen_attention_mask"), "rejected": tokens("rejected_attention_mask")},
            "prompt_tokens": tokens("prompt_attention_mask"),
            "context": context,
            "adapter_sha256": adapter_digest(model), "base_sha256": base_digest(model),
            "frozen_state": frozen_state_digest(model)}


def float16_nearest(value: float) -> float:
    """IEEE-754 round-to-nearest-even into binary16, the conversion a float32 -> float16 cast performs."""
    return struct.unpack("<e", struct.pack("<e", value))[0]


def compare_reference_probes(before: dict, after: dict) -> dict:
    """Is the adapter-disabled reference the same function of the same frozen state in both probes? Zero tolerance.

    * The frozen-state digest must be identical in every component, and the completion token counts equal.
    * Reference values returned in the same dtype must be bitwise equal.
    * A float16 value may be compared with a float32 value only by the IEEE-754 identity float16(float32 value) ==
      float16 value: the float16 result is the float32 result after the one representation change, and nothing else.
    Any other dtype pairing is not comparable and fails. No numeric tolerance is involved anywhere.
    """
    problems, modes, differences = [], {}, []
    state_a, state_b = before.get("frozen_state") or {}, after.get("frozen_state") or {}
    components_a, components_b = state_a.get("components") or {}, state_b.get("components") or {}
    differing = sorted(k for k in set(components_a) | set(components_b) if components_a.get(k) != components_b.get(k))
    if not state_a.get("sha256") or not components_a or differing or state_a.get("sha256") != state_b.get("sha256"):
        problems.append(f"frozen state missing or different: {differing or 'missing'}")
    tokens_a, tokens_b = before.get("completion_tokens"), after.get("completion_tokens")
    if not tokens_a or tokens_a != tokens_b:
        problems.append("completion token counts missing or different; the probes did not score the same sequences")
    for side in ("chosen", "rejected"):
        key = f"reference_{side}_logps"
        values_a, values_b = before.get(key), after.get(key)
        dtype_a = (before.get("output_dtypes") or {}).get(f"reference_{side}")
        dtype_b = (after.get("output_dtypes") or {}).get(f"reference_{side}")
        valid = (isinstance(values_a, list) and isinstance(values_b, list) and values_a and len(values_a) == len(values_b)
                 and all(isinstance(v, float) and math.isfinite(v) for v in values_a + values_b))
        if not valid:
            problems.append(f"{key}: missing, empty, unequal length or non-finite")
            modes[side] = "invalid"
            continue
        differences.extend(abs(a - b) for a, b in zip(values_a, values_b))
        if dtype_a is None or dtype_b is None:
            problems.append(f"{key}: output dtype not recorded")
            modes[side] = "invalid"
        elif dtype_a == dtype_b:
            modes[side] = "exact"
            if values_a != values_b:
                problems.append(f"{key}: same dtype {dtype_a} but values differ")
        elif {dtype_a, dtype_b} == {"torch.float16", "torch.float32"}:
            modes[side] = "float16_rounding_identity"
            half, full = (values_a, values_b) if dtype_a == "torch.float16" else (values_b, values_a)
            if any(float16_nearest(h) != h for h in half):
                problems.append(f"{key}: a float16-labelled value is not representable in float16")
            elif any(float16_nearest(f) != h for h, f in zip(half, full)):
                problems.append(f"{key}: float16 values are not the float16 rounding of the float32 values")
        else:
            modes[side] = "not_comparable"
            problems.append(f"{key}: dtypes {dtype_a} and {dtype_b} are not comparable")
    return {"ok": not problems, "problems": problems, "modes": modes, "frozen_state_components_differing": differing,
            "max_abs_difference": max(differences) if differences else None}


LIFECYCLE_COMPARISONS = {
    "train_update": [("fresh", "fresh_repeat"), ("after_train_update", "after_train_update_repeat"),
                     ("fresh", "after_train_update"), ("after_train_update", "after_for_inference"),
                     ("after_train_update", "after_for_training")],
    "wrapper_only": [("fresh", "after_for_inference"), ("fresh", "after_for_training"),
                     ("fresh", "after_accelerate_prepare"),
                     ("after_accelerate_prepare", "after_accelerate_prepare_repeat")],
}


def reference_lifecycle(trainer, rows: int, variant: str, *, for_inference=None, for_training=None) -> dict:
    """Probe the reference across lifecycle transitions, in one process, with nothing else changed.

    train_update: fresh, repeat, the trainer's --stop_after_steps real updates through trainer.train() (Accelerate
    preparation, Unsloth's train wrapper, optimizer steps), repeat, then explicit inference and training modes. Under
    linear warmup the first update has learning rate 0, so two updates are needed for the adapter to change.
    wrapper_only: fresh, explicit inference and training modes, then only Accelerate's prepare_model (no optimizer, no
    update), repeated. If this reproduces the post-update reference, the update is not what changed it.
    """
    if variant not in LIFECYCLE_COMPARISONS:
        raise ValueError(f"unknown lifecycle variant {variant!r}")
    stages = []

    def probe(label):
        stages.append({"label": label, **reference_probe(trainer, rows, label=label)})

    extra = {}
    if variant == "train_update":
        probe("fresh")
        probe("fresh_repeat")
        trainer.train()
        extra["global_step_after_train"] = int(trainer.state.global_step)
        probe("after_train_update")
        probe("after_train_update_repeat")
        if for_inference is not None:
            for_inference(trainer.model)
            probe("after_for_inference")
        if for_training is not None:
            for_training(trainer.model)
            probe("after_for_training")
    else:
        probe("fresh")
        if for_inference is not None:
            for_inference(trainer.model)
            probe("after_for_inference")
        if for_training is not None:
            for_training(trainer.model)
            probe("after_for_training")
        trainer.accelerator.prepare_model(trainer.model)
        probe("after_accelerate_prepare")
        probe("after_accelerate_prepare_repeat")
    by_label = {stage["label"]: stage for stage in stages}
    comparisons = {f"{a}__{b}": compare_reference_probes(by_label[a], by_label[b])
                   for a, b in LIFECYCLE_COMPARISONS[variant] if a in by_label and b in by_label}
    return {"variant": variant, "rows": rows, "stages": stages, "comparisons": comparisons, **extra}


def make_reference_guard_callback(trainer, rows: int, every_steps: int, record: dict, stop_record: dict):
    """Same-process, post-preparation, zero-tolerance reference guard for real training.

    The train-begin probe runs after Accelerate has prepared the model (and after any checkpoint load), which is the
    lifecycle point the reference-probe diagnostic showed to be stable. Every later probe (first update, every
    `every_steps` updates, train end) must equal it in `exact` mode with an identical frozen-state digest. The first
    mismatch stops training with a forced save, so a changed reference costs minutes and keeps the state.
    """
    from transformers import TrainerCallback

    record.update(probes=[], comparisons=[], failed=False, first_failure=None, every_steps=every_steps, rows=rows)

    def take(label, state):
        probe = reference_probe(trainer, rows, label=label)
        probe["label"] = label
        probe["global_step"] = int(state.global_step)
        record["probes"].append(probe)
        side_effects = (probe.get("context") or {}).get("side_effects") or {}
        side_effects_ok = side_effects and all(value is True for value in side_effects.values())
        if len(record["probes"]) == 1:
            exact = bool(side_effects_ok)
            record["baseline_side_effects"] = side_effects
            if not exact:
                record["failed"], record["first_failure"] = True, label
                stop_record["reference_guard_fired"] = True
                stop_record["reference_guard_global_step"] = probe["global_step"]
            return exact
        result = compare_reference_probes(record["probes"][0], probe)
        exact = bool(side_effects_ok and result["ok"] and set(result["modes"].values()) == {"exact"})
        problems = list(result["problems"])
        if not side_effects_ok:
            problems.append(f"reference probe changed process state: {side_effects}")
        record["comparisons"].append({"label": label, "global_step": probe["global_step"], "ok": exact,
                                      "modes": result["modes"], "problems": problems, "side_effects": side_effects,
                                      "frozen_state_components_differing": result["frozen_state_components_differing"],
                                      "max_abs_difference": result["max_abs_difference"]})
        if not exact and not record["failed"]:
            record["failed"], record["first_failure"] = True, label
            stop_record["reference_guard_fired"] = True
            stop_record["reference_guard_global_step"] = probe["global_step"]
        return exact

    class ReferenceGuardCallback(TrainerCallback):
        first_update_seen = False

        def on_train_begin(self, args, state, control, **kwargs):
            if not take("train_begin", state):
                control.should_training_stop = True
                control.should_save = True
            return control

        def on_step_end(self, args, state, control, **kwargs):
            label = None
            if not self.first_update_seen:
                self.first_update_seen = True
                label = "first_update"
            elif state.global_step % every_steps == 0:
                label = f"step_{state.global_step}"
            if label is not None and not take(label, state):
                control.should_training_stop = True
                control.should_save = True
            return control

        def on_train_end(self, args, state, control, **kwargs):
            take("train_end", state)

    return ReferenceGuardCallback()


def trainer_record(trainer) -> dict:
    """Facts read from the constructed trainer, not restated from the arguments passed to it."""
    model = trainer.model
    return {
        "class": type(trainer).__name__,
        "ref_model_is_none": getattr(trainer, "ref_model", "missing") is None,
        "is_peft_model": getattr(trainer, "is_peft_model", None),
        "reference_free": getattr(trainer, "reference_free", None),
        "precompute_ref_log_probs": getattr(trainer.args, "precompute_ref_log_probs", None),
        "logging_nan_inf_filter": trainer.args.logging_nan_inf_filter,
        "restore_callback_states_from_checkpoint": trainer.args.restore_callback_states_from_checkpoint,
        "model_accepts_loss_kwargs": getattr(trainer, "model_accepts_loss_kwargs", None),
        "gradient_accumulation_steps": trainer.args.gradient_accumulation_steps,
        # Transformers 4.x accumulates itself and leaves Accelerate at 1; anything else would divide the loss twice.
        "accelerator_gradient_accumulation_steps": getattr(getattr(trainer, "accelerator", None),
                                                           "gradient_accumulation_steps", None),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "trainable_non_adapter": sorted(n for n, p in model.named_parameters() if p.requires_grad and "lora_" not in n),
    }


PREFLIGHT_TRAINER = {"ref_model_is_none": True, "is_peft_model": True, "reference_free": False,
                     "precompute_ref_log_probs": False, "logging_nan_inf_filter": False,
                     "restore_callback_states_from_checkpoint": True, "model_accepts_loss_kwargs": False}
PROBE_ARRAYS = ("policy_chosen_logps", "policy_rejected_logps", "reference_chosen_logps", "reference_rejected_logps")


def preflight_problems(record: dict, args: argparse.Namespace) -> list[str]:
    """What a DPO pilot's post-run checks would reject but can already be known before the first update."""
    if args.method != "dpo":
        return ["--fail_fast_checks is defined for DPO only"]
    problems = []
    trainer = record.get("trainer") or {}
    for key, expected in PREFLIGHT_TRAINER.items():
        if trainer.get(key, "missing") is not expected:
            problems.append(f"trainer.{key} is {trainer.get(key, 'missing')!r}, expected {expected!r}")
    if trainer.get("accelerator_gradient_accumulation_steps") != 1:
        problems.append(f"accelerator gradient_accumulation_steps is {trainer.get('accelerator_gradient_accumulation_steps')!r}, "
                        "expected 1 (the Trainer accumulates)")
    if trainer.get("gradient_accumulation_steps") != args.grad_accum:
        problems.append(f"trainer gradient_accumulation_steps is {trainer.get('gradient_accumulation_steps')!r}, "
                        f"expected requested {args.grad_accum}")
    if trainer.get("trainable_non_adapter") != []:
        problems.append(f"trainable parameters outside the adapter: {trainer.get('trainable_non_adapter')!r}")
    tokenizer = record.get("tokenizer") or {}
    if args.expected_marker_ids and tokenizer.get("expected_marker_ids_match") is not True:
        problems.append("chat marker ids not confirmed")
    if args.require_template_unchanged and tokenizer.get("template_unchanged") is not True:
        problems.append("chat template change not ruled out")
    if args.reference_probe_rows:
        probe = record.get("probe_before")
        if not isinstance(probe, dict):
            problems.append("reference probe missing")
        else:
            for key in PROBE_ARRAYS:
                values = probe.get(key)
                if not (isinstance(values, list) and len(values) == args.reference_probe_rows
                        and all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in values)):
                    problems.append(f"probe {key} is not {args.reference_probe_rows} finite numbers: {values!r}")
            for key in ("adapter_sha256", "base_sha256"):
                value = probe.get(key)
                if not (isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)):
                    problems.append(f"probe {key} is not a sha256 hex digest")
    return problems


def write_json(path: str | Path, payload: dict) -> None:
    target = Path(path)
    staging = target.with_name(target.name + ".partial")
    staging.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    staging.replace(target)
