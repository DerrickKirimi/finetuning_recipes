"""Measure the DPO memory envelope, one candidate batch size per process.

Why one process per candidate
-----------------------------
An in-process sweep cannot be trusted. Wrapping `trainer.training_step` keeps a bound
method alive that references the previous trainer, so the old model is still resident when
the next candidate is built, and a *smaller* batch can OOM merely because it is loading
beside its predecessor. That inverts the result the probe exists to produce. This module
therefore runs exactly one candidate and exits; the caller runs it once per size.

What it measures, and what it does not
--------------------------------------
Memory is driven by the longest pair the trainer will ever see, so stress rows are the
longest in the split rather than the first. DPO puts chosen AND rejected through the model,
so the vocabulary-sized logits buffer behind the SFT step-52 OOM is entered twice per
example: SFT's batch 16 is an upper bound here, not a starting point.

Fit is judged on *optimizer steps*, not `training_step` calls. Those are microbatches, and
with gradient accumulation a run can issue many without completing a single update. A
candidate counts as fitting only when optimizer steps actually advanced, the loss stayed
finite, and an optimizer exists with state.

The `--with-evaluation` phase covers the hazard that actually killed the SFT run: the first
update *after* a validation pass, in the same process.

Construction mirrors `preference_optimization/train_preference.py` exactly, including
`ref_model=None`, which makes TRL use adapter disabling rather than a second model copy.
Pass a MERGED parent, as production does; loading an adapter and disabling it would make
the reference the pre-adapter base and measure the wrong topology.

Stress is measured at the model's own input, not at the collator
------------------------------------------------------------------
TRL concatenates prompt and completion and truncates AFTER the collator, so collated shapes
bound the model input rather than measure it. Every training forward is therefore recorded
from two hooks - the top-level model call (whose attention mask gives each row's real
length) and the input-embedding layer (whose columns are the longest row after TRL's
flush_left) - and a phase reaches the stress target only on grad-enabled forwards actually
observed in that phase. Reference forwards (adapters disabled, no grad) and evaluation
forwards are recorded but never credited. If no forward is observed the verdict fails
closed. The collator's pre-truncation numbers are kept as diagnostics only.

Substituted rows make any weights here diagnostic. They must never enter the DPO lineage.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

SEED = 3407


def json_safe(value):
    """Replace nonfinite floats with tagged strings so a NaN finding can be REPORTED.

    `emit` serialises with `allow_nan=False`, deliberately: a bare NaN in JSON is not valid
    and downstream readers differ on it. But the verdict carries the observed loss lists, so
    a detected NaN made its own report unserialisable - `ValueError: Out of range float
    values are not JSON compliant: nan` - and no candidate_result was written at all. The
    probe detected the failure and then lost it.
    """
    import math
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
        return value
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def select_stress_pairs(dataset, count, tokenizer, max_length, max_prompt_length):
    """Return the `count` rows with the largest post-truncation concatenated footprint.

    Scores what the concatenated forward actually materialises: prompt (capped at
    `max_prompt_length`) plus the longer of the two completions, capped at `max_length`.
    Summing chosen+rejected would overstate rows with two long completions and understate
    one very long completion, which is the shape that actually binds.
    """
    scored = []
    for index, row in enumerate(dataset):
        prompt, chosen, rejected = row.get("prompt"), row.get("chosen"), row.get("rejected")
        if chosen is None or rejected is None:
            continue
        def n_tokens(value):
            if value is None:
                return 0
            if isinstance(value, str):
                return len(tokenizer(value, add_special_tokens=False)["input_ids"])
            return len(tokenizer.apply_chat_template(value, tokenize=True))
        n_p = min(n_tokens(prompt), max_prompt_length)
        n_c, n_r = n_tokens(chosen), n_tokens(rejected)
        footprint = min(n_p + max(n_c, n_r), max_length)
        scored.append((footprint, index, row, {"prompt": n_p, "chosen": n_c, "rejected": n_r}))
    if len(scored) < count:
        raise AssertionError(f"need {count} usable preference rows, found {len(scored)}")
    scored.sort(key=lambda item: item[0], reverse=True)
    top = scored[:count]
    return ([row for _, _, row, _ in top],
            [index for _, index, _, _ in top],
            [{"footprint": f, **lens} for f, _, _, lens in top])


def memory_snapshot():
    import torch

    free, total = torch.cuda.mem_get_info()
    return {
        "allocated": torch.cuda.memory_allocated(),
        "reserved": torch.cuda.memory_reserved(),
        "peak_allocated": torch.cuda.max_memory_allocated(),
        "peak_reserved": torch.cuda.max_memory_reserved(),
        "device_free": free,
        "device_total": total,
    }


class BoundedProbe:
    """Counts optimizer steps, records memory per step, and stops at the limit.

    Built as a plain class and adapted to TrainerCallback in `attach` so the module can be
    unit-tested without transformers installed.
    """

    def __init__(self, emit, max_optimizer_steps, post_eval_steps=0, stress_target_tokens=0):
        self.emit = emit
        self.max_optimizer_steps = max_optimizer_steps
        self.post_eval_steps = post_eval_steps
        # The workload a stress pass must actually carry. Zero disables the gate, which is
        # only appropriate for unit tests, never for a reported candidate.
        self.stress_target_tokens = stress_target_tokens
        self.optimizer_steps = 0          # attempts: on_step_end fires even when skipped
        self.applied_steps = 0            # attempts that actually updated weights
        self.unknown_steps = 0
        self.post_eval_applied_steps = 0
        self.in_evaluation = False
        self.microbatches = 0
        self.evaluated = False
        self.post_eval_optimizer_steps = 0
        self.losses = []
        self.eval_losses = []
        self.shapes = []
        self.forwards = []                # observed model inputs, see forward_record

    def current_phase(self):
        """Training, evaluation, or training after an evaluation has completed."""
        if self.in_evaluation:
            return "evaluation"
        return "post_evaluation" if self.evaluated else "training"

    def on_substep(self):
        self.microbatches += 1

    def on_optimizer_step(self, memory=None, skipped=None):
        """`skipped` is the trainer's own FP16 flag; None means it was unavailable.

        `on_step_end` fires whether or not the optimizer actually stepped: with FP16 the
        grad scaler skips an update on a non-finite gradient. Counting attempts as updates
        reports a candidate trained when it has not.
        """
        self.optimizer_steps += 1
        applied = (skipped is False)
        if skipped is None:
            self.unknown_steps += 1
        if applied:
            self.applied_steps += 1
        if self.evaluated:
            self.post_eval_optimizer_steps += 1
            if applied:
                self.post_eval_applied_steps += 1
        self.emit("optimizer_step", optimizer_steps=self.optimizer_steps,
                  applied_steps=self.applied_steps, skipped=skipped,
                  microbatches=self.microbatches, after_evaluation=self.evaluated,
                  memory=memory)
        return self.should_stop()

    def on_evaluation(self, memory=None):
        self.evaluated = True
        self.emit("evaluation_complete", optimizer_steps=self.optimizer_steps, memory=memory)

    def should_stop(self):
        if self.applied_steps < self.max_optimizer_steps:
            return False
        if self.post_eval_steps and self.post_eval_applied_steps < self.post_eval_steps:
            return False
        return True

    def verdict(self):
        """A candidate fits only if it demonstrably trained, not merely if it did not raise."""
        observed = self.losses + self.eval_losses
        finite = all(math.isfinite(value) for value in observed) if self.losses else False
        nonfinite = [v for v in observed if not math.isfinite(v)]
        # Gate EACH phase separately. A global maximum lets a 2048-token training batch
        # carry a 32-token post-evaluation batch over the line - and the post-evaluation
        # phase is the one that actually matters, being what killed the SFT run.
        required_phases = ["training"] + (["post_evaluation"] if self.post_eval_steps else [])
        # Supervision is judged once over every TRAINING collation, not per phase. The
        # collator's phase tag is unreliable by one batch: Accelerate's DataLoaderShard
        # fetches the next batch before yielding the current one, so the first batch trained
        # after an evaluation was collated before it and is tagged "training". Per-phase
        # supervision therefore found zero post-evaluation collations in a real TRL run while
        # the model received the full post-evaluation workload. Supervision is a property of
        # the rows, which repeat across windows; stress, which is phase-sensitive, is taken
        # from forwards, whose phase is observed at the moment the model runs.
        training_collations = [r for r in self.shapes if r.get("phase") != "evaluation"]
        supervised = bool(training_collations) and all(
            r.get("all_rows_supervised", False) for r in training_collations)
        per_phase = {}
        for phase in required_phases:
            records = [r for r in self.shapes if r.get("phase") == phase]
            rows = [r.get("max_row_pre_truncation_tokens", 0) for r in records]
            # Only grad-enabled forwards in this phase carry the training workload. The
            # reference pass (adapters disabled under no_grad) and evaluation forwards use
            # the same model and would otherwise be credited.
            forwards = [f for f in self.forwards
                        if f.get("phase") == phase and f.get("grad_enabled")]
            calls = [f for f in forwards if f.get("source") == "model_call"]
            embeddings = [f for f in forwards if f.get("source") == "input_embeddings"]
            # Do not let padded embedding columns overrule the attention mask. Both
            # observation points must fire and agree, in forward order, in each phase.
            agreed = bool(calls) and len(calls) == len(embeddings) and all(
                call.get("basis") == "attention_mask"
                and call.get("rows") == embedding.get("rows")
                and call.get("columns") == embedding.get("columns")
                and call.get("max_row_tokens") == embedding.get("max_row_tokens")
                for call, embedding in zip(calls, embeddings))
            forward_tokens = [f.get("max_row_tokens", 0) for f in calls]
            per_phase[phase] = {
                "collations": len(records),      # tagged at collation; may lag by one batch
                "max_row_pre_truncation_tokens": max(rows) if rows else 0,
                "all_rows_supervised": supervised,
                "training_forwards": len(forwards),
                "hook_measurements_agree": agreed,
                "model_calls": len(calls),
                "embedding_calls": len(embeddings),
                "max_row_forward_tokens": max(forward_tokens) if forward_tokens else 0,
                # Measured at the model input. The pre-truncation bound above can exceed
                # what the model processed, so it no longer decides this.
                "reached_target": bool(agreed
                                       and max(forward_tokens) >= self.stress_target_tokens
                                       and supervised),
            }
        stress = {
            "collations_recorded": len(self.shapes),
            "phases_seen": sorted({r.get("phase") for r in self.shapes if r.get("phase")}),
            "per_phase": per_phase,
            "target_tokens": self.stress_target_tokens,
            "forwards_recorded": len(self.forwards),
            "measured_at": "model forward input; grad-enabled forwards in each phase only",
        }
        # A candidate that did not carry the intended workload has not demonstrated a
        # worst case. Report that as inconclusive rather than as a comfortable envelope.
        #
        # A zero target DISABLES the gate, for unit tests of the other criteria only. The
        # CLI always supplies one (defaulting to --max-seq-length), so a reported candidate
        # is always gated; `gated` records which applied so no reader has to infer it.
        gated = bool(self.stress_target_tokens)
        stress["gated"] = gated
        stress["reached_target"] = (not gated) or all(
            per_phase[phase]["reached_target"] for phase in required_phases)
        return {
            "stress": stress,
            "training_losses": json_safe(list(self.losses)),
            "eval_losses": json_safe(list(self.eval_losses)),
            "nonfinite_observations": len(nonfinite),
            "optimizer_steps": self.optimizer_steps,            # attempts
            "applied_steps": self.applied_steps,                # attempts that updated
            "skipped_steps": self.optimizer_steps - self.applied_steps - self.unknown_steps,
            "unknown_steps": self.unknown_steps,
            "microbatches": self.microbatches,
            "post_eval_optimizer_steps": self.post_eval_optimizer_steps,
            "post_eval_applied_steps": self.post_eval_applied_steps,
            "losses_finite": finite,
            "evaluated": self.evaluated,
            "fit": (self.applied_steps >= self.max_optimizer_steps
                    and finite
                    and stress["reached_target"]
                    and (not self.post_eval_steps
                         or self.post_eval_applied_steps >= self.post_eval_steps)),
        }


def forward_record(source, phase, grad_enabled, shape, attention_row_sums=None):
    """One observed model input. Pure Python so the verdict is testable without torch.

    `attention_row_sums` gives each row's real token count and is exact. Without it the
    column count is used, which equals the longest row only because TRL's flush_left drops
    columns that are padding in every row; `basis` records which applied.
    """
    shape = tuple(int(v) for v in shape)
    rows, columns = (shape[0], shape[1]) if len(shape) >= 2 else (1, shape[0])
    if attention_row_sums is not None:
        max_row, basis = max(int(v) for v in attention_row_sums), "attention_mask"
    else:
        max_row, basis = columns, "columns"
    return {"source": source, "phase": phase, "grad_enabled": bool(grad_enabled),
            "rows": rows, "columns": columns, "max_row_tokens": max_row, "basis": basis}


def install_forward_recorders(model, probe):
    """Record every forward the model actually receives, from two independent points.

    The top-level call is what TRL invokes (`model(input_ids, **model_kwargs)`) and carries
    the attention mask. The input-embedding layer is reached through `Module.__call__` in
    both transformers' and Unsloth's Llama paths, so it still fires if a wrapper calls an
    inner `.forward` directly - PEFT's BaseTuner does exactly that, which is why a hook on
    the innermost causal LM would never fire.
    """
    import torch

    handles = []

    def on_model_call(module, args, kwargs):
        input_ids = kwargs.get("input_ids", args[0] if args else None)
        if input_ids is None or not hasattr(input_ids, "shape") or input_ids.dim() != 2:
            return
        mask = kwargs.get("attention_mask")
        sums = (mask.sum(dim=1).tolist()
                if mask is not None and hasattr(mask, "dim") and mask.dim() == 2 else None)
        probe.forwards.append(forward_record("model_call", probe.current_phase(),
                                             torch.is_grad_enabled(), input_ids.shape, sums))

    def on_embeddings(module, args):
        if not args or not hasattr(args[0], "shape") or args[0].dim() != 2:
            return
        probe.forwards.append(forward_record("input_embeddings", probe.current_phase(),
                                             torch.is_grad_enabled(), args[0].shape))

    handles.append(model.register_forward_pre_hook(on_model_call, with_kwargs=True))
    embeddings = model.get_input_embeddings() if hasattr(model, "get_input_embeddings") else None
    if embeddings is not None:
        handles.append(embeddings.register_forward_pre_hook(on_embeddings))
    return handles


def attach(trainer, probe, record_shapes=True):
    """Wire the probe into a real trainer.

    Losses come from `on_log` and `on_evaluate`, never from peeking at the last history
    entry: the installed Trainer calls `on_step_end` BEFORE logging and evaluation, so with
    evaluation every update the previous entry is normally an eval record and training
    losses go unrecorded entirely.
    """
    from transformers import TrainerCallback

    # TRL's preference collator emits prompt/chosen/rejected ids and attention masks, and
    # no `*labels` key at all - so a recorder searching for one recorded nothing. The model
    # sees prompt+completion concatenated, so the binding dimension is the longest
    # concatenation, measured over both sides after truncation.
    PAIR_KEYS = ("prompt_input_ids", "prompt_attention_mask",
                 "chosen_input_ids", "chosen_attention_mask",
                 "rejected_input_ids", "rejected_attention_mask")

    if record_shapes:
        collator = trainer.data_collator

        def recording_collator(features):
            batch = collator(features)
            record = {"shapes": {k: list(batch[k].shape) for k in PAIR_KEYS if k in batch}}
            if "prompt_attention_mask" in batch:
                prompt_per_row = batch["prompt_attention_mask"].sum(dim=1)
                chosen = batch.get("chosen_attention_mask")
                rejected = batch.get("rejected_attention_mask")
                chosen_per_row = chosen.sum(dim=1) if chosen is not None else None
                rejected_per_row = rejected.sum(dim=1) if rejected is not None else None

                # PER ROW, then take the maximum. max(prompts) + max(completions) is an
                # upper bound over DIFFERENT rows: prompts [1536, 1] with completions
                # [1, 512] reports 2048 while the largest real example is 1537. A bound
                # reaching the target does not mean any example does.
                per_row = []
                for position in range(int(prompt_per_row.shape[0])):
                    longer_side = max(
                        int(chosen_per_row[position]) if chosen_per_row is not None else 0,
                        int(rejected_per_row[position]) if rejected_per_row is not None else 0)
                    per_row.append(int(prompt_per_row[position]) + longer_side)
                # Named for both properties it has: PER ROW (not a cross-row maximum) and
                # PRE-TRUNCATION (TRL concatenates and truncates after the collator, so
                # this bounds the model input rather than measuring it).
                record["per_row_pre_truncation_tokens"] = per_row
                record["max_row_pre_truncation_tokens"] = max(per_row) if per_row else 0
                record["max_prompt_tokens"] = int(prompt_per_row.max())

                # EVERY pair must supervise both sides. "at least one row per side" passes
                # a batch of 15 empty pairs and one real one.
                supervised_rows = 0
                for position in range(int(prompt_per_row.shape[0])):
                    has_chosen = chosen_per_row is not None and int(chosen_per_row[position]) > 0
                    has_rejected = rejected_per_row is not None and int(rejected_per_row[position]) > 0
                    supervised_rows += 1 if (has_chosen and has_rejected) else 0
                record["rows_supervised_both_sides"] = supervised_rows
                record["batch_rows"] = int(batch["prompt_input_ids"].shape[0])
                record["all_rows_supervised"] = (supervised_rows == record["batch_rows"])
            # Real phase identity. Tagging by "has an evaluation finished" credits the
            # evaluation's OWN collations as post-evaluation stress, because the eval
            # dataloader uses this same collator.
            record["phase"] = probe.current_phase()
            # NB: these are the collator's outputs, BEFORE TRL concatenates and truncates
            # for the forward pass (dpo_trainer.py:1526-1550). They bound the model input,
            # they do not measure it. Named accordingly so no reader mistakes them for it.
            probe.shapes.append(record)
            return batch

        trainer.data_collator = recording_collator
        probe.forward_hooks = install_forward_recorders(trainer.model, probe)

    class ProbeCallback(TrainerCallback):
        def on_substep_end(self, args, state, control, **kwargs):
            probe.on_substep()
            return control

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs and "loss" in logs:
                probe.losses.append(float(logs["loss"]))
            return control

        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            if metrics and "eval_loss" in metrics:
                probe.eval_losses.append(float(metrics["eval_loss"]))
            probe.on_evaluation(memory_snapshot())
            return control

        def on_step_end(self, args, state, control, **kwargs):
            # The installed Trainer consults this flag itself for the scheduler; attach()
            # closes over the trainer, so the callback can read it too. Recording this as
            # impossible was wrong: it is an observation gap, not an API limit.
            skipped = None
            accelerator = getattr(trainer, "accelerator", None)
            if accelerator is not None:
                skipped = getattr(accelerator, "optimizer_step_was_skipped", None)
            if probe.on_optimizer_step(memory_snapshot(), skipped=skipped):
                control.should_training_stop = True
            return control

    # Mark the evaluation window explicitly. The eval dataloader uses this same collator,
    # so without this its batches are indistinguishable from post-evaluation training.
    original_evaluate = trainer.evaluate

    def bracketed_evaluate(*args, **kwargs):
        probe.in_evaluation = True
        try:
            return original_evaluate(*args, **kwargs)
        finally:
            probe.in_evaluation = False

    trainer.evaluate = bracketed_evaluate
    trainer.add_callback(ProbeCallback())
    return trainer
def build_production_trainer(base_model_id, rows, eval_rows, batch_size, grad_accum,
                             max_seq_length, max_prompt_length, beta, lora_r, output_dir,
                             max_steps):
    """Mirror `train_preference.py`'s construction exactly, differing only in data and size.

    `ref_model=None` is deliberate and load-bearing: TRL then uses adapter disabling rather
    than holding a second full model, which changes the memory story entirely.
    """
    import sys
    from datasets import Dataset
    from unsloth import FastLanguageModel, PatchDPOTrainer, is_bfloat16_supported
    from unsloth.chat_templates import get_chat_template

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "preference_optimization"))
    from train_preference import ensure_trl_warning_state, patch_trl_optional_dependency_checks

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model_id,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        full_finetuning=False,
    )
    tokenizer = get_chat_template(tokenizer, chat_template="chatml")
    model = FastLanguageModel.get_peft_model(
        model, r=lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=lora_r, lora_dropout=0, bias="none",
        use_gradient_checkpointing="unsloth", random_state=SEED,
        use_rslora=lora_r >= 64, loftq_config=None,
    )
    ensure_trl_warning_state(model)
    PatchDPOTrainer()
    patch_trl_optional_dependency_checks()
    from trl import DPOTrainer, DPOConfig

    trainer = DPOTrainer(
        model=model, ref_model=None,
        args=DPOConfig(
            output_dir=output_dir,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=grad_accum,
            # An explicit step horizon, not an epoch count. With one accumulation
            # window of rows, num_train_epochs=1 yields ONE optimizer update, so a
            # verdict requiring two could never be satisfied.
            warmup_ratio=0.03, warmup_steps=5, max_steps=max_steps,
            learning_rate=2e-4, logging_steps=1, logging_nan_inf_filter=False,
            dataloader_num_workers=0,
            optim="adamw_8bit", weight_decay=0.001, lr_scheduler_type="linear",
            report_to="none", seed=SEED,
            fp16=not is_bfloat16_supported(), bf16=is_bfloat16_supported(),
            save_strategy="no",
            eval_strategy="steps" if eval_rows is not None else "no",
            eval_steps=1,
            beta=beta, max_length=max_seq_length, max_prompt_length=max_prompt_length,
        ),
        train_dataset=Dataset.from_list(rows),
        eval_dataset=Dataset.from_list(eval_rows) if eval_rows is not None else None,
        processing_class=tokenizer,
    )
    return trainer, tokenizer


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-model-id", required=True,
                        help="MERGED parent, as production uses. Not a bare adapter.")
    parser.add_argument("--dataset", default="paperbd/paper_preference_150K-v1")
    parser.add_argument("--dataset-revision", default=None)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--max-prompt-length", type=int, default=1536)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--optimizer-steps", type=int, default=2)
    parser.add_argument("--with-evaluation", action="store_true",
                        help="Run a bounded validation pass and at least one update after it.")
    parser.add_argument("--post-eval-steps", type=int, default=1)
    parser.add_argument("--stress-target-tokens", type=int, default=0,
                        help="Minimum longest-row length a stress pass must deliver to the "
                             "model's input, measured at forward. "
                             "Defaults to --max-seq-length.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--output-dir", default="/tmp/dpo-probe-out")
    args = parser.parse_args(argv)

    import torch
    from datasets import load_dataset

    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    assert not path.exists(), f"refuse to overwrite probe evidence: {path}"

    def emit(event, **fields):
        record = json_safe({"event": event, "batch_size": args.batch_size,
                            "monotonic_seconds": time.monotonic(), **fields})
        with path.open("a") as handle:
            handle.write(json.dumps(record, sort_keys=True, allow_nan=False, default=str) + "\n")
        print("DPO_MEMORY_PROBE " + json.dumps(record, sort_keys=True, default=str), flush=True)
        return record

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "preference_optimization"))
    from chat_formatting import normalize_explicit_preference_example

    dataset = load_dataset(args.dataset, split="train", revision=args.dataset_revision)
    dataset = dataset.map(
        normalize_explicit_preference_example,
        remove_columns=[c for c in dataset.column_names
                        if c not in {"prompt", "chosen", "rejected"}],
    )
    split = dataset.train_test_split(test_size=0.02, seed=SEED)

    # Tokenizer only. `_, tok = FastLanguageModel.from_pretrained(...)` bound a whole model
    # to `_`, and Python evaluates the next call before rebinding it, so a second model
    # loaded while the first was still resident - putting a duplicate-load peak into the
    # very envelope this probe reports.
    # unsloth FIRST. It patches transformers at import time, so importing transformers
    # first can leave those patches unapplied - and the patched paths are precisely what
    # determines memory here. train_preference.py imports unsloth at line 21 and
    # transformers at line 35; this probe must match that order or it measures a different
    # code path than production. Unsloth emits an explicit warning when the order is wrong.
    from unsloth.chat_templates import get_chat_template   # imports the unsloth package
    from transformers import AutoTokenizer
    tok = get_chat_template(AutoTokenizer.from_pretrained(args.base_model_id),
                            chat_template="chatml")

    # One optimizer step consumes batch_size * grad_accum rows. Supply a full window per
    # requested step, plus one more when a post-evaluation update is required.
    windows = args.optimizer_steps + (args.post_eval_steps if args.with_evaluation else 0)
    per_window = args.batch_size * args.grad_accum
    unique_rows, indices, lengths = select_stress_pairs(
        split["train"], per_window, tok, args.max_seq_length, args.max_prompt_length)
    rows = (unique_rows * windows)[: per_window * windows]   # repeat the worst case
    eval_rows = unique_rows[: args.batch_size] if args.with_evaluation else None

    emit("configuration", diagnostic_only=True, grad_accum=args.grad_accum,
         optimizer_steps_requested=args.optimizer_steps,
         accumulation_windows=windows, rows_supplied=len(rows),
         with_evaluation=bool(args.with_evaluation),
         selected_indices=indices, selected_lengths=lengths,
         max_seq_length=args.max_seq_length, max_prompt_length=args.max_prompt_length,
         base_model_id=args.base_model_id, reference_topology="ref_model=None (adapter disabling)")

    torch.cuda.reset_peak_memory_stats()
    probe = BoundedProbe(emit, args.optimizer_steps,
                         args.post_eval_steps if args.with_evaluation else 0,
                         stress_target_tokens=args.stress_target_tokens or args.max_seq_length)
    try:
        trainer, _ = build_production_trainer(
            args.base_model_id, rows, eval_rows, args.batch_size, args.grad_accum,
            args.max_seq_length, args.max_prompt_length, args.beta, args.lora_r,
            args.output_dir, max_steps=windows)
        attach(trainer, probe)
        # The input shape the stress measurement depends on. padding_free flattens every
        # row into one, so column counts would sum rows rather than bound one; refuse it.
        trainer_args = trainer.args
        padding_free = bool(getattr(trainer_args, "padding_free", False))
        emit("trainer_ready", padding_free=padding_free,
             truncation_mode=getattr(trainer_args, "truncation_mode", None),
             max_length=getattr(trainer_args, "max_length", None),
             max_prompt_length=getattr(trainer_args, "max_prompt_length", None),
             use_logits_to_keep=getattr(trainer_args, "use_logits_to_keep", None),
             precompute_ref_log_probs=getattr(trainer_args, "precompute_ref_log_probs", None),
             forward_hooks=len(getattr(probe, "forward_hooks", [])))
        started = time.monotonic()
        trainer.train()
        torch.cuda.synchronize()
        verdict = probe.verdict()
        optimizer = getattr(trainer, "optimizer", None)
        verdict["optimizer_initialised"] = bool(
            optimizer is not None and getattr(optimizer, "state", None))
        verdict["padding_free"] = padding_free
        verdict["fit"] = bool(verdict["fit"] and verdict["optimizer_initialised"]
                              and not padding_free)
        emit("candidate_result", seconds=time.monotonic() - started,
             memory=memory_snapshot(), batch_shapes=probe.shapes[:4],
             forward_shapes=probe.forwards[:8], **verdict)
        return 0 if verdict["fit"] else 2
    except torch.cuda.OutOfMemoryError as exc:
        # Merge once. Passing fit= alongside **verdict(), which also carries fit, raised
        # TypeError - so an OOM was reported as a broken handler rather than as the memory
        # result this probe exists to produce.
        emit("candidate_result", **{**probe.verdict(), "fit": False,
                                    "error": "OutOfMemoryError", "message": str(exc)[:600],
                                    "memory": memory_snapshot()})
        return 3
    except Exception as exc:  # a non-OOM failure is a different finding entirely
        emit("candidate_result", **{**probe.verdict(), "fit": False,
                                    "error": type(exc).__name__, "message": str(exc)[:600]})
        return 4


if __name__ == "__main__":
    raise SystemExit(main())
