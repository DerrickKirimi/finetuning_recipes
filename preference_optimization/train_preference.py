"""
Preference tuning with Unsloth + TRL.
Supports DPO and ORPO.

Usage:
    uv run python preference_optimization/train_preference.py \
        --method dpo \
        --output_model_id preference_tuned

    uv run python preference_optimization/train_preference.py \
        --method orpo \
        --output_model_id preference_orpo

Bounded, resumable segments are opt-in: pass a full-schedule --max_steps, then --stop_after_steps to end a segment
with its optimizer, scheduler, scaler and RNG state saved, and --resume_from_checkpoint to continue the same schedule
in a new process. Everything that does not need Unsloth lives in dpo_setup.py so CPU tests exercise it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from unsloth import FastLanguageModel, PatchDPOTrainer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from preference_optimization import dpo_setup as setup
    from preference_optimization.training_controls import (make_stop_callback, select_precision, terminal_record,
                                                           validate_resume_checkpoint)
    from preference_optimization.trl_compat import patch_trl_optional_dependency_checks
except ModuleNotFoundError:
    import dpo_setup as setup
    from training_controls import make_stop_callback, select_precision, terminal_record, validate_resume_checkpoint
    from trl_compat import patch_trl_optional_dependency_checks

import torch
from transformers import EarlyStoppingCallback
from unsloth.chat_templates import get_chat_template

SEED = setup.SEED


def ensure_trl_warning_state(model) -> None:
    """Ensure TRL can write trainer warning flags on PEFT/Unsloth models."""
    for candidate in (
        model,
        getattr(model, "base_model", None),
        getattr(getattr(model, "base_model", None), "model", None),
    ):
        if candidate is not None and not hasattr(candidate, "warnings_issued"):
            candidate.warnings_issued = {}


def parse_args():
    return setup.build_parser().parse_args()


def main() -> None:
    args = parse_args()
    setup.check_args(args)
    output_dir = args.output_dir or f"models/{args.output_model_id}"
    record = {"argv": sys.argv[1:], "output_dir": output_dir}
    stop_record: dict = {}

    def save_record():
        if args.run_record:
            setup.write_json(args.run_record, record)

    precision = select_precision(torch)
    record["precision"] = precision
    if args.resume_from_checkpoint:
        # fp16 training saves GradScaler state; resuming without it would restart the loss scale.
        record["resume"] = validate_resume_checkpoint(args.resume_from_checkpoint, expected_max_steps=args.max_steps,
                                                      require_scaler=precision["fp16"])
    record["stage"] = "data"
    save_record()

    print(f"Method: {args.method.upper()}")
    print(f"Base model: {args.base_model_id}")
    print(f"Dataset: {args.dataset_file or args.dataset}")

    # Data identity is checked before a model is allocated.
    train_dataset, val_dataset, record["data"] = setup.load_preference_datasets(args)
    record["horizon"] = {
        "max_steps": args.max_steps,
        "epoch_equivalent_max_steps": setup.epoch_equivalent_max_steps(len(train_dataset), args.batch_size,
                                                                       args.grad_accum, args.epochs),
        "stop_after_steps": args.stop_after_steps,
    }
    print(f"Rows: {record['data']['rows']}  horizon: {record['horizon']}")
    record["stage"] = "model"
    save_record()

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base_model_id,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
        full_finetuning=False,
    )
    template_before = setup.template_fingerprint(tokenizer) if getattr(tokenizer, "chat_template", None) else None
    tokenizer = get_chat_template(tokenizer, chat_template="chatml")
    template_after = setup.template_fingerprint(tokenizer)
    expected_markers = json.loads(args.expected_marker_ids) if args.expected_marker_ids else None
    record["tokenizer"] = {**setup.marker_report(tokenizer, expected_markers),
                           "template_before": template_before, "template_after": template_after,
                           "template_unchanged": template_before == template_after}
    save_record()
    if args.require_template_unchanged and template_before != template_after:
        raise SystemExit(f"get_chat_template changed tokenization: {template_before} -> {template_after}")

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=setup.LORA_TARGET_MODULES,
        lora_alpha=args.lora_r,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=SEED,
        use_rslora=args.lora_r >= 64,
        loftq_config=None,
    )

    ensure_trl_warning_state(model)
    config_kwargs = setup.preference_config_kwargs(args, precision, output_dir)

    if args.method == "dpo":
        PatchDPOTrainer()
        patch_trl_optional_dependency_checks()
        from trl import DPOTrainer, DPOConfig

        trainer = DPOTrainer(
            model=model,
            ref_model=None,
            args=DPOConfig(**config_kwargs),
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            processing_class=tokenizer,
        )
    else:
        # ORPO
        patch_trl_optional_dependency_checks()
        from trl import ORPOTrainer, ORPOConfig

        trainer = ORPOTrainer(
            model=model,
            args=ORPOConfig(**config_kwargs),
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            processing_class=tokenizer,
        )

    trainer.add_callback(
        EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience, early_stopping_threshold=0.0)
    )
    if args.stop_after_steps is not None:
        trainer.add_callback(make_stop_callback(args.stop_after_steps, stop_record))
    if args.stop_by_unix is not None:
        trainer.add_callback(setup.make_deadline_callback(args.stop_by_unix, stop_record))
    if args.telemetry:
        trainer.add_callback(setup.make_telemetry_callback(args.telemetry, stop_record, trainer))
    guard_record: dict = {}
    if args.reference_guard_every:
        trainer.add_callback(setup.make_reference_guard_callback(trainer, args.reference_probe_rows,
                                                                 args.reference_guard_every, guard_record, stop_record))
    if args.batch_fingerprints:
        trainer.data_collator = setup.FingerprintingCollator(trainer.data_collator, args.batch_fingerprints)

    record["trainer"] = setup.trainer_record(trainer)
    if args.method == "dpo" and args.reference_probe_rows:
        record["probe_before"] = setup.reference_probe(trainer, args.reference_probe_rows)
    if args.fail_fast_checks:
        record["preflight_problems"] = setup.preflight_problems(record, args)
        if record["preflight_problems"]:
            record["stage"] = "preflight_failed"
            save_record()
            raise SystemExit(f"preflight checks failed before training: {record['preflight_problems']}")
    if args.reference_lifecycle_diagnostic:
        record["stage"] = "reference_lifecycle"
        save_record()
        use_gc = getattr(trainer.args, "gradient_checkpointing", True)     # what Unsloth's train wrapper passes
        record["lifecycle"] = setup.reference_lifecycle(
            trainer, args.reference_probe_rows, args.reference_lifecycle_diagnostic,
            for_inference=FastLanguageModel.for_inference,
            for_training=lambda m: FastLanguageModel.for_training(m, use_gradient_checkpointing=use_gc))
        record["stop"] = {**stop_record, "global_step": trainer.state.global_step}
        record["stage"] = "reference_lifecycle_complete"
        setup.write_json(args.diagnostic_record, record)
        save_record()
        return

    record["stage"] = "train"
    save_record()

    try:
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    except BaseException as exc:
        record["stage"] = "train_failed"
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["stop"] = {**stop_record, "global_step": trainer.state.global_step}
        if args.reference_guard_every:
            record["reference_guard"] = guard_record
        save_record()
        raise

    # The trainer has already restored the best checkpoint into `model` (load_best_model_at_end); the terminal state
    # of this segment is the checkpoint saved at the stop, recorded here by step.
    record["stop"] = terminal_record(trainer.state, stop_record, args.stop_after_steps)
    record["callback_states_after_train"] = setup.callback_states(trainer)
    if args.reference_guard_every:
        record["reference_guard"] = guard_record
    if args.batch_fingerprints:
        record["data_position"] = setup.check_data_position(
            trainer.train_dataset, trainer.eval_dataset, args.batch_fingerprints, seed=trainer.args.seed,
            first_step=record.get("resume", {}).get("global_step", 0), last_step=trainer.state.global_step,
            batch_size=args.batch_size, grad_accum=args.grad_accum)
    save_record()
    if args.method == "dpo" and args.reference_probe_rows:
        record["probe_after_best_restored"] = setup.reference_probe(trainer, args.reference_probe_rows)

    if record["stop"]["stop_reason"] == "reference_guard":
        record["stage"] = "reference_guard_failed"
        print(f"Reference guard stopped training at step {trainer.state.global_step}: {guard_record.get('first_failure')}")
    elif record["stop"]["stop_reason"] in ("target_step", "deadline"):
        record["stage"] = "segment_complete"
        print(f"Segment stopped at step {trainer.state.global_step} ({record['stop']['stop_reason']}); "
              f"best checkpoint {trainer.state.best_model_checkpoint}")
    else:
        final_dir = f"{output_dir}/final"
        model.save_pretrained(final_dir)
        tokenizer.save_pretrained(final_dir)
        record["final_dir"] = final_dir
        record["stage"] = "complete"
        print(f"Saved to {final_dir}")
    save_record()


if __name__ == "__main__":
    main()
