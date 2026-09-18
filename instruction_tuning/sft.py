import argparse
import hashlib
import json
import sys
from pathlib import Path

from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template, to_sharegpt, standardize_data_formats, train_on_responses_only

if sys.platform != "darwin":
    import transformers.utils.generic

    transformers.utils.generic._is_mlx_available = False

from datasets import load_dataset
from datasets.combine import concatenate_datasets

from trl import SFTTrainer, SFTConfig
from transformers import EarlyStoppingCallback, TrainerCallback
import torch
SEED = 3407

parser = argparse.ArgumentParser(description="ChatML instruction fine-tuning with Unsloth on alpaca-format data.")
parser.add_argument("--base_model_id", "-i", type=str, default="paperbd/smollm_135M_arxiv_cpt",
                    help="Path to a model in models/ directory or a HF model ID.")
parser.add_argument("--output_model_id", "-o", type=str, default="instruction_tuned",
                    help="Output subdirectory under models/.")
parser.add_argument("--dataset", "-d", type=str, default="paperbd/paper_instructions_300K-v1",
                    help="HF dataset to train on (alpaca format: instruction, input, output).")
parser.add_argument("--dataset_revision", type=str, default=None,
                    help="Immutable Hugging Face dataset revision.")
parser.add_argument("--split_manifest", type=Path, default=None,
                    help="Grouped train/validation source-index manifest.")
parser.add_argument("--truncation_manifest", type=Path, default=None,
                    help="Audited fully-masked source rows expected at max_seq_length.")
parser.add_argument("--max_seq_length", type=int, default=2048)
parser.add_argument("--batch_size", "-bs", type=int, default=32)
parser.add_argument("--eval_batch_size", type=int, default=None,
                    help="Evaluation batch size; defaults to --batch_size.")
parser.add_argument("--grad_accum", type=int, default=4)
parser.add_argument("--epochs", "-e", type=int, default=3)
parser.add_argument("--lora_r", type=int, default=32)
parser.add_argument("--load_in_4bit", action="store_true", default=True)
parser.add_argument("--conversation_extension", type=int, default=1)
parser.add_argument("--variations", type=int, default=1)
parser.add_argument("--learning_rate", "-lr", type=float, default=2e-4) 
parser.add_argument("--max_steps", type=int, default=-1,
                    help="Positive values override epochs for bounded pilots.")
parser.add_argument("--stop_after_steps", type=int, default=None,
                    help="Stop a bounded pilot early while retaining the --max_steps schedule.")
parser.add_argument("--save_steps", type=int, default=50)
parser.add_argument("--eval_steps", type=int, default=50)
parser.add_argument("--logging_steps", type=int, default=10)
parser.add_argument("--resume_from_checkpoint", type=Path, default=None)

args = parser.parse_args()

if args.conversation_extension == 1:
    args.variations = 1

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=args.base_model_id,
    max_seq_length=args.max_seq_length,
    load_in_4bit=args.load_in_4bit,
    full_finetuning=False,
)
tokenizer = get_chat_template(tokenizer, chat_template="chatml")

def prepare_dataset(source):
    variations = []
    for i in range(args.variations):
        variations.append(to_sharegpt(
            source,
            merged_prompt="{instruction}\n\n{input}",
            output_column_name="output",
            conversation_extension=args.conversation_extension,
            random_state=SEED + i,
        ))
    prepared = concatenate_datasets(variations)
    return standardize_data_formats(prepared)


source_dataset = load_dataset(args.dataset, revision=args.dataset_revision, split="train")
expected_removed = {"train": 0, "validation": 0}

if args.split_manifest is not None:
    manifest = json.loads(args.split_manifest.read_text())
    assert manifest["dataset"] == args.dataset
    assert manifest["dataset_revision"] == args.dataset_revision
    assert manifest["train_source"]["rows"] == len(source_dataset)
    split_spec = manifest["internal_validation"]
    assert split_spec["cross_split_overlap"] == {"prompt_rows": 0, "passage_rows": 0}
    train_indices, val_indices = split_spec["train_indices"], split_spec["validation_indices"]
    all_indices = train_indices + val_indices
    assert len(all_indices) == len(source_dataset)
    assert len(set(all_indices)) == len(source_dataset)
    assert min(all_indices) == 0 and max(all_indices) == len(source_dataset) - 1
    train_source = source_dataset.select(train_indices)
    val_source = source_dataset.select(val_indices)
    train_dataset = prepare_dataset(train_source)
    val_dataset = prepare_dataset(val_source)
else:
    dataset = prepare_dataset(source_dataset)
    source_split = dataset.train_test_split(test_size=0.02, seed=SEED)
    train_dataset, val_dataset = source_split["train"], source_split["test"]

if args.truncation_manifest is not None:
    assert args.split_manifest is not None
    truncation = json.loads(args.truncation_manifest.read_text())
    assert truncation["dataset"] == args.dataset
    assert truncation["dataset_revision"] == args.dataset_revision
    assert truncation["max_length"] == args.max_seq_length
    assert truncation["source_rows_scanned"] == len(source_dataset)
    assert truncation["split_manifest_sha256"] == hashlib.sha256(
        args.split_manifest.read_bytes()
    ).hexdigest()
    expected_removed = truncation["fully_masked_counts"]
    rejected = truncation["fully_masked_rows"]
    assert len(rejected) == expected_removed["train"] + expected_removed["validation"]
    assert len({row["source_index"] for row in rejected}) == len(rejected)
    validation_indices = set(val_indices)
    assert all(row["split"] == (
        "validation" if row["source_index"] in validation_indices else "train"
    ) for row in rejected)

def formatting_func(examples, tokenizer):
    # Step 3: serialize each conversation to a flat text string using the
    # ChatML Jinja template now set on the tokenizer.
    # add_generation_prompt=False because we include the full assistant turn
    # (including <|im_end|>) during training — we're not doing inference here.

    SYSTEM_PROMPT = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.
You are an expert in AI, deep learning, and machine learning research and its applications.
Your answers are concise and helps directly solve any user query truthfully.
If you do not know the answer, you will inform the user that you do not know instead of making answers up.
    """
    convos = examples["conversations"]
    system_part = [{"role": "system", "content": SYSTEM_PROMPT}]
    texts = [
        tokenizer.apply_chat_template(system_part + c, tokenize=False, add_generation_prompt=False)
        for c in convos
    ]
    return {"text": texts}

train_dataset = train_dataset.map(
    lambda examples: formatting_func(examples, tokenizer),
    batched=True,
    remove_columns=train_dataset.column_names,
)
val_dataset = val_dataset.map(
    lambda examples: formatting_func(examples, tokenizer),
    batched=True,
    remove_columns=val_dataset.column_names,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=args.lora_r,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=args.lora_r,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=SEED,
    use_rslora=args.lora_r >= 64,
    loftq_config=None,
)


max_grad_norm = 1.0
native_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported(
    including_emulation=False
)

trainer = SFTTrainer(
    model = model,
    processing_class = tokenizer,
    train_dataset = train_dataset,
    eval_dataset = val_dataset,
    args = SFTConfig(
        output_dir = f"models/{args.output_model_id}",
        dataset_text_field = "text",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=(args.eval_batch_size or args.batch_size),
        gradient_accumulation_steps=args.grad_accum,
        warmup_ratio=0.03,
        warmup_steps = 5,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate = args.learning_rate,
        logging_steps = args.logging_steps,
        dataloader_num_workers=8,
        optim = "adamw_8bit",
        weight_decay = 0.001,
        lr_scheduler_type = "linear",
        report_to = "none", # Use TrackIO/WandB etc,
        max_grad_norm=max_grad_norm,
        seed=SEED,
        max_length=args.max_seq_length,
        packing=False,
        padding_free=False,
        dataset_num_proc=8,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        bf16=native_bf16,
        fp16=torch.cuda.is_available() and not native_bf16,
        ddp_find_unused_parameters=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    ),
)

train_rows_before_masking = len(trainer.train_dataset)
eval_rows_before_masking = len(trainer.eval_dataset)
trainer = train_on_responses_only(
    trainer,
    instruction_part="<|im_start|>user\n",
    response_part="<|im_start|>assistant\n",
)
removed_train_rows = train_rows_before_masking - len(trainer.train_dataset)
removed_eval_rows = eval_rows_before_masking - len(trainer.eval_dataset)
observed_removed = {"train": removed_train_rows, "validation": removed_eval_rows}
if observed_removed != expected_removed:
    raise RuntimeError(
        "Fully masked exclusion drift after truncation: "
        f"expected={expected_removed}, observed={observed_removed}"
    )
print("Verified fully masked exclusions:", json.dumps(observed_removed, sort_keys=True))

stop_record = {"stop_after_steps_fired": False}

if args.stop_after_steps is not None:
    if args.max_steps <= 0 or not 0 < args.stop_after_steps < args.max_steps:
        raise ValueError("--stop_after_steps requires 0 < stop_after_steps < max_steps")

    class StopAfterStepsCallback(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step >= stop_after_steps:
                control.should_training_stop = True
                # Transformers saves only when global_step is a multiple of save_steps
                # or has reached max_steps. A segment stop is usually neither -- 1942 is
                # not a multiple of 50, and max_steps stays at the full horizon -- so
                # without forcing this the terminal optimizer state is never written and
                # the newest checkpoint on disk is an earlier, save-aligned one.
                control.should_save = True
                stop_record["stop_after_steps_fired"] = True
            return control

    stop_after_steps = args.stop_after_steps
    trainer.add_callback(StopAfterStepsCallback())

trainer.add_callback(
    EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.0)
)

# Check the fully constructed trainer, before training. If this is False, transformers
# falls back to averaging per-microbatch losses instead of normalising by the token count
# of the whole accumulation window, and batch/accumulation splits stop being equivalent.
# Asserted here rather than by patching Trainer.__init__, which would alter an
# introspected constructor signature and run before Unsloth's own patches are installed.
_accepts_loss_kwargs = getattr(trainer, "model_accepts_loss_kwargs", None)
print("LOSS_NORMALIZATION " + json.dumps({"model_accepts_loss_kwargs": _accepts_loss_kwargs}), flush=True)
assert _accepts_loss_kwargs is True, (
    f"token-weighted loss normalization inactive: {_accepts_loss_kwargs!r}"
)

trainer.train(resume_from_checkpoint=(
    str(args.resume_from_checkpoint) if args.resume_from_checkpoint else None
))

# Record why training ended, so a consumer never has to infer it from which checkpoint
# happens to be newest. An old checkpoint is not evidence of early stopping.
_terminal = {
    "global_step": int(trainer.state.global_step),
    "max_steps": int(trainer.state.max_steps),
    "stop_after_steps": args.stop_after_steps,
    "stop_after_steps_fired": stop_record["stop_after_steps_fired"],
    "best_model_checkpoint": trainer.state.best_model_checkpoint,
    "best_metric": trainer.state.best_metric,
}
_terminal["stop_reason"] = (
    "target_step" if stop_record["stop_after_steps_fired"]
    else "horizon" if trainer.state.global_step >= trainer.state.max_steps
    else "early_stopping_or_other"
)
Path(f"models/{args.output_model_id}").mkdir(parents=True, exist_ok=True)
Path(f"models/{args.output_model_id}/stop_reason.json").write_text(
    json.dumps(_terminal, indent=2, sort_keys=True) + "\n"
)
print("STOP_REASON " + json.dumps(_terminal, sort_keys=True), flush=True)

model.save_pretrained(f"models/{args.output_model_id}/final")
tokenizer.save_pretrained(f"models/{args.output_model_id}/final")
print(f"Saved to models/{args.output_model_id}/final")
