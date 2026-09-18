"""Inspect response masking in actual Unsloth SFTTrainer dataloader batches."""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path


SYSTEM_PROMPT = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.
You are an expert in AI, deep learning, and machine learning research and its applications.
Your answers are concise and helps directly solve any user query truthfully.
If you do not know the answer, you will inform the user that you do not know instead of making answers up.
    """


def file_sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def find_subsequence(values, target, start=0, end=None):
    end = len(values) if end is None else end
    width = len(target)
    for index in range(start, end - width + 1):
        if values[index:index + width] == target:
            return index
    return None


def logical_spans(input_ids, position_ids, attention_mask):
    """Return (physical row, start, end) spans, using packed position resets."""
    spans = []
    for row, tokens in enumerate(input_ids):
        valid = len(tokens)
        if attention_mask is not None:
            valid = sum(attention_mask[row])
        starts = [0]
        if position_ids is not None:
            starts = [index for index, value in enumerate(position_ids[row][:valid]) if value == 0]
            if not starts or starts[0] != 0:
                starts.insert(0, 0)
        starts = sorted(set(starts))
        spans.extend((row, start, starts[offset + 1] if offset + 1 < len(starts) else valid)
                     for offset, start in enumerate(starts) if start < valid)
    return spans


def inspect_batch(batch, assistant_marker):
    input_ids = batch["input_ids"].detach().cpu()
    labels = batch["labels"].detach().cpu()
    if input_ids.ndim == 1:
        input_ids, labels = input_ids.unsqueeze(0), labels.unsqueeze(0)
    attention = batch.get("attention_mask")
    positions = batch.get("position_ids")
    attention = attention.detach().cpu().tolist() if attention is not None else None
    positions = positions.detach().cpu().tolist() if positions is not None else None
    token_rows, label_rows = input_ids.tolist(), labels.tolist()
    result = {
        "physical_rows": len(token_rows), "logical_examples": 0,
        "assistant_markers_missing": 0, "prompt_tokens_in_loss": 0,
        "response_tokens_masked": 0, "response_tokens_in_loss": 0,
        "padding_tokens_in_loss": 0, "fully_masked_examples": 0,
    }
    if attention is not None:
        result["padding_tokens_in_loss"] = sum(
            not keep and label != -100
            for row_mask, row_labels in zip(attention, label_rows)
            for keep, label in zip(row_mask, row_labels)
        )
    for row, start, end in logical_spans(token_rows, positions, attention):
        result["logical_examples"] += 1
        marker = find_subsequence(token_rows[row], assistant_marker, start, end)
        if marker is None:
            result["assistant_markers_missing"] += 1
            result["fully_masked_examples"] += int(
                all(label == -100 for label in label_rows[row][start:end])
            )
            continue
        response_start = marker + len(assistant_marker)
        result["prompt_tokens_in_loss"] += sum(
            label != -100 for label in label_rows[row][start:response_start]
        )
        response_labels = label_rows[row][response_start:end]
        result["response_tokens_masked"] += sum(label == -100 for label in response_labels)
        result["response_tokens_in_loss"] += sum(label != -100 for label in response_labels)
        result["fully_masked_examples"] += int(
            all(label == -100 for label in label_rows[row][start:end])
        )
    return result


def add_counts(total, current):
    for key, value in current.items():
        total[key] = total.get(key, 0) + value


def read_rows(path):
    rows = []
    with Path(path).open() as handle:
        for line in handle:
            value = json.loads(line)
            rows.append(value.get("row", value))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--sample", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--reference-sft", required=True)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    import unsloth  # noqa: F401 -- must patch Transformers/TRL before either is imported
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import (
        get_chat_template, standardize_data_formats, to_sharegpt, train_on_responses_only,
    )
    import torch
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    assert torch.cuda.is_available(), "This gate requires a CUDA GPU."
    native_bf16 = torch.cuda.is_bf16_supported(including_emulation=False)
    output = Path(args.output).resolve()
    reference = Path(args.reference_sft).resolve()
    if output.is_relative_to(reference.parents[1]):
        raise ValueError("Gate output must be outside the public repository")
    output.mkdir(parents=True, exist_ok=True)

    rows = read_rows(args.sample)
    dataset = Dataset.from_list(rows)
    dataset = to_sharegpt(dataset, merged_prompt="{instruction}\n\n{input}",
                          output_column_name="output", conversation_extension=1,
                          random_state=3407)
    dataset = standardize_data_formats(dataset)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model, max_seq_length=args.max_seq_length,
        load_in_4bit=True, full_finetuning=False,
    )
    tokenizer = get_chat_template(tokenizer, chat_template="chatml")

    def format_batch(examples):
        system = [{"role": "system", "content": SYSTEM_PROMPT}]
        return {"text": [tokenizer.apply_chat_template(
            system + conversation, tokenize=False, add_generation_prompt=False
        ) for conversation in examples["conversations"]]}

    dataset = dataset.map(format_batch, batched=True, remove_columns=dataset.column_names)
    model = FastLanguageModel.get_peft_model(
        model, r=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=32, lora_dropout=0, bias="none",
        use_gradient_checkpointing="unsloth", random_state=3407,
        use_rslora=False, loftq_config=None,
    )
    assistant_marker = tokenizer(
        "<|im_start|>assistant\n", add_special_tokens=False
    )["input_ids"]

    report = {
        "pass": False,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Actual Unsloth SFTTrainer preprocessing, response masking, collator and train dataloader; no optimizer update",
        "model": args.model,
        "model_revision": args.model_revision,
        "sample": {"path": str(Path(args.sample).resolve()),
                   "sha256": file_sha256(args.sample), "rows": len(rows)},
        "reference_sft_sha256": file_sha256(reference),
        "max_seq_length": args.max_seq_length,
        "batch_size": args.batch_size,
        "gpu": torch.cuda.get_device_name(0),
        "native_bf16": native_bf16,
        "versions": {name: importlib.metadata.version(name) for name in
                     ["torch", "transformers", "trl", "datasets", "peft",
                      "unsloth", "unsloth-zoo", "bitsandbytes"]},
        "configurations": {},
    }

    for packing in (True, False):
        trainer = SFTTrainer(
            model=model, processing_class=tokenizer, train_dataset=dataset,
            args=SFTConfig(
                output_dir=str(output / ("packed" if packing else "unpacked")),
                dataset_text_field="text", per_device_train_batch_size=args.batch_size,
                gradient_accumulation_steps=4, max_steps=1, learning_rate=2e-4,
                dataloader_num_workers=0, optim="adamw_8bit", report_to="none",
                max_length=args.max_seq_length, packing=packing, dataset_num_proc=1,
                bf16=native_bf16, fp16=not native_bf16,
            ),
        )
        prepared_rows = len(trainer.train_dataset)
        trainer = train_on_responses_only(
            trainer, instruction_part="<|im_start|>user\n",
            response_part="<|im_start|>assistant\n", num_proc=1,
        )
        masked_rows = len(trainer.train_dataset)
        counts = {}
        batches = 0
        for batch in trainer.get_train_dataloader():
            add_counts(counts, inspect_batch(batch, assistant_marker))
            batches += 1
        counts.update({
            "source_rows": len(rows), "prepared_rows": prepared_rows,
            "masked_rows": masked_rows,
            "rows_filtered_as_fully_masked": prepared_rows - masked_rows,
            "dataloader_batches": batches,
        })
        counts["safe"] = all(counts[key] == 0 for key in (
            "assistant_markers_missing", "prompt_tokens_in_loss",
            "response_tokens_masked", "padding_tokens_in_loss",
            "fully_masked_examples", "rows_filtered_as_fully_masked",
        )) and counts["response_tokens_in_loss"] > 0
        report["configurations"]["packed" if packing else "unpacked"] = counts

    packed = report["configurations"]["packed"]
    unpacked = report["configurations"]["unpacked"]
    report["packing_decision"] = (
        "disable" if unpacked["safe"] and not packed["safe"] else
        "unresolved"
    )
    report["pass"] = unpacked["safe"] and not packed["safe"]
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    (output / "sft_trainer_gate.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    assert report["pass"], report


if __name__ == "__main__":
    main()
