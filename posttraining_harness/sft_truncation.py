"""Identify exact SFT source rows whose response is fully masked after truncation."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from assets import output_directory


SYSTEM_PROMPT = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.
You are an expert in AI, deep learning, and machine learning research and its applications.
Your answers are concise and helps directly solve any user query truthfully.
If you do not know the answer, you will inform the user that you do not know instead of making answers up.
    """


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    import unsloth  # noqa: F401 -- patch before importing Transformers
    from datasets import load_dataset
    from transformers import AutoTokenizer
    from unsloth.chat_templates import get_chat_template, train_on_responses_only

    manifest = json.loads(args.split_manifest.read_text())
    assert manifest["dataset"] == args.dataset
    assert manifest["dataset_revision"] == args.revision
    source = load_dataset(args.dataset, revision=args.revision, split="train")
    assert len(source) == manifest["train_source"]["rows"]
    validation = set(manifest["internal_validation"]["validation_indices"])
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    tokenizer = get_chat_template(tokenizer, chat_template="chatml")
    mask = train_on_responses_only(
        None, tokenizer=tokenizer, return_function=True,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )
    system = [{"role": "system", "content": SYSTEM_PROMPT}]
    rejected = []
    candidates = []
    candidate_count = 0
    scanned = 0

    def inspect(items):
        if not items:
            return
        texts = [item[1] for item in items]
        encoded = tokenizer(texts, add_special_tokens=False, truncation=True,
                            max_length=args.max_length, padding=False)["input_ids"]
        labels = mask({"input_ids": encoded})["labels"]
        for (index, text, row, prefix_bytes), ids, row_labels in zip(items, encoded, labels):
            if any(label != -100 for label in row_labels):
                continue
            full_length = len(tokenizer(text, add_special_tokens=False)["input_ids"])
            rejected.append({
                "source_index": index,
                "split": "validation" if index in validation else "train",
                "prompt_sha256": digest(row["instruction"] + "\n\n" + row.get("input", "")),
                "passage_sha256": digest(row.get("input", "")) if row.get("input") else None,
                "instruction_chars": len(row["instruction"]),
                "input_chars": len(row.get("input", "")),
                "output_chars": len(row["output"]),
                "assistant_prefix_utf8_bytes": prefix_bytes,
                "full_tokens": full_length,
                "discarded_tokens": max(0, full_length - args.max_length),
            })

    marker = "<|im_start|>assistant\n"
    for index, row in enumerate(source):
        conversation = [
            {"role": "user", "content": row["instruction"] + "\n\n" + row.get("input", "")},
            {"role": "assistant", "content": row["output"]},
        ]
        text = tokenizer.apply_chat_template(
            system + conversation, tokenize=False, add_generation_prompt=False
        )
        prefix = text.index(marker) + len(marker)
        prefix_bytes = len(text[:prefix].encode())
        candidate_count += 1
        candidates.append((index, text, row, prefix_bytes))
        if len(candidates) == args.batch_size:
            inspect(candidates)
            candidates.clear()
        scanned += 1
    inspect(candidates)
    counts = {"train": 0, "validation": 0}
    for row in rejected:
        counts[row["split"]] += 1
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset, "dataset_revision": args.revision,
        "model": str(Path(args.model).resolve()),
        "split_manifest_sha256": hashlib.sha256(args.split_manifest.read_bytes()).hexdigest(),
        "max_length": args.max_length, "source_rows_scanned": scanned,
        "rows_tokenized": candidate_count,
        "fully_masked_counts": counts, "fully_masked_rows": rejected,
        "criterion": "all labels -100 under the installed Unsloth response-only helper after exact ChatML rendering and truncation",
    }
    out = output_directory(args.output)
    (out / "sft_truncation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in (
        "dataset", "dataset_revision", "max_length", "source_rows_scanned",
        "rows_tokenized", "fully_masked_counts",
    )}, indent=2), flush=True)


if __name__ == "__main__":
    main()
