"""Offline tokenizer/collator audit; no model load or training, no Unsloth import."""

import argparse
import ast
import hashlib
import json
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoTokenizer, DataCollatorForSeq2Seq
from trl import pack_dataset
from trl.trainer.sft_trainer import DataCollatorForLanguageModeling

from assets import output_directory


def source_objects(path, names, namespace=None):
    """Execute only explicitly selected top-level definitions from supplied source."""
    tree = ast.parse(Path(path).read_text())
    nodes = [node for node in tree.body if
             isinstance(node, ast.FunctionDef) and node.name in names or
             isinstance(node, ast.Assign) and any(
                 isinstance(t, ast.Name) and t.id in names for t in node.targets)]
    scope = {} if namespace is None else dict(namespace)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), scope)
    return scope


def chatml_tokenizer(path, template):
    tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
    # Mirror get_chat_template's fast-tokenizer EOS swap without loading Unsloth.
    old, stop = tok.eos_token, "<|im_end|>"
    vocab = tok.backend_tokenizer.to_str()
    if stop in vocab:
        vocab = vocab.replace(old, "<|:__TEMP//STOP//TOKEN__:|>").replace(stop, old).replace(
            "<|:__TEMP//STOP//TOKEN__:|>", stop)
    else:
        vocab = vocab.replace(old, stop)
    tok = tok.__class__(tokenizer_object=tok.backend_tokenizer.from_str(vocab),
                        bos_token=tok.bos_token, eos_token=stop, unk_token=tok.unk_token,
                        pad_token=stop if tok.pad_token == old else tok.pad_token)
    tok.chat_template = template
    return tok


def encode_rows(rows, tok, formatter):
    conversations = [[{"role": "user", "content": r["instruction"] + "\n\n" + r["input"]},
                      {"role": "assistant", "content": r["output"]}] for r in rows]
    texts = formatter({"conversations": conversations}, tok)["text"]
    result = []
    marker = "<|im_start|>assistant\n"
    for text in texts:
        # SFTTrainer adds EOS when the rendered text ends in a newline, not EOS.
        if not text.endswith(tok.eos_token):
            text += tok.eos_token
        boundary = text.index(marker) + len(marker)
        encoded = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        expected = [int(start >= boundary and end > start) for start, end in encoded["offset_mapping"]]
        result.append({"input_ids": encoded["input_ids"], "expected": expected})
    return result


def check(rows, tok, mask_fn, length, packing):
    original_lengths = [len(r["input_ids"]) for r in rows]
    if packing:
        examples = list(pack_dataset(Dataset.from_list(rows), seq_length=length, strategy="bfd"))
    else:
        examples = [{k: v[:length] for k, v in row.items()} for row in rows]
    labels = mask_fn({"input_ids": [r["input_ids"] for r in examples]})["labels"]
    features = [{"input_ids": r["input_ids"], "labels": label,
                 **({"seq_lengths": r["seq_lengths"]} if packing else {})}
                for r, label in zip(examples, labels)]
    collator = (DataCollatorForLanguageModeling(pad_token_id=tok.pad_token_id,
                 completion_only_loss=False, padding_free=True) if packing else
                DataCollatorForSeq2Seq(tokenizer=tok))
    prompt_unmasked = answer_masked = padding_unmasked = empty = eos_kept = 0
    shape = None
    for feature, row in zip(features, examples):
        batch = collator([feature])
        shape = list(batch["input_ids"].shape)
        actual = batch["labels"][0].tolist()
        expected = row["expected"]
        prompt_unmasked += sum(y != -100 and not e for y, e in zip(actual, expected))
        answer_masked += sum(y == -100 and e for y, e in zip(actual, expected))
        empty += int(all(y == -100 for y in actual))
        eos_kept += int(row["input_ids"][-1] == tok.eos_token_id)
        if "attention_mask" in batch:
            padding_unmasked += int(((batch["attention_mask"] == 0) & (batch["labels"] != -100)).sum())
    # Exercise unequal-length, genuinely padded batches as well.
    if not packing:
        batch = collator(features[:8])
        padding_unmasked += int(((batch["attention_mask"] == 0) & (batch["labels"] != -100)).sum())
        assert padding_unmasked == 0
        assert prompt_unmasked == 0
    return {"source_examples": len(rows), "collated_examples": len(examples),
            "max_length": length, "packing": packing,
            "overlength_source_examples": sum(n > length for n in original_lengths),
            "discarded_tail_tokens": sum(max(0, n-length) for n in original_lengths),
            "prompt_tokens_in_loss": prompt_unmasked, "response_tokens_masked": answer_masked,
            "padding_tokens_in_loss": padding_unmasked, "fully_masked_examples": empty,
            "examples_ending_in_eos": eos_kept, "last_single_example_batch_shape": shape,
            "multi_document_packs": sum(len(r.get("seq_lengths", [])) > 1 for r in examples)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ["sample", "tokenizer", "helper-source", "template-source", "sft-source", "output"]:
        p.add_argument("--" + arg, required=True)
    args = p.parse_args()
    template = source_objects(args.template_source, {"chatml_template"})["chatml_template"]
    tok = chatml_tokenizer(args.tokenizer, template)
    helper = source_objects(args.helper_source,
        {"_longest_common_sublist", "_find_common_token_ids", "train_on_responses_only"}, {"torch": torch})
    mask_fn = helper["train_on_responses_only"](None, tokenizer=tok, return_function=True,
        instruction_part="<|im_start|>user\n", response_part="<|im_start|>assistant\n")
    formatter = source_objects(args.sft_source, {"formatting_func"})["formatting_func"]
    rows = [json.loads(line)["row"] for line in Path(args.sample).read_text().splitlines()]
    encoded = encode_rows(rows, tok, formatter)
    synthetic = encode_rows([{"instruction": "Answer.", "input": "One.", "output": "Two."},
                             {"instruction": "Answer.", "input": "Three.", "output": "Four."}], tok, formatter)
    long_prompt = encode_rows([{"instruction": "Answer.", "input": "word " * 3000,
                               "output": "The answer is cut off."}], tok, formatter)
    report = {"scope": "Instrumented CPU preprocessing: reference formatter, mirrored fast-tokenizer EOS swap, extracted Unsloth return_function, installed TRL packer/collators. NOT a full Unsloth trainer or GPU test.",
        "source_sha256": {path: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in
                          [args.sample, args.helper_source, args.template_source, args.sft_source]},
        "tokenizer_path": str(Path(args.tokenizer).resolve()), "eos_token": tok.eos_token,
        "eos_token_id": tok.eos_token_id, "pad_token_id": tok.pad_token_id,
        "sample_unpacked": check(encoded, tok, mask_fn, 2048, False),
        "sample_bfd_packed": check(encoded, tok, mask_fn, 2048, True),
        "two_conversation_regression": check(synthetic, tok, mask_fn, 2048, True),
        "overlength_prompt_regression": check(long_prompt, tok, mask_fn, 2048, False)}
    out = output_directory(args.output)
    (out / "batch_rows.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    (out / "batches.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
