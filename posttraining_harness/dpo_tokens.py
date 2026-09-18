"""Exercise pinned DPO chat formatting and tokenization without loading weights."""

import argparse
import hashlib
import inspect
import json
from pathlib import Path
import sys

from trl import DPOTrainer
from trl.data_utils import maybe_apply_chat_template

from assets import output_directory
from batches import chatml_tokenizer, source_objects

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# Match train_preference.py's direct-module fallback; package __init__ imports a
# moved diversity module and cannot currently be imported on its own.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "preference_optimization"))
from chat_formatting import normalize_explicit_preference_example


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["sample", "tokenizer", "template-source", "output"]:
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    template = source_objects(args.template_source, {"chatml_template"})["chatml_template"]
    tok = chatml_tokenizer(args.tokenizer, template)
    rows = [json.loads(line)["row"] for line in Path(args.sample).read_text().splitlines()]
    prompt_cut = combined_over = eos_missing = max_combined = 0
    for row in rows:
        text = maybe_apply_chat_template(normalize_explicit_preference_example(row), tokenizer=tok)
        result = DPOTrainer.tokenize_row(text, tok, max_prompt_length=1536,
                                        max_completion_length=None, add_special_tokens=False)
        original = tok(text["prompt"], add_special_tokens=False)["input_ids"]
        assert result["prompt_input_ids"] == original[-1536:]
        prompt_cut += int(len(original) > 1536)
        for key in ["chosen_input_ids", "rejected_input_ids"]:
            size = len(result["prompt_input_ids"]) + len(result[key])
            combined_over += int(size > 2048)
            max_combined = max(max_combined, size)
            eos_missing += int(result[key][-1] != tok.eos_token_id)
    # Verify left truncation with a prompt deliberately longer than the limit.
    synthetic = {"prompt": "word " * 2000, "chosen": "yes", "rejected": "no"}
    result = DPOTrainer.tokenize_row(synthetic, tok, max_prompt_length=1536,
                                    add_special_tokens=False)
    assert len(result["prompt_input_ids"]) == 1536
    assert result["prompt_input_ids"] == tok(synthetic["prompt"], add_special_tokens=False)["input_ids"][-1536:]
    assert eos_missing == 0
    report = {"scope": "Pinned TRL DPO tokenizer and reference normalizer; mirrored ChatML EOS swap. No trainer forward, model weights or Unsloth patches exercised.",
              "rows": len(rows), "prompt_limit": 1536, "combined_limit_in_reference": 2048,
              "prompts_left_truncated": prompt_cut, "chosen_rejected_sequences_over_combined_limit": combined_over,
              "max_combined_tokens_before_forward": max_combined,
              "completion_sequences_missing_eos": eos_missing, "synthetic_left_truncation_assertion": "PASS",
              "tokenizer": str(Path(args.tokenizer).resolve()),
              "sample_sha256": hashlib.sha256(Path(args.sample).read_bytes()).hexdigest(),
              "tokenize_row_source_sha256": hashlib.sha256(inspect.getsource(DPOTrainer.tokenize_row).encode()).hexdigest()}
    (output_directory(args.output) / "tokens.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
