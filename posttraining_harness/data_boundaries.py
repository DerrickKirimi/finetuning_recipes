"""Measure exact prompt/passage leakage; do not infer missing document provenance."""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random

import numpy as np

from assets import output_directory


def normalized(value):
    return " ".join(str(value).split())


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def identities(row):
    if "instruction" in row:
        prompt = normalized(row["instruction"]) + "\n\n" + normalized(row.get("input", ""))
        passage = normalized(row.get("input", ""))
    else:
        value = row.get("prompt", [])
        prompt = normalized(value) if isinstance(value, str) else "\n".join(
            normalized(m["content"]) for m in value if m.get("role") != "system")
        passage = ""  # Cannot reconstruct source passages from arbitrary preference prompts.
    return digest(prompt), digest(passage) if passage else None


def load_identities(path, seed, sample_size=128):
    prompts, passages, sample = [], [], []
    columns, source_ids = Counter(), Counter()
    rng = random.Random(seed)
    with Path(path).open() as f:
        for i, line in enumerate(f):
            row = json.loads(line)
            columns.update(row.keys())
            for key in ["document_id", "source_document_id", "arxiv_id", "source_id"]:
                if row.get(key) is not None:
                    source_ids[key] += 1
            prompt, passage = identities(row)
            prompts.append(prompt)
            passages.append(passage)
            item = {"source_row_index": i, "row": row}
            if len(sample) < sample_size:
                sample.append(item)
            else:
                index = rng.randrange(i + 1)
                if index < sample_size:
                    sample[index] = item
    return prompts, passages, sample, dict(columns), dict(source_ids)


def overlap(left, right):
    left = {x for x in left if x is not None}
    right = [x for x in right if x is not None]
    return {"shared_unique_keys": len(left.intersection(right)),
            "right_rows_matching_left": sum(x in left for x in right),
            "right_rows_with_key": len(right)}


def audit(train, test, seed, holdout):
    tp, ti, sample, columns, ids = load_identities(train, seed)
    ep, ei, _, test_columns, test_ids = load_identities(test, seed)
    permutation = np.random.default_rng(seed).permutation(len(tp))
    n_test = math.ceil(holdout * len(tp))
    eval_indexes, train_indexes = permutation[:n_test], permutation[n_test:]
    return {"train_rows": len(tp), "test_rows": len(ep), "train_columns": columns,
            "test_columns": test_columns, "source_id_nonnull_counts": {"train": ids, "test": test_ids},
            "document_independence": "NOT VERIFIED: exact keys cannot establish document independence",
            "published_prompt_overlap": overlap(tp, ep),
            "published_passage_overlap": overlap(ti, ei),
            "train_duplicate_prompt_rows": len(tp) - len(set(tp)),
            "reference_row_split": {"seed": seed, "test_size": holdout,
                "algorithm": "NumPy default_rng permutation, test prefix (datasets shuffle split)",
                "prompt_overlap": overlap([tp[i] for i in train_indexes], [tp[i] for i in eval_indexes]),
                "passage_overlap": overlap([ti[i] for i in train_indexes], [ti[i] for i in eval_indexes])},
            "coverage": "All JSONL rows; normalized exact matches only, not fuzzy/semantic matching"}, sample


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--holdout", type=float, default=0.02)
    args = parser.parse_args()
    if not 0 < args.holdout < 1:
        parser.error("holdout must be between 0 and 1")
    out = output_directory(args.output)
    report, sample = audit(args.train, args.test, args.seed, args.holdout)
    report["input_files"] = {}
    for name in [args.train, args.test]:
        with open(name, "rb") as f:
            report["input_files"][name] = hashlib.file_digest(f, "sha256").hexdigest()
    (out / "data_boundaries.json").write_text(json.dumps(report, indent=2) + "\n")
    (out / "sample.jsonl").write_text("".join(json.dumps(x) + "\n" for x in sample))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
