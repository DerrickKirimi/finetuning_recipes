"""Build grouped SFT validation and a train-disjoint fixed test battery."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random

from assets import output_directory
from data_boundaries import identities


class UnionFind:
    def __init__(self):
        self.parent = []
        self.size = []

    def add(self):
        index = len(self.parent)
        self.parent.append(index)
        self.size.append(1)
        return index

    def find(self, item):
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left, right):
        left, right = self.find(left), self.find(right)
        if left == right:
            return
        if self.size[left] < self.size[right]:
            left, right = right, left
        self.parent[right] = left
        self.size[left] += self.size[right]


def file_sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_keys(path):
    keys = []
    with Path(path).open() as handle:
        for line in handle:
            keys.append(identities(json.loads(line)))
    return keys


def components(keys):
    union_find = UnionFind()
    owners = {}
    for prompt, passage in keys:
        index = union_find.add()
        for key in (("prompt", prompt), ("passage", passage) if passage else None):
            if key is None:
                continue
            if key in owners:
                union_find.union(index, owners[key])
            else:
                owners[key] = index
    groups = defaultdict(list)
    for index in range(len(keys)):
        groups[union_find.find(index)].append(index)
    return list(groups.values())


def grouped_validation(keys, fraction, seed):
    groups = components(keys)
    random.Random(seed).shuffle(groups)
    target = math.ceil(len(keys) * fraction)
    selected, count = [], 0
    for group in groups:
        if count >= target:
            break
        selected.extend(group)
        count += len(group)
    validation = sorted(selected)
    validation_set = set(validation)
    training = [index for index in range(len(keys)) if index not in validation_set]
    return training, validation, groups


def overlap(left, right):
    left_prompts = {prompt for prompt, _ in left}
    left_passages = {passage for _, passage in left if passage}
    return {
        "prompt_rows": sum(prompt in left_prompts for prompt, _ in right),
        "passage_rows": sum(bool(passage and passage in left_passages) for _, passage in right),
    }


def fixed_battery(train_keys, test_keys, size, seed):
    train_prompts = {prompt for prompt, _ in train_keys}
    train_passages = {passage for _, passage in train_keys if passage}
    candidates = [index for index, (prompt, passage) in enumerate(test_keys)
                  if prompt not in train_prompts and not (passage and passage in train_passages)]
    candidate_keys = [test_keys[index] for index in candidates]
    groups = components(candidate_keys)
    random.Random(seed).shuffle(groups)
    selected = []
    for group in groups:
        # One representative per connected prompt/passage group avoids weighting duplicates.
        selected.append(candidates[min(group)])
        if len(selected) == size:
            break
    if len(selected) < size:
        raise ValueError(f"only {len(selected)} clean test groups for battery size {size}")
    return sorted(selected), len(candidates), len(groups)


def build(train_path, test_path, validation_fraction, battery_size, seed):
    train_keys, test_keys = read_keys(train_path), read_keys(test_path)
    training, validation, train_groups = grouped_validation(train_keys, validation_fraction, seed)
    train_overlap = overlap([train_keys[index] for index in training],
                            [train_keys[index] for index in validation])
    assert train_overlap == {"prompt_rows": 0, "passage_rows": 0}
    battery, clean_test_rows, clean_test_groups = fixed_battery(
        train_keys, test_keys, battery_size, seed
    )
    battery_overlap = overlap(train_keys, [test_keys[index] for index in battery])
    assert battery_overlap == {"prompt_rows": 0, "passage_rows": 0}
    sizes = Counter(len(group) for group in train_groups)
    return {
        "algorithm": "connected components over exact normalized prompt or nonempty passage hashes",
        "seed": seed,
        "validation_fraction_requested": validation_fraction,
        "train_source": {"path": str(Path(train_path).resolve()),
                         "sha256": file_sha256(train_path), "rows": len(train_keys)},
        "test_source": {"path": str(Path(test_path).resolve()),
                        "sha256": file_sha256(test_path), "rows": len(test_keys)},
        "internal_validation": {
            "train_indices": training,
            "validation_indices": validation,
            "train_rows": len(training),
            "validation_rows": len(validation),
            "validation_fraction_actual": len(validation) / len(train_keys),
            "component_count": len(train_groups),
            "component_size_counts": dict(sorted(sizes.items())),
            "cross_split_overlap": train_overlap,
        },
        "fixed_test_battery": {
            "indices": battery,
            "rows": len(battery),
            "clean_candidate_rows": clean_test_rows,
            "clean_candidate_components": clean_test_groups,
            "overlap_with_full_training_source": battery_overlap,
            "selection": "one lowest source index per seeded shuffled clean component",
            "prompt_sha256": [test_keys[index][0] for index in battery],
            "passage_sha256": [test_keys[index][1] for index in battery],
        },
        "limits": [
            "Exact normalized matches only; no fuzzy, semantic or source-document identity.",
            "Published rows contain no source document ID, so document independence is not established.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset", default="paperbd/paper_instructions_300K-v1")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--validation-fraction", type=float, default=0.02)
    parser.add_argument("--battery-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()
    if not 0 < args.validation_fraction < 1:
        parser.error("validation-fraction must be between zero and one")
    if args.battery_size < 1:
        parser.error("battery-size must be positive")
    report = build(args.train, args.test, args.validation_fraction, args.battery_size, args.seed)
    report["dataset"] = args.dataset
    report["dataset_revision"] = args.revision
    out = output_directory(args.output)
    path = out / "sft_split_manifest.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "output": str(path),
        "dataset": report["dataset"],
        "revision": report["dataset_revision"],
        "train_rows": report["internal_validation"]["train_rows"],
        "validation_rows": report["internal_validation"]["validation_rows"],
        "battery_rows": report["fixed_test_battery"]["rows"],
        "limits": report["limits"],
    }, indent=2))


if __name__ == "__main__":
    main()
