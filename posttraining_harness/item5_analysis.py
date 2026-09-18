"""Measure a generated instruction slice: schema validity, task mix, lengths, degeneracy."""

import argparse
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from assets import output_directory

# main.py prints one line per batch it attempts and one per failure it swallows.
ATTEMPT_RE = re.compile(r"^Running batch augmentation for (\S+) on (\d+) chunks$", re.M)
SKIP_RE = re.compile(r"^Skipping (.+?): (.*)$", re.M)
CROSS_RE = re.compile(r"^Running (\S+) augmentation", re.M)

# The Alpaca instruction text is the only task marker on a saved row.
TASK_MARKERS = [
    ("bullets", ("important points", "key points", "markdown bullet", "python list")),
    ("qa_pairs", ("question and", "question answer", "questions and answers",
                  "generate a question", "answer the user's question")),
    ("rephrase", ("rephrase", "rewrite", "paraphrase")),
    ("triplets", ("triplet", "subject", "relation")),
    ("retrieval", ("retriev", "which passage", "given the provided passage")),
    ("comparison", ("compare", "difference between", "contrast")),
    ("continuation", ("continue the passage", "generate the rest")),
    ("fact", ("important fact", "piece of information")),
]


def classify(instruction):
    text = (instruction or "").lower()
    for name, markers in TASK_MARKERS:
        if any(m in text for m in markers):
            return name
    return "unclassified"


def quantiles(values):
    if not values:
        return {}
    ordered = sorted(values)
    pick = lambda q: ordered[min(len(ordered) - 1, int(q * len(ordered)))]
    return {"n": len(ordered), "min": ordered[0], "p25": pick(0.25),
            "median": pick(0.5), "p75": pick(0.75), "p95": pick(0.95), "max": ordered[-1],
            "mean": round(statistics.fmean(ordered), 1)}


def schema_validity(logs):
    attempts, failures, reasons = 0, 0, Counter()
    for log in logs:
        text = Path(log).read_text(errors="replace")
        for _task, count in ATTEMPT_RE.findall(text):
            attempts += 1
        attempts += len(CROSS_RE.findall(text))
        for label, reason in SKIP_RE.findall(text):
            failures += 1
            reasons[reason.strip()[:120]] += 1
    return {"generation_calls_attempted": attempts, "calls_failed": failures,
            "schema_validity_rate": None if attempts == 0 else round(1 - failures / attempts, 4),
            "failure_reasons": reasons.most_common(10)}


def degenerate(rows):
    """Outputs that satisfy the schema but are unusable; the check that is usually skipped."""
    empty, copied, dup_within_doc = 0, 0, 0
    seen = defaultdict(set)
    for row in rows:
        out = (row.get("output") or "").strip()
        passage = (row.get("input") or "").strip()
        if not out:
            empty += 1
            continue
        if passage and len(out) > 40 and out in passage:
            copied += 1
        key = (passage[:200], out)
        if key in seen[passage[:200]]:
            dup_within_doc += 1
        seen[passage[:200]].add(key)
    total = max(1, len(rows))
    return {"rows": len(rows), "empty_output": empty, "output_copied_from_passage": copied,
            "duplicate_output_within_passage": dup_within_doc,
            "empty_rate": round(empty / total, 4),
            "copied_rate": round(copied / total, 4),
            "duplicate_rate": round(dup_within_doc / total, 4),
            "degenerate_rate": round((empty + copied + dup_within_doc) / total, 4)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated", nargs="+", required=True, help="generated.jsonl files")
    parser.add_argument("--log", nargs="+", default=[], help="generation.log files")
    parser.add_argument("--output", required=True)
    parser.add_argument("--handread", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = []
    for path in args.generated:
        rows.extend(json.loads(l) for l in Path(path).open() if l.strip())

    by_task = Counter(classify(r.get("instruction")) for r in rows)
    lengths = defaultdict(list)
    for row in rows:
        lengths[classify(row.get("instruction"))].append(len((row.get("output") or "").split()))

    report = {
        "scope": "Slice measurement only; no comparison against the published dataset",
        "rows": len(rows),
        "schema": schema_validity(args.log) if args.log else "NO LOGS SUPPLIED",
        "task_distribution": by_task.most_common(),
        "task_share": {k: round(v / max(1, len(rows)), 4) for k, v in by_task.most_common()},
        "output_words_by_task": {k: quantiles(v) for k, v in sorted(lengths.items())},
        "degenerate": degenerate(rows),
    }

    import random
    rng = random.Random(args.seed)
    buckets = defaultdict(list)
    for row in rows:
        buckets[classify(row.get("instruction"))].append(row)
    picks, tasks = [], sorted(buckets)
    while len(picks) < min(args.handread, len(rows)) and tasks:
        for task in list(tasks):
            if not buckets[task]:
                tasks.remove(task); continue
            picks.append(buckets[task].pop(rng.randrange(len(buckets[task]))))
            if len(picks) >= args.handread:
                break
    out = output_directory(args.output)
    (out / "handread_sample.jsonl").write_text(
        "".join(json.dumps(p) + "\n" for p in picks))
    report["handread_written"] = len(picks)
    (out / "slice_measurements.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "output_words_by_task"}, indent=2))


if __name__ == "__main__":
    main()
