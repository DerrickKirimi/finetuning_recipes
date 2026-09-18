"""Measure peak process memory when materializing a corpus, streamed vs. fully listed."""

import argparse
import json
import platform
import resource
import subprocess
import sys
from pathlib import Path

from assets import output_directory

CHILD = """
import json, resource, sys
sys.path.insert(0, {cpt!r})
from datasets import Dataset
import corpus as C
if {streamed!r}:
    build = C.documents_to_dataset
else:
    build = lambda documents: Dataset.from_dict(
        {{"text": [document.text for document in documents]}}
    )
dataset = build(C.load_corpus({path!r}))
print(json.dumps({{"rows": dataset.num_rows,
                  "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024}}))
"""


def measure(path, streamed, cpt_dir, cache_dir):
    """Run one materialization in a fresh interpreter so peak RSS is attributable."""

    proc = subprocess.run(
        [sys.executable, "-c", CHILD.format(cpt=cpt_dir, streamed=streamed, path=path)],
        capture_output=True, text=True,
        env={"PATH": "/usr/bin:/bin", "HOME": str(Path.home()),
             "HF_DATASETS_CACHE": f"{cache_dir}/{'streamed' if streamed else 'listed'}"},
    )
    if proc.returncode != 0:
        raise SystemExit(proc.stderr)
    return json.loads(proc.stdout.strip().splitlines()[-1])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True, help="JSONL corpus with a 'text' field")
    parser.add_argument("--cache-dir", required=True, help="Scratch dataset cache, outside the repo")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    cpt_dir = str((Path(__file__).resolve().parents[1] / "cpt"))
    corpus_bytes = Path(args.corpus).stat().st_size
    listed = measure(args.corpus, False, cpt_dir, args.cache_dir)
    streamed = measure(args.corpus, True, cpt_dir, args.cache_dir)
    assert listed["rows"] == streamed["rows"], (listed, streamed)

    report = {
        "scope": "Peak RSS of corpus materialization only; not a training measurement",
        "corpus_bytes": corpus_bytes,
        "rows": streamed["rows"],
        "python": platform.python_version(),
        "listed_from_dict": listed,
        "streamed_from_generator": streamed,
        "peak_ratio_listed": round(listed["peak_rss_bytes"] / corpus_bytes, 2),
        "peak_ratio_streamed": round(streamed["peak_rss_bytes"] / corpus_bytes, 2),
        "reduction": round(listed["peak_rss_bytes"] / streamed["peak_rss_bytes"], 2),
    }
    (output_directory(args.output) / "corpus_memory.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
