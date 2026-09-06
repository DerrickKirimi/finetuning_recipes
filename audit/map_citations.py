"""Validate map citation locations against an immutable git revision, not claims."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

from assets import output_directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-map", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    revision = subprocess.check_output(["git", "rev-parse", "--verify", args.revision + "^{commit}"],
                                      cwd=repo, text=True).strip()
    content = Path(args.reference_map).read_text()
    citations, failures = [], []
    cache = {}
    for number, line in enumerate(content.splitlines(), 1):
        matches = re.findall(r"`([\w./-]+):(\d+)(?:[-–](\d+))?`", line)
        if line.startswith("- ") and not matches:
            failures.append({"map_line": number, "error": "entry has no citation"})
        for filename, start, end in matches:
            if filename not in cache:
                proc = subprocess.run(["git", "show", f"{revision}:{filename}"], cwd=repo,
                                      capture_output=True, text=True)
                cache[filename] = proc.stdout.splitlines() if proc.returncode == 0 else []
            start, end = int(start), int(end or start)
            valid = 1 <= start <= end <= len(cache[filename])
            record = {"map_line": number, "file": filename, "start": start, "end": end,
                      "valid_location": valid}
            citations.append(record)
            if not valid:
                failures.append(record)
    report = {"revision": revision, "map_sha256": hashlib.sha256(content.encode()).hexdigest(),
              "scope": "Citation locations and per-bullet coverage only; not semantic entailment",
              "citations": citations, "failures": failures,
              "summary": {"citations": len(citations), "source_files": len(cache), "failures": len(failures)}}
    (output_directory(args.output) / "map_citations.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"]))
    raise SystemExit(bool(failures))


if __name__ == "__main__":
    main()
