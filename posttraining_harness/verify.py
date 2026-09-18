"""Rerun offline audits and retain commands, outputs, exit codes and code hashes."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from assets import output_directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--reference-map", required=True)
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()
    out = output_directory(args.results_root)
    repo = Path(__file__).resolve().parents[1]
    assets = json.loads((out / "assets/assets.json").read_text())

    def asset(kind, name):
        rev = assets[f"{kind}/{name}"]["revision"]
        return str(out / "assets" / kind / name.replace("/", "--") / rev)

    sources = out / "dependencies"
    template = str(sources / "unsloth-2026.6.1/chat_templates.py")
    helper = str(sources / "unsloth-zoo-2026.6.1/dataset_utils.py")
    commands = []
    for task, name in [("sft", "paperbd/paper_instructions_300K-v1"),
                       ("dpo", "paperbd/paper_preference_150K-v1")]:
        folder = asset("datasets", name)
        commands.append([sys.executable, "posttraining_harness/data_boundaries.py", "--train", folder + "/train.jsonl",
                         "--test", folder + "/test.jsonl", "--output", str(out / task)])
    commands += [
        [sys.executable, "posttraining_harness/map_citations.py", "--reference-map", str(Path(args.reference_map).resolve()),
         "--revision", args.revision, "--output", str(out)],
        [sys.executable, "posttraining_harness/batches.py", "--sample", str(out / "sft/sample.jsonl"),
         "--tokenizer", asset("models", "paperbd/smollm_135M_arxiv_cpt"),
         "--helper-source", helper, "--template-source", template,
         "--sft-source", "instruction_tuning/sft.py", "--output", str(out / "sft")],
        [sys.executable, "posttraining_harness/dpo_tokens.py", "--sample", str(out / "dpo/sample.jsonl"),
         "--tokenizer", asset("models", "paperbd/smollm_135M_neuraltxt_v1"),
         "--template-source", template, "--output", str(out / "dpo")],
        ["bash", "check_env.sh", "--verify", "--output", str(out / "local")],
        [sys.executable, "-m", "pytest", "-q", "-o", f"cache_dir={out / 'pytest-cache'}",
         "--basetemp", str(out / "pytest-tmp"), "posttraining_harness/test_audit.py", "posttraining_harness/test_cpt_gate.py",
         "tests/reasoning", "reasoning/tests",
         "tests/data_prep/reasoning"],
        ["git", "diff", "--check"],
    ]
    env = dict(os.environ)
    overrides = {"PY": sys.executable, "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
                 "HF_DATASETS_CACHE": str(out / "datasets-cache"), "TMPDIR": str(out / "tmp"),
                 "PAPER_GRPO_TEST_DATASET": str(out / "sft/batch_rows.jsonl")}
    Path(overrides["TMPDIR"]).mkdir(exist_ok=True)
    env.update(overrides)
    records = []
    for command in commands:
        print("Running", command[1:3], flush=True)
        proc = subprocess.run(command, cwd=repo, env=env, text=True, capture_output=True)
        records.append({"argv": command, "exit_code": proc.returncode,
                        "stdout": proc.stdout, "stderr": proc.stderr})
        (out / "checks.json").write_text(json.dumps({"cwd": str(repo), "environment_overrides": overrides,
            "commands": records, "scope": "Audit execution only, not GPU/training readiness"}, indent=2) + "\n")
        if proc.returncode:
            print(proc.stdout + proc.stderr)
            raise SystemExit(proc.returncode)
    files = [*Path(__file__).parent.glob("*.py"), repo / "pyproject.toml", repo / "uv.lock",
             repo / "check_env.sh", repo / "posttraining_harness/pyproject.toml", repo / "posttraining_harness/uv.lock",
             repo / "tests/reasoning/test_paper_grpo_pipeline.py"]
    (out / "verification_sources.json").write_text(json.dumps({str(p.relative_to(repo)):
        hashlib.sha256(p.read_bytes()).hexdigest() for p in files}, indent=2) + "\n")
    print(f"{len(records)} audit commands completed; see checks.json. GPU gates remain unverified.")


if __name__ == "__main__":
    main()
