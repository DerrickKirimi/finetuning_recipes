"""Package the corpus-adapter delta and emit its Colab equivalence-gate driver."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import zipfile

from assets import output_directory

DELTA = ("cpt/sft.py", "cpt/corpus.py", "posttraining_harness/cpt_gate.py")

DRIVER = '''"""Apply the hashed corpus-adapter delta and run its equivalence gate on Colab."""

import hashlib, json, os, shutil, subprocess, zipfile
from datetime import datetime, timezone
from pathlib import Path

ARCHIVE = Path("/content/{archive_name}")
ARCHIVE_SHA256 = "{archive_sha256}"
RUN = Path("/content/posttraining-cpt-runs/{run_id}")
BASELINE_SHA256 = "{baseline_sha256}"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


assert ARCHIVE.is_file(), ARCHIVE
assert sha256(ARCHIVE) == ARCHIVE_SHA256
project, results = RUN / "code", RUN / "results"
baseline = results / "formal/report.json"
assert baseline.is_file(), baseline
assert sha256(baseline) == BASELINE_SHA256, "Baseline report is not the expected formal run"
assert (project / ".venv/bin/python").is_file()
assert (results / "model-snapshot/model.safetensors").is_file()

with zipfile.ZipFile(ARCHIVE) as bundle:
    manifest = json.loads(bundle.read("bundle-manifest.json"))
    assert set(bundle.namelist()) == set(manifest["files"]) | {{"bundle-manifest.json"}}
    for name, expected in manifest["files"].items():
        destination = (project / name).resolve()
        assert destination.is_relative_to(project.resolve()), name
        data = bundle.read(name)
        assert hashlib.sha256(data).hexdigest() == expected, name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        assert sha256(destination) == expected

output = results / "adapter"
assert not output.exists(), output
command = [str(project / ".venv/bin/python"), "-u", str(project / "posttraining_harness/cpt_gate.py"),
           "--project", str(project), "--output", str(output),
           "--model-revision", manifest["model_revision"],
           "--mode", "adapter", "--baseline-report", str(baseline)]
started = datetime.now(timezone.utc).isoformat()
environment = os.environ.copy()
environment.update(PYTHONNOUSERSITE="1", WANDB_MODE="disabled")
process = subprocess.run(command, env=environment)
report = json.loads((output / "report.json").read_text())
runner = {{"pass": process.returncode == 0 and report.get("pass") is True,
          "started_utc": started, "finished_utc": datetime.now(timezone.utc).isoformat(),
          "archive": str(ARCHIVE), "archive_sha256": ARCHIVE_SHA256, "manifest": manifest,
          "command": command, "exit_code": process.returncode,
          "report_sha256": sha256(output / "report.json")}}
(output / "runner.json").write_text(json.dumps(runner, indent=2) + "\\n")
archive_path = Path("/content/cpt-adapter-results-{run_id}.zip")
shutil.make_archive(str(archive_path.with_suffix("")), "zip", output.parent, output.name)
print("\\nCORPUS ADAPTER CLI RESULT\\n" + json.dumps(runner, indent=2))
print("Result archive:", archive_path)
assert runner["pass"], runner
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="Private results directory")
    parser.add_argument("--run-id", required=True, help="Remote run id holding the formal baseline")
    parser.add_argument("--baseline-sha256", required=True, help="sha256 of that run's formal report.json")
    args = parser.parse_args()

    project = Path(__file__).resolve().parents[1]
    created = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    out = output_directory(args.output) / created
    out.mkdir(parents=True, exist_ok=True)

    files = {name: hashlib.sha256((project / name).read_bytes()).hexdigest() for name in DELTA}
    manifest = {
        "created_utc": created,
        "scope": "Exact public source delta for the CPT corpus-adapter equivalence gate",
        "files": files,
        "baseline_run": args.run_id,
        "baseline_report_sha256": args.baseline_sha256,
        "model": "HuggingFaceTB/SmolLM-135M",
        "model_revision": "1d461723eec654e65efdc40cf49301c89c0c92f4",
    }
    archive_name = f"cpt-adapter-inputs-{created}.zip"
    archive = out / archive_name
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
        for name in DELTA:
            bundle.write(project / name, name)
        bundle.writestr("bundle-manifest.json", json.dumps(manifest, indent=2) + "\n")
    archive_sha256 = hashlib.sha256(archive.read_bytes()).hexdigest()

    (out / "bundle-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    driver = out / f"cpt-adapter-cli-{created}.py"
    driver.write_text(DRIVER.format(archive_name=archive_name, archive_sha256=archive_sha256,
                                    run_id=args.run_id, baseline_sha256=args.baseline_sha256))
    compile(driver.read_text(), str(driver), "exec")
    print(json.dumps({"archive": str(archive), "archive_sha256": archive_sha256,
                      "driver": str(driver), "files": files}, indent=2))


if __name__ == "__main__":
    main()
