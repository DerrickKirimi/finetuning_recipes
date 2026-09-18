"""Package the item-6 trainer-batch gate and emit a private Colab CLI driver."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import textwrap
import zipfile

from assets import output_directory


MODEL = "paperbd/smollm_135M_arxiv_cpt"
MODEL_REVISION = "94c5e5c15117559f3f867ab952df909c991ca22f"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    sample = Path(args.sample).resolve()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    out = output_directory(args.output) / stamp
    out.mkdir()
    names = [
        "pyproject.toml", "uv.lock", "README.md", "check_env.sh",
        "instruction_tuning/sft.py", "posttraining_harness/assets.py", "posttraining_harness/hardware.py",
        "posttraining_harness/colab_gate.py", "posttraining_harness/sft_trainer_gate.py",
    ]
    payload = {name: (root / name).read_bytes() for name in names}
    payload["input/sample.jsonl"] = sample.read_bytes()
    git = lambda *arguments: subprocess.check_output(
        ["git", *arguments], cwd=root, text=True
    ).strip()
    manifest = {
        "created_utc": stamp,
        "git_head": git("rev-parse", "HEAD"),
        "git_branch": git("branch", "--show-current"),
        "git_status": git("status", "--short"),
        "scope": "Actual SFTTrainer masking gate; no optimizer update",
        "model": MODEL,
        "model_revision": MODEL_REVISION,
        "sample_source": str(sample),
        "files": {name: hashlib.sha256(data).hexdigest() for name, data in payload.items()},
        "uv_version": "0.8.13",
        "python_version": "3.12.10",
    }
    archive = out / f"sft-trainer-gate-inputs-{stamp}.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
        for name, data in payload.items():
            bundle.writestr(name, data)
        bundle.writestr("bundle-manifest.json", json.dumps(manifest, indent=2) + "\n")
    archive_sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    (out / "bundle-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    driver = textwrap.dedent(f'''\
        """Run the pinned item-6 SFTTrainer masking gate on one Colab T4."""
        import hashlib, json, os, shutil, subprocess, sys, zipfile
        from datetime import datetime, timezone
        from pathlib import Path

        import torch

        ARCHIVE = Path("/content/{archive.name}")
        ARCHIVE_SHA = "{archive_sha}"
        MODEL = "{MODEL}"
        MODEL_REVISION = "{MODEL_REVISION}"

        def run_logged(name, command, cwd, env, log_path):
            print("\\n>>>", name, flush=True)
            with log_path.open("w") as log:
                process = subprocess.Popen(list(map(str, command)), cwd=cwd, env=env,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                for line in process.stdout:
                    log.write(line); log.flush(); print(line, end="", flush=True)
                code = process.wait()
            if code:
                print(name, "failed; log tail:", flush=True)
                print("".join(log_path.read_text().splitlines(keepends=True)[-80:]), flush=True)
            assert code == 0, (name, code)

        assert torch.cuda.is_available(), "Allocate a Colab GPU session."
        assert ARCHIVE.is_file(), ARCHIVE
        assert hashlib.sha256(ARCHIVE.read_bytes()).hexdigest() == ARCHIVE_SHA
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        work = Path("/content/posttraining-sft-runs") / run_id
        project, results = work / "code", work / "results"
        project.mkdir(parents=True); results.mkdir()
        with zipfile.ZipFile(ARCHIVE) as bundle:
            manifest = json.loads(bundle.read("bundle-manifest.json"))
            assert set(bundle.namelist()) == set(manifest["files"]) | {{"bundle-manifest.json"}}
            for name, expected in manifest["files"].items():
                data = bundle.read(name)
                assert hashlib.sha256(data).hexdigest() == expected, name
                destination = (project / name).resolve()
                assert destination.is_relative_to(project.resolve()), name
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(data)
            (project / "bundle-manifest.json").write_text(json.dumps(manifest, indent=2) + "\\n")

        environment = results / "environment"
        environment.mkdir()
        gate = subprocess.run([sys.executable, "-u", project / "posttraining_harness/colab_gate.py",
            "--project", project, "--output", environment], text=True)
        assert gate.returncode == 0, "locked environment gate failed"
        python = project / ".venv/bin/python"
        env = os.environ.copy()
        env.pop("VIRTUAL_ENV", None); env.pop("PYTHONPATH", None)
        env.update(PYTHONNOUSERSITE="1", PYTHONUNBUFFERED="1", WANDB_MODE="disabled",
                   MPLBACKEND="Agg", HF_HOME=str(work / "hf-cache"))
        model = work / "model-snapshot"
        download = [python, "-c",
            "from huggingface_hub import snapshot_download; "
            f"snapshot_download(repo_id={{MODEL!r}}, revision={{MODEL_REVISION!r}}, "
            f"local_dir={{str(model)!r}})"]
        run_logged("pinned-model-download", download, project, env, results / "download.log")
        masking = results / "masking"
        command = [python, "-u", project / "posttraining_harness/sft_trainer_gate.py",
            "--model", model, "--model-revision", MODEL_REVISION,
            "--sample", project / "input/sample.jsonl", "--output", masking,
            "--reference-sft", project / "instruction_tuning/sft.py"]
        run_logged("actual-trainer-batches", command, project, env, results / "masking.log")
        report = json.loads((masking / "sft_trainer_gate.json").read_text())
        summary = {{
            "run_id": run_id, "finished_utc": datetime.now(timezone.utc).isoformat(),
            "input_bundle_sha256": ARCHIVE_SHA, "report": report, "pass": report["pass"],
        }}
        (results / "summary.json").write_text(json.dumps(summary, indent=2) + "\\n")
        result_archive = Path(f"/content/sft-trainer-gate-results-{{run_id}}.zip")
        with zipfile.ZipFile(result_archive, "w", zipfile.ZIP_DEFLATED) as bundle:
            for path in sorted(results.rglob("*")):
                if path.is_file(): bundle.write(path, path.relative_to(results))
        print("\\nSFT TRAINER GATE SUMMARY\\n" + json.dumps(summary, indent=2), flush=True)
        print("RESULT_ARCHIVE:", result_archive, flush=True)
        assert summary["pass"]
    ''')
    driver_path = out / "sft-trainer-gate-colab.py"
    compile(driver, str(driver_path), "exec")
    driver_path.write_text(driver)
    print(json.dumps({
        "directory": str(out), "archive": str(archive),
        "archive_sha256": archive_sha, "driver": str(driver_path),
    }, indent=2))


if __name__ == "__main__":
    main()
