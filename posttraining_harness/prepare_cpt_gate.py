"""Package exact CPT gate inputs and generate a private Colab execution notebook."""

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import textwrap
import zipfile

from assets import output_directory


MODEL_REVISION = "1d461723eec654e65efdc40cf49301c89c0c92f4"


def cell(kind, source):
    result = {
        "cell_type": kind,
        "metadata": {},
        "source": textwrap.dedent(source).strip().splitlines(keepends=True),
    }
    if kind == "code":
        result.update(execution_count=None, outputs=[])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    out = output_directory(args.output) / stamp
    out.mkdir()
    names = [
        "pyproject.toml",
        "uv.lock",
        "README.md",
        "check_env.sh",
        "cpt/sft.py",
        "posttraining_harness/assets.py",
        "posttraining_harness/hardware.py",
        "posttraining_harness/colab_gate.py",
        "posttraining_harness/cpt_gate.py",
    ]
    payload = {name: (root / name).read_bytes() for name in names}
    git = lambda *arguments: subprocess.check_output(
        ["git", *arguments], cwd=root, text=True
    ).strip()
    manifest = {
        "created_utc": stamp,
        "git_head": git("rev-parse", "HEAD"),
        "git_branch": git("branch", "--show-current"),
        "git_status": git("status", "--short"),
        "scope": "Exact environment and unchanged CPT gate inputs; outputs excluded",
        "model": "HuggingFaceTB/SmolLM-135M",
        "model_revision": MODEL_REVISION,
        "files": {
            name: hashlib.sha256(data).hexdigest() for name, data in payload.items()
        },
        "uv_version": "0.8.13",
        "python_version": "3.12.10",
    }
    archive = out / f"colab-cpt-inputs-{stamp}.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
        for name, data in payload.items():
            bundle.writestr(name, data)
        bundle.writestr("bundle-manifest.json", json.dumps(manifest, indent=2) + "\n")
    archive_hash = hashlib.sha256(archive.read_bytes()).hexdigest()
    (out / "bundle-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "accelerator": "GPU",
            "kernelspec": {
                "display_name": "Python 3",
                "name": "python3",
                "language": "python",
            },
            "language_info": {"name": "python"},
        },
        "cells": [],
    }
    notebook["cells"].append(cell("markdown", f"""
        # CPT optimizer gate

        Open this local notebook in VS Code with the official Colab extension. Select a
        T4 GPU server, upload `{archive.name}` to `/content`, and run cells in order.
        The environment cell installs the locked project in a fresh Python 3.12.10
        environment. The formal cell then invokes the bundled, unchanged `cpt/sft.py`
        for exactly one logged optimizer step. Save the notebook after every cell.

        The optional 20-step cell is deliberately separate. Run it only after the
        formal cell prints `FORMAL CPT GATE PASSED`. It reaches the reference script's
        hardcoded `save_steps=20`, checks `checkpoint-20`, and compares adapter tensors.
    """))
    setup = """
        import hashlib, json, shutil, subprocess, sys, zipfile
        from datetime import datetime, timezone
        from pathlib import Path
        import torch

        assert torch.cuda.is_available(), "Select a Colab GPU server first."
        archive = Path("/content") / ARCHIVE_NAME
        assert archive.is_file(), f"Upload {archive.name} to /content first."
        assert hashlib.sha256(archive.read_bytes()).hexdigest() == ARCHIVE_HASH
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        work = Path("/content/posttraining-cpt-runs") / run_id
        project, results = work / "code", work / "results"
        project.mkdir(parents=True)
        results.mkdir()
        with zipfile.ZipFile(archive) as bundle:
            manifest = json.loads(bundle.read("bundle-manifest.json"))
            assert set(bundle.namelist()) == set(manifest["files"]) | {"bundle-manifest.json"}
            for name, expected in manifest["files"].items():
                destination = (project / name).resolve()
                assert destination.is_relative_to(project.resolve()), name
                data = bundle.read(name)
                assert hashlib.sha256(data).hexdigest() == expected, name
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(data)
            (project / "bundle-manifest.json").write_text(json.dumps(manifest, indent=2) + "\\n")
        print("GPU:", torch.cuda.get_device_name(0))
        print("Native BF16:", torch.cuda.is_bf16_supported(including_emulation=False))
        print("Fresh run directory:", work)
        print("Bundle verified:", ARCHIVE_HASH)
    """
    setup = setup.replace("ARCHIVE_NAME", repr(archive.name)).replace(
        "ARCHIVE_HASH", repr(archive_hash)
    )
    notebook["cells"].append(cell("code", setup))
    notebook["cells"].append(cell("markdown", """
        ## Install and verify the locked environment

        This repeats the successful environment gate because Colab runtimes are
        ephemeral. It performs no training. Stop and save the output if it fails.
    """))
    notebook["cells"].append(cell("code", """
        environment_output = results / "environment"
        environment_command = [sys.executable, "-u", str(project / "posttraining_harness/colab_gate.py"),
                               "--project", str(project), "--output", str(environment_output)]
        process = subprocess.run(environment_command, text=True)
        assert process.returncode == 0, "Locked environment gate failed. Save and return the output."
        print("LOCKED ENVIRONMENT PASSED")
    """))
    notebook["cells"].append(cell("markdown", """
        ## Formal gate: one optimizer step

        This is the pass/fail CPT gate. The first learning rate is zero because the
        unchanged reference script hardcodes 100 warmup steps. This cell therefore
        requires one reported step and finite loss; parameter change is checked later.
    """))
    formal = """
        python = project / ".venv/bin/python"
        formal_output = results / "formal"
        formal_command = [python, "-u", project / "posttraining_harness/cpt_gate.py",
                          "--project", project, "--output", formal_output,
                          "--model-revision", MODEL_REVISION, "--mode", "formal"]
        formal_command = list(map(str, formal_command))
        process = subprocess.run(formal_command, text=True)
        if (formal_output / "report.json").exists():
            print((formal_output / "report.json").read_text())
        assert process.returncode == 0, "Formal CPT gate failed. Save and return this notebook."
        formal_report = json.loads((formal_output / "report.json").read_text())
        assert formal_report["pass"] and formal_report["expected_steps"] == 1
        print("FORMAL CPT GATE PASSED. Save the notebook now.")
    """.replace("MODEL_REVISION", repr(MODEL_REVISION))
    notebook["cells"].append(cell("code", formal))
    notebook["cells"].append(cell("markdown", """
        ## Optional: checkpoint writer and parameter changes

        Run only after the formal gate passes. This is a fresh 20-step invocation from
        the same pinned base and fixture, not a resume, because the unchanged CLI has no
        resume argument. It checks the reference `checkpoint-20` writer and compares its
        final adapter with the one-step adapter.
    """))
    optional = """
        checkpoint_output = results / "checkpoint"
        baseline_adapter = Path(formal_report["final_adapter"]["path"])
        checkpoint_command = [python, "-u", project / "posttraining_harness/cpt_gate.py",
                              "--project", project, "--output", checkpoint_output,
                              "--model-revision", MODEL_REVISION, "--mode", "checkpoint",
                              "--baseline-adapter", baseline_adapter]
        checkpoint_command = list(map(str, checkpoint_command))
        process = subprocess.run(checkpoint_command, text=True)
        if (checkpoint_output / "report.json").exists():
            print((checkpoint_output / "report.json").read_text())
        assert process.returncode == 0, "Optional checkpoint run failed. Save and return this notebook."
        print("OPTIONAL CHECKPOINT RUN PASSED. Save the notebook now.")
    """.replace("MODEL_REVISION", repr(MODEL_REVISION))
    notebook["cells"].append(cell("code", optional))
    notebook["cells"].append(cell("markdown", """
        ## Optional: persist the run evidence to Drive

        This excludes the reproducible base-model snapshot and package caches. It keeps
        reports, logs, fixtures, the small adapters and checkpoint state.
    """))
    notebook["cells"].append(cell("code", """
        from google.colab import drive
        if not Path("/content/drive/MyDrive").is_dir():
            drive.mount("/content/drive")
        destination = Path("/content/drive/MyDrive/posttraining-results/cpt-gate") / run_id
        shutil.copytree(results, destination, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("model-snapshot", "hf-cache"))
        print("Persistent evidence:", destination)
    """))
    for index, notebook_cell in enumerate(notebook["cells"]):
        notebook_cell["id"] = f"cpt-gate-{index}"
        if notebook_cell["cell_type"] == "code":
            ast.parse("".join(notebook_cell["source"]))
    notebook_path = out / "cpt-optimizer-gate.ipynb"
    notebook_path.write_text(json.dumps(notebook, indent=2) + "\n")
    print(json.dumps({
        "directory": str(out),
        "archive": str(archive),
        "archive_sha256": archive_hash,
        "notebook": str(notebook_path),
    }, indent=2))


if __name__ == "__main__":
    main()
