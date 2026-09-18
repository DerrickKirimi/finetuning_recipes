"""Run and verify the CPT entry point on a tiny private Colab fixture."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys


ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_training_log(text):
    """Extract per-step loss and learning-rate records from Trainer output."""
    clean = ANSI.sub("", text).replace("\r", "\n")
    losses = [float(value) for value in re.findall(
        rf"(?<![A-Za-z_])['\"]loss['\"]\s*:\s*({NUMBER})", clean
    )]
    learning_rates = [float(value) for value in re.findall(
        rf"['\"]learning_rate['\"]\s*:\s*({NUMBER})", clean
    )]
    train_losses = [float(value) for value in re.findall(
        rf"['\"]train_loss['\"]\s*:\s*({NUMBER})", clean
    )]
    return {
        "step_losses": losses,
        "learning_rates_reported": learning_rates,
        "train_losses": train_losses,
    }


def write_fixture(directory):
    """Write deterministic train/eval text rows; outputs stay outside the code tree."""
    directory.mkdir(parents=True, exist_ok=True)
    passages = [
        (
            "Gradient descent updates model parameters using derivatives of a loss function. "
            "A learning-rate schedule controls the magnitude of each update. "
            "Warmup begins with small learning rates and increases them gradually. "
            "Reproducible experiments record data, software, hardware, seeds, and commands. "
        ) * 8,
        (
            "Neural language models estimate conditional token probabilities from context. "
            "Continued pre-training adapts those probabilities to a selected text domain. "
            "A held-out corpus measures whether loss transfers beyond the training examples. "
            "Careful evaluation separates successful execution from a quality improvement. "
        ) * 8,
    ]
    eval_passage = (
        "Optimization tests should distinguish a finite forward pass from a parameter update. "
        "Saved metadata connects a result to its exact source and dependency environment. "
        "A small smoke run detects interface and device failures before expensive training. "
    ) * 8
    paths = {"train": directory / "train.jsonl", "eval": directory / "eval.jsonl"}
    paths["train"].write_text("".join(json.dumps({"text": text}) + "\n" for text in passages))
    paths["eval"].write_text(json.dumps({"text": eval_passage}) + "\n")
    return paths


def adapter_file(final_directory):
    candidates = sorted(final_directory.glob("adapter_model*.safetensors"))
    if not candidates:
        candidates = sorted(final_directory.glob("*.safetensors"))
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one saved adapter tensor file, found {candidates}")
    return candidates[0]


def compare_adapters(baseline_path, candidate_path):
    from safetensors.torch import load_file

    baseline = load_file(str(baseline_path), device="cpu")
    candidate = load_file(str(candidate_path), device="cpu")
    if baseline.keys() != candidate.keys():
        raise RuntimeError("Adapter tensor keys differ between the formal and optional runs")
    changed = {}
    for name in baseline:
        delta = (candidate[name].float() - baseline[name].float()).abs().max().item()
        if delta:
            changed[name] = delta
    return {
        "tensor_count": len(baseline),
        "changed_tensor_count": len(changed),
        "maximum_absolute_delta": max(changed.values(), default=0.0),
    }


def compare_adapter_run(status, baseline_path):
    """Require the corpus-adapter run to reproduce the formal gate's inputs and loss."""
    baseline_path = baseline_path.resolve()
    baseline = json.loads(baseline_path.read_text())
    assert baseline["pass"] is True
    assert baseline["mode"] == "formal"
    assert baseline["expected_steps"] == 1
    assert baseline["model"] == status["model"]
    assert baseline["model_revision"] == status["model_revision"]
    assert {
        name: details["sha256"] for name, details in baseline["inputs"].items()
    } == {
        name: details["sha256"] for name, details in status["inputs"].items()
    }
    assert baseline["trainer_output"]["step_losses"] == status["trainer_output"]["step_losses"]
    assert baseline["trainer_output"]["train_losses"] == status["trainer_output"]["train_losses"]
    return {
        "report": str(baseline_path),
        "report_sha256": sha256(baseline_path),
        "step_losses_equal": True,
        "train_losses_equal": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument(
        "--mode", choices=("formal", "checkpoint", "adapter"), default="formal"
    )
    parser.add_argument("--baseline-adapter", type=Path)
    parser.add_argument("--baseline-report", type=Path)
    args = parser.parse_args()

    project, output = args.project.resolve(), args.output.resolve()
    if not Path("/content").is_dir() or not shutil.which("nvidia-smi"):
        raise SystemExit("Run this gate on a Colab GPU server, not locally.")
    if output.is_relative_to(project):
        raise SystemExit("Outputs must be outside the public code directory.")
    if args.mode == "checkpoint" and not args.baseline_adapter:
        raise SystemExit("--baseline-adapter is required for checkpoint mode")
    if args.mode == "adapter" and not args.baseline_report:
        raise SystemExit("--baseline-report is required for adapter mode")

    output.mkdir(parents=True, exist_ok=False)
    log_path = output / "training.log"
    report_path = output / "report.json"
    expected_steps = 20 if args.mode == "checkpoint" else 1
    status = {
        "pass": False,
        "mode": args.mode,
        "scope": {
            "formal": "One unchanged-reference optimizer step and finite loss",
            "checkpoint": (
                "Fresh 20-step unchanged-reference run; checkpoint and parameter-change check"
            ),
            "adapter": "One corpus-adapter step reproducing the formal gate loss",
        }[args.mode],
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "expected_steps": expected_steps,
        "reference_script": str(project / "cpt/sft.py"),
        "reference_script_sha256": sha256(project / "cpt/sft.py"),
        "corpus_adapter_sha256": (
            sha256(project / "cpt/corpus.py")
            if (project / "cpt/corpus.py").is_file()
            else None
        ),
        "model": "HuggingFaceTB/SmolLM-135M",
        "model_revision": args.model_revision,
    }

    def save():
        report_path.write_text(json.dumps(status, indent=2) + "\n")

    try:
        import torch
        from huggingface_hub import snapshot_download

        assert torch.cuda.is_available(), "CUDA is unavailable"
        status["hardware"] = {
            "name": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
            "native_bf16": torch.cuda.is_bf16_supported(including_emulation=False),
        }
        work = output / "work"
        fixture = write_fixture(work / "input")
        status["inputs"] = {
            name: {"path": str(path), "sha256": sha256(path)}
            for name, path in fixture.items()
        }
        model_path = Path(snapshot_download(
            repo_id=status["model"],
            revision=args.model_revision,
            local_dir=output.parent / "model-snapshot",
        )).resolve()
        status["model_snapshot"] = {
            "path": str(model_path),
            "commit": args.model_revision,
            "files": {
                str(path.relative_to(model_path)): sha256(path)
                for path in sorted(model_path.rglob("*"))
                if path.is_file() and ".cache" not in path.parts
            },
        }
        run_dir = work / "run"
        run_dir.mkdir(parents=True)
        output_id = {
            "formal": "cpt_gate_one_update",
            "checkpoint": "cpt_gate_checkpoint",
            "adapter": "cpt_gate_corpus_adapter",
        }[args.mode]
        command = [
            sys.executable, "-u", str(project / "cpt/sft.py"),
            "--base_model_id", str(model_path),
            "--output_model_id", output_id,
            "--dataset_path", str(fixture["train"]),
            "--test_dataset_path", str(fixture["eval"]),
            "--batch_size", "32",
            "--epochs", str(expected_steps),
            "--max_seq_length", "128",
            "--split_by_words", "0",
        ]
        status["command"] = command
        status["cwd"] = str(run_dir)
        save()
        env = os.environ.copy()
        env.update(
            PYTHONUNBUFFERED="1",
            PYTHONNOUSERSITE="1",
            TOKENIZERS_PARALLELISM="false",
            WANDB_MODE="disabled",
            HF_HOME=str(output.parent / "hf-cache"),
        )
        with log_path.open("w") as log:
            process = subprocess.Popen(
                command,
                cwd=run_dir,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
            status["exit_code"] = process.wait()
        parsed = parse_training_log(log_path.read_text(errors="replace"))
        status["trainer_output"] = parsed
        assert status["exit_code"] == 0, f"CPT exited {status['exit_code']}"
        assert len(parsed["step_losses"]) == expected_steps, parsed
        assert all(math.isfinite(value) for value in parsed["step_losses"]), parsed
        assert parsed["train_losses"] and all(
            math.isfinite(value) for value in parsed["train_losses"]
        ), parsed
        final = run_dir / "models" / output_id / "final"
        assert final.is_dir(), f"Missing final model directory: {final}"
        adapter = adapter_file(final)
        status["final_adapter"] = {"path": str(adapter), "sha256": sha256(adapter)}
        if args.mode == "adapter":
            status["baseline_comparison"] = compare_adapter_run(
                status, args.baseline_report
            )
        if args.mode == "checkpoint":
            checkpoint = run_dir / "models" / output_id / "checkpoint-20"
            assert checkpoint.is_dir(), f"Missing reference checkpoint: {checkpoint}"
            assert (checkpoint / "trainer_state.json").is_file()
            comparison = compare_adapters(args.baseline_adapter.resolve(), adapter)
            assert comparison["changed_tensor_count"] > 0, comparison
            status["checkpoint"] = str(checkpoint)
            status["adapter_comparison"] = comparison
        status["pass"] = True
    except Exception as exc:
        status["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        status["finished_utc"] = datetime.now(timezone.utc).isoformat()
        save()
        print("\nCPT GATE RESULT\n" + json.dumps(status, indent=2), flush=True)


if __name__ == "__main__":
    main()
