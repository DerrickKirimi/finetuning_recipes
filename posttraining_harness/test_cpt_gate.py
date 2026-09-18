import json
import math

import pytest

from cpt_gate import compare_adapter_run, parse_training_log


def test_parse_training_log_separates_step_and_summary_losses():
    parsed = parse_training_log(
        "\x1b[32m{'loss': 2.5, 'grad_norm': 1.0, 'learning_rate': 2e-06}\x1b[0m\r"
        "{'train_runtime': 1.0, 'train_loss': 2.5}"
    )
    assert parsed == {
        "step_losses": [2.5],
        "learning_rates_reported": [2e-6],
        "train_losses": [2.5],
    }
    assert all(math.isfinite(value) for value in parsed["step_losses"])


def test_adapter_comparison_requires_same_inputs_model_and_losses(tmp_path):
    inputs = {
        "train": {"sha256": "train-hash"},
        "eval": {"sha256": "eval-hash"},
    }
    baseline = {
        "pass": True,
        "mode": "formal",
        "expected_steps": 1,
        "model": "model",
        "model_revision": "revision",
        "inputs": inputs,
        "trainer_output": {"step_losses": [2.0], "train_losses": [2.0]},
    }
    path = tmp_path / "baseline.json"
    path.write_text(json.dumps(baseline))
    status = {
        "model": "model",
        "model_revision": "revision",
        "inputs": inputs,
        "trainer_output": {"step_losses": [2.0], "train_losses": [2.0]},
    }

    comparison = compare_adapter_run(status, path)

    assert comparison["step_losses_equal"] is True
    assert comparison["train_losses_equal"] is True
    status["trainer_output"]["step_losses"] = [2.1]
    with pytest.raises(AssertionError):
        compare_adapter_run(status, path)
