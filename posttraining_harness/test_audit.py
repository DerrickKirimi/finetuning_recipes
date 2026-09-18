import json
from types import SimpleNamespace

from datasets import Dataset
import numpy as np
import pytest

from assets import output_directory
from data_boundaries import identities, overlap
from hardware import settings
from schema_index_probe import parse_events, parser
from sft_split import build
from sft_trainer_gate import find_subsequence, inspect_batch, logical_spans


def test_split_matches_installed_datasets():
    split = Dataset.from_dict({"index": list(range(150))}).train_test_split(test_size=.02, seed=3407)
    indexes = np.random.default_rng(3407).permutation(150)
    assert split["test"]["index"] == indexes[:3].tolist()
    assert split["train"]["index"] == indexes[3:].tolist()


def test_identity_ignores_answer_and_whitespace_but_keeps_input():
    row = {"instruction": " Say\nwhy ", "input": "A  B", "output": "x"}
    same = {"instruction": "Say why", "input": "A B", "output": "different answer"}
    assert identities(row) == identities(same)
    assert identities(row)[0] != identities(dict(row, input="Other"))[0]
    assert overlap(["a", None], ["a", "a", "b", None]) == {
        "shared_unique_keys": 1, "right_rows_matching_left": 2, "right_rows_with_key": 3}


@pytest.mark.parametrize("cuda,bf16,dtype", [(False, False, "float32"),
    (True, False, "float16"), (True, True, "bfloat16")])
def test_hardware_capability_branches(cuda, bf16, dtype):
    torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: cuda,
        is_bf16_supported=lambda including_emulation=True: True if including_emulation else bf16))
    profile = settings(torch)
    assert profile["dtype"] == dtype
    assert profile["bf16"] is (cuda and bf16)
    assert profile["fp16"] is (cuda and not bf16)
    assert profile["attn_implementation"] == "sdpa"
    assert not profile["packing"] and not profile["padding_free"]


def test_output_guard_rejects_public_tree():
    with pytest.raises(ValueError):
        output_directory(__file__)


def test_schema_probe_keeps_only_json_events():
    output = 'noise\n{"event":"phase","phase":"regex_built"}\n{"other":1}\n'
    assert parse_events(output) == [{"event": "phase", "phase": "regex_built"}]


def test_schema_probe_child_receives_a_single_length_list():
    args = parser().parse_args(
        ["--child", "--tokenizer", "unused", "--max-length", "1000"]
    )
    assert args.max_length == [1000]


def test_schema_probe_can_request_uncapped_preferred_oom_child():
    args = parser().parse_args(
        [
            "--tokenizer",
            "unused",
            "--max-length",
            "20000",
            "--memory-gib",
            "0",
            "--prefer-child-oom-kill",
        ]
    )
    assert args.memory_gib == 0
    assert args.prefer_child_oom_kill


def test_sft_split_groups_prompt_and_passage_and_builds_clean_battery(tmp_path):
    train = tmp_path / "train.jsonl"
    test = tmp_path / "test.jsonl"
    rows = [
        {"instruction": "p1", "input": "shared passage", "output": "a"},
        {"instruction": "p2", "input": "shared passage", "output": "b"},
        {"instruction": "p1", "input": "another passage", "output": "c"},
        {"instruction": "p3", "input": "third", "output": "d"},
        {"instruction": "p4", "input": "fourth", "output": "e"},
    ]
    test_rows = [
        {"instruction": "p1", "input": "new", "output": "overlap prompt"},
        {"instruction": "new", "input": "shared passage", "output": "overlap passage"},
        {"instruction": "clean1", "input": "clean passage 1", "output": "x"},
        {"instruction": "clean2", "input": "clean passage 2", "output": "y"},
    ]
    train.write_text("".join(json.dumps(row) + "\n" for row in rows))
    test.write_text("".join(json.dumps(row) + "\n" for row in test_rows))
    report = build(train, test, validation_fraction=0.2, battery_size=2, seed=3407)
    split = report["internal_validation"]
    assert split["cross_split_overlap"] == {"prompt_rows": 0, "passage_rows": 0}
    connected = {0, 1, 2}
    assert connected.issubset(split["train_indices"]) or connected.issubset(
        split["validation_indices"]
    )
    assert report["fixed_test_battery"]["indices"] == [2, 3]
    assert report["fixed_test_battery"]["overlap_with_full_training_source"] == {
        "prompt_rows": 0, "passage_rows": 0
    }


def test_sft_trainer_gate_detects_prompt_leak_across_packed_position_reset():
    import torch

    assistant = [7, 8]
    # Two logical examples flattened into one physical row. The second starts where
    # position_ids resets; its prompt labels deliberately leak into the loss.
    batch = {
        "input_ids": torch.tensor([[1, 7, 8, 20, 21, 1, 2, 7, 8, 30]]),
        "labels": torch.tensor([[-100, -100, -100, 20, 21, 1, 2, -100, -100, 30]]),
        "position_ids": torch.tensor([[0, 1, 2, 3, 4, 0, 1, 2, 3, 4]]),
    }
    assert find_subsequence(batch["input_ids"][0].tolist(), assistant) == 1
    assert logical_spans(batch["input_ids"].tolist(),
                         batch["position_ids"].tolist(), None) == [(0, 0, 5), (0, 5, 10)]
    result = inspect_batch(batch, assistant)
    assert result["logical_examples"] == 2
    assert result["prompt_tokens_in_loss"] == 2
    assert result["response_tokens_masked"] == 0
    assert result["fully_masked_examples"] == 0
