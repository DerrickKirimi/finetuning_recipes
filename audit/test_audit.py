from types import SimpleNamespace

from datasets import Dataset
import numpy as np
import pytest

from assets import output_directory
from data_boundaries import identities, overlap
from hardware import settings


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
