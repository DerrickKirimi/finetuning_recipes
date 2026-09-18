"""Offline tests of grouped-split verification and selection for DPO training. No model, no network."""
import copy
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

from datasets import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "preference_optimization"))
from split_manifest import SplitError, prompt_key, select_split, sha256_file, verify_split_files  # noqa: E402


def rows(n):
    return [{"prompt": [{"role": "user", "content": f"Answer it\n\nPassage {i}."}],
             "chosen": [{"role": "assistant", "content": "yes"}],
             "rejected": [{"role": "assistant", "content": "no"}]} for i in range(n)]


class Fixture:
    """A dataset file, manifest and spot-check that verify, written to a temporary directory."""

    def __init__(self, root: Path, n=10, train=(1, 2, 5), val=(8,)):
        self.root, self.data = root, rows(n)
        self.dataset_file = root / "train.jsonl"
        self.dataset_file.write_text("".join(json.dumps(r) + "\n" for r in self.data))
        self.manifest = {"sources": {"dpo_train": {"rows": n, "sha256": sha256_file(self.dataset_file)}},
                         "train_indices": list(train), "validation_indices": list(val),
                         "train_rows": len(train), "validation_rows": len(val),
                         "checks": {"indices_disjoint": True, "train_validation_prompt_overlap": 0},
                         "all_checks_pass": True}
        self.write()

    def write(self, manifest=None, spot_rows=None, spot_manifest_sha=None):
        self.manifest_path = self.root / "manifest.json"
        self.manifest_path.write_text(json.dumps(manifest or self.manifest))
        self.manifest_sha = sha256_file(self.manifest_path)
        spot = {"manifest_sha256": spot_manifest_sha or self.manifest_sha,
                "rows": spot_rows if spot_rows is not None else
                [{"index": i, "prompt_key": prompt_key(self.data[i])} for i in (1, 8)]}
        self.spot_path = self.root / "spot.json"
        self.spot_path.write_text(json.dumps(spot))
        self.spot_sha = sha256_file(self.spot_path)

    def verify(self, **overrides):
        kwargs = dict(manifest_sha256=self.manifest_sha, spotcheck_sha256=self.spot_sha, dataset_file=self.dataset_file)
        kwargs.update(overrides)
        return verify_split_files(self.manifest_path, self.spot_path, **kwargs)


class VerifyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.fx = Fixture(Path(self.tmp.name))

    def tearDown(self):
        self.tmp.cleanup()

    def mutated(self, **changes):
        manifest = copy.deepcopy(self.fx.manifest)
        manifest.update(changes)
        self.fx.write(manifest=manifest)
        return self.fx

    def test_a_consistent_split_verifies_and_selects_in_order(self):
        manifest, spot = self.fx.verify()
        train, val = select_split(Dataset.from_list(self.fx.data), manifest, spot)
        self.assertEqual([r["prompt"][0]["content"] for r in train],
                         [self.fx.data[i]["prompt"][0]["content"] for i in (1, 2, 5)])
        self.assertEqual(len(val), 1)

    def test_wrong_manifest_or_spotcheck_hash_is_refused(self):
        with self.assertRaises(SplitError):
            self.fx.verify(manifest_sha256="0" * 64)
        with self.assertRaises(SplitError):
            self.fx.verify(spotcheck_sha256="0" * 64)

    def test_changed_dataset_content_is_refused_even_with_same_row_count(self):
        data = rows(10)
        data[3]["chosen"][0]["content"] = "a different chosen answer"
        self.fx.dataset_file.write_text("".join(json.dumps(r) + "\n" for r in data))
        with self.assertRaises(SplitError) as ctx:
            self.fx.verify()
        self.assertIn("dataset file sha256", str(ctx.exception))

    def test_spotcheck_for_another_manifest_is_refused(self):
        self.fx.write(spot_manifest_sha="f" * 64)
        with self.assertRaises(SplitError):
            self.fx.verify()

    def test_failed_or_missing_checks_are_refused(self):
        for checks in ({"indices_disjoint": False}, {"train_validation_prompt_overlap": 2}, {}):
            with self.subTest(checks=checks):
                self.mutated(checks=checks)
                with self.assertRaises(SplitError):
                    self.fx.verify()
        self.mutated(checks={"indices_disjoint": True}, all_checks_pass=False)
        with self.assertRaises(SplitError):
            self.fx.verify()

    def test_bad_indices_are_refused(self):
        cases = [dict(train_indices=[1, 1, 5]), dict(train_indices=[1, 2, 99]), dict(train_indices=[1, True, 5]),
                 dict(train_indices=[1, 2]), dict(validation_indices=[], validation_rows=0),
                 dict(train_indices=[1, 2, 8])]
        for change in cases:
            with self.subTest(change=change):
                self.mutated(**change)
                with self.assertRaises(SplitError):
                    self.fx.verify()

    def test_empty_spotcheck_is_refused(self):
        self.fx.write(spot_rows=[])
        with self.assertRaises(SplitError):
            self.fx.verify()


class SelectTests(unittest.TestCase):
    def test_reordered_rows_fail_the_spot_check(self):
        with tempfile.TemporaryDirectory() as d:
            fx = Fixture(Path(d))
            manifest, spot = fx.verify()
            with self.assertRaises(SplitError) as ctx:
                select_split(Dataset.from_list(list(reversed(fx.data))), manifest, spot)
            self.assertIn("does not match", str(ctx.exception))

    def test_prompt_key_ignores_system_messages_and_whitespace(self):
        data = rows(10)
        with_system = {"prompt": [{"role": "system", "content": "sys"},
                                  {"role": "user", "content": "Answer   it\n\nPassage 3."}]}
        self.assertEqual(prompt_key(with_system), prompt_key(data[3]))
        self.assertEqual(len(prompt_key(data[0])), len(hashlib.sha256(b"").hexdigest()))


if __name__ == "__main__":
    unittest.main()
