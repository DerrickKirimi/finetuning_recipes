import json
import tempfile
import unittest
from pathlib import Path

import torch
from transformers import BertConfig, BertModel, BertTokenizerFast

from reward_models.adapt_grouped_reward import (
    AdaptationConfig,
    grouped_loss,
    load_grouped_records,
    run_adaptation,
)
from reward_models.reward_model import load_reward_model
from reward_models.reward_training import RewardRegressor, save_export


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


class GroupedRewardAdaptationTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        source = self.root / "source"
        source.mkdir()
        vocabulary = [
            "[PAD]",
            "[UNK]",
            "[CLS]",
            "[SEP]",
            "[MASK]",
            "reference",
            "right",
            "wrong",
            "answer",
        ]
        (source / "vocab.txt").write_text("\n".join(vocabulary) + "\n", encoding="utf-8")
        tokenizer = BertTokenizerFast(vocab_file=str(source / "vocab.txt"))
        tokenizer.save_pretrained(source)
        torch.manual_seed(123)
        BertModel(
            BertConfig(
                vocab_size=len(tokenizer),
                hidden_size=12,
                num_hidden_layers=2,
                num_attention_heads=2,
                intermediate_size=16,
                max_position_embeddings=32,
            )
        ).save_pretrained(source)
        model = RewardRegressor(
            str(source),
            model_revision=None,
            dropout=0.2,
            unfreeze_layers=2,
            pooling="meanmax",
            attention_implementation="eager",
            local_files_only=True,
        )
        self.parent = self.root / "parent"
        save_export(
            self.parent,
            model,
            tokenizer,
            {
                "tokenization_contract": "balanced_pair",
                "pooling": "meanmax",
                "max_length": 16,
            },
        )

        self.data = self.root / "groups.jsonl"
        rows = []
        for fold in range(5):
            for rollout_index in range(2):
                rows.append(
                    {
                        "group_id": f"group-{fold}",
                        "rollout_index": rollout_index,
                        "fold": fold,
                        "reference": "reference answer",
                        "response": "right answer" if rollout_index else "wrong answer",
                        "score": float(rollout_index),
                        "scoreable": not (fold == 4 and rollout_index == 0),
                    }
                )
        write_jsonl(self.data, rows)
        self.candidates = self.root / "candidates.json"
        write_json(
            self.candidates,
            [
                {
                    "name": "tiny",
                    "epochs": 1,
                    "learning_rate_head": 0.001,
                    "learning_rate_encoder": 0.0001,
                    "advantage_beta": 0.1,
                    "raw_score_weight": 0.25,
                }
            ],
        )

    def tearDown(self):
        self.temporary.cleanup()

    def config(self, output: Path) -> AdaptationConfig:
        return AdaptationConfig(
            data_file=str(self.data),
            parent_model_dir=str(self.parent),
            candidates_file=str(self.candidates),
            output_dir=str(output),
            folds=5,
            group_size=2,
            groups_per_batch=2,
            max_length=16,
            unfreeze_layers=2,
            seed=19,
            device="cpu",
            development_mae_gate=1.0,
        )

    def test_grouped_cross_validation_exports_and_reuses_terminal_result(self):
        output = self.root / "output"
        first = run_adaptation(self.config(output))
        self.assertEqual(first["status"], "development_pass")
        self.assertEqual(first["selected_candidate"], "tiny")
        self.assertEqual(len(first["candidates"][0]["folds"]), 5)
        self.assertEqual(first["parent_development"]["groups"], 5)
        self.assertTrue((output / "selected-model/model.safetensors").is_file())
        self.assertTrue((output / "selected-model/head_weights.pt").is_file())
        fold_results = sorted((output / "folds/tiny").glob("fold-*.json"))
        self.assertEqual(len(fold_results), 5)
        invalid_prediction = next(
            row
            for row in json.loads(fold_results[4].read_text())["predictions"]
            if not row["scoreable"]
        )
        self.assertEqual(invalid_prediction["prediction"], 0.0)

        scorer, _ = load_reward_model(output / "selected-model")
        self.assertTrue(
            torch.isfinite(torch.tensor(scorer.score("reference answer", "right answer")))
        )
        second = run_adaptation(self.config(output))
        self.assertEqual(first, second)

    def test_advantage_loss_ignores_groupwise_offsets_without_raw_anchor(self):
        predictions = torch.tensor([[0.1, 0.7], [0.3, 0.4]])
        targets = torch.tensor([[0.0, 1.0], [0.5, 0.75]])
        shifted_predictions = predictions + torch.tensor([[5.0], [-3.0]])
        first, _ = grouped_loss(
            predictions, targets, advantage_beta=0.1, raw_score_weight=0.0
        )
        shifted, _ = grouped_loss(
            shifted_predictions, targets, advantage_beta=0.1, raw_score_weight=0.0
        )
        torch.testing.assert_close(first, shifted)

    def test_loader_rejects_a_group_split_across_folds(self):
        rows = [json.loads(line) for line in self.data.read_text().splitlines()]
        rows[1]["fold"] = 1
        bad = self.root / "bad.jsonl"
        write_jsonl(bad, rows)
        with self.assertRaisesRegex(ValueError, "spans multiple folds"):
            load_grouped_records(bad, folds=5, group_size=2)


if __name__ == "__main__":
    unittest.main()
