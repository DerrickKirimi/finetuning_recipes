import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from posttraining_harness import scalar_judge


def groups():
    return [
        {
            "group_id": "validation-000",
            "task": "Answer from the passage.",
            "reference": "correct",
            "candidates": [
                {"candidate_id": f"validation-000:{index}", "rollout_index": index, "answer": f"answer {index}"}
                for index in range(8)
            ],
        }
    ]


def payload(expected_ids, *, usage=True):
    text = json.dumps(
        {
            "ratings": [
                {"candidate_id": value, "score": index % 5, "rationale": "Specific concise reason."}
                for index, value in enumerate(expected_ids)
            ]
        }
    )
    return {
        "candidates": [{"finishReason": "STOP", "content": {"parts": [{"text": text}]}}],
        "usageMetadata": {"promptTokenCount": 100, "candidatesTokenCount": 100} if usage else {},
        "modelVersion": "test-model",
    }


class Transport:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def post(self, url, headers, body, timeout):
        self.calls.append((url, headers, body, timeout))
        return self.responses.pop(0)


class ScalarJudgeTests(unittest.TestCase):
    def config(self, cap=1.0):
        return scalar_judge.ScalarConfig(
            model="gemini-test",
            cap_usd=cap,
            price_input_per_million=0.30,
            price_output_per_million=2.50,
        )

    def test_deterministic_forward_and_reverse_orders(self):
        group = groups()[0]
        forward = scalar_judge.candidate_order(group, "forward", 7)
        reverse = scalar_judge.candidate_order(group, "reverse", 7)
        self.assertEqual([row["candidate_id"] for row in forward], [row["candidate_id"] for row in reversed(reverse)])

    def test_parser_requires_all_exact_ids_and_integer_scores(self):
        expected = [row["candidate_id"] for row in groups()[0]["candidates"]]
        self.assertEqual(scalar_judge.parse_response(payload(expected), expected)["outcome"], "ok")
        bad = payload(expected[:-1] + ["wrong"])
        self.assertEqual(scalar_judge.parse_response(bad, expected)["outcome"], "parse_failure")
        boolean = payload(expected)
        content = json.loads(boolean["candidates"][0]["content"]["parts"][0]["text"])
        content["ratings"][0]["score"] = True
        boolean["candidates"][0]["content"]["parts"][0]["text"] = json.dumps(content)
        self.assertEqual(scalar_judge.parse_response(boolean, expected)["outcome"], "parse_failure")

    def test_run_resumes_success_and_key_is_only_in_header(self):
        group = groups()[0]
        _, expected = scalar_judge.build_prompt(group, "forward", self.config().seed)
        transport = Transport([(200, payload(expected))])
        with TemporaryDirectory() as scratch:
            log = Path(scratch) / "log.jsonl"
            first = scalar_judge.score_groups(groups(), self.config(), log, "secret", transport=transport, sleep=lambda _: None, echo=lambda _: None)
            second = scalar_judge.score_groups(groups(), self.config(), log, "secret", transport=transport, sleep=lambda _: None, echo=lambda _: None)
            text = log.read_text()
        self.assertEqual(first["calls"], 1)
        self.assertEqual(second["skipped_cached"], 1)
        self.assertNotIn("secret", text)
        self.assertEqual(transport.calls[0][1]["x-goog-api-key"], "secret")
        self.assertNotIn("secret", transport.calls[0][0])

    def test_budget_stops_before_request(self):
        transport = Transport([])
        with TemporaryDirectory() as scratch:
            with self.assertRaises(scalar_judge.judge.BudgetExhausted):
                scalar_judge.score_groups(groups(), self.config(cap=0.000001), Path(scratch) / "log.jsonl", "key", transport=transport, echo=lambda _: None)
        self.assertEqual(transport.calls, [])

    def test_summary_averages_two_orders_by_candidate_identity(self):
        config = self.config()
        group = groups()[0]
        responses = []
        for order in ("forward", "reverse"):
            _, expected = scalar_judge.build_prompt(group, order, config.seed)
            responses.append((200, payload(expected)))
        with TemporaryDirectory() as scratch:
            log = Path(scratch) / "log.jsonl"
            scalar_judge.score_groups(groups(), config, log, "key", transport=Transport(responses), sleep=lambda _: None, orders=("forward", "reverse"), echo=lambda _: None)
            report = scalar_judge.summarize(groups(), log)
        self.assertEqual(report["groups_with_scores"], 1)
        self.assertEqual(len(report["labels"]), 8)
        self.assertTrue(all(row["orders"] == 2 for row in report["labels"]))

    def test_validation_rejects_missing_candidate(self):
        value = groups()
        value[0]["candidates"].pop()
        with self.assertRaises(ValueError):
            scalar_judge.validate_groups(value)

    def test_dry_run_needs_no_key_and_creates_nested_report(self):
        with TemporaryDirectory() as scratch:
            root = Path(scratch)
            groups_path = root / "groups.jsonl"
            groups_path.write_text(json.dumps(groups()[0]) + "\n")
            report = root / "nested/dry-run.json"
            status = scalar_judge.main(
                [
                    "dry-run",
                    "--groups",
                    str(groups_path),
                    "--log",
                    str(report),
                    "--model",
                    "gemini-test",
                    "--cap-usd",
                    "1",
                    "--price-input",
                    "0.30",
                    "--price-output",
                    "2.50",
                ]
            )
            result = json.loads(report.read_text())
        self.assertEqual(status, 0)
        self.assertEqual(result["calls"], 1)
        self.assertTrue(result["fits_cap_if_every_call_succeeds_first_attempt"])


if __name__ == "__main__":
    unittest.main()
