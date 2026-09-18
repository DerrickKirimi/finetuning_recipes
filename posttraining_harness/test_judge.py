"""Offline tests of the judge client's key, budget, cache, retry and parsing controls. No network."""
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from posttraining_harness import judge

KEY = "TEST-KEY-abcdefghijklmnopqrstuvwxyz0123456789"


def pair(pair_id="instruction:1", a="good answer", b="bad answer"):
    return {"pair_id": pair_id, "task": "Passage: x.\n\nWhat is x?", "reference": "x is y.",
            "answers": {"sft": a, "cpt": b}, "model_a": "sft", "model_b": "cpt"}


def ok_payload(verdict="A", rationale="A matches the reference.", finish="STOP", prompt=900, out=40, thoughts=0):
    return {"candidates": [{"content": {"parts": [{"text": json.dumps({"rationale": rationale, "verdict": verdict})}]},
                            "finishReason": finish}],
            "usageMetadata": {"promptTokenCount": prompt, "candidatesTokenCount": out, "thoughtsTokenCount": thoughts},
            "modelVersion": "gemini-2.5-flash", "responseId": "r1"}


class FakeTransport:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def post(self, url, headers, body, timeout):
        self.calls.append({"url": url, "headers": dict(headers), "body": body})
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def config(**overrides):
    base = dict(
        model="gemini-3.5-flash-lite",
        thinking_level="minimal",
        cap_usd=1.0,
        backoff_seconds=0.0,
    )
    base.update(overrides)
    return judge.JudgeConfig(**base)


class KeyTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.root = Path(self.dir.name)

    def tearDown(self):
        self.dir.cleanup()

    def secrets(self, mode=0o600, content=f"GEMINI_API_KEY={KEY}\n", where=None):
        folder = where or self.root / "config"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / "gemini.env"
        path.write_text(content)
        os.chmod(path, mode)
        return path

    def test_reads_key_from_private_file(self):
        with mock.patch.object(judge, "_inside_git_work_tree", return_value=None):
            self.assertEqual(judge.load_api_key(environ={}, secrets_file=self.secrets()), KEY)

    def test_environment_variable_wins(self):
        self.assertEqual(judge.load_api_key(environ={"GEMINI_API_KEY": "from-env"}, secrets_file=self.secrets()), "from-env")

    def test_refuses_group_or_world_readable_file(self):
        path = self.secrets(mode=0o644)
        with self.assertRaises(judge.SecretError) as ctx:
            judge.load_api_key(environ={}, secrets_file=path)
        self.assertNotIn(KEY, str(ctx.exception))

    def test_refuses_file_inside_a_git_work_tree(self):
        repo = self.root / "repo"
        (repo / ".git").mkdir(parents=True)
        path = self.secrets(where=repo / "secrets")
        with self.assertRaises(judge.SecretError) as ctx:
            judge.load_api_key(environ={}, secrets_file=path)
        self.assertIn("git work tree", str(ctx.exception))
        self.assertNotIn(KEY, str(ctx.exception))

    def test_missing_file_and_missing_line_are_refused(self):
        with self.assertRaises(judge.SecretError):
            judge.load_api_key(environ={}, secrets_file=self.root / "absent.env")
        with self.assertRaises(judge.SecretError):
            judge.load_api_key(environ={}, secrets_file=self.secrets(content="OTHER=1\n"))

    def test_quoted_value_is_unquoted(self):
        path = self.secrets(content=f'GEMINI_API_KEY="{KEY}"\n')
        with mock.patch.object(judge, "_inside_git_work_tree", return_value=None):
            self.assertEqual(judge.load_api_key(environ={}, secrets_file=path), KEY)


class RequestTests(unittest.TestCase):
    def test_key_only_in_header_and_settings_frozen(self):
        with tempfile.TemporaryDirectory() as d:
            transport = FakeTransport([(200, ok_payload()), (200, ok_payload(verdict="B"))])
            judge.judge_pairs([pair()], config(), Path(d) / "log.jsonl", KEY, transport, echo=lambda *_: None)
            call = transport.calls[0]
            self.assertEqual(call["headers"]["x-goog-api-key"], KEY)
            self.assertNotIn(KEY, call["url"])
            self.assertNotIn(KEY, json.dumps(call["body"]))
            generation = call["body"]["generationConfig"]
            self.assertEqual(generation["thinkingConfig"], {"thinkingLevel": "minimal"})
            self.assertEqual(generation["temperature"], 0.0)
            self.assertEqual(generation["maxOutputTokens"], 512)
            self.assertEqual(generation["candidateCount"], 1)
            self.assertEqual(generation["responseSchema"]["properties"]["verdict"]["enum"], ["A", "B", "tie"])
            self.assertTrue(call["url"].endswith("/models/gemini-3.5-flash-lite:generateContent"))

    def test_both_orders_swap_the_models_shown(self):
        _, first = judge.build_prompt(pair(), "AB")
        user, second = judge.build_prompt(pair(), "BA")
        self.assertEqual(first, {"A": "sft", "B": "cpt"})
        self.assertEqual(second, {"A": "cpt", "B": "sft"})
        self.assertLess(user.index("bad answer"), user.index("good answer"))

    def test_empty_answer_is_shown_as_empty(self):
        user, _ = judge.build_prompt(pair(a="   "), "AB")
        self.assertEqual(json.loads(user.split("\n", 1)[1])["answer_a"], "(empty)")


class ThinkingConfigTests(unittest.TestCase):
    def test_level_takes_precedence_and_none_omits(self):
        self.assertEqual(config().generation_config()["thinkingConfig"], {"thinkingLevel": "minimal"})
        self.assertNotIn(
            "thinkingConfig",
            config(thinking_level=None, thinking_budget=None).generation_config(),
        )
        self.assertEqual(
            config(thinking_level=None, thinking_budget=0).generation_config()["thinkingConfig"],
            {"thinkingBudget": 0},
        )

    def test_reservation_is_bounded_by_max_output_tokens(self):
        cfg = config(max_output_tokens=1000, price_input_per_million=0.0, price_output_per_million=10.0)
        self.assertAlmostEqual(cfg.reserve_usd(""), 1000 * 10.0 / 1e6)

    def test_invalid_budget_and_attempt_settings_are_rejected(self):
        for overrides in (
            {"model": " "},
            {"max_output_tokens": 0},
            {"price_input_per_million": -1},
            {"cap_usd": 0},
            {"max_attempts": 0},
            {"timeout_seconds": 0},
            {"backoff_seconds": -1},
            {"thinking_budget": 0},
        ):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                config(**overrides)


class RunTests(unittest.TestCase):
    def test_log_never_contains_the_key_even_in_error_bodies(self):
        with tempfile.TemporaryDirectory() as d:
            log = Path(d) / "log.jsonl"
            transport = FakeTransport([(503, {"error": {"message": f"bad key {KEY}"}}),
                                       (200, ok_payload()), (200, ok_payload())])
            judge.judge_pairs([pair()], config(), log, KEY, transport, sleep=lambda _: None, echo=lambda *_: None)
            self.assertNotIn(KEY, log.read_text())
            self.assertIn("[redacted]", log.read_text())

    def test_resume_skips_successful_calls(self):
        with tempfile.TemporaryDirectory() as d:
            log = Path(d) / "log.jsonl"
            judge.judge_pairs([pair()], config(), log, KEY, FakeTransport([(200, ok_payload()), (200, ok_payload("B"))]),
                              echo=lambda *_: None)
            second = FakeTransport([])
            result = judge.judge_pairs([pair()], config(), log, KEY, second, echo=lambda *_: None)
            self.assertEqual(second.calls, [])
            self.assertEqual(result["skipped_cached"], 2)

    def test_changed_prompt_or_settings_is_not_served_from_cache(self):
        with tempfile.TemporaryDirectory() as d:
            log = Path(d) / "log.jsonl"
            judge.judge_pairs([pair()], config(), log, KEY, FakeTransport([(200, ok_payload()), (200, ok_payload())]),
                              echo=lambda *_: None)
            again = FakeTransport([(200, ok_payload()), (200, ok_payload())])
            judge.judge_pairs([pair()], config(seed=7), log, KEY, again, echo=lambda *_: None)
            self.assertEqual(len(again.calls), 2)

    def test_retries_rate_limits_and_counts_each_attempt(self):
        with tempfile.TemporaryDirectory() as d:
            log = Path(d) / "log.jsonl"
            transport = FakeTransport([(429, {"error": {}}), (200, ok_payload()), (200, ok_payload("B"))])
            judge.judge_pairs([pair()], config(), log, KEY, transport, sleep=lambda _: None, echo=lambda *_: None)
            records = judge.read_log(log)
            self.assertEqual([r["outcome"] for r in records], ["http_error", "ok", "ok"])
            self.assertGreater(records[0]["charged_usd"], 0)

    def test_client_error_stops_the_run_without_retry_or_charge(self):
        with tempfile.TemporaryDirectory() as d:
            log = Path(d) / "log.jsonl"
            transport = FakeTransport([(400, {"error": {"message": f"unknown field; key {KEY}"}}), (200, ok_payload())])
            with self.assertRaises(judge.RequestRejected) as ctx:
                judge.judge_pairs([pair("p1"), pair("p2")], config(), log, KEY, transport, echo=lambda *_: None)
            self.assertEqual(len(transport.calls), 1)
            self.assertNotIn(KEY, str(ctx.exception))
            records = judge.read_log(log)
            self.assertEqual([r["outcome"] for r in records], ["http_error"])
            self.assertEqual(records[0]["charged_usd"], 0.0)

    def test_transport_failure_charges_its_reservation(self):
        with tempfile.TemporaryDirectory() as d:
            log = Path(d) / "log.jsonl"
            transport = FakeTransport([TimeoutError("slow"), (200, ok_payload()), (200, ok_payload())])
            judge.judge_pairs([pair()], config(), log, KEY, transport, sleep=lambda _: None, echo=lambda *_: None)
            first = judge.read_log(log)[0]
            self.assertEqual(first["outcome"], "transport_error")
            self.assertEqual(first["charged_usd"], first["reserved_usd"])

    def test_budget_stops_before_the_cap_is_crossed(self):
        with tempfile.TemporaryDirectory() as d:
            log = Path(d) / "log.jsonl"
            cfg = config(cap_usd=0.0015)
            transport = FakeTransport([(200, ok_payload(prompt=900, out=400))] * 10)
            with self.assertRaises(judge.BudgetExhausted):
                judge.judge_pairs([pair("p1"), pair("p2")], cfg, log, KEY, transport, echo=lambda *_: None)
            self.assertLessEqual(judge.spent_usd(judge.read_log(log)), cfg.cap_usd)

    def test_reported_usage_above_reservation_stops_and_is_not_cached(self):
        with tempfile.TemporaryDirectory() as d:
            log = Path(d) / "log.jsonl"
            payload = ok_payload(prompt=1_000_000, out=1)
            with self.assertRaises(judge.ReservationUnderflow):
                judge.judge_pairs(
                    [pair()],
                    config(),
                    log,
                    KEY,
                    FakeTransport([(200, payload)]),
                    echo=lambda *_: None,
                )
            record = judge.read_log(log)[0]
            self.assertEqual(record["outcome"], "reservation_underflow")
            self.assertGreater(record["charged_usd"], record["reserved_usd"])

    def test_concurrent_owner_cannot_share_a_log_budget(self):
        with tempfile.TemporaryDirectory() as d:
            log = Path(d) / "log.jsonl"
            with judge.exclusive_log(log), self.assertRaises(judge.LogLocked):
                judge.judge_pairs(
                    [pair()],
                    config(),
                    log,
                    KEY,
                    FakeTransport([]),
                    echo=lambda *_: None,
                )


class ParseTests(unittest.TestCase):
    def test_accepts_a_clean_verdict(self):
        self.assertEqual(judge.parse_response(ok_payload("tie"))["verdict"], "tie")

    def test_truncated_output_is_a_failure(self):
        self.assertEqual(judge.parse_response(ok_payload(finish="MAX_TOKENS"))["outcome"], "parse_failure")

    def test_invalid_json_or_verdict_is_a_failure(self):
        bad_json = ok_payload()
        bad_json["candidates"][0]["content"]["parts"][0]["text"] = "A is better"
        self.assertEqual(judge.parse_response(bad_json)["outcome"], "parse_failure")

    def test_rationale_word_limit_is_a_parser_setting(self):
        long = json.dumps({"rationale": " ".join(["word"] * 120), "verdict": "A"})
        payload = {"candidates": [{"finishReason": "STOP", "content": {"parts": [{"text": long}]}}]}
        self.assertEqual(judge.parse_response(payload)["outcome"], "parse_failure")
        self.assertEqual(judge.parse_response(payload, max_rationale_words=200)["outcome"], "ok")
        # a parser tolerance must not change request identity, or cached verdicts would silently fork
        base = judge.JudgeConfig(model="m")
        wide = judge.JudgeConfig(model="m", max_rationale_words=200)
        self.assertEqual(judge.cache_key(base, "p", "AB", "f"), judge.cache_key(wide, "p", "AB", "f"))
        self.assertEqual(judge.parse_response(ok_payload(verdict="C"))["outcome"], "parse_failure")
        self.assertEqual(judge.parse_response(ok_payload(rationale=" "))["outcome"], "parse_failure")
        too_long = " ".join(["word"] * 81)
        self.assertEqual(judge.parse_response(ok_payload(rationale=too_long))["outcome"], "parse_failure")

    def test_blocked_prompt_is_recorded(self):
        self.assertEqual(judge.parse_response({"promptFeedback": {"blockReason": "SAFETY"}})["outcome"], "blocked")

    def test_measured_cost_includes_thinking_tokens(self):
        cfg = config()
        self.assertAlmostEqual(cfg.measured_usd({"promptTokenCount": 1_000_000, "candidatesTokenCount": 0,
                                                 "thoughtsTokenCount": 1_000_000}), 0.30 + 2.50)


class SummaryTests(unittest.TestCase):
    def test_order_averaged_credit_and_consistency(self):
        with tempfile.TemporaryDirectory() as d:
            log = Path(d) / "log.jsonl"
            pairs = [pair("p1"), pair("p2")]
            # p1: sft preferred in both orders; p2: position-biased, always answer A (sft first, then cpt first)
            transport = FakeTransport([(200, ok_payload("A")), (200, ok_payload("B")),
                                       (200, ok_payload("A")), (200, ok_payload("A"))])
            judge.judge_pairs(pairs, config(), log, KEY, transport, echo=lambda *_: None)
            summary = judge.summarize(pairs, log)
            self.assertEqual(summary["per_pair"][0]["preferences"], ["sft", "sft"])
            self.assertEqual(summary["per_pair"][1]["preferences"], ["sft", "cpt"])
            self.assertAlmostEqual(summary["order_averaged_credit_model_a"], 0.75)
            self.assertAlmostEqual(summary["position_consistency"], 0.5)

    def test_refuses_to_mix_successful_judge_configurations(self):
        with tempfile.TemporaryDirectory() as d:
            log = Path(d) / "log.jsonl"
            judge.judge_pairs(
                [pair()],
                config(seed=1),
                log,
                KEY,
                FakeTransport([(200, ok_payload()), (200, ok_payload())]),
                echo=lambda *_: None,
            )
            judge.judge_pairs(
                [pair()],
                config(seed=2),
                log,
                KEY,
                FakeTransport([(200, ok_payload()), (200, ok_payload())]),
                echo=lambda *_: None,
            )
            with self.assertRaisesRegex(ValueError, "multiple judge configurations"):
                judge.summarize([pair()], log)

    def test_does_not_reuse_a_verdict_after_pair_content_changes(self):
        with tempfile.TemporaryDirectory() as d:
            log = Path(d) / "log.jsonl"
            original = pair()
            judge.judge_pairs(
                [original],
                config(),
                log,
                KEY,
                FakeTransport([(200, ok_payload()), (200, ok_payload())]),
                echo=lambda *_: None,
            )
            changed = pair(a="different answer")
            summary = judge.summarize([changed], log)
            self.assertEqual(summary["pairs_judged_in_both_orders"], 0)
            self.assertEqual(summary["ignored_successes_for_other_prompt_content"], 2)


class PairValidationTests(unittest.TestCase):
    def write(self, root: str, rows: list[dict]) -> Path:
        path = Path(root) / "pairs.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        return path

    def test_rejects_duplicate_ids_and_missing_model_answers(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(ValueError, "unique"):
                judge.read_pairs(self.write(d, [pair("same"), pair("same")]))
            broken = pair()
            del broken["answers"]["cpt"]
            with self.assertRaisesRegex(ValueError, "both models"):
                judge.read_pairs(self.write(d, [broken]))


class DryRunTests(unittest.TestCase):
    def test_dry_run_needs_no_key_and_makes_no_calls(self):
        with tempfile.TemporaryDirectory() as d:
            pairs_path = Path(d) / "pairs.jsonl"
            pairs_path.write_text(json.dumps(pair()) + "\n")
            out = Path(d) / "estimate.json"
            env = dict(os.environ)
            os.environ.pop("GEMINI_API_KEY", None)
            try:
                code = judge.main(["dry-run", "--model", "gemini-3.5-flash-lite", "--thinking-level", "minimal",
                                   "--pairs", str(pairs_path), "--log", str(out),
                                   "--cap-usd", "0.1", "--price-input", "0.30", "--price-output", "2.50",
                                   "--secrets-file", str(Path(d) / "absent.env")])
            finally:
                os.environ.clear()
                os.environ.update(env)
            self.assertEqual(code, 0)
            estimate = json.loads(out.read_text())
            self.assertEqual(estimate["calls"], 2)
            self.assertTrue(estimate["fits_cap_if_every_call_succeeds_first_attempt"])
            self.assertGreater(
                estimate["all_attempts_reserved_usd_total"],
                estimate["single_attempt_reserved_usd_total"],
            )


if __name__ == "__main__":
    unittest.main()
