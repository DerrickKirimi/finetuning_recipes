"""CPU checks of the evaluation battery. No GPU, no model, no network.

Several of these encode findings from the 2026-09-12 review: the original Track A scored
`input` against a task-specific `output` and called it continuation, when only 46 of the
500 pinned rows are continuation tasks; Track B rendered through whatever template the
tokenizer carried; and outputs recorded a label rather than provenance.
"""
import json
import ast
import tempfile
import unittest
import unittest.mock
from pathlib import Path

from posttraining_harness.eval_battery import (
    CHATML_TEMPLATE, build_manifest, continuation_split, extract_system_prompt,
    identity_prompt, load_rows, render_continuation,
    render_instruction, run_track, score_track, sha256_text, validate_manifest,
    verify_bertscore_snapshot, write_outputs,
)


def row(i, words=80):
    return {"instruction": f"Answer the question given the passage {i}",
            "input": " ".join(f"w{i}x{j}" for j in range(words)),
            "output": f"answer {i}"}


class Fixture(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.source = self.tmp / "test.jsonl"
        self.rows = {3: row(3, 80), 7: row(7, 80), 11: row(11, 5)}   # 11 is too short
        with self.source.open("w") as handle:
            for i in range(20):
                handle.write(json.dumps(self.rows.get(i, row(i, 80))) + "\n")
        indices = [3, 7, 11]
        self.split = self.tmp / "split.json"
        self.split.write_text(json.dumps({
            "dataset": "d", "dataset_revision": "rev", "test_source": {"sha256": "abc"},
            "fixed_test_battery": {
                "indices": indices, "selection": "seeded",
                "overlap_with_full_training_source": {"prompt_rows": 0, "passage_rows": 0},
                "prompt_sha256": [sha256_text(identity_prompt(self.rows[i]["instruction"],
                                                              self.rows[i]["input"]))
                                  for i in indices]}}))
        self.sft = self.tmp / "sft.py"
        self.sft.write_text('SYSTEM_PROMPT = """be helpful"""\n')
        self.manifest = build_manifest(self.split, self.source, self.sft, count=3)
        self.loaded = load_rows(self.manifest, self.source)


class ContinuationTrackTests(Fixture):
    def test_split_is_a_real_prefix_and_suffix_of_the_passage(self):
        prefix, rest = continuation_split(" ".join(str(i) for i in range(100)))
        self.assertEqual(prefix.split()[0], "0")
        self.assertEqual(rest.split()[-1], "99")
        self.assertEqual(len(prefix.split()) + len(rest.split()), 100,
                         "the split must lose nothing")

    def test_short_passages_are_excluded_not_degraded(self):
        self.assertIsNone(continuation_split("too short"),
                          "a two-word reference would score as noise")

    def test_output_is_not_used_by_the_continuation_track(self):
        prompt, reference = render_continuation(self.loaded[0])
        self.assertNotIn("answer 3", reference,
                         "the task-specific output must not be the continuation reference")
        self.assertIn("w3x", reference)

    def test_manifest_records_which_rows_each_track_covers(self):
        tracks = self.manifest["tracks"]
        self.assertEqual(tracks["continuation"]["indices"], [3, 7],
                         "row 11 is too short and must be excluded, visibly")
        self.assertEqual(tracks["instruction"]["indices"], [3, 7, 11])
        self.assertNotEqual(tracks["continuation"]["row_count"],
                            tracks["instruction"]["row_count"])


class InstructionTrackTests(Fixture):
    def test_rendering_does_not_depend_on_a_tokenizer_template(self):
        prompt, reference = render_instruction(self.loaded[0], "be helpful")
        self.assertIn("<|im_start|>system\nbe helpful<|im_end|>", prompt)
        self.assertIn(self.loaded[0]["merged_prompt"], prompt)
        self.assertTrue(prompt.endswith("<|im_start|>assistant\n"))
        self.assertEqual(reference, "answer 3")

    def test_template_hash_is_pinned_in_the_manifest(self):
        self.assertEqual(self.manifest["tracks"]["instruction"]["template_sha256"],
                         sha256_text(CHATML_TEMPLATE))

    def test_system_prompt_comes_from_sft_not_a_copy(self):
        self.assertEqual(extract_system_prompt(self.sft), "be helpful")


class ManifestGovernsTests(Fixture):
    def test_validate_passes_on_the_pinned_manifest(self):
        self.assertTrue(validate_manifest(self.manifest))

    def test_changed_generation_setting_is_refused(self):
        self.manifest["generation"]["max_new_tokens"] = 128
        with self.assertRaises(AssertionError) as caught:
            validate_manifest(self.manifest)
        self.assertIn("max_new_tokens", str(caught.exception))

    def test_changed_template_is_refused(self):
        self.manifest["tracks"]["instruction"]["template_sha256"] = "0" * 64
        with self.assertRaises(AssertionError):
            validate_manifest(self.manifest)

    def test_changed_continuation_rule_is_refused(self):
        self.manifest["tracks"]["continuation"]["min_words"] = 5
        with self.assertRaises(AssertionError):
            validate_manifest(self.manifest)

    def test_changed_bertscore_revision_is_refused(self):
        self.manifest["bertscore_model_revision"] = "0" * 40
        with self.assertRaises(AssertionError) as caught:
            validate_manifest(self.manifest)
        self.assertIn("bertscore_model_revision", str(caught.exception))

    def test_drifted_source_is_rejected(self):
        lines = self.source.read_text().splitlines()
        changed = json.loads(lines[7]); changed["input"] = "TAMPERED " * 80
        lines[7] = json.dumps(changed)
        self.source.write_text("\n".join(lines) + "\n")
        with self.assertRaises(AssertionError):
            load_rows(self.manifest, self.source)


class BertScoreSnapshotTests(Fixture):
    def _snapshot(self):
        revision = self.manifest["bertscore_model_revision"]
        path = self.tmp / revision
        path.mkdir()
        hashes = {}
        for name in ("config.json", "model.safetensors", "tokenizer.json",
                     "tokenizer_config.json", "vocab.txt"):
            payload = f"fixture:{name}".encode()
            (path / name).write_bytes(payload)
            hashes[name] = __import__("hashlib").sha256(payload).hexdigest()
        self.manifest["bertscore_model_files_sha256"] = hashes
        return path

    def test_exact_snapshot_is_accepted_and_identified(self):
        path = self._snapshot()
        out = verify_bertscore_snapshot(path, self.manifest)
        self.assertEqual(out["revision"], self.manifest["bertscore_model_revision"])
        self.assertEqual(out["num_layers"], 5)
        self.assertEqual(out["files_sha256"],
                         self.manifest["bertscore_model_files_sha256"])

    def test_wrong_revision_is_refused(self):
        path = self._snapshot()
        wrong = path.with_name("0" * 40)
        path.rename(wrong)
        with self.assertRaises(AssertionError) as caught:
            verify_bertscore_snapshot(wrong, self.manifest)
        self.assertIn("not pinned revision", str(caught.exception))

    def test_changed_weight_is_refused(self):
        path = self._snapshot()
        (path / "model.safetensors").write_bytes(b"changed")
        with self.assertRaises(AssertionError) as caught:
            verify_bertscore_snapshot(path, self.manifest)
        self.assertIn("model.safetensors", str(caught.exception))


class RunAndWriteTests(Fixture):
    def _result(self, track, predictions=None):
        seen = {}
        def generate(model, tokenizer, prompts, **kwargs):
            seen.update(kwargs)
            return predictions if predictions is not None else ["gen"] * len(prompts)
        return run_track(None, None, self.loaded, track, "be helpful",
                         generate, lambda *a, **k: {"perplexity": 12.5, "scored_rows": 2,
                                    "supervised_tokens": 40, "skipped": []}), seen

    def test_continuation_track_skips_short_rows(self):
        result, _ = self._result("continuation")
        self.assertEqual(result["indices"], [3, 7], "row 11 is too short")

    def test_instruction_track_uses_every_row(self):
        result, _ = self._result("instruction")
        self.assertEqual(result["indices"], [3, 7, 11])

    def test_generation_parameters_reach_the_generator(self):
        _, seen = self._result("instruction")
        self.assertEqual(seen["max_new_tokens"], 256)
        self.assertEqual(seen["repetition_penalty"], 1.2)

    def test_mismatched_generation_count_is_rejected(self):
        with self.assertRaises(AssertionError):
            self._result("instruction", predictions=["only one"])

    def test_unknown_track_is_refused(self):
        with self.assertRaises(AssertionError):
            self._result("track-a")

    def test_outputs_require_provenance_and_refuse_overwrite(self):
        result, _ = self._result("instruction")
        metrics = {"instruction": score_track(result, lambda p, r: {"rouge1": 0.5})}
        provenance = {"checkpoint_label": "sft-1900", "model_path": "/m",
                      "weight_sha256": {"model.safetensors": "a" * 64},
                      "parent": "cpt-merged", "adapter_sha256": "b" * 64}
        out = self.tmp / "out"
        write_outputs(out, self.manifest, provenance, {"instruction": result}, metrics)
        written = json.loads((out / "metrics.json").read_text())
        self.assertEqual(written["provenance"]["adapter_sha256"], "b" * 64)
        self.assertEqual(written["rows_per_track"]["instruction"], 3)
        with self.assertRaises(AssertionError):
            write_outputs(out, self.manifest, provenance, {"instruction": result}, metrics)

    def test_outputs_reject_a_bare_label(self):
        result, _ = self._result("instruction")
        with self.assertRaises(AssertionError):
            write_outputs(self.tmp / "o2", self.manifest, {"checkpoint_label": "sft"},
                          {"instruction": result}, {})

    def test_each_track_reports_the_wall_time_that_produced_it(self):
        """A bounded smoke cannot forecast a full track; a timed one calibrates the next."""
        result, _ = self._result("instruction")
        self.assertIn("timing", result)
        for key in ("generation_seconds", "perplexity_seconds", "rows", "generated_words"):
            self.assertIn(key, result["timing"])
        self.assertEqual(result["timing"]["rows"], 3)
        metrics = score_track(result, lambda p, r: {"rouge1": 0.5})
        self.assertIn("generation_seconds", metrics,
                      "the timing has to survive into the published metrics, not stay in "
                      "the in-memory result")
        self.assertNotIn("rows", metrics, "row counts already have their own keys")

    def test_a_nonfinite_metric_is_named_not_absorbed(self):
        """A NaN is not a low score. It has to arrive as a name, not as a number."""
        result, _ = self._result("instruction")
        metrics = score_track(result, lambda p, r: {"rouge1": float("nan"),
                                                    "bleu": float("inf"),
                                                    "bertscore_f1": 0.5})
        self.assertEqual(metrics["nonfinite_metrics"], ["bleu", "rouge1"])
        self.assertEqual(score_track(result, lambda p, r: {"rouge1": 0.5})
                         ["nonfinite_metrics"], [],
                         "a clean run must report the empty list, not omit the key")

    def test_a_nonfinite_metric_is_refused_publication(self):
        """The generations already cost the run; the consolidation is what gets refused."""
        result, _ = self._result("instruction")
        metrics = {"instruction": score_track(
            result, lambda p, r: {"rouge1": float("nan")})}
        provenance = {"checkpoint_label": "sft-1900", "model_path": "/m",
                      "weight_sha256": {"model.safetensors": "a" * 64},
                      "parent": "cpt-merged", "adapter_sha256": "b" * 64}
        out = self.tmp / "nonfinite"
        with self.assertRaises(AssertionError) as caught:
            write_outputs(out, self.manifest, provenance, {"instruction": result}, metrics)
        self.assertIn("nonfinite", str(caught.exception))
        self.assertIn("rouge1", str(caught.exception))
        self.assertIn("instruction", str(caught.exception))
        self.assertFalse((out / "metrics.json").exists(),
                         "a refused result must not leave a published metrics file")
        self.assertFalse((out / "SHA256SUMS").exists(),
                         "and must not acquire a checksum that vouches for it")

    def test_empty_generations_are_counted(self):
        result, _ = self._result("instruction", predictions=["ok", "", "  "])
        metrics = score_track(result, lambda p, r: {"rouge1": 0.1})
        self.assertEqual(metrics["empty_generations"], 2)
        self.assertEqual(metrics["perplexity_scored_rows"], 2)
        self.assertEqual(metrics["perplexity_skipped_rows"], 0)


class FakeTensor(list):
    """Enough of a tensor for the perplexity path, on CPU, without torch."""
    def clone(self): return FakeTensor([list(r) for r in self])
    def __setitem__(self, key, value):
        if isinstance(key, tuple):
            row, span = key
            self[row][span] = [value] * len(range(*span.indices(len(self[row]))))
        else:
            super().__setitem__(key, value)
    def __ne__(self, other): return FakeTensor([[x != other for x in r] for r in self])
    def sum(self): return sum(sum(1 for x in r if x) for r in self)


class PerplexityGuardTests(unittest.TestCase):
    """The injected harness could score the wrong tokens, or none, and still return a number."""

    def setUp(self):
        import sys, types
        self.words = lambda s: s.split()
        fake = types.SimpleNamespace(
            tensor=lambda data, device=None: FakeTensor(data),
            no_grad=lambda: __import__("contextlib").nullcontext(),
        )
        self.patch = unittest.mock.patch.dict(sys.modules, {"torch": fake})

    def _tokenizer(self):
        class T:
            def __call__(self, text, add_special_tokens=False):
                return {"input_ids": [hash(w) % 1000 for w in text.split()]}
        return T()

    def _model(self, loss=1.0):
        class M:
            device = "cpu"
            def __call__(self, input_ids=None, labels=None):
                return types_ns(loss=loss)
        import types as _t
        def types_ns(**kw): return _t.SimpleNamespace(**kw)
        return M()

    def test_long_prompt_keeps_the_reference_whole(self):
        """Right-truncating the joined text would discard the reference and still score."""
        from posttraining_harness.eval_battery import teacher_forced_perplexity
        prompt = " ".join(f"p{i}" for i in range(3000))
        reference = "r1 r2 r3"
        with self.patch:
            out = teacher_forced_perplexity(self._model(), self._tokenizer(),
                                            [prompt], [reference], max_length=2048)
        self.assertEqual(out["supervised_tokens"], 3,
                         "every reference token must be supervised, none truncated away")
        self.assertEqual(out["scored_rows"], 1)
        self.assertEqual(out["skipped"], [])

    def test_a_row_with_no_reference_is_skipped_not_silently_absorbed(self):
        from posttraining_harness.eval_battery import teacher_forced_perplexity
        with self.patch:
            out = teacher_forced_perplexity(self._model(), self._tokenizer(),
                                            ["a b c", "d e f"], ["", "ref"], max_length=2048)
        self.assertEqual(out["scored_rows"], 1)
        self.assertEqual(len(out["skipped"]), 1)
        self.assertIn("tokenizes to nothing", out["skipped"][0]["reason"])

    def test_no_scoreable_row_raises_rather_than_dividing_by_zero(self):
        from posttraining_harness.eval_battery import teacher_forced_perplexity
        with self.patch:
            with self.assertRaises(AssertionError) as caught:
                teacher_forced_perplexity(self._model(), self._tokenizer(),
                                          ["a"], [""], max_length=2048)
        self.assertIn("no row produced a supervised token", str(caught.exception))


class ParentVerificationTests(unittest.TestCase):
    """A check that cannot fail is not a check. These make it fail."""

    def test_mismatched_parent_is_rejected(self):
        from posttraining_harness.eval_battery import verify_parent
        with self.assertRaises(AssertionError) as caught:
            verify_parent({"adapter_declared_base": "/p",
                           "weight_sha256": {"model.safetensors": "a" * 64}},
                          expected_parent_sha256="b" * 64)
        self.assertIn("matches neither any weight file", str(caught.exception))

    def test_matching_parent_is_accepted_and_says_so(self):
        from posttraining_harness.eval_battery import verify_parent
        out = verify_parent({"adapter_declared_base": "/p",
                             "weight_sha256": {"model.safetensors": "a" * 64}},
                            expected_parent_sha256="a" * 64)
        self.assertTrue(out["checked"])
        self.assertTrue(out["matched"])

    def test_no_expected_digest_records_that_nothing_was_checked(self):
        from posttraining_harness.eval_battery import verify_parent
        out = verify_parent({"adapter_declared_base": "/p",
                             "weight_sha256": {"model.safetensors": "a" * 64}})
        self.assertFalse(out["checked"], "silence must not read as success")
        self.assertIn("no expected parent digest", out["reason"])


class ReferenceTruncationTests(unittest.TestCase):
    """The whole-reference contract must hold or the row must be refused, never clipped."""

    def _run(self, references, max_length):
        import sys, types
        from posttraining_harness.eval_battery import teacher_forced_perplexity
        fake = types.SimpleNamespace(
            tensor=lambda data, device=None: FakeTensor(data),
            no_grad=lambda: __import__("contextlib").nullcontext())

        class T:
            def __call__(self, text, add_special_tokens=False):
                return {"input_ids": [1] * len(text.split())}

        class M:
            device = "cpu"
            def __call__(self, input_ids=None, labels=None):
                import types as _t
                return _t.SimpleNamespace(loss=1.0)

        prompts = ["p " * 5] * len(references)
        with unittest.mock.patch.dict(sys.modules, {"torch": fake}):
            return teacher_forced_perplexity(M(), T(), prompts, references,
                                             max_length=max_length,
                                             indices=list(range(len(references))))

    def test_oversized_reference_is_refused_with_its_row_id(self):
        out = self._run(["w " * 10, "short ref"], max_length=8)
        self.assertEqual(out["scored_rows"], 1, "only the row that fits may be scored")
        self.assertEqual(len(out["skipped"]), 1)
        skip = out["skipped"][0]
        self.assertEqual(skip["index"], 0, "the skipped row must be identifiable")
        self.assertIn("does not fit", skip["reason"])
        self.assertEqual(skip["reference_tokens"], 10)

    def test_every_scored_reference_is_whole(self):
        out = self._run(["a b c"], max_length=64)
        self.assertEqual(out["supervised_tokens"], 3,
                         "a scored row supervises exactly its whole reference")


class CudaRoutingTests(unittest.TestCase):
    """Execute the CUDA branch of load_model with a double. Passing tests that bypass this
    route were exactly the gap the review identified."""

    class Tok:
        """load_model sets padding_side and pad_token on whatever it returns."""
        def __init__(self, name): self.name, self.padding_side, self.pad_token = name, "right", None
        eos_token = "</s>"
        def __eq__(self, other): return getattr(other, "name", other) == self.name

    def _load_model_source(self):
        import ast
        source = Path(__file__).resolve().parents[1] / "cpt" / "inference.py"
        tree = ast.parse(source.read_text())
        fn = [n for n in tree.body
              if isinstance(n, ast.FunctionDef) and n.name == "load_model"][0]
        return ast.unparse(fn)

    def test_cuda_branch_loads_the_explicit_base_then_applies_the_adapter(self):
        import os, sys, tempfile, types
        adapter = Path(tempfile.mkdtemp())
        (adapter / "adapter_config.json").write_text("{}")
        seen = {}

        class FLM:
            @staticmethod
            def from_pretrained(model_name=None, **kwargs):
                seen["base_loaded"] = model_name
                return "BASE_MODEL", CudaRoutingTests.Tok("BASE_TOK")
            @staticmethod
            def for_inference(model):
                seen["for_inference"] = model

        class Peft:
            @staticmethod
            def from_pretrained(model, path):
                seen["adapter_applied_to"] = model
                seen["adapter_path"] = path
                return "BASE+ADAPTER"

        torch_double = types.SimpleNamespace(
            cuda=types.SimpleNamespace(is_available=lambda: True),
            backends=types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: False)),
            float16="fp16", float32="fp32")
        namespace = {"torch": torch_double, "os": os, "PeftModel": Peft,
                     "AutoTokenizer": types.SimpleNamespace(
                         from_pretrained=lambda p: CudaRoutingTests.Tok("ADAPTER_TOK")),
                     "AutoModelForCausalLM": types.SimpleNamespace(from_pretrained=None),
                     "print": lambda *a, **k: None}
        with unittest.mock.patch.dict(sys.modules, {
                "unsloth": types.SimpleNamespace(FastLanguageModel=FLM)}):
            exec(compile(self._load_model_source(), "inference", "exec"), namespace)
            model, tokenizer = namespace["load_model"](
                str(adapter), base_model_id="/the/explicit/merged/parent")

        self.assertEqual(seen["base_loaded"], "/the/explicit/merged/parent",
                         "CUDA must load the SUPPLIED base, not the adapter path")
        self.assertEqual(seen["adapter_path"], str(adapter))
        self.assertEqual(model, "BASE+ADAPTER")

    def test_cuda_branch_unchanged_for_a_full_model(self):
        import os, sys, tempfile, types
        full = Path(tempfile.mkdtemp())          # no adapter_config.json
        seen = {}

        class FLM:
            @staticmethod
            def from_pretrained(model_name=None, **kwargs):
                seen["loaded"] = model_name
                return "MODEL", CudaRoutingTests.Tok("TOK")
            @staticmethod
            def for_inference(model): pass

        torch_double = types.SimpleNamespace(
            cuda=types.SimpleNamespace(is_available=lambda: True),
            backends=types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: False)),
            float16="fp16", float32="fp32")
        namespace = {"torch": torch_double, "os": os, "PeftModel": None,
                     "AutoTokenizer": None, "AutoModelForCausalLM": None,
                     "print": lambda *a, **k: None}
        with unittest.mock.patch.dict(sys.modules, {
                "unsloth": types.SimpleNamespace(FastLanguageModel=FLM)}):
            exec(compile(self._load_model_source(), "inference", "exec"), namespace)
            namespace["load_model"](str(full), base_model_id="/ignored/for/full/models")
        self.assertEqual(seen["loaded"], str(full),
                         "a full model still loads directly, base_model_id is irrelevant")


class RuntimeIdentityTests(unittest.TestCase):
    """Weight hashes do not identify the evaluated model; dtype and quantization do."""

    def _model(self, dtype="torch.float32", quant=None):
        import types
        param = types.SimpleNamespace(dtype=dtype, device="cpu")
        config = types.SimpleNamespace(quantization_config=quant, name_or_path="/p",
                                       torch_dtype=dtype)
        return types.SimpleNamespace(parameters=lambda: [param], config=config)

    def _tokenizer(self, merges="a b"):
        class T:
            padding_side = "left"
            def get_vocab(self): return {"a": 0, "b": 1}
            def save_pretrained(self, directory):
                Path(directory).mkdir(parents=True, exist_ok=True)
                (Path(directory) / "vocab.json").write_text('{"a": 0, "b": 1}')
                (Path(directory) / "merges.txt").write_text(merges)
        return T()

    def test_float32_and_4bit_are_distinguishable(self):
        from posttraining_harness.eval_battery import describe_runtime
        import types
        cpu = describe_runtime(self._model("torch.float32"), self._tokenizer())
        gpu = describe_runtime(
            self._model("torch.float16", types.SimpleNamespace(load_in_4bit=True,
                                                               bnb_4bit_quant_type="nf4")),
            self._tokenizer())
        self.assertEqual(cpu["dtype"], "torch.float32")
        self.assertIsNone(cpu["quantization"])
        self.assertEqual(gpu["quantization"]["load_in_4bit"], True)
        self.assertNotEqual(cpu["dtype"], gpu["dtype"],
                            "the same weights run two ways must not look identical")

    def test_tokenizer_is_hashed(self):
        from posttraining_harness.eval_battery import describe_runtime
        out = describe_runtime(self._model(), self._tokenizer())
        self.assertEqual(out["tokenizer_vocab_size"], 2)
        self.assertTrue(out["tokenizer_vocab_sha256"])
        self.assertEqual(out["padding_side"], "left",
                         "left padding is required for batch generation; record it")

    def test_same_vocabulary_with_different_merges_is_not_the_same_tokenizer(self):
        """The exact gap the vocabulary digest cannot see."""
        from posttraining_harness.eval_battery import describe_runtime
        one = describe_runtime(self._model(), self._tokenizer(merges="a b"))
        two = describe_runtime(self._model(), self._tokenizer(merges="b a"))
        self.assertEqual(one["tokenizer_vocab_sha256"], two["tokenizer_vocab_sha256"],
                         "the vocabulary really is identical; that is the point")
        self.assertNotEqual(one["tokenizer_serialisation_sha256"],
                            two["tokenizer_serialisation_sha256"],
                            "different merges segment text differently and must not "
                            "compare equal")
        self.assertIn("merges.txt", one["tokenizer_files_sha256"])

    def test_a_tokenizer_that_cannot_serialise_records_the_error(self):
        from posttraining_harness.eval_battery import describe_runtime
        class T:
            padding_side = "left"
            def get_vocab(self): return {"a": 0}
        out = describe_runtime(self._model(), T())
        self.assertIn("tokenizer_serialisation_error", out,
                      "a missing identity must not read as a matching one")
        self.assertNotIn("tokenizer_serialisation_sha256", out)

    def test_the_loader_is_named_not_inferred(self):
        """Device and 4-bit metadata imply Unsloth; the archived output should say so."""
        from posttraining_harness.eval_battery import describe_runtime
        out = describe_runtime(self._model(), self._tokenizer())
        self.assertIn("model_class", out)
        self.assertIn("model_module", out)
        self.assertIs(out["unsloth_imported"], False,
                      "this suite never imports unsloth; a true here would be a false "
                      "record of the CUDA route")
        self.assertIn("unsloth_version", out,
                      "absent is indistinguishable from not-recorded; None plus an error "
                      "string is not")

    def test_a_broken_model_records_the_error_rather_than_omitting_the_field(self):
        from posttraining_harness.eval_battery import describe_runtime
        import types
        broken = types.SimpleNamespace(parameters=lambda: (_ for _ in ()).throw(RuntimeError("x")),
                                       config=None)
        out = describe_runtime(broken, self._tokenizer())
        self.assertIn("dtype_error", out, "a missing field must not read as absence of quantization")


class OutputReservationTests(unittest.TestCase):
    """Refusing after the battery has run wastes the battery."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def test_completed_directory_is_refused_before_work(self):
        from posttraining_harness.eval_battery import reserve_output_dir
        (self.tmp / "metrics.json").write_text("{}")
        with self.assertRaises(AssertionError) as caught:
            reserve_output_dir(self.tmp)
        self.assertIn("before spending the run", str(caught.exception))

    def test_fresh_directory_is_claimed(self):
        from posttraining_harness.eval_battery import reserve_output_dir
        out = reserve_output_dir(self.tmp / "fresh")
        self.assertTrue(out.is_dir())

    def test_overwrite_is_allowed_only_deliberately(self):
        from posttraining_harness.eval_battery import reserve_output_dir
        (self.tmp / "metrics.json").write_text("{}")
        self.assertTrue(reserve_output_dir(self.tmp, overwrite=True).is_dir())


class DigestContractTests(unittest.TestCase):
    """The expected digest must be what a human would compute, not a bespoke aggregate."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.weights = self.tmp / "model.safetensors"
        self.weights.write_bytes(b"some weights")
        import hashlib
        self.raw = hashlib.sha256(b"some weights").hexdigest()

    def test_a_plain_sha256sum_of_the_weight_file_is_accepted(self):
        """The project records raw file digests; the check must accept them."""
        from posttraining_harness.eval_battery import describe_checkpoint, verify_parent
        provenance = describe_checkpoint(self.tmp)
        self.assertEqual(provenance["weight_sha256"]["model.safetensors"], self.raw)
        out = verify_parent(provenance, expected_parent_sha256=self.raw)
        self.assertTrue(out["matched"])
        self.assertEqual(out["matched_via"], "weight_file")

    def test_the_aggregate_is_named_not_passed_off_as_a_file_digest(self):
        from posttraining_harness.eval_battery import describe_checkpoint
        provenance = describe_checkpoint(self.tmp)
        self.assertNotEqual(provenance["model_sha256_aggregate"], self.raw,
                            "the aggregate folds in filenames, so it differs by design")
        self.assertIn("filename bytes", provenance["model_sha256_aggregate_algorithm"])

    def test_the_aggregate_still_verifies_when_supplied(self):
        from posttraining_harness.eval_battery import describe_checkpoint, verify_parent
        provenance = describe_checkpoint(self.tmp)
        out = verify_parent(provenance,
                            expected_parent_sha256=provenance["model_sha256_aggregate"])
        self.assertEqual(out["matched_via"], "aggregate")

    def test_a_wrong_digest_is_still_rejected(self):
        from posttraining_harness.eval_battery import describe_checkpoint, verify_parent
        provenance = describe_checkpoint(self.tmp)
        with self.assertRaises(AssertionError):
            verify_parent(provenance, expected_parent_sha256="c" * 64)


class QuantizationShapeTests(unittest.TestCase):
    """describe_runtime runs after the model loads; a crash there wastes the load."""

    class Tok:
        padding_side = "left"
        def get_vocab(self): return {"a": 0}

    def _model(self, quantization):
        import types
        return types.SimpleNamespace(
            parameters=lambda: [types.SimpleNamespace(dtype="torch.float16", device="cuda:0")],
            config=types.SimpleNamespace(quantization_config=quantization,
                                         name_or_path="/p", torch_dtype="fp16"))

    def test_unsloth_writes_a_dict_and_it_must_not_crash(self):
        """Installed unsloth/models/loader.py assigns a dict for 4-bit; vars() raised."""
        from posttraining_harness.eval_battery import describe_runtime
        out = describe_runtime(
            self._model({"load_in_4bit": True, "bnb_4bit_quant_type": "nf4"}), self.Tok())
        self.assertEqual(out["quantization"]["load_in_4bit"], True)
        self.assertEqual(out["quantization"]["bnb_4bit_quant_type"], "nf4")

    def test_a_config_object_with_to_dict_is_used(self):
        from posttraining_harness.eval_battery import describe_runtime
        import types
        config = types.SimpleNamespace(to_dict=lambda: {"load_in_4bit": True, "x": object()})
        out = describe_runtime(self._model(config), self.Tok())
        self.assertEqual(out["quantization"]["load_in_4bit"], True)
        self.assertIsInstance(out["quantization"]["x"], str, "non-scalars are stringified")

    def test_a_plain_object_still_works(self):
        from posttraining_harness.eval_battery import describe_runtime
        import types
        out = describe_runtime(self._model(types.SimpleNamespace(load_in_4bit=True)), self.Tok())
        self.assertEqual(out["quantization"]["load_in_4bit"], True)

    def test_none_is_reported_as_none_not_omitted(self):
        from posttraining_harness.eval_battery import describe_runtime
        self.assertIsNone(describe_runtime(self._model(None), self.Tok())["quantization"])


class PartialProgressTests(unittest.TestCase):
    """Generation is the expensive half; losing it to an interruption is the waste."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def _result(self):
        return {"indices": [3, 7], "prompt_sha256": ["aa", "bb"],
                "predictions": ["p0", "p1"], "references": ["r0", "r1"]}

    def test_a_finished_track_is_on_disk_before_the_next_one_starts(self):
        from posttraining_harness.eval_battery import write_track_partial
        path = write_track_partial(self.tmp, "continuation", {"battery_version": "1"},
                                   {"checkpoint_label": "checkpoint-1900"},
                                   self._result(), {"bleu": 0.5})
        saved = json.loads(path.read_text())
        self.assertEqual([row["index"] for row in saved["rows"]], [3, 7])
        self.assertEqual(saved["rows"][0]["prediction"], "p0")
        self.assertEqual(saved["metrics"]["bleu"], 0.5)
        self.assertEqual(saved["checkpoint_label"], "checkpoint-1900")

    def test_a_partial_cannot_be_mistaken_for_a_result(self):
        from posttraining_harness.eval_battery import write_track_partial
        path = write_track_partial(self.tmp, "instruction", {"battery_version": "1"},
                                   {"checkpoint_label": "x"},
                                   self._result(), {})
        self.assertTrue(path.name.startswith("partial-"))
        self.assertIn("not a published result", json.loads(path.read_text())["warning"])
        self.assertFalse((self.tmp / "metrics.json").exists(),
                         "a partial must not claim the name reserve_output_dir guards")
        self.assertFalse((self.tmp / "SHA256SUMS").exists())

    def test_a_matching_partial_is_reusable(self):
        from posttraining_harness.eval_battery import load_track_partial, write_track_partial
        manifest = {"battery_version": "1", "content_sha256": "abc"}
        provenance = {"checkpoint_label": "x", "weight_sha256": {"m": "1"},
                      "adapter_sha256": "2", "limit": 10,
                      "runtime": {"dtype": "float32"}}
        write_track_partial(self.tmp, "continuation", manifest, provenance,
                            self._result(), {"bleu": 0.5})
        loaded = load_track_partial(self.tmp, "continuation", manifest, provenance)
        self.assertEqual(loaded["result"]["predictions"], ["p0", "p1"])
        self.assertEqual(loaded["metrics"]["bleu"], 0.5)

    def test_a_partial_from_another_limit_is_refused(self):
        from posttraining_harness.eval_battery import load_track_partial, write_track_partial
        manifest = {"battery_version": "1", "content_sha256": "abc"}
        first = {"checkpoint_label": "x", "limit": 10}
        write_track_partial(self.tmp, "continuation", manifest, first,
                            self._result(), {})
        with self.assertRaises(AssertionError) as caught:
            load_track_partial(self.tmp, "continuation", manifest,
                               {"checkpoint_label": "x", "limit": 500})
        self.assertIn("different manifest, checkpoint, runtime, or limit", str(caught.exception))

    def test_a_partial_with_changed_rows_is_refused(self):
        from posttraining_harness.eval_battery import (continuation_split, load_track_partial,
                                        sha256_text, write_track_partial)
        manifest = {"battery_version": "1", "content_sha256": "abc"}
        provenance = {"checkpoint_label": "x", "limit": 2}
        rendered_rows = [
            {"index": 3, "input": " ".join(f"a{i}" for i in range(80))},
            {"index": 7, "input": " ".join(f"b{i}" for i in range(80))},
        ]
        splits = [continuation_split(row["input"]) for row in rendered_rows]
        result = {"indices": [3, 7],
                  "prompt_sha256": [sha256_text(split[0]) for split in splits],
                  "predictions": ["p0", "p1"],
                  "references": [split[1] for split in splits]}
        write_track_partial(self.tmp, "continuation", manifest, provenance, result, {})
        saved = json.loads((self.tmp / "partial-continuation.json").read_text())
        saved["rows"][0]["reference"] = "damaged"
        (self.tmp / "partial-continuation.json").write_text(json.dumps(saved))
        with self.assertRaises(AssertionError) as caught:
            load_track_partial(self.tmp, "continuation", manifest, provenance,
                               rendered_rows, None)
        self.assertIn("do not match the currently rendered track", str(caught.exception))

    def test_evaluator_source_change_invalidates_a_partial(self):
        from posttraining_harness.eval_battery import load_track_partial, write_track_partial
        manifest = {"battery_version": "1", "content_sha256": "abc"}
        provenance = {"checkpoint_label": "x", "limit": 2,
                      "metric_runtime": {"rouge": "old"}}
        write_track_partial(self.tmp, "continuation", manifest, provenance,
                            self._result(), {})
        changed = dict(provenance, metric_runtime={"rouge": "new"})
        with self.assertRaises(AssertionError):
            load_track_partial(self.tmp, "continuation", manifest, changed)

    def test_completed_checksums_do_not_name_deleted_partials(self):
        from posttraining_harness.eval_battery import write_outputs, write_track_partial
        manifest = {"battery_version": "1", "content_sha256": "abc", "generation": {}}
        provenance = {"checkpoint_label": "x", "model_path": "/m",
                      "weight_sha256": {"model.safetensors": "a"}, "parent": "p",
                      "adapter_sha256": None}
        result = self._result()
        write_track_partial(self.tmp, "continuation", manifest, provenance, result, {})
        write_outputs(self.tmp, manifest, provenance, {"continuation": result},
                      {"continuation": {}})
        sums = (self.tmp / "SHA256SUMS").read_text()
        self.assertNotIn("partial-continuation.json", sums)


class TokenizerMarkerTests(unittest.TestCase):
    """base and SFT list the same 49,152 strings and disagree on two of their ids."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def _checkpoint(self, mapping):
        (self.tmp / "tokenizer.json").write_text(json.dumps(
            {"added_tokens": [{"id": i, "content": c} for c, i in mapping.items()]}))
        return self.tmp

    def _tokenizer(self, mapping):
        class T:
            def convert_tokens_to_ids(self, token): return mapping.get(token)
        return T()

    def test_matching_ids_pass_and_are_recorded(self):
        from posttraining_harness.eval_battery import verify_tokenizer_markers
        mapping = {"<|endoftext|>": 2, "<|im_start|>": 1, "<|im_end|>": 0}
        out = verify_tokenizer_markers(self._tokenizer(mapping), self._checkpoint(mapping))
        self.assertEqual(out["disagreements"], {})
        self.assertEqual(out["marker_ids"]["<|im_end|>"], 0,
                         "the id in force belongs in the record, not just its agreement")

    def test_the_real_base_against_sft_swap_is_refused(self):
        """The base tokenizer loaded beside the SFT adapter: ids 0 and 2 exchanged."""
        from posttraining_harness.eval_battery import verify_tokenizer_markers
        sft = {"<|endoftext|>": 2, "<|im_start|>": 1, "<|im_end|>": 0}
        base = {"<|endoftext|>": 0, "<|im_start|>": 1, "<|im_end|>": 2}
        self.assertEqual(set(sft), set(base),
                         "identical token sets: what a vocabulary check cannot catch")
        with self.assertRaises(AssertionError) as caught:
            verify_tokenizer_markers(self._tokenizer(base), self._checkpoint(sft))
        self.assertIn("<|im_end|>", str(caught.exception))
        self.assertIn("differently from training", str(caught.exception))

    def test_a_checkpoint_without_a_tokenizer_file_is_skipped_not_failed(self):
        from posttraining_harness.eval_battery import verify_tokenizer_markers
        out = verify_tokenizer_markers(self._tokenizer({}), self.tmp)
        self.assertIn("skipped", out)
        self.assertNotIn("disagreements", out)


class DeterministicRougeTests(unittest.TestCase):
    """Exercise the actual calculate_metrics body without importing remote metric modules."""

    def test_rouge_is_the_direct_mean_and_disables_bootstrap(self):
        source = Path(__file__).resolve().parents[1] / "cpt/evals.py"
        node = next(node for node in ast.parse(source.read_text()).body
                    if isinstance(node, ast.FunctionDef) and node.name == "calculate_metrics")
        calls = {}

        class Rouge:
            def compute(self, **kwargs):
                calls["use_aggregator"] = kwargs.get("use_aggregator")
                return {"rouge1": [0.0, 1.0], "rouge2": [0.2, 0.4],
                        "rougeL": [0.1, 0.7]}

        class Bleu:
            def compute(self, **kwargs): return {"bleu": 0.25}

        class Bert:
            def compute(self, **kwargs):
                calls["bertscore_model"] = kwargs.get("model_type")
                calls["bertscore_num_layers"] = kwargs.get("num_layers")
                return {"f1": [0.5, 0.9]}

        namespace = {"rouge": Rouge(), "bleu": Bleu(), "bertscore": Bert()}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
        first = namespace["calculate_metrics"](["a", "b"], ["c", "d"])
        second = namespace["calculate_metrics"](["a", "b"], ["c", "d"])
        self.assertIs(calls["use_aggregator"], False)
        self.assertEqual(first, second)
        self.assertEqual(first["rouge1"], 0.5)
        self.assertAlmostEqual(first["rouge2"], 0.3)
        self.assertAlmostEqual(first["rougeL"], 0.4)
        self.assertEqual(calls["bertscore_model"], "distilbert-base-uncased")
        self.assertEqual(calls["bertscore_num_layers"], 5)

    def test_metric_identity_ignores_local_download_metadata(self):
        source = Path(__file__).resolve().parents[1] / "cpt/evals.py"
        node = next(node for node in ast.parse(source.read_text()).body
                    if isinstance(node, ast.FunctionDef)
                    and node.name == "metric_runtime_identity")
        model = Path(tempfile.mkdtemp())
        (model / "model.safetensors").write_bytes(b"weights")
        metadata = model / ".cache" / "huggingface" / "download"
        metadata.mkdir(parents=True)
        (metadata / "model.metadata").write_text("run-specific timestamp")
        # __import__("importlib") does not bind the metadata submodule; the function
        # under test reaches importlib.metadata, so import the submodule by name. The
        # attribute exists only incidentally when some earlier import in the process
        # happened to load it, which made this test pass or error by suite ordering.
        namespace = {"Path": Path, "hashlib": __import__("hashlib"),
                     "importlib": __import__("importlib.metadata"),
                     "rouge": object(), "bleu": object(), "bertscore": object()}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
        out = namespace["metric_runtime_identity"](model, 5)
        self.assertEqual(set(out["bertscore_model_files_sha256"]), {"model.safetensors"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
