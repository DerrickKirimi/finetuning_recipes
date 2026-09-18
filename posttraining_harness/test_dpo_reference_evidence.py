"""Offline tests of the fixed-reference evidence: the frozen-state digest, the dtype-aware zero-tolerance probe
comparison, and the reference lifecycle diagnostic driven through a real TRL DPOTrainer on CPU.

The comparison fixtures include the DPO pilot's launch-r3 reference values. They are used as a mutation fixture for the
IEEE-754 float16 rounding identity, not to choose a tolerance: the criterion has no numeric tolerance.
"""
import copy
import importlib.util
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

REQUIRED = ("torch", "transformers", "trl", "peft", "tokenizers", "datasets")
MISSING = [name for name in REQUIRED if importlib.util.find_spec(name) is None]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "preference_optimization")]

R3_FLOAT16 = {"chosen": [-209.625, -315.75, -209.625, -209.625], "rejected": [-266.25, -266.25, -275.75, -315.75]}
R3_FLOAT32 = {"chosen": [-209.58914184570312, -315.865478515625, -209.58914184570312, -209.58914184570312],
              "rejected": [-266.148681640625, -266.148681640625, -275.69110107421875, -315.865478515625]}


def frozen(sha="a" * 64):
    components = {k: sha for k in ("parameters", "buffers", "quant_state", "module_attributes", "adapter_topology")}
    return {"sha256": sha, "components": components, "counts": {k: 1 for k in components}}


def probe(values, dtype, tokens=((131, 90, 131, 131), (60, 60, 88, 131)), state=None):
    return {"reference_chosen_logps": list(values["chosen"]), "reference_rejected_logps": list(values["rejected"]),
            "output_dtypes": {"reference_chosen": dtype, "reference_rejected": dtype},
            "completion_tokens": {"chosen": list(tokens[0]), "rejected": list(tokens[1])},
            "frozen_state": state or frozen()}


@unittest.skipIf(MISSING, f"needs {MISSING}; run under the fork's main .venv")
class CompareReferenceProbesTests(unittest.TestCase):
    def compare(self, a, b):
        import dpo_setup as setup
        return setup.compare_reference_probes(a, b)

    def test_float16_nearest_is_ieee_binary16(self):
        import dpo_setup as setup
        import numpy as np

        for value in R3_FLOAT32["chosen"] + R3_FLOAT32["rejected"] + [0.1, -1e-4, 65504.0, 1.0000001]:
            self.assertEqual(setup.float16_nearest(value), float(np.float16(value)))

    def test_same_dtype_requires_bitwise_equality(self):
        self.assertTrue(self.compare(probe(R3_FLOAT32, "torch.float32"), probe(R3_FLOAT32, "torch.float32"))["ok"])
        nudged = copy.deepcopy(R3_FLOAT32)
        nudged["chosen"][1] = math.nextafter(nudged["chosen"][1], 0.0)
        result = self.compare(probe(R3_FLOAT32, "torch.float32"), probe(nudged, "torch.float32"))
        self.assertFalse(result["ok"])
        self.assertEqual(result["modes"]["chosen"], "exact")

    def test_launch_r3_values_satisfy_the_float16_rounding_identity_in_either_order(self):
        for a, b in ((probe(R3_FLOAT16, "torch.float16"), probe(R3_FLOAT32, "torch.float32")),
                     (probe(R3_FLOAT32, "torch.float32"), probe(R3_FLOAT16, "torch.float16"))):
            result = self.compare(a, b)
            self.assertTrue(result["ok"], result)
            self.assertEqual(set(result["modes"].values()), {"float16_rounding_identity"})
            self.assertAlmostEqual(result["max_abs_difference"], 0.115478515625)

    def test_the_identity_rejects_anything_but_the_one_rounding(self):
        one_step = copy.deepcopy(R3_FLOAT16)
        one_step["chosen"][1] = -315.5            # a float16 value, but not the nearest one to the float32 value
        off_grid = copy.deepcopy(R3_FLOAT16)
        off_grid["rejected"][0] = -266.2          # not representable in float16 at all
        self.assertFalse(self.compare(probe(one_step, "torch.float16"), probe(R3_FLOAT32, "torch.float32"))["ok"])
        self.assertFalse(self.compare(probe(off_grid, "torch.float16"), probe(R3_FLOAT32, "torch.float32"))["ok"])

    def test_documented_limit_a_float32_change_inside_one_float16_step_is_invisible_across_dtypes(self):
        # Across a float16/float32 pair the identity cannot see a change smaller than the float16 rounding. That is why
        # the diagnostic's decisive comparisons are same-dtype and bitwise; the identity only explains cross-dtype pairs.
        moved = copy.deepcopy(R3_FLOAT32)
        moved["chosen"][1] = -315.87
        self.assertTrue(self.compare(probe(R3_FLOAT16, "torch.float16"), probe(moved, "torch.float32"))["ok"])
        self.assertFalse(self.compare(probe(R3_FLOAT32, "torch.float32"), probe(moved, "torch.float32"))["ok"])

    def test_state_tokens_dtypes_and_values_must_all_be_present_and_consistent(self):
        base = probe(R3_FLOAT32, "torch.float32")
        mutations = {
            "frozen_component": lambda p: p["frozen_state"]["components"].update(quant_state="b" * 64),
            "frozen_missing": lambda p: p.pop("frozen_state"),
            "tokens": lambda p: p["completion_tokens"]["chosen"].__setitem__(0, 130),
            "tokens_missing": lambda p: p.pop("completion_tokens"),
            "dtype_missing": lambda p: p["output_dtypes"].pop("reference_rejected"),
            "bfloat16": lambda p: p["output_dtypes"].update(reference_chosen="torch.bfloat16"),
            "nan": lambda p: p["reference_rejected_logps"].__setitem__(2, float("nan")),
            "empty": lambda p: p.update(reference_chosen_logps=[]),
            "short": lambda p: p["reference_chosen_logps"].pop(),
            "int_value": lambda p: p["reference_chosen_logps"].__setitem__(0, -209),
        }
        for name, mutate in mutations.items():
            after = copy.deepcopy(base)
            mutate(after)
            with self.subTest(mutation=name):
                self.assertFalse(self.compare(base, after)["ok"])
                self.assertFalse(self.compare(after, base)["ok"])


@unittest.skipIf(MISSING, f"needs {MISSING}; run under the fork's main .venv")
class FrozenStateDigestTests(unittest.TestCase):
    def setUp(self):
        import torch
        from test_dpo_segments_real_trainer import make_model

        self.torch = torch
        self.model = make_model()
        self.model.register_buffer("rotary_cache", torch.arange(8, dtype=torch.float32), persistent=False)
        layer = self.model.base_model.model.model.layers[0].self_attn.q_proj
        self.base_weight = layer.base_layer.weight
        self.base_weight.quant_state = SimpleNamespace(
            absmax=torch.ones(4), shape=torch.Size([16, 16]), code=torch.linspace(-1, 1, 16), dtype=torch.float16,
            blocksize=64, quant_type="nf4", offset=torch.tensor(0.5), nested=True, packing_format_for_cpu=False,
            state2=SimpleNamespace(absmax=torch.ones(2), shape=None, code=torch.linspace(-1, 1, 256), dtype=torch.float32,
                                   blocksize=256, quant_type=None, offset=None, nested=False,
                                   packing_format_for_cpu=False, state2=None))
        layer.base_layer.compute_dtype = torch.float16
        self.layer = layer

    def digest(self):
        import dpo_setup as setup
        return setup.frozen_state_digest(self.model)

    def assert_changes(self, component, mutate):
        before = self.digest()
        mutate()
        after = self.digest()
        changed = sorted(k for k in before["components"] if before["components"][k] != after["components"][k])
        self.assertEqual(changed, [component])
        self.assertNotEqual(before["sha256"], after["sha256"])

    def test_counts_and_stability(self):
        first, second = self.digest(), self.digest()
        self.assertEqual(first, second)
        self.assertGreater(first["counts"]["parameters"], 0)
        self.assertEqual(first["counts"]["quant_state"], 1)
        self.assertGreaterEqual(first["counts"]["buffers"], 1)
        self.assertGreater(first["counts"]["adapter_topology"], 0)

    def test_each_component_detects_its_own_mutation(self):
        torch = self.torch
        cases = {
            "parameters": lambda: self.base_weight.data.view(-1)[37].add_(1e-3),
            "buffers": lambda: self.model.rotary_cache.__setitem__(5, 5.5),
            "quant_state_absmax": lambda: self.base_weight.quant_state.absmax.__setitem__(2, 1.0001),
            "quant_state_blocksize": lambda: setattr(self.base_weight.quant_state, "blocksize", 128),
            "quant_state_nested": lambda: self.base_weight.quant_state.state2.code.__setitem__(200, 0.0),
            "quant_state_nested_flag": lambda: setattr(self.base_weight.quant_state, "nested", False),
            "quant_state_cpu_packing": lambda: setattr(self.base_weight.quant_state, "packing_format_for_cpu", True),
            "quant_state_dtype": lambda: setattr(self.base_weight.quant_state, "dtype", torch.float32),
            "module_attributes": lambda: setattr(self.layer.base_layer, "compute_dtype", torch.float32),
            "adapter_topology_scaling": lambda: self.layer.scaling.update(default=2.0),
            "adapter_topology_merged": lambda: self.layer.merged_adapters.append("default"),
        }
        component = {"parameters": "parameters", "buffers": "buffers", "module_attributes": "module_attributes"}
        for name, mutate in cases.items():
            with self.subTest(mutation=name):
                self.setUp()
                expected = component.get(name, "quant_state" if name.startswith("quant_state") else "adapter_topology")
                self.assert_changes(expected, mutate)

    def test_adapter_weights_are_not_frozen_state(self):
        before = self.digest()
        with self.torch.no_grad():
            for name, param in self.model.named_parameters():
                if "lora_" in name:
                    param.add_(0.5)
        self.assertEqual(before, self.digest())

    @unittest.skipIf(importlib.util.find_spec("bitsandbytes") is None, "needs bitsandbytes")
    def test_digesting_a_real_params4bit_never_moves_its_shared_quant_state(self):
        # Params4bit.to()/.cpu() relocate the shared QuantState in place; hashing must never go through them.
        import bitsandbytes as bnb
        import dpo_setup as setup

        moves = []

        class RecordingState(SimpleNamespace):
            def to(self, device):
                moves.append(str(device))

        state = RecordingState(absmax=self.torch.ones(4), shape=self.torch.Size([8, 8]), code=self.torch.linspace(-1, 1, 16),
                               dtype=self.torch.float16, blocksize=64, quant_type="nf4", offset=None, nested=False,
                               packing_format_for_cpu=False, state2=None)
        weight = bnb.nn.Params4bit(self.torch.zeros(8, 8, dtype=self.torch.uint8), requires_grad=False, quant_state=state,
                                   quant_type="nf4", bnb_quantized=True)
        self.layer.base_layer.weight = weight
        digest = setup.frozen_state_digest(self.model)
        self.assertEqual(moves, [])
        self.assertEqual(digest["counts"]["quant_state"], 1)
        self.assertIs(self.layer.base_layer.weight.quant_state, state)

    def test_unknown_quant_state_value_types_fail_loudly(self):
        import dpo_setup as setup

        self.base_weight.quant_state.blocksize = object()
        with self.assertRaises(TypeError):
            setup.frozen_state_digest(self.model)


@unittest.skipIf(MISSING, f"needs {MISSING}; run under the fork's main .venv")
class ReferenceLifecycleTests(unittest.TestCase):
    """The orchestration production runs, on a real TRL trainer (CPU: no fp16, so every comparison is exact)."""

    def run_variant(self, variant):
        import dpo_setup as setup
        from test_dpo_segments_real_trainer import make_rows, make_trainer
        from training_controls import make_stop_callback

        calls = []
        with tempfile.TemporaryDirectory() as d:
            record = Path(d) / "diagnostic.json"
            argv = ["--batch_size", "2", "--grad_accum", "2", "--max_steps", "4", "--dataloader_num_workers", "0",
                    "--reference_probe_rows", "2", "--reference_lifecycle_diagnostic", variant,
                    "--diagnostic_record", str(record)]
            if variant == "train_update":
                argv += ["--stop_after_steps", "2"]
            args, trainer = make_trainer(d, make_rows(12), make_rows(4, offset=20), argv)
            if variant == "train_update":
                trainer.add_callback(make_stop_callback(2, {}))
            result = setup.reference_lifecycle(
                trainer, args.reference_probe_rows, variant,
                for_inference=lambda m: (calls.append("inference"), m.eval()),
                for_training=lambda m: (calls.append("training"), m.train()))
            json.dumps(result)          # the record must be JSON-serializable as written
        return result, calls

    def assert_common(self, result, labels):
        self.assertEqual([s["label"] for s in result["stages"]], labels)
        self.assertTrue(result["comparisons"])
        for name, comparison in result["comparisons"].items():
            self.assertTrue(comparison["ok"], (name, comparison))
            self.assertEqual(set(comparison["modes"].values()), {"exact"})
        states = {s["frozen_state"]["sha256"] for s in result["stages"]}
        self.assertEqual(len(states), 1)
        for stage in result["stages"]:
            self.assertEqual(stage["output_dtypes"]["reference_chosen"], "torch.float32")
            self.assertEqual(len(stage["completion_tokens"]["chosen"]), 2)
            self.assertEqual(stage["context"]["label"], stage["label"])
            self.assertEqual(len(stage["context"]["forward_logits_dtypes"]), 2)

    def test_train_update_variant(self):
        result, calls = self.run_variant("train_update")
        self.assert_common(result, ["fresh", "fresh_repeat", "after_train_update", "after_train_update_repeat",
                                    "after_for_inference", "after_for_training"])
        self.assertEqual(result["global_step_after_train"], 2)
        self.assertEqual(calls, ["inference", "training"])
        by = {s["label"]: s for s in result["stages"]}
        # Two updates: the first runs at learning rate 0 under warmup, the second moves the adapter.
        self.assertNotEqual(by["fresh"]["adapter_sha256"], by["after_train_update"]["adapter_sha256"])
        self.assertNotEqual(by["fresh"]["policy_chosen_logps"], by["after_train_update"]["policy_chosen_logps"])
        self.assertEqual(set(result["comparisons"]), {"fresh__fresh_repeat", "after_train_update__after_train_update_repeat",
                                                      "fresh__after_train_update", "after_train_update__after_for_inference",
                                                      "after_train_update__after_for_training"})

    def test_wrapper_only_variant_applies_no_update(self):
        result, calls = self.run_variant("wrapper_only")
        self.assert_common(result, ["fresh", "after_for_inference", "after_for_training", "after_accelerate_prepare",
                                    "after_accelerate_prepare_repeat"])
        self.assertEqual(calls, ["inference", "training"])
        self.assertEqual({s["adapter_sha256"] for s in result["stages"]}, {result["stages"][0]["adapter_sha256"]})
        # CPU Accelerate has no mixed precision, so preparation adds no fp32 output wrapper; the T4 run records it.
        self.assertFalse(any(s["context"]["forward_wrapped_by_accelerate"] for s in result["stages"]))

    def test_unknown_variant_is_refused(self):
        import dpo_setup as setup

        with self.assertRaises(ValueError):
            setup.reference_lifecycle(None, 2, "wrapper")


@unittest.skipIf(MISSING, f"needs {MISSING}; run under the fork's main .venv")
class DiagnosticArgumentTests(unittest.TestCase):
    def parse(self, extra):
        import dpo_setup as setup

        args = setup.build_parser().parse_args(["--max_steps", "10"] + extra)
        setup.check_args(args)
        return args

    def test_diagnostic_argument_rules(self):
        ok = ["--reference_probe_rows", "4", "--diagnostic_record", "x.json"]
        self.parse(ok + ["--reference_lifecycle_diagnostic", "wrapper_only"])
        self.parse(ok + ["--reference_lifecycle_diagnostic", "train_update", "--stop_after_steps", "1"])
        self.parse(ok + ["--reference_lifecycle_diagnostic", "train_update", "--stop_after_steps", "2"])
        refused = [
            ok + ["--reference_lifecycle_diagnostic", "train_update"],
            ok + ["--reference_lifecycle_diagnostic", "train_update", "--stop_after_steps", "6"],
            ["--reference_lifecycle_diagnostic", "wrapper_only", "--diagnostic_record", "x.json"],
            ["--reference_lifecycle_diagnostic", "wrapper_only", "--reference_probe_rows", "4"],
            ["--diagnostic_record", "x.json"],
            ok + ["--reference_lifecycle_diagnostic", "wrapper_only", "--method", "orpo"],
            ["--train_rows_limit", "0"],
        ]
        for extra in refused:
            with self.subTest(extra=extra), self.assertRaises(SystemExit):
                self.parse(extra)

    def test_row_limits_select_the_first_rows_after_the_split(self):
        import hashlib

        import dpo_setup as setup

        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "rows.jsonl"
            rows = [{"prompt": [{"role": "user", "content": f"q{i}"}], "chosen": [{"role": "assistant", "content": "a"}],
                     "rejected": [{"role": "assistant", "content": "b"}]} for i in range(50)]
            path.write_text("".join(json.dumps(r) + "\n" for r in rows))
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            base = ["--dataset_file", str(path), "--dataset_sha256", digest]
            full_train, full_val, _ = setup.load_preference_datasets(self.parse(base))
            train, val, record = setup.load_preference_datasets(self.parse(base + ["--train_rows_limit", "7",
                                                                                   "--validation_rows_limit", "1"]))
        self.assertEqual(record["rows"], {"train": 7, "validation": 1})
        self.assertEqual(record["rows_before_limits"], {"train": len(full_train), "validation": len(full_val)})
        self.assertEqual(train["prompt"], full_train.select(range(7))["prompt"])


if __name__ == "__main__":
    unittest.main()
