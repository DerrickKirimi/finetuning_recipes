"""Offline tests of the resumable preference-training controls, against the installed transformers where it matters."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from transformers.trainer_callback import DefaultFlowCallback, TrainerControl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "preference_optimization"))
import training_controls as tc  # noqa: E402


def installed_flow(global_step, save_steps=50, max_steps=494):
    args = SimpleNamespace(logging_strategy="steps", logging_steps=10, logging_first_step=False,
                           eval_strategy="steps", eval_delay=0, save_strategy="steps", save_steps=save_steps)
    state = SimpleNamespace(global_step=global_step, max_steps=max_steps, logging_steps=10, eval_steps=50,
                            save_steps=save_steps)
    return DefaultFlowCallback().on_step_end(args, state, TrainerControl())


class StopCallbackTests(unittest.TestCase):
    def test_installed_transformers_does_not_save_at_the_pilot_stop(self):
        self.assertFalse(installed_flow(51).should_save, "if this passes, the save rule changed; forcing may be redundant")
        self.assertTrue(installed_flow(50).should_save)

    def test_callback_stops_and_forces_the_save(self):
        record = {}
        callback = tc.make_stop_callback(51, record)
        control = TrainerControl()
        callback.on_step_end(None, SimpleNamespace(global_step=50), control)
        self.assertFalse(control.should_training_stop)
        callback.on_step_end(None, SimpleNamespace(global_step=51), control)
        self.assertTrue(control.should_training_stop and control.should_save)
        self.assertEqual(record, {"stop_after_steps_fired": True, "stopped_at_global_step": 51})


class SegmentArgTests(unittest.TestCase):
    def test_valid_and_invalid_segments(self):
        tc.validate_segment_args(max_steps=494, stop_after_steps=51, save_steps=50, eval_steps=50)
        tc.validate_segment_args(max_steps=494, stop_after_steps=None, save_steps=50, eval_steps=50)
        for kwargs in (dict(max_steps=-1, stop_after_steps=51), dict(max_steps=494, stop_after_steps=494),
                       dict(max_steps=494, stop_after_steps=0), dict(max_steps=51, stop_after_steps=51)):
            with self.subTest(kwargs=kwargs), self.assertRaises(tc.ContinuationError):
                tc.validate_segment_args(save_steps=50, eval_steps=50, **kwargs)


class ResumeCheckpointTests(unittest.TestCase):
    def make(self, root, step=51, max_steps=494, skip=(), empty=(), state=None):
        checkpoint = Path(root) / f"checkpoint-{step}"
        checkpoint.mkdir()
        for name in tc.REQUIRED_CHECKPOINT_FILES:
            if name in skip:
                continue
            content = "" if name in empty else "x"
            if name == "trainer_state.json" and name not in empty:
                content = json.dumps(state or {"global_step": step, "max_steps": max_steps,
                                               "best_model_checkpoint": None, "best_metric": None})
            (checkpoint / name).write_text(content)
        return checkpoint

    def test_complete_checkpoint_validates(self):
        with tempfile.TemporaryDirectory() as d:
            info = tc.validate_resume_checkpoint(self.make(d), expected_max_steps=494)
            self.assertEqual((info["global_step"], info["max_steps"]), (51, 494))

    def test_every_missing_or_empty_required_file_is_refused(self):
        for name in tc.REQUIRED_CHECKPOINT_FILES:
            for mode in ("skip", "empty"):
                with self.subTest(name=name, mode=mode), tempfile.TemporaryDirectory() as d:
                    checkpoint = self.make(d, **{mode: (name,)})
                    with self.assertRaises(tc.ContinuationError):
                        tc.validate_resume_checkpoint(checkpoint, expected_max_steps=494)

    def test_schedule_and_step_mismatches_are_refused(self):
        cases = [dict(max_steps=300), dict(state={"global_step": 50, "max_steps": 494})]
        for kwargs in cases:
            with self.subTest(kwargs=kwargs), tempfile.TemporaryDirectory() as d:
                with self.assertRaises(tc.ContinuationError):
                    tc.validate_resume_checkpoint(self.make(d, **kwargs), expected_max_steps=494)

    def test_missing_directory_scaler_and_bad_json_are_refused(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(tc.ContinuationError):
                tc.validate_resume_checkpoint(Path(d) / "checkpoint-51", expected_max_steps=494)
            checkpoint = self.make(d)
            with self.assertRaises(tc.ContinuationError):
                tc.validate_resume_checkpoint(checkpoint, expected_max_steps=494, require_scaler=True)
            (checkpoint / "trainer_state.json").write_text("{not json")
            with self.assertRaises(tc.ContinuationError):
                tc.validate_resume_checkpoint(checkpoint, expected_max_steps=494)


class PrecisionTests(unittest.TestCase):
    def fake_torch(self, cuda, native):
        calls = []

        def is_bf16_supported(including_emulation=True):
            calls.append(including_emulation)
            return native if not including_emulation else True
        return SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: cuda, is_bf16_supported=is_bf16_supported)), calls

    def test_t4_style_emulated_bf16_selects_fp16(self):
        torch, calls = self.fake_torch(cuda=True, native=False)
        self.assertEqual(tc.select_precision(torch), {"cuda": True, "native_bf16": False, "bf16": False, "fp16": True})
        self.assertEqual(calls, [False])

    def test_native_bf16_and_cpu(self):
        torch, _ = self.fake_torch(cuda=True, native=True)
        self.assertTrue(tc.select_precision(torch)["bf16"])
        torch, _ = self.fake_torch(cuda=False, native=False)
        self.assertEqual(tc.select_precision(torch), {"cuda": False, "native_bf16": False, "bf16": False, "fp16": False})


class TerminalRecordTests(unittest.TestCase):
    def test_reasons(self):
        state = SimpleNamespace(global_step=51, max_steps=494, best_model_checkpoint="x/checkpoint-50", best_metric=0.6)
        self.assertEqual(tc.terminal_record(state, {"stop_after_steps_fired": True}, 51)["stop_reason"], "target_step")
        state.global_step = 120
        self.assertEqual(tc.terminal_record(state, {"deadline_fired": True}, 51 * 4)["stop_reason"], "deadline")
        both = tc.terminal_record(state, {"reference_guard_fired": True, "stop_after_steps_fired": True}, 120)
        self.assertEqual(both["stop_reason"], "reference_guard")      # a guard failure outranks any other stop
        state.global_step = 494
        self.assertEqual(tc.terminal_record(state, {}, None)["stop_reason"], "horizon")
        state.global_step = 200
        self.assertEqual(tc.terminal_record(state, {}, None)["stop_reason"], "early_stopping_or_other")


if __name__ == "__main__":
    unittest.main()
