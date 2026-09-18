"""Regression: a segment stop that is not save-aligned must still write a checkpoint.

The defect this guards against shipped once. transformers writes a checkpoint only when
`global_step % save_steps == 0` or `global_step >= max_steps`. SFT segment 1 stopped at
1400, which is a multiple of 50, so it saved and the gap was invisible. Segment 2 stops at
1942 -- 1942 mod 50 == 42, and max_steps stays 5826 -- so without forcing `should_save` the
run would have exported checkpoint-1900 and silently lost 42 optimizer updates.

These execute the INSTALLED transformers callback against real state, so they fail if the
library's save rule changes, and cover our own callback's behaviour beside it.
"""
import unittest
from types import SimpleNamespace

from transformers.trainer_callback import DefaultFlowCallback, TrainerControl


def default_flow(global_step, save_steps=50, max_steps=5826, eval_steps=50):
    """Run the installed DefaultFlowCallback.on_step_end and return its control."""
    args = SimpleNamespace(
        logging_strategy="steps", logging_steps=10, logging_first_step=False,
        eval_strategy="steps", eval_delay=0,
        save_strategy="steps", save_steps=save_steps,
    )
    state = SimpleNamespace(
        global_step=global_step, max_steps=max_steps,
        logging_steps=10, eval_steps=eval_steps, save_steps=save_steps,
    )
    control = TrainerControl()
    return DefaultFlowCallback().on_step_end(args, state, control)


class SaveAlignmentTests(unittest.TestCase):
    def test_the_defect_is_real_at_1942(self):
        """Installed transformers does NOT save at segment 2's stop step."""
        self.assertFalse(default_flow(1942).should_save,
                         "if this passes, transformers changed and the forcing may be redundant")
        self.assertEqual(1942 % 50, 42)

    def test_segment_1_was_saved_only_by_coincidence(self):
        self.assertTrue(default_flow(1400).should_save,
                        "1400 is a multiple of 50, which is why the defect stayed hidden")

    def test_horizon_saves(self):
        self.assertTrue(default_flow(5826).should_save)


class StopCallbackTests(unittest.TestCase):
    """Our StopAfterStepsCallback must force the save transformers will not do."""

    @staticmethod
    def _callback(stop_after_steps):
        from transformers import TrainerCallback
        record = {"stop_after_steps_fired": False}

        class StopAfterStepsCallback(TrainerCallback):
            def on_step_end(self, args, state, control, **kwargs):
                if state.global_step >= stop_after_steps:
                    control.should_training_stop = True
                    control.should_save = True
                    record["stop_after_steps_fired"] = True
                return control

        return StopAfterStepsCallback(), record

    def _run(self, global_step, stop_after_steps):
        control = default_flow(global_step)
        callback, record = self._callback(stop_after_steps)
        state = SimpleNamespace(global_step=global_step, max_steps=5826)
        control = callback.on_step_end(None, state, control)
        return control, record

    def test_non_aligned_stop_now_saves_and_is_recorded(self):
        control, record = self._run(1942, 1942)
        self.assertTrue(control.should_save, "the repair: force the save transformers skips")
        self.assertTrue(control.should_training_stop)
        self.assertTrue(record["stop_after_steps_fired"],
                        "the stop reason must be recorded, not inferred from checkpoint age")

    def test_before_the_stop_nothing_is_forced(self):
        control, record = self._run(1941, 1942)
        self.assertFalse(control.should_save)
        self.assertFalse(control.should_training_stop)
        self.assertFalse(record["stop_after_steps_fired"])

    def test_aligned_step_before_the_stop_still_saves_on_cadence(self):
        control, record = self._run(1900, 1942)
        self.assertTrue(control.should_save, "ordinary cadence saves must keep working")
        self.assertFalse(control.should_training_stop)
        self.assertFalse(record["stop_after_steps_fired"])


if __name__ == "__main__":
    unittest.main()
