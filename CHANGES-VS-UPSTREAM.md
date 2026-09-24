# Changes relative to the upstream course repository

This branch builds on [`avbiswas/finetuning_recipes`](https://github.com/avbiswas/finetuning_recipes) at commit
`9b3154a`. The course's pipeline, configurations, datasets and training code are the foundation of everything here.
The goal of the changes is to run that pipeline end to end on a single 16 GB Tesla T4, and to be able to check that
each stage did what it claims. Existing files change only where the port required it; most additions are new modules.

## Across all stages

| area | what changed | where |
|---|---|---|
| Hardware | capability-based precision (fp16 with loss scaling where the GPU has no native bf16), an executable environment check | `posttraining_harness/hardware.py`, `check_env.sh` |
| Inputs | immutable Hub revisions and file hashes for every model, tokenizer and dataset input | `posttraining_harness/assets.py`, `posttraining_harness/dependency_sources.py` |
| Environment | pinned GPU environment for the stages; a separate locked CPU environment for the harness | `pyproject.toml`, `uv.lock`, `posttraining_harness/pyproject.toml` |
| Data boundaries | exact prompt/passage leakage measurement; grouped splits recorded as manifests of indices and hashes, never as copied rows | `posttraining_harness/data_boundaries.py`, `posttraining_harness/sft_split.py`, `preference_optimization/split_manifest.py` |

## Continued pre-training

- `cpt/corpus.py` introduces a document interface, `load_corpus() -> Iterable[Document]`, so the trainer no longer
  parses a specific source format; `cpt/sft.py` reads through it.
- `cpt/evals.py` and `cpt/inference.py` load the exact trained adapter and parent recorded at training time.
- `posttraining_harness/cpt_gate.py` and `prepare_cpt_gate.py` run the unchanged entry point on a small fixture and
  verify that an optimizer step actually happened; `corpus_memory.py` measures the memory cost of materializing a
  corpus.

## Supervised fine-tuning

- `instruction_tuning/sft.py` gains a separate evaluation batch size and bounded, resumable segments with a fixed
  schedule horizon.
- `posttraining_harness/sft_truncation.py` finds rows whose entire response is masked after truncation;
  `sft_trainer_gate.py` inspects response masking in real trainer batches; `sft_memory_probe.py` instruments memory.

## Preference optimization (DPO)

- `preference_optimization/training_controls.py`: bounded training segments, a forced save at a stop,
  checkpoint-restoration validation, and precision selection that uses native bf16 only when the hardware has it.
- `preference_optimization/dpo_setup.py`, importable without a GPU so the logic is testable on CPU:
  - a five-component digest of all frozen state, including 4-bit quantization metadata;
  - a reference-model probe and an in-training guard that re-checks the adapter-disabled reference bitwise and stops
    with state saved on any mismatch;
  - a full optimizer-state digest, including 8-bit optimizer state;
  - verification that a resumed run trains exactly the rows the seeded sampler would have, across epoch boundaries and
    the data loader's one-batch prefetch.
- `train_preference.py` keeps its defaults; every new control is opt-in.
- Two DPO-only loss options, both off unless asked for: `--rpo_alpha` (RPO's likelihood term on the chosen answer)
  and `--ld_alpha` (LD-DPO's weight on the part of a response beyond what the two answers share). They are passed to
  TRL only for `--method dpo`, rejected for ORPO, and reported in the run record as the constructed trainer holds
  them, so a silent no-op cannot pass for a configured run.
- `preference_optimization/trl_compat.py` corrects TRL 0.24's LD-DPO masking. Upstream selects the shared-prefix
  tokens by absolute position while the per-token log-probabilities span prompt + completion and are rolled one place
  right, so every masked sum is zero: the loss stops depending on the policy and training runs to completion with a
  gradient norm of exactly 0.0. The correction ranks tokens by the cumulative completion mask on the rolled grid,
  which makes `ld_alpha = 1.0` reproduce plain DPO exactly, as the paper defines it. Applied only when `--ld_alpha`
  is set; idempotent, reversible, and recorded. Fixed upstream in TRL v1.0.0, so it matters only for the 0.2x series
  pinned here.
- `posttraining_harness/dpo_memory_probe.py` measures the memory envelope one batch size per process;
  `dpo_tokens.py` checks chat formatting and tokenization without loading weights.

## Reward model

- `reward_models/reward_training.py` and `train_reward_model_repro.py`: restartable training with explicit data and
  tokenization contracts, checkpoints that carry optimizer, cursor, best-validation and random state, and optional
  activation checkpointing. Reference and response are tokenized as a pair so truncation never removes the entire
  candidate answer; the original behaviour remains available by name.
- `reward_models/adapt_grouped_reward.py` adapts a trained reward to grouped reference scores.

## Reinforcement learning (GRPO)

- `reasoning/grpo/train_t4.py` is a restartable single-GPU port of the reference trainer. It keeps the reference's
  effective optimizer batch and optimizer updates per prompt through gradient accumulation, and checkpoints only at
  event boundaries.
- `reasoning/grpo/repro_runtime.py` holds the deterministic checkpoint controls: exact resume of the optimizer, loss
  scaler, random streams and data cursor; checkpoint-selection evaluation with isolated random state; and
  preservation of both the latest and the selected state.
- Frozen base weights are snapshotted and restored exactly around merged-adapter decoding, because merging and
  unmerging a LoRA adapter in float16 does not return the original bytes.
- `reasoning/generate_grouped_rollouts.py`, `reward_diagnostics.py`, `score_reward_diagnostics.py` and
  `analyze_reward_diagnostics.py` generate grouped rollouts and evaluate reward models in GRPO's advantage space,
  where only within-group differences matter.

## Evaluation

- `posttraining_harness/eval_battery.py`: a frozen cross-stage battery with separate instruction and continuation
  tracks, deterministic ROUGE aggregation (the default aggregator reports the median of unseeded bootstrap
  resamples), BERTScore from pinned, hash-verified weights, refusal to publish any non-finite aggregate, and an
  immutable identity for every evaluated artifact.
- `posttraining_harness/judge.py`: a reference-guided pairwise judge with both answer orders, rationale before
  verdict, response length recorded, a hard spending cap, a cache and full request logs. The API key is read from a
  file outside any working tree, refused if readable by others, and sent only in a request header.
- `posttraining_harness/scalar_judge.py` and `scalar_adjudication.py`: budgeted, resumable grouped scalar
  adjudication.
- `posttraining_harness/kev_judge.py` and `kev_analysis.py`: a pinned, open Kev-4B decision judge. Each pair is
  scored in both answer placements and three option-label rotations; raw probabilities are retained, resumable rows
  are bound to exact input bytes, and a separate analyzer recomputes every aggregate. Runtime instructions and the
  measured T4 envelope are in `posttraining_harness/KEV_JUDGE.md`.

## Tests

Each stage's additions come with tests: the harness's own suite in `posttraining_harness/test_*.py`, and stage tests
under `tests/cpt/`, `tests/reasoning/` and `tests/reward_models/`. Several run the real trainers on CPU with tiny
models to prove equivalences — for example that accumulating over 32 microbatches produces the same update as one
batch of 128, and that a paused and resumed run matches an uninterrupted one.
