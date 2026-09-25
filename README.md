# Finetuning Recipes

End-to-end post-training recipes for small language models, from continued pre-training to preference alignment.
This fork is adapted from the
[Neural Breakdown YouTube Model Training course](https://www.youtube.com/playlist?list=PLHl0PLdiWkLg)
and extends its recipes for reproducible single-T4 execution and evaluation.

## What This Covers

| Stage | Description |
|-------|-------------|
| **CPT** (Continued Pre-Training) | Further pre-train a base model on domain-specific tasks |
| **SFT** (Supervised Fine-Tuning) | Instruction-tune the CPT model on instruction data |
| **DPO** (Direct Preference Optimization) | Align the SFT model using pairwise human/AI preferences |
| **RL** (Reinforcement learning) | Training Reward models and using them to train neural models |


## Related Repos

- **[text-albumentations](https://github.com/avbiswas/text-albumentations)** — dataset generation & augmentation library
- **[neural-txt](https://github.com/avbiswas/neural-txt)** — Inference harness
- **[paper_instructions_300K-v1](https://huggingface.co/datasets/paperbd/paper_instructions_300K-v1)** — instruction dataset generated from arXiv papers

## This fork: a single-T4 port with a post-training harness

I ran the course pipeline — continued pre-training, SFT, DPO, a reward model and GRPO — for
`HuggingFaceTB/SmolLM-135M` on one 16 GB Tesla T4. This fork adds resumable stage runners, explicit
data boundaries, evaluation tools and hardware-aware configurations. The course code remains the
foundation: changes to existing files are limited, while most additions live in new modules.
[`CHANGES-VS-UPSTREAM.md`](CHANGES-VS-UPSTREAM.md) is the stage-by-stage technical delta.

In brief:

- **Precision and memory for a T4.** fp16 with loss scaling where bf16 is unavailable, 4-bit NF4 base weights with
  LoRA, measured batch geometries that preserve each stage's effective batch, and activation checkpointing for the
  reward model.
- **Exact resumption.** Every long stage can stop at a boundary and resume in a fresh process with its optimizer,
  scheduler, loss scaler, early-stopping state, random state and data position restored and checked.
- **Runtime checks.** A frozen-state digest and in-training reference-model probe for DPO; frozen-weight
  preservation across GRPO's merged-adapter decoding; data-position verification across epochs.
- **Data boundaries.** Grouped splits keep related source material on one side of each boundary and are recorded as
  manifests of row indices and hashes.
- **Evaluation.** A frozen evaluation battery with deterministic scoring, pinned scorer weights and paired
  bootstrap intervals, pairwise and scalar LLM-judge clients, and a pinned open Kev-4B decision judge with
  answer-order and option-rotation controls.

### A self-hosted System 1 decision judge

I wanted a second judge for a frozen 500-pair model comparison. TypeSafe had
[temporarily paused new Jev signups](https://x.com/typesafeai/status/2102281508950307159) to preserve service quality
for existing users, so I used an open implementation of the same public SystemOne interface instead.

In this context, a **System 1 decision model** reads one state and a set of typed questions, then returns a
probability distribution for each question in one forward pass. It does not generate an answer or a rationale token
by token. I self-hosted
[`jaredpalmer/kev-4b` at the exact revision used here](https://huggingface.co/jaredpalmer/kev-4b/blob/485ace8703592fcf405488b262449990824cfed1/README.md)
on a Kaggle T4 and used it as a pairwise judge. Kev implements TypeSafe's public `/v1/systemone` contract, but it is
a different model with different training data and failure modes.

The frozen run scored every answer pair in both placements and under three option-label rotations. It completed 500
pairs in 24.7 minutes with 8.80 GiB peak reserved VRAM. Kev assigned the tested LD-DPO model continuous credit
0.5958 (95% bootstrap interval 0.5744–0.6171), directionally corroborating the earlier judge result. That agreement
is evidence from a second instrument, not ground truth. The pinned runtime, scorer, limitations and reproduction
command are in [`posttraining_harness/KEV_JUDGE.md`](posttraining_harness/KEV_JUDGE.md).

### The harness

`posttraining_harness/` holds the checks, probes, evaluation and judge code. It has its own locked CPU environment,
separate from the GPU training environment:

```bash
uv sync --project posttraining_harness --locked
posttraining_harness/.venv/bin/python -m pytest -q posttraining_harness/test_audit.py
```

Hardware capability report:

```bash
./check_env.sh --verify --output path/to/report
```

Generated fixtures, reports, logs and model artifacts belong in an output directory you supply, outside the
repository.

The Kev scorer retains all six raw probability distributions per pair. Its analyzer reports credit, bootstrap
intervals, placement sensitivity, option-rotation sensitivity and optional correlation with another pairwise judge.

Continued pre-training reads local arXiv JSONL through `load_corpus() -> Iterable[Document]` in `cpt/corpus.py`, so
other corpus sources can implement the same interface without changing the trainer.

The root project keeps CUDA training dependencies in `pyproject.toml` and `uv.lock`; MLX dependencies are installed
only on Darwin systems.


## Support

If you find this helpful, consider supporting on Patreon — it hosts all code, projects, slides, and write-ups from the YouTube channel.

[<img src="https://c5.patreon.com/external/logo/become_a_patron_button.png" alt="Become a Patron!" width="200">](https://www.patreon.com/NeuralBreakdownwithAVB)
