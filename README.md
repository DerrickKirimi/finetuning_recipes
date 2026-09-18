# Finetuning Recipes

End-to-end post-training recipes for small language models — from continued pre-training to preference alignment. Part of the [Neural Breakdown YouTube course](https://www.youtube.com/@avb_fj).

## What This Covers

| Stage | Description |
|-------|-------------|
| **CPT** (Continued Pre-Training) | Further pre-train a base model on domain-specific tasks |
| **SFT** (Supervised Fine-Tuning) | Instruction-tune the CPT model on instruction data |
| **DPO** (Direct Preference Optimization) | Align the SFT model using pairwise human/AI preferences |
| **RLVR** | Coming next — reinforcement learning with verifiable rewards |

Videos for [CPT](https://youtu.be/B8Ur62D3J3U) and [SFT](https://youtu.be/gvZIUEL6Ruc?si=da47dP3Fad-ggAhH) are already on the channel. DPO and RLVR videos are in the works.

## Related Repos

- **[text-albumentations](https://github.com/avbiswas/text-albumentations)** — dataset generation & augmentation library
- **[neural-txt](https://github.com/avbiswas/neural-txt)** — Inference harness
- **[paper_instructions_300K-v1](https://huggingface.co/datasets/paperbd/paper_instructions_300K-v1)** — instruction dataset generated from arXiv papers

## This branch: a single-T4 port with a post-training harness

The `smollm` branch runs the full recipe — continued pre-training, SFT, DPO, a reward model and GRPO — for
SmolLM2-135M on one 16 GB Tesla T4, and adds the machinery needed to trust the results on hardware the original
configurations were not written for. The course code is extended rather than rewritten: changes to existing files are
small, and most additions are new modules. [`CHANGES-VS-UPSTREAM.md`](CHANGES-VS-UPSTREAM.md) lists every change by
stage.

In brief:

- **Precision and memory for a T4.** fp16 with loss scaling where bf16 is unavailable, 4-bit NF4 base weights with
  LoRA, measured batch geometries that preserve each stage's effective batch, and activation checkpointing for the
  reward model.
- **Exact resumption.** Every long stage can stop at a boundary and resume in a fresh process with its optimizer,
  scheduler, loss scaler, early-stopping state, random state and data position restored and checked.
- **Correctness guards.** A frozen-state digest and in-training reference-model probe for DPO; frozen-weight
  preservation across GRPO's merged-adapter decoding; data-position verification across epochs.
- **Data boundaries.** Grouped, contamination-filtered splits recorded as manifests of row indices and hashes.
- **Evaluation.** A frozen evaluation battery with deterministic scoring, pinned scorer weights and paired
  bootstrap intervals, plus pairwise and scalar LLM-judge clients.

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

Continued pre-training reads local arXiv JSONL through `load_corpus() -> Iterable[Document]` in `cpt/corpus.py`, so
other corpus sources can implement the same interface without changing the trainer.

The root project keeps CUDA training dependencies in `pyproject.toml` and `uv.lock`; MLX dependencies are installed
only on Darwin systems.


## Support

If you find this helpful, consider supporting on Patreon — it hosts all code, projects, slides, and write-ups from the YouTube channel.

[<img src="https://c5.patreon.com/external/logo/become_a_patron_button.png" alt="Become a Patron!" width="200">](https://www.patreon.com/NeuralBreakdownwithAVB)
