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

## Reproducibility audit

The `audit/` directory contains small, reusable checks for dataset boundaries,
tokenization, dependency provenance, hardware capability and citation coverage. It
also contains its own locked CPU audit environment, separate from the GPU training
environment. The checks inspect source and fixtures; they do not claim that training
or an optimizer step has succeeded.

From the repository root, create the audit environment and run its tests with:

```bash
uv sync --project audit --locked
audit/.venv/bin/python -m pytest -q audit/test_audit.py
```

To write a hardware capability report, provide an output directory:

```bash
./check_env.sh --verify --output path/to/report
```

The root project keeps CUDA training dependencies in `pyproject.toml` and `uv.lock`;
MLX dependencies are installed only on Darwin systems.


## Support

If you find this helpful, consider supporting on Patreon — it hosts all code, projects, slides, and write-ups from the YouTube channel.

[<img src="https://c5.patreon.com/external/logo/become_a_patron_button.png" alt="Become a Patron!" width="200">](https://www.patreon.com/NeuralBreakdownwithAVB)
