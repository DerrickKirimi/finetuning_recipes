# Kev-4B pairwise decision judge

`kev_judge.py` runs an open decision model as an independent pairwise judge. It was built for comparisons where two
candidate models answer the same task and a reference answer is available. It uses
[`jaredpalmer/kev-4b`](https://huggingface.co/jaredpalmer/kev-4b) with the model and base revisions frozen in code,
and verifies the required checkpoint files byte for byte before loading them. The matching Kev source is
[`jaredpalmer/kev`](https://github.com/jaredpalmer/kev).

This path is separate from `judge.py`, which calls a text-generating API judge. Kev returns a probability
distribution in one forward pass. It is an independent instrument, not a human label or ground truth.

## Protocol

For every pair the scorer:

1. presents candidate A on the left and candidate B on the right;
2. rotates the three option labels (`answer_a`, `answer_b`, `tie`) three times;
3. repeats the rotations with the candidates swapped;
4. combines rotations within a placement with a normalized geometric mean;
5. averages `P(candidate A) + 0.5 P(tie)` across the two placements.

All six raw distributions, token counts, placement aggregates, and the exact input-row hash are retained. The
separate `kev_analysis.py` program recomputes the aggregates from those raw distributions and refuses changed,
missing, duplicated, or reordered inputs.

The input is newline-delimited JSON. Candidate names are arbitrary, but `answers` must contain exactly the two names
given by `candidate_a` and `candidate_b`:

```json
{"pair_id":"example:1","task":"Answer the question.","reference":"Reference answer.","answers":{"model_x":"First answer.","model_y":"Second answer."},"candidate_a":"model_x","candidate_b":"model_y"}
```

## Frozen implementation

The scorer pins and verifies:

| component | revision |
|---|---|
| Kev source | `557598fced1dada75dfbf36ed144dce309ac6ceb` |
| `jaredpalmer/kev-4b` | `485ace8703592fcf405488b262449990824cfed1` |
| `Qwen/Qwen3.5-4B-Base` | `1001bb4d826a52d1f399e183466143f4da7b741b` |

The validated T4 runtime used Python 3.12.13, Torch 2.10.0+cu128, Transformers 5.17.0, PEFT 0.21.0, Accelerate
1.15.0, Flash Linear Attention 0.5.2, Pydantic 2.12.3, and Triton 3.6.0. It loaded the LoRA adapter unmerged in FP16
with SDPA. Kaggle's then-preinstalled `torchao==0.10.0` was unused and incompatible with PEFT 0.21, so the isolated
runtime removed it. The scorer now stops with an explanation if an incompatible old `torchao` remains installed.

Use a dedicated environment because those serving versions intentionally differ from the training and CPU-harness
locks in this repository:

```bash
git clone https://github.com/jaredpalmer/kev.git /workspace/kev
git -C /workspace/kev checkout 557598fced1dada75dfbf36ed144dce309ac6ceb

python -m pip install transformers==5.17.0 peft==0.21.0 accelerate==1.15.0 \
  pydantic==2.12.3 flash-linear-attention==0.5.2
python -m pip uninstall -y torchao
```

The exact CUDA Torch build should match the host image. Do not replace a working provider CUDA stack merely to match
the recorded patch version.

## Score and verify

Run from a checkout of this branch. The cache must be outside the Git working tree. On Kaggle, `/kaggle/temp` avoids
promoting the roughly 9.3 GB base-model download as notebook output.

```bash
python -m posttraining_harness.kev_judge \
  --pairs /workspace/inputs/pairs.jsonl \
  --kev-source /workspace/kev \
  --cache /workspace/cache/kev \
  --output-dir /workspace/output/kev \
  --require-device-substring T4 \
  --max-rejections 0

python -m posttraining_harness.kev_analysis \
  --pairs /workspace/inputs/pairs.jsonl \
  --rows /workspace/output/kev/kev-rows.jsonl \
  --summary /workspace/output/kev/kev-summary.json \
  --max-rejections 0 \
  --output /workspace/output/kev/kev-analysis.json
```

Pass `--checkpoint /path/to/snapshot` to avoid another adapter download. Local and downloaded snapshots are held to
the same embedded file hashes. `--comparison other-judge-summary.json` adds pair-aligned Pearson and Spearman
correlations when the comparison file contains `per_pair` records with `pair_id` and `credit_model_a`.

The scorer is append-only and resumable. It accepts an existing output only when its rows form an in-order prefix of
the current input and every saved row hash still matches. A complete passing summary makes a repeated invocation a
no-op.

## Measured envelope and limits

The validated 500-pair run completed on one Tesla T4 in 24.7 minutes, including 20.7 minutes of scoring, and reserved
8.80 GiB of VRAM at peak. That measurement is a reference point, not a provider guarantee.

Kev-4B was trained with a shorter state limit than some evaluated requests. The serving encoder accepts the longer
states, but this remains an out-of-distribution use. Both answer placement and option-label rotation had measurable
effects, which is why neither control is optional. A result can corroborate or challenge another judge's direction;
it does not establish factual correctness by itself.

No provider account identifier or credential is part of this workflow. Public Hub assets do not require a token. If
a platform requires authentication, supply it through that platform's secret store or a process environment variable;
never put it in the pair file, command line, output directory, or Git tree.
