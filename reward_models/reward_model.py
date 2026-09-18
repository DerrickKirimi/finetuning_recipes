import json
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from huggingface_hub import snapshot_download
from pathlib import Path

BASE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# BASE_MODEL = "distilbert/distilbert-base-cased"

EMBED_DIM = None  # auto-detected from config if None


def mean_pool(hidden, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(hidden.size()).float()
    return (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)


def meanmax_pool(hidden, attention_mask):
    mask_f = attention_mask.unsqueeze(-1).float()
    mean = (hidden * mask_f).sum(1) / mask_f.sum(1).clamp(min=1e-9)
    mx = hidden.masked_fill(mask_f == 0, float("-inf")).max(1).values
    return torch.cat([mean, mx], dim=-1)


def max_pool(hidden, attention_mask):
    mask = attention_mask.unsqueeze(-1).bool()
    return hidden.masked_fill(~mask, float("-inf")).max(1).values


def load_reward_model(path):
    """
    Load a trained MiniLM reward model.

    Usage:
        model, tokenizer = load_reward_model("paperbd/neuraltxt-reward-22M")
        score = model.score("reference text", "candidate text")
    """
    raw_path = Path(path).expanduser()
    local_path = (
        raw_path.resolve()
        if raw_path.exists()
        else Path(snapshot_download(repo_id=str(path))).resolve()
    )
    model_source = str(local_path)
    metadata = {}
    if (local_path / "reward_model_metadata.json").is_file():
        metadata = json.loads(
            (local_path / "reward_model_metadata.json").read_text(encoding="utf-8")
        )
    tokenization_contract = metadata.get("tokenization_contract", "reference_tail")
    max_length = int(metadata.get("max_length", 512))
    pooling = metadata.get("pooling")
    if tokenization_contract not in {"balanced_pair", "reference_tail"}:
        raise ValueError(f"Unsupported reward tokenization contract: {tokenization_contract}")

    # Older exports may contain only encoder weights and a head. In that case,
    # use the declared base tokenizer; do not hide errors from an included tokenizer.
    tokenizer_files = ("tokenizer.json", "tokenizer_config.json", "vocab.txt", "vocab.json")
    tokenizer_source = (
        model_source if any((local_path / name).is_file() for name in tokenizer_files) else BASE_MODEL
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)

    # Encoder
    if (local_path / "model.safetensors").exists():
        encoder = AutoModel.from_pretrained(model_source)
    else:
        encoder = AutoModel.from_pretrained(BASE_MODEL)
    encoder.eval()

    # Head (~3KB)
    # Auto-detect embedding dim and pooling from the saved head's input width:
    # heads trained with mean+max concat pooling are 2x hidden_size wide.
    dim = EMBED_DIM or encoder.config.hidden_size
    head_state = None
    for fname in ["head_weights.pt", "head_weights.bin"]:
        hp = local_path / fname
        if hp.exists():
            head_state = torch.load(str(hp), weights_only=True, map_location="cpu")
            break
    if head_state is None:
        raise FileNotFoundError(f"No head_weights.pt or head_weights.bin in {local_path}")
    pool_fn = mean_pool
    in_dim = head_state["1.weight"].shape[1]
    if pooling is not None:
        pool_fns = {"mean": mean_pool, "max": max_pool, "meanmax": meanmax_pool}
        if pooling not in pool_fns:
            raise ValueError(f"Unsupported reward pooling contract: {pooling}")
        expected_dim = dim * (2 if pooling == "meanmax" else 1)
        if in_dim != expected_dim:
            raise ValueError(
                f"Reward head width {in_dim} does not match {pooling} pooling width {expected_dim}"
            )
        pool_fn = pool_fns[pooling]
    elif in_dim == 2 * dim:
        pooling = "meanmax"
        pool_fn = meanmax_pool
    else:
        pooling = "mean"
    dim = in_dim
    head = nn.Sequential(nn.Dropout(0.1), nn.Linear(dim, 1))
    if head_state is not None:
        head.load_state_dict(head_state)
    head.eval()

    class RewardScorer:
        def __init__(self):
            self.encoder = encoder
            self.head = head
            self.tokenization_contract = tokenization_contract
            self.pooling = pooling
            self.max_length = max_length

        def score(self, reference, response):
            if tokenization_contract == "balanced_pair":
                enc = tokenizer(
                    reference,
                    response,
                    return_tensors="pt",
                    truncation="longest_first",
                    max_length=max_length,
                )
            else:
                text = f"{reference} [SEP] {response}"
                enc = tokenizer(
                    text, return_tensors="pt", truncation=True, max_length=max_length
                )
            with torch.no_grad():
                outputs = self.encoder(
                    input_ids=enc["input_ids"],
                    attention_mask=enc["attention_mask"],
                )
                pooled = pool_fn(outputs.last_hidden_state, enc["attention_mask"])
                return self.head(pooled).item()

        def score_batch(self, references, responses, batch_size=128):
            if len(references) != len(responses):
                raise ValueError("references and responses must have the same length")
            scores = []
            for start in range(0, len(references), batch_size):
                batch_refs = references[start:start + batch_size]
                batch_resps = responses[start:start + batch_size]
                if tokenization_contract == "balanced_pair":
                    enc = tokenizer(
                        batch_refs,
                        batch_resps,
                        padding=True,
                        truncation="longest_first",
                        max_length=max_length,
                        return_tensors="pt",
                    )
                else:
                    texts = [f"{r} [SEP] {c}" for r, c in zip(batch_refs, batch_resps)]
                    enc = tokenizer(
                        texts,
                        padding=True,
                        truncation=True,
                        max_length=max_length,
                        return_tensors="pt",
                    )
                with torch.no_grad():
                    outputs = self.encoder(
                        input_ids=enc["input_ids"],
                        attention_mask=enc["attention_mask"],
                    )
                    pooled = pool_fn(outputs.last_hidden_state, enc["attention_mask"])
                    scores.extend(self.head(pooled).squeeze(-1).tolist())
            return scores

        def batch_score(self, responses, references, batch_size=128):
            """Match the ``NeuralTxtReward`` batch interface used by training.

            The historical local loader exposed ``score_batch(reference,
            response)`` while :mod:`reasoning.env` detects and calls
            ``batch_score(response, reference)``.  Keeping both spellings makes
            old callers work and lets GRPO score one rollout group in a single
            encoder pass instead of falling back to one forward per candidate.
            """
            return self.score_batch(
                references=references,
                responses=responses,
                batch_size=batch_size,
            )

    return RewardScorer(), tokenizer
