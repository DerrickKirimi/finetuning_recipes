"""Compatibility helpers for TRL/Transformers version skew."""

from __future__ import annotations


def patch_trl_optional_dependency_checks() -> None:
    """Make TRL 0.24 optional dependency flags real booleans.

    TRL 0.24 imports `transformers.utils.import_utils._is_package_available`
    and expects a bool when `return_version=False`. Newer Transformers returns
    `(available, version)` in that case. Non-empty tuples are truthy, so TRL
    tries to import optional extras like mergekit and llm-blender even when
    they are not installed.
    """

    import trl.import_utils as import_utils

    for name, value in vars(import_utils).items():
        if name.endswith("_available") and isinstance(value, tuple):
            setattr(import_utils, name, value[0])

    # DPO/ORPO training here does not use TRL's PairRM judge. If llm-blender is
    # present in the environment, TRL imports it while loading trainer callbacks;
    # current llm-blender releases still import Transformers' removed
    # TRANSFORMERS_CACHE symbol.
    import_utils._llm_blender_available = False


# The exact lines TRL 0.24.0 uses to split a response into its "public" prefix and the rest for LD-DPO, and the
# corrected replacement. TRL masks by absolute position, but `per_token_logps` spans prompt + completion, so
# `position_ids < completion_lengths` selects the first N positions — prompt positions, whose per-token log-probs are
# zero. Every masked sum is then 0, `all_logps` is 0 for every row, and the loss is constant: no gradient at all.
# The fix ranks tokens *within the completion* (cumulative loss mask) instead of by absolute position.
LD_ALPHA_BUGGY = """            seq_len = per_token_logps.size(1)
            position_ids = torch.arange(seq_len, device=per_token_logps.device).expand_as(per_token_logps)

            ld_mask = position_ids < public_lengths.unsqueeze(1)
            mask = position_ids < completion_lengths.unsqueeze(1)

            front_mask = (ld_mask & mask).float()
            rear_mask = (~ld_mask & mask).float()
"""
# `per_token_logps` is rolled one position right just before this block, and the baseline sums it as
# `per_token_logps[:, 1:]`. The masks must therefore live on that rolled grid and exclude column 0, so that
# ld_alpha = 1 reproduces the baseline sum exactly (the paper's definition of LD-DPO at alpha = 1).
LD_ALPHA_FIXED = """            shifted_mask = torch.roll(loss_mask, shifts=1, dims=1)
            shifted_mask[:, 0] = False
            completion_rank = shifted_mask.int().cumsum(dim=1)   # 1-based index within the completion, 0 elsewhere
            front_mask = (shifted_mask & (completion_rank <= public_lengths.unsqueeze(1))).float()
            rear_mask = (shifted_mask & (completion_rank > public_lengths.unsqueeze(1))).float()
"""


def patch_ld_dpo_mask(trainer) -> dict:
    """Correct TRL 0.24's LD-DPO masking on the trainer's own class. Returns a record of what was patched.

    Verified by tests: unpatched, any `ld_alpha` gives zero gradients; patched, `ld_alpha=1.0` reproduces plain DPO
    exactly (the paper's definition) and `ld_alpha=0.5` trains with non-zero gradients.
    """
    import hashlib
    import inspect
    import sys

    cls = type(trainer)
    if getattr(cls, "_ld_dpo_mask_patch", None):        # idempotent: patching twice would not find the block again
        return dict(cls._ld_dpo_mask_patch)
    source = inspect.getsource(cls.concatenated_forward)
    if LD_ALPHA_BUGGY not in source:
        raise RuntimeError("LD-DPO mask patch: the expected TRL 0.24 block is not in concatenated_forward")
    if source.count(LD_ALPHA_BUGGY) != 1:
        raise RuntimeError("LD-DPO mask patch: the expected block appears more than once")
    patched = source.replace(LD_ALPHA_BUGGY, LD_ALPHA_FIXED)
    # Compile inside a throwaway class so the method keeps its original indentation, and in the defining module's
    # namespace so every name it references resolves exactly as before.
    namespace = dict(vars(sys.modules[cls.__module__]))
    exec(compile("class _LDPatched:\n" + patched, "<ld_dpo_mask_patch>", "exec"), namespace)
    cls._ld_dpo_mask_original = cls.concatenated_forward
    cls.concatenated_forward = namespace["_LDPatched"].concatenated_forward
    cls._ld_dpo_mask_patch = {"class": f"{cls.__module__}.{cls.__qualname__}",
                              "original_sha256": hashlib.sha256(source.encode()).hexdigest(),
                              "patched_sha256": hashlib.sha256(patched.encode()).hexdigest()}
    return dict(cls._ld_dpo_mask_patch)


def unpatch_ld_dpo_mask(cls) -> bool:
    """Undo patch_ld_dpo_mask on a class. For tests that need the unpatched behaviour in the same process."""
    if not getattr(cls, "_ld_dpo_mask_patch", None):
        return False
    cls.concatenated_forward = cls._ld_dpo_mask_original
    del cls._ld_dpo_mask_original, cls._ld_dpo_mask_patch
    return True
