"""Capability-based settings and an executable environment check (no model training)."""

import argparse
import importlib.metadata
import json
import platform

from assets import output_directory


def settings(torch):
    cuda = torch.cuda.is_available()
    bf16 = cuda and torch.cuda.is_bf16_supported(including_emulation=False)
    return {"device": "cuda" if cuda else "cpu", "bf16": bool(bf16),
            "fp16": bool(cuda and not bf16), "dtype": "bfloat16" if bf16 else "float16" if cuda else "float32",
            "attn_implementation": "sdpa", "packing": False, "padding_free": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()
    import torch
    profile = settings(torch)
    report = {"profile": profile, "python": platform.python_version(),
              "torch": torch.__version__, "cuda_build": torch.version.cuda,
              "devices": [{"name": torch.cuda.get_device_name(i),
                           "capability": list(torch.cuda.get_device_capability(i)),
                           "total_memory": torch.cuda.get_device_properties(i).total_memory}
                          for i in range(torch.cuda.device_count())],
              "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
              "training_readiness": "NOT VERIFIED: no model or optimizer step in this check"}
    # Exercise the selected dtype/device; this is a tensor check, not a training run.
    a = torch.ones((8, 8), device=profile["device"], dtype=getattr(torch, profile["dtype"]))
    report["finite_matmul"] = bool(torch.isfinite(a @ a).all())
    report["pass"] = report["finite_matmul"] and (not args.require_cuda or bool(report["devices"]))
    (output_directory(args.output) / "environment.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "packages"}, indent=2))
    raise SystemExit(0 if report["pass"] else 1)


if __name__ == "__main__":
    main()
