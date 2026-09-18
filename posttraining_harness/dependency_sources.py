"""Archive installed source and hash-verified wheel source for dependency audits."""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import tomllib
import zipfile

from assets import download, output_directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--lock", required=True)
    args = parser.parse_args()
    out = output_directory(args.output)
    records = {}
    for package in ["trl", "transformers", "text-albumentations", "neural-txt", "datasets"]:
        dist = importlib.metadata.distribution(package)
        selected = [str(f) for f in dist.files if str(f).endswith(("sft_config.py", "sft_trainer.py",
            "dpo_config.py", "dpo_trainer.py", "training_args.py", "data_utils.py", "runtime.py", "reward.py", "models.py"))]
        for filename in selected:
            data = Path(dist.locate_file(filename)).read_bytes()
            target = out / f"{package}-{dist.version}" / filename
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            records[str(target.relative_to(out))] = {"sha256": hashlib.sha256(data).hexdigest(), "origin": "installed wheel"}
    lock = tomllib.loads(Path(args.lock).read_text())
    for package in ["unsloth", "unsloth-zoo"]:
        candidates = [p for p in lock["package"] if p["name"] == package]
        p = max(candidates, key=lambda p: tuple(int(x) for x in p["version"].split(".")))
        wheel = p["wheels"][0]
        target = out / wheel["url"].rsplit("/", 1)[-1]
        actual = download(wheel["url"], target)
        if "sha256:" + actual != wheel["hash"]:
            raise ValueError("Wheel hash does not match lock")
        with zipfile.ZipFile(target) as archive:
            for name in archive.namelist():
                if not name.endswith(("chat_templates.py", "dataset_utils.py")):
                    continue
                path = out / f"{package}-{p['version']}" / Path(name).name
                path.parent.mkdir(parents=True, exist_ok=True)
                data = archive.read(name)
                path.write_bytes(data)
                records[str(path.relative_to(out))] = {"sha256": hashlib.sha256(data).hexdigest(),
                    "wheel_url": wheel["url"], "wheel_sha256": actual, "member": name}
    (out / "sources.json").write_text(json.dumps(records, indent=2) + "\n")
    print(f"Archived {len(records)} source files")


if __name__ == "__main__":
    main()
