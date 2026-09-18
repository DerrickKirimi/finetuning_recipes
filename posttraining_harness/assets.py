"""Resolve immutable Hub revisions and download audit inputs outside the code tree."""

import argparse
import hashlib
import json
from pathlib import Path
import urllib.request


def output_directory(value):
    path = Path(value).resolve()
    if path.is_relative_to(Path(__file__).resolve().parents[1]):
        raise ValueError("Audit outputs must be outside the repository working tree")
    path.mkdir(parents=True, exist_ok=True)
    return path


def download(url, target):
    partial = target.with_suffix(target.suffix + ".partial")
    digest = hashlib.sha256()
    with urllib.request.urlopen(url, timeout=120) as response, partial.open("wb") as out:
        while chunk := response.read(1024 * 1024):
            out.write(chunk)
            digest.update(chunk)
    partial.replace(target)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset", action="append", default=[])
    parser.add_argument("--model", action="append", default=[])
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    out = output_directory(args.output)
    manifest_path = out / "assets.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    for kind, names in [("datasets", args.dataset), ("models", args.model)]:
        for name in names:
            key = f"{kind}/{name}"
            if key not in manifest:
                url = f"https://huggingface.co/api/{kind}/{name}"
                with urllib.request.urlopen(url, timeout=60) as response:
                    metadata = json.load(response)
                manifest[key] = {"revision": metadata["sha"], "metadata_url": url,
                                 "files": [s["rfilename"] for s in metadata["siblings"]],
                                 "downloads": {}}
                manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
            entry = manifest[key]
            folder = out / kind / name.replace("/", "--") / entry["revision"]
            folder.mkdir(parents=True, exist_ok=True)
            for filename in entry["files"]:
                wanted = (filename in {"train.jsonl", "test.jsonl"} if kind == "datasets"
                          else filename in {"config.json", "tokenizer.json", "tokenizer_config.json",
                                            "special_tokens_map.json", "merges.txt", "vocab.json"})
                if not args.download or not wanted:
                    continue
                target = folder / filename
                previous = entry["downloads"].get(filename)
                if previous and target.exists():
                    with target.open("rb") as f:
                        if hashlib.file_digest(f, "sha256").hexdigest() == previous["sha256"]:
                            continue
                prefix = "datasets/" if kind == "datasets" else ""
                url = f"https://huggingface.co/{prefix}{name}/resolve/{entry['revision']}/{filename}"
                print(f"Downloading {key}@{entry['revision']}/{filename}", flush=True)
                digest = download(url, target)
                if previous and digest != previous["sha256"]:
                    raise ValueError(f"Downloaded content changed for pinned asset {key}/{filename}")
                entry["downloads"][filename] = {"path": str(target), "sha256": digest, "url": url}
                manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
            print(key, entry["revision"], flush=True)


if __name__ == "__main__":
    main()
