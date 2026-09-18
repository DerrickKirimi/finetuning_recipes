import argparse
import concurrent.futures
import json
import re
from io import BytesIO
from pathlib import Path

import arxiv
import requests
from pypdf import PdfReader


TOPIC_QUERIES = [
    ("cs.AI", "cat:cs.AI"),
    ("cs.CL", "cat:cs.CL"),
    ("cs.CV", "cat:cs.CV"),
    ("cs.LG", "cat:cs.LG"),
    ("cs.RO", "cat:cs.RO"),
    ("cs.IR", "cat:cs.IR"),
    ("cs.NE", "cat:cs.NE"),
    ("cs.HC", "cat:cs.HC"),
    ("cs.MA", "cat:cs.MA"),
    ("cs.CR", "cat:cs.CR"),
]

ML_TOPIC_QUERIES = [
    ("cs.LG", "cat:cs.LG"),
    ("stat.ML", "cat:stat.ML"),
    ("cs.AI", "cat:cs.AI"),
    ("cs.CL", "cat:cs.CL"),
    ("cs.CV", "cat:cs.CV"),
    ("cs.RO", "cat:cs.RO"),
    ("cs.IR", "cat:cs.IR"),
    ("cs.NE", "cat:cs.NE"),
    ("cs.MA", "cat:cs.MA"),
    ("cs.HC", "cat:cs.HC AND (machine learning OR learning OR neural OR model)"),
]


def normalize_arxiv_id(raw_id: str) -> str:
    return re.sub(r"v\d+$", "", raw_id.rsplit("/", 1)[-1])


def existing_ids(paths: list[Path]) -> set[str]:
    ids = set()
    for path in paths:
        if not path.exists():
            continue
        for txt_file in path.rglob("*.txt"):
            ids.add(txt_file.stem)
    return ids


def collect_candidates(
    *,
    target_count: int,
    max_results_per_topic: int,
    existing: set[str],
    balanced: bool,
    topic_queries: list[tuple[str, str]],
) -> list[dict]:
    selected = []
    seen = set(existing)
    per_topic_limit = (
        (target_count + len(topic_queries) - 1) // len(topic_queries)
        if balanced
        else target_count
    )

    for topic, query in topic_queries:
        topic_count = 0
        search = arxiv.Search(
            query=query,
            max_results=max_results_per_topic,
            sort_by=arxiv.SortCriterion.SubmittedDate,
        )
        # arxiv 2.0 removed Search.results(); the client owns paging and rate limiting.
        for result in arxiv.Client().results(search):
            arxiv_id = normalize_arxiv_id(result.entry_id)
            if arxiv_id in seen:
                continue
            seen.add(arxiv_id)
            selected.append(
                {
                    "arxiv_id": arxiv_id,
                    "topic": topic,
                    "title": " ".join(result.title.split()),
                    "published": result.published.isoformat(),
                    "entry_id": result.entry_id,
                }
            )
            topic_count += 1
            if len(selected) >= target_count:
                return selected
            if topic_count >= per_topic_limit:
                break

    return selected


def download_text(arxiv_id: str) -> str:
    url = f"https://export.arxiv.org/pdf/{arxiv_id}"
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    reader = PdfReader(BytesIO(response.content))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def download_one(paper: dict, output_dir: Path) -> dict:
    arxiv_id = paper["arxiv_id"]
    output_path = output_dir / f"{arxiv_id}.txt"

    if output_path.exists():
        return {**paper, "status": "exists", "chars": output_path.stat().st_size}

    try:
        text = download_text(arxiv_id)
        output_path.write_text(text, encoding="utf-8")
        return {**paper, "status": "downloaded", "chars": len(text)}
    except Exception as exc:
        return {**paper, "status": "failed", "error": str(exc)}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data_prep/cpt/papers_new_500"),
    )
    parser.add_argument(
        "--existing-dir",
        type=Path,
        action="append",
        default=[Path("data_prep/cpt/papers")],
        help="Directory of existing .txt papers to avoid by arXiv ID",
    )
    parser.add_argument("--target-count", type=int, default=500)
    parser.add_argument("--max-results-per-topic", type=int, default=200)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument(
        "--unbalanced",
        action="store_true",
        help="Fill from earlier topics first instead of applying per-topic quotas",
    )
    parser.add_argument(
        "--topic-set",
        choices=["cs-ai", "ml"],
        default="cs-ai",
        help="Topic query set to use for candidate selection",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    collisions = existing_ids(args.existing_dir + [args.output_dir])
    candidates = collect_candidates(
        target_count=args.target_count,
        max_results_per_topic=args.max_results_per_topic,
        existing=collisions,
        balanced=not args.unbalanced,
        topic_queries=(
            ML_TOPIC_QUERIES if args.topic_set == "ml" else TOPIC_QUERIES
        ),
    )
    print(
        f"selected_candidates={len(candidates)} "
        f"existing_ids={len(collisions)} output_dir={args.output_dir}"
    )

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_paper = {
            executor.submit(download_one, paper, args.output_dir): paper
            for paper in candidates
        }
        for idx, future in enumerate(
            concurrent.futures.as_completed(future_to_paper),
            start=1,
        ):
            result = future.result()
            results.append(result)
            print(
                f"[{idx}/{len(candidates)}] {result['arxiv_id']} "
                f"{result['topic']} {result['status']}"
            )

    manifest_path = args.output_dir / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for result in sorted(results, key=lambda row: row["arxiv_id"]):
            manifest.write(json.dumps(result) + "\n")

    downloaded = sum(1 for result in results if result["status"] == "downloaded")
    failed = sum(1 for result in results if result["status"] == "failed")
    print(
        f"downloaded={downloaded} failed={failed} "
        f"manifest={manifest_path}"
    )


if __name__ == "__main__":
    main()
