"""Corpus boundary for continued pre-training documents."""

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Iterable, Iterator

from datasets import Dataset, load_dataset


@dataclass(frozen=True)
class Document:
    """The content required by the CPT trainer, independent of its source."""

    text: str


class ArxivJsonlCorpus:
    """Read the existing arXiv-derived JSONL representation in source order."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def fingerprint(self) -> str:
        """Content hash, so a rebuilt corpus at one path is never served from a stale cache."""

        digest = hashlib.sha256()
        with self.path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def __iter__(self) -> Iterator[Document]:
        rows = load_dataset("json", data_files=str(self.path), split="train")
        for row in rows:
            yield Document(text=row["text"])


def load_corpus(path: str | Path) -> Iterable[Document]:
    """Load the configured corpus without exposing its storage format to training."""

    return ArxivJsonlCorpus(path)


def _text_rows(documents: Iterable[Document], fingerprint: str | None = None):
    """Yield trainer rows one document at a time; `fingerprint` only keys the Arrow cache."""

    for document in documents:
        yield {"text": document.text}


def documents_to_dataset(documents: Iterable[Document]) -> Dataset:
    """Materialize documents in the schema expected by TRL's SFT trainer.

    Streams through Arrow instead of building one in-memory list of every document,
    so corpus size is bounded by disk rather than RAM.
    """

    fingerprint = getattr(documents, "fingerprint", None)
    return Dataset.from_generator(
        _text_rows,
        gen_kwargs={
            "documents": documents,
            "fingerprint": fingerprint() if callable(fingerprint) else None,
        },
    )
