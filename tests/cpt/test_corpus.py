import json

from datasets import load_dataset

from cpt.corpus import Document, documents_to_dataset, load_corpus


def test_arxiv_jsonl_adapter_preserves_reference_text_rows(tmp_path):
    path = tmp_path / "corpus.jsonl"
    texts = ["first paper text", "second paper text"]
    path.write_text("".join(json.dumps({"text": text}) + "\n" for text in texts))

    reference = load_dataset("json", data_files=str(path), split="train")
    documents = list(load_corpus(path))
    adapted = documents_to_dataset(documents)

    assert documents == [Document(text=text) for text in texts]
    assert adapted.column_names == ["text"]
    assert adapted[:]["text"] == reference[:]["text"] == texts


def test_corpus_is_reiterable_and_order_preserving(tmp_path):
    """The trainer materializes the corpus once, but the boundary must not be one-shot."""

    path = tmp_path / "corpus.jsonl"
    texts = [f"document {index}" for index in range(50)]
    path.write_text("".join(json.dumps({"text": text}) + "\n" for text in texts))

    corpus = load_corpus(path)
    assert [document.text for document in corpus] == texts
    assert [document.text for document in corpus] == texts
    assert documents_to_dataset(corpus)[:]["text"] == texts


def test_rebuilt_corpus_at_one_path_is_not_served_from_a_stale_cache(tmp_path):
    """Streaming through Arrow keys a cache; the key must follow content, not path."""

    path = tmp_path / "corpus.jsonl"
    path.write_text(json.dumps({"text": "original"}) + "\n")
    first = load_corpus(path)
    original_fingerprint = first.fingerprint()
    assert documents_to_dataset(first)[:]["text"] == ["original"]

    path.write_text(json.dumps({"text": "replaced"}) + "\n")
    second = load_corpus(path)
    assert second.fingerprint() != original_fingerprint
    assert documents_to_dataset(second)[:]["text"] == ["replaced"]
