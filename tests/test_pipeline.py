from __future__ import annotations

from ragstudio.pipeline import (
    _reciprocal_rank_fusion,
    bm25_scores,
    chunk_text,
    overlap_scores,
)


def test_chunking_respects_size_and_overlap() -> None:
    text = " ".join(f"word{i}" for i in range(25))
    chunks = chunk_text(text, size=10, overlap=2, strategy="fixed-size")
    assert len(chunks) == 4
    assert all(len(chunk.split()) <= 10 for chunk in chunks)
    assert chunks[0].split()[-2:] == chunks[1].split()[:2]


def test_bm25_prefers_relevant_document() -> None:
    docs = [
        "bananas are yellow fruit",
        "quantum field equations describe particle physics",
        "banana banana banana ripening fruit",
    ]
    scores = bm25_scores("banana fruit", docs)
    assert scores[2] > scores[1]
    assert scores[0] > scores[1]


def test_dense_lexical_overlap_and_rrf_are_distinct_rankers() -> None:
    docs = ["red apple fruit", "apple pie recipe", "blue ocean water"]
    lexical = sorted(range(len(docs)), key=lambda i: bm25_scores("apple recipe", docs)[i], reverse=True)
    overlap = sorted(range(len(docs)), key=lambda i: overlap_scores("apple recipe", docs)[i], reverse=True)
    fused = _reciprocal_rank_fusion(lexical, [2, 0, 1])
    assert lexical[0] == 1
    assert overlap[0] == 1
    assert fused[0][0] in {0, 1, 2}
    assert len({item[0] for item in fused}) == 3


def test_sentence_aware_chunking_keeps_sentence_boundaries_when_possible() -> None:
    chunks = chunk_text("First short sentence. Second short sentence. Third short sentence.", 5, 0, "sentence-aware")
    assert chunks[0] == "First short sentence."
    assert all(chunk.strip() for chunk in chunks)
