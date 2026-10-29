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


def test_dense_retrieval_uses_embedding_vectors_and_can_differ_from_bm25(monkeypatch) -> None:
    from ragstudio import pipeline

    monkeypatch.setattr(pipeline, "get_embedding_model_runtime_config", lambda _model_id: {"provider": "mock", "model_id": "embed"})
    monkeypatch.setattr(pipeline, "get_model_runtime_config", lambda _model_id: {"provider": "mock", "model_id": "generate"})

    def fake_vectors(_config, texts):
        if len(texts) > 1:
            return [[1.0, 0.0], [0.0, 1.0]]
        return [[0.0, 1.0]]

    monkeypatch.setattr(pipeline, "_embedding_vectors", fake_vectors)
    monkeypatch.setattr(pipeline, "_generate", lambda *args, **kwargs: ("mock answer", {}))
    docs = [
        {"id": "doc1", "filename": "lexical.txt", "extracted_text": "search term"},
        {"id": "doc2", "filename": "semantic.txt", "extracted_text": "completely different wording"},
    ]
    base = {
        "chunking_strategy": "fixed-size", "chunk_size": 100, "overlap": 0,
        "top_k": 1, "reranking": False, "query_rewrite": False,
        "embedding_model": "embedding-id", "generation_model": "generation-id",
    }
    question = [{"question": "search term", "relevant_passages": []}]
    dense = pipeline.execute_experiment({}, docs, question, {**base, "retrieval_method": "dense"})
    sparse = pipeline.execute_experiment({}, docs, question, {**base, "retrieval_method": "bm25"})
    assert dense["cases"][0]["retrieved"][0]["filename"] == "semantic.txt"
    assert sparse["cases"][0]["retrieved"][0]["filename"] == "lexical.txt"
