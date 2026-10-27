from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any, Sequence


@dataclass
class DocumentChunk:
    id: str
    text: str
    source: str
    page: int = 1


class RetrievalEngine:
    def __init__(self, chunks: Sequence[DocumentChunk] | None = None):
        self.chunks = list(chunks or [])

    def set_chunks(self, chunks: Sequence[DocumentChunk]) -> None:
        self.chunks = list(chunks)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"[a-z0-9]+", text.lower())

    def retrieve(self, query: str, top_k: int = 5, strategy: str = "hybrid") -> list[dict[str, Any]]:
        if not self.chunks:
            return []

        normalized_strategy = (strategy or "hybrid").lower()
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        scored: list[tuple[str, float]] = []
        for chunk in self.chunks:
            dense_score = self._dense_score(query_tokens, chunk.text)
            bm25_score = self._bm25_score(query_tokens, chunk.text)
            if normalized_strategy == "dense":
                final_score = dense_score
            elif normalized_strategy == "bm25":
                final_score = bm25_score
            else:
                final_score = dense_score + bm25_score
            scored.append((chunk.id, final_score))

        ranked = sorted(scored, key=lambda item: item[1], reverse=True)[:top_k]
        results: list[dict[str, Any]] = []
        chunk_by_id = {chunk.id: chunk for chunk in self.chunks}
        for chunk_id, score in ranked:
            chunk = chunk_by_id[chunk_id]
            results.append(
                {
                    "chunk_id": chunk.id,
                    "score": round(score, 4),
                    "source": chunk.source,
                    "page": chunk.page,
                    "text": chunk.text,
                }
            )
        return results

    @staticmethod
    def _dense_score(query_tokens: list[str], text: str) -> float:
        text_tokens = RetrievalEngine._tokenize(text)
        if not text_tokens:
            return 0.0
        query_set = set(query_tokens)
        overlap = sum(1 for token in query_set if token in text_tokens)
        return overlap / len(text_tokens)

    @staticmethod
    def _bm25_score(query_tokens: list[str], text: str) -> float:
        tokens = RetrievalEngine._tokenize(text)
        if not tokens:
            return 0.0

        doc_len = len(tokens)
        tf = {}
        for token in tokens:
            tf[token] = tf.get(token, 0) + 1

        score = 0.0
        for token in set(query_tokens):
            if token not in tf:
                continue
            term_freq = tf[token]
            idf = math.log((1 + 1) / (1 + term_freq)) + 1
            normalized = 1.5 * (1 - 0.75 + 0.75 * (doc_len / max(doc_len, 1)))
            score += idf * ((term_freq * (1.5 + 1)) / (term_freq + 1.5 * normalized))
        return score


__all__ = ["DocumentChunk", "RetrievalEngine"]
