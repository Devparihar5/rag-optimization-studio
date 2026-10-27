from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence
import re


@dataclass(slots=True)
class DocumentChunk:
    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


class RetrievalEngine:
    def __init__(self, chunks: Sequence[DocumentChunk] | None = None):
        self.chunks = list(chunks or [])

    def set_chunks(self, chunks: Sequence[DocumentChunk]) -> None:
        self.chunks = list(chunks)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"[a-z0-9]+", text.lower())

    @staticmethod
    def _score_term(query_tokens: list[str], document_tokens: list[str]) -> float:
        if not query_tokens:
            return 0.0

        query_set = set(query_tokens)
        overlap = sum(1 for token in document_tokens if token in query_set)
        return overlap / len(query_set)

    def retrieve(self, query: str, top_k: int = 5, strategy: str = "hybrid") -> list[dict[str, Any]]:
        if not query or not self.chunks:
            return []

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        scored: list[tuple[float, DocumentChunk]] = []
        for chunk in self.chunks:
            document_tokens = self._tokenize(chunk.text)
            semantic_overlap = self._score_term(query_tokens, document_tokens)
            bm25_like = sum(document_tokens.count(token) for token in set(query_tokens))
            score = semantic_overlap + (bm25_like * 0.1)
            if strategy == "exact":
                score = float(query.lower() in chunk.text.lower())
            scored.append((score, chunk))

        ranked = sorted(scored, key=lambda item: item[0], reverse=True)[:top_k]
        return [
            {
                "id": chunk.id,
                "text": chunk.text,
                "score": round(score, 4),
                "metadata": chunk.metadata,
            }
            for score, chunk in ranked
        ]


__all__ = ["DocumentChunk", "RetrievalEngine"]
