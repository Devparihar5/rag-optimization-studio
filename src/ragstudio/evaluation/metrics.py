from __future__ import annotations

from typing import Iterable


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def precision_at_k(retrieved: Iterable[str], relevant: Iterable[str], k: int) -> float:
    retrieved_list = list(retrieved)[:k]
    relevant_set = set(relevant)
    hits = sum(1 for item in retrieved_list if item in relevant_set)
    return _safe_divide(hits, len(retrieved_list))


def recall_at_k(retrieved: Iterable[str], relevant: Iterable[str], k: int) -> float:
    retrieved_list = list(retrieved)[:k]
    relevant_set = set(relevant)
    hits = sum(1 for item in retrieved_list if item in relevant_set)
    return _safe_divide(hits, len(relevant_set))


def mrr(retrieved: Iterable[str], relevant: Iterable[str]) -> float:
    relevant_set = set(relevant)
    for idx, item in enumerate(retrieved, start=1):
        if item in relevant_set:
            return 1.0 / idx
    return 0.0


def ndcg_at_k(retrieved: Iterable[str], relevant: Iterable[str], k: int) -> float:
    retrieved_list = list(retrieved)[:k]
    relevant_set = set(relevant)
    dcg = 0.0
    for rank, item in enumerate(retrieved_list, start=1):
        if item in relevant_set:
            dcg += 1.0 / __import__("math").log2(rank + 1)

    ideal_items = list(relevant_set)[:k]
    ideal_dcg = 0.0
    for rank, _ in enumerate(ideal_items, start=1):
        ideal_dcg += 1.0 / __import__("math").log2(rank + 1)

    return _safe_divide(dcg, ideal_dcg)


def compute_metrics(retrieved: Iterable[str], relevant: Iterable[str], *, k: int = 5) -> dict[str, float]:
    retrieved_list = list(retrieved)
    relevant_list = list(relevant)

    return {
        "precision_at_k": precision_at_k(retrieved_list, relevant_list, k),
        "recall_at_k": recall_at_k(retrieved_list, relevant_list, k),
        "mrr": mrr(retrieved_list, relevant_list),
        "ndcg_at_k": ndcg_at_k(retrieved_list, relevant_list, k),
    }


__all__ = [
    "compute_metrics",
    "precision_at_k",
    "recall_at_k",
    "mrr",
    "ndcg_at_k",
]
