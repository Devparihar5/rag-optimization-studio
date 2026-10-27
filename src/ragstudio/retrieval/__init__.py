from __future__ import annotations


def recall_at_k(relevant_ids: list[str], retrieved_ids: list[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    retrieved_top_k = retrieved_ids[:k]
    hits = sum(1 for item in retrieved_top_k if item in relevant_ids)
    return hits / len(relevant_ids)


def precision_at_k(relevant_ids: list[str], retrieved_ids: list[str], k: int) -> float:
    if not retrieved_ids:
        return 0.0
    retrieved_top_k = retrieved_ids[:k]
    hits = sum(1 for item in retrieved_top_k if item in relevant_ids)
    return hits / len(retrieved_top_k)


def mrr(relevant_ids: list[str], retrieved_ids: list[str]) -> float:
    for index, item in enumerate(retrieved_ids, start=1):
        if item in relevant_ids:
            return 1.0 / index
    return 0.0


def ndcg_at_k(relevant_ids: list[str], retrieved_ids: list[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    dcg = 0.0
    for index, item in enumerate(retrieved_ids[:k], start=1):
        if item in relevant_ids:
            dcg += 1.0 / (index ** 0.5)
    ideal = 0.0
    for index in range(1, min(k, len(relevant_ids)) + 1):
        ideal += 1.0 / (index ** 0.5)
    return dcg / ideal if ideal else 0.0


def compute_metrics(relevant_ids: list[str], retrieved_ids: list[str], k: int) -> dict[str, float]:
    return {
        "recall_at_k": recall_at_k(relevant_ids, retrieved_ids, k),
        "precision_at_k": precision_at_k(relevant_ids, retrieved_ids, k),
        "mrr": mrr(relevant_ids, retrieved_ids),
        "ndcg_at_k": ndcg_at_k(relevant_ids, retrieved_ids, k),
    }
