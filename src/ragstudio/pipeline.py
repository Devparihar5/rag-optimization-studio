"""Deterministic RAG pipeline execution and provider inference adapters."""
from __future__ import annotations

import json
import math
import re
import time
from collections import Counter
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen
from typing import Any

from fastapi import HTTPException

try:
    from ragstudio.model_registry import get_embedding_model_runtime_config, get_model_runtime_config
except ModuleNotFoundError:
    from src.ragstudio.model_registry import get_embedding_model_runtime_config, get_model_runtime_config


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def chunk_text(text: str, size: int, overlap: int, strategy: str) -> list[str]:
    words = (text or "").split()
    if not words:
        return []
    if strategy == "sentence-aware":
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        chunks: list[str] = []
        current: list[str] = []
        for sentence in sentences:
            sentence_words = sentence.split()
            if current and len(current) + len(sentence_words) > size:
                chunks.append(" ".join(current))
                current = current[-overlap:] if overlap else []
            if len(sentence_words) > size:
                if current:
                    chunks.append(" ".join(current))
                    current = []
                for start in range(0, len(sentence_words), max(1, size - overlap)):
                    chunks.append(" ".join(sentence_words[start:start + size]))
            else:
                current.extend(sentence_words)
        if current:
            chunks.append(" ".join(current))
        return chunks
    # Recursive currently uses paragraph/sentence boundaries, then word windows.
    if strategy == "recursive":
        pieces = re.split(r"\n\s*\n|(?<=[.!?])\s+", text.strip())
        chunks = []
        current: list[str] = []
        for piece in pieces:
            piece_words = piece.split()
            if current and len(current) + len(piece_words) > size:
                chunks.append(" ".join(current))
                current = current[-overlap:] if overlap else []
            while len(piece_words) > size:
                take = max(1, size - overlap)
                chunks.append(" ".join(current + piece_words[:max(0, size-len(current))]))
                current = []
                piece_words = piece_words[take:]
            current.extend(piece_words)
        if current:
            chunks.append(" ".join(current))
        return [chunk for chunk in chunks if chunk.strip()]
    step = max(1, size - overlap)
    return [" ".join(words[i:i + size]) for i in range(0, len(words), step)]


def bm25_scores(query: str, documents: list[str]) -> list[float]:
    q = set(tokens(query))
    docs = [tokens(doc) for doc in documents]
    if not docs:
        return []
    avg_len = sum(map(len, docs)) / len(docs) or 1
    df = Counter(token for doc in docs for token in set(doc))
    scores = []
    for doc in docs:
        freq = Counter(doc)
        score = 0.0
        for term in q:
            if not freq[term]:
                continue
            idf = math.log(1 + (len(docs) - df[term] + 0.5) / (df[term] + 0.5))
            score += idf * (freq[term] * 2.2) / (freq[term] + 1.2 * (0.25 + 0.75 * len(doc) / avg_len))
        scores.append(score)
    return scores


def overlap_scores(query: str, documents: list[str]) -> list[float]:
    q = set(tokens(query))
    return [len(q.intersection(tokens(doc))) / max(1, len(q)) for doc in documents]


def _json_request(url: str, payload: dict[str, Any], headers: dict[str, str] | None = None, timeout: int = 45) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    request_headers = {"Content-Type": "application/json", **(headers or {})}
    req = Request(url, data=body, headers=request_headers, method="POST")
    try:
        with urlopen(req, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:1000]
        raise RuntimeError(f"Provider request failed ({exc.code}): {detail}") from exc
    except (URLError, TimeoutError) as exc:
        raise RuntimeError(f"Could not reach model provider: {exc}") from exc


def _embedding_vectors(config: dict[str, Any], texts: list[str]) -> list[list[float]]:
    provider, model = config["provider"], config["model_id"]
    key, base = config.get("api_key"), (config.get("base_url") or "").rstrip("/")
    if provider == "openai":
        result = _json_request(f"{base or 'https://api.openai.com/v1'}/embeddings",
                               {"model": model, "input": texts},
                               {"Authorization": f"Bearer {key}"})
        data = sorted(result.get("data", []), key=lambda item: item.get("index", 0))
        vectors = [item.get("embedding") for item in data]
    elif provider == "gemini":
        vectors = []
        for text in texts:
            result = _json_request(
                f"https://generativelanguage.googleapis.com/v1beta/models/{quote(model, safe='')}:embedContent",
                {"model": f"models/{model}", "content": {"parts": [{"text": text}]}},
                {"x-goog-api-key": key or ""},
            )
            vectors.append(result.get("embedding", {}).get("values"))
    elif provider == "ollama":
        result = _json_request(f"{base or 'http://localhost:11434'}/api/embed",
                               {"model": model, "input": texts})
        vectors = result.get("embeddings", [])
    else:
        raise RuntimeError(f"Unsupported embedding provider: {provider}")
    if len(vectors) != len(texts) or any(not isinstance(v, list) or not v for v in vectors):
        raise RuntimeError("Embedding provider returned an invalid vector response.")
    return vectors


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def _generate(config: dict[str, Any], prompt: str, system: str = "Answer using only the supplied context. If it is insufficient, say so.") -> tuple[str, dict[str, Any]]:
    provider, model = config["provider"], config["model_id"]
    key, base = config.get("api_key"), (config.get("base_url") or "").rstrip("/")
    if provider == "openai":
        result = _json_request(f"{base or 'https://api.openai.com/v1'}/chat/completions",
                               {"model": model, "messages": [{"role": "system", "content": system},
                                                             {"role": "user", "content": prompt}]},
                               {"Authorization": f"Bearer {key}"})
        choices = result.get("choices", [])
        if not choices:
            raise RuntimeError("Generation provider returned no choices.")
        return choices[0].get("message", {}).get("content", ""), result.get("usage", {})
    if provider == "gemini":
        result = _json_request(
            f"https://generativelanguage.googleapis.com/v1beta/models/{quote(model, safe='')}:generateContent",
            {"systemInstruction": {"parts": [{"text": system}]},
             "contents": [{"parts": [{"text": prompt}]}]},
            {"x-goog-api-key": key or ""},
        )
        candidates = result.get("candidates", [])
        if not candidates:
            raise RuntimeError("Generation provider returned no candidates.")
        answer = "".join(part.get("text", "") for part in candidates[0].get("content", {}).get("parts", []))
        usage = result.get("usageMetadata", {})
        return answer, {"prompt_tokens": usage.get("promptTokenCount"),
                        "completion_tokens": usage.get("candidatesTokenCount"),
                        "total_tokens": usage.get("totalTokenCount")}
    if provider == "ollama":
        result = _json_request(f"{base or 'http://localhost:11434'}/api/chat",
                               {"model": model, "stream": False,
                                "messages": [{"role": "system", "content": system},
                                             {"role": "user", "content": prompt}]})
        return result.get("message", {}).get("content", ""), {
            "prompt_tokens": result.get("prompt_eval_count"),
            "completion_tokens": result.get("eval_count"),
        }
    raise RuntimeError(f"Unsupported generation provider: {provider}")


def _reciprocal_rank_fusion(*rankings: list[int], k: int = 60) -> list[tuple[int, float]]:
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, index in enumerate(ranking, 1):
            scores[index] = scores.get(index, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)


def _metric_results(question: dict[str, Any], retrieved: list[dict[str, Any]], top_k: int) -> dict[str, Any]:
    relevant = [str(item).strip() for item in question.get("relevant_passages", []) if str(item).strip()]
    if not relevant:
        return {"precision_at_k": None, "recall_at_k": None, "mrr": None, "ndcg_at_k": None}
    def is_relevant(chunk: str) -> bool:
        chunk_words = set(tokens(chunk))
        for target in relevant:
            target_words = set(tokens(target))
            if target_words and (target_words.issubset(chunk_words) or len(target_words & chunk_words) / len(target_words) >= 0.8):
                return True
        return False
    flags = [is_relevant(item["text"]) for item in retrieved[:top_k]]
    hits = sum(flags)
    precision = hits / max(1, top_k)
    recall = min(1.0, hits / len(relevant))
    reciprocal_rank = next((1.0 / i for i, flag in enumerate(flags, 1) if flag), 0.0)
    dcg = sum((1.0 / math.log2(i + 1)) for i, flag in enumerate(flags, 1) if flag)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(relevant), top_k) + 1))
    return {"precision_at_k": precision, "recall_at_k": recall, "mrr": reciprocal_rank,
            "ndcg_at_k": dcg / ideal if ideal else 0.0}


def execute_experiment(dataset: dict[str, Any], documents: list[dict[str, Any]],
                       questions: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    if not documents:
        raise HTTPException(status_code=422, detail="The selected dataset has no uploaded documents.")
    if not questions:
        raise HTTPException(status_code=422, detail="Attach a benchmark with at least one question before running.")
    chunk_size, overlap, top_k = config["chunk_size"], config["overlap"], config["top_k"]
    if overlap >= chunk_size:
        raise HTTPException(status_code=422, detail="Chunk overlap must be smaller than chunk size.")
    chunks: list[dict[str, Any]] = []
    for document in documents:
        for index, text in enumerate(chunk_text(document.get("extracted_text", ""), chunk_size, overlap,
                                                 config.get("chunking_strategy", "fixed-size"))):
            chunks.append({"text": text, "filename": document["filename"], "document_id": document["id"],
                           "chunk_index": index})
    if not chunks:
        raise HTTPException(status_code=422, detail="No text could be extracted from the selected dataset.")
    texts = [item["text"] for item in chunks]
    embedding_config = get_embedding_model_runtime_config(config["embedding_model"])
    generation_config = get_model_runtime_config(config["generation_model"])
    # Dense vectors are computed once per run and reused for all benchmark questions.
    vectors = _embedding_vectors(embedding_config, texts) if config["retrieval_method"] in {"dense", "hybrid_rrf"} else []
    results = []
    run_started = time.perf_counter()
    for case_index, question in enumerate(questions):
        question_text = str(question.get("question", "")).strip()
        if not question_text:
            continue
        started = time.perf_counter()
        retrieval_query = question_text
        if config.get("query_rewrite"):
            rewritten, _ = _generate(generation_config,
                f"Rewrite the following user question as a concise search query. Return only the query.\n\nQuestion: {question_text}",
                "You rewrite search queries faithfully without answering them.")
            retrieval_query = rewritten.strip() or question_text
        rankings: list[list[int]] = []
        lexical_scores = bm25_scores(retrieval_query, texts)
        lexical_rank = sorted(range(len(texts)), key=lambda i: lexical_scores[i], reverse=True)
        if config["retrieval_method"] == "bm25":
            ranked = [(i, lexical_scores[i]) for i in lexical_rank]
        elif config["retrieval_method"] == "dense":
            query_vector = _embedding_vectors(embedding_config, [retrieval_query])[0]
            dense_scores = [_cosine(query_vector, vector) for vector in vectors]
            ranked = sorted(enumerate(dense_scores), key=lambda item: item[1], reverse=True)
        elif config["retrieval_method"] == "hybrid_rrf":
            query_vector = _embedding_vectors(embedding_config, [retrieval_query])[0]
            dense_scores = [_cosine(query_vector, vector) for vector in vectors]
            dense_rank = sorted(range(len(texts)), key=lambda i: dense_scores[i], reverse=True)
            fused = _reciprocal_rank_fusion(lexical_rank, dense_rank)
            ranked = [(i, score) for i, score in fused]
        else:
            raise HTTPException(status_code=422, detail=f"Unsupported retrieval method: {config['retrieval_method']}")
        retrieved = [dict(chunks[i], score=float(score)) for i, score in ranked[:top_k]]
        if config.get("reranking"):
            rerank = overlap_scores(retrieval_query, [item["text"] for item in retrieved])
            retrieved = [item for _, item in sorted(zip(rerank, retrieved), key=lambda pair: pair[0], reverse=True)]
            for item, score in zip(retrieved, sorted(rerank, reverse=True)):
                item["rerank_score"] = score
        context = "\n\n".join(f"[{i+1}] {item['filename']} (chunk {item['chunk_index']}): {item['text']}"
                              for i, item in enumerate(retrieved))
        answer, usage = _generate(generation_config,
            f"Question: {question_text}\n\nRetrieved context:\n{context}\n\nAnswer the question based on the context.",
            "Answer using only the supplied context. Cite evidence by source filename when possible. If insufficient, say so.")
        elapsed_ms = (time.perf_counter() - started) * 1000
        metrics = _metric_results(question, retrieved, top_k)
        results.append({
            "index": case_index, "question": question_text, "expected_answer": question.get("answer"),
            "relevant_passages": question.get("relevant_passages", []), "retrieved": retrieved,
            "answer": answer, "metrics": metrics, "latency_ms": round(elapsed_ms, 2),
            "token_usage": usage or None, "cost": None, "retrieval_query": retrieval_query,
        })
    numeric_metrics: dict[str, float | None] = {}
    for metric in ("precision_at_k", "recall_at_k", "mrr", "ndcg_at_k"):
        values = [case["metrics"][metric] for case in results if case["metrics"][metric] is not None]
        numeric_metrics[metric] = round(sum(values) / len(values), 4) if values else None
    token_counts = [case["token_usage"].get("total_tokens") for case in results
                    if case["token_usage"] and isinstance(case["token_usage"].get("total_tokens"), (int, float))]
    if token_counts:
        numeric_metrics["total_tokens"] = float(sum(token_counts))
    numeric_metrics["avg_latency_ms"] = round(sum(case["latency_ms"] for case in results) / len(results), 2) if results else None
    numeric_metrics["duration_ms"] = round((time.perf_counter() - run_started) * 1000, 2)
    numeric_metrics["cost"] = None
    return {"cases": results, "metrics": numeric_metrics, "question_count": len(results),
            "evaluated_count": sum(case["metrics"]["mrr"] is not None for case in results)}
