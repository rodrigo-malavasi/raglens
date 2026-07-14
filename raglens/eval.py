"""Harness de qualidade — Fase 3. Generico: roda um gabarito contra um ranker.

O ranker e um callable query->[doc_ids]. Desacopla a metrica do RAG concreto:
- CLI usa `adapter.search` (contrato portavel).
- Servidor usa o rerank com pesos custom (tuning ao vivo), mesma funcao `evaluate`.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path

from .metrics import aggregate, hit_at_k, recall_at_k, reciprocal_rank


def load_golden(path: str | Path) -> list[dict]:
    cases = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        cases.append(json.loads(line))
    return cases


def first_hit_rank(ranked: Sequence[str], expected) -> int | None:
    exp = set(expected)
    for i, doc in enumerate(ranked, start=1):
        if doc in exp:
            return i
    return None


def evaluate(
    golden: list[dict],
    rank_fn: Callable[[str], list[str]],
    ks: Sequence[int] = (1, 5),
    k_ret: int = 10,
) -> dict:
    cases = []
    pairs = []
    for g in golden:
        q = g["query"]
        exp = g.get("expected_paths", [])
        ranked = list(rank_fn(q))[:k_ret]
        pairs.append((ranked, exp))
        rank = first_hit_rank(ranked, exp)
        cases.append(
            {
                "query": q,
                "expected": exp,
                "ranked": ranked[:5],
                "tipo": g.get("tipo", "canon"),
                "first_hit_rank": rank,
                "hit1": hit_at_k(ranked, exp, 1),
                "hit5": hit_at_k(ranked, exp, 5),
                "recall5": round(recall_at_k(ranked, exp, 5), 3),
                "rr": round(reciprocal_rank(ranked, exp), 3),
            }
        )
    return {"aggregate": aggregate(pairs, ks=ks), "cases": cases}
