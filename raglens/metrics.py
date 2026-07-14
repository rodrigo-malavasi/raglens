"""Metricas de qualidade de retrieval — Fase 3. Funcoes puras, sem estado.

Cada caso de avaliacao = (ranked, expected):
- ranked: lista ordenada de doc_ids que o RAG devolveu (melhor primeiro)
- expected: conjunto de doc_ids que o gabarito diz que deveriam aparecer

`aggregate` roda um lote de casos e devolve as medias — o que a tela do RagLens
mostra no topo (acertou top-1? top-5? em que posicao?).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence


def hit_at_k(ranked: Sequence[str], expected: Iterable[str], k: int) -> bool:
    """Algum doc esperado apareceu nos primeiros k?"""
    exp = set(expected)
    return any(d in exp for d in ranked[:k])


def reciprocal_rank(ranked: Sequence[str], expected: Iterable[str]) -> float:
    """1/posicao do primeiro acerto (0 se nao achou). Base do MRR."""
    exp = set(expected)
    for i, doc in enumerate(ranked, start=1):
        if doc in exp:
            return 1.0 / i
    return 0.0


def recall_at_k(ranked: Sequence[str], expected: Iterable[str], k: int) -> float:
    """Fracao dos esperados que apareceu nos primeiros k."""
    exp = set(expected)
    if not exp:
        return 0.0
    top = set(ranked[:k])
    return len(exp & top) / len(exp)


def aggregate(
    cases: Iterable[tuple[Sequence[str], Iterable[str]]],
    ks: Sequence[int] = (1, 5),
) -> dict:
    """Media das metricas sobre um lote de casos.

    Retorna hit@k pra cada k pedido, MRR, recall@k(maior k) e a contagem de casos.
    """
    cases = list(cases)
    n = len(cases)
    if n == 0:
        return {"n": 0}

    out: dict = {"n": n}
    for k in ks:
        out[f"hit@{k}"] = sum(hit_at_k(r, e, k) for r, e in cases) / n
    out["mrr"] = sum(reciprocal_rank(r, e) for r, e in cases) / n
    kmax = max(ks)
    out[f"recall@{kmax}"] = sum(recall_at_k(r, e, kmax) for r, e in cases) / n
    return out
