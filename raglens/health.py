"""Saude do indice — Fase 1. Generico, roda sobre qualquer RagAdapter, zero API.

Tres modos de falha silenciosa que isto pega:
- unindexed: doc-fonte existe mas nunca entrou no indice (hook nao rodou / doc novo).
- stale: doc indexado mas o fonte mudou depois (indice desatualizado).
- orphan: doc no indice cujo fonte sumiu (deletado/movido sem poda).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .adapter import RagAdapter

# tolerancia de clock skew ao comparar mtimes (segundos)
_MTIME_EPS = 0.001


@dataclass
class HealthReport:
    adapter: str
    n_source: int
    n_indexed: int
    n_covered: int
    coverage_pct: float
    fresh_pct: float
    n_chunks: int
    n_tokens: int
    avg_chunks: float
    unindexed: list[str] = field(default_factory=list)
    stale: list[str] = field(default_factory=list)
    orphan: list[str] = field(default_factory=list)

    @property
    def is_healthy(self) -> bool:
        return not (self.unindexed or self.stale or self.orphan)

    @property
    def grade(self) -> str:
        if self.is_healthy:
            return "green"
        # orfaos/stale sao mais graves que um doc novo ainda nao indexado
        if self.orphan or self.stale:
            return "red"
        return "yellow"

    def to_dict(self) -> dict:
        return {
            "adapter": self.adapter,
            "grade": self.grade,
            "n_source": self.n_source,
            "n_indexed": self.n_indexed,
            "n_covered": self.n_covered,
            "coverage_pct": round(self.coverage_pct, 2),
            "fresh_pct": round(self.fresh_pct, 2),
            "n_chunks": self.n_chunks,
            "n_tokens": self.n_tokens,
            "avg_chunks": round(self.avg_chunks, 2),
            "unindexed": self.unindexed,
            "stale": self.stale,
            "orphan": self.orphan,
        }


def check_health(adapter: RagAdapter) -> HealthReport:
    src = {d.doc_id: d for d in adapter.source_docs()}
    idx = {d.doc_id: d for d in adapter.indexed_docs()}
    src_ids, idx_ids = set(src), set(idx)

    both = src_ids & idx_ids
    unindexed = sorted(src_ids - idx_ids)
    orphan = sorted(idx_ids - src_ids)
    stale = sorted(
        i for i in both if src[i].mtime > idx[i].indexed_mtime + _MTIME_EPS
    )

    n_source = len(src_ids)
    n_covered = len(both)
    coverage = 100.0 * n_covered / n_source if n_source else 100.0
    fresh = 100.0 * (n_covered - len(stale)) / n_covered if n_covered else 100.0

    n_chunks = sum(d.n_chunks for d in idx.values())
    n_tokens = sum(d.tokens for d in idx.values())
    avg_chunks = n_chunks / len(idx) if idx else 0.0

    return HealthReport(
        adapter=adapter.name,
        n_source=n_source,
        n_indexed=len(idx_ids),
        n_covered=n_covered,
        coverage_pct=coverage,
        fresh_pct=fresh,
        n_chunks=n_chunks,
        n_tokens=n_tokens,
        avg_chunks=avg_chunks,
        unindexed=unindexed,
        stale=stale,
        orphan=orphan,
    )
