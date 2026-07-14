"""Formatacao humana dos relatorios (a versao --json sai direto do to_dict)."""

from __future__ import annotations

from .health import HealthReport

_LAMP = {"green": "🟢", "yellow": "🟡", "red": "🔴"}


def _sample(items: list[str], limit: int = 10) -> str:
    shown = items[:limit]
    extra = len(items) - len(shown)
    lines = [f"    - {x}" for x in shown]
    if extra > 0:
        lines.append(f"    … +{extra}")
    return "\n".join(lines)


def format_health_text(rep: HealthReport) -> str:
    lamp = _LAMP.get(rep.grade, "")
    out = [
        f"{lamp} RagLens · saude do indice · adapter='{rep.adapter}'",
        "",
        f"  cobertura   {rep.coverage_pct:5.1f}%  ({rep.n_covered}/{rep.n_source} docs-fonte no indice)",
        f"  frescor     {rep.fresh_pct:5.1f}%  (dos cobertos, quantos nao estao stale)",
        f"  chunks      {rep.n_chunks}  (media {rep.avg_chunks:.1f}/doc · {rep.n_tokens} tokens)",
        "",
        f"  ⚠ nao-indexados : {len(rep.unindexed)}",
        f"  ⚠ stale         : {len(rep.stale)}",
        f"  ⚠ orfaos        : {len(rep.orphan)}",
    ]
    if rep.unindexed:
        out += ["", "  nao-indexados (fonte existe, falta no indice):", _sample(rep.unindexed)]
    if rep.stale:
        out += ["", "  stale (fonte mudou depois de indexar):", _sample(rep.stale)]
    if rep.orphan:
        out += ["", "  orfaos (no indice, fonte sumiu):", _sample(rep.orphan)]
    if rep.is_healthy:
        out += ["", "  ✓ indice integro — nada a fazer."]
    return "\n".join(out)
