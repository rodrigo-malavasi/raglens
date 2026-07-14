"""Adapter do Second Brain — primeiro cliente / dogfood do RagLens.

E o UNICO ponto que conhece o RAG do SB (sqlite-vec + FTS5 + voyage). Reaproveita
`sb_vector_lib` (a lib real do indexador, em ~/.claude/scripts) pra nao duplicar a
regra de "o que deveria estar indexado" (SKIP_PATTERNS) — se duplicasse, driftaria.

Health nao precisa de API: le doc_meta (tabela comum) + varre os .md do vault.
search() so e usado na Fase 3 e ai sim sobe voyage + sqlite-vec via connect_db.
"""

from __future__ import annotations

import os
import sqlite3
import sys
from pathlib import Path

from raglens.adapter import IndexedDoc, RagAdapter, SourceDoc


def _load_sb_lib():
    """Localiza e importa sb_vector_lib (override por env SB_SCRIPTS)."""
    scripts = Path(os.environ.get("SB_SCRIPTS") or (Path.home() / ".claude" / "scripts"))
    if not (scripts / "sb_vector_lib.py").exists():
        raise SystemExit(
            f"sb_vector_lib.py nao encontrado em {scripts}. "
            "Aponte SB_SCRIPTS pra pasta de scripts do SB."
        )
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    import sb_vector_lib as lib

    return lib


class SBAdapter(RagAdapter):
    name = "second-brain"

    def __init__(self):
        self.lib = _load_sb_lib()

    def source_docs(self) -> list[SourceDoc]:
        return [
            SourceDoc(self.lib.relpath(f), f.stat().st_mtime)
            for f in self.lib.all_md_files()
        ]

    def indexed_docs(self) -> list[IndexedDoc]:
        # doc_meta e tabela comum -> sqlite3 puro, sem carregar a extensao vec
        con = sqlite3.connect(str(self.lib.DB_PATH))
        try:
            rows = con.execute(
                "SELECT path, mtime, n_chunks, total_tokens FROM doc_meta"
            ).fetchall()
        finally:
            con.close()
        return [IndexedDoc(p, m, n, t) for p, m, n, t in rows]

    def search(self, query: str, k: int = 5) -> list[str]:
        con = self.lib.connect_db()
        try:
            vo = self.lib.get_client()
            results, _ = self.lib.hybrid_search(con, vo, query, k=k * 3)
        finally:
            con.close()
        # colapsa chunks -> doc_ids unicos preservando ordem de ranking
        seen: list[str] = []
        for r in results:
            if r["path"] not in seen:
                seen.append(r["path"])
        return seen[:k]

    def search_scored(self, query: str, fetch: int = 30) -> dict:
        """Candidatos a nivel de doc com os componentes crus (vec/lex/boost).

        Base do tuning ao vivo: embeda a query 1x, devolve os componentes; a tela
        recombina com pesos diferentes SEM chamar a API de novo (via `rerank`).
        """
        con = self.lib.connect_db()
        try:
            vo = self.lib.get_client()
            results, _ = self.lib.hybrid_search(con, vo, query, k=fetch, fetch=fetch)
        finally:
            con.close()
        best: dict = {}
        for r in results:
            p = r["path"]
            if p not in best or r["score"] > best[p]["score"]:
                best[p] = {"vec": r["vec"], "lex": r["lex"], "boost": r["boost"], "score": r["score"]}
        return best


def rerank(candidates: dict, w_vec: float = 0.6, w_lex: float = 0.3, w_boost: float = 0.1) -> list[str]:
    """Recombina componentes com pesos custom e devolve doc_ids ordenados."""
    scored = [
        (p, w_vec * c["vec"] + w_lex * c["lex"] + w_boost * c["boost"])
        for p, c in candidates.items()
    ]
    scored.sort(key=lambda x: -x[1])
    return [p for p, _ in scored]
