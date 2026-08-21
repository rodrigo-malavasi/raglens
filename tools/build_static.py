"""Gera a versao ESTATICA do dashboard — um unico .html autocontido, sem servidor.

Por que existe: publicar a tela (GitHub Pages, anexo de email) sem expor `serve.py`,
que e um ThreadingHTTPServer de stdlib com rotas de escrita e sem auth. Aqui nao ha
servidor: os numeros ja vao embutidos no HTML.

O que embute: metrica agregada, saude do indice, projecao 2D e — a parte que faz o
tuning ao vivo continuar funcionando offline — os componentes crus (vec/lex/boost)
de cada candidato por query. `localEval()` no dashboard.html recombina esses
componentes com pesos novos, entao mexer nos sliders na pagina estatica recalcula
hit@k/MRR de verdade, sem chamar a API de embedding.

O que NAO embute: texto de documento. Sai path e numero, nada mais — checavel na
lista que `--print-paths` imprime.

Uso:
    SB_ROOT=<vault> python tools/build_static.py --out docs/index.html
    SB_ROOT=<vault> python tools/build_static.py --print-paths   # so a auditoria
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from raglens.eval import evaluate, load_golden  # noqa: E402
from raglens.health import check_health  # noqa: E402

_WEB = _ROOT / "raglens" / "web" / "dashboard.html"

# Pesos default do ranker — mesmos de `rerank()` em adapters/sb_adapter.py.
BASE_W = (0.6, 0.3, 0.1)

# Trilha de experimentos que vira o grafico de tendencia. Sao rodadas REAIS: cada
# tupla e reavaliada contra o golden com os componentes ja em cache.
TREND_W = [
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.7, 0.2, 0.1),
    (0.6, 0.3, 0.1),
    (0.5, 0.4, 0.1),
]


def _vault_root() -> Path:
    return Path(os.environ.get("SB_ROOT") or r"C:/Second Brain")


def _golden_path() -> Path:
    return _vault_root() / "_META" / "AVALIACAO" / "rag" / "golden.jsonl"


def _top_folder(rel: str) -> str:
    seg = rel.split("/")[0]
    return seg if seg else "(raiz)"


def _projection(adapter) -> dict:
    import numpy as np

    con = adapter.lib.connect_db()
    try:
        rows = con.execute(
            "SELECT c.path, v.embedding FROM vec_chunks v JOIN chunks c ON c.id = v.id"
        ).fetchall()
    finally:
        con.close()
    paths = [r[0] for r in rows]
    M = np.vstack([np.frombuffer(r[1], dtype=np.float32) for r in rows])
    Mc = M - M.mean(0)
    _, S, Vt = np.linalg.svd(Mc, full_matrices=False)
    P = Mc @ Vt[:2].T
    P = P / (np.abs(P).max(0) + 1e-9)
    groups = sorted({_top_folder(p) for p in paths})
    gi = {g: i for i, g in enumerate(groups)}
    pts = [
        [round(float(P[k, 0]), 3), round(float(P[k, 1]), 3), gi[_top_folder(paths[k])],
         paths[k].split("/")[-1]]
        for k in range(len(paths))
    ]
    var2 = float((S[:2] ** 2).sum() / (S ** 2).sum())
    return {"points": pts, "groups": groups, "var": round(var2, 3)}


def _coverage_folders(adapter) -> dict:
    src, idx = {}, {}
    idx_ids = {d.doc_id for d in adapter.indexed_docs()}
    for d in adapter.source_docs():
        f = _top_folder(d.doc_id)
        src[f] = src.get(f, 0) + 1
        if d.doc_id in idx_ids:
            idx[f] = idx.get(f, 0) + 1
    rows = [
        {"folder": f, "source": src[f], "indexed": idx.get(f, 0)}
        for f in sorted(src, key=lambda x: -src[x])
    ]
    return {"folders": rows}


def build_data(adapter) -> dict:
    from adapters.sb_adapter import rerank

    golden = load_golden(_golden_path())

    # Um embed de query por caso; dai em diante tudo recombina em memoria.
    cands = {g["query"]: adapter.search_scored(g["query"], fetch=30) for g in golden}

    def eval_at(w):
        return evaluate(golden, lambda q: rerank(cands[q], *w))

    base = eval_at(BASE_W)["aggregate"]

    trend = []
    for w in TREND_W:
        a = eval_at(w)["aggregate"]
        trend.append({
            "ts": f"w={w[0]}/{w[1]}/{w[2]}",
            "hit1": a["hit@1"], "hit5": a["hit@5"], "mrr": a["mrr"],
            "weights": {"vec": w[0], "lex": w[1], "boost": w[2]},
        })

    golden_js = [
        {
            "query": g["query"],
            "expected": g["expected_paths"],
            "tipo": g.get("tipo", "canon"),
            # dict -> array [vec, lex, boost]: e a forma que localEval() consome.
            "cands": {
                p: [round(c["vec"], 5), round(c["lex"], 5), round(c["boost"], 5)]
                for p, c in cands[g["query"]].items()
            },
        }
        for g in golden
    ]

    return {
        "health": check_health(adapter).to_dict(),
        "projection": _projection(adapter),
        "coverage_folders": _coverage_folders(adapter),
        "golden": golden_js,
        "baseline": {
            "hit1": base["hit@1"], "hit5": base["hit@5"],
            "mrr": base["mrr"], "recall5": base["recall@5"],
        },
        "trend": trend,
    }


def collect_paths(data: dict) -> list[str]:
    """Todo path/nome de arquivo que sai no HTML. E a lista de auditoria."""
    out = set()
    h = data["health"]
    for key in ("unindexed", "stale", "orphan"):
        out.update(h.get(key, []))
    for g in data["golden"]:
        out.update(g["expected"])
        out.update(g["cands"].keys())
    # projecao guarda so basename
    out.update(p[3] for p in data["projection"]["points"])
    out.update(f["folder"] for f in data["coverage_folders"]["folders"])
    return sorted(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(_ROOT / "docs" / "index.html"))
    ap.add_argument("--print-paths", action="store_true",
                    help="Lista tudo que sai no HTML e sai sem gravar")
    ap.add_argument("--data-out", help="Grava tambem o JSON cru (auditoria)")
    args = ap.parse_args()

    from adapters.sb_adapter import SBAdapter

    data = build_data(SBAdapter())

    if args.print_paths:
        for p in collect_paths(data):
            print(p)
        return 0

    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    # `</script>` dentro de string JS fecha a tag cedo. Nao ocorre com paths, mas
    # gerar HTML sem essa guarda e como gerar SQL sem escapar aspas.
    payload = payload.replace("</", "<\\/")

    html = _WEB.read_text(encoding="utf-8").replace("__RAGLENS_DATA__", payload)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")

    if args.data_out:
        Path(args.data_out).write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    kb = out.stat().st_size / 1024
    print(f"{out}  ({kb:.0f} KB)")
    print(f"  golden: {len(data['golden'])} casos")
    print(f"  paths embutidos: {len(collect_paths(data))}")
    print(f"  hit@1 {data['baseline']['hit1']:.3f} · mrr {data['baseline']['mrr']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
