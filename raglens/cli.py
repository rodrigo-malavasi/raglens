"""CLI do RagLens.  Rodar da raiz do repo:  python -m raglens health

Comandos:
  health   saude do indice (cobertura/staleness/orfaos) — Fase 1, sem API
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Windows: console default cp1252 quebra nos emojis/acentos. Forca UTF-8 (idem sb-search.py).
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

# Garante que a raiz do repo esteja no path pra achar o pacote `adapters/`,
# independente de onde o comando foi disparado.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from raglens.health import check_health  # noqa: E402
from raglens.report import format_health_text  # noqa: E402


def _get_adapter(name: str):
    if name in ("sb", "second-brain"):
        from adapters.sb_adapter import SBAdapter

        return SBAdapter()
    raise SystemExit(f"adapter desconhecido: '{name}' (disponiveis: sb)")


def _cmd_health(args) -> int:
    rep = check_health(_get_adapter(args.adapter))
    if args.json:
        print(json.dumps(rep.to_dict(), ensure_ascii=False, indent=2))
    else:
        print(format_health_text(rep))
    return 0 if rep.is_healthy else 2


def _cmd_eval(args) -> int:
    from raglens.eval import evaluate, load_golden

    adapter = _get_adapter(args.adapter)
    golden = load_golden(args.golden)
    res = evaluate(golden, lambda q: adapter.search(q, k=10))
    if args.json:
        print(json.dumps(res, ensure_ascii=False, indent=2))
        return 0
    agg = res["aggregate"]
    print(f"RagLens · eval · adapter='{adapter.name}' · {agg['n']} perguntas\n")
    print(f"  hit@1    {agg['hit@1']:.1%}")
    print(f"  hit@5    {agg['hit@5']:.1%}")
    print(f"  MRR      {agg['mrr']:.3f}")
    print(f"  recall@5 {agg['recall@5']:.1%}\n")
    misses = [c for c in res["cases"] if not c["hit5"]]
    if misses:
        print(f"  {len(misses)} falhas (esperado nao apareceu no top-5):")
        for c in misses:
            print(f"    ✗ {c['query']}  →  esperava {c['expected'][0]}")
    return 0


def _cmd_serve(args) -> int:
    from raglens.serve import serve

    serve(port=args.port)
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="raglens", description="Dashboard de qualidade de RAG")
    sub = p.add_subparsers(dest="cmd", required=True)

    h = sub.add_parser("health", help="Saude do indice: cobertura, staleness, orfaos")
    h.add_argument("--adapter", default="sb", help="RAG a medir (default: sb)")
    h.add_argument("--json", action="store_true", help="Saida JSON em vez de texto")
    h.set_defaults(fn=_cmd_health)

    _default_golden = str(
        Path(os.environ.get("SB_ROOT") or r"C:/Second Brain")
        / "_META" / "AVALIACAO" / "rag" / "golden.jsonl"
    )
    e = sub.add_parser("eval", help="Qualidade de retrieval contra o gabarito")
    e.add_argument("--adapter", default="sb")
    e.add_argument("--golden", default=_default_golden, help="Path do golden.jsonl")
    e.add_argument("--json", action="store_true")
    e.set_defaults(fn=_cmd_eval)

    s = sub.add_parser("serve", help="Sobe a tela (dashboard web)")
    s.add_argument("--port", type=int, default=7878)
    s.set_defaults(fn=_cmd_serve)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
