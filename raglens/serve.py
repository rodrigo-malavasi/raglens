"""Servidor da tela — Fase 4. stdlib + numpy (só pra projeção). Sem framework.

Serve o dashboard e a API. Tuning ao vivo reusa cache de candidatos por query
(mexer peso nao re-embeda). Projeção 2D dos embeddings via PCA (cache no processo).

Rodar:  python -m raglens serve   (http://localhost:7878)
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from raglens.eval import evaluate, load_golden
from raglens.health import check_health

_WEB = Path(__file__).resolve().parent / "web"


def _vault_root() -> Path:
    return Path(os.environ.get("SB_ROOT") or r"C:/Second Brain")


def _golden_path() -> Path:
    return _vault_root() / "_META" / "AVALIACAO" / "rag" / "golden.jsonl"


def _snap_dir() -> Path:
    d = _vault_root() / "_META" / "AVALIACAO" / "rag" / "snapshots"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _top_folder(rel: str) -> str:
    seg = rel.split("/")[0]
    return seg if seg else "(raiz)"


class _State:
    def __init__(self):
        self._adapter = None
        self.cand_cache: dict = {}
        self.projection: dict | None = None

    @property
    def adapter(self):
        if self._adapter is None:
            from adapters.sb_adapter import SBAdapter

            self._adapter = SBAdapter()
        return self._adapter

    def candidates(self, query: str) -> dict:
        if query not in self.cand_cache:
            self.cand_cache[query] = self.adapter.search_scored(query, fetch=30)
        return self.cand_cache[query]


STATE = _State()


# ---------- eval / tuning ----------
def _run_eval(wv: float, wl: float, wb: float) -> dict:
    from adapters.sb_adapter import rerank

    golden = load_golden(_golden_path())
    res = evaluate(golden, lambda q: rerank(STATE.candidates(q), wv, wl, wb))
    res["weights"] = {"vec": wv, "lex": wl, "boost": wb}
    return res


def _sweep(steps: int = 11) -> dict:
    from adapters.sb_adapter import rerank

    golden = load_golden(_golden_path())
    for g in golden:
        STATE.candidates(g["query"])  # aquece cache

    def metric_at(wv, wl, wb):
        h5 = mrr = 0
        for g in golden:
            ranked = rerank(STATE.candidates(g["query"]), wv, wl, wb)[:10]
            exp = set(g["expected_paths"])
            if any(d in exp for d in ranked[:5]):
                h5 += 1
            for i, d in enumerate(ranked, 1):
                if d in exp:
                    mrr += 1 / i
                    break
        n = len(golden)
        return h5 / n, mrr / n

    xs = [round(i / (steps - 1), 3) for i in range(steps)]
    fixed = {"vec": (0.3, 0.1), "lex": (0.6, 0.1), "boost": (0.6, 0.3)}
    out = {}
    for axis, (a, b) in fixed.items():
        curve = []
        for w in xs:
            if axis == "vec":
                h5, mrr = metric_at(w, a, b)
            elif axis == "lex":
                h5, mrr = metric_at(a, w, b)
            else:
                h5, mrr = metric_at(a, b, w)
            curve.append({"w": w, "hit5": h5, "mrr": mrr})
        out[axis] = curve
    return {"sweep": out}


def _save_snapshot(res: dict) -> str:
    now = datetime.now()
    ts = now.strftime("%Y-%m-%dT%H-%M-%S")
    fname = f"{ts}-{now.microsecond:06d}"
    snap = {"ts": ts, "weights": res["weights"], "aggregate": res["aggregate"]}
    (_snap_dir() / f"{fname}.json").write_text(
        json.dumps(snap, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return fname


def _trend() -> list[dict]:
    out = []
    for f in sorted(_snap_dir().glob("*.json")):
        try:
            s = json.loads(f.read_text(encoding="utf-8"))
            a = s["aggregate"]
            out.append({
                "ts": s["ts"],
                "hit1": a.get("hit@1", 0),
                "hit5": a.get("hit@5", 0),
                "mrr": a.get("mrr", 0),
                "weights": s.get("weights", {}),
            })
        except (json.JSONDecodeError, KeyError):
            continue
    return out


# ---------- projeção / cobertura ----------
def _projection() -> dict:
    if STATE.projection is not None:
        return STATE.projection
    import numpy as np

    lib = STATE.adapter.lib
    con = lib.connect_db()
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
    # normaliza pra caixa [-1,1]
    P = P / (np.abs(P).max(0) + 1e-9)
    groups = sorted({_top_folder(p) for p in paths})
    gi = {g: i for i, g in enumerate(groups)}
    pts = [[round(float(P[k, 0]), 3), round(float(P[k, 1]), 3), gi[_top_folder(paths[k])], paths[k].split("/")[-1]]
           for k in range(len(paths))]
    var2 = float((S[:2] ** 2).sum() / (S ** 2).sum())
    STATE.projection = {"points": pts, "groups": groups, "var": round(var2, 3)}
    return STATE.projection


def _coverage_folders() -> dict:
    ad = STATE.adapter
    src, idx = {}, {}
    idx_ids = {d.doc_id for d in ad.indexed_docs()}
    for d in ad.source_docs():
        f = _top_folder(d.doc_id)
        src[f] = src.get(f, 0) + 1
        if d.doc_id in idx_ids:
            idx[f] = idx.get(f, 0) + 1
    rows = [{"folder": f, "source": src[f], "indexed": idx.get(f, 0)} for f in sorted(src, key=lambda x: -src[x])]
    return {"folders": rows}


def _suggest(rel: str) -> dict:
    f = _vault_root() / rel
    if not f.exists():
        return {"doc": rel, "suggestions": [], "error": "doc nao encontrado"}
    heads = re.findall(r"^##\s+(.+)$", f.read_text(encoding="utf-8"), flags=re.MULTILINE)
    seen, sugg = set(), []
    for h in heads:
        h = re.sub(r"[*`#>|]", "", h).strip().lower()
        h = re.sub(r"\s*[—\-].*$", "", h)
        h = re.sub(r"^\d+[.)]\s*", "", h)
        if 3 < len(h) < 60 and h not in seen:
            seen.add(h)
            sugg.append(f"como funciona {h}?")
    return {"doc": rel, "expected_paths": [rel], "suggestions": sugg[:10]}


def _write_golden(items: list[dict]) -> None:
    with open(_golden_path(), "w", encoding="utf-8") as fh:
        for it in items:
            fh.write(json.dumps({
                "query": it["query"],
                "expected_paths": it.get("expected_paths", []),
                "tipo": it.get("tipo", "manual"),
            }, ensure_ascii=False) + "\n")


def _append_golden(item: dict) -> None:
    items = load_golden(_golden_path())
    items.append(item)
    _write_golden(items)


def _edit_golden(index: int, item: dict) -> None:
    items = load_golden(_golden_path())
    if not (0 <= index < len(items)):
        raise IndexError("indice fora do gabarito")
    items[index] = item
    _write_golden(items)


def _delete_golden(index: int) -> None:
    items = load_golden(_golden_path())
    if not (0 <= index < len(items)):
        raise IndexError("indice fora do gabarito")
    del items[index]
    _write_golden(items)


def _generate(n: int = 8) -> dict:
    import random

    idx_ids = {d.doc_id for d in STATE.adapter.indexed_docs()}
    existing = {g["query"] for g in load_golden(_golden_path())}
    pool = [d.doc_id for d in STATE.adapter.source_docs()
            if d.doc_id in idx_ids and d.doc_id.endswith(".md")]
    random.shuffle(pool)
    props = []
    for rel in pool:
        s = _suggest(rel)
        for q in s.get("suggestions", []):
            if q not in existing:
                props.append({"query": q, "expected_paths": [rel], "tipo": "gerada"})
                break
        if len(props) >= n:
            break
    return {"proposals": props}


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

    def log_message(self, *a):
        pass

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        try:
            if u.path in ("/", "/index.html"):
                html = (_WEB / "dashboard.html").read_text(encoding="utf-8").replace("__RAGLENS_DATA__", "null")
                return self._send(200, html.encode("utf-8"), "text/html; charset=utf-8")
            if u.path == "/api/health":
                return self._json(check_health(STATE.adapter).to_dict())
            if u.path == "/api/golden":
                return self._json({"golden": load_golden(_golden_path())})
            if u.path == "/api/eval":
                wv = float(q.get("wv", ["0.6"])[0]); wl = float(q.get("wl", ["0.3"])[0]); wb = float(q.get("wb", ["0.1"])[0])
                res = _run_eval(wv, wl, wb)
                if q.get("save", ["0"])[0] == "1":
                    res["saved"] = _save_snapshot(res)
                return self._json(res)
            if u.path == "/api/sweep":
                return self._json(_sweep())
            if u.path == "/api/trend":
                return self._json({"trend": _trend()})
            if u.path == "/api/projection":
                return self._json(_projection())
            if u.path == "/api/coverage_folders":
                return self._json(_coverage_folders())
            if u.path == "/api/suggest":
                return self._json(_suggest(q.get("path", [""])[0]))
            if u.path == "/api/generate":
                return self._json(_generate(int(q.get("n", ["8"])[0])))
            return self._json({"error": "rota nao encontrada"}, 404)
        except Exception as e:  # noqa: BLE001
            return self._json({"error": str(e)}, 500)

    def _body(self):
        n = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(n) or b"{}")

    def do_POST(self):
        u = urlparse(self.path)
        try:
            body = self._body()
            if u.path == "/api/golden":
                if not body.get("query") or not body.get("expected_paths"):
                    return self._json({"error": "query e expected_paths obrigatorios"}, 400)
                _append_golden(body)
                return self._json({"ok": True})
            return self._json({"error": "rota nao encontrada"}, 404)
        except Exception as e:  # noqa: BLE001
            return self._json({"error": str(e)}, 500)

    def do_PUT(self):
        u = urlparse(self.path)
        try:
            body = self._body()
            if u.path == "/api/golden":
                _edit_golden(int(body["index"]), body)
                return self._json({"ok": True})
            return self._json({"error": "rota nao encontrada"}, 404)
        except Exception as e:  # noqa: BLE001
            return self._json({"error": str(e)}, 500)

    def do_DELETE(self):
        u = urlparse(self.path)
        try:
            body = self._body()
            if u.path == "/api/golden":
                _delete_golden(int(body["index"]))
                return self._json({"ok": True})
            return self._json({"error": "rota nao encontrada"}, 404)
        except Exception as e:  # noqa: BLE001
            return self._json({"error": str(e)}, 500)


def serve(host: str = "127.0.0.1", port: int = 7878) -> None:
    httpd = ThreadingHTTPServer((host, port), Handler)
    print(f"RagLens rodando em http://{host}:{port}  (Ctrl+C pra parar)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nencerrado.")
        httpd.shutdown()
