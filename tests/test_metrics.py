"""Testes das metricas puras. Roda standalone (sem pytest):  python tests/test_metrics.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from raglens.metrics import aggregate, hit_at_k, recall_at_k, reciprocal_rank


def test_hit_at_k():
    assert hit_at_k(["a", "b", "c"], {"c"}, 5) is True
    assert hit_at_k(["a", "b", "c"], {"c"}, 2) is False  # c esta na pos 3
    assert hit_at_k(["a", "b", "c"], {"a"}, 1) is True
    assert hit_at_k([], {"a"}, 5) is False


def test_reciprocal_rank():
    assert reciprocal_rank(["a", "b", "c"], {"a"}) == 1.0
    assert reciprocal_rank(["a", "b", "c"], {"b"}) == 0.5
    assert reciprocal_rank(["a", "b", "c"], {"z"}) == 0.0


def test_recall_at_k():
    assert recall_at_k(["a", "b", "c"], {"a", "b"}, 5) == 1.0
    assert recall_at_k(["a", "b", "c"], {"a", "z"}, 5) == 0.5
    assert recall_at_k(["a", "b", "c"], set(), 5) == 0.0


def test_aggregate():
    cases = [
        (["a", "b", "c"], {"a"}),  # hit@1 sim, rr 1.0
        (["x", "y", "z"], {"z"}),  # hit@1 nao, hit@5 sim, rr 1/3
    ]
    agg = aggregate(cases, ks=(1, 5))
    assert agg["n"] == 2
    assert agg["hit@1"] == 0.5
    assert agg["hit@5"] == 1.0
    assert abs(agg["mrr"] - (1.0 + 1 / 3) / 2) < 1e-9
    assert aggregate([], ks=(1, 5)) == {"n": 0}


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} testes passaram.")
