"""第34回 演習の模範解答（全 PASS）。

  uv run python lectures/34_inference_profiling/exercises_solutions.py

exercises.py の採点ランナー grade() をそのまま使い、各 ex* を埋めた版を渡して全 PASS を確認する。
解き方のポイントはコメントに残してある。まず自分で exercises.py を解いてから答え合わせに使うこと。
"""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np
import torch
import torch.nn as nn

from exercises import grade


def ex1_percentile(values: list[float], q: float) -> float:
    # np.percentile は線形補間。p50=中央値、p99=最悪寄り。外れ値に頑健なのが分位点の強み。
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def ex2_throughput(latency_s: float, batch_size: int) -> float:
    # スループット = 単位時間に何枚さばけるか。1反復で batch_size 枚を latency_s 秒で処理。
    return batch_size / latency_s


def ex3_drop_warmup(times: list[float], n_warmup: int) -> list[float]:
    # 先頭 n_warmup 個（初回の遅い計測）を捨てるだけ。
    return times[n_warmup:]


def ex4_speedup(baseline_ms: float, candidate_ms: float) -> float:
    return baseline_ms / candidate_ms


def ex5_summarize(latencies_ms: list[float]) -> dict[str, float]:
    arr = np.asarray(latencies_ms, dtype=np.float64)
    return {
        "p50": float(np.percentile(arr, 50)),
        "p99": float(np.percentile(arr, 99)),
        "mean": float(arr.mean()),
    }


def ex6_run_benchmark(fn: Callable[[], object], n_warmup: int, n_iter: int) -> tuple[float, int]:
    # 正しいベンチの最小形: ウォームアップ → 反復計測 → 中央値。
    for _ in range(n_warmup):
        fn()  # 計測しない（キャッシュ/遅延ロードを温める）
    rec: list[float] = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        fn()
        rec.append((time.perf_counter() - t0) * 1e3)  # ms
    return float(np.median(rec)), n_iter


def ex7_top_ops(records: list[tuple[str, float]], k: int) -> list[tuple[str, float]]:
    # 自己CPU時間の降順に並べて上位 k 件（律速演算子の特定）。
    return sorted(records, key=lambda kv: kv[1], reverse=True)[:k]


def ex8_no_grad_output(model: nn.Module, x: torch.Tensor) -> bool:
    # inference_mode 下では自動微分グラフを作らない → 出力 requires_grad=False。
    with torch.inference_mode():
        out = model(x)
    return bool(not out.requires_grad)


def ex9_comparison_table(results: dict[str, float], baseline: str) -> list[dict[str, object]]:
    base = results[baseline]
    rows = [{"label": k, "p50_ms": v, "speedup": base / v} for k, v in results.items()]
    rows.sort(key=lambda r: r["p50_ms"])  # 速い順（p50 昇順）
    return rows


def current_funcs() -> dict:
    return {
        "ex1": ex1_percentile,
        "ex2": ex2_throughput,
        "ex3": ex3_drop_warmup,
        "ex4": ex4_speedup,
        "ex5": ex5_summarize,
        "ex6": ex6_run_benchmark,
        "ex7": ex7_top_ops,
        "ex8": ex8_no_grad_output,
        "ex9": ex9_comparison_table,
    }


if __name__ == "__main__":
    ok = grade(current_funcs())
    if not ok:
        raise SystemExit(1)
