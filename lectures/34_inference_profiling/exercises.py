"""第34回 演習問題（推論プロファイリング / 正しいベンチマーク）。

使い方:
  1. 各 exN_*() の中の TODO を自分で実装する（最初は NotImplementedError で FAIL）。
  2. 自己採点（未実装でも例外で落とさず PASS/FAIL を表示し、必ず exit 0）:
         uv run python lectures/34_inference_profiling/exercises.py
  3. 全問の模範解答（全 PASS）はこちら:
         uv run python lectures/34_inference_profiling/exercises_solutions.py

テーマ（易→難）:
  1 分位点(p50/p99)   2 スループット   3 ウォームアップ除去   4 speedup
  5 レイテンシ要約     6 ベンチの中身(ウォームアップ→反復→中央値)
  7 律速演算子トップk  8 inference_mode で勾配が切れるか   9 公平比較表の作成

採点は決定的な合成データ / 極小ネットだけで行うので、モデル DL 不要で速く再現的。
本物の resnet18 / profiler 連携は 01〜05・mini_project.py を参照。
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch
import torch.nn as nn


# =====================================================================
# 演習（ここを実装する）
# =====================================================================

def ex1_percentile(values: list[float], q: float) -> float:
    """演習1: レイテンシ列 values の q パーセンタイル（0〜100）を返す。

    平均ではなく分位点で語るのがベンチの作法。np.percentile（線形補間）と一致させること。
    """
    # TODO: float(np.percentile(np.asarray(values, dtype=float), q)) を返す
    raise NotImplementedError


def ex2_throughput(latency_s: float, batch_size: int) -> float:
    """演習2: 1反復のレイテンシ（秒）とバッチ枚数から スループット(img/s) を返す。

    throughput = batch_size / latency_s。レイテンシとスループットは別物（逆数関係＋バッチ）。
    """
    # TODO: batch_size / latency_s を返す（latency_s>0 前提）
    raise NotImplementedError


def ex3_drop_warmup(times: list[float], n_warmup: int) -> list[float]:
    """演習3: 計測列の先頭 n_warmup 個（ウォームアップ）を捨てた残りを返す。

    初回は遅延ロード/キャッシュ miss で遅い。捨ててから集計するのが正しい手順。
    """
    # TODO: times[n_warmup:] を返す
    raise NotImplementedError


def ex4_speedup(baseline_ms: float, candidate_ms: float) -> float:
    """演習4: speedup = baseline / candidate を返す（>1 で速くなった）。"""
    # TODO: baseline_ms / candidate_ms を返す
    raise NotImplementedError


def ex5_summarize(latencies_ms: list[float]) -> dict[str, float]:
    """演習5: レイテンシ列から {'p50','p99','mean'} を作って返す。

    p50/p99 は np.percentile、mean は np.mean。すべて float で。
    """
    # TODO: {"p50":..., "p99":..., "mean":...} を返す
    raise NotImplementedError


def ex6_run_benchmark(fn: Callable[[], object], n_warmup: int, n_iter: int) -> tuple[float, int]:
    """演習6: 「ウォームアップ→反復計測→中央値」を行い (中央値ms, 計測した反復数) を返す。

    手順:
      - fn を n_warmup 回呼ぶ（計測しない＝ウォームアップ）。
      - 続けて n_iter 回、各回 time.perf_counter で挟んで経過(ms)を記録。
      - 記録列の中央値(ms, np.median) と、計測した反復数 n_iter を返す。
    （fn は引数なしで1回の処理を行うクロージャ。GPU同期は今回は考えない＝CPU前提。）
    """
    # TODO: import time; ウォームアップ→計測ループ→ (float(np.median(rec)), n_iter)
    raise NotImplementedError


def ex7_top_ops(records: list[tuple[str, float]], k: int) -> list[tuple[str, float]]:
    """演習7: (演算子名, 自己CPU時間) のリストを“時間の降順”に並べ上位 k 件を返す。

    profiler の key_averages から律速演算子を取り出す処理の核。タイは入力順を保てなくてよい。
    """
    # TODO: time 降順 sort して [:k]
    raise NotImplementedError


def ex8_no_grad_output(model: nn.Module, x: torch.Tensor) -> bool:
    """演習8: model(x) を inference_mode 下で実行し、出力が勾配を追跡していないか（True/False）を返す。

    推論で勾配を切ると速く・省メモリになる。inference_mode 下の出力は requires_grad=False。
    返り値: 「勾配グラフを作っていない」= (not out.requires_grad) を bool で返す。
    """
    # TODO: with torch.inference_mode(): out = model(x); return bool(not out.requires_grad)
    raise NotImplementedError


def ex9_comparison_table(results: dict[str, float], baseline: str) -> list[dict[str, object]]:
    """演習9: 条件名→p50(ms) の dict から、speedup 付きの比較行リストを作る。

    各行は {"label": 名前, "p50_ms": 値, "speedup": baseline_p50/値}。
    リストは p50 の昇順（速い順）に並べる。baseline 自身の speedup は 1.0 になる。
    """
    # TODO: baseline の p50 を取り、各条件で speedup を計算し、p50 昇順に並べた行リストを返す
    raise NotImplementedError


# =====================================================================
# 自己採点ランナー（exercises_solutions.py からも import して再利用）
# =====================================================================

class _TinyNet(nn.Module):
    """ex8 用の極小ネット（DL 不要・決定的）。"""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(8, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def grade(funcs: dict) -> bool:
    """funcs（名前→関数）を採点して PASS/FAIL を表示。全PASSなら True。"""
    results: list[tuple[str, bool, str]] = []

    def check(name: str, fn) -> None:
        try:
            ok, detail = fn()
            results.append((name, bool(ok), detail))
        except NotImplementedError:
            results.append((name, False, "未実装（TODOを埋めてください）"))
        except Exception as e:  # noqa: BLE001
            results.append((name, False, f"例外: {type(e).__name__}: {e}"))

    def c1():
        vals = [1.0, 2.0, 3.0, 4.0, 100.0]
        p50 = funcs["ex1"](vals, 50)
        p99 = funcs["ex1"](vals, 99)
        ok = abs(p50 - 3.0) < 1e-9 and abs(p99 - float(np.percentile(vals, 99))) < 1e-9
        return ok, f"p50={p50:.3f}, p99={p99:.3f}（外れ値100に p50 は頑健）"

    def c2():
        tp = funcs["ex2"](0.05, 8)  # 50ms で 8枚 → 160 img/s
        return abs(tp - 160.0) < 1e-6, f"throughput={tp:.1f} img/s"

    def c3():
        out = funcs["ex3"]([9.0, 9.0, 1.0, 1.1, 1.0], 2)
        return out == [1.0, 1.1, 1.0], f"warmup除去後={out}"

    def c4():
        s = funcs["ex4"](40.0, 10.0)
        return abs(s - 4.0) < 1e-9, f"speedup={s:.2f}x"

    def c5():
        d = funcs["ex5"]([1.0, 2.0, 3.0, 4.0])
        ok = abs(d["p50"] - 2.5) < 1e-9 and abs(d["mean"] - 2.5) < 1e-9 and "p99" in d
        return ok, f"p50={d['p50']:.2f}, p99={d['p99']:.2f}, mean={d['mean']:.2f}"

    def c6():
        calls = {"n": 0}

        def fake() -> object:
            calls["n"] += 1
            _ = sum(i * i for i in range(2000))  # ごく軽い決定的処理
            return None

        med, n = funcs["ex6"](fake, n_warmup=3, n_iter=10)
        ok = calls["n"] == 13 and n == 10 and med >= 0.0
        return ok, f"総呼び出し={calls['n']}（=warmup3+iter10）, median={med:.4f}ms"

    def c7():
        recs = [("conv", 5.0), ("bn", 1.0), ("relu", 0.5), ("matmul", 3.0)]
        top = funcs["ex7"](recs, 2)
        return top == [("conv", 5.0), ("matmul", 3.0)], f"top2={top}"

    def c8():
        torch.manual_seed(0)
        net = _TinyNet().eval()
        x = torch.randn(2, 8)
        no_grad = funcs["ex8"](net, x)
        return no_grad is True, f"inference_mode下で勾配なし={no_grad}"

    def c9():
        rows = funcs["ex9"]({"A": 40.0, "B": 20.0, "C": 10.0}, baseline="A")
        labels = [r["label"] for r in rows]
        a = next(r for r in rows if r["label"] == "A")
        c = next(r for r in rows if r["label"] == "C")
        ok = labels == ["C", "B", "A"] and abs(a["speedup"] - 1.0) < 1e-9 and abs(c["speedup"] - 4.0) < 1e-9
        return ok, f"速い順={labels}, C.speedup={c['speedup']:.1f}x"

    check("ex1_percentile", c1)
    check("ex2_throughput", c2)
    check("ex3_drop_warmup", c3)
    check("ex4_speedup", c4)
    check("ex5_summarize", c5)
    check("ex6_run_benchmark", c6)
    check("ex7_top_ops", c7)
    check("ex8_no_grad_output", c8)
    check("ex9_comparison_table", c9)

    print("=== 採点結果 ===")
    all_ok = True
    for name, ok, detail in results:
        mark = "PASS" if ok else "FAIL"
        all_ok = all_ok and ok
        print(f"  [{mark}] {name:24s} {detail}")
    print("\nALL PASS 🎉" if all_ok else "\nまだ未達の演習があります。TODO を埋めましょう。")
    return all_ok


def current_funcs() -> dict:
    """このファイルで定義された ex* を採点ランナーへ渡す形にまとめる。"""
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
    grade(current_funcs())
