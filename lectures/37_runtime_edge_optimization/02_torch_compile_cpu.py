"""02: torch.compile を CPU で — Inductor の仕組み・初回コスト・グラフブレイク・環境ガード。

torch.compile は PyTorch 2 の目玉。中身は2段:
  - TorchDynamo : Python バイトコードをフックして計算グラフを切り出す(捕捉)。
  - Inductor    : 切り出したグラフを最適化し、CPU では C++/OpenMP カーネルを*生成*して
                  コンパイルする(GPU では Triton カーネル)。カーネル融合でメモリ往復を減らす。

CPU でも効くが、現実には次の注意がある:
  - 初回呼び出しでコンパイルが走るため*非常に遅い*(数秒〜)。2回目以降が速い。
    → ベンチでは必ずウォームアップ後に測る(初回を混ぜると誤判定)。
  - C++ ツールチェイン(g++ と python3-dev の Python.h)が無いと*初回でコンパイル失敗*。
    本スクリプトはその場合でも落ちないよう try/except でガードし、概念だけ示す。
  - グラフブレイク(未対応構文・データ依存分岐)や dynamic shape で再コンパイルが多発し、
    期待した高速化が出ないことがある。fullgraph=True でブレイクを検出できる。

実行:
  uv run python lectures/37_runtime_edge_optimization/02_torch_compile_cpu.py
結果は lectures/37_runtime_edge_optimization/outputs/ に保存される(コンパイル不可環境では概念のみ)。
"""

from __future__ import annotations

import pathlib
import sys
import time
import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from bench_lab import (  # noqa: E402
    apply_bench_threads,
    ensure_output_dir,
    make_resnet18,
    measure_latency,
    synthetic_batch,
)


def explain_modes() -> None:
    """torch.compile の mode と引数の意味を一覧で示す(暗記より使い分けの地図)。"""
    print("[1] torch.compile の主な mode / 引数:")
    rows = [
        ("mode='default'", "標準。コンパイル時間と速度のバランス。まずこれ。"),
        ("mode='reduce-overhead'", "起動オーバーヘッド削減(CUDA Graph 寄り)。小バッチ反復向け。"),
        ("mode='max-autotune'", "カーネルを探索し最速を選ぶ。コンパイルは長い。本番固定形向け。"),
        ("fullgraph=True", "グラフブレイクを許さず例外化。ブレイク箇所の発見に使う。"),
        ("dynamic=False", "形状を固定とみなす。可変長で再コンパイル多発するなら True を検討。"),
    ]
    for k, v in rows:
        print(f"    {k:24s} {v}")


def try_compile_bench(out: pathlib.Path) -> bool:
    """torch.compile を実測する。コンパイル不可環境なら概念に切り替えて False を返す。"""
    apply_bench_threads()
    model = make_resnet18()
    x1 = synthetic_batch(1, seed=11)
    example = torch.from_numpy(x1)

    print("\n[2] torch.compile を試す(初回はコンパイルで遅い)...")
    try:
        compiled = torch.compile(model)  # mode 既定
        t0 = time.perf_counter()
        with torch.inference_mode():
            compiled(example)  # ★ ここで初回コンパイルが発火(失敗もここで出る)
        compile_sec = time.perf_counter() - t0
        print(f"    初回コンパイル+実行: {compile_sec:.2f} 秒(2回目以降は速い)")
    except Exception as e:  # noqa: BLE001
        print(f"    [利用不可] {type(e).__name__}: {str(e)[:100]}")
        print("    → この環境は C++ ツールチェイン(g++ / python3-dev)が無く Inductor が動かない。")
        print("       torch.compile は『使えれば効く』が必須にしない。概念を理解しておけば OK。")
        return False

    # 定常状態のレイテンシを eager と比較
    def eager_infer(x: np.ndarray) -> np.ndarray:
        with torch.inference_mode():
            return model(torch.from_numpy(x)).numpy()

    def compiled_infer(x: np.ndarray) -> np.ndarray:
        with torch.inference_mode():
            return compiled(torch.from_numpy(x)).numpy()

    eager_lat = measure_latency(eager_infer, x1)
    comp_lat = measure_latency(compiled_infer, x1)
    print("\n[3] レイテンシ比較(ウォームアップ後 p50):")
    print(f"    eager        p50={eager_lat.p50_ms:7.2f} ms")
    print(f"    torch.compile p50={comp_lat.p50_ms:7.2f} ms  ({eager_lat.p50_ms / comp_lat.p50_ms:.2f}x)")

    # 数値一致の確認(高速化しても出力が変わってはいけない)
    with torch.inference_mode():
        diff = (compiled(example) - model(example)).abs().max().item()
    print(f"    eager との最大絶対誤差: {diff:.2e}(微小なら OK)")
    return True


def explain_graph_break() -> None:
    """グラフブレイクの概念と、fullgraph=True での検出方法を文章で示す。"""
    print("\n[4] グラフブレイク(graph break)とは:")
    print("    Dynamo が捕捉できない箇所(print, 一部の Python 構文, データ依存分岐 等)で")
    print("    グラフが分断される現象。分断ごとに Python に戻るため最適化効果が削れる。")
    print("    - 発見: torch.compile(model, fullgraph=True) でブレイク時に例外を出させる。")
    print("    - 再コンパイル: 形状が変わるたび再コンパイルが走ると逆に遅い。")
    print("      固定形なら dynamic=False、可変なら dynamic=True で挙動を制御する。")
    print("    - 計測: 初回(コンパイル) と 定常(2回目以降) を分けて測るのが鉄則。")


def main() -> None:
    out = ensure_output_dir()
    explain_modes()
    ok = try_compile_bench(out)
    explain_graph_break()

    print("\n[まとめ]")
    if ok:
        print("  - torch.compile は CPU でも Inductor が C++ カーネルを生成して効くことがある")
    else:
        print("  - この環境では Inductor が動かないが、概念(捕捉→生成→融合)は他環境で同じ")
    print("  - 必ず『初回コンパイルコスト』と『定常レイテンシ』を分けて評価する")
    print("  - グラフブレイク/再コンパイルに注意。fullgraph=True で隠れたブレイクを炙り出す")


if __name__ == "__main__":
    main()
