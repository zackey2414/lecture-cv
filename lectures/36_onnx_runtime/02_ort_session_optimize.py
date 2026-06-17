"""02: onnxruntime のセッション設定 — グラフ最適化レベルとスレッド数。

学ぶこと:
  - InferenceSession は SessionOptions で挙動を制御する。最重要は2つ:
      * graph_optimization_level … 定数畳み込み・冗長ノード除去・演算子融合（Conv+ReLU 等）を
        どこまでやるか。ORT_DISABLE_ALL（無効）〜ORT_ENABLE_ALL（全部）。
      * intra_op_num_threads … 1 演算の内部並列スレッド数。CPU の物理コアに合わせて調整する。
  - 「最適化レベルを上げれば必ず速い」わけではない（小モデルでは差が出にくい/逆転もある）。
    だから推測せず必ず計測する。これは前章までの『計測ファースト』原則の延長。
  - 公平なベンチのため torch 側も onnxruntime 側もスレッド数を揃える。

実行:
  uv run python lectures/36_onnx_runtime/02_ort_session_optimize.py
結果（図）は lectures/36_onnx_runtime/outputs/ に保存される。
"""

from __future__ import annotations

import pathlib
import sys
import warnings

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from onnx_helpers import (  # noqa: E402
    benchmark,
    ensure_output_dir,
    export_onnx,
    make_dataset,
    make_session,
    pin_threads,
    run_session,
    set_seed,
    train_tiny_cnn,
)


def _bench_session(sess, x_np: np.ndarray, batch: int) -> dict:
    """セッションを 1 つベンチして latency/throughput を返す。"""
    return benchmark(lambda: run_session(sess, x_np), batch_size=batch, warmup=10, iters=80)


def main() -> None:
    warnings.filterwarnings("ignore")
    set_seed(0)
    pin_threads(1)  # torch を 1 スレッドに固定（onnxruntime 側も 1 に揃える）
    out = ensure_output_dir()

    # === 1. モデルと入力を用意 ===========================================
    print("[1] TinyCNN を用意して ONNX 化")
    model = train_tiny_cnn(epochs=3, verbose=False)
    x, _ = make_dataset(n_per_class=16, seed=7)  # 48 枚バッチ
    batch = x.shape[0]
    x_np = x.numpy()
    onnx_path = export_onnx(model, x[:1], out / "tinycnn.onnx", dynamic_batch=True)

    # === 2. eager torch のベンチ（比較の基準） ===========================
    def torch_call() -> None:
        with torch.inference_mode():
            model(x)

    base = benchmark(torch_call, batch_size=batch, warmup=10, iters=80)
    print(f"\n[2] eager torch         : p50={base['p50_ms']:.3f}ms  "
          f"p99={base['p99_ms']:.3f}ms  {base['throughput']:.0f} img/s")

    # === 3. グラフ最適化レベルの比較（DISABLE vs ENABLE_ALL） ============
    sess_off = make_session(onnx_path, optimize=False, intra_threads=1)
    sess_on = make_session(onnx_path, optimize=True, intra_threads=1)
    r_off = _bench_session(sess_off, x_np, batch)
    r_on = _bench_session(sess_on, x_np, batch)
    print("\n[3] グラフ最適化レベルの比較（intra_op=1）")
    print(f"    ORT_DISABLE_ALL : p50={r_off['p50_ms']:.3f}ms  {r_off['throughput']:.0f} img/s")
    print(f"    ORT_ENABLE_ALL  : p50={r_on['p50_ms']:.3f}ms  {r_on['throughput']:.0f} img/s")
    speedup = r_off["p50_ms"] / max(r_on["p50_ms"], 1e-9)
    print(f"    → 最適化ありは {speedup:.2f}x（小モデルでは差が小さい/逆転もあり得る = 要計測）")

    # === 4. スレッド数スイープ ===========================================
    # intra_op_num_threads を変えて p50 がどう動くか観察。多ければ速いとは限らない
    # （スレッド起動コスト・小モデルでは 1〜2 が最速のことも多い）。
    print("\n[4] intra_op_num_threads のスイープ（ORT_ENABLE_ALL）")
    thread_results: dict[int, dict] = {}
    for t in (1, 2, 4):
        sess_t = make_session(onnx_path, optimize=True, intra_threads=t)
        thread_results[t] = _bench_session(sess_t, x_np, batch)
        print(f"    threads={t}: p50={thread_results[t]['p50_ms']:.3f}ms  "
              f"{thread_results[t]['throughput']:.0f} img/s")

    # === 5. 図に保存 =====================================================
    _save_chart(out, base, r_off, r_on, thread_results)
    print("\n[5] 図を保存:", (out / "02_session_optimize.png").name)

    print("\n[まとめ]")
    print("  - SessionOptions の graph_optimization_level と intra_op_num_threads が二大ツマミ。")
    print("  - 『上げれば速い』は思い込み。小モデルでは効果が小さく逆転もある → 必ず計測。")
    print("  - torch と onnxruntime のスレッド数を揃えないとベンチは不公平になる。")


def _save_chart(out: pathlib.Path, base: dict, r_off: dict, r_on: dict, threads: dict) -> None:
    """最適化レベル比較とスレッドスイープを 2 枚の棒グラフにして保存する。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    labels = ["eager\ntorch", "ORT\nDISABLE", "ORT\nENABLE_ALL"]
    vals = [base["p50_ms"], r_off["p50_ms"], r_on["p50_ms"]]
    axes[0].bar(labels, vals, color=["tab:gray", "tab:orange", "tab:blue"])
    axes[0].set_ylabel("latency p50 (ms, 小さいほど良い)")
    axes[0].set_title("backend / graph optimization")
    for i, v in enumerate(vals):
        axes[0].text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)

    ts = sorted(threads)
    tp = [threads[t]["throughput"] for t in ts]
    axes[1].plot(ts, tp, "o-", color="tab:green")
    axes[1].set_xlabel("intra_op_num_threads")
    axes[1].set_ylabel("throughput (img/s, 大きいほど良い)")
    axes[1].set_title("thread sweep (ORT_ENABLE_ALL)")
    axes[1].set_xticks(ts)

    fig.tight_layout()
    fig.savefig(out / "02_session_optimize.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
