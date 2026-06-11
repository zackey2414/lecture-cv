"""03: torch.profiler でボトルネックを特定する — 当て推量をやめる。

学ぶこと:
  - torch.profiler.profile(activities=[ProfilerActivity.CPU]) で演算子ごとの時間を計測する。
  - record_function("名前") で“自分の関心区間”に注釈を付け、後で集計に現れるようにする。
  - key_averages().table(sort_by="self_cpu_time_total") で重い演算子を降順に並べ、律速を特定する。
      * self_cpu_time  … その演算子“自身”が使った時間（子を含まない）。律速の特定にはこれ。
      * cpu_time_total … 子（内部で呼ぶ別演算子）も含む合計時間。
  - export_chrome_trace() で chrome://tracing / Perfetto 用のタイムラインを書き出す。
  - 「推測するな、計測せよ」: resnet18 では畳み込み（mkldnn_convolution）が支配的、という事実を“見る”。

実行:
  uv run python lectures/34_inference_profiling/03_torch_profiler.py
結果（表テキスト・chrome trace・図）は outputs/34_inference_profiling/ に保存される。
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from profiling_lab import (  # noqa: E402
    build_model,
    ensure_out_dir,
    get_device,
    make_input,
    profile_ops,
    top_self_cpu_ops,
)


def save_op_bars(ops: list[tuple[str, float]], path: pathlib.Path) -> None:
    """律速演算子（自己CPU時間）の横棒グラフを保存する。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [name.replace("aten::", "") for name, _ in ops][::-1]
    times = [t for _, t in ops][::-1]
    y = np.arange(len(ops))
    fig, ax = plt.subplots(figsize=(8.5, 0.55 * len(ops) + 1.5))
    ax.barh(y, times, color="tab:red")
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("self CPU time (ms, summed over iters)")
    ax.set_title("resnet18 inference: top operators by self CPU time")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> None:
    out = ensure_out_dir()
    device = get_device()
    model = build_model("resnet18").to(device)
    x = make_input(batch_size=2).to(device)

    def run() -> object:
        with torch.inference_mode():
            return model(x)

    # === 1. プロファイル実行 =============================================
    print("[1] torch.profiler で resnet18 推論を計測（CPU活動）")
    prof = profile_ops(run, n_warmup=3, n_iter=10, region="resnet18_inference")

    # === 2. 演算子別テーブル =============================================
    # sort_by を変えると見える景色が変わる: self_cpu は“自分の重さ”、cpu_time_total は“配下込み”。
    table = prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=12)
    print("\n[2] 演算子別 自己CPU時間 トップ12（key_averages().table）")
    print(table)

    # 入力 shape ごとに割って見ることもできる（どの解像度/チャネルが重いかの分析に有用）
    table_by_shape = prof.key_averages(group_by_input_shape=True).table(
        sort_by="self_cpu_time_total", row_limit=6
    )

    # === 3. 律速演算子を機械可読に取り出す ==============================
    ops = top_self_cpu_ops(prof, k=8)
    print("\n[3] 律速演算子（自己CPU時間 上位8件）")
    total = sum(t for _, t in ops)
    for name, t in ops:
        share = 100.0 * t / total if total > 0 else 0.0
        print(f"    {name:32s} {t:8.3f}ms  ({share:4.1f}% of top8)")
    print(f"    → 最重量: {ops[0][0]}。CNN の推論は畳み込みが支配的、という“事実”が見えた。")
    print("       次の一手（bf16 autocast / compile / 量子化）はここを狙うのが筋。")

    # === 4. 成果物を保存 =================================================
    (out / "03_profiler_table.txt").write_text(
        "# sort_by=self_cpu_time_total\n" + table + "\n\n# group_by_input_shape\n" + table_by_shape,
        encoding="utf-8",
    )
    trace_path = out / "03_chrome_trace.json"
    prof.export_chrome_trace(str(trace_path))  # chrome://tracing か https://ui.perfetto.dev で開ける
    save_op_bars(ops, out / "03_top_ops.png")

    print("\n[4] 保存物:")
    print("    -", (out / "03_profiler_table.txt").name, "（演算子別テーブルのテキスト）")
    print("    -", trace_path.name, "（chrome://tracing / Perfetto で開くタイムライン）")
    print("    -", (out / "03_top_ops.png").name, "（律速演算子の棒グラフ）")
    print("\n[まとめ] 最適化の前に profiler で律速を特定する。推測ではなく計測が“次の一手”を決める。")


if __name__ == "__main__":
    main()
