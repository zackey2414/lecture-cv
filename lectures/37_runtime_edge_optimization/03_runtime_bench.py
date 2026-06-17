"""03: ランタイム横並びベンチ — eager / bf16 / TorchScript / compile / ONNX / ONNX-int8。

「どのランタイムが速い/小さい/正確か」を*同一手順*で公平に測る統合ベンチ。
評価軸は4つを必ずセットで見る(1つだけ見ると誤った最適化に走る):

  - レイテンシ(p50/p99, batch=1) : 1リクエストの応答時間。p99 はテール遅延。
  - スループット(img/s, batched)  : まとめ処理の効率。並列性を活かせるか。
  - モデルサイズ(MB)             : 配布・メモリ。量子化で効く。
  - 精度保持(fp32 eager 基準)    : top1 一致率 と 最大絶対誤差。速くても壊れていたら無意味。

この環境で実際に動くランタイムだけを自動収集する(torch.compile が C++ ツールチェイン
不在で使えなければ表から自動的に外れる)。OpenVINO / CoreML 等のエッジ系は 04 で扱う。

実行:
  uv run python lectures/37_runtime_edge_optimization/03_runtime_bench.py
結果(表+図)は lectures/37_runtime_edge_optimization/outputs/ に保存される。
"""

from __future__ import annotations

import pathlib
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from bench_lab import (  # noqa: E402
    apply_bench_threads,
    benchmark_runtimes,
    collect_runtimes,
    ensure_output_dir,
    make_resnet18,
    print_table,
    save_bench_plot,
    synthetic_batch,
)


def main() -> None:
    threads = apply_bench_threads()
    out = ensure_output_dir()
    print(f"[setup] model=resnet18, threads={threads}（公平比較のため固定）\n")

    model = make_resnet18()

    # 1) この環境で使えるランタイムを集める(使えないものは自動スキップ)
    print("[1] ランタイムを構築...")
    runtimes = collect_runtimes(model, out, include_compile=True)
    print(f"    収集: {[r.name for r in runtimes]}\n")

    # 2) 精度保持を測るための評価バッチ(fp32 eager の出力が物差し)
    eval_batch = synthetic_batch(24, seed=99)

    # 3) 全ランタイムを同一手順でベンチ
    print("[2] ベンチ実行中(各ランタイム: warmup→反復→分位点)...\n")
    rows = benchmark_runtimes(runtimes, eval_batch, latency_iters=40, throughput_batch=16)

    # 4) 表示
    print("[3] 結果(x速 は eager_fp32 の p50 を 1.00x とした相対):")
    print_table(rows)

    # 5) 図を保存
    plot_path = out / "03_runtime_bench.png"
    save_bench_plot(rows, plot_path, "Runtime comparison (resnet18, CPU)")
    print(f"\n[4] 図を保存: {plot_path.name}")

    # 6) 観察ポイント(読者が表から学ぶべきこと)
    print("\n[5] 読み解きのポイント:")
    print("  - ONNX Runtime(fp32)は eager より速いことが多い(グラフ最適化+専用カーネル)。")
    print("  - ONNX int8 動的量子化は『サイズ ~1/4』が*確実*。ただし速度は別問題。")
    print("    動的量子化は Linear/MatMul 主体(Transformer 等)で効きやすく、")
    print("    Conv 主体の CNN(resnet18)では quant/dequant の overhead で*逆に遅くなる*ことも。")
    print("    → 「サイズは縮むが速度は伸びない」= 実効速度の罠。必ず実測で確かめる。")
    print("  - bf16 autocast はサイズ不変(重みは fp32 のまま)。速度効果は環境依存。")
    print("  - 精度保持: fp32 系は top1 一致 100%・誤差ほぼ 0。int8 は誤差が増えるので要確認。")
    print("  - p50 だけでなく p99(テール)も見る。SLA はテールで決まることが多い。")


if __name__ == "__main__":
    main()
