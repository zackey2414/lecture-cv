"""章末ミニプロジェクト: 圧縮手法ベンチ — 三角関係(精度/速度/サイズ)で比較する。

統合課題:
  これまでの部品(動的/静的量子化・QAT・非構造/構造プルーニング)を1つにまとめ、
  「同じ小分類モデルに各手法を適用し、accuracy / latency(p50,p99) / モデルサイズ(MB) を
  同一指標で公平に比較し、用途別の最適手法を意思決定する」ベンチを完成させる。
  これがこのモジュールの deliverable。

完成形がやること:
  1. Linear 主体(TinyMLP) と Conv 主体(TinyConvNet) を学習。
  2. 各モデルに適切な圧縮を適用:
       MLP → 動的量子化 / 非構造化90%プルーニング(罠) / 構造化リビルド(本当に縮む)
       CNN → 静的PTQ / QAT（任意で torchao があれば int8 dynamic も）
  3. 三指標を1つの表にまとめ、図と JSON に保存。
  4. 速度・サイズ・精度のトレードオフから「意思決定順序」を提示する。

実行:
  uv run python lectures/35_quantization_pruning/mini_project.py
結果は outputs/35_quantization_pruning/ に保存される。
"""

from __future__ import annotations

import copy
import json
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from quant_lab import (  # noqa: E402
    accuracy,
    benchmark,
    configure_backend,
    dynamic_quantize,
    ensure_output_dir,
    get_data,
    get_trained_cnn,
    get_trained_mlp,
    global_unstructured_prune,
    qat_quantize,
    quiet_warnings,
    remove_prune_reparam,
    state_dict_size_mb,
    static_quantize,
    structured_compress_mlp,
    weight_sparsity,
)

SAMPLE = torch.randn(128, 1, 16, 16)  # 公平比較のため全手法を同一バッチで計測


class CompressionBench:
    """1つのベースモデルに複数の圧縮手法を適用し、三指標で比較する。"""

    def __init__(self, family: str, ascii_title: str, base: torch.nn.Module, x_te, y_te) -> None:
        self.family = family  # 表示用（日本語可）
        self.ascii_title = ascii_title  # 図用（ASCII。matplotlib のフォント警告回避）
        self.base = base
        self.x_te = x_te
        self.y_te = y_te
        self.rows: dict[str, dict] = {}

    def add(self, name: str, model: torch.nn.Module, note: str = "") -> None:
        """1手法を評価して表に1行追加する（accuracy / size / latency）。"""
        b = benchmark(model, SAMPLE)
        self.rows[name] = {
            "accuracy": round(accuracy(model, self.x_te, self.y_te), 4),
            "size_mb": round(state_dict_size_mb(model), 4),
            "p50_ms": round(b["p50"], 4),
            "p99_ms": round(b["p99"], 4),
            "sparsity": round(weight_sparsity(model), 3),
            "note": note,
        }

    def print_table(self) -> None:
        base_size = self.rows[next(iter(self.rows))]["size_mb"]
        print(f"\n=== {self.family} ===")
        print(f"    {'method':<22}{'acc':>7}{'size(MB)':>10}{'x':>6}"
              f"{'p50(ms)':>9}{'p99(ms)':>9}{'spars':>7}")
        for name, m in self.rows.items():
            shrink = base_size / max(m["size_mb"], 1e-9)
            print(f"    {name:<22}{m['accuracy']:>7.3f}{m['size_mb']:>10.4f}"
                  f"{shrink:>5.1f}x{m['p50_ms']:>9.3f}{m['p99_ms']:>9.3f}{m['sparsity']:>7.2f}")


def _maybe_torchao(bench: CompressionBench, base: torch.nn.Module) -> None:
    """torchao があれば int8 dynamic を追加（任意・未導入ならスキップ）。"""
    try:
        from torchao.quantization import (  # noqa: PLC0415
            Int8DynamicActivationInt8WeightConfig,
            quantize_,
        )
    except Exception:  # noqa: BLE001
        print("    [info] torchao 未導入のため torchao 行はスキップ（概念は 05 を参照）。")
        return
    m = copy.deepcopy(base)
    try:
        quantize_(m, Int8DynamicActivationInt8WeightConfig())
        bench.add("torchao int8(dyn)", m, note="torchao quantize_")
    except Exception as exc:  # noqa: BLE001
        print(f"    [info] torchao quantize_ は環境依存で失敗（{type(exc).__name__}）。スキップ。")


def build_mlp_bench(x_tr, y_tr, x_te, y_te) -> CompressionBench:
    """Linear 主体モデルの圧縮ベンチ（動的量子化 / プルーニング各種）。"""
    mlp = get_trained_mlp()
    bench = CompressionBench("TinyMLP (Linear 主体)", "TinyMLP (Linear-heavy)", mlp, x_te, y_te)
    bench.add("fp32 baseline", mlp)
    bench.add("dynamic int8", dynamic_quantize(mlp), note="Linear→int8, 手軽")

    pruned = copy.deepcopy(mlp)
    global_unstructured_prune(pruned, 0.9)
    remove_prune_reparam(pruned)
    bench.add("unstructured 90%", pruned, note="罠: 密のまま縮まない")

    bench.add("structured keep=0.5", structured_compress_mlp(mlp, 0.5), note="本当に縮む密モデル")
    _maybe_torchao(bench, mlp)
    return bench


def build_cnn_bench(x_tr, y_tr, x_te, y_te) -> CompressionBench:
    """Conv 主体モデルの圧縮ベンチ（静的PTQ / QAT）。"""
    cnn = get_trained_cnn()
    bench = CompressionBench("TinyConvNet (Conv 主体)", "TinyConvNet (Conv-heavy)", cnn, x_te, y_te)
    bench.add("fp32 baseline", cnn)
    bench.add("dynamic int8", dynamic_quantize(cnn), note="Conv に効かず")
    bench.add("static PTQ int8", static_quantize(cnn, x_tr[:256]), note="Conv も int8")
    bench.add("QAT int8", qat_quantize(cnn, x_tr, y_tr, epochs=3), note="精度回復策")
    return bench


def main() -> None:
    quiet_warnings()
    engine = configure_backend()
    out = ensure_output_dir()
    x_tr, y_tr, x_te, y_te = get_data()
    print(f"[setup] backend={engine}, threads={torch.get_num_threads()}, "
          f"batch={SAMPLE.shape[0]}")

    # === 1. 2系統のベンチを構築 ========================================
    mlp_bench = build_mlp_bench(x_tr, y_tr, x_te, y_te)
    cnn_bench = build_cnn_bench(x_tr, y_tr, x_te, y_te)
    mlp_bench.print_table()
    cnn_bench.print_table()

    # === 2. 図と JSON に保存 ===========================================
    _save_figure(out, mlp_bench, cnn_bench)
    payload = {
        "backend": engine,
        "batch": SAMPLE.shape[0],
        "TinyMLP": mlp_bench.rows,
        "TinyConvNet": cnn_bench.rows,
    }
    (out / "mini_compression_bench.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\n[保存]", (out / "mini_compression_bench.png").name, "/",
          (out / "mini_compression_bench.json").name)

    # === 3. 意思決定順序（実務の手引き） ===============================
    print("\n[意思決定順序 — まず安く効く順に試す]")
    print("  1) model.eval() + torch.inference_mode()（34回・タダで効く前提条件）")
    print("  2) bf16 autocast + torch.compile（34回・学習も変換も不要）")
    print("  3) 動的量子化 / ONNX 動的量子化（Linear/Transformer 主体に即効・キャリブ不要）")
    print("  4) 静的PTQ（Conv 主体・キャリブレーションデータがある）")
    print("  5) QAT（PTQ で精度が落ちすぎる時のみ・学習コストを払う）")
    print("  6) 構造化プルーニング+リビルド（さらに小さくしたい時。非構造化は実速度に効かない）")
    print("\n完了。三角関係(精度/速度/サイズ)で手法を選べるベンチが動きました。")


def _save_figure(out: pathlib.Path, mlp_bench: CompressionBench,
                 cnn_bench: CompressionBench) -> None:
    """MLP/CNN それぞれの size と p50 を手法ごとに棒グラフ化して保存。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for col, bench in enumerate((mlp_bench, cnn_bench)):
        names = list(bench.rows.keys())
        short = [n.replace(" int8", "").replace(" baseline", "") for n in names]
        sizes = [bench.rows[n]["size_mb"] for n in names]
        p50s = [bench.rows[n]["p50_ms"] for n in names]
        accs = [bench.rows[n]["accuracy"] for n in names]

        ax = axes[0][col]
        bars = ax.bar(short, sizes, color="tab:blue")
        ax.set_title(f"{bench.ascii_title}\nmodel size (MB)")
        ax.set_ylabel("size (MB)")
        ax.tick_params(axis="x", labelrotation=25, labelsize=8)
        for bar, a in zip(bars, accs):  # 精度をバー上に注記
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"acc {a:.2f}", ha="center", va="bottom", fontsize=7)

        ax = axes[1][col]
        ax.bar(short, p50s, color="tab:orange")
        ax.set_title("p50 latency (ms, batch=128)")
        ax.set_ylabel("p50 (ms)")
        ax.tick_params(axis="x", labelrotation=25, labelsize=8)

    fig.tight_layout()
    fig.savefig(out / "mini_compression_bench.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
