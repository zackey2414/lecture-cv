"""02: 動的PTQ — quantize_dynamic で Linear を int8 化（CPU で最も手軽）。

学ぶこと:
  - 動的量子化(Post-Training Quantization, dynamic)は「重みを事前に int8 で保存し、
    activation は推論のたびにその場で量子化する」方式。キャリブレーション不要・1 行で適用でき、
    CPU 推論で最も手軽に効く。
  - 対象は既定で Linear / LSTM。だから **Linear 主体のモデル(MLP/Transformer)では縮むが、
    Conv 主体の CNN ではほとんど効かない**（Conv の重みは fp32 のまま）。
  - 三角評価: サイズ(MB) は約 1/4 に縮む、精度はほぼ保たれる、レイテンシは
    「小モデル/小バッチでは量子化のオーバヘッドで逆に遅くなることもある」現実を実測する。

実行:
  uv run python lectures/35_quantization_pruning/02_dynamic_ptq.py
結果は lectures/35_quantization_pruning/outputs/ に保存される。
"""

from __future__ import annotations

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
    quiet_warnings,
    state_dict_size_mb,
)


def _evaluate(model: torch.nn.Module, x_te, y_te, sample) -> dict:
    """1 モデルの 三角指標(accuracy / size / latency) をまとめて返す。"""
    bench = benchmark(model, sample)
    return {
        "accuracy": round(accuracy(model, x_te, y_te), 4),
        "size_mb": round(state_dict_size_mb(model), 4),
        "p50_ms": round(bench["p50"], 4),
        "p99_ms": round(bench["p99"], 4),
        "throughput": round(bench["throughput"], 1),
    }


def _print_table(title: str, rows: dict[str, dict]) -> None:
    """before/after を 1 つの表で表示する。"""
    print(title)
    print(f"    {'condition':<16}{'acc':>7}{'size(MB)':>11}{'p50(ms)':>10}"
          f"{'p99(ms)':>10}{'img/s':>10}")
    for name, m in rows.items():
        print(f"    {name:<16}{m['accuracy']:>7.3f}{m['size_mb']:>11.4f}"
              f"{m['p50_ms']:>10.3f}{m['p99_ms']:>10.3f}{m['throughput']:>10.1f}")


def main() -> None:
    quiet_warnings()
    engine = configure_backend()
    out = ensure_output_dir()
    _, _, x_te, y_te = get_data()
    print(f"[setup] backend={engine}, threads={torch.get_num_threads()}\n")

    # === 1. Linear 主体の MLP に動的量子化 → よく効く ====================
    mlp = get_trained_mlp()
    mlp_q = dynamic_quantize(mlp)
    # 小バッチ(=1)と大バッチ(=128)の両方で測り、レイテンシの罠を見る。
    sample1 = torch.randn(1, 1, 16, 16)
    sample128 = torch.randn(128, 1, 16, 16)

    mlp_rows = {
        "fp32 (bs=1)": _evaluate(mlp, x_te, y_te, sample1),
        "int8 (bs=1)": _evaluate(mlp_q, x_te, y_te, sample1),
        "fp32 (bs=128)": _evaluate(mlp, x_te, y_te, sample128),
        "int8 (bs=128)": _evaluate(mlp_q, x_te, y_te, sample128),
    }
    _print_table("[1] TinyMLP（Linear 主体）— 動的量子化はよく効く", mlp_rows)
    shrink = mlp_rows["fp32 (bs=1)"]["size_mb"] / max(mlp_rows["int8 (bs=1)"]["size_mb"], 1e-9)
    print(f"    → サイズ {shrink:.1f}x 縮小・精度ほぼ維持。"
          "ただし bs=1 では int8 が fp32 より速いとは限らない（オーバヘッド）。\n")

    # === 2. Conv 主体の CNN に動的量子化 → ほとんど効かない ==============
    cnn = get_trained_cnn()
    cnn_q = dynamic_quantize(cnn)
    cnn_rows = {
        "fp32 (bs=128)": _evaluate(cnn, x_te, y_te, sample128),
        "int8 (bs=128)": _evaluate(cnn_q, x_te, y_te, sample128),
    }
    _print_table("[2] TinyConvNet（Conv 主体）— 動的量子化はほぼ効かない", cnn_rows)
    cnn_shrink = cnn_rows["fp32 (bs=128)"]["size_mb"] / max(cnn_rows["int8 (bs=128)"]["size_mb"], 1e-9)
    print(f"    → 縮小は {cnn_shrink:.2f}x どまり（Linear ヘッドだけ int8、Conv は fp32 のまま）。")
    print("       Conv を量子化したいなら次回 03 の『静的PTQ』が必要。\n")

    # === 3. 図と JSON に保存 ============================================
    _save_figure(out, mlp_rows, cnn_rows)
    payload = {"backend": engine, "mlp": mlp_rows, "cnn": cnn_rows}
    (out / "02_dynamic_ptq.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("[3] 保存:", (out / "02_dynamic_quant.png").name, "/", (out / "02_dynamic_ptq.json").name)

    print("\n[まとめ]")
    print("  - 動的量子化は 1 行・キャリブレーション不要で、Linear 主体モデルに最適。")
    print("  - サイズは約 1/4・精度ほぼ維持。CNN(Conv 主体)には効かない → 静的PTQ へ。")
    print("  - 速度は『大きいモデル/大きいバッチ』ほど int8 が有利。小規模では逆転しうる。")


def _save_figure(out: pathlib.Path, mlp_rows: dict, cnn_rows: dict) -> None:
    """サイズ縮小（MLP vs CNN）と MLP のバッチ別レイテンシを棒グラフで保存。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))

    # 左: サイズ縮小の効き目（MLP は効く / CNN は効かない）
    ax = axes[0]
    labels = ["MLP fp32", "MLP int8", "CNN fp32", "CNN int8"]
    sizes = [
        mlp_rows["fp32 (bs=1)"]["size_mb"], mlp_rows["int8 (bs=1)"]["size_mb"],
        cnn_rows["fp32 (bs=128)"]["size_mb"], cnn_rows["int8 (bs=128)"]["size_mb"],
    ]
    colors = ["tab:blue", "tab:cyan", "tab:red", "tab:orange"]
    ax.bar(labels, sizes, color=colors)
    ax.set_ylabel("state_dict size (MB)")
    ax.set_title("dynamic int8: MLP shrinks, CNN barely")
    ax.tick_params(axis="x", labelrotation=20)

    # 右: MLP の p50 レイテンシ（bs=1 / bs=128）で fp32 vs int8
    ax = axes[1]
    groups = ["bs=1", "bs=128"]
    fp32 = [mlp_rows["fp32 (bs=1)"]["p50_ms"], mlp_rows["fp32 (bs=128)"]["p50_ms"]]
    int8 = [mlp_rows["int8 (bs=1)"]["p50_ms"], mlp_rows["int8 (bs=128)"]["p50_ms"]]
    x = range(len(groups))
    ax.bar([i - 0.2 for i in x], fp32, width=0.4, label="fp32", color="tab:blue")
    ax.bar([i + 0.2 for i in x], int8, width=0.4, label="int8", color="tab:cyan")
    ax.set_xticks(list(x))
    ax.set_xticklabels(groups)
    ax.set_ylabel("p50 latency (ms)")
    ax.set_title("speed depends on batch size")
    ax.legend()

    fig.tight_layout()
    fig.savefig(out / "02_dynamic_quant.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
