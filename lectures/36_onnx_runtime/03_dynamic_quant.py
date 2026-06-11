"""03: ONNX 動的量子化 — CPU で最も手軽に効く int8 化。

学ぶこと:
  - onnxruntime.quantization.quantize_dynamic は、重みを int8 に量子化し、活性は推論時に
    その場で量子化する（キャリブレーション用データ不要）方式。CPU 推論で最も導入が容易で
    費用対効果が高い圧縮。Linear/MatMul/Conv の重みが int8 になりファイルサイズが約 1/4 に。
  - CPU の int8 行列演算は U8(activation)×S8(weight) の組み合わせが基本。ORT の動的量子化は
    既定でこの構成になり、weight_type には QUInt8 を指定すれば素直に動く。
  - 圧縮は『三角関係（速度・サイズ・精度）』で評価する。サイズと速度だけ見て精度を測らないのは
    最悪。ここでは torch fp32 / ONNX fp32 / ONNX int8 の3者を、精度・p50・サイズで比較する。
  - 落とし穴: int8 が CPU で必ず速くなるとは限らない（小モデル・少バッチでは量子化/逆量子化の
    オーバーヘッドが勝つこともある）。だから必ず実測する。

実行:
  uv run python lectures/36_onnx_runtime/03_dynamic_quant.py
結果（図）は outputs/36_onnx_runtime/ に保存される。
"""

from __future__ import annotations

import pathlib
import sys
import warnings

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from onnx_helpers import (  # noqa: E402
    accuracy_session,
    accuracy_torch,
    benchmark,
    ensure_output_dir,
    export_onnx,
    file_size_mb,
    make_dataset,
    make_session,
    pin_threads,
    run_session,
    set_seed,
    train_tiny_cnn,
)


def main() -> None:
    warnings.filterwarnings("ignore")  # quantize_dynamic の前処理推奨 Warning 等を抑制
    set_seed(0)
    pin_threads(1)
    out = ensure_output_dir()

    # === 1. モデルと評価データ ===========================================
    print("[1] TinyCNN を学習し、評価用の合成テストセットを用意")
    model = train_tiny_cnn(epochs=3, verbose=False)
    x_test, y_test = make_dataset(n_per_class=80, seed=2024)  # 240 枚の評価セット
    batch = x_test.shape[0]
    x_np = x_test.numpy()

    # === 2. fp32 ONNX をエクスポート =====================================
    fp32_path = export_onnx(model, x_test[:1], out / "tinycnn_fp32.onnx", dynamic_batch=True)

    # === 3. 動的量子化（int8）===========================================
    from onnxruntime.quantization import QuantType, quantize_dynamic

    int8_path = out / "tinycnn_int8.onnx"
    quantize_dynamic(
        model_input=str(fp32_path),
        model_output=str(int8_path),
        weight_type=QuantType.QUInt8,  # CPU は U8X8 構成。重みを uint8 へ。
    )
    print(f"\n[2] 動的量子化 完了: {int8_path.name}")

    # === 4. 三者をベンチ＆精度評価 =======================================
    sess_fp32 = make_session(fp32_path, optimize=True, intra_threads=1)
    sess_int8 = make_session(int8_path, optimize=True, intra_threads=1)

    def torch_call() -> None:
        with torch.inference_mode():
            model(x_test)

    rows = []
    # torch fp32
    b = benchmark(torch_call, batch_size=batch, warmup=8, iters=40)
    rows.append(("torch fp32", accuracy_torch(model, x_test, y_test),
                 b["p50_ms"], file_size_mb(fp32_path)))
    # ONNX fp32
    b = benchmark(lambda: run_session(sess_fp32, x_np), batch_size=batch, warmup=8, iters=40)
    rows.append(("ONNX fp32", accuracy_session(sess_fp32, x_test, y_test),
                 b["p50_ms"], file_size_mb(fp32_path)))
    # ONNX int8
    b = benchmark(lambda: run_session(sess_int8, x_np), batch_size=batch, warmup=8, iters=40)
    rows.append(("ONNX int8", accuracy_session(sess_int8, x_test, y_test),
                 b["p50_ms"], file_size_mb(int8_path)))

    # === 5. 比較表を表示 =================================================
    print("\n[3] 速度・サイズ・精度の三角関係（評価 240 枚）")
    print(f"    {'手法':<12}{'accuracy':>10}{'p50(ms)':>12}{'size(MB)':>12}")
    for name, acc, p50, mb in rows:
        print(f"    {name:<12}{acc:>10.3f}{p50:>12.3f}{mb:>12.3f}")

    size_ratio = rows[1][3] / max(rows[2][3], 1e-9)
    acc_drop = rows[1][1] - rows[2][1]
    print(f"\n    int8 でサイズ {size_ratio:.2f}x 縮小 / 精度劣化 {acc_drop:+.3f}")
    print("    （小モデルでは速度差が小さい/逆転もある＝サイズ削減が主効果。要実測）")

    # === 6. 図に保存 =====================================================
    _save_chart(out, rows)
    print("\n[4] 図を保存:", (out / "03_dynamic_quant.png").name)

    print("\n[まとめ]")
    print("  - quantize_dynamic はキャリブ不要・1 行で int8 化、サイズ約 1/4 が手堅く効く。")
    print("  - CPU は U8X8 なので weight_type=QuantType.QUInt8 を使う。")
    print("  - 必ず accuracy / p50 / size の三角で評価する。速度は必ずしも上がらない。")


def _save_chart(out: pathlib.Path, rows: list) -> None:
    """精度・p50・サイズの 3 指標を 3 枚の棒グラフにして保存する。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [r[0].replace(" ", "\n") for r in rows]
    accs = [r[1] for r in rows]
    p50s = [r[2] for r in rows]
    sizes = [r[3] for r in rows]
    colors = ["tab:gray", "tab:blue", "tab:red"]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, vals, title, fmt in (
        (axes[0], accs, "accuracy (高いほど良い)", "{:.3f}"),
        (axes[1], p50s, "latency p50 ms (低いほど良い)", "{:.2f}"),
        (axes[2], sizes, "model size MB (低いほど良い)", "{:.2f}"),
    ):
        ax.bar(names, vals, color=colors)
        ax.set_title(title)
        for i, v in enumerate(vals):
            ax.text(i, v, fmt.format(v), ha="center", va="bottom", fontsize=9)
    axes[0].set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(out / "03_dynamic_quant.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
