"""章末ミニプロジェクト: 『PyTorch → ONNX デプロイ・ベンチ一式』。

この回で学んだ要素を 1 本に統合する完成形:
  (1) 学習済みモデルを用意（TinyCNN を合成データで軽く学習）
  (2) ONNX エクスポート（dynamo=False / 動的バッチ）
  (3) 検証: onnx.checker + torch との数値一致(atol/rtol, 最大絶対誤差)
  (4) onnxruntime 推論（グラフ最適化・スレッド固定）
  (5) 動的量子化(int8)で圧縮
  (6) torch fp32 / ONNX fp32 / ONNX int8 を accuracy・latency(p50/p99)・throughput・size で
      公平比較し、用途別の推奨を意思決定。表・図・JSON レポートを出力。

実行:
  uv run python lectures/36_onnx_runtime/mini_project.py
成果物（図・JSON）は lectures/36_onnx_runtime/outputs/ に保存される。
"""

from __future__ import annotations

import json
import pathlib
import sys
import warnings

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from onnx_helpers import (  # noqa: E402
    accuracy_session,
    accuracy_torch,
    benchmark,
    check_onnx,
    ensure_output_dir,
    export_onnx,
    file_size_mb,
    make_dataset,
    make_session,
    numerical_agreement,
    pin_threads,
    run_session,
    set_seed,
    train_tiny_cnn,
)


def main() -> None:
    warnings.filterwarnings("ignore")
    set_seed(0)
    pin_threads(1)  # 公平なベンチのため torch / ORT を 1 スレッドに固定
    out = ensure_output_dir()
    print("=" * 64)
    print("ミニプロジェクト: PyTorch → ONNX デプロイ・ベンチ一式")
    print("=" * 64)

    # === (1) 学習済みモデル =============================================
    print("\n[1] TinyCNN を合成図形データで学習")
    model = train_tiny_cnn(epochs=4, verbose=False)
    x_test, y_test = make_dataset(n_per_class=80, seed=2024)  # 240 枚の評価セット
    batch = x_test.shape[0]
    x_np = x_test.numpy()

    # === (2) ONNX エクスポート =========================================
    fp32_path = export_onnx(model, x_test[:1], out / "mp_fp32.onnx", dynamic_batch=True)
    print(f"[2] ONNX 書き出し: {fp32_path.name} ({file_size_mb(fp32_path):.3f} MB)")

    # === (3) 検証（壊れた ONNX を出さない最重要ステップ） ==============
    check_onnx(fp32_path)
    sess_fp32 = make_session(fp32_path, optimize=True, intra_threads=1)
    with torch.inference_mode():
        ref = model(x_test).numpy()
    got = run_session(sess_fp32, x_np)
    agree = numerical_agreement(got, ref)
    verify_ok = bool(np.allclose(got, ref, atol=1e-4, rtol=1e-3))
    print(f"[3] 数値一致: 最大絶対誤差={agree['max_abs']:.3e} allclose={verify_ok} "
          f"→ {'OK ✅' if verify_ok else 'NG ❌（デプロイ中止）'}")
    if not verify_ok:
        print("    数値一致しないモデルは本番に出してはいけない。export 設定を見直すこと。")
        return

    # === (4)+(5) 動的量子化 ============================================
    from onnxruntime.quantization import QuantType, quantize_dynamic

    int8_path = out / "mp_int8.onnx"
    quantize_dynamic(str(fp32_path), str(int8_path), weight_type=QuantType.QUInt8)
    sess_int8 = make_session(int8_path, optimize=True, intra_threads=1)
    print(f"[4] 動的量子化(int8): {int8_path.name} ({file_size_mb(int8_path):.3f} MB)")

    # === (6) 公平比較 ==================================================
    def torch_call() -> None:
        with torch.inference_mode():
            model(x_test)

    rows: list[dict] = []
    rows.append(_row("torch fp32", accuracy_torch(model, x_test, y_test),
                     benchmark(torch_call, batch_size=batch, warmup=8, iters=50),
                     file_size_mb(fp32_path)))
    rows.append(_row("ONNX fp32", accuracy_session(sess_fp32, x_test, y_test),
                     benchmark(lambda: run_session(sess_fp32, x_np), batch_size=batch, warmup=8, iters=50),
                     file_size_mb(fp32_path)))
    rows.append(_row("ONNX int8", accuracy_session(sess_int8, x_test, y_test),
                     benchmark(lambda: run_session(sess_int8, x_np), batch_size=batch, warmup=8, iters=50),
                     file_size_mb(int8_path)))

    print("\n[5] 公平比較（評価 240 枚・1 スレッド）")
    print(f"    {'手法':<12}{'acc':>8}{'p50(ms)':>10}{'p99(ms)':>10}{'img/s':>10}{'MB':>9}")
    for r in rows:
        print(f"    {r['name']:<12}{r['accuracy']:>8.3f}{r['p50_ms']:>10.3f}"
              f"{r['p99_ms']:>10.3f}{r['throughput']:>10.0f}{r['size_mb']:>9.3f}")

    # === 意思決定（推奨） =============================================
    rec = _recommend(rows)
    print("\n[6] 意思決定（この結果からの推奨）")
    for line in rec:
        print("    " + line)

    # === レポート保存 =================================================
    report = {
        "verify": {"max_abs": agree["max_abs"], "allclose": verify_ok},
        "comparison": rows,
        "recommendation": rec,
    }
    (out / "mini_project_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _save_chart(out, rows)
    print("\n[7] 保存: mini_project_report.json / mini_project.png")
    print("\n完了 🎉  export→verify→optimize→quantize→公平比較 の一連を踏破した。")


def _row(name: str, acc: float, bench: dict, size_mb: float) -> dict:
    """比較表 1 行分の dict にまとめる。"""
    return {
        "name": name,
        "accuracy": acc,
        "p50_ms": bench["p50_ms"],
        "p99_ms": bench["p99_ms"],
        "throughput": bench["throughput"],
        "size_mb": size_mb,
    }


def _recommend(rows: list[dict]) -> list[str]:
    """比較結果から実務的な推奨を文章で導く（速度・サイズ・精度の三角で判断）。"""
    by_name = {r["name"]: r for r in rows}
    fp32 = by_name["ONNX fp32"]
    int8 = by_name["ONNX int8"]
    torch_row = by_name["torch fp32"]
    lines: list[str] = []

    # ONNX が eager torch より速いか
    if fp32["p50_ms"] < torch_row["p50_ms"]:
        lines.append(f"ONNX fp32 は eager torch より p50 が速い "
                     f"({torch_row['p50_ms']:.2f}→{fp32['p50_ms']:.2f}ms)。CPU 配布の既定候補。")
    else:
        lines.append("この環境では ONNX fp32 が eager torch を上回らない。"
                     "効果は op・モデル・スレッドで変わる＝必ず計測する。")

    # int8 のサイズと精度劣化
    size_ratio = fp32["size_mb"] / max(int8["size_mb"], 1e-9)
    acc_drop = fp32["accuracy"] - int8["accuracy"]
    lines.append(f"int8 はサイズ {size_ratio:.2f}x 縮小・精度劣化 {acc_drop:+.3f}。"
                 "配布サイズ/メモリが課題なら有力。")

    # 速度面で int8 が勝つか
    if int8["p50_ms"] <= fp32["p50_ms"]:
        lines.append("int8 は速度でも fp32 以下のレイテンシ。サイズと速度を両取りできる。")
    else:
        lines.append("ただし小モデルでは int8 が量子化/逆量子化コストで遅くなることがある"
                     "（速度目的なら fp32+スレッド調整も検討）。")

    if acc_drop > 0.05:
        lines.append("精度劣化が大きいので、静的PTQ/QAT（第35回）での回復を検討。")
    return lines


def _save_chart(out: pathlib.Path, rows: list[dict]) -> None:
    """accuracy / p50 / size の 3 指標を 1 枚にまとめて保存する。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [r["name"].replace(" ", "\n") for r in rows]
    colors = ["tab:gray", "tab:blue", "tab:red"]
    metrics = [
        ("accuracy", [r["accuracy"] for r in rows], "accuracy (高いほど良い)", "{:.3f}"),
        ("p50_ms", [r["p50_ms"] for r in rows], "latency p50 ms (低いほど良い)", "{:.2f}"),
        ("size_mb", [r["size_mb"] for r in rows], "size MB (低いほど良い)", "{:.2f}"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, (_, vals, title, fmt) in zip(axes, metrics):
        ax.bar(names, vals, color=colors)
        ax.set_title(title)
        for i, v in enumerate(vals):
            ax.text(i, v, fmt.format(v), ha="center", va="bottom", fontsize=9)
    axes[0].set_ylim(0, 1.05)
    fig.suptitle("mini project: torch vs ONNX vs ONNX-int8", fontsize=12)
    fig.tight_layout()
    fig.savefig(out / "mini_project.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
