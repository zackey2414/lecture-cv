"""03: 静的PTQ と QAT — Conv も int8 化する（fuse + observer + キャリブレーション）。

学ぶこと:
  - 静的PTQ(Post-Training Quantization, static)は activation の範囲を**事前に**代表データで
    観測(キャリブレーション)して固定する方式。Conv 主体の CNN でも int8 で動かせる。
    必要な部品:
        QuantStub / DeQuantStub … 入口で fp32→int8、出口で int8→fp32 の境界
        fuse_modules            … conv→bn→relu を 1 演算へ畳む（精度・速度に効く）
        get_default_qconfig     … observer の構成（per-channel 重み + activation 観測）
        prepare → calibrate → convert … observer 挿入 → 範囲観測 → int8 確定
  - QAT(Quantization-Aware Training)は学習中に fake-quant(丸めの模擬)を挟み、
    量子化誤差込みで重みを微調整する。静的PTQ で精度が落ちる時の次の一手。
  - 三角評価で fp32 / 動的 / 静的 / QAT を並べ、サイズ・精度・レイテンシを比較する。

実行:
  uv run python lectures/35_quantization_pruning/03_static_ptq_qat.py
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
    qat_quantize,
    quiet_warnings,
    state_dict_size_mb,
    static_quantize,
)


def _row(model: torch.nn.Module, x_te, y_te, sample) -> dict:
    """1 条件の 三角指標をまとめる。"""
    b = benchmark(model, sample)
    return {
        "accuracy": round(accuracy(model, x_te, y_te), 4),
        "size_mb": round(state_dict_size_mb(model), 4),
        "p50_ms": round(b["p50"], 4),
        "p99_ms": round(b["p99"], 4),
    }


def main() -> None:
    quiet_warnings()
    engine = configure_backend()
    out = ensure_output_dir()
    x_tr, y_tr, x_te, y_te = get_data()
    sample = torch.randn(128, 1, 16, 16)
    print(f"[setup] backend={engine}, threads={torch.get_num_threads()}\n")

    # === 1. fp32 ベースライン ==========================================
    cnn = get_trained_cnn()
    print("[1] TinyConvNet（Conv 主体）を題材に 4 方式を比較\n")

    # === 2. 動的量子化（比較用: CNN にはほぼ効かない） =================
    dyn = dynamic_quantize(cnn)

    # === 3. 静的PTQ（fuse + キャリブレーション + convert） =============
    # キャリブレーションは学習データの一部で十分（ラベル不要・推論を流すだけ）。
    static = static_quantize(cnn, calib_x=x_tr[:256], backend=engine)

    # === 4. QAT（fake-quant で数エポック微調整 → convert） =============
    qat = qat_quantize(cnn, x_tr, y_tr, backend=engine, epochs=3)

    rows = {
        "fp32 (baseline)": _row(cnn, x_te, y_te, sample),
        "dynamic int8": _row(dyn, x_te, y_te, sample),
        "static PTQ int8": _row(static, x_te, y_te, sample),
        "QAT int8": _row(qat, x_te, y_te, sample),
    }
    print(f"    {'method':<18}{'acc':>7}{'size(MB)':>11}{'p50(ms)':>10}{'p99(ms)':>10}")
    for name, m in rows.items():
        print(f"    {name:<18}{m['accuracy']:>7.3f}{m['size_mb']:>11.4f}"
              f"{m['p50_ms']:>10.3f}{m['p99_ms']:>10.3f}")

    base = rows["fp32 (baseline)"]
    s_shrink = base["size_mb"] / max(rows["static PTQ int8"]["size_mb"], 1e-9)
    d_shrink = base["size_mb"] / max(rows["dynamic int8"]["size_mb"], 1e-9)
    print(f"\n    → 静的PTQ は Conv も int8 化して {s_shrink:.1f}x 縮小。"
          f"動的量子化は {d_shrink:.2f}x どまり（Conv は fp32 のまま）。")
    print("    → QAT は静的PTQ で精度が落ちる場合の回復策（本タスクは元から易しく差は小さい）。")

    # === 5. 保存 =======================================================
    _save_figure(out, rows)
    (out / "03_static_qat.json").write_text(
        json.dumps({"backend": engine, "results": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("\n[5] 保存:", (out / "03_static_vs_qat.png").name, "/", (out / "03_static_qat.json").name)

    print("\n[まとめ]")
    print("  - 静的PTQ = QuantStub/DeQuantStub + fuse + キャリブレーション + convert。Conv に効く。")
    print("  - 動的との違い: activation の範囲を事前固定するので Conv も int8 実行できる。")
    print("  - QAT は学習で量子化誤差を吸収し、PTQ で精度劣化が大きいときに使う上位互換。")


def _save_figure(out: pathlib.Path, rows: dict) -> None:
    """4 方式の サイズ と 精度 を並べた棒グラフを保存する。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(rows.keys())
    sizes = [rows[n]["size_mb"] for n in names]
    accs = [rows[n]["accuracy"] for n in names]
    short = [n.replace(" int8", "").replace(" (baseline)", "") for n in names]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
    axes[0].bar(short, sizes, color=["tab:gray", "tab:cyan", "tab:green", "tab:purple"])
    axes[0].set_ylabel("state_dict size (MB)")
    axes[0].set_title("static PTQ shrinks Conv too")
    axes[0].tick_params(axis="x", labelrotation=15)

    axes[1].bar(short, accs, color=["tab:gray", "tab:cyan", "tab:green", "tab:purple"])
    axes[1].set_ylim(0, 1.05)
    axes[1].set_ylabel("test accuracy")
    axes[1].set_title("accuracy retained after int8")
    axes[1].tick_params(axis="x", labelrotation=15)

    fig.tight_layout()
    fig.savefig(out / "03_static_vs_qat.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
