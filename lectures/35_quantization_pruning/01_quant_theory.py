"""01: 量子化の理論基礎 — scale / zero-point・対称/非対称・per-tensor/per-channel。

学ぶこと:
  - int8 量子化とは「実数の範囲 [min, max] を 256 段階の整数 [qmin, qmax] に写す」こと。
    写像は次の2パラメータで決まる:
        scale      … 1段階あたりの実数の刻み幅
        zero_point … 実数の 0.0 がどの整数に対応するか（非対称量子化で使う）
        量子化:   q = round(x/scale) + zero_point        （[qmin,qmax] にクランプ）
        逆量子化: x ≈ scale * (q - zero_point)
  - 対称(symmetric, zero_point=0) と 非対称(asymmetric/affine) の違い。
    重みは 0 を中心に分布しがちなので対称、activation は ReLU 後など非対称分布が多い。
  - per-tensor(テンソル全体で1つの scale) と per-channel(出力チャネルごとに scale)。
    チャネルごとにスケールが大きく違う重みでは per-channel が誤差を大幅に下げる。

実行:
  uv run python lectures/35_quantization_pruning/01_quant_theory.py
結果は lectures/35_quantization_pruning/outputs/ に保存される。
"""

from __future__ import annotations

import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from quant_lab import (  # noqa: E402
    dequantize_tensor,
    ensure_output_dir,
    quantize_tensor,
    quiet_warnings,
    set_seed,
)


def _roundtrip_error(x: torch.Tensor, symmetric: bool) -> tuple[torch.Tensor, float, float, int]:
    """量子化→逆量子化した復元値と平均絶対誤差を返す。"""
    q, scale, zp = quantize_tensor(x, num_bits=8, symmetric=symmetric)
    x_hat = dequantize_tensor(q, scale, zp)
    mae = float((x - x_hat).abs().mean())
    return x_hat, mae, scale, zp


def _per_channel_quant_error(w: torch.Tensor) -> tuple[float, float]:
    """重み行列 (out, in) を per-tensor / per-channel で量子化したときの MAE を比較。"""
    # per-tensor: 全体で 1 つの scale（対称）
    _, mae_tensor, _, _ = _roundtrip_error(w, symmetric=True)

    # per-channel: 出力チャネル(行)ごとに scale を決める
    errs = []
    for row in w:
        _, mae_row, _, _ = _roundtrip_error(row, symmetric=True)
        errs.append(mae_row * row.numel())
    mae_channel = float(sum(errs) / w.numel())
    return mae_tensor, mae_channel


def main() -> None:
    quiet_warnings()
    set_seed(0)
    out = ensure_output_dir()

    # === 1. scale / zero-point を手で求める ============================
    # ReLU 後を想定した非負＆非対称な分布（0〜6 付近）。
    x = torch.tensor([0.0, 0.7, 1.3, 2.1, 3.4, 4.8, 5.9, 6.0])
    fmt = lambda t: "[" + ", ".join(f"{v:.2f}" for v in t.tolist()) + "]"  # noqa: E731
    print("[1] 量子化の正体 — 実数を int8 の格子に丸める")
    print(f"    入力 x        = {fmt(x)}")
    for sym in (False, True):
        q, scale, zp = quantize_tensor(x, num_bits=8, symmetric=sym)
        x_hat = dequantize_tensor(q, scale, zp)
        name = "対称  " if sym else "非対称"
        print(f"    [{name}] scale={scale:.4f} zero_point={zp:>4d}  "
              f"q={q.to(torch.int32).tolist()}")
        print(f"             復元 x̂ = {fmt(x_hat)}  "
              f"MAE={(x - x_hat).abs().mean():.4f}")
    print("    → 非対称は zero_point で 0.0 を整数にぴったり合わせられ、片側分布で誤差が小さい。\n")

    # === 2. ビット幅と誤差 =============================================
    print("[2] ビット幅を下げるほど刻みが粗くなり誤差が増える")
    g = torch.Generator().manual_seed(1)
    sig = torch.randn(4096, generator=g)
    for bits in (8, 4, 2):
        q, scale, zp = quantize_tensor(sig, num_bits=bits, symmetric=True)
        x_hat = dequantize_tensor(q, scale, zp)
        print(f"    {bits}-bit: 段階数={2 ** bits:>3d}  scale={scale:.4f}  "
              f"MAE={(sig - x_hat).abs().mean():.4f}")
    print("    → int8(256段階) は fp32 をほぼ保てるが、int4/int2 は急速に荒くなる。\n")

    # === 3. per-tensor vs per-channel ==================================
    # チャネルごとにスケールが大きく違う重みを作る（行 i の振幅 = 10^i 倍）。
    rows = []
    base = torch.randn(6, 32, generator=torch.Generator().manual_seed(2))
    for i in range(6):
        rows.append(base[i] * (10.0 ** (i - 2)))  # 0.01 倍 〜 1000 倍まで桁が違う
    w = torch.stack(rows)
    mae_tensor, mae_channel = _per_channel_quant_error(w)
    print("[3] per-tensor vs per-channel（チャネルでスケールが桁違いの重み）")
    print(f"    per-tensor  MAE = {mae_tensor:.5f}  （全体で1つの scale → 小振幅チャネルが潰れる）")
    print(f"    per-channel MAE = {mae_channel:.5f}  （行ごとに scale → 各チャネルを活かせる）")
    print(f"    → per-channel で誤差が約 {mae_tensor / max(mae_channel, 1e-9):.0f} 倍改善。"
          "実際の重み量子化は per-channel が既定。\n")

    # === 4. 図に保存 ===================================================
    _save_figure(out, sig)
    print("[4] 図を保存:", (out / "01_quantization_levels.png").name)

    print("\n[まとめ]")
    print("  - 量子化 = scale と zero_point で実数↔整数を往復する写像。")
    print("  - 対称(重み向き) / 非対称(activation向き) を使い分ける。")
    print("  - per-channel はチャネル間スケール差を吸収し、per-tensor より誤差が小さい。")


def _save_figure(out: pathlib.Path, signal: torch.Tensor) -> None:
    """連続値→量子化格子の可視化と、ビット幅ごとの誤差を保存する。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))

    # 左: 連続な直線を int8/int4 で量子化した階段関数
    xs = torch.linspace(-3, 3, 400)
    ax = axes[0]
    ax.plot(xs.numpy(), xs.numpy(), color="gray", lw=1.2, label="fp32 (continuous)")
    for bits, color in [(8, "tab:blue"), (4, "tab:orange"), (2, "tab:red")]:
        q, scale, zp = quantize_tensor(xs, num_bits=bits, symmetric=True)
        x_hat = dequantize_tensor(q, scale, zp)
        ax.plot(xs.numpy(), x_hat.numpy(), lw=1.2, color=color, label=f"int{bits} (staircase)")
    ax.set_title("quantization = rounding to a grid")
    ax.set_xlabel("real value")
    ax.set_ylabel("dequantized value")
    ax.legend(fontsize=8)

    # 右: ビット幅ごとの量子化誤差(MAE)
    ax = axes[1]
    bits_list = [8, 7, 6, 5, 4, 3, 2]
    maes = []
    for bits in bits_list:
        q, scale, zp = quantize_tensor(signal, num_bits=bits, symmetric=True)
        x_hat = dequantize_tensor(q, scale, zp)
        maes.append(float((signal - x_hat).abs().mean()))
    ax.plot(bits_list, maes, "o-", color="tab:purple")
    ax.invert_xaxis()
    ax.set_title("fewer bits -> larger error")
    ax.set_xlabel("bit width")
    ax.set_ylabel("mean abs error")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out / "01_quantization_levels.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
