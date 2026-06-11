"""第35回 演習の模範解答（全 PASS）。

実行:
  uv run python lectures/35_quantization_pruning/exercises_solutions.py

exercises.py の採点ランナー grade() をそのまま import して使い、
ここでは各 exN_*() を「正しく実装した版」に差し替えて採点する。
モデル DL も学習も不要・決定的なので一瞬で全 PASS する。
"""

from __future__ import annotations

import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from exercises import grade  # noqa: E402  （採点ランナーを再利用）


# =====================================================================
# 模範解答
# =====================================================================

def ex1_affine_scale_zero_point(x_min: float, x_max: float) -> tuple[float, int]:
    """非対称(affine) uint8 の scale / zero_point。0.0 を整数に正確に対応させる。"""
    qmin, qmax = 0, 255
    if x_max <= x_min:
        return 1.0, 0
    scale = (x_max - x_min) / (qmax - qmin)
    zero_point = int(round(qmin - x_min / scale))
    zero_point = max(qmin, min(qmax, zero_point))
    return scale, zero_point


def ex2_quantize_dequantize(x: torch.Tensor, scale: float, zero_point: int) -> torch.Tensor:
    """uint8 へ量子化 → 逆量子化（量子化誤差込みの復元値）。"""
    q = torch.clamp(torch.round(x / scale) + zero_point, 0, 255)
    return scale * (q - zero_point)


def ex3_roundtrip_mae(x: torch.Tensor, num_bits: int) -> float:
    """対称 num_bits 量子化の往復 平均絶対誤差。"""
    qmax = 2 ** (num_bits - 1) - 1
    max_abs = float(x.abs().max())
    scale = max_abs / qmax if max_abs > 0 else 1.0
    q = torch.clamp(torch.round(x / scale), -qmax - 1, qmax)
    x_hat = scale * q
    return float((x - x_hat).abs().mean())


def ex4_per_channel_scales(w: torch.Tensor) -> torch.Tensor:
    """per-channel(行ごと) 対称int8 scale = max(|row|)/127（ゼロ行は 1.0）。"""
    max_abs = w.abs().amax(dim=1)  # 行ごとの最大絶対値 (out,)
    scale = max_abs / 127.0
    scale[scale == 0] = 1.0
    return scale


def ex5_weight_sparsity(weight: torch.Tensor) -> float:
    """スパース率 = 0 の割合。"""
    return float((weight == 0).float().mean())


def ex6_magnitude_prune(weight: torch.Tensor, amount: float) -> torch.Tensor:
    """非構造化(magnitude)枝刈り: |w| の小さい amount 割合を 0 に（形は不変）。"""
    if amount <= 0:
        return weight.clone()
    if amount >= 1:
        return torch.zeros_like(weight)
    thr = torch.quantile(weight.abs().flatten(), amount)
    mask = weight.abs() > thr  # 閾値より大きい重みだけ残す（密のまま）
    return weight * mask


def ex7_int8_bytes_saved(num_params: int) -> int:
    """fp32(4byte) → int8(1byte) の節約バイト数。"""
    return num_params * (4 - 1)


def ex8_structured_keep_rows(weight: torch.Tensor, keep_frac: float) -> torch.Tensor:
    """構造化枝刈りで残す出力ニューロン(行)の index（L2 上位・昇順）。"""
    out = weight.shape[0]
    keep = max(1, int(round(out * keep_frac)))
    scores = weight.flatten(1).norm(dim=1)  # 各行(出力ニューロン)の大きさ
    idx = torch.topk(scores, keep).indices
    return torch.sort(idx).values


def ex9_recommend_method(model_kind: str, has_calibration: bool, accuracy_drop_ok: bool) -> str:
    """状況から量子化手法を選ぶ意思決定。"""
    if model_kind == "linear":
        return "dynamic"
    if has_calibration and accuracy_drop_ok:
        return "static"
    return "qat"


def solution_funcs() -> dict:
    return {
        "ex1": ex1_affine_scale_zero_point,
        "ex2": ex2_quantize_dequantize,
        "ex3": ex3_roundtrip_mae,
        "ex4": ex4_per_channel_scales,
        "ex5": ex5_weight_sparsity,
        "ex6": ex6_magnitude_prune,
        "ex7": ex7_int8_bytes_saved,
        "ex8": ex8_structured_keep_rows,
        "ex9": ex9_recommend_method,
    }


if __name__ == "__main__":
    grade(solution_funcs())
