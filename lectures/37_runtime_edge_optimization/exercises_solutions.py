"""第37回 演習の模範解答(全 PASS)。

実行:
  uv run python lectures/37_runtime_edge_optimization/exercises_solutions.py

exercises.py の採点ランナー grade() をそのまま import して使い、
各 exN_*() を「正しく実装した版」に差し替えて採点する。小モデル・合成データのみで
決定的なので数秒で全 PASS する。
"""

from __future__ import annotations

import pathlib
import sys
import warnings

import numpy as np
import torch
from torch import nn

warnings.filterwarnings("ignore")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from exercises import grade  # noqa: E402  (採点ランナーを再利用)


# =====================================================================
# 模範解答
# =====================================================================
def ex1_percentile(sorted_samples: list[float], q: float) -> float:
    """昇順済みリストの q パーセンタイル(線形補間)。"""
    n = len(sorted_samples)
    if n == 0:
        return float("nan")
    if n == 1:
        return sorted_samples[0]
    pos = (q / 100.0) * (n - 1)
    lo = int(pos)
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return sorted_samples[lo] * (1 - frac) + sorted_samples[hi] * frac


def ex2_speedup(baseline_ms: float, candidate_ms: float) -> float:
    """速度比 = baseline / candidate(>1 で高速化)。0除算は nan。"""
    return baseline_ms / candidate_ms if candidate_ms > 0 else float("nan")


def ex3_throughput(num_images: int, elapsed_sec: float) -> float:
    """スループット = 総枚数 / 総秒数(images/sec)。0除算は nan。"""
    return num_images / elapsed_sec if elapsed_sec > 0 else float("nan")


def ex4_top1_agreement(logits: np.ndarray, ref_logits: np.ndarray) -> float:
    """基準に対する top1 一致率(argmax の一致割合)。"""
    return float((logits.argmax(axis=1) == ref_logits.argmax(axis=1)).mean())


def ex5_max_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    """最大絶対誤差 max(|a-b|)(atol 検証の素)。"""
    return float(np.abs(a - b).max())


def ex6_compression_ratio(fp32_mb: float, quant_mb: float) -> float:
    """圧縮率 = fp32 / 量子化後(int8 で ~4 が目安)。0除算は nan。"""
    return fp32_mb / quant_mb if quant_mb > 0 else float("nan")


def ex7_steady_median(times_ms: list[float], warmup: int) -> float:
    """先頭 warmup 件を除外した残りの中央値(初回コスト除去)。"""
    steady = times_ms[warmup:]
    return float(np.median(steady)) if steady else float("nan")


def ex8_choose_runtime(rows: list[dict], max_latency_ms: float, min_top1: float) -> str:
    """制約(精度保持・レイテンシ)を満たす中で p50 最小のランタイム名。無ければ none。"""
    feasible = [
        r for r in rows if r["top1"] >= min_top1 and r["p50_ms"] <= max_latency_ms
    ]
    if not feasible:
        return "none"
    return min(feasible, key=lambda r: r["p50_ms"])["name"]


def ex9_torchscript_roundtrip(model: nn.Module, x: torch.Tensor) -> float:
    """trace → eager との最大絶対誤差(変換の正しさ検証)。"""
    model.eval()
    with torch.inference_mode():
        ref = model(x)
        traced = torch.jit.trace(model, x)
        out = traced(x)
        return float((out - ref).abs().max().item())


def ex10_onnx_export_match(model: nn.Module, x: torch.Tensor, tmp_path: str) -> float:
    """ONNX 化 → onnxruntime 実行 → torch 出力との最大絶対誤差。"""
    import onnxruntime as ort

    model.eval()
    with torch.inference_mode():
        ref = model(x).numpy()
        torch.onnx.export(
            model,
            (x,),
            tmp_path,
            input_names=["input"],
            output_names=["logits"],
            opset_version=17,
            dynamo=False,  # onnxscript 非依存の安定経路
        )
    sess = ort.InferenceSession(tmp_path, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    out = sess.run(None, {name: x.numpy()})[0]
    return float(np.abs(out - ref).max())


def solution_funcs() -> dict:
    return {
        "ex1": ex1_percentile,
        "ex2": ex2_speedup,
        "ex3": ex3_throughput,
        "ex4": ex4_top1_agreement,
        "ex5": ex5_max_abs_diff,
        "ex6": ex6_compression_ratio,
        "ex7": ex7_steady_median,
        "ex8": ex8_choose_runtime,
        "ex9": ex9_torchscript_roundtrip,
        "ex10": ex10_onnx_export_match,
    }


if __name__ == "__main__":
    grade(solution_funcs())
