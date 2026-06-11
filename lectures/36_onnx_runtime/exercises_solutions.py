"""第36回 演習の模範解答（全 PASS）。

実行:
  uv run python lectures/36_onnx_runtime/exercises_solutions.py

exercises.py の採点ランナー grade() をそのまま import して使い、
ここでは各 exN_*() を「正しく実装した版」に差し替えて採点する。
"""

from __future__ import annotations

import collections
import pathlib

import numpy as np
import torch
from torch import nn

import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from exercises import grade  # noqa: E402  （採点ランナーを再利用）

OPSET = 18


# =====================================================================
# 模範解答
# =====================================================================

def ex1_max_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    """2 配列の最大絶対誤差。ONNX 化の正しさ検証の中核指標。"""
    return float(np.max(np.abs(a - b)))


def ex2_outputs_agree(a: np.ndarray, b: np.ndarray, atol: float, rtol: float) -> bool:
    """atol/rtol 内で一致するか。デプロイ前の必須チェック。"""
    return bool(np.allclose(a, b, atol=atol, rtol=rtol))


def ex3_export_onnx(model: nn.Module, example: torch.Tensor, path: str | pathlib.Path) -> pathlib.Path:
    """TorchScript ベース exporter（dynamo=False）で動的バッチ ONNX を書き出す。"""
    model.eval()
    path = pathlib.Path(path)
    torch.onnx.export(
        model, (example,), str(path), dynamo=False, opset_version=OPSET,
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
    )
    return path


def ex4_make_session_options(threads: int):
    """全グラフ最適化・intra スレッド固定の SessionOptions。"""
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.intra_op_num_threads = threads
    return so


def ex5_predict_topk(session, x_np: np.ndarray, k: int) -> np.ndarray:
    """セッション推論 → 各行のロジット降順 top-k index。"""
    name = session.get_inputs()[0].name
    logits = session.run(None, {name: x_np.astype(np.float32)})[0]
    return np.argsort(-logits, axis=1)[:, :k]


def ex6_quantize_int8(fp32_path: str | pathlib.Path, int8_path: str | pathlib.Path) -> pathlib.Path:
    """動的量子化(int8, QUInt8)。CPU で手堅く効くサイズ削減。"""
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(str(fp32_path), str(int8_path), weight_type=QuantType.QUInt8)
    return pathlib.Path(int8_path)


def ex7_op_histogram(path: str | pathlib.Path) -> dict[str, int]:
    """ONNX グラフの op_type ヒストグラム。"""
    import onnx

    model = onnx.load(str(path))
    return dict(collections.Counter(n.op_type for n in model.graph.node))


def ex8_size_reduction_ratio(fp32_path: str | pathlib.Path, int8_path: str | pathlib.Path) -> float:
    """サイズ削減比 = fp32 バイト / int8 バイト。"""
    fp32 = pathlib.Path(fp32_path).stat().st_size
    int8 = pathlib.Path(int8_path).stat().st_size
    return fp32 / int8


def ex9_choose_deployment(metrics: dict, max_acc_drop: float) -> str:
    """精度劣化が許容内なら小さい int8、超えるなら fp32。"""
    drop = metrics["fp32"]["acc"] - metrics["int8"]["acc"]
    return "int8" if drop <= max_acc_drop else "fp32"


def ex10_dynamic_batch_shapes(session, batches: list[int]) -> list[tuple]:
    """動的バッチで各サイズの出力 shape を集める。"""
    name = session.get_inputs()[0].name
    shapes: list[tuple] = []
    for b in batches:
        xb = np.random.randn(b, 3, 16, 16).astype(np.float32)
        out = session.run(None, {name: xb})[0]
        shapes.append(tuple(out.shape))
    return shapes


def solution_funcs() -> dict:
    return {
        "ex1": ex1_max_abs_diff,
        "ex2": ex2_outputs_agree,
        "ex3": ex3_export_onnx,
        "ex4": ex4_make_session_options,
        "ex5": ex5_predict_topk,
        "ex6": ex6_quantize_int8,
        "ex7": ex7_op_histogram,
        "ex8": ex8_size_reduction_ratio,
        "ex9": ex9_choose_deployment,
        "ex10": ex10_dynamic_batch_shapes,
    }


if __name__ == "__main__":
    grade(solution_funcs())
