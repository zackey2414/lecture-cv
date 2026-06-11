"""第36回 演習問題（ONNX エクスポート / onnxruntime / グラフ最適化 / 動的量子化）。

使い方:
  1. 各 exN_*() の中の TODO を自分で実装する（最初は NotImplementedError で FAIL）。
  2. 自己採点（未実装でも例外で落とさず PASS/FAIL を表示し、必ず exit 0）:
         uv run python lectures/36_onnx_runtime/exercises.py
  3. 全問の模範解答（全 PASS）はこちら:
         uv run python lectures/36_onnx_runtime/exercises_solutions.py

テーマ（易→難）:
  1 最大絶対誤差        2 数値一致 allclose      3 ONNX エクスポート(dynamo=False, 動的batch)
  4 SessionOptions 設定 5 top-k 予測             6 動的量子化(int8)
  7 op ヒストグラム      8 サイズ削減比           9 デプロイ意思決定
  10 動的バッチ推論

採点はランダム初期化の小モデル（学習不要・seed 固定）で決定的に行うので速く再現的。
本物のフローは 01〜05・mini_project.py を参照。
"""

from __future__ import annotations

import pathlib
import tempfile

import numpy as np
import torch
from torch import nn

OPSET = 18


# =====================================================================
# 演習（ここを実装する）
# =====================================================================

def ex1_max_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    """演習1: 2 つの配列の最大絶対誤差 max(|a-b|) を float で返す。

    ONNX 化の正しさは「torch と onnxruntime の出力の最大絶対誤差」で測るのが基本。
    """
    # TODO: float(np.max(np.abs(a - b))) を返す
    raise NotImplementedError


def ex2_outputs_agree(a: np.ndarray, b: np.ndarray, atol: float, rtol: float) -> bool:
    """演習2: a と b が atol/rtol 内で一致するか（bool）。np.allclose を使う。

    エクスポート後は必ずこの判定をしてからデプロイする（壊れた ONNX を出さない）。
    """
    # TODO: bool(np.allclose(a, b, atol=atol, rtol=rtol)) を返す
    raise NotImplementedError


def ex3_export_onnx(model: nn.Module, example: torch.Tensor, path: str | pathlib.Path) -> pathlib.Path:
    """演習3: model を ONNX へエクスポートして path を返す。

    要件:
      - dynamo=False（onnxscript 不要の旧経路）, opset_version=OPSET
      - input_names=['input'], output_names=['logits']
      - バッチ次元(0)を動的軸 'batch' にする（dynamic_axes）
    """
    # TODO: model.eval() の上で torch.onnx.export(...) を呼び、path を返す
    raise NotImplementedError


def ex4_make_session_options(threads: int):
    """演習4: グラフ最適化を全有効・intra スレッド数を threads にした SessionOptions を返す。

    ヒント: ort.SessionOptions() を作り、graph_optimization_level に
    ort.GraphOptimizationLevel.ORT_ENABLE_ALL、intra_op_num_threads に threads を設定。
    """
    # TODO: SessionOptions を構成して返す
    raise NotImplementedError


def ex5_predict_topk(session, x_np: np.ndarray, k: int) -> np.ndarray:
    """演習5: セッションに x_np を流し、各行の top-k クラス index を (N,k) で返す。

    ロジットの降順に並べた上位 k 個の index。session.get_inputs()[0].name を使って入力名を取る。
    """
    # TODO: 出力ロジットを取り、np.argsort(-logits, axis=1)[:, :k] を返す
    raise NotImplementedError


def ex6_quantize_int8(fp32_path: str | pathlib.Path, int8_path: str | pathlib.Path) -> pathlib.Path:
    """演習6: fp32 の ONNX を動的量子化(int8)して int8_path に書き、その path を返す。

    onnxruntime.quantization.quantize_dynamic を使い、weight_type は QuantType.QUInt8（CPU は U8X8）。
    """
    # TODO: quantize_dynamic(str(fp32_path), str(int8_path), weight_type=QuantType.QUInt8)
    raise NotImplementedError


def ex7_op_histogram(path: str | pathlib.Path) -> dict[str, int]:
    """演習7: ONNX を読み、op_type ごとのノード数の辞書を返す（例 {'Conv':1,'Gemm':2,...}）。

    onnx.load → model.graph.node を走査して op_type を数える。
    """
    # TODO: collections.Counter で op_type を集計して dict で返す
    raise NotImplementedError


def ex8_size_reduction_ratio(fp32_path: str | pathlib.Path, int8_path: str | pathlib.Path) -> float:
    """演習8: サイズ削減比 = fp32 のバイト数 / int8 のバイト数 を float で返す。

    1.0 より大きいほど縮んでいる。pathlib.Path(...).stat().st_size を使う。
    """
    # TODO: fp32_bytes / int8_bytes を返す
    raise NotImplementedError


def ex9_choose_deployment(metrics: dict, max_acc_drop: float) -> str:
    """演習9: 精度劣化許容量から 'fp32'/'int8' を選ぶ。

    metrics = {'fp32': {'acc':..,'size_mb':..}, 'int8': {'acc':..,'size_mb':..}}
    ルール: (fp32.acc - int8.acc) <= max_acc_drop なら小さい 'int8' を選ぶ。さもなくば 'fp32'。
    """
    # TODO: 精度劣化を計算して 'int8' か 'fp32' を返す
    raise NotImplementedError


def ex10_dynamic_batch_shapes(session, batches: list[int]) -> list[tuple]:
    """演習10: 動的バッチの session に各 batch サイズで (b,3,16,16) を流し、出力 shape の list を返す。

    export 時と違うバッチでも通る（dynamic_axes のおかげ）ことを確認する演習。
    """
    # TODO: 各 b について乱数入力を作り run、出力 shape をタプルで集める
    raise NotImplementedError


# =====================================================================
# 採点ランナー（exercises_solutions.py からも import して再利用）
# =====================================================================

class GradeNet(nn.Module):
    """採点用の固定小モデル（学習しない・seed 固定の乱数重み）。Conv と Linear を含む。"""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 4, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 16->8
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(4 * 8 * 8, 32), nn.ReLU(), nn.Linear(32, 10)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def _fixed_model() -> GradeNet:
    """seed 固定の決定的なモデルを返す。"""
    torch.manual_seed(0)
    m = GradeNet()
    m.eval()
    return m


def _ref_export(path: pathlib.Path) -> pathlib.Path:
    """採点が独立に使う『正解の』エクスポート（exercise の ex3 に依存しないため）。"""
    m = _fixed_model()
    example = torch.randn(1, 3, 16, 16)
    torch.onnx.export(
        m, (example,), str(path), dynamo=False, opset_version=OPSET,
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
    )
    return path


def grade(funcs: dict) -> bool:
    """funcs（名前→関数）を採点して PASS/FAIL を表示。全 PASS なら True。"""
    import warnings

    warnings.filterwarnings("ignore")
    import onnxruntime as ort

    results: list[tuple[str, bool, str]] = []
    tmp = pathlib.Path(tempfile.mkdtemp())
    ref_path = _ref_export(tmp / "ref.onnx")  # 採点共通の参照 ONNX
    model = _fixed_model()
    x = torch.randn(6, 3, 16, 16)
    x_np = x.numpy().astype(np.float32)
    with torch.inference_mode():
        ref_logits = model(x).numpy()

    def check(name: str, fn) -> None:
        try:
            ok, detail = fn()
            results.append((name, bool(ok), detail))
        except NotImplementedError:
            results.append((name, False, "未実装（TODOを埋めてください）"))
        except Exception as e:  # noqa: BLE001
            results.append((name, False, f"例外: {type(e).__name__}: {e}"))

    def c1():
        v = funcs["ex1"](np.array([1.0, 2.0, 5.0]), np.array([1.0, 2.0, 1.0]))
        return (abs(v - 4.0) < 1e-9, f"max|a-b|={v:.3f}")

    def c2():
        a = np.array([1.0, 2.0])
        b = np.array([1.0 + 1e-6, 2.0 - 1e-6])
        ok1 = funcs["ex2"](a, b, 1e-4, 1e-3)
        ok2 = funcs["ex2"](a, np.array([1.5, 2.0]), 1e-4, 1e-3)
        return (ok1 is True and ok2 is False, f"近い={ok1} 遠い={ok2}")

    def c3():
        p = funcs["ex3"](_fixed_model(), torch.randn(1, 3, 16, 16), tmp / "cand.onnx")
        import onnx
        onnx.checker.check_model(onnx.load(str(p)))
        sess = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
        got = sess.run(None, {sess.get_inputs()[0].name: x_np})[0]
        return (np.allclose(got, ref_logits, atol=1e-4, rtol=1e-3), "checker OK & torch一致")

    def c4():
        so = funcs["ex4"](2)
        ok = (so.graph_optimization_level == ort.GraphOptimizationLevel.ORT_ENABLE_ALL
              and so.intra_op_num_threads == 2)
        return (ok, f"ENABLE_ALL & threads={so.intra_op_num_threads}")

    def c5():
        sess = ort.InferenceSession(str(ref_path), providers=["CPUExecutionProvider"])
        top = funcs["ex5"](sess, x_np, 3)
        exp = np.argsort(-ref_logits, axis=1)[:, :3]
        top = np.asarray(top)
        return (top.shape == (6, 3) and np.array_equal(top, exp), f"top3 shape={top.shape}")

    def c6():
        ip = funcs["ex6"](ref_path, tmp / "cand_int8.onnx")
        sess = ort.InferenceSession(str(ip), providers=["CPUExecutionProvider"])
        got = sess.run(None, {sess.get_inputs()[0].name: x_np})[0]
        smaller = pathlib.Path(ip).stat().st_size < ref_path.stat().st_size
        return (smaller and got.shape == (6, 10), f"int8 smaller={smaller}")

    def c7():
        hist = funcs["ex7"](ref_path)
        ok = hist.get("Conv") == 1 and hist.get("Gemm") == 2 and hist.get("MaxPool") == 1
        return (ok, f"hist={hist}")

    def c8():
        i8 = tmp / "ratio_int8.onnx"
        from onnxruntime.quantization import QuantType, quantize_dynamic
        quantize_dynamic(str(ref_path), str(i8), weight_type=QuantType.QUInt8)
        r = funcs["ex8"](ref_path, i8)
        true_r = ref_path.stat().st_size / i8.stat().st_size
        return (abs(r - true_r) < 1e-6 and r > 1.0, f"ratio={r:.2f}x")

    def c9():
        metrics = {"fp32": {"acc": 0.90, "size_mb": 4.0}, "int8": {"acc": 0.88, "size_mb": 1.1}}
        a = funcs["ex9"](metrics, max_acc_drop=0.05)   # 劣化0.02 <= 0.05 → int8
        b = funcs["ex9"](metrics, max_acc_drop=0.01)   # 劣化0.02 > 0.01 → fp32
        return (a == "int8" and b == "fp32", f"許容0.05->{a} / 0.01->{b}")

    def c10():
        sess = ort.InferenceSession(str(ref_path), providers=["CPUExecutionProvider"])
        shapes = funcs["ex10"](sess, [1, 4, 8])
        shapes = [tuple(s) for s in shapes]
        exp = [(1, 10), (4, 10), (8, 10)]
        return (shapes == exp, f"shapes={shapes}")

    check("ex1_max_abs_diff", c1)
    check("ex2_outputs_agree", c2)
    check("ex3_export_onnx", c3)
    check("ex4_make_session_options", c4)
    check("ex5_predict_topk", c5)
    check("ex6_quantize_int8", c6)
    check("ex7_op_histogram", c7)
    check("ex8_size_reduction_ratio", c8)
    check("ex9_choose_deployment", c9)
    check("ex10_dynamic_batch_shapes", c10)

    print("=== 採点結果 ===")
    all_ok = True
    for name, ok, detail in results:
        mark = "PASS" if ok else "FAIL"
        all_ok = all_ok and ok
        print(f"  [{mark}] {name:28s} {detail}")
    print("\nALL PASS 🎉" if all_ok else "\nまだ未達の演習があります。TODO を埋めましょう。")
    return all_ok


def current_funcs() -> dict:
    """このファイルで定義された ex* を採点ランナーへ渡す形にまとめる。"""
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
    grade(current_funcs())
