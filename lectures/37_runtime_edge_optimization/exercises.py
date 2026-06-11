"""第37回 演習問題(ランタイム/エッジ最適化 — 計測・変換・量子化・意思決定)。

使い方:
  1. 各 exN_*() の TODO を自分で実装する(最初は NotImplementedError で FAIL)。
  2. 自己採点(未実装でも例外で落とさず PASS/FAIL を表示し、必ず exit 0):
         uv run python lectures/37_runtime_edge_optimization/exercises.py
  3. 全問の模範解答(全 PASS):
         uv run python lectures/37_runtime_edge_optimization/exercises_solutions.py

テーマ(易→難):
  1 分位点(p50/p99)   2 速度比     3 スループット   4 top1一致率(精度保持)
  5 最大絶対誤差(atol) 6 圧縮率     7 ウォームアップ除外の中央値
  8 ランタイム選択(意思決定)  9 TorchScript trace 往復一致  10 ONNX エクスポート数値一致

採点は小さな合成データ・小モデルのみで決定的に行う(重い DL 不要・数秒)。
本物のベンチは 01〜05・mini_project.py を参照。
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


# =====================================================================
# 演習(ここを実装する)
# =====================================================================
def ex1_percentile(sorted_samples: list[float], q: float) -> float:
    """演習1: 昇順済みリストの q パーセンタイル(線形補間)を返す。

    レイテンシは平均でなく p50(中央値)/p99(テール)で見るのが鉄則。
    pos = (q/100)*(n-1) を計算し、前後インデックスを frac で線形補間する。
    n==1 なら唯一要素、空なら nan。
    """
    # TODO: pos=(q/100)*(n-1) → lo=int(pos), hi=min(lo+1,n-1), frac=pos-lo → 線形補間
    raise NotImplementedError


def ex2_speedup(baseline_ms: float, candidate_ms: float) -> float:
    """演習2: 速度比 = baseline_ms / candidate_ms を返す(>1 なら高速化)。

    candidate_ms<=0 のときは 0除算回避で nan を返す。
    """
    # TODO: candidate_ms>0 なら baseline/candidate、さもなくば nan
    raise NotImplementedError


def ex3_throughput(num_images: int, elapsed_sec: float) -> float:
    """演習3: スループット(images/sec)= 総枚数 / 総秒数 を返す。

    elapsed_sec<=0 のときは nan(0除算回避)。
    """
    # TODO: elapsed_sec>0 なら num_images/elapsed_sec、さもなくば nan
    raise NotImplementedError


def ex4_top1_agreement(logits: np.ndarray, ref_logits: np.ndarray) -> float:
    """演習4: 基準 ref に対する top1 一致率(argmax の一致割合, 0〜1)を返す。

    量子化/変換の『精度保持』は fp32 基準と argmax がどれだけ一致するかで測る。
    各行 argmax を比較し、一致したサンプルの割合を返す。
    """
    # TODO: logits.argmax(axis=1) と ref_logits.argmax(axis=1) の一致割合
    raise NotImplementedError


def ex5_max_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    """演習5: 2配列の最大絶対誤差 max(|a-b|) を返す(atol 検証の素)。

    ONNX/TorchScript 変換後は『torch 出力と一致するか』をこの誤差で検証する。
    """
    # TODO: float(np.abs(a-b).max())
    raise NotImplementedError


def ex6_compression_ratio(fp32_mb: float, quant_mb: float) -> float:
    """演習6: 圧縮率 = fp32サイズ / 量子化後サイズ を返す(int8 なら ~4 が目安)。

    quant_mb<=0 のときは nan(0除算回避)。
    """
    # TODO: quant_mb>0 なら fp32_mb/quant_mb、さもなくば nan
    raise NotImplementedError


def ex7_steady_median(times_ms: list[float], warmup: int) -> float:
    """演習7: 先頭 warmup 件(初回コンパイル等)を捨てた残りの中央値を返す。

    計測ファースト原則: 初回には JIT/コンパイル/キャッシュmiss の代金が乗るので
    必ずウォームアップ分を除外してから中央値を取る。残りが空なら nan。
    """
    # TODO: times_ms[warmup:] を取り、空なら nan、さもなくば中央値(np.median)
    raise NotImplementedError


def ex8_choose_runtime(rows: list[dict], max_latency_ms: float, min_top1: float) -> str:
    """演習8: 制約を満たす中で p50 最小のランタイム名を返す(意思決定)。

    rows は [{"name":..., "p50_ms":..., "top1":...}, ...]。
    条件: top1 >= min_top1 かつ p50_ms <= max_latency_ms。
    満たすものが無ければ "none"。複数あれば p50_ms 最小を選ぶ。
    """
    # TODO: 制約フィルタ → 空なら "none" → さもなくば p50_ms 最小の name
    raise NotImplementedError


def ex9_torchscript_roundtrip(model: nn.Module, x: torch.Tensor) -> float:
    """演習9: model を trace → 同じ入力で eager と TorchScript の最大絶対誤差を返す。

    手順: model.eval() 前提 → torch.jit.trace(model, x) → inference_mode で両者を実行
          → max(|ts(x) - model(x)|) を float で返す(0 に近ければ変換成功)。
    """
    # TODO: trace して eager 出力との最大絶対誤差を返す
    raise NotImplementedError


def ex10_onnx_export_match(model: nn.Module, x: torch.Tensor, tmp_path: str) -> float:
    """演習10: model を ONNX 化 → onnxruntime で実行し、torch 出力との最大絶対誤差を返す。

    手順:
      - torch.onnx.export(model, (x,), tmp_path, opset_version=17, dynamo=False,
                          input_names=["input"], output_names=["logits"])
      - onnxruntime.InferenceSession(tmp_path, providers=["CPUExecutionProvider"])
      - sess.run(None, {入力名: x.numpy()})[0] と model(x) の最大絶対誤差を返す。
    """
    # TODO: ONNX export → ort で実行 → torch 出力との最大絶対誤差
    raise NotImplementedError


# =====================================================================
# 自己採点ランナー(exercises_solutions.py からも import して再利用)
# =====================================================================
def _tiny_model(seed: int = 0) -> nn.Module:
    """採点用の小さな決定的モデル(高速・DL不要)。conv→relu→pool→flatten→linear。"""
    torch.manual_seed(seed)
    model = nn.Sequential(
        nn.Conv2d(3, 4, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((4, 4)),
        nn.Flatten(),
        nn.Linear(4 * 4 * 4, 10),
    )
    model.eval()
    return model


def grade(funcs: dict) -> bool:
    """funcs(名前→関数)を採点して PASS/FAIL を表示。全PASSなら True。"""
    results: list[tuple[str, bool, str]] = []

    def check(name: str, fn) -> None:
        try:
            ok, detail = fn()
            results.append((name, bool(ok), detail))
        except NotImplementedError:
            results.append((name, False, "未実装(TODOを埋めてください)"))
        except Exception as e:  # noqa: BLE001
            results.append((name, False, f"例外: {type(e).__name__}: {e}"))

    def c1():
        s = [1.0, 2.0, 3.0, 4.0, 5.0]
        p50 = funcs["ex1"](s, 50)
        p99 = funcs["ex1"](s, 99)
        return (abs(p50 - 3.0) < 1e-9 and abs(p99 - 4.96) < 1e-6, f"p50={p50:.3f}, p99={p99:.3f}")

    def c2():
        v = funcs["ex2"](10.0, 5.0)
        bad = funcs["ex2"](10.0, 0.0)
        return (abs(v - 2.0) < 1e-9 and np.isnan(bad), f"speedup={v:.2f}, zero→{bad}")

    def c3():
        v = funcs["ex3"](160, 2.0)
        return (abs(v - 80.0) < 1e-9, f"throughput={v:.1f} img/s")

    def c4():
        ref = np.array([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]], dtype=np.float32)
        logits = np.array([[0.2, 0.8], [0.1, 0.9], [0.4, 0.6]], dtype=np.float32)
        # 行0: 1==1 OK, 行1: 0 vs 1 NG, 行2: 1==1 OK → 2/3
        v = funcs["ex4"](logits, ref)
        return (abs(v - 2 / 3) < 1e-6, f"top1一致={v:.3f}")

    def c5():
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.0, 2.5, 2.0])
        v = funcs["ex5"](a, b)
        return (abs(v - 1.0) < 1e-9, f"max|a-b|={v:.3f}")

    def c6():
        v = funcs["ex6"](46.8, 11.7)
        bad = funcs["ex6"](46.8, 0.0)
        return (abs(v - 4.0) < 1e-6 and np.isnan(bad), f"compression={v:.2f}x")

    def c7():
        times = [50.0, 12.0, 10.0, 11.0, 9.0]  # 先頭2件はウォームアップ(初回コンパイル等)
        v = funcs["ex7"](times, warmup=2)  # 残り [10,11,9] の中央値=10
        return (abs(v - 10.0) < 1e-9, f"steady median={v:.2f}")

    def c8():
        rows = [
            {"name": "eager", "p50_ms": 9.0, "top1": 1.0},
            {"name": "onnx_fp32", "p50_ms": 5.0, "top1": 1.0},
            {"name": "onnx_int8", "p50_ms": 11.0, "top1": 0.96},
        ]
        # 制約 lat<=8, top1>=0.99 → onnx_fp32 のみ該当
        v = funcs["ex8"](rows, max_latency_ms=8.0, min_top1=0.99)
        none = funcs["ex8"](rows, max_latency_ms=1.0, min_top1=0.99)
        return (v == "onnx_fp32" and none == "none", f"choose={v}, 不能→{none}")

    def c9():
        model = _tiny_model(seed=1)
        x = torch.randn(2, 3, 16, 16)
        v = funcs["ex9"](model, x)
        return (v < 1e-4, f"trace往復 max|diff|={v:.2e}")

    def c10():
        import tempfile

        model = _tiny_model(seed=2)
        x = torch.randn(2, 3, 16, 16)
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            path = f.name
        v = funcs["ex10"](model, x, path)
        return (v < 1e-4, f"ONNX一致 max|diff|={v:.2e}")

    check("ex1_percentile", c1)
    check("ex2_speedup", c2)
    check("ex3_throughput", c3)
    check("ex4_top1_agreement", c4)
    check("ex5_max_abs_diff", c5)
    check("ex6_compression_ratio", c6)
    check("ex7_steady_median", c7)
    check("ex8_choose_runtime", c8)
    check("ex9_torchscript_roundtrip", c9)
    check("ex10_onnx_export_match", c10)

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
    import warnings

    warnings.filterwarnings("ignore")
    grade(current_funcs())
