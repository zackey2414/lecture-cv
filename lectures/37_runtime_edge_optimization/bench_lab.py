"""第37回 共通ヘルパ — 複数ランタイムの公平ベンチを1か所にまとめる。

ここに「計測の作法」と「ランタイムの組み立て」を集約し、各スクリプト
(01〜05・mini_project)はこのモジュールを import して使い回す。狙いは2つ:

  1. 計測ファースト原則を*同じコード*で全ランタイムに適用する。
     - ウォームアップ → 多数回反復 → 中央値(p50)/分位点(p99) で評価する。
     - CPU なので cuda.synchronize は不要。time.perf_counter で十分。
     - レイテンシ(1枚の応答, ms)とスループット(まとめて何枚/秒)は別物として測る。

  2. ランタイムを「numpy を入れて logits(numpy) を返す」共通インタフェースに
     そろえる(Runtime.infer)。こうすると eager torch / TorchScript /
     torch.compile / ONNX Runtime を 1 つのループで横並び比較できる。

設計メモ(この環境の現実):
  - torch 2.12+cpu / onnx 1.21 / onnxruntime 1.26。
  - torch.onnx.export は 2.9+ で dynamo=True が既定だが onnxscript が要る。
    本講座は onnxscript 非依存で確実に動かすため dynamo=False(legacy 経路)を使う。
  - torch.compile(Inductor)は C++ ツールチェイン(g++ と Python.h)が要る。
    無い環境では初回呼び出しで失敗するので *必ず try/except でガード* し、
    使えないときは表から外す(概念だけ README で説明)。
  - OpenVINO / CoreML / LiteRT / TensorRT は重い/プラットフォーム依存なので
    実行経路では必須にせず、04 で import ガードして概念紹介に留める。
"""

from __future__ import annotations

import contextlib
import io
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn

MODULE_ID = "37_runtime_edge_optimization"

# 公平比較のためスレッド数を固定する。多コア機ほどばらつくので、教材では少なめに固定。
# (本番では torch.set_num_threads / ORT の intra_op_num_threads を実機に合わせて探索する)
BENCH_THREADS = 4

# numpy で入力を受け、logits(numpy float32, 形 (N, 1000))を返す推論関数の型。
InferFn = Callable[[np.ndarray], np.ndarray]


# =====================================================================
# 出力先・モデル・合成データ
# =====================================================================
def ensure_output_dir() -> Path:
    """lectures/37_runtime_edge_optimization/outputs/ を作って返す(スクリプト隣基準)。"""
    out = Path(__file__).resolve().parent / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    return out


def apply_bench_threads(n: int = BENCH_THREADS) -> int:
    """torch のスレッド数を固定し、実際に設定された値を返す。

    レイテンシ計測は内部並列でばらつくため、まず固定して再現性を上げる。
    """
    n = max(1, min(n, torch.get_num_threads() if torch.get_num_threads() > 0 else n))
    torch.set_num_threads(n)
    return n


def make_resnet18(seed: int = 0) -> nn.Module:
    """評価モード済みの resnet18 を返す(重み DL 不要・決定的)。

    ベンチ目的では「重みが本物か」は無関係(計算量・グラフ構造が同じ)なので、
    オフラインでも確実に動くよう weights=None + 固定シードで初期化する。
    実データで精度まで見たいときは weights="IMAGENET1K_V1" に差し替えてよい。
    """
    import torchvision

    torch.manual_seed(seed)
    model = torchvision.models.resnet18(weights=None)
    model.eval()
    return model


def synthetic_batch(n: int, seed: int = 1234) -> np.ndarray:
    """(n, 3, 224, 224) の合成入力(float32, C連続)を返す。"""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, 3, 224, 224)).astype(np.float32)
    return np.ascontiguousarray(x)


# =====================================================================
# 計測コア(ウォームアップ → 反復 → 分位点)
# =====================================================================
@dataclass
class Latency:
    """1 サンプル応答時間の要約(ミリ秒)。"""

    p50_ms: float
    p90_ms: float
    p99_ms: float
    mean_ms: float
    first_ms: float  # ウォームアップ前の初回(JIT/コンパイルコストが乗る)


def measure_latency(infer: InferFn, x1: np.ndarray, warmup: int = 8, iters: int = 50) -> Latency:
    """バッチ1のレイテンシを p50/p90/p99 で測る(計測ファーストの正準形)。

    手順:
      1. 初回(first)を1回だけ単独計測 → JIT/コンパイル/キャッシュmiss の代金を可視化。
      2. warmup 回まわして定常状態に入れる(初回コストを計測から除く)。
      3. iters 回計測して分位点を取る。平均だけだと外れ値に弱いので中央値/分位点を主役にする。
    """
    assert x1.shape[0] == 1, "レイテンシはバッチ1で測る(1枚の応答時間)"

    t0 = time.perf_counter()
    infer(x1)
    first_ms = (time.perf_counter() - t0) * 1e3

    for _ in range(warmup):
        infer(x1)

    samples: list[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        infer(x1)
        samples.append((time.perf_counter() - t0) * 1e3)

    samples.sort()
    return Latency(
        p50_ms=_percentile(samples, 50),
        p90_ms=_percentile(samples, 90),
        p99_ms=_percentile(samples, 99),
        mean_ms=statistics.fmean(samples),
        first_ms=first_ms,
    )


def measure_throughput(infer: InferFn, xb: np.ndarray, warmup: int = 3, iters: int = 12) -> float:
    """スループット(images/sec)を測る。バッチを丸ごと流して総枚数/総時間。

    レイテンシと違い「並列性をどれだけ使えるか」を見る指標。バッチ1のまま
    比較すると並列性を活かせず不当に遅く見えるので、必ずバッチで測る。
    """
    batch = xb.shape[0]
    for _ in range(warmup):
        infer(xb)
    t0 = time.perf_counter()
    for _ in range(iters):
        infer(xb)
    elapsed = time.perf_counter() - t0
    return (batch * iters) / elapsed if elapsed > 0 else float("nan")


def _percentile(sorted_samples: list[float], q: float) -> float:
    """昇順済みリストの q パーセンタイル(線形補間)。numpy を使わず軽量に。"""
    if not sorted_samples:
        return float("nan")
    if len(sorted_samples) == 1:
        return sorted_samples[0]
    pos = (q / 100.0) * (len(sorted_samples) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_samples) - 1)
    frac = pos - lo
    return sorted_samples[lo] * (1 - frac) + sorted_samples[hi] * frac


# =====================================================================
# 精度保持(fp32 eager を基準にした一致度)とサイズ
# =====================================================================
def fidelity_vs_reference(logits: np.ndarray, ref_logits: np.ndarray) -> tuple[float, float]:
    """基準(fp32 eager)に対する忠実度を返す: (top1 一致率, 最大絶対誤差)。

    合成データなので「本物のラベル」は無い。ランタイム変換・量子化の正しさは
    『fp32 と同じ予測・近い数値か』で測るのが定石。検出/分類どちらでも、
    まず top1 一致率(argmax の一致)と数値誤差(atol/rtol の感覚)を見る。
    """
    pred = logits.argmax(axis=1)
    ref = ref_logits.argmax(axis=1)
    top1_agree = float((pred == ref).mean())
    max_abs = float(np.abs(logits - ref_logits).max())
    return top1_agree, max_abs


def state_dict_size_mb(module: nn.Module) -> float:
    """state_dict をメモリ上で直列化したバイト数から MB を概算する。"""
    buf = io.BytesIO()
    torch.save(module.state_dict(), buf)
    return buf.getbuffer().nbytes / 1e6


def file_size_mb(path: str | Path) -> float:
    """ファイルサイズ(MB)。存在しなければ NaN。"""
    p = Path(path)
    return p.stat().st_size / 1e6 if p.exists() else float("nan")


# =====================================================================
# ランタイム(numpy in → logits numpy out に統一)
# =====================================================================
@dataclass
class Runtime:
    """横並び比較の単位。infer は numpy→numpy の関数。"""

    name: str
    infer: InferFn
    size_mb: float = float("nan")
    note: str = ""


@dataclass
class BenchRow:
    """1 ランタイムのベンチ結果(表の1行)。"""

    name: str
    latency: Latency
    throughput_ips: float
    size_mb: float
    top1_agree: float
    max_abs_diff: float
    note: str = ""
    extra: dict = field(default_factory=dict)


def build_eager(model: nn.Module) -> Runtime:
    """素の PyTorch(fp32 eager)。比較の基準(baseline)になる。"""

    def infer(x: np.ndarray) -> np.ndarray:
        with torch.inference_mode():
            return model(torch.from_numpy(x)).numpy()

    return Runtime("eager_fp32", infer, state_dict_size_mb(model), "基準(baseline)")


def build_eager_bf16(model: nn.Module) -> Runtime:
    """CPU の混合精度(bf16 autocast)。CPU では fp16 は遅く bf16 が現実的。

    重みは fp32 のまま、計算を bf16 で行う。サイズは fp32 と同じ(重みは未変換)。
    """

    def infer(x: np.ndarray) -> np.ndarray:
        with torch.inference_mode(), torch.autocast("cpu", dtype=torch.bfloat16):
            return model(torch.from_numpy(x)).float().numpy()

    return Runtime("eager_bf16_autocast", infer, state_dict_size_mb(model), "計算のみ bf16")


def build_torchscript_trace(model: nn.Module, example: torch.Tensor, out_dir: Path) -> Runtime:
    """TorchScript(trace)。実行を1回なぞってグラフ化する。

    trace は「与えた入力の通り道」を固定するので、if 分岐やデータ依存の制御フローは
    捕捉できない(その場合は script を使う)。resnet18 は分岐が無いので trace で十分。
    保存した .pt は Python 非依存で配布でき、torch.jit.load でロードできる。
    """
    with torch.inference_mode():
        ts = torch.jit.trace(model, example)
    ts = torch.jit.freeze(ts)  # 推論用に定数畳み込み等を適用(eval 済み前提)
    path = out_dir / "resnet18_traced.pt"
    ts.save(str(path))

    def infer(x: np.ndarray) -> np.ndarray:
        with torch.inference_mode():
            return ts(torch.from_numpy(x)).numpy()

    return Runtime("torchscript_trace", infer, file_size_mb(path), "freeze 済み .pt")


def build_torchscript_script(model: nn.Module, out_dir: Path) -> Runtime | None:
    """TorchScript(script)。ソースを解析して制御フローごとグラフ化する。

    trace と違い分岐/ループを保持できる。モデルによっては未対応構文で失敗するため
    ガードして None を返せるようにする(失敗しても全体は止めない)。
    """
    try:
        sc = torch.jit.script(model)
        sc = torch.jit.freeze(sc.eval())
        path = out_dir / "resnet18_scripted.pt"
        sc.save(str(path))

        def infer(x: np.ndarray) -> np.ndarray:
            with torch.inference_mode():
                return sc(torch.from_numpy(x)).numpy()

        return Runtime("torchscript_script", infer, file_size_mb(path), "制御フロー保持")
    except Exception as e:  # noqa: BLE001
        print(f"  [skip] torchscript_script: {type(e).__name__}: {str(e)[:80]}")
        return None


def build_torch_compile(model: nn.Module, example: torch.Tensor) -> Runtime | None:
    """torch.compile(TorchDynamo + Inductor)。CPU では C++/OpenMP カーネルを生成。

    速くなる可能性がある一方、初回コンパイルが重く、g++ や Python.h が無い環境では
    *初回呼び出しで失敗する*。教材では失敗しても落ちないよう、ここで一度試走して
    例外を握り、使えなければ None を返す(README で概念だけ説明する)。
    """
    try:
        compiled = torch.compile(model)  # mode 既定。max-autotune は CPU だと探索が長い
        with torch.inference_mode():
            compiled(example)  # 初回でコンパイル発火 → ここで失敗を検知

        def infer(x: np.ndarray) -> np.ndarray:
            with torch.inference_mode():
                return compiled(torch.from_numpy(x)).numpy()

        return Runtime("torch_compile", infer, state_dict_size_mb(model), "Inductor 生成")
    except Exception as e:  # noqa: BLE001
        print(f"  [skip] torch.compile 利用不可: {type(e).__name__}: {str(e)[:90]}")
        print("        → C++ ツールチェイン(g++ / python3-dev の Python.h)が必要。概念は README 参照。")
        return None


def build_onnx(
    model: nn.Module, example: torch.Tensor, out_dir: Path, quantize: bool = False
) -> Runtime | None:
    """ONNX へエクスポート → onnxruntime(CPUExecutionProvider)で推論するランタイム。

    - dynamo=False(legacy 経路)で onnxscript 非依存に。opset 17 で広い互換性。
    - dynamic_axes でバッチ次元を可変にし、レイテンシ(1)とスループット(N)を同一モデルで測る。
    - SessionOptions で graph 最適化を ORT_ENABLE_ALL、スレッドを固定。
    - quantize=True なら quantize_dynamic で int8 化(CPU で最も手軽に効く圧縮)。
    """
    try:
        import onnx
        import onnxruntime as ort
    except Exception as e:  # noqa: BLE001
        print(f"  [skip] onnx/onnxruntime 不在: {e}")
        return None

    fp32_path = out_dir / "resnet18.onnx"
    with torch.inference_mode():
        torch.onnx.export(
            model,
            (example,),
            str(fp32_path),
            input_names=["input"],
            output_names=["logits"],
            dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
            opset_version=17,
            dynamo=False,  # ★ onnxscript 不要の安定経路
        )
    onnx.checker.check_model(onnx.load(str(fp32_path)))  # 構造の妥当性検証

    model_path = fp32_path
    name = "onnxruntime_fp32"
    note = "graph最適化ALL"
    if quantize:
        from onnxruntime.quantization import QuantType, quantize_dynamic

        int8_path = out_dir / "resnet18_int8.onnx"
        # CPU は U8X8 経路のため weight_type=QUInt8。Conv も int8 行列積に落ちる。
        with contextlib.redirect_stderr(io.StringIO()):
            quantize_dynamic(str(fp32_path), str(int8_path), weight_type=QuantType.QUInt8)
        model_path = int8_path
        name = "onnxruntime_int8_dynamic"
        note = "int8 動的量子化"

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.intra_op_num_threads = BENCH_THREADS
    so.inter_op_num_threads = 1
    sess = ort.InferenceSession(str(model_path), so, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    def infer(x: np.ndarray) -> np.ndarray:
        return sess.run(None, {input_name: x})[0]

    return Runtime(name, infer, file_size_mb(model_path), note)


# =====================================================================
# オーケストレーション(全ランタイムを横並びでベンチ)
# =====================================================================
def collect_runtimes(model: nn.Module, out_dir: Path, *, include_compile: bool = True) -> list[Runtime]:
    """この環境で利用可能なランタイムを集める(使えないものは自動で除外)。"""
    example = torch.from_numpy(synthetic_batch(1, seed=7))
    runtimes: list[Runtime] = [
        build_eager(model),
        build_eager_bf16(model),
        build_torchscript_trace(model, example, out_dir),
    ]
    sc = build_torchscript_script(model, out_dir)
    if sc is not None:
        runtimes.append(sc)
    if include_compile:
        cm = build_torch_compile(model, example)
        if cm is not None:
            runtimes.append(cm)
    onnx_fp32 = build_onnx(model, example, out_dir, quantize=False)
    if onnx_fp32 is not None:
        runtimes.append(onnx_fp32)
    onnx_int8 = build_onnx(model, example, out_dir, quantize=True)
    if onnx_int8 is not None:
        runtimes.append(onnx_int8)
    return runtimes


def benchmark_runtimes(
    runtimes: list[Runtime],
    eval_batch: np.ndarray,
    *,
    latency_iters: int = 40,
    throughput_batch: int = 16,
) -> list[BenchRow]:
    """各ランタイムを同一手順でベンチし、fp32 eager を基準に忠実度も測る。"""
    x1 = synthetic_batch(1, seed=21)
    xb = synthetic_batch(throughput_batch, seed=22)

    # 基準(eager_fp32)の logits を先に確定させ、以降の一致率の物差しにする。
    ref_name = runtimes[0].name
    ref_logits = runtimes[0].infer(eval_batch)

    rows: list[BenchRow] = []
    for rt in runtimes:
        lat = measure_latency(rt.infer, x1, iters=latency_iters)
        thr = measure_throughput(rt.infer, xb)
        logits = rt.infer(eval_batch)
        agree, max_abs = fidelity_vs_reference(logits, ref_logits)
        rows.append(
            BenchRow(
                name=rt.name,
                latency=lat,
                throughput_ips=thr,
                size_mb=rt.size_mb,
                top1_agree=agree,
                max_abs_diff=max_abs,
                note=rt.note + ("（=基準）" if rt.name == ref_name else ""),
            )
        )
    return rows


def print_table(rows: list[BenchRow], baseline_p50: float | None = None) -> None:
    """ベンチ結果を整形表示する。p50 を基準にした相対速度も出す。"""
    if baseline_p50 is None:
        baseline_p50 = rows[0].latency.p50_ms
    header = (
        f"{'runtime':26s} {'p50(ms)':>8s} {'p99(ms)':>8s} {'x速':>6s} "
        f"{'img/s':>8s} {'size(MB)':>9s} {'top1一致':>8s} {'maxdiff':>9s}  note"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        speedup = baseline_p50 / r.latency.p50_ms if r.latency.p50_ms > 0 else float("nan")
        size = f"{r.size_mb:9.2f}" if np.isfinite(r.size_mb) else f"{'—':>9s}"
        print(
            f"{r.name:26s} {r.latency.p50_ms:8.2f} {r.latency.p99_ms:8.2f} "
            f"{speedup:5.2f}x {r.throughput_ips:8.1f} {size} "
            f"{r.top1_agree:7.1%} {r.max_abs_diff:9.4f}  {r.note}"
        )


def save_bench_plot(rows: list[BenchRow], out_path: Path, title: str) -> None:
    """p50 レイテンシ(低いほど良い)とスループット(高いほど良い)を横棒で保存。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [r.name for r in rows]
    p50 = [r.latency.p50_ms for r in rows]
    ips = [r.throughput_ips for r in rows]
    y = np.arange(len(names))

    fig, axes = plt.subplots(1, 2, figsize=(13, 0.6 * len(names) + 1.8))
    axes[0].barh(y, p50, color="tab:red")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(names, fontsize=9)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("latency p50 (ms)  ← lower is better")
    axes[0].set_title("Latency (batch=1)")
    for yi, v in zip(y, p50):
        axes[0].text(v, yi, f" {v:.1f}", va="center", fontsize=8)

    axes[1].barh(y, ips, color="tab:green")
    axes[1].set_yticks(y)
    axes[1].set_yticklabels([])
    axes[1].invert_yaxis()
    axes[1].set_xlabel("throughput (img/s)  → higher is better")
    axes[1].set_title("Throughput (batched)")
    for yi, v in zip(y, ips):
        axes[1].text(v, yi, f" {v:.0f}", va="center", fontsize=8)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    # 単体スモークテスト: ヘルパが一通り動くか(各スクリプトの土台確認)。
    out = ensure_output_dir()
    th = apply_bench_threads()
    print(f"[bench_lab] threads={th}, out={out}")
    model = make_resnet18()
    rts = collect_runtimes(model, out)
    print(f"[bench_lab] 利用可能ランタイム: {[r.name for r in rts]}")
