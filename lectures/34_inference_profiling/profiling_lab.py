"""第34回 共通の道具箱 — 正しいベンチマークとプロファイリングの部品。

このモジュールは「読み物」ではなく、各スクリプト（01〜05・mini_project）が import して
使う再利用部品です。第34回のテーマ「計測ファースト」を支えるため、ここに

  1. get_device() / sync()      … device 判定と、GPU でのみ必要な同期（CPUは no-op）
  2. build_model() / make_input() … 題材（resnet18・合成入力）。重み DL もデータ DL も不要
  3. benchmark()                … ウォームアップ→多数回反復→p50/p90/p99/throughput を返す“正準手順”
  4. profile_ops()              … torch.profiler で自己CPU時間の重い演算子を集計
  5. make_compiled_runner()     … torch.compile を安全に試す（C++ツールチェーン不備なら eager に退避）
  6. 表・図の保存ユーティリティ

を一度だけ定義します。ベンチマークの「正しさ」はこのファイルに閉じ込め、各スクリプトは
「何を比較するか」だけに集中できるようにしています。

なぜ resnet18 + 合成入力か:
  本講座は GPU もデータセット DL も無しに必ず完走する必要があります。resnet18 は
  torchvision に同梱され `weights=None`（ランダム初期化）ならネット不要で、CNN として
  畳み込み・BN・ReLU・pooling と「プロファイルすると面白い演算子」が一通り含まれます。
  推論の“速さ”は重みの値に依存しないので、ベンチ題材としてはランダム重みで十分です。
"""

from __future__ import annotations

import dataclasses
import pathlib
import statistics
import time
from collections.abc import Callable

import matplotlib

matplotlib.use("Agg")  # ヘッドレス環境向け（画面ではなくファイルに描く）

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.profiler import ProfilerActivity, profile, record_function

# このモジュール（34回）の入出力ディレクトリ。__file__ から絶対パスで解決する。
MODULE_ID = "34_inference_profiling"
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT_DIR = pathlib.Path(__file__).resolve().parent / "outputs"
DATA_DIR = _REPO_ROOT / "data" / MODULE_ID


def ensure_out_dir() -> pathlib.Path:
    """lectures/34_inference_profiling/outputs/ を作って返す（既にあっても OK）。"""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUT_DIR


# =====================================================================
# device と同期
# =====================================================================
def get_device() -> torch.device:
    """cuda → mps → cpu の優先順で torch.device を返す（全回共通の定石）。

    本講座は CPU 前提なので既定では cpu。重要なのは「計測時に device を意識する」こと。
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sync(device: torch.device | None = None) -> None:
    """GPU 計測でのみ必要な同期。

    ★最重要の落とし穴: CUDA はカーネルを“非同期”に発行する。perf_counter で挟むだけだと
    GPU の処理完了を待たずに時刻を読むため『爆速』に誤計測する。計測の前後で
    torch.cuda.synchronize() を呼んで初めて正しい壁時計時間になる。
    CPU は同期不要（命令は同期実行）なので、ここでは device を見て CUDA のときだけ同期する。
    """
    if device is not None and device.type == "cuda":
        torch.cuda.synchronize()


def set_threads(n: int | None) -> int:
    """intra-op スレッド数を設定し、設定後の実効スレッド数を返す（None なら現状維持）。

    CPU 推論のレイテンシ/スループットはスレッド数に強く依存する。スレッドを増やすと
    1 枚のレイテンシは下がりやすいが、メモリ帯域で頭打ちになり線形には伸びない。
    """
    if n is not None:
        torch.set_num_threads(n)
    return torch.get_num_threads()


# =====================================================================
# 題材（モデル・入力）
# =====================================================================
def build_model(name: str = "resnet18") -> torch.nn.Module:
    """ベンチ題材のモデルを eval() で返す（重み DL 不要のランダム初期化）。

    eval() を最初から掛けておくのは、BatchNorm を“移動統計を使う推論モード”に、
    Dropout を“素通し”に固定するため。学習用の挙動のままベンチすると BN がバッチ統計を
    計算してしまい、結果も速度も推論時と変わる。
    """
    import torchvision

    if name == "resnet18":
        model = torchvision.models.resnet18(weights=None)
    elif name == "resnet50":
        model = torchvision.models.resnet50(weights=None)
    elif name == "mobilenet_v3_small":
        model = torchvision.models.mobilenet_v3_small(weights=None)
    else:
        raise ValueError(f"未知のモデル名: {name}")
    return model.eval()


def make_input(batch_size: int = 1, size: int = 224, seed: int = 0) -> torch.Tensor:
    """合成入力 (B,3,size,size) を返す（決定的）。推論時間は値に依存しないので乱数でよい。"""
    g = torch.Generator().manual_seed(seed)
    return torch.randn(batch_size, 3, size, size, generator=g)


# =====================================================================
# 正しいベンチマーク（ウォームアップ → 反復 → 分位点）
# =====================================================================
@dataclasses.dataclass
class BenchResult:
    """1 条件のベンチ結果。レイテンシは ms、スループットは img/s で保持する。"""

    label: str
    batch_size: int
    n_iter: int
    latencies_ms: list[float]  # 各反復のレイテンシ（ms）

    @property
    def p50_ms(self) -> float:
        return float(np.percentile(self.latencies_ms, 50))

    @property
    def p90_ms(self) -> float:
        return float(np.percentile(self.latencies_ms, 90))

    @property
    def p99_ms(self) -> float:
        return float(np.percentile(self.latencies_ms, 99))

    @property
    def mean_ms(self) -> float:
        return float(np.mean(self.latencies_ms))

    @property
    def std_ms(self) -> float:
        return float(np.std(self.latencies_ms))

    @property
    def throughput_img_s(self) -> float:
        """中央値レイテンシ基準のスループット（バッチ枚数 / p50秒）。"""
        return self.batch_size / (self.p50_ms / 1e3)

    def as_row(self) -> dict[str, float | str | int]:
        return {
            "label": self.label,
            "batch": self.batch_size,
            "p50_ms": round(self.p50_ms, 3),
            "p90_ms": round(self.p90_ms, 3),
            "p99_ms": round(self.p99_ms, 3),
            "mean_ms": round(self.mean_ms, 3),
            "std_ms": round(self.std_ms, 3),
            "img_s": round(self.throughput_img_s, 1),
        }


def benchmark(
    fn: Callable[[], object],
    *,
    label: str,
    batch_size: int = 1,
    n_warmup: int = 10,
    n_iter: int = 50,
    device: torch.device | None = None,
) -> BenchResult:
    """fn を「ウォームアップ→多数回反復→各回の時間を記録」で計測して BenchResult を返す。

    正しいベンチの 4 原則をすべてここに実装している:
      (1) ウォームアップ … 初回は遅延ロード/キャッシュ miss/JIT で遅い。n_warmup 回は捨てる。
      (2) 多数回反復     … 1 回計測はブレるので n_iter 回測り、分位点で評価する。
      (3) 同期          … GPU は sync() を入れて非同期実行の誤計測を防ぐ（CPUは no-op）。
      (4) 分位点         … 平均より p50（中央値）/p99（最悪寄り）が外れ値に強く実態を表す。

    fn は「引数なしで 1 回の推論を行う」クロージャを渡す（torch.inference_mode 等は呼び出し側で）。
    """
    # (1) ウォームアップ: 計測には含めない
    for _ in range(n_warmup):
        fn()
    sync(device)

    # (2)(3) 反復計測。各回ごとに sync して“その 1 回”の壁時計時間を測る
    latencies_ms: list[float] = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        fn()
        sync(device)
        latencies_ms.append((time.perf_counter() - t0) * 1e3)

    return BenchResult(label=label, batch_size=batch_size, n_iter=n_iter, latencies_ms=latencies_ms)


def benchmark_no_warmup(
    fn: Callable[[], object], *, label: str, batch_size: int = 1, n_iter: int = 50
) -> BenchResult:
    """わざとウォームアップを省いた“誤った”計測（初回の遅さを混ぜてしまう例）。

    benchmark() との差を見せて「ウォームアップを省くと p99 や mean が膨らむ」ことを体感させる教材用。
    """
    latencies_ms: list[float] = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        fn()
        latencies_ms.append((time.perf_counter() - t0) * 1e3)
    return BenchResult(label=label, batch_size=batch_size, n_iter=n_iter, latencies_ms=latencies_ms)


# =====================================================================
# プロファイリング（torch.profiler）
# =====================================================================
def profile_ops(
    fn: Callable[[], object],
    *,
    n_warmup: int = 3,
    n_iter: int = 10,
    region: str = "inference",
) -> profile:
    """fn を torch.profiler で計測し、profiler オブジェクトを返す。

    record_function(region) で計測区間に注釈を付ける。返り値に対して
    key_averages().table(sort_by="self_cpu_time_total") を呼ぶと演算子別の時間集計が得られる。
    """
    for _ in range(n_warmup):  # コンパイル/キャッシュを温めてから計測（プロファイルでもウォームアップは要る）
        fn()
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function(region):
            for _ in range(n_iter):
                fn()
    return prof


def top_self_cpu_ops(prof: profile, k: int = 8) -> list[tuple[str, float]]:
    """profiler から (演算子名, 自己CPU時間[ms]) の上位 k 件を返す（律速演算子の特定）。

    self_cpu_time_total は「その演算子“自身”が使った時間（子の時間を含まない）」。
    どの演算が支配的かを知るにはこれで降順に並べるのが定石。
    """
    rows = [(e.key, e.self_cpu_time_total / 1e3) for e in prof.key_averages()]
    rows.sort(key=lambda kv: kv[1], reverse=True)
    return rows[:k]


# =====================================================================
# torch.compile を“安全に”試す
# =====================================================================
def make_compiled_runner(
    model: torch.nn.Module, x: torch.Tensor, **compile_kwargs: object
) -> tuple[Callable[[], object] | None, bool, str]:
    """torch.compile を試し、(実行クロージャ, 成功フラグ, メッセージ) を返す。

    ★現場の落とし穴: torch.compile の CPU バックエンド（Inductor）は C++ コードを生成して
    g++ 等でコンパイルする。コンパイラや Python 開発ヘッダ（Python.h）が無い環境では
    “最初の呼び出し時”に CppCompileError で失敗する。失敗を握りつぶして eager に退避できるよう、
    トリガ呼び出しまでをここで try/except する（これにより全スクリプトが exit 0 を保てる）。
    """
    try:
        compiled = torch.compile(model, **compile_kwargs)
        with torch.inference_mode():
            compiled(x)  # ← ここで初めてコンパイルが走る（失敗するならここ）
        return (lambda: _run(compiled, x)), True, "compiled"
    except Exception as e:  # noqa: BLE001  失敗しても講座を止めない
        return None, False, f"{type(e).__name__}: {str(e).splitlines()[0][:120]}"


def _run(model: torch.nn.Module, x: torch.Tensor) -> object:
    with torch.inference_mode():
        return model(x)


# =====================================================================
# 表・図の保存ユーティリティ
# =====================================================================
def format_table(results: list[BenchResult], baseline_label: str | None = None) -> str:
    """BenchResult のリストを ASCII 表に整形して返す（speedup 列つき）。"""
    base_p50 = None
    if baseline_label is not None:
        for r in results:
            if r.label == baseline_label:
                base_p50 = r.p50_ms
                break

    header = f"{'condition':28s} {'p50(ms)':>9s} {'p99(ms)':>9s} {'img/s':>8s} {'speedup':>8s}"
    lines = [header, "-" * len(header)]
    for r in results:
        speed = f"{base_p50 / r.p50_ms:6.2f}x" if base_p50 else "   -  "
        lines.append(
            f"{r.label:28s} {r.p50_ms:9.3f} {r.p99_ms:9.3f} {r.throughput_img_s:8.1f} {speed:>8s}"
        )
    return "\n".join(lines)


def save_latency_bars(
    results: list[BenchResult], path: pathlib.Path, title: str = "latency (p50) by condition"
) -> None:
    """各条件の p50 レイテンシ（と p99 ひげ）を横棒で保存する。"""
    labels = [r.label for r in results]
    p50 = [r.p50_ms for r in results]
    p99 = [r.p99_ms for r in results]
    y = np.arange(len(results))
    fig, ax = plt.subplots(figsize=(8.5, 0.6 * len(results) + 1.5))
    ax.barh(y, p50, color="tab:blue", label="p50")
    # p99 までの差を薄く重ねて“ばらつき”を見せる
    ax.barh(y, [b - a for a, b in zip(p50, p99)], left=p50, color="tab:orange", alpha=0.5, label="p50→p99")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("latency (ms)")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def save_throughput_curve(
    batch_sizes: list[int], throughputs: list[float], path: pathlib.Path
) -> None:
    """バッチサイズ vs スループット曲線を保存する（並列性で img/s が伸びる様子）。"""
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.plot(batch_sizes, throughputs, "o-", color="tab:green")
    for bs, tp in zip(batch_sizes, throughputs):
        ax.annotate(f"{tp:.0f}", (bs, tp), textcoords="offset points", xytext=(0, 6), fontsize=8)
    ax.set_xlabel("batch size")
    ax.set_ylabel("throughput (img/s)")
    ax.set_title("throughput vs batch size (CPU)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# =====================================================================
# 純関数（演習と本編で共有する“ベンチの中身”）
# =====================================================================
def percentile(values: list[float], q: float) -> float:
    """np.percentile（線形補間）と一致する分位点を返す。q は 0〜100。"""
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def throughput(latency_s: float, batch_size: int) -> float:
    """1 反復のレイテンシ（秒）とバッチ枚数から img/s を返す。"""
    return batch_size / latency_s if latency_s > 0 else float("inf")


def speedup(baseline_ms: float, candidate_ms: float) -> float:
    """speedup = baseline / candidate（>1 で速くなった）。"""
    return baseline_ms / candidate_ms if candidate_ms > 0 else float("inf")


def summarize_latencies(latencies_ms: list[float]) -> dict[str, float]:
    """レイテンシ列から p50/p90/p99/mean/std をまとめて返す。"""
    arr = np.asarray(latencies_ms, dtype=np.float64)
    return {
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p99": float(np.percentile(arr, 99)),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "median_stats": float(statistics.median(latencies_ms)),  # statistics でも一致する確認用
    }
