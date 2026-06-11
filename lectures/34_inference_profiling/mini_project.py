"""第34回 章末ミニプロジェクト — 推論ベンチマーク・ユーティリティの完成形。

このスクリプトは「任意モデルのレイテンシ/スループットを“正しい手順”で計測し、profiler で
律速演算子を特定し、各最適化条件（eval+inference_mode / bf16 autocast / torch.compile）の
効果を一枚の比較表・図・JSON にまとめる」再利用可能なベンチマーカーです。

比較する条件（同一モデル・同一入力・同一手順で公平に）:
  A. naive (train, grad on)   … 推論なのに train モード＆勾配追跡（よくある誤り）
  B. eval + inference_mode    … 推論の正しい既定（“無料の高速化”）
  C. + bf16 autocast          … 低精度の混合精度推論（CPUは bf16）
  D. + torch.compile          … Inductor（環境次第で不可なら自動スキップ）
さらに:
  - ウォームアップを省いた“誤計測”との差を併記して、計測手順の重要性も示す。
  - torch.profiler で B 条件の律速演算子トップを出す。
出力:
  - outputs/34_inference_profiling/mini_benchmark.png（条件別 p50/p99 横棒）
  - outputs/34_inference_profiling/mini_report.json（全数値・環境情報）

実行:
  uv run python lectures/34_inference_profiling/mini_project.py
"""

from __future__ import annotations

import json
import pathlib
import platform
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from profiling_lab import (  # noqa: E402
    BenchResult,
    benchmark,
    benchmark_no_warmup,
    build_model,
    ensure_out_dir,
    format_table,
    get_device,
    make_compiled_runner,
    make_input,
    profile_ops,
    save_latency_bars,
    top_self_cpu_ops,
)


def build_conditions(
    model: torch.nn.Module, x: torch.Tensor, device: torch.device, batch_size: int
) -> list[BenchResult]:
    """A〜D の各条件を同じ手順でベンチして BenchResult のリストを返す。"""
    results: list[BenchResult] = []

    # A. naive: train モード＆勾配あり（推論には不適切）。train モードの BN は batch>1 が必要。
    def run_naive() -> object:
        model.train()
        return model(x)

    results.append(
        benchmark(run_naive, label="A: train + grad", batch_size=batch_size, n_warmup=5, n_iter=25, device=device)
    )
    model.eval()  # 以降は eval に固定

    # B. 推論の正しい既定
    def run_infer() -> object:
        with torch.inference_mode():
            return model(x)

    results.append(
        benchmark(run_infer, label="B: eval+inference_mode", batch_size=batch_size, n_warmup=5, n_iter=25, device=device)
    )

    # C. bf16 autocast を上乗せ
    def run_bf16() -> object:
        with torch.inference_mode(), torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            return model(x)

    results.append(
        benchmark(run_bf16, label="C: + bf16 autocast", batch_size=batch_size, n_warmup=5, n_iter=25, device=device)
    )

    # D. torch.compile（不可なら自動スキップ）
    runner, ok, msg = make_compiled_runner(model, x)
    if ok and runner is not None:
        results.append(
            benchmark(runner, label="D: + torch.compile", batch_size=batch_size, n_warmup=5, n_iter=25, device=device)
        )
    else:
        print(f"[info] D: torch.compile はこの環境では不可のためスキップ（{msg}）")

    return results


def main() -> None:
    out = ensure_out_dir()
    device = get_device()
    model_name = "resnet18"
    batch_size = 4
    model = build_model(model_name).to(device)
    x = make_input(batch_size=batch_size).to(device)

    print("=" * 72)
    print(f"推論ベンチマーク: model={model_name}  batch={batch_size}  device={device}  threads={torch.get_num_threads()}")
    print("=" * 72)

    # === 1. 計測手順の重要性（ウォームアップ有無） =======================
    def run_infer() -> object:
        model.eval()
        with torch.inference_mode():
            return model(x)

    bad = benchmark_no_warmup(run_infer, label="no-warmup", batch_size=batch_size, n_iter=25)
    good = benchmark(run_infer, label="warmup", batch_size=batch_size, n_warmup=5, n_iter=25, device=device)
    print("\n[1] 計測手順の検証（同じ処理・ウォームアップ有無だけが違う）")
    print(f"    no-warmup : p50={bad.p50_ms:7.3f}ms  p99={bad.p99_ms:7.3f}ms  mean={bad.mean_ms:7.3f}ms")
    print(f"    warmup    : p50={good.p50_ms:7.3f}ms  p99={good.p99_ms:7.3f}ms  mean={good.mean_ms:7.3f}ms")
    print("    → ウォームアップを省くと初回の遅さが p99/mean を汚す。計測の前提条件は結論を変える。")

    # === 2. 条件別ベンチ =================================================
    print("\n[2] 最適化条件の公平比較")
    results = build_conditions(model, x, device, batch_size)
    print(format_table(results, baseline_label="A: train + grad"))
    save_latency_bars(results, out / "mini_benchmark.png", title=f"{model_name} inference: condition comparison")

    # === 3. profiler で律速を特定（条件 B） ==============================
    model.eval()

    def run_for_profile() -> object:
        with torch.inference_mode():
            return model(x)

    prof = profile_ops(run_for_profile, n_warmup=3, n_iter=10, region="inference")
    top_ops = top_self_cpu_ops(prof, k=5)
    print("\n[3] 律速演算子トップ5（条件 B / 自己CPU時間）")
    for name, t in top_ops:
        print(f"    {name:32s} {t:8.3f}ms")
    print(f"    → 最重量は {top_ops[0][0]}。次の最適化（量子化/ONNX）はここを狙う（第35〜37回へ）。")

    # === 4. レポート JSON ===============================================
    report = {
        "env": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
            "num_threads": torch.get_num_threads(),
            "platform": platform.platform(),
        },
        "model": model_name,
        "batch_size": batch_size,
        "warmup_check": {"no_warmup": bad.as_row(), "warmup": good.as_row()},
        "conditions": [r.as_row() for r in results],
        "top_ops_self_cpu_ms": [{"name": n, "self_cpu_ms": round(t, 3)} for n, t in top_ops],
    }
    (out / "mini_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n[4] 保存物:")
    print("    -", (out / "mini_benchmark.png").name, "（条件別 p50/p99 横棒）")
    print("    -", (out / "mini_report.json").name, "（全数値・環境情報のレポート）")
    print("\n完成: 計測ファースト → 正しいベンチ → profiler で律速特定 → 条件比較、までを一気通貫で実行した。")


if __name__ == "__main__":
    main()
