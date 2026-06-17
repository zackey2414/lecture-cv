"""01: 正しいベンチマークの取り方 — ウォームアップ・分位点・スレッド・スループット。

学ぶこと:
  - なぜ「最適化の前に計測」なのか。まず“正しく測れる”ようにならないと、改善の真偽が判定できない。
  - 正しい手順の4原則:
      (1) ウォームアップ … 初回は遅延ロード/キャッシュ miss で遅い → 数回捨てる。
      (2) 多数回反復     … 1回計測はブレる → 何度も測る。
      (3) 同期          … GPU は torch.cuda.synchronize() が必須（CPUは不要）。
      (4) 分位点で評価   … mean ではなく p50（中央値）/ p99（最悪寄り）で語る。
  - レイテンシ（1件あたりの応答時間）と スループット（単位時間あたりの処理枚数）の違い。
  - torch.set_num_threads でスレッド数を変えると CPU の速度が大きく変わること。
  - torch.utils.benchmark.Timer という“正しい計測”を内蔵した公式ツール。

実行:
  uv run python lectures/34_inference_profiling/01_benchmark_basics.py
結果は lectures/34_inference_profiling/outputs/ に保存される。
"""

from __future__ import annotations

import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from profiling_lab import (  # noqa: E402
    benchmark,
    benchmark_no_warmup,
    build_model,
    ensure_out_dir,
    get_device,
    make_input,
    save_throughput_curve,
    set_threads,
)


def main() -> None:
    out = ensure_out_dir()
    device = get_device()
    model = build_model("resnet18").to(device)
    print(f"[setup] device={device} / threads={torch.get_num_threads()} / model=resnet18(eval)\n")

    # 推論クロージャ。inference_mode は“勾配を作らない＝速い/省メモリ”の推論用文脈。
    x1 = make_input(batch_size=1).to(device)

    def run_bs1() -> object:
        with torch.inference_mode():
            return model(x1)

    # === 1. ウォームアップ有無の差 =======================================
    # 同じモデル・同じ入力でも、ウォームアップを省くと初回の遅さが mean/p99 を膨らませる。
    bad = benchmark_no_warmup(run_bs1, label="no-warmup (wrong)", batch_size=1, n_iter=50)
    good = benchmark(run_bs1, label="warmup (correct)", batch_size=1, n_warmup=10, n_iter=50, device=device)

    print("[1] ウォームアップ有無で“同じ処理”の数字がどう変わるか")
    print(f"    誤(no warmup): p50={bad.p50_ms:7.3f}ms  p99={bad.p99_ms:7.3f}ms  mean={bad.mean_ms:7.3f}ms")
    print(f"    正(warmup)   : p50={good.p50_ms:7.3f}ms  p99={good.p99_ms:7.3f}ms  mean={good.mean_ms:7.3f}ms")
    print(f"    → 1回目: {bad.latencies_ms[0]:.3f}ms（初回は遅い）。これを捨てるのがウォームアップ。\n")

    # === 2. mean と p50/p99 の違い =======================================
    # 外れ値（GC・OSスケジューラ等の突発的な遅延）に mean は引きずられる。p50 は頑健、p99 は最悪寄り。
    print("[2] 平均より分位点で語る理由")
    print(f"    mean={good.mean_ms:.3f}ms / p50={good.p50_ms:.3f}ms / p99={good.p99_ms:.3f}ms")
    print("    SLA（応答保証）は通常 p99 で設計する。『たまに遅い』を mean は隠してしまう。\n")

    # === 3. レイテンシ vs スループット（バッチサイズ実験） ===============
    # レイテンシ = 1反復の時間。スループット = batch / レイテンシ秒（= img/s）。
    # CPU でもバッチを増やすと並列性が効いて img/s は伸びる（が、いずれ頭打ち）。
    print("[3] レイテンシ vs スループット（バッチを変える）")
    batch_sizes = [1, 2, 4, 8]
    throughputs: list[float] = []
    for bs in batch_sizes:
        xb = make_input(batch_size=bs).to(device)

        def run_bs(xb: torch.Tensor = xb) -> object:
            with torch.inference_mode():
                return model(xb)

        r = benchmark(run_bs, label=f"bs={bs}", batch_size=bs, n_warmup=5, n_iter=20, device=device)
        throughputs.append(r.throughput_img_s)
        print(f"    bs={bs:2d}: latency(p50)={r.p50_ms:8.3f}ms  throughput={r.throughput_img_s:7.1f} img/s")
    print("    → バッチを増やすとレイテンシは伸びるが img/s は上がる（スループット最適化の基本）。\n")
    save_throughput_curve(batch_sizes, throughputs, out / "01_throughput_vs_batch.png")

    # === 4. スレッド数の効果 =============================================
    # CPU 推論は intra-op スレッド数に強く依存。1スレッドに絞ると遅くなる（並列性を捨てるため）。
    print("[4] torch.set_num_threads の効果（CPUのみ意味を持つ）")
    original = torch.get_num_threads()
    x4 = make_input(batch_size=4).to(device)

    def run_bs4() -> object:
        with torch.inference_mode():
            return model(x4)

    for n in [1, max(1, original // 2), original]:
        set_threads(n)
        r = benchmark(run_bs4, label=f"threads={n}", batch_size=4, n_warmup=5, n_iter=15, device=device)
        print(f"    threads={n:2d}: p50={r.p50_ms:8.3f}ms  throughput={r.throughput_img_s:7.1f} img/s")
    set_threads(original)  # 後続に影響しないよう必ず戻す
    print(f"    （計測後はスレッド数を {original} に戻した）\n")

    # === 5. 公式ツール torch.utils.benchmark.Timer ======================
    # Timer は内部でウォームアップ・反復・（GPUなら）同期を正しく行う“ベンチの決定版”。
    # ★注意: Timer は既定で num_threads=1 で測る。自前ループ（既定スレッド数）と公平に比べるには
    #        num_threads を明示して揃える（揃えないと“別条件”を比べて食い違う）。
    from torch.utils.benchmark import Timer

    timer = Timer(stmt="m(x)", globals={"m": model, "x": x1}, num_threads=torch.get_num_threads())
    measurement = timer.timeit(50)  # 内部で適切に回す
    print("[5] torch.utils.benchmark.Timer（自前ループの答え合わせ）")
    print(f"    Timer median = {measurement.median * 1e3:.3f}ms  /  自前 p50 = {good.p50_ms:.3f}ms")
    print("    → スレッド数を揃えれば概ね一致する（一致＝自前ループの計測が正しい証拠）。\n")

    print("[まとめ] 計測は『ウォームアップ→多数回→(GPUは同期)→分位点』。")
    print("         レイテンシとスループットは別物。CPU はスレッド数で激変する。")
    print("         図を保存:", (out / "01_throughput_vs_batch.png").name)


if __name__ == "__main__":
    main()
