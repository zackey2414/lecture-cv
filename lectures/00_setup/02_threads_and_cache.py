"""02: CPU スレッド調整と HuggingFace キャッシュの定石。

実行:
    uv run python lectures/00_setup/02_threads_and_cache.py

学ぶこと:
  - torch.set_num_threads / OMP_NUM_THREADS でスレッド数を制御する意味
  - スレッド数を変えると行列積の速度がどう変わるかを実測（推測するな、測れ）
  - 「torch と OpenMP を揃える」べき理由（オーバーサブスクライブ）
  - HF_HOME / HF_HUB_OFFLINE でモデルキャッシュとオフライン動作を制御する

CPU のみ・合成データ・ネット不要で完走する。結果図は lectures/00_setup/outputs/ に保存。
"""

from __future__ import annotations

import os
import pathlib
import sys
import time

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from device import configure_threads, hf_home  # noqa: E402


def _output_dir() -> pathlib.Path:
    """このモジュール専用の出力先 lectures/00_setup/outputs/ を作って返す。"""
    d = pathlib.Path(__file__).resolve().parent / "outputs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _bench_matmul(threads: int, size: int = 1024, iters: int = 5) -> float:
    """指定スレッド数での行列積 1 回あたりの中央値時間(ms)を測る（単一責務）。"""
    torch.set_num_threads(threads)
    a = torch.randn(size, size)
    b = torch.randn(size, size)
    # ウォームアップ（初回は確保・最適化が走るので計測から除外する）
    for _ in range(2):
        a @ b
    times: list[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        a @ b
        times.append((time.perf_counter() - t0) * 1000.0)
    times.sort()
    return times[len(times) // 2]  # 中央値（外れ値に強い）


def main() -> None:
    cpu = os.cpu_count() or 1
    print("=== 1. スレッド数の既定と制御 ===")
    print(f"  os.cpu_count()         = {cpu}")
    print(f"  torch.get_num_threads()= {torch.get_num_threads()}（既定）")
    used = configure_threads()  # 既定では cpu を上限に妥当値へ
    print(f"  configure_threads()    -> {used} に固定（OMP_NUM_THREADS={os.environ['OMP_NUM_THREADS']}）")

    # --- 2. スレッド数 × 速度の実測 -----------------------------------
    # スレッドを増やせば必ず速くなるとは限らない（メモリ帯域・小さい行列では頭打ち）。
    print("\n=== 2. スレッド数を変えて行列積を実測（1024x1024）===")
    candidates = sorted({1, 2, 4, cpu})
    candidates = [t for t in candidates if 1 <= t <= cpu]
    results: list[tuple[int, float]] = []
    for t in candidates:
        ms = _bench_matmul(t)
        results.append((t, ms))
        print(f"  threads={t:3d} : {ms:8.2f} ms / 1 回")

    base = results[0][1]
    print("\n  （threads=1 を基準にした速度比）")
    for t, ms in results:
        print(f"    threads={t:3d} : x{base / ms:5.2f}")

    # 後片付け: 計測で変えたスレッド数を妥当値に戻す。
    configure_threads(used)

    # --- 3. オーバーサブスクライブの注意 ------------------------------
    print("\n=== 3. なぜ torch と OMP を揃えるのか ===")
    print("  torch だけ／OMP だけ設定すると、cv2・numpy(OpenMP) と torch が別々に")
    print("  スレッドを立て、物理コア数を超える『オーバーサブスクライブ』で遅くなる。")
    print("  configure_threads() は両方を同じ値に揃えるのでこの事故を防げる。")

    # --- 4. HuggingFace キャッシュ ------------------------------------
    print("\n=== 4. HuggingFace キャッシュ（HF_HOME / HF_HUB_OFFLINE）===")
    print(f"  HF_HOME        = {hf_home()}")
    print(f"  存在するか      = {hf_home().exists()}（無ければ初回 DL 時に作られる）")
    print(f"  HF_HUB_OFFLINE = {os.environ.get('HF_HUB_OFFLINE', '0')}"
          "（1 ならキャッシュ済みだけでネット非依存に動く）")
    print("  Docker では HF_HOME をボリュームにマウントして再 DL を防ぐ（README 参照）。")

    _save_chart(results)
    print(f"\n保存: {_output_dir() / '02_threads.png'}")


def _save_chart(results: list[tuple[int, float]]) -> None:
    """スレッド数 × 1 回あたり時間の棒グラフを保存する（headless: Agg）。"""
    import matplotlib

    matplotlib.use("Agg")  # 画面不要のバックエンドに固定（import pyplot より前）
    import matplotlib.pyplot as plt

    threads = [str(t) for t, _ in results]
    ms = [m for _, m in results]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(threads, ms, color="tab:blue")
    ax.set_xlabel("num_threads")
    ax.set_ylabel("matmul time (ms, lower is better)")
    ax.set_title("CPU threads vs matmul latency (1024x1024)")
    for i, m in enumerate(ms):
        ax.text(i, m, f"{m:.0f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(_output_dir() / "02_threads.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
