"""03: 取得と処理のスレッド分離（producer/consumer）とフレームドロップ。

リアルタイム処理で最も怖いのは『レイテンシの雪だるま』。映像は一定間隔で届くのに、
処理が1枚あたりそれより長くかかると、未処理フレームがどんどん溜まり、表示が現実より
何秒も遅れていく。これを防ぐ2つの定石を実測で学ぶ:

  1. スレッド分離: cap.read() は I/O 待ちの間 GIL を解放する。だから『取得専用スレッド』と
     『処理側』を分けると、読み込み待ちの隙に処理を進められる（producer/consumer 構成）。
  2. フレームドロップ: queue.Queue(maxsize=1) に put_nowait で入れ、満杯なら新フレームを捨てる。
     『古い在庫を処理し続けて遅れる』より『最新だけ見て少し落とす』方がリアルタイムでは正しい。

このスクリプトは2つの戦略を同じ入力で走らせ、各処理フレームの『齢（=生成から処理までの遅れ）』を
比較する:
  - 無制限キュー（捨てない）: 齢がどんどん増える（遅延が溜まる）
  - maxsize=1 ＋ ドロップ: 齢はほぼ一定（最新に追従）だが一部フレームを捨てる

CPUバウンドな推論は GIL のためスレッドでは並列化しない、という限界と、その場合に
multiprocessing でプロセス分離する考え方も（CV_MP=1 のときだけ）実演する。

実行（スレッド版のみ）:
  uv run python lectures/11_realtime_stream/03_threaded_capture.py
プロセス分離も試す（任意）:
  CV_MP=1 uv run python lectures/11_realtime_stream/03_threaded_capture.py
"""

from __future__ import annotations

import os
import pathlib
import queue
import sys
import threading
import time

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from stream_helpers import SyntheticCapture, output_dir  # noqa: E402

PRODUCE_FPS = 60.0     # 入力が届く速さ（1秒に60枚）
CONSUME_COST_S = 0.04  # 1枚の処理にかかる時間の模擬（25fps相当）。実際の推論レイテンシの代役。
NUM_FRAMES = 90


def producer(
    source: SyntheticCapture,
    q: "queue.Queue",
    stop_event: threading.Event,
    counters: dict,
    drop_when_full: bool,
) -> None:
    """取得専用スレッド: 一定間隔でフレームを読み、キューに入れる。

    drop_when_full=True なら put_nowait で入れ、満杯(queue.Full)なら新フレームを捨てる
    （maxsize=1 と組み合わせると『最新1枚だけ』を保つ）。
    drop_when_full=False なら put でブロック待ち＝捨てない（在庫が溜まりレイテンシが増える）。
    各フレームには生成時刻を添えて、後で『齢』を測れるようにする。
    """
    interval = 1.0 / PRODUCE_FPS
    while not stop_event.is_set():
        ret, frame = source.read()
        if not ret:
            break
        counters["produced"] += 1
        item = (time.perf_counter(), frame)   # (生成時刻, フレーム)
        if drop_when_full:
            try:
                q.put_nowait(item)
            except queue.Full:
                counters["dropped"] += 1       # 満杯 → 新フレームを捨てる
        else:
            q.put(item)                        # 捨てない（満杯なら空くまで待つ）
        time.sleep(interval)                   # 次フレームが届くまでの間隔を模擬
    stop_event.set()                           # 入力が尽きたら停止を通知


def consume(
    q: "queue.Queue",
    stop_event: threading.Event,
    counters: dict,
    ages_ms: list[float],
) -> None:
    """処理側（メインスレッドから呼ぶ）: キューからフレームを取り出して『重い処理』を行う。

    各フレームの齢（age = 取り出した時刻 - 生成時刻）を記録する。これが『映像と処理のズレ』。
    入力が尽き（stop_event）かつキューが空になったら終了する。
    """
    while True:
        try:
            stamp, _frame = q.get(timeout=0.5)
        except queue.Empty:
            if stop_event.is_set():
                break       # producer が終わっていて在庫も無い → 完了
            continue
        # 重い処理の代役（一定時間スリープ＝推論レイテンシの模擬）。
        time.sleep(CONSUME_COST_S)
        ages_ms.append((time.perf_counter() - stamp) * 1000.0)
        counters["consumed"] += 1


def run_scenario(drop_when_full: bool, maxsize: int) -> dict:
    """1シナリオ実行: producer スレッドを起動し、consumer をメインで回し、計測を返す。"""
    source = SyntheticCapture(NUM_FRAMES, size=(360, 640))
    q: "queue.Queue" = queue.Queue(maxsize=maxsize)
    stop_event = threading.Event()
    counters = {"produced": 0, "dropped": 0, "consumed": 0}
    ages_ms: list[float] = []

    # daemon=True にしておくと、万一 producer が残っても本体終了を妨げない。
    t = threading.Thread(
        target=producer, args=(source, q, stop_event, counters, drop_when_full), daemon=True
    )
    t.start()
    consume(q, stop_event, counters, ages_ms)
    t.join(timeout=2.0)

    counters["ages_ms"] = ages_ms
    return counters


def main() -> None:
    out = output_dir()

    print("[1] 無制限キュー（捨てない）: 処理が間に合わず齢=遅延が増えていく…")
    unbounded = run_scenario(drop_when_full=False, maxsize=10_000)
    print(f"    produced={unbounded['produced']} consumed={unbounded['consumed']} "
          f"dropped={unbounded['dropped']}")

    print("[2] maxsize=1 + put_nowait ドロップ: 最新に追従し齢はほぼ一定（一部を捨てる）")
    dropping = run_scenario(drop_when_full=True, maxsize=1)
    print(f"    produced={dropping['produced']} consumed={dropping['consumed']} "
          f"dropped={dropping['dropped']}")

    # 齢の推移を1枚に重ねて可視化（無制限は右肩上がり、ドロップは横ばいになるはず）。
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(unbounded["ages_ms"], label="unbounded queue (no drop)", color="crimson")
    ax.plot(dropping["ages_ms"], label="maxsize=1 + drop", color="#2a9d8f")
    ax.set_xlabel("consumed frame index")
    ax.set_ylabel("frame age at consumption [ms]  (lower = lower latency)")
    ax.set_title("thread separation: dropping keeps latency bounded")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "03_latency_vs_drop.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("\n[plot] 齢(レイテンシ)の推移 -> 03_latency_vs_drop.png")

    # 数値サマリ: ドロップ版は齢が低く保たれる代わりにフレームを捨てている。
    def _avg(xs: list[float]) -> float:
        return float(np.mean(xs)) if xs else 0.0

    print("\n=== まとめ ===")
    print(f"  無制限キュー : 平均齢 {_avg(unbounded['ages_ms']):7.1f} ms / drop {unbounded['dropped']:3d}")
    print(f"  ドロップあり : 平均齢 {_avg(dropping['ages_ms']):7.1f} ms / drop {dropping['dropped']:3d}")
    drop_rate = dropping["dropped"] / max(dropping["produced"], 1)
    print(f"  → ドロップ率 {drop_rate*100:.0f}% と引き換えに、遅延の蓄積を断ち切れている。")

    # 任意: プロセス分離（GIL回避）の最小デモ。
    if os.environ.get("CV_MP") == "1":
        _run_multiprocessing_demo()
    else:
        print("\n[mp] CV_MP=1 を付けると multiprocessing 版（GIL回避）も実演します。")

    print("\n完了。lectures/11_realtime_stream/outputs/ にレイテンシ比較を保存しました。")


# =====================================================================
# 任意: multiprocessing によるプロセス分離（CPUバウンド処理向け）
# =====================================================================

def _mp_producer(q, n: int) -> None:
    """別プロセスのフレーム生成役。スレッドと違い GIL を共有しないので真に並列。"""
    import time as _t

    from stream_helpers import synthetic_frames
    for _, frame in synthetic_frames(n, size=(240, 320), seed=5):
        stamp = _t.perf_counter()
        try:
            q.put_nowait((stamp, frame))   # 満杯なら捨てる（プロセス間でも同じドロップ戦略）
        except Exception:                  # mp.Queue の Full は queue.Full
            pass
        _t.sleep(1.0 / PRODUCE_FPS)
    q.put(None)  # 終了の番兵


def _run_multiprocessing_demo() -> None:
    """CPUバウンドな処理を別プロセスへ逃がす最小例（GIL の影響を受けない）。

    スレッドは I/O 待ちには有効だが、純粋なCPU計算は GIL のため並列化しない。
    その場合は multiprocessing で『取得プロセス』と『処理プロセス（=本体）』を分ける。
    ここでは取得を別プロセス化し、本体がキューから取り出して処理する最小構成を示す。
    """
    import multiprocessing as mp

    print("\n[mp] multiprocessing デモ（取得を別プロセスへ。GILを共有しない）")
    ctx = mp.get_context("spawn")          # fork 依存を避け移植性を上げる
    q: mp.Queue = ctx.Queue(maxsize=1)
    p = ctx.Process(target=_mp_producer, args=(q, 30), daemon=True)
    p.start()

    consumed = 0
    t0 = time.perf_counter()
    while True:
        try:
            item = q.get(timeout=3.0)
        except Exception:                  # queue.Empty 等
            break
        if item is None:                   # 番兵 → 終了
            break
        stamp, frame = item
        # 別プロセスから来たフレームをここで処理（本体プロセス）。
        time.sleep(0.01)
        consumed += 1
    p.join(timeout=2.0)
    if p.is_alive():
        p.terminate()                      # 念のため後始末（ハング防止）
    print(f"[mp] 別プロセスから {consumed} 枚を処理（{time.perf_counter()-t0:.2f}s）。")


if __name__ == "__main__":
    main()
