"""02: CPUで実時間に追いつかせる定石 — 早期縮小・フレームスキップ・grab/retrieve。

『1フレームの処理に 50ms かかるのに、映像は 33ms ごとに来る』——このときパイプラインは
だんだん遅れ、レイテンシ（映像と表示のズレ）が雪だるま式に膨らむ。GPU が無くても実時間に
追いつかせるための、OpenCV における3つの定石を実測で学ぶ:

  1. 早期縮小: 重い処理の『前』に cv2.resize(..., INTER_AREA) で小さくする。
     画素数が 1/4 になれば計算量もおおむね 1/4。縮小は INTER_AREA が定石（モアレが出にくい）。
  2. フレームスキップ: 毎フレームではなく N フレームに1回だけ重い処理を走らせる。
     間のフレームは前回の結果を使い回す。フレームレートを稼ぐ最も手軽な手段。
  3. grab / retrieve: cap.read() は「デコード＋numpy変換＋コピー」を毎回行う。
     cap.grab() は内部バッファを進めるだけ（安価）で、本当に必要なフレームだけ
     cap.retrieve() でデコードする。カメラ/RTSP では溜まったフレームを安価に捨てて
     『最新』に追いつくのにも使う。

このスクリプトはまず合成動画を mp4 で書き出し（書けなければ合成ソースにフォールバック）、
それを VideoCapture で読み戻して各シナリオの処理FPS・レイテンシ(p50/p99)を比較する。

実行:
  uv run python lectures/11_realtime_stream/02_frameskip_grab_retrieve.py
"""

from __future__ import annotations

import pathlib
import sys
import time

import cv2

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from stream_helpers import (  # noqa: E402
    output_dir,
    percentiles,
    simulate_heavy_work,
    synthetic_frames,
)

SRC_SIZE = (480, 640)   # (H, W) もとの解像度（重い）
SMALL_SIZE = (240, 320)  # (H, W) 縮小後（軽い）
NUM_FRAMES = 120
SRC_FPS = 30.0


def write_demo_video(path: pathlib.Path) -> bool:
    """合成フレームを mp4 動画として書き出す。書けたら True。

    FOURCC は headless でも使える "mp4v"(.mp4)。サイズは (幅, 高さ) の順で渡す点に注意
    （cv2 の dsize と同じく shape の逆順）。出力サイズが実フレームと違うと空/壊れた動画になる。
    """
    h, w = SRC_SIZE
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, SRC_FPS, (w, h))  # (W, H) 順
    if not writer.isOpened():
        return False
    for _, frame in synthetic_frames(NUM_FRAMES, SRC_SIZE, seed=2):
        writer.write(frame)
    writer.release()
    return path.exists() and path.stat().st_size > 0


def run_read_every_frame(cap, resize_to: tuple[int, int] | None) -> dict:
    """シナリオ A/B: 毎フレーム read() して重い処理。resize_to があれば『先に縮小』してから処理。

    返り値は計測サマリ（処理枚数・総時間・処理FPS・処理レイテンシの p50/p99）。
    """
    latencies_ms: list[float] = []
    processed = 0
    t0 = time.perf_counter()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if resize_to is not None:
            h, w = resize_to
            frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)  # 早期縮小
        t = time.perf_counter()
        simulate_heavy_work(frame)            # 重い処理（解像度に比例して重い）
        latencies_ms.append((time.perf_counter() - t) * 1000.0)
        processed += 1
    total = time.perf_counter() - t0
    return _summary(processed, total, latencies_ms)


def run_resize_and_skip(cap, resize_to: tuple[int, int], every_n: int) -> dict:
    """シナリオ C: 早期縮小 ＋ N フレームに1回だけ重い処理（間は前回結果を使い回す）。"""
    latencies_ms: list[float] = []
    processed = 0
    seen = 0
    t0 = time.perf_counter()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        seen += 1
        if seen % every_n != 0:
            continue  # スキップ: ここでは重い処理をしない（前回結果を使う想定）
        h, w = resize_to
        small = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
        t = time.perf_counter()
        simulate_heavy_work(small)
        latencies_ms.append((time.perf_counter() - t) * 1000.0)
        processed += 1
    total = time.perf_counter() - t0
    return _summary(processed, total, latencies_ms, seen=seen)


def run_grab_retrieve(cap, resize_to: tuple[int, int], every_n: int) -> dict:
    """シナリオ D: grab() で安価にフレームを進め、N回に1回だけ retrieve() でデコードして処理。

    cap.grab() はデコード前の安価な前進。cap.retrieve() が実際の numpy 変換を行う。
    『要らないフレームは grab だけで読み飛ばす』ことで、read() より無駄なデコード/コピーを省ける。
    """
    latencies_ms: list[float] = []
    processed = 0
    seen = 0
    t0 = time.perf_counter()
    while True:
        grabbed = cap.grab()          # 安価: 内部バッファを1つ進めるだけ（デコードしない）
        if not grabbed:
            break
        seen += 1
        if seen % every_n != 0:
            continue                  # retrieve() を呼ばない＝デコードしないので速い
        ret, frame = cap.retrieve()   # ここで初めてデコードして numpy 化
        if not ret:
            break
        h, w = resize_to
        small = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
        t = time.perf_counter()
        simulate_heavy_work(small)
        latencies_ms.append((time.perf_counter() - t) * 1000.0)
        processed += 1
    total = time.perf_counter() - t0
    return _summary(processed, total, latencies_ms, seen=seen)


def _summary(processed: int, total_s: float, latencies_ms: list[float], seen: int | None = None) -> dict:
    """計測結果を辞書にまとめる。

    リアルタイム性の主指標は『ストリーム消化レート』= 入力フレームを1秒に何枚さばけたか
    (seen / 総時間)。これが入力のソースFPS(例:30)を上回っていれば実時間に追いつける。
    縮小・スキップ・grab はこのレートを上げるための手段なので、この指標で効果が見える。
    """
    seen_n = seen if seen is not None else processed
    p50, p99 = percentiles(latencies_ms)
    stream_rate = seen_n / total_s if total_s > 0 else 0.0  # 入力消化レート[frames/s]
    return {
        "seen": seen_n,
        "processed": processed,
        "total_s": total_s,
        "stream_rate": stream_rate,
        "p50_ms": p50,
        "p99_ms": p99,
    }


def open_video(path: pathlib.Path):
    """書き出した動画を VideoCapture で開く。read()/grab()/retrieve() が本物のAPIで使える。"""
    return cv2.VideoCapture(str(path))


def main() -> None:
    out = output_dir()
    video_path = out / "02_demo_source.mp4"

    have_video = write_demo_video(video_path)
    if have_video:
        print(f"[setup] 合成動画を書き出しました: {video_path.name} "
              f"({video_path.stat().st_size // 1024} KiB)")
    else:
        print("[setup] mp4 を書き出せませんでした。grab/retrieve シナリオはスキップします。")

    results: dict[str, dict] = {}

    # --- シナリオ A: 原寸・毎フレーム（ベースライン、最も重い） ---
    cap = open_video(video_path) if have_video else None
    if cap is not None and cap.isOpened():
        results["A: full-res every frame"] = run_read_every_frame(cap, resize_to=None)
        cap.release()

    # --- シナリオ B: 早期縮小・毎フレーム ---
    cap = open_video(video_path) if have_video else None
    if cap is not None and cap.isOpened():
        results["B: resize every frame"] = run_read_every_frame(cap, resize_to=SMALL_SIZE)
        cap.release()

    # --- シナリオ C: 早期縮小 ＋ 3フレームに1回処理 ---
    cap = open_video(video_path) if have_video else None
    if cap is not None and cap.isOpened():
        results["C: resize + skip(3)"] = run_resize_and_skip(cap, SMALL_SIZE, every_n=3)
        cap.release()

    # --- シナリオ D: grab/retrieve で安価スキップ ＋ 縮小 ---
    cap = open_video(video_path) if have_video else None
    if cap is not None and cap.isOpened():
        results["D: grab/retrieve skip(3)"] = run_grab_retrieve(cap, SMALL_SIZE, every_n=3)
        cap.release()

    if not results:
        # 動画が書けない環境でも必ず exit 0 で何か残す（合成ソースで縮小+スキップだけ実演）。
        print("[fallback] 動画が使えないため、合成ソースで縮小+スキップのみ計測します。")
        from stream_helpers import SyntheticCapture
        results["C: resize + skip(3) [synthetic]"] = run_resize_and_skip(
            SyntheticCapture(NUM_FRAMES, SRC_SIZE), SMALL_SIZE, every_n=3
        )

    # --- 結果を表で表示 ---
    print(f"\n=== リアルタイム最適化の比較（ソースFPS={SRC_FPS:.0f}を上回れば実時間に追いつける）===")
    print(f"{'scenario':30s} {'seen':>5s} {'proc':>5s} {'rate':>8s} {'p50ms':>8s} {'p99ms':>8s}")
    for name, r in results.items():
        print(f"{name:30s} {r['seen']:5d} {r['processed']:5d} "
              f"{r['stream_rate']:8.1f} {r['p50_ms']:8.2f} {r['p99_ms']:8.2f}")

    # --- ストリーム消化レートの棒グラフ ---
    names = list(results.keys())
    rate_vals = [results[n]["stream_rate"] for n in names]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(range(len(names)), rate_vals, color="#4c78a8")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([n.split(":")[0] for n in names])  # 軸ラベルは ASCII の見出しだけ
    ax.axvline(SRC_FPS, color="crimson", linestyle="--", label=f"source FPS = {SRC_FPS:.0f}")
    ax.set_xlabel("stream consumption rate [frames/s] (higher is better)")
    ax.set_title("CPU real-time optimization: resize + skip + grab raise the rate")
    ax.legend(loc="lower right")
    for i, v in enumerate(rate_vals):
        ax.text(v, i, f" {v:.0f}", va="center")
    fig.tight_layout()
    fig.savefig(out / "02_throughput_compare.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("\n[plot] ストリーム消化レートの比較 -> 02_throughput_compare.png")

    print("\n完了。縮小とスキップで処理FPSが上がること、grab/retrieve の使い方を確認しました。")


if __name__ == "__main__":
    main()
