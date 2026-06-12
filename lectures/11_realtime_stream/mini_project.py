"""mini_project.py — 章末ミニプロジェクト: CPUだけで成立させるリアルタイム動体検出ストリーム。

この回の学び（背景差分・CPU最適化・スレッド分離+ドロップ・性能プロファイル）を
1 本の「ストリームアプリ」に統合する完成形。合成フレーム（静止背景の上を円が動く）を
入力に、次の 4 段を通して『GPU 無しでも実時間に追いつく』ことを数値で示す:

  STAGE 1  動体検出パイプライン:
           早期 resize(INTER_AREA) → MOG2 背景差分 → モルフォロジ掃除 → 輪郭→外接矩形
  STAGE 2  CPU 最適化（同期計測・決定的）:
           「原寸・毎フレーム」 vs 「縮小＋3枚に1回」 でストリーム消化レートを比較
  STAGE 3  スレッド分離（producer/consumer）＋フレームドロップ:
           「無制限キュー（捨てない）」 vs 「maxsize=1 + put_nowait（最新だけ）」で
           各フレームの『齢(age=生成→処理の遅れ)』とドロップ率を比較
  STAGE 4  プロファイル出力: p50/p99 レイテンシ・処理FPS(EMA)・ドロップ率を JSON と図に

外部データもネットも GPU も不要（合成フレームのみ・CPU 完結）。実運用では入力を
Webカメラ/動画/RTSP/ライブ配信に差し替えれば、同じ骨格（取得と処理を分け、キューで
つなぎ、満杯なら落とす）がそのまま効く。これは最終章の Cluster-CLIP ストリーム
パイプラインの最小核でもある。

実行:
    uv run python lectures/11_realtime_stream/mini_project.py
結果（画像・JSON）は outputs/11_realtime_stream/ に保存される（画面表示はしない）。
"""

from __future__ import annotations

import json
import pathlib
import queue
import sys
import threading
import time

import cv2
import numpy as np

# headless 環境でも図を保存できるよう、pyplot より前に Agg を指定する。
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# 共通ヘルパ（合成フレーム生成・出力先・FPS/レイテンシ計測）は再利用する。
# ※ digit 始まりの 01_/02_… は import できないので、検出処理はこの中に自己完結で書く。
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from stream_helpers import (  # noqa: E402
    FpsMeter,
    output_dir,
    percentiles,
    synthetic_frames,
)

# --- パラメータ（環境差を吸収しつつ短時間で完走するよう控えめに） ---
SRC_SIZE = (360, 640)     # (H, W) 原寸（やや重い）
SMALL_SIZE = (180, 320)   # (H, W) 早期縮小後（軽い）
WARMUP_FRAMES = 70        # STAGE1/2 で使うフレーム数（最初の数十枚は背景 warm-up）
PRODUCE_FPS = 60.0        # STAGE3 で入力が届く速さ（1秒60枚 ≒ 16.7ms間隔）
CONSUME_EXTRA_S = 0.045   # STAGE3 consumer の追加コスト（下流の重い推論を模擬）。
                          # producer 間隔(≈20ms)より大きくして『処理が追いつかない』状況を作る。
STREAM_FRAMES = 72        # STAGE3 のフレーム数


# =====================================================================
# STAGE 1 の部品（背景差分による動体検出。自己完結で実装）
# =====================================================================

def clean_mask(raw_mask: np.ndarray) -> np.ndarray:
    """生の前景マスクを 0/255 の2値マスクへ掃除する（影127除去＋モルフォロジ）。

    1. threshold(>200) で影(127)を捨て、確実な前景(255)だけ残す。
    2. オープニング(iterations=1)で『ぽつぽつ誤検出』を消す。
    3. クロージング(iterations=2)で『物体内部の穴』を埋める。
    """
    _, binary = cv2.threshold(raw_mask, 200, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    return cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=2)


def detect_boxes(mask: np.ndarray, min_area: int = 120) -> list[tuple[int, int, int, int]]:
    """2値マスクから輪郭を取り、面積 >= min_area のものだけ外接矩形 (x,y,w,h) で返す。

    findContours は OpenCV 4 系で (contours, hierarchy) の2つ返し（3系の3つ返しと違う）。
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes: list[tuple[int, int, int, int]] = []
    for c in contours:
        if cv2.contourArea(c) < min_area:
            continue
        boxes.append(cv2.boundingRect(c))
    return boxes


def detect_stage(
    subtractor, frame: np.ndarray, resize_to: tuple[int, int] | None
) -> tuple[list[tuple[int, int, int, int]], np.ndarray]:
    """1フレーム分の検出（早期縮小→MOG2→掃除→輪郭）。boxes と掃除後マスクを返す。

    resize_to を渡すと『重い処理の前』に縮小する（CPU 最適化の定石）。box 座標は
    縮小後の座標系のままにしておく（呼び出し側で必要なら拡大すればよい）。
    """
    if resize_to is not None:
        h, w = resize_to
        frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
    raw = subtractor.apply(frame)
    clean = clean_mask(raw)
    return detect_boxes(clean), clean


def annotate(frame_bgr: np.ndarray, boxes: list[tuple[int, int, int, int]]) -> np.ndarray:
    """検出矩形を緑枠で描く（元フレームは壊さずコピーに描く）。"""
    out = frame_bgr.copy()
    for (x, y, w, h) in boxes:
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(out, f"objects: {len(boxes)}", (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    return out


def stage1_detection_sample(out: pathlib.Path) -> dict:
    """STAGE 1: 背景差分の動体検出を回し、十分に学習済みのフレームを 3 枚パネルで保存する。"""
    frames = list(synthetic_frames(num_frames=WARMUP_FRAMES, size=SRC_SIZE, seed=11))
    sub = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=24, detectShadows=True)

    boxes: list[tuple[int, int, int, int]] = []
    clean = np.zeros(SRC_SIZE, dtype=np.uint8)
    snapshot_at = WARMUP_FRAMES - 12   # 十分に背景を学習し終えたあたり
    for idx, frame in frames:
        boxes, clean = detect_stage(sub, frame, resize_to=None)
        if idx == snapshot_at:
            panel = np.hstack([
                frame,
                cv2.cvtColor(clean, cv2.COLOR_GRAY2BGR),
                annotate(frame, boxes),
            ])
            cv2.imwrite(str(out / "mini_project_detection.png"), panel)
    print(f"[STAGE1] 動体検出: 最終フレームで {len(boxes)} 個を検出 -> mini_project_detection.png")
    return {"detected_objects": len(boxes), "frames": WARMUP_FRAMES}


# =====================================================================
# STAGE 2: CPU 最適化（同期・決定的）— 縮小＋スキップで消化レートが上がる
# =====================================================================

def run_sync_pipeline(resize_to: tuple[int, int] | None, every_n: int) -> dict:
    """合成フレームを同期処理し、ストリーム消化レートと検出段レイテンシ(p50/p99)を測る。

    every_n=1 なら毎フレーム、3 なら3枚に1回だけ検出する（間引き）。
    消化レート = 見たフレーム数 / 総時間。ソースFPS を上回れば実時間に追いつける。
    """
    sub = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=24, detectShadows=True)
    latencies_ms: list[float] = []
    seen = 0
    processed = 0
    t0 = time.perf_counter()
    for _idx, frame in synthetic_frames(num_frames=WARMUP_FRAMES, size=SRC_SIZE, seed=12):
        seen += 1
        if seen % every_n != 0:
            continue  # スキップ: 重い検出を呼ばない（前回結果を使う想定）
        t = time.perf_counter()
        detect_stage(sub, frame, resize_to=resize_to)
        latencies_ms.append((time.perf_counter() - t) * 1000.0)
        processed += 1
    total = time.perf_counter() - t0
    p50, p99 = percentiles(latencies_ms)
    return {
        "seen": seen,
        "processed": processed,
        "total_s": total,
        "stream_rate": seen / total if total > 0 else 0.0,
        "p50_ms": p50,
        "p99_ms": p99,
    }


def stage2_cpu_optimization(out: pathlib.Path) -> dict:
    """STAGE 2: 「原寸・毎フレーム」 vs 「縮小＋3枚に1回」 を消化レートで比較し棒グラフに。"""
    full = run_sync_pipeline(resize_to=None, every_n=1)
    opt = run_sync_pipeline(resize_to=SMALL_SIZE, every_n=3)

    print("[STAGE2] CPU 最適化（消化レート [frames/s]、高いほど良い）:")
    print(f"          full-res every frame : rate={full['stream_rate']:8.1f}  "
          f"p50={full['p50_ms']:.2f}ms p99={full['p99_ms']:.2f}ms")
    print(f"          resize + skip(3)     : rate={opt['stream_rate']:8.1f}  "
          f"p50={opt['p50_ms']:.2f}ms p99={opt['p99_ms']:.2f}ms")
    speedup = opt["stream_rate"] / full["stream_rate"] if full["stream_rate"] > 0 else 0.0
    print(f"          → 縮小＋スキップで消化レート約 {speedup:.1f} 倍（不要な重い検出を省ける）")

    names = ["full-res\nevery frame", "resize +\nskip(3)"]
    rates = [full["stream_rate"], opt["stream_rate"]]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(names, rates, color=["#bbbbbb", "#4c78a8"])
    ax.axhline(30.0, color="crimson", linestyle="--", label="source FPS = 30")
    ax.set_ylabel("stream consumption rate [frames/s] (higher is better)")
    ax.set_title("STAGE2: early resize + frame skip raise the throughput")
    ax.legend(loc="upper left")
    for i, v in enumerate(rates):
        ax.text(i, v, f"{v:.0f}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(str(out / "mini_project_cpu_opt.png"), dpi=120)
    plt.close(fig)
    return {"full": full, "resize_skip": opt, "speedup": round(speedup, 2)}


# =====================================================================
# STAGE 3: スレッド分離（producer/consumer）＋フレームドロップ
# =====================================================================

def _producer(
    q: "queue.Queue",
    stop_event: threading.Event,
    counters: dict,
    drop_when_full: bool,
) -> None:
    """取得専用スレッド: 一定間隔でフレームを生成し、生成時刻を添えてキューへ入れる。

    drop_when_full=True なら put_nowait で、満杯(queue.Full)なら新フレームを捨てる
    （maxsize=1 と組み合わせて『最新1枚だけ』を保つ）。False なら捨てずに溜める。
    """
    interval = 1.0 / PRODUCE_FPS
    for _idx, frame in synthetic_frames(num_frames=STREAM_FRAMES, size=SRC_SIZE, seed=13):
        if stop_event.is_set():
            break
        counters["produced"] += 1
        item = (time.perf_counter(), frame)
        if drop_when_full:
            try:
                q.put_nowait(item)
            except queue.Full:
                counters["dropped"] += 1   # 満杯 → 新フレームを捨てる（遅延蓄積を防ぐ）
        else:
            q.put(item)                    # 捨てない（満杯なら空くまで待つ＝在庫が溜まる）
        time.sleep(interval)
    stop_event.set()                       # 入力が尽きたら停止を通知


def _consume(
    q: "queue.Queue",
    stop_event: threading.Event,
    counters: dict,
    ages_ms: list[float],
    meter: FpsMeter,
    resize_to: tuple[int, int] | None,
) -> None:
    """処理側（メインスレッド）: キューからフレームを取り出し、背景差分で動体検出する。

    各フレームの齢(age=取り出した時刻-生成時刻)を記録する＝『映像と処理のズレ』。
    consumer は MOG2 検出＋追加コストで、producer より少し遅い（＝溜まりやすい）状況を作る。
    """
    sub = cv2.createBackgroundSubtractorMOG2(history=120, varThreshold=24, detectShadows=False)
    while True:
        try:
            stamp, frame = q.get(timeout=0.5)
        except queue.Empty:
            if stop_event.is_set():
                break       # producer が終わっていて在庫も無い → 完了
            continue
        detect_stage(sub, frame, resize_to=resize_to)  # 実際の検出（背景差分）
        time.sleep(CONSUME_EXTRA_S)                     # 下流の重い推論を模擬（決定的に遅くする）
        ages_ms.append((time.perf_counter() - stamp) * 1000.0)
        meter.tick()
        counters["consumed"] += 1


def run_threaded(drop_when_full: bool, maxsize: int, resize_to: tuple[int, int] | None) -> dict:
    """1シナリオ実行: producer スレッドを起こし、consumer をメインで回して計測を返す。"""
    q: "queue.Queue" = queue.Queue(maxsize=maxsize)
    stop_event = threading.Event()
    counters = {"produced": 0, "dropped": 0, "consumed": 0}
    ages_ms: list[float] = []
    meter = FpsMeter(alpha=0.2)

    t = threading.Thread(
        target=_producer, args=(q, stop_event, counters, drop_when_full), daemon=True
    )
    t.start()
    _consume(q, stop_event, counters, ages_ms, meter, resize_to)
    t.join(timeout=2.0)

    p50, p99 = percentiles(ages_ms)
    drop_rate = counters["dropped"] / max(counters["produced"], 1)
    counters.update({
        "ages_ms": ages_ms,
        "mean_age_ms": float(np.mean(ages_ms)) if ages_ms else 0.0,
        "p50_age_ms": p50,
        "p99_age_ms": p99,
        "drop_rate": drop_rate,
        "proc_fps_ema": meter.fps,
    })
    return counters


def stage3_thread_drop(out: pathlib.Path) -> dict:
    """STAGE 3: 「無制限キュー（捨てない）」 vs 「maxsize=1+ドロップ＋縮小」 を齢で比較。"""
    print("[STAGE3] スレッド分離 + フレームドロップ:")
    naive = run_threaded(drop_when_full=False, maxsize=10_000, resize_to=None)
    print(f"          無制限キュー : produced={naive['produced']} consumed={naive['consumed']} "
          f"drop={naive['dropped']} mean_age={naive['mean_age_ms']:.0f}ms "
          f"p99_age={naive['p99_age_ms']:.0f}ms")
    realtime = run_threaded(drop_when_full=True, maxsize=1, resize_to=SMALL_SIZE)
    print(f"          maxsize=1+drop: produced={realtime['produced']} consumed={realtime['consumed']} "
          f"drop={realtime['dropped']} mean_age={realtime['mean_age_ms']:.0f}ms "
          f"p99_age={realtime['p99_age_ms']:.0f}ms")
    print(f"          → ドロップ率 {realtime['drop_rate']*100:.0f}% と引き換えに齢(遅延)の蓄積を断ち切る")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(naive["ages_ms"], label="unbounded queue (no drop)", color="crimson")
    ax.plot(realtime["ages_ms"], label="maxsize=1 + drop + resize", color="#2a9d8f")
    ax.set_xlabel("consumed frame index")
    ax.set_ylabel("frame age at consumption [ms]  (lower = lower latency)")
    ax.set_title("STAGE3: dropping the newest keeps latency bounded")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(out / "mini_project_latency.png"), dpi=120)
    plt.close(fig)

    # JSON 用に ages_ms（巨大になり得る）は要約だけ残す。
    def _summ(d: dict) -> dict:
        return {
            "produced": d["produced"], "consumed": d["consumed"], "dropped": d["dropped"],
            "mean_age_ms": round(d["mean_age_ms"], 2),
            "p50_age_ms": round(d["p50_age_ms"], 2),
            "p99_age_ms": round(d["p99_age_ms"], 2),
            "drop_rate": round(d["drop_rate"], 3),
            "proc_fps_ema": round(d["proc_fps_ema"], 2),
        }
    return {"naive": _summ(naive), "realtime": _summ(realtime)}


# =====================================================================
# メイン
# =====================================================================

def main() -> None:
    out = output_dir()
    print("=" * 68)
    print("章末ミニプロジェクト: CPUだけで成立させるリアルタイム動体検出ストリーム")
    print("=" * 68)

    metrics: dict = {}
    metrics["stage1_detection"] = stage1_detection_sample(out)
    metrics["stage2_cpu_opt"] = stage2_cpu_optimization(out)
    metrics["stage3_thread_drop"] = stage3_thread_drop(out)

    # --- まとめ図（検出サンプル＋2つの比較を1枚に） ---
    det = cv2.imread(str(out / "mini_project_detection.png"))
    opt = cv2.imread(str(out / "mini_project_cpu_opt.png"))
    lat = cv2.imread(str(out / "mini_project_latency.png"))
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    axes[0, 0].imshow(cv2.cvtColor(det, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("STAGE1: background-subtraction detection")
    axes[0, 1].imshow(cv2.cvtColor(opt, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title("STAGE2: CPU optimization (throughput)")
    axes[1, 0].imshow(cv2.cvtColor(lat, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title("STAGE3: thread split + drop (latency)")
    axes[1, 1].axis("off")
    summary_text = (
        f"CPU opt speedup: x{metrics['stage2_cpu_opt']['speedup']}\n\n"
        f"no-drop  mean age: {metrics['stage3_thread_drop']['naive']['mean_age_ms']:.0f} ms\n"
        f"drop     mean age: {metrics['stage3_thread_drop']['realtime']['mean_age_ms']:.0f} ms\n"
        f"drop rate        : {metrics['stage3_thread_drop']['realtime']['drop_rate']*100:.0f} %\n"
        f"proc FPS (EMA)   : {metrics['stage3_thread_drop']['realtime']['proc_fps_ema']:.1f}"
    )
    axes[1, 1].text(0.02, 0.95, summary_text, va="top", ha="left",
                    family="monospace", fontsize=12)
    for ax in (axes[0, 0], axes[0, 1], axes[1, 0]):
        ax.axis("off")
    fig.suptitle("Mini Project: realtime motion-detection stream on CPU only")
    fig.tight_layout()
    fig.savefig(str(out / "mini_project_summary.png"), dpi=110)
    plt.close(fig)

    # --- 指標を JSON で保存（再現性・自動評価のため） ---
    json_path = out / "mini_project_metrics.json"
    json_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n  保存物:")
    print("    - 検出サンプル : mini_project_detection.png  ([入力 | 掃除後マスク | 検出枠])")
    print("    - CPU最適化図  : mini_project_cpu_opt.png")
    print("    - レイテンシ図 : mini_project_latency.png")
    print("    - まとめ図     : mini_project_summary.png")
    print(f"    - 指標 JSON    : {json_path.name}")
    print("\n完成: 背景差分→CPU最適化→スレッド分離+ドロップ→プロファイル、を外部依存ゼロで通した。")
    print("発展: 入力を Webカメラ/動画/RTSP/ライブ配信に差し替えれば同じ骨格が実運用で効く。")


if __name__ == "__main__":
    main()
