"""use_case.py — 実践ユースケース: 防犯カメラ風『動体検知アラート録画』ツール（ミニDVR）。

この章で学んだ「背景差分による動体検出」を、そのまま現場で使える**小さな実アプリ**に
仕立てた“動く出発点”です。固定カメラの映像を監視し、動くもの（侵入者・通行人・荷物など）が
現れた区間を **1つの「アラートイベント」** としてまとめ、

  1. イベントごとに代表フレームの**スナップショット画像**（検出枠つき）を保存し、
  2. いつ・どれくらいの大きさの動きがあったかを **アラートログ（CSV / JSON Lines）** に残す。

という、市販の防犯カメラ（モーション録画機能）の最小核をCPUだけで再現します。
ネット・GPU・Webカメラが無くても **必ず exit 0** で完走します（合成フォールバック）。

--------------------------------------------------------------------------
data の置き方（実データ優先・無ければ合成）
--------------------------------------------------------------------------
  - 実映像を使う場合: 監視映像を `data/11_realtime_stream/` に動画ファイルで置いてください。
        対応拡張子: .mp4 / .mov / .avi / .mkv / .webm （複数あれば名前順で先頭を使用）
        例:  data/11_realtime_stream/front_door.mp4
    特定ファイルを直接指定したいときは環境変数 `USECASE_VIDEO=/path/to/clip.mp4`。
  - データが無い場合: 「静止背景の上を“侵入者”が3回横切る」合成監視映像をその場で生成して
    使います（途中に“何も動かない静かな区間”を挟むので、アラートのON/OFFがはっきり学べる）。

--------------------------------------------------------------------------
mini_project.py との違い（役割分担）
--------------------------------------------------------------------------
  - `mini_project.py` は「CPUで実時間に追いつかせる**性能エンジニアリング**」が主役で、
    縮小・スキップ・スレッド分離・フレームドロップを“数値（レイテンシ/スループット/ドロップ率）”で
    比較するのが目的でした。検出は手段で、結果はプロファイル図です。
  - こちら `use_case.py` は「検出結果を**業務的に使い切る**」のが主役です。動体を
    『イベント』として状態機械でまとめ、デバウンス（短いノイズで誤発報しない）と
    クールダウン（連続発報の抑制）を入れ、**スナップショット＋構造化アラートログ**という
    “そのまま運用に乗る成果物”を出します。＝検出そのものよりも「**録画トリガと記録**」が主題。

--------------------------------------------------------------------------
拡張アイデア（ここから先は自分で育てる練習）
--------------------------------------------------------------------------
  - 通知連携: イベント発火時に Slack/Discord の Webhook や メール、MQTT/LINE へ送る。
  - 実時計: フレーム番号→秒の代わりに `datetime.now()` を使い、実際の発生時刻でログを残す。
  - 監視ROI: 画面の一部だけ監視したいとき、関心領域のマスクを掛けて屋外の木の揺れ等を無視する。
  - プリロール録画: スナップショットの代わりに、イベント前後を含む短い動画クリップを
    リングバッファ（`collections.deque`）から `cv2.VideoWriter` で書き出す。
  - 人だけに反応: 背景差分は“動いた画素”しか分からないので、検出枠を第18回以降の
    物体検出（人クラスのみ通す）に通して『人が来た時だけ』アラートにする。
  - 実運用化: 第3〜5節のスレッド/プロセス分離＋フレームドロップと組み合わせ、RTSP/ライブ入力に
    差し替えれば、低遅延で動き続ける本物の監視アプリになる。

実行:
    uv run python lectures/11_realtime_stream/use_case.py
結果（スナップショット・ログ・タイムライン図）は outputs/11_realtime_stream/ に保存（画面表示なし）。
"""

from __future__ import annotations

import csv
import json
import os
import pathlib
import sys
from collections.abc import Iterator

import cv2
import numpy as np

# headless 環境でも図を保存できるよう、pyplot より前に Agg バックエンドを固定する。
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# 共通ヘルパ（出力先・背景生成・プロジェクトルート）は再利用する。
# ※ 数字始まりの 01_/02_… は import できないため、検出ロジックはこの中に自己完結で書く。
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from stream_helpers import (  # noqa: E402
    PROJECT_ROOT,
    make_static_background,
    output_dir,
)

MODULE_ID = "11_realtime_stream"
DATA_DIR = PROJECT_ROOT / "data" / MODULE_ID
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

# --- 監視・検出パラメータ（コメントで意味を明示。現場では映像に合わせて調整する） ---
PROC_H = 360          # 処理解像度（高さ）。重い処理の前に縮小して CPU を節約する定石。
WARMUP = 25           # 背景モデルの立ち上がり。最初の数十枚は前景だらけになるのでアラート対象外。
MIN_AREA = 400        # この画素数未満の動き（ノイズ・葉のゆらぎ等）は無視する下限面積。
MIN_CONSEC = 3        # この枚数連続で動きがあって初めてイベント開始（単発ノイズで誤発報しない＝デバウンス）。
COOLDOWN = 12         # この枚数連続で動きが無くなったらイベント終了（細切れ発報を防ぐ＝クールダウン）。

# 合成監視映像のパラメータ
SYN_FRAMES = 200
SYN_FPS = 20.0


# =====================================================================
# 入力ソース: 実データ（data/）優先・無ければ合成監視映像
# =====================================================================

def find_video() -> pathlib.Path | None:
    """監視に使う動画を1本選ぶ。環境変数 USECASE_VIDEO 優先、無ければ data/ を名前順で探索。"""
    env = os.environ.get("USECASE_VIDEO")
    if env:
        p = pathlib.Path(env)
        return p if p.exists() else None
    if DATA_DIR.exists():
        vids = sorted(p for p in DATA_DIR.iterdir() if p.suffix.lower() in VIDEO_EXTS)
        if vids:
            return vids[0]
    return None


def resize_to_height(frame: np.ndarray, target_h: int) -> np.ndarray:
    """アスペクト比を保ったまま高さを target_h に縮小する（縮小は INTER_AREA が定石）。"""
    h, w = frame.shape[:2]
    if h == target_h:
        return frame
    new_w = max(1, int(round(w * target_h / h)))
    return cv2.resize(frame, (new_w, target_h), interpolation=cv2.INTER_AREA)


def synthetic_security_frames(
    num_frames: int, size: tuple[int, int], seed: int = 11
) -> Iterator[tuple[int, np.ndarray]]:
    """合成の監視映像（静止背景＋たまに横切る“侵入者”＋センサノイズ）を1枚ずつ返す。

    本物の監視カメラ同様、ほとんどの時間は「何も動かない」。決められた区間だけ明るい塊
    （侵入者）が画面を横切る。これにより『静か→動き→静か』が繰り返され、アラートの
    発火・終了・クールダウンを現実的に体験できる（mini_project の“常に円が動く”映像とは別物）。
    """
    height, width = size
    rng = np.random.default_rng(seed)
    background = make_static_background(height, width, rng)

    # 侵入者が現れる区間 [start, end)。間に“静かな区間”を挟む。
    windows = [(35, 75), (110, 150), (175, num_frames - 2)]
    paths = []  # 各区間ごとの入場点→退場点（直線移動）
    for _s, _e in windows:
        entry = np.array([rng.uniform(width * 0.05, width * 0.20),
                          rng.uniform(height * 0.20, height * 0.80)])
        leave = np.array([rng.uniform(width * 0.80, width * 0.95),
                          rng.uniform(height * 0.20, height * 0.80)])
        paths.append((entry, leave))

    radius = int(min(height, width) * 0.06)  # 侵入者の大きさ
    for idx in range(num_frames):
        frame = background.copy()
        for (s, e), (entry, leave) in zip(windows, paths):
            if s <= idx < e:
                t = (idx - s) / max(e - s - 1, 1)            # 0→1 で横断
                pos = entry * (1.0 - t) + leave * t
                cv2.circle(frame, (int(pos[0]), int(pos[1])), radius,
                           (245, 245, 245), thickness=-1, lineType=cv2.LINE_AA)
        noise = rng.normal(0.0, 4.0, size=frame.shape).astype(np.float32)  # センサノイズ
        frame = np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        yield idx, frame


def build_source(proc_h: int) -> tuple[Iterator[tuple[int, np.ndarray]], float, str]:
    """入力を準備して (フレーム列, fps, ラベル) を返す。実データがあれば動画、無ければ合成。"""
    video = find_video()
    if video is not None:
        cap = cv2.VideoCapture(str(video))
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS)
            fps = fps if fps and fps > 0 else 30.0

            def gen() -> Iterator[tuple[int, np.ndarray]]:
                i = 0
                while True:
                    ret, frame = cap.read()      # ret=False で末尾/切断 → ループ終了
                    if not ret:
                        break
                    yield i, resize_to_height(frame, proc_h)
                    i += 1
                cap.release()

            return gen(), fps, f"real video: {video.name}"
        cap.release()
        print(f"[source] {video} を開けませんでした。合成監視映像にフォールバックします。")

    size = (proc_h, int(round(proc_h * 16 / 9)))  # 16:9 の合成画面（例: 360x640）
    return synthetic_security_frames(SYN_FRAMES, size, seed=11), SYN_FPS, "synthetic security feed"


# =====================================================================
# 動体検出（背景差分 → 影除去 → モルフォロジ掃除 → 輪郭→外接矩形）
# =====================================================================

def detect_motion(
    subtractor, frame: np.ndarray
) -> tuple[list[tuple[int, int, int, int]], int]:
    """1フレームの動体検出。MIN_AREA 以上の外接矩形リストと、最大輪郭面積を返す。

    流れは本章の定石どおり:
      apply()→前景マスク(影は127) → threshold(>200)で影を捨て2値化 →
      オープニング(ノイズ除去)→クロージング(穴埋め) → findContours(4系は2つ返し)。
    """
    raw = subtractor.apply(frame)
    _, binary = cv2.threshold(raw, 200, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    clean = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes: list[tuple[int, int, int, int]] = []
    largest = 0
    for c in contours:
        area = int(cv2.contourArea(c))
        if area < MIN_AREA:
            continue                       # 小さすぎる動き（ノイズ）は無視
        boxes.append(cv2.boundingRect(c))
        largest = max(largest, area)
    return boxes, largest


def annotate_alert(
    img: np.ndarray, boxes: list[tuple[int, int, int, int]],
    event_id: int, time_s: float, area: int
) -> np.ndarray:
    """アラート用スナップショットを作る（検出枠＋ヘッダ。元画像は壊さずコピーに描く）。"""
    out = img.copy()
    for (x, y, w, h) in boxes:
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 0, 255), 2)  # 赤枠（BGR）
    header = f"ALERT #{event_id}  t={time_s:.1f}s  boxes={len(boxes)}  area={area}"
    cv2.rectangle(out, (0, 0), (out.shape[1], 26), (0, 0, 0), thickness=-1)  # 黒帯
    cv2.putText(out, header, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (0, 0, 255), 1, cv2.LINE_AA)  # 図中文字は ASCII のみ
    return out


# =====================================================================
# イベント状態機械: 動きの始まり/終わりを「1イベント」にまとめる
# =====================================================================

def process_stream(
    frames: Iterator[tuple[int, np.ndarray]],
    fps: float,
    snap_dir: pathlib.Path,
    max_frames: int,
) -> tuple[list[dict], list[tuple[int, int, bool]]]:
    """フレーム列を1枚ずつ検出し、動体イベントを検出してスナップショット＋ログ化する。

    状態機械（録画ON/OFF）:
      - 非録画中に MIN_CONSEC 枚連続で動きを見たら → イベント開始（録画ON）。
      - 録画中に COOLDOWN 枚連続で動きが無くなったら → イベント終了（スナップショット保存＋ログ追記）。
      - 録画中は「最も動きが大きかったフレーム」を代表（peak）として覚えておく。
    戻り値: (確定イベントのログ辞書リスト, タイムライン用の(frame, largest_area, is_warmup)列)。
    """
    sub = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=24, detectShadows=True)

    events: list[dict] = []
    series: list[tuple[int, int, bool]] = []
    event: dict | None = None
    event_id = 0
    motion_run = 0   # 非録画中: 連続で動きを見た枚数
    quiet_run = 0    # 録画中: 連続で動きが無かった枚数

    for idx, frame in frames:
        if idx >= max_frames:
            break
        boxes, largest = detect_motion(sub, frame)
        warming = idx < WARMUP
        series.append((idx, largest, warming))
        if warming:
            continue  # 立ち上がり中は背景が未学習なのでアラート判定しない

        motion = largest >= MIN_AREA

        if event is None:
            # --- 非録画中: イベント開始を待つ ---
            if motion:
                motion_run += 1
                if motion_run >= MIN_CONSEC:
                    event_id += 1
                    event = {
                        "event_id": event_id,
                        "start_frame": idx,
                        "end_frame": idx,
                        "peak_frame": idx,
                        "peak_area": largest,
                        "peak_boxes": boxes,
                        "peak_img": frame.copy(),
                    }
                    quiet_run = 0
            else:
                motion_run = 0
        else:
            # --- 録画中: ピーク更新と終了判定 ---
            if motion:
                quiet_run = 0
                event["end_frame"] = idx
                if largest > event["peak_area"]:
                    event["peak_area"] = largest
                    event["peak_frame"] = idx
                    event["peak_boxes"] = boxes
                    event["peak_img"] = frame.copy()
            else:
                quiet_run += 1
                if quiet_run >= COOLDOWN:
                    events.append(_finalize_event(event, fps, snap_dir))
                    event = None
                    motion_run = 0

    # ストリーム末尾でまだ録画中なら、開いたままのイベントを確定させる。
    if event is not None:
        events.append(_finalize_event(event, fps, snap_dir))

    return events, series


def _finalize_event(event: dict, fps: float, snap_dir: pathlib.Path) -> dict:
    """1イベントを確定: 代表フレームを注釈付きで保存し、ログ用の辞書（画像なし）を返す。"""
    eid = event["event_id"]
    start_t = event["start_frame"] / fps
    end_t = event["end_frame"] / fps
    duration = max(end_t - start_t, 1.0 / fps)
    snapshot = f"alert_{eid:04d}.png"

    annotated = annotate_alert(
        event["peak_img"], event["peak_boxes"], eid,
        event["peak_frame"] / fps, event["peak_area"],
    )
    cv2.imwrite(str(snap_dir / snapshot), annotated)

    return {
        "event_id": eid,
        "start_frame": event["start_frame"],
        "end_frame": event["end_frame"],
        "start_time_s": round(start_t, 2),
        "duration_s": round(duration, 2),
        "peak_frame": event["peak_frame"],
        "peak_boxes": len(event["peak_boxes"]),
        "peak_area_px": event["peak_area"],
        "snapshot": snapshot,
    }


# =====================================================================
# 成果物の出力: アラートログ（CSV / JSONL）とタイムライン図
# =====================================================================

def write_alert_log(events: list[dict], out: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    """アラートを CSV（表計算/Excel向け）と JSON Lines（プログラム連携向け）の両方で保存。"""
    csv_path = out / "use_case_alerts.csv"
    jsonl_path = out / "use_case_alerts.jsonl"
    fields = ["event_id", "start_frame", "end_frame", "start_time_s",
              "duration_s", "peak_frame", "peak_boxes", "peak_area_px", "snapshot"]

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(events)

    with jsonl_path.open("w", encoding="utf-8") as f:
        for ev in events:
            f.write(json.dumps(ev, ensure_ascii=False) + "\n")

    return csv_path, jsonl_path


def plot_timeline(
    series: list[tuple[int, int, bool]], events: list[dict], out: pathlib.Path
) -> pathlib.Path:
    """動き量の時系列に、warm-up 区間と各アラート区間を重ねたタイムライン図を保存。"""
    idxs = [s[0] for s in series]
    areas = [s[1] for s in series]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(idxs, areas, color="#4c78a8", linewidth=1.4, label="largest motion area [px]")
    ax.axhline(MIN_AREA, color="gray", linestyle="--", linewidth=1.0,
               label=f"trigger threshold = {MIN_AREA}")
    if idxs:
        ax.axvspan(idxs[0], min(WARMUP, idxs[-1]), color="#dddddd", alpha=0.6,
                   label="warm-up (no alert)")
    for i, ev in enumerate(events):
        ax.axvspan(ev["start_frame"], ev["end_frame"], color="#e45756", alpha=0.25,
                   label="alert event" if i == 0 else None)
    ax.set_xlabel("frame index")
    ax.set_ylabel("largest motion area [px]")
    ax.set_title("Use Case: motion-triggered security alerts (event spans shaded)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = out / "use_case_timeline.png"
    fig.savefig(str(path), dpi=120)
    plt.close(fig)
    return path


# =====================================================================
# メイン
# =====================================================================

def main() -> None:
    out = output_dir()
    snap_dir = out / "use_case_snapshots"
    snap_dir.mkdir(parents=True, exist_ok=True)
    # 再実行時に古いスナップショットが残らないよう掃除（イベント数が前回と変わっても整合する）。
    for old in snap_dir.glob("alert_*.png"):
        old.unlink()

    print("=" * 68)
    print("実践ユースケース: 防犯カメラ風『動体検知アラート録画』ツール（CPUのみ）")
    print("=" * 68)

    frames, fps, label = build_source(PROC_H)
    max_frames = int(os.environ.get("USECASE_MAX_FRAMES", "900"))
    print(f"  入力ソース   : {label}   (fps={fps:.1f}, 最大{max_frames}フレームまで処理)")
    print(f"  検出設定     : warmup={WARMUP}  min_area={MIN_AREA}  "
          f"debounce={MIN_CONSEC}枚  cooldown={COOLDOWN}枚")

    events, series = process_stream(frames, fps, snap_dir, max_frames)

    csv_path, jsonl_path = write_alert_log(events, out)
    timeline_path = plot_timeline(series, events, out)

    # --- コンソールにアラート一覧を表示（運用時はこの行がそのまま監視ログになる） ---
    print(f"\n  検出アラート: {len(events)} 件")
    for ev in events:
        print(f"    - ALERT #{ev['event_id']:02d}  "
              f"t={ev['start_time_s']:6.1f}s  dur={ev['duration_s']:4.1f}s  "
              f"boxes={ev['peak_boxes']}  peak_area={ev['peak_area_px']:6d}px  "
              f"-> {ev['snapshot']}")

    print("\n  保存物 (outputs/11_realtime_stream/):")
    print(f"    - スナップショット : use_case_snapshots/alert_*.png  ({len(events)} 枚, 検出枠つき)")
    print(f"    - アラートログCSV  : {csv_path.name}")
    print(f"    - アラートログJSONL: {jsonl_path.name}")
    print(f"    - タイムライン図   : {timeline_path.name}")
    print("\n完成: 背景差分→イベント状態機械(デバウンス/クールダウン)→スナップ＋構造化ログ、を外部依存ゼロで通した。")
    print("発展: data/11_realtime_stream/ に実映像を置く / 通知Webhook連携 / ROI監視 / プリロール動画録画 へ。")


if __name__ == "__main__":
    main()
