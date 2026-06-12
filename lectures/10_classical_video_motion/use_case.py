"""use_case.py — 動き検知録画ツール（動いた区間だけを切り出して保存する小ツール）

この回の「実践ユースケース」: 監視カメラ・ドライブレコーダ・野生動物トラップカメラで
定番の **モーション録画（motion-activated recording）** を、古典的な背景差分だけで作ります。
入力映像をフレームごとに「動きあり/なし」で判定し、**動きのあった区間だけ** をイベントとして
切り出し、個別のクリップ（動画）として保存します。ずっと録りっぱなしにせず動いた所だけ残すので
ストレージを大幅に節約できる——これが現場のモーション録画の狙いです。

────────────────────────────────────────────────────────────────────────
mini_project.py との違い（役割分担）
────────────────────────────────────────────────────────────────────────
- `mini_project.py` は四本柱（背景差分・LK・Farneback・CamShift）を 1 本に束ねて
  動体数・EPE・窓成長率・処理FPS を一括レポートする **総合演習（読み物の完成形）** です。
- 本 `use_case.py` は解析結果のレポートではなく、**アプリの最終出力（イベント・クリップ）** を
  作る **現実の小ツール** です。使う技術は背景差分 1 つに絞り、その上に
  「動いた区間をイベントに区切って録画する」という **録画機ロジック**（プリロール/ポストロール/
  クールダウン）だけを薄く足しています。そのまま実運用の動き検知レコーダに拡張できます。

────────────────────────────────────────────────────────────────────────
データ配置（実データ優先・合成フォールバック）
────────────────────────────────────────────────────────────────────────
- `data/10_classical_video_motion/` に **動画ファイル**（*.mp4 / *.avi / *.mov / *.mkv）か
  **連番画像**（*.png / *.jpg）を置くと、それを実入力として使います。
  例: 固定カメラで撮った `data/10_classical_video_motion/hallway.mp4` を置けば、
  人が通った区間だけを自動でクリップ化する実用的なモーションレコーダになります。
- 置かなければ **合成シーン**（静止背景＋ときどき横切る物体）で必ず完走します（exit 0）。
  合成は「静止 → 物体が通る → 静止 → また通る → 静止」と、わざと無音区間を挟むので、
  録画区間とそうでない区間がはっきり分かれ、ツールの動作が一目で確認できます。
- 固定カメラ前提です（背景差分は背景静止が前提）。手ぶれ・激しい照明変化があると
  全面が「動き」と判定されやすいので、その場合は MOTION_THRESH を上げてください。

────────────────────────────────────────────────────────────────────────
拡張アイデア（練習用）
────────────────────────────────────────────────────────────────────────
- しきい値を「前景画素率」ではなく「最大ブロブ面積」「動体の個数」に変えて誤検知を減らす。
- 関心領域（ROI）マスクを掛け、画面の特定エリア（ドア付近など）の動きだけで録画する。
- イベントごとにメール/Slack 通知やサムネイル生成を足して簡易セキュリティ通知システムへ。
- 背景差分(MOG2)を YOLO 等の検出器に差し替え「人が映った時だけ録画」に高度化する。
- リングバッファ＋ `cv2.VideoCapture(0)` のライブループに替えれば実カメラの常時監視録画になる。

実行:
    uv run python lectures/10_classical_video_motion/use_case.py

出力（outputs/10_classical_video_motion/ に保存）:
    use_case_event{k}.mp4         : 動きのあった区間ごとに切り出したクリップ（mp4v）
                                    ※VideoWriter が開けない環境では連番モンタージュ PNG に退避
    use_case_event_montage.png    : 各イベントの代表フレーム（最初/ピーク/最後）を並べた一覧
    use_case_motion_timeline.png  : フレーム毎の動き量の折れ線＋しきい値＋録画区間の帯
    use_case_events.json          : イベント表（開始/終了/長さ/ピーク動き量/節約率など, 機械可読）
"""

from __future__ import annotations

import glob
import json
import math
import os
import sys

# headless 前提: matplotlib は必ず Agg バックエンド（GUI を開かない）。
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402

# 同じフォルダの cv_helpers.py（合成データ生成・出力先管理）だけを借用する。
# ※数字始まりの 01〜04 は import 不可なので、録画ロジックはここに自己完結で書く。
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cv_helpers import make_background, make_sprite, output_dir  # noqa: E402

# ----------------------------------------------------------------------------
# 設定（ここを変えるだけで挙動を調整できる）
# ----------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
DATA_DIR = os.path.join(REPO, "data", "10_classical_video_motion")

VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv")
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp")

MAX_FRAMES = 300        # 実動画は CPU 負荷を抑えるためこのフレーム数で打ち切る
MAX_WIDTH = 480         # 実動画はこの幅に縮小（処理を軽く）
DEFAULT_FPS = 20.0      # FPS が取れないときの既定値（合成・連番画像用）

# --- 録画機の判定パラメータ（モーションレコーダの心臓部）---------------------
WARMUP = 5              # 背景モデルの慣らし運転。最初の数フレームは判定に使わない
MOTION_THRESH = 0.0020  # 前景画素率がこれ以上なら「動きあり」とする（0〜1）
MIN_AREA = 80           # この面積未満の前景ブロブはノイズとして無視（描画・面積集計用）
PRE_ROLL = 3            # 動き検出の少し前から録画に含める（イベントの頭切れ防止）
POST_ROLL = 6           # 動きが止まってからも数フレーム録り続ける（クールダウン）
MIN_EVENT_LEN = 3       # 実際に動いたフレームがこれ未満のイベントは捨てる（誤検知除去）

KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


# ----------------------------------------------------------------------------
# 入力フレームの用意（実データ優先 → 合成フォールバック）
# ----------------------------------------------------------------------------
def find_input():
    """data/10_classical_video_motion/ から動画 or 連番画像を探す。無ければ None。"""
    if not os.path.isdir(DATA_DIR):
        return None
    for ext in VIDEO_EXTS:  # 1) 動画を優先
        hits = sorted(glob.glob(os.path.join(DATA_DIR, f"*{ext}")))
        if hits:
            return ("video", hits[0])
    imgs = []               # 2) 連番画像
    for ext in IMAGE_EXTS:
        imgs += glob.glob(os.path.join(DATA_DIR, f"*{ext}"))
    if len(imgs) >= 2:
        return ("images", sorted(imgs))
    return None


def read_video(path):
    """動画を読み、必要なら縮小して (frames, fps) を返す。"""
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or math.isnan(fps) or fps <= 1.0:
        fps = DEFAULT_FPS  # ライブ/壊れたメタでは 0 が返ることがあるので保険
    frames = []
    while len(frames) < MAX_FRAMES:
        ok, frame = cap.read()
        if not ok:  # ret(成功フラグ)が False なら末尾。必ず確認してから使う
            break
        h, w = frame.shape[:2]
        if w > MAX_WIDTH:  # CPU 負荷を抑えるため縮小
            scale = MAX_WIDTH / w
            frame = cv2.resize(frame, (MAX_WIDTH, int(round(h * scale))))
        frames.append(frame)
    cap.release()
    return frames, float(fps)


def read_images(paths):
    """連番画像をフレームとして読む（サイズは先頭に合わせる）。"""
    frames = []
    target = None
    for p in paths[:MAX_FRAMES]:
        img = cv2.imread(p)  # 失敗時は例外でなく None が返る点に注意
        if img is None:
            continue
        if target is None:
            target = (img.shape[1], img.shape[0])
        else:
            img = cv2.resize(img, target)
        frames.append(img)
    return frames


def make_surveillance_clip(n_frames=130, h=240, w=320, seed=0):
    """監視カメラ風の合成クリップを作る（静止背景＋ときどき横切る物体）。

    わざと「無音(静止)区間」を挟むのがポイント。録画されるべき区間とされない区間が
    はっきり分かれるので、モーションレコーダが正しく区切れているか一目で確認できる。
    背景・物体は cv_helpers から借り、毎フレーム微小ノイズを足して実カメラっぽくする。
    """
    rng = np.random.default_rng(seed)
    bg = make_background(h, w, seed)
    sprite = make_sprite()
    sh, sw = sprite.shape[:2]
    # 物体が画面を横切る区間（[start, end)）。それ以外は静止＝録画されないはず。
    active_spans = [(22, 50), (78, 108)]
    frames = []
    for i in range(n_frames):
        f = bg.copy()
        for k, (s, e) in enumerate(active_spans):
            if s <= i < e:
                t = (i - s) / max(1, e - s - 1)  # 0→1
                # 偶数番イベントは左→右、奇数番は右→左に動かす（向きを変えて変化をつける）。
                tt = t if k % 2 == 0 else 1.0 - t
                sx = int(20 + (w - 40 - sw) * tt)
                sy = int(h * 0.42 + 16 * np.sin(i / 5.0))
                f[sy:sy + sh, sx:sx + sw] = sprite
        # 実カメラ風の弱い撮像ノイズ（静止区間でも完全な無変化にはしない）。
        f = np.clip(f.astype(np.int16) + rng.normal(0, 1.5, f.shape), 0, 255).astype(np.uint8)
        frames.append(f)
    return frames


def load_frames():
    """(frames, source, fps) を返す。frames は BGR uint8 のリスト。"""
    found = find_input()
    if found is not None:
        kind, ref = found
        if kind == "video":
            frames, fps = read_video(ref)
            if frames:
                return frames, f"video:{os.path.basename(ref)}", fps
        else:
            frames = read_images(ref)
            if frames:
                return frames, f"images:{len(frames)}枚", DEFAULT_FPS
        print(f"[警告] {DATA_DIR} の入力を読めませんでした。合成にフォールバックします。")
    return make_surveillance_clip(), "synthetic(make_surveillance_clip)", DEFAULT_FPS


# ----------------------------------------------------------------------------
# 動き量の計測（背景差分で各フレームの「動きスコア」と動体ボックスを出す）
# ----------------------------------------------------------------------------
def measure_motion(frames):
    """全フレームの (動きスコア, 最大動体ボックス) を計算して返す。

    動きスコア = 前景画素の割合(0〜1)。背景差分器(MOG2)を順に apply して背景を学習し、
    影(127)を閾値で落とし、形態学でごま塩ノイズを掃除してから画素率を測る。
    ボックスは最大面積の前景ブロブの外接矩形（描画用。無ければ None）。
    """
    h, w = frames[0].shape[:2]
    area = float(h * w)
    mog2 = cv2.createBackgroundSubtractorMOG2(history=120, varThreshold=30,
                                              detectShadows=True)
    scores, boxes = [], []
    for f in frames:
        raw = mog2.apply(f)                                   # 背景更新＋前景マスク
        _, fg = cv2.threshold(raw, 200, 255, cv2.THRESH_BINARY)  # 影(127)を 0 に
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, KERNEL)     # ごま塩ノイズ除去
        scores.append(float(cv2.countNonZero(fg)) / area)     # 前景画素率＝動きスコア

        contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        big = [c for c in contours if cv2.contourArea(c) >= MIN_AREA]
        boxes.append(cv2.boundingRect(max(big, key=cv2.contourArea)) if big else None)
    return scores, boxes


# ----------------------------------------------------------------------------
# イベント抽出（動いた区間を録画イベントに区切る＝レコーダの中核ロジック）
# ----------------------------------------------------------------------------
def detect_events(scores, n):
    """動きスコア列から録画イベント [(start, end, peak_idx), ...] を作る。

    手順は実機のモーションレコーダと同じ:
      1. しきい値超え（かつ慣らし運転後）のフレームを「動きあり」とする。
      2. 近接する動きを 1 イベントにまとめる（POST_ROLL 以内の隙間は同一イベント扱い）。
      3. 短すぎるイベント（チラつき）は捨てる。
      4. 頭切れ防止に PRE_ROLL、余韻に POST_ROLL を付けて区間を前後に広げる。
    返す (start, end) は録画する閉区間 [start, end]（両端含む）。
    """
    active = [i for i in range(n) if i >= WARMUP and scores[i] >= MOTION_THRESH]
    if not active:
        return []

    # 隣接する動きフレームを束ねる（隙間が POST_ROLL 以内なら同じイベント）。
    raw_events = []
    s = prev = active[0]
    for i in active[1:]:
        if i - prev <= POST_ROLL:
            prev = i
        else:
            raw_events.append((s, prev))
            s = prev = i
    raw_events.append((s, prev))

    # 短いチラつきを捨て、前後にのりしろ（プリ/ポストロール）を付けて確定。
    events = []
    for s, e in raw_events:
        if (e - s + 1) < MIN_EVENT_LEN:
            continue
        start = max(0, s - PRE_ROLL)
        end = min(n - 1, e + POST_ROLL)
        peak = max(range(s, e + 1), key=lambda i: scores[i])  # 最も動いたフレーム
        events.append((start, end, peak))
    return events


# ----------------------------------------------------------------------------
# 可視化（フレームに録画オーバレイを焼き込む）
# ----------------------------------------------------------------------------
def annotate(frame, idx, score, box, recording, event_id):
    """フレームに動体枠・動きスコア・録画インジケータ(REC)を描いたコピーを返す。"""
    vis = frame.copy()
    h, w = vis.shape[:2]
    if box is not None:  # 最大動体に黄枠
        x, y, bw, bh = box
        cv2.rectangle(vis, (x, y), (x + bw, y + bh), (0, 255, 255), 2)
    # 左上にフレーム番号と動きスコア（ASCII のみ＝headless 安全）。
    cv2.putText(vis, f"frame {idx:03d}  motion {score*100:5.2f}%", (8, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    if recording:  # 録画中は赤丸 REC と赤枠で強調
        cv2.circle(vis, (w - 70, 16), 6, (0, 0, 255), -1)
        cv2.putText(vis, f"REC E{event_id}", (w - 56, 21),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.rectangle(vis, (1, 1), (w - 2, h - 2), (0, 0, 255), 2)
    return vis


def write_clip(path, annotated, span, fps, size):
    """イベント区間 [s, e] の注釈付きフレームを mp4v クリップに書き出す。

    VideoWriter が開けない環境（コーデック無し等）では連番モンタージュ PNG に退避する。
    返り値は実際に保存したファイルパス。
    """
    s, e, _ = span
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, size)
    if writer.isOpened():
        for i in range(s, e + 1):
            writer.write(annotated[i])
        writer.release()
        return path
    # フォールバック: 代表フレームを横に並べた PNG（動画が作れなくても結果を残す）。
    png_path = path.rsplit(".", 1)[0] + "_frames.png"
    idxs = sorted(set(np.linspace(s, e, min(6, e - s + 1)).astype(int).tolist()))
    cv2.imwrite(png_path, np.hstack([annotated[i] for i in idxs]))
    return png_path


def save_montage(path, annotated, events):
    """各イベントの代表3枚（最初/ピーク/最後）を縦に積んだ一覧画像を保存する。"""
    rows = []
    for s, e, peak in events[:6]:  # 多すぎる場合は先頭 6 イベントまで
        rows.append(np.hstack([annotated[s], annotated[peak], annotated[e]]))
    if not rows:  # イベントゼロでも空にせず、最初の数枚を並べて状況を残す
        idxs = sorted(set(np.linspace(0, len(annotated) - 1, 3).astype(int).tolist()))
        rows.append(np.hstack([annotated[i] for i in idxs]))
    cv2.imwrite(path, np.vstack(rows))


def save_timeline(path, scores, events, n):
    """動きスコアの折れ線＋しきい値線＋録画区間の帯を描く（matplotlib/Agg/ASCII）。"""
    fig, ax = plt.subplots(figsize=(8, 3.2))
    xs = list(range(n))
    ax.plot(xs, [s * 100 for s in scores], color="tab:blue", label="motion %")
    ax.axhline(MOTION_THRESH * 100, color="tab:red", ls="--", lw=1, label="threshold")
    for k, (s, e, _) in enumerate(events):  # 録画区間を薄い緑で塗る
        ax.axvspan(s, e, color="tab:green", alpha=0.2,
                   label="recorded" if k == 0 else None)
    ax.set_title("Motion-Activated Recording - per-frame motion & recorded spans")
    ax.set_xlabel("frame")
    ax.set_ylabel("motion (% of pixels)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ----------------------------------------------------------------------------
# メイン
# ----------------------------------------------------------------------------
def main():
    out = output_dir()
    frames, source, fps = load_frames()
    n = len(frames)
    h, w = frames[0].shape[:2]

    print("=" * 64)
    print("動き検知録画ツール（動いた区間だけを切り出して保存）")
    print(f"  入力: {source}  / フレーム {n} / サイズ {w}x{h} / {fps:.1f} FPS")
    print(f"  しきい値: 前景画素率 {MOTION_THRESH*100:.2f}% 以上で『動きあり』"
          f"  プリ/ポストロール: {PRE_ROLL}/{POST_ROLL} フレーム")

    # ① 各フレームの動き量を測る → ② 録画イベントに区切る
    scores, boxes = measure_motion(frames)
    events = detect_events(scores, n)

    # ③ 録画フレーム集合とイベントID対応を作る（描画用）
    recorded = set()
    frame_event = {}
    for k, (s, e, _) in enumerate(events, start=1):
        for i in range(s, e + 1):
            recorded.add(i)
            frame_event[i] = k

    # ④ 全フレームに録画オーバレイを焼き込む（クリップ・モンタージュで共用）
    annotated = [
        annotate(frames[i], i, scores[i], boxes[i], i in recorded, frame_event.get(i, 0))
        for i in range(n)
    ]

    # ⑤ イベントごとにクリップを書き出す
    event_records = []
    for k, (s, e, peak) in enumerate(events, start=1):
        clip_path = os.path.join(out, f"use_case_event{k}.mp4")
        saved = write_clip(clip_path, annotated, (s, e, peak), fps, (w, h))
        num = e - s + 1
        event_records.append({
            "event_id": k,
            "start_frame": s,
            "end_frame": e,
            "num_frames": num,
            "duration_sec": round(num / fps, 3),
            "peak_frame": peak,
            "peak_motion_pct": round(scores[peak] * 100, 3),
            "clip_file": os.path.basename(saved),
        })

    # ⑥ 一覧モンタージュ・タイムライン図
    montage_path = os.path.join(out, "use_case_event_montage.png")
    timeline_path = os.path.join(out, "use_case_motion_timeline.png")
    save_montage(montage_path, annotated, events)
    save_timeline(timeline_path, scores, events, n)

    # ⑦ 機械可読な結果 JSON（録画区間と『節約率』を記録）
    recorded_frames = len(recorded)
    saved_ratio = 1.0 - recorded_frames / n if n else 0.0
    result = {
        "source": source,
        "num_frames": n,
        "frame_size": [w, h],
        "fps": round(fps, 3),
        "params": {
            "motion_thresh_pct": round(MOTION_THRESH * 100, 3),
            "warmup": WARMUP,
            "pre_roll": PRE_ROLL,
            "post_roll": POST_ROLL,
            "min_event_len": MIN_EVENT_LEN,
        },
        "num_events": len(events),
        "recorded_frames": recorded_frames,
        "storage_saved_ratio": round(saved_ratio, 4),
        "events": event_records,
    }
    json_path = os.path.join(out, "use_case_events.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # ⑧ サマリ表示
    print(f"\n  検出イベント数: {len(events)}")
    for r in event_records:
        print(f"    - E{r['event_id']}: frame {r['start_frame']}-{r['end_frame']} "
              f"({r['num_frames']}枚 / {r['duration_sec']}s, "
              f"ピーク動き {r['peak_motion_pct']}%) -> {r['clip_file']}")
    print(f"  録画フレーム: {recorded_frames}/{n}  "
          f"=> 録りっぱなしに対し約 {saved_ratio*100:.1f}% を録画せず節約")
    print("\n  保存物:")
    print("    - イベントクリップ : use_case_event*.mp4（録画区間ごと）")
    print(f"    - 代表フレーム一覧 : {os.path.basename(montage_path)}")
    print(f"    - 動き量タイムライン: {os.path.basename(timeline_path)}")
    print(f"    - イベント表 JSON  : {os.path.basename(json_path)}")
    if source.startswith("synthetic"):
        print("  ※ 合成入力で完走。data/10_classical_video_motion/ に固定カメラ動画を置けば実録画になります。")
    print("\n発展: しきい値を最大ブロブ面積に、背景差分を検出器に替えると誤検知が減り実用度が上がります。")


if __name__ == "__main__":
    main()
