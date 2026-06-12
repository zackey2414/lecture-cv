"""use_case.py — ライン通過カウンター（指定ラインを横切った回数を数える小ツール）

この回の「実践ユースケース」: 動画中の物体を追跡し、画面に引いた **仮想ライン** を
横切った回数を **方向別** に数える、現実の監視・計測でよく使う小ツールです。
店舗の入店者カウント、横断歩道の通行量、生産ラインの製品計数、道路の交通量調査 ——
どれも「ライン交差 × 物体追跡」の組み合わせで作れます。

────────────────────────────────────────────────────────────────────────
mini_project.py との違い（役割分担）
────────────────────────────────────────────────────────────────────────
- `mini_project.py` は「検出 → 追跡 → **MOT 評価(MOTA/IDF1/HOTA)**」を一通り通す
  統合課題で、指標を厳密に検算するために合成 GT を重視した **ベンチ寄り** の構成です。
- 本 `use_case.py` は評価ではなく **アプリの出力（カウント値）** を作ることが目的の
  **現実の小ツール** です。追跡器（自前 SORT）を借りつつ、その上に
  「ラインを横切ったら +1」というアプリ固有ロジックだけを薄く足しています。
  そのまま実運用の計数器に拡張できます。

────────────────────────────────────────────────────────────────────────
データ配置（実データ優先・合成フォールバック）
────────────────────────────────────────────────────────────────────────
- `data/28_tracking/` に **動画ファイル**（*.mp4 / *.avi / *.mov / *.mkv）か
  **連番画像**（*.png / *.jpg）を置くと、それを実入力として使います。
  例: `data/28_tracking/street.mp4` を置いて再生すれば実用的な通行量カウンタになります。
- 置かなければ **合成シーン**（壁で反射する複数の長方形）で必ず完走します（exit 0）。
- 実動画では背景差分(MOG2)で前景ブロブを検出します。簡易検出なので、固定カメラ・
  動く前景のシーンで効きます（手ぶれ・激しい照明変化では反応が弱くなります）。
  本格運用したい場合は detect 部を YOLO / torchvision 検出器に差し替えてください。

────────────────────────────────────────────────────────────────────────
拡張アイデア（練習用）
────────────────────────────────────────────────────────────────────────
- 複数ラインや多角形ゾーンを引いて「ゾーン内滞在数」を測る（ヒートマップ化）。
- ライン方向ごとに別カウンタを持ち、入店/退店の差分から「現在の在室人数」を出す。
- detect を実検出器（fasterrcnn_mobilenet_v3_large_320_fpn 等の小型）に差し替え、
  クラス別（人だけ/車だけ）にカウントする。
- SORT を ByteTrack 流の 2 段対応へ拡張し、低スコア検出も拾って取りこぼしを減らす。
- カウント時刻を記録して時間帯別ヒストグラム（ピーク時間の可視化）を作る。

実行:
    uv run python lectures/28_tracking/use_case.py

出力（outputs/28_tracking/ に保存）:
    use_case_line_counter.png  : ライン・ID 枠・軌跡・累計カウントを描いたモンタージュ
    use_case_count_timeline.png: 累計通過数の時系列グラフ（matplotlib, Agg）
    use_case_counts.json       : 方向別/ID 別の通過数と交差ログ（機械可読）
"""

from __future__ import annotations

import glob
import importlib.util
import json
import os
import sys

# headless 前提: matplotlib は必ず Agg バックエンド（GUI を開かない）。
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import ensure_outdir, id_color, make_mot_scene  # noqa: E402

# ----------------------------------------------------------------------------
# 設定（ここを変えるだけで挙動を調整できる）
# ----------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
DATA_DIR = os.path.join(REPO, "data", "28_tracking")

VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv")
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp")

MAX_FRAMES = 240        # 実動画は CPU 負荷を抑えるためこのフレーム数で打ち切る
MAX_WIDTH = 480         # 実動画はこの幅に縮小（処理を軽く）

# 仮想ラインの向きと位置（"horizontal" = 横線で上下通過を数える）。
# "vertical" にすれば縦線で左右通過を数える。位置は画面比率(0〜1)で指定。
LINE_ORIENTATION = "horizontal"
LINE_POS_FRAC = 0.5

TRAIL_LEN = 16          # 軌跡として残す中心点の数


# ----------------------------------------------------------------------------
# 数字始まりのスクリプト（import 不可）を importlib で読み込む
# ----------------------------------------------------------------------------
def _load_module(filename: str, mod_name: str):
    path = os.path.join(HERE, filename)
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# 02 の自前 SORT を再利用（追跡ロジックを重複させない）
_sort_mod = _load_module("02_sort_from_scratch.py", "uc_sort_mod")
Sort = _sort_mod.Sort
KalmanBoxTracker = _sort_mod.KalmanBoxTracker


# ----------------------------------------------------------------------------
# 入力フレームの用意（実データ優先 → 合成フォールバック）
# ----------------------------------------------------------------------------
def _find_input():
    """data/28_tracking/ から動画 or 連番画像を探す。無ければ None を返す。"""
    if not os.path.isdir(DATA_DIR):
        return None
    # 1) 動画を優先
    for ext in VIDEO_EXTS:
        hits = sorted(glob.glob(os.path.join(DATA_DIR, f"*{ext}")))
        if hits:
            return ("video", hits[0])
    # 2) 連番画像
    imgs = []
    for ext in IMAGE_EXTS:
        imgs += glob.glob(os.path.join(DATA_DIR, f"*{ext}"))
    if len(imgs) >= 2:
        return ("images", sorted(imgs))
    return None


def _read_video(path: str):
    """動画を読み、必要なら縮小して BGR フレームのリストを返す。"""
    cap = cv2.VideoCapture(path)
    frames = []
    while len(frames) < MAX_FRAMES:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        if w > MAX_WIDTH:  # CPU 負荷を抑えるため縮小
            scale = MAX_WIDTH / w
            frame = cv2.resize(frame, (MAX_WIDTH, int(round(h * scale))))
        frames.append(frame)
    cap.release()
    return frames


def _read_images(paths):
    """連番画像をフレームとして読む（サイズは先頭に合わせる）。"""
    frames = []
    target = None
    for p in paths[:MAX_FRAMES]:
        img = cv2.imread(p)
        if img is None:
            continue
        if target is None:
            target = (img.shape[1], img.shape[0])
        else:
            img = cv2.resize(img, target)
        frames.append(img)
    return frames


def _render_synthetic_frame(scene, t):
    """合成 GT を色付き塗りつぶしで描いた BGR フレーム（検出器の入力画像）。"""
    img = np.full((scene.height, scene.width, 3), 25, np.uint8)
    for obj_id, box in scene.gt[t].items():
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(img, (x1, y1), (x2, y2), scene.colors[obj_id], -1)
    return img


def load_frames():
    """(frames, source, is_synthetic) を返す。frames は BGR uint8 のリスト。"""
    found = _find_input()
    if found is not None:
        kind, ref = found
        if kind == "video":
            frames = _read_video(ref)
            if frames:
                return frames, f"video:{os.path.basename(ref)}", False
        else:
            frames = _read_images(ref)
            if frames:
                return frames, f"images:{len(frames)}枚", False
        print(f"[警告] {DATA_DIR} の入力を読めませんでした。合成にフォールバックします。")

    # --- 合成フォールバック: 壁で反射する 4 物体（中央線を何度も横切る）---
    scene = make_mot_scene(num_objects=4, num_frames=72, width=360, height=260,
                           seed=5, p_miss=0.05, p_fp=0.05)
    frames = [_render_synthetic_frame(scene, t) for t in range(scene.num_frames)]
    return frames, "synthetic(make_mot_scene)", True


# ----------------------------------------------------------------------------
# 検出（毎フレーム独立にボックスを得る・ID は付かない）
# ----------------------------------------------------------------------------
def detect_synthetic(frame, min_area=120):
    """合成フレーム用: 明るい前景ブロブを輝度しきい値＋連結成分で検出。"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
    return _blobs_from_mask(mask, min_area)


def detect_real(frame, bgsub, min_area=300):
    """実動画用: 背景差分(MOG2)で前景マスクを作り、連結成分でボックス化。"""
    fg = bgsub.apply(frame)
    fg = cv2.medianBlur(fg, 5)
    # 影(127)を除き、確かな前景(255)だけ残す
    _, mask = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    return _blobs_from_mask(mask, min_area)


def _blobs_from_mask(mask, min_area):
    """二値マスクの連結成分から (K,5)=[x1,y1,x2,y2,score] を作る。"""
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    dets = []
    for i in range(1, n):  # 0 は背景
        x, y, w, h, area = stats[i]
        if area < min_area:
            continue
        dets.append([x, y, x + w, y + h, 0.9])
    return np.array(dets, dtype=float).reshape(-1, 5)


# ----------------------------------------------------------------------------
# ライン交差の判定
# ----------------------------------------------------------------------------
def make_line(width, height):
    """画面サイズから仮想ラインの端点 (p1, p2) と方向ラベルを決める。"""
    if LINE_ORIENTATION == "vertical":
        x = int(width * LINE_POS_FRAC)
        return (x, 0), (x, height), ("right", "left")  # 正側=右, 負側=左
    y = int(height * LINE_POS_FRAC)
    return (0, y), (width, y), ("down", "up")           # 正側=下, 負側=上


def signed_side(pt, p1, p2):
    """点 pt がライン(p1→p2)のどちら側かを符号で返す（外積の z 成分）。

    > 0 / < 0 で左右（上下）を表し、フレーム間で符号が反転したら「横切った」。
    """
    return (p2[0] - p1[0]) * (pt[1] - p1[1]) - (p2[1] - p1[1]) * (pt[0] - p1[0])


# ----------------------------------------------------------------------------
# メイン: 追跡しながらライン交差を数える
# ----------------------------------------------------------------------------
def main():
    out_dir = ensure_outdir()
    KalmanBoxTracker._count = 0  # 再現性のため ID カウンタを初期化

    frames, source, is_synth = load_frames()
    height, width = frames[0].shape[:2]
    p1, p2, (label_pos, label_neg) = make_line(width, height)

    tracker = Sort(iou_threshold=0.3, max_age=8, min_hits=2)
    bgsub = None if is_synth else cv2.createBackgroundSubtractorMOG2(
        history=120, varThreshold=40, detectShadows=True
    )

    # カウント状態
    counts = {label_pos: 0, label_neg: 0}     # 方向別の通過数
    per_id: dict[int, dict] = {}              # ID 別の通過数
    prev_side: dict[int, float] = {}          # ID ごとの前フレームのライン側
    crossings = []                            # 交差ログ（フレーム/ID/方向）
    trails: dict[int, list] = {}              # 軌跡（描画用）
    cum_timeline = {label_pos: [], label_neg: []}  # 累計の時系列（グラフ用）

    # 可視化用に等間隔で 6 枚抜き出す
    n_frames = len(frames)
    tile_idxs = sorted(set(np.linspace(0, n_frames - 1, 6).astype(int).tolist()))
    tiles = []

    print("=" * 64)
    print("ライン通過カウンター（物体追跡 × ライン交差）")
    print(f"  入力: {source}  / フレーム {n_frames} / サイズ {width}x{height}")
    print(f"  ライン: {LINE_ORIENTATION}  位置 {LINE_POS_FRAC:.2f}  "
          f"（正側={label_pos} / 負側={label_neg}）")

    for t, frame in enumerate(frames):
        # ① 検出 → ② 追跡（ID 付与）
        dets = detect_synthetic(frame) if is_synth else detect_real(frame, bgsub)
        tracks = tracker.update(dets)

        vis = frame.copy()
        cv2.line(vis, p1, p2, (0, 0, 255), 2)  # 仮想ライン（赤）

        for x1, y1, x2, y2, tid in tracks:
            tid = int(tid)
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            side = signed_side((cx, cy), p1, p2)
            ps = prev_side.get(tid)

            # ③ 符号が反転 = ラインを横切った
            if ps is not None and ps != 0 and side != 0 and (ps > 0) != (side > 0):
                direction = label_pos if side > 0 else label_neg
                counts[direction] += 1
                per_id.setdefault(tid, {label_pos: 0, label_neg: 0})[direction] += 1
                crossings.append({"frame": t, "id": tid, "direction": direction})
            prev_side[tid] = side

            # 描画（枠・ID・軌跡）
            color = id_color(tid)
            cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(vis, f"ID{tid}", (int(x1), max(10, int(y1) - 3)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
            pts = trails.setdefault(tid, [])
            pts.append((int(cx), int(cy)))
            if len(pts) > TRAIL_LEN:
                del pts[0]
            for i in range(1, len(pts)):
                cv2.line(vis, pts[i - 1], pts[i], color, 1)

        # 累計カウントを画面に焼き込む（ASCII で headless 安全に）
        banner = f"{label_pos}:{counts[label_pos]}  {label_neg}:{counts[label_neg]}"
        cv2.putText(vis, banner, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 2, cv2.LINE_AA)

        cum_timeline[label_pos].append(counts[label_pos])
        cum_timeline[label_neg].append(counts[label_neg])
        if t in tile_idxs:
            tiles.append(vis.copy())

    total = counts[label_pos] + counts[label_neg]

    # --- 可視化 1: モンタージュ PNG（6 枚を 2x3 に並べる）---
    while len(tiles) < 6:  # 念のため枚数を揃える
        tiles.append(tiles[-1].copy())
    montage = np.vstack([np.hstack(tiles[:3]), np.hstack(tiles[3:6])])
    montage_path = os.path.join(out_dir, "use_case_line_counter.png")
    cv2.imwrite(montage_path, montage)

    # --- 可視化 2: 累計通過数の時系列グラフ（matplotlib / Agg / ASCII タイトル）---
    fig, ax = plt.subplots(figsize=(7, 3.2))
    xs = list(range(n_frames))
    ax.plot(xs, cum_timeline[label_pos], label=label_pos, color="tab:blue")
    ax.plot(xs, cum_timeline[label_neg], label=label_neg, color="tab:orange")
    ax.set_title("Line Crossing Counter - cumulative crossings")  # ASCII
    ax.set_xlabel("frame")
    ax.set_ylabel("cumulative count")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    timeline_path = os.path.join(out_dir, "use_case_count_timeline.png")
    fig.savefig(timeline_path, dpi=110)
    plt.close(fig)

    # --- 機械可読な結果 JSON ---
    result = {
        "source": source,
        "num_frames": n_frames,
        "frame_size": [width, height],
        "line": {
            "orientation": LINE_ORIENTATION,
            "pos_frac": LINE_POS_FRAC,
            "p1": list(p1),
            "p2": list(p2),
            "positive_label": label_pos,
            "negative_label": label_neg,
        },
        "counts": {**counts, "total": total},
        "per_id": {str(k): v for k, v in sorted(per_id.items())},
        "num_crossings": len(crossings),
        "crossings": crossings,
    }
    json_path = os.path.join(out_dir, "use_case_counts.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # --- サマリ表示 ---
    print(f"  通過合計: {total}  （{label_pos}={counts[label_pos]}, "
          f"{label_neg}={counts[label_neg]}）")
    print(f"  交差したユニーク ID 数: {len(per_id)}")
    print(f"  モンタージュ : {montage_path}")
    print(f"  時系列グラフ : {timeline_path}")
    print(f"  カウント JSON: {json_path}")
    if is_synth:
        print("  ※ 合成入力で完走。data/28_tracking/ に動画を置けば実通行量カウンタになります。")
    print("\n発展: detect 部を実検出器(YOLO/torchvision)に、ラインを複数/ゾーンに拡張すると実用度が上がります。")


if __name__ == "__main__":
    main()
