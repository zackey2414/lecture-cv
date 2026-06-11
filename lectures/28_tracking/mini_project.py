"""mini_project.py — 章末ミニプロジェクト: 検出 → 追跡 → 評価 の統合パイプライン

これまでの部品を 1 本につなぐ「完成形」:

    合成カラー動画(複数の動く物体)
      → ① 検出器: cv2 の連結成分でボックスを得る（毎フレーム独立・ID 無し）
      → ② 追跡器: 自前 SORT(02) でフレーム間に同一 ID を付与
      → ③ 可視化: ID 別の色で軌跡と枠を描き、動画/モンタージを保存
      → ④ 評価: 自前 MOT 指標(04) で MOTA/MOTP/IDF1/HOTA を算出

外部データもネットも不要（モデル重みすら不要）。実運用では ① をそのまま
YOLO/torchvision 検出器に、SORT を ByteTrack/DeepSORT に差し替えればよい。

実行:
    uv run python lectures/28_tracking/mini_project.py
"""

from __future__ import annotations

import importlib.util
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import ensure_outdir, id_color, make_mot_scene  # noqa: E402


def _load_module(filename: str, mod_name: str):
    """同ディレクトリの数字始まりファイル（import 不可）を importlib で読み込む。"""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# 02 の SORT と 04 の評価器を再利用（コードを重複させない）
_sort_mod = _load_module("02_sort_from_scratch.py", "sort_mod")
_metrics_mod = _load_module("04_mot_metrics.py", "metrics_mod")
Sort = _sort_mod.Sort
KalmanBoxTracker = _sort_mod.KalmanBoxTracker
evaluate_mot = _metrics_mod.evaluate_mot


def render_colored_frame(scene, t):
    """GT を色つき塗りつぶしで描いた BGR フレーム（検出器の入力画像）。"""
    img = np.full((scene.height, scene.width, 3), 25, np.uint8)
    for obj_id, box in scene.gt[t].items():
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(img, (x1, y1), (x2, y2), scene.colors[obj_id], -1)
    return img


def detect_blobs(frame, min_area=120):
    """① 検出器: 背景(暗) との差で前景マスクを作り、連結成分のボックスを返す。

    これは「毎フレーム独立に物体を探す」検出器の最小実装。ID は付かない。
    物体が重なると 1 つの blob に融合することがある（=検出器の限界）→ 追跡が効いてくる。
    返り値: (K,5) = [x1,y1,x2,y2,score]
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    dets = []
    for i in range(1, n):  # 0 は背景
        x, y, w, h, area = stats[i]
        if area < min_area:
            continue
        dets.append([x, y, x + w, y + h, 0.9])
    return np.array(dets, dtype=float).reshape(-1, 5)


def main():
    out_dir = ensure_outdir()
    KalmanBoxTracker._count = 0  # ID カウンタを初期化（再現性）

    # --- 合成シーン（GT は評価のため保持）---
    scene = make_mot_scene(num_objects=4, num_frames=48, seed=11, p_miss=0.0, p_fp=0.0)
    frames = [render_colored_frame(scene, t) for t in range(scene.num_frames)]

    sort = Sort(iou_threshold=0.3, max_age=5, min_hits=2)

    print("=" * 64)
    print("ミニプロジェクト: 検出 → SORT 追跡 → MOT 評価")

    # 動画ライター（headless でもファイル書き出しは可能。失敗時は PNG 連番へ）
    video_path = os.path.join(out_dir, "mini_project_tracking.mp4")
    writer = cv2.VideoWriter(
        video_path, cv2.VideoWriter_fourcc(*"mp4v"), 12, (scene.width, scene.height)
    )
    use_video = writer.isOpened()

    pred_list = []      # 評価用: 各フレーム {pred_id: box}
    trails: dict[int, list] = {}
    n_det_total = 0
    montage_idxs = set(np.linspace(0, scene.num_frames - 1, 6).astype(int).tolist())
    tiles = []

    for t in range(scene.num_frames):
        dets = detect_blobs(frames[t])     # ① 検出
        n_det_total += len(dets)
        tracks = sort.update(dets)         # ② 追跡（ID 付与）

        pred_frame = {}
        vis = frames[t].copy()
        for x1, y1, x2, y2, tid in tracks:
            tid = int(tid)
            pred_frame[tid] = np.array([x1, y1, x2, y2], dtype=float)
            color = id_color(tid)
            cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(vis, f"ID{tid}", (int(x1), max(10, int(y1) - 3)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            trails.setdefault(tid, []).append((cx, cy))
            for i in range(1, len(trails[tid])):  # ③ 軌跡
                cv2.line(vis, trails[tid][i - 1], trails[tid][i], color, 1)
        pred_list.append(pred_frame)

        if use_video:
            writer.write(vis)
        if t in montage_idxs:
            tiles.append(vis.copy())

    if use_video:
        writer.release()

    # モンタージ PNG（動画が見られない環境向けの保険）
    montage = np.vstack([np.hstack(tiles[:3]), np.hstack(tiles[3:6])])
    montage_path = os.path.join(out_dir, "mini_project_montage.png")
    cv2.imwrite(montage_path, montage)

    # --- ④ 評価 ---
    res = evaluate_mot(scene.gt, pred_list, iou_thr=0.5)

    print(f"  フレーム数={scene.num_frames}  真の物体数={len(scene.colors)}  "
          f"検出総数={n_det_total}")
    print(f"  MOTA={res['MOTA']:.3f}  MOTP={res['MOTP']:.3f}  "
          f"IDF1={res['IDF1']:.3f}  HOTA={res['HOTA']:.3f}")
    print(f"  内訳: FN={res['FN']} FP={res['FP']} IDSW={res['IDSW']} GT={res['GT']}")
    if use_video:
        print(f"  追跡動画: {video_path}")
    else:
        print("  （この環境では mp4 書き出し不可。モンタージのみ保存）")
    print(f"  モンタージ: {montage_path}")
    print("\n完成: 検出器→トラッカ→評価の一連を、外部依存ゼロで回せた。")
    print("発展: detect_blobs を YOLO に、Sort を ByteTrack/DeepSORT に差し替えれば実運用へ。")


if __name__ == "__main__":
    main()
