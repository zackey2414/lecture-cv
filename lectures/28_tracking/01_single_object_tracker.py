"""01_single_object_tracker.py — 検出器なしの単一物体トラッカ（OpenCV）

学べること:
- 「検出 (detection)」と「追跡 (tracking)」の違い。検出は各フレーム独立に物体を探す。
  単一物体トラッカは **最初の 1 フレームで ROI を 1 回与えるだけ** で、以降は
  前フレームの見え方を手かかりに同じ物体を追い続ける（検出器が不要）。
- OpenCV のトラッカ API（init / update）の正準的な使い方。
- ビルドによって使えるトラッカが違う問題（CSRT は main、KCF/MOSSE は contrib の
  cv2.legacy 名前空間）。本講座は headless 版なので `hasattr` で **実在するものだけ** 使う。

実行:
    uv run python lectures/28_tracking/01_single_object_tracker.py

注意（落とし穴）:
- opencv-python-headless / 非 contrib ビルドでは TrackerCSRT_create や cv2.legacy が
  存在しないことがある。記事を鎅呑みにせず `hasattr` で確認してから呼ぶ。
- GOTURN / Nano / DaSiamRPN は別途モデル重み(.onnx 等)が必要なので、重み無しで使える
  MIL / CSRT / KCF を優先する。
"""

from __future__ import annotations

import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import ensure_outdir, iou_xyxy  # noqa: E402


def list_available_trackers():
    """この OpenCV ビルドで「重み無しに使える」単一物体トラッカを集める。

    返り値: [(名前, ファクトリ関数), ...]
    優先順位は CSRT(高精度) → KCF(高速) → MIL(頑健・依存少)。
    実在チェックは hasattr で行い、無いものはスキップする。
    """
    candidates = []
    # CSRT: 高精度。新しめのビルドは main（cv2.TrackerCSRT_create）にある
    if hasattr(cv2, "TrackerCSRT_create"):
        candidates.append(("CSRT", cv2.TrackerCSRT_create))
    elif hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerCSRT_create"):
        candidates.append(("CSRT(legacy)", cv2.legacy.TrackerCSRT_create))
    # KCF: 高速。contrib では cv2.legacy 側のことが多い
    if hasattr(cv2, "TrackerKCF_create"):
        candidates.append(("KCF", cv2.TrackerKCF_create))
    elif hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerKCF_create"):
        candidates.append(("KCF(legacy)", cv2.legacy.TrackerKCF_create))
    # MIL: ほぼ全ビルドにあり重み不要。最後の砦
    if hasattr(cv2, "TrackerMIL_create"):
        candidates.append(("MIL", cv2.TrackerMIL_create))
    return candidates


def make_single_object_clip(num_frames=60, width=320, height=240, seed=0):
    """1 個の長方形がジグザグに動く合成クリップと GT を返す。

    返り値:
        frames: list[np.ndarray(H,W,3) BGR uint8]
        gts   : list[np.ndarray([x1,y1,x2,y2])]  各フレームの正解ボックス
    """
    rng = np.random.default_rng(seed)
    cx, cy = 60.0, 120.0
    vx, vy = 4.0, 2.4
    w, h = 40.0, 36.0
    frames, gts = [], []
    for _ in range(num_frames):
        cx += vx
        cy += vy
        # 壁で反射
        if cx < w / 2 or cx > width - w / 2:
            vx *= -1
            cx = float(np.clip(cx, w / 2, width - w / 2))
        if cy < h / 2 or cy > height - h / 2:
            vy *= -1
            cy = float(np.clip(cy, h / 2, height - h / 2))
        # 背景に弱いノイズ + テクスチャ（トラッカに手かかりを与える）
        img = (rng.normal(40, 6, (height, width, 3))).clip(0, 255).astype(np.uint8)
        x1, y1 = int(cx - w / 2), int(cy - h / 2)
        x2, y2 = int(cx + w / 2), int(cy + h / 2)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 170, 255), -1)
        cv2.circle(img, (int(cx), int(cy)), 6, (255, 255, 255), -1)  # 内部模様
        frames.append(img)
        gts.append(np.array([x1, y1, x2, y2], dtype=float))
    return frames, gts


def run_tracker(name, factory, frames, gts):
    """1 つのトラッカを最初のフレームで初期化し、全フレームを update して IoU を測る。"""
    tracker = factory()
    x1, y1, x2, y2 = gts[0]
    # OpenCV トラッカの init は (x, y, w, h) のタプルで ROI を渡す
    init_box = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))
    tracker.init(frames[0], init_box)

    ious = []
    pred_boxes = [gts[0].copy()]  # フレーム0は初期化に使ったので GT 扱い
    for t in range(1, len(frames)):
        ok, box = tracker.update(frames[t])
        if not ok:
            # ロスト時は前回の予測を流用（実務では再検出にフォールバックする）
            pred = pred_boxes[-1].copy()
        else:
            bx, by, bw, bh = box
            pred = np.array([bx, by, bx + bw, by + bh], dtype=float)
        pred_boxes.append(pred)
        ious.append(iou_xyxy(pred, gts[t]))

    ious = np.array(ious)
    mean_iou = float(ious.mean()) if len(ious) else 0.0
    success = float((ious > 0.5).mean()) if len(ious) else 0.0  # IoU>0.5 を成功とみなす割合
    return mean_iou, success, pred_boxes


def main():
    out_dir = ensure_outdir()
    frames, gts = make_single_object_clip()
    trackers = list_available_trackers()

    print("=" * 64)
    print("単一物体トラッカ（検出器なし）— OpenCV")
    print(f"このビルドで使えるトラッカ: {[n for n, _ in trackers]}")
    if not trackers:
        # どれも無い極端な環境でも exit 0 で案内のみ（実行経路を壊さない）
        print("使える単一物体トラッカが見つかりませんでした。")
        print("contrib 版が必要です: uv add opencv-contrib-python（headless と排他）")
        return

    results = {}
    best_name, best_pred = None, None
    for name, factory in trackers:
        mean_iou, success, preds = run_tracker(name, factory, frames, gts)
        results[name] = (mean_iou, success)
        print(f"  {name:14s}: 平均IoU={mean_iou:.3f}  成功率(IoU>0.5)={success:.3f}")
        if best_name is None:
            best_name, best_pred = name, preds

    # --- 可視化: 数フレームを並べて GT(緑) と 予測(赤) を重ねて保存 ---
    idxs = np.linspace(0, len(frames) - 1, 6).astype(int)
    tiles = []
    for t in idxs:
        vis = frames[t].copy()
        g = gts[t].astype(int)
        p = best_pred[t].astype(int)
        cv2.rectangle(vis, (g[0], g[1]), (g[2], g[3]), (0, 255, 0), 2)  # GT 緑
        cv2.rectangle(vis, (p[0], p[1]), (p[2], p[3]), (0, 0, 255), 2)  # 予測 赤
        cv2.putText(vis, f"t={t}", (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(vis)
    montage = np.vstack([np.hstack(tiles[:3]), np.hstack(tiles[3:])])
    save_path = os.path.join(out_dir, "01_single_object_tracker.png")
    cv2.imwrite(save_path, montage)
    print(f"\n可視化を保存: {save_path}  （緑=GT, 赤={best_name} 予測）")
    print("ポイント: ROI は最初の1回だけ。以降は前フレームの見え方で追従している。")


if __name__ == "__main__":
    main()
