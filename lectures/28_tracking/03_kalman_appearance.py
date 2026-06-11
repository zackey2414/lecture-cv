"""03_kalman_appearance.py — 外見特徴で ID スイッチを減らす（DeepSORT の考え方を自前実装）

SORT（02）は動き（IoU）だけで対応付けるため、物体同士が **交差・重なる** と ID が入れ替わり
やすい（ID スイッチ）。DeepSORT はここに「外見特徴（ReID 埋め込み）」を足し、見た目が似て
いる組を優先することで ID スイッチを減らす。

本スクリプトは巨大依存 deep_sort_realtime を使わず、外見特徴を **色ヒストグラム** で代用して
DeepSORT のエッセンス（動き + 外見の合成コスト）を numpy/opencv だけで再現する。実運用では
この色ヒストグラムを小さな ReID CNN の埋め込みに差し替える（README 参照）。

実行:
    uv run python lectures/28_tracking/03_kalman_appearance.py
"""

from __future__ import annotations

import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import ensure_outdir, iou_matrix, linear_assignment, make_mot_scene  # noqa: E402


def render_colored_frame(scene, t):
    """シーンの GT を「色つき塗りつぶし長方形」として描いた BGR フレームを返す。

    各物体が固有色を持つので、検出ボックスを切り出すと外見特徴が物体ごとに区別できる。
    """
    img = np.full((scene.height, scene.width, 3), 25, np.uint8)
    for obj_id, box in scene.gt[t].items():
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(img, (x1, y1), (x2, y2), scene.colors[obj_id], -1)
    return img


def appearance_feature(frame, box, bins=8):
    """ボックス領域の HSV 色ヒストグラム（正規化済み 1 次元ベクトル）を返す。

    これが「外見の指紋」。同じ物体なら見た目（色）が安定するので特徴も似る。
    """
    h, w = frame.shape[:2]
    x1 = max(0, int(box[0]))
    y1 = max(0, int(box[1]))
    x2 = min(w, int(box[2]))
    y2 = min(h, int(box[3]))
    if x2 <= x1 or y2 <= y1:
        return np.zeros(bins * bins, dtype=np.float32)
    crop = frame[y1:y2, x1:x2]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [bins, bins], [0, 180, 0, 256])
    hist = hist.flatten().astype(np.float32)
    norm = np.linalg.norm(hist)
    return hist / norm if norm > 0 else hist


def cosine_distance_matrix(feats_a, feats_b):
    """正規化済み特徴同士のコサイン距離 (1 - 類似度) 行列を返す。"""
    if len(feats_a) == 0 or len(feats_b) == 0:
        return np.zeros((len(feats_a), len(feats_b)), dtype=float)
    A = np.asarray(feats_a, dtype=float)
    B = np.asarray(feats_b, dtype=float)
    sim = A @ B.T  # 既に L2 正規化済みなので内積 = コサイン類似度
    return 1.0 - sim


class AppearanceTracker:
    """動き(IoU) と 外見(色ヒスト) を合成コストで対応付ける簡易トラッカ。

    use_appearance=False なら純粋な IoU 追跡（SORT 的）になり、比較ができる。
    lambda_app は外見の重み（0=動きのみ, 1=外見のみ）。
    """

    def __init__(self, iou_threshold=0.2, lambda_app=0.5, use_appearance=True, max_age=3):
        self.iou_threshold = iou_threshold
        self.lambda_app = lambda_app
        self.use_appearance = use_appearance
        self.max_age = max_age
        self.tracks = []  # 各 track: dict(id, box, feat, age)
        self._next_id = 1

    def update(self, dets, feats):
        dets = np.asarray(dets, dtype=float).reshape(-1, 4)
        result = {}
        if len(self.tracks) == 0:
            for i in range(len(dets)):
                self._spawn(dets[i], feats[i])
                result[self.tracks[-1]["id"]] = dets[i]
            return result

        trk_boxes = np.array([t["box"] for t in self.tracks]).reshape(-1, 4)
        iou = iou_matrix(dets, trk_boxes)              # (D, T)
        motion_cost = 1.0 - iou
        if self.use_appearance:
            trk_feats = [t["feat"] for t in self.tracks]
            app_cost = cosine_distance_matrix(feats, trk_feats)  # (D, T)
            cost = (1 - self.lambda_app) * motion_cost + self.lambda_app * app_cost
        else:
            cost = motion_cost

        pairs = linear_assignment(cost)
        matched_d, matched_t = set(), set()
        for d, t in pairs:
            if iou[d, t] < self.iou_threshold:  # 動きが離れすぎる組は外見が似てても結ばない
                continue
            self.tracks[t]["box"] = dets[d]
            # 外見特徴は指数移動平均で滑らかに更新（急な見えの変化に頑健）
            self.tracks[t]["feat"] = 0.7 * self.tracks[t]["feat"] + 0.3 * feats[d]
            self.tracks[t]["age"] = 0
            matched_d.add(int(d))
            matched_t.add(int(t))
            result[self.tracks[t]["id"]] = dets[d]

        # 未マッチ検出 → 新規 ID
        for d in range(len(dets)):
            if d not in matched_d:
                self._spawn(dets[d], feats[d])
                result[self.tracks[-1]["id"]] = dets[d]

        # 未マッチトラック → 加齢、max_age 超で破棄
        for t in range(len(self.tracks)):
            if t not in matched_t:
                self.tracks[t]["age"] += 1
        self.tracks = [t for t in self.tracks if t["age"] <= self.max_age]
        return result

    def _spawn(self, box, feat):
        self.tracks.append({"id": self._next_id, "box": box.copy(), "feat": feat.copy(), "age": 0})
        self._next_id += 1


def count_id_switches(per_frame_pred, scene, iou_thr=0.5):
    """GT 視点の ID スイッチ数を数える。

    各フレームで GT と予測を IoU で対応付け、ある GT 物体に紐づく予測 ID が
    前回と変わったら 1 スイッチ。少ないほど良い。
    """
    gt_to_pred_prev = {}
    switches = 0
    for t, pred in enumerate(per_frame_pred):
        gt = scene.gt[t]
        gt_ids = list(gt.keys())
        pred_ids = list(pred.keys())
        if not gt_ids or not pred_ids:
            continue
        gt_boxes = np.array([gt[i] for i in gt_ids])
        pred_boxes = np.array([pred[i] for i in pred_ids])
        iou = iou_matrix(gt_boxes, pred_boxes)
        pairs = linear_assignment(1 - iou)
        for g, p in pairs:
            if iou[g, p] < iou_thr:
                continue
            gid, pid = gt_ids[g], pred_ids[p]
            if gid in gt_to_pred_prev and gt_to_pred_prev[gid] != pid:
                switches += 1
            gt_to_pred_prev[gid] = pid
    return switches


def run(scene, frames, use_appearance, lambda_app=0.5):
    """1 つの設定でシーン全体を追跡し、フレーム毎の {pred_id: box} を返す。"""
    tracker = AppearanceTracker(use_appearance=use_appearance, lambda_app=lambda_app)
    per_frame = []
    for t in range(scene.num_frames):
        dets = scene.det[t][:, :4] if len(scene.det[t]) else np.empty((0, 4))
        feats = [appearance_feature(frames[t], b) for b in dets]
        pred = tracker.update(dets, feats)
        per_frame.append(pred)
    return per_frame


def main():
    out_dir = ensure_outdir()
    # 交差が多くて ID が入れ替わりやすいシーンを作る（物体多め・取りこぼし少なめ）
    scene = make_mot_scene(num_objects=6, num_frames=50, seed=23, p_miss=0.05, p_fp=0.05)
    frames = [render_colored_frame(scene, t) for t in range(scene.num_frames)]

    print("=" * 64)
    print("外見特徴で ID スイッチを抑える（DeepSORT の核を自前実装）")

    pred_motion = run(scene, frames, use_appearance=False)
    pred_appear = run(scene, frames, use_appearance=True, lambda_app=0.6)

    sw_motion = count_id_switches(pred_motion, scene)
    sw_appear = count_id_switches(pred_appear, scene)
    print(f"  動きのみ(IoU)       の ID スイッチ数: {sw_motion}")
    print(f"  動き+外見(色ヒスト)  の ID スイッチ数: {sw_appear}")
    if sw_appear < sw_motion:
        print("  → 外見特徴を足すと ID スイッチが減った（DeepSORT の狙い通り）。")
    elif sw_appear == sw_motion:
        print("  → このシーンでは同等。交差が激しいほど外見の効果が出やすい。")
    else:
        print("  → 今回は外見の重み/特徴が噛み合わず増加。lambda や特徴設計が重要と分かる。")

    # 任意: deep_sort_realtime があれば概念紹介（実行経路では使わない）
    try:
        import deep_sort_realtime  # noqa: F401

        print("  [note] deep_sort_realtime 検出: 本番では ReID 埋め込みでさらに頑健になります。")
    except Exception:
        print("  [note] deep_sort_realtime 未導入（任意）: uv add --group track deep-sort-realtime")

    # --- 可視化: 最後のフレームに ID を重ねて保存 ---
    t = scene.num_frames - 1
    vis = frames[t].copy()
    for pid, box in pred_appear[t].items():
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 255, 255), 2)
        cv2.putText(vis, f"ID{pid}", (x1, max(10, y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
    save_path = os.path.join(out_dir, "03_appearance_tracking.png")
    cv2.imwrite(save_path, vis)
    print(f"\n可視化を保存: {save_path}")


if __name__ == "__main__":
    main()
