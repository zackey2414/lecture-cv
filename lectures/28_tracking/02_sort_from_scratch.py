"""02_sort_from_scratch.py — SORT 多物体追跡を numpy で自前実装する

「検出器の出力（毎フレームのボックス群、ID 無し）」を入力に、フレーム間で同じ物体に
同じ ID を振り続ける = 多物体追跡 (Multi-Object Tracking, MOT)。その最小にして王道が
SORT (Simple Online and Realtime Tracking)。構成は次の 2 部品だけ:

  1. カルマンフィルタ（等速モデル）: 各トラックの次フレーム位置を予測する。
  2. ハンガリアン法: 予測ボックスと新しい検出を IoU コストで 1対1 対応付ける。

巨大依存（supervision の ByteTrack や deep_sort_realtime）は使わず、numpy だけで実装する。
ByteTrack / DeepSORT との違いは README の「発展」を参照（本実装はその土台）。

実行:
    uv run python lectures/28_tracking/02_sort_from_scratch.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import ensure_outdir, id_color, iou_matrix, linear_assignment, make_mot_scene  # noqa: E402


# ----------------------------------------------------------------------------
# 1) カルマンフィルタ付きの 1 トラック
# ----------------------------------------------------------------------------
# SORT の状態ベクトルは 7 次元: [u, v, s, r, u', v', s']
#   u,v : ボックス中心 / s : 面積(scale) / r : アスペクト比 / ' は速度
# 観測（検出）は 4 次元 [u, v, s, r]。等速モデルで「次フレームの中心と面積」を予測する。


def box_to_z(box):
    """[x1,y1,x2,y2] → 観測ベクトル [u,v,s,r]（中心・面積・アスペクト比）。"""
    w = box[2] - box[0]
    h = box[3] - box[1]
    u = box[0] + w / 2
    v = box[1] + h / 2
    s = w * h
    r = w / float(h) if h > 0 else 1.0
    return np.array([u, v, s, r], dtype=float)


def x_to_box(x):
    """状態ベクトル先頭 [u,v,s,r,...] → [x1,y1,x2,y2]。"""
    u, v, s, r = x[0], x[1], x[2], x[3]
    s = max(s, 1.0)
    r = max(r, 1e-3)
    w = np.sqrt(s * r)
    h = s / w
    return np.array([u - w / 2, v - h / 2, u + w / 2, v + h / 2], dtype=float)


class KalmanBoxTracker:
    """1 物体ぶんのカルマンフィルタ（定数速度モデル）。"""

    _count = 0

    def __init__(self, box):
        # 状態遷移行列 F: u <- u + u' のように位置に速度を足す（等速）
        self.F = np.eye(7)
        for i in range(3):  # u,v,s に対応する速度を足し込む
            self.F[i, i + 4] = 1.0
        # 観測行列 H: 状態の先頭 4 次元（u,v,s,r）だけが観測できる
        self.H = np.zeros((4, 7))
        self.H[:4, :4] = np.eye(4)

        # 共分散の初期化（速度成分は不確実なので大きめに）
        self.P = np.eye(7)
        self.P[4:, 4:] *= 1000.0
        self.P *= 10.0
        self.Q = np.eye(7) * 0.01      # プロセスノイズ
        self.Q[4:, 4:] *= 0.01
        self.R = np.eye(4) * 1.0       # 観測ノイズ

        self.x = np.zeros(7)
        self.x[:4] = box_to_z(box)

        # トラック管理用メタ情報
        KalmanBoxTracker._count += 1
        self.id = KalmanBoxTracker._count
        self.time_since_update = 0   # 何フレーム検出に当たっていないか
        self.hits = 0                # 累計マッチ数
        self.age = 0

    def predict(self):
        """1 フレーム先を予測して状態を進める。返り値は予測ボックス。"""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.age += 1
        self.time_since_update += 1
        return x_to_box(self.x)

    def update(self, box):
        """検出 box でカルマン更新（観測を取り込む）。"""
        z = box_to_z(box)
        y = z - self.H @ self.x                      # 残差
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)     # カルマンゲイン
        self.x = self.x + K @ y
        self.P = (np.eye(7) - K @ self.H) @ self.P
        self.time_since_update = 0
        self.hits += 1

    def get_box(self):
        return x_to_box(self.x)


# ----------------------------------------------------------------------------
# 2) SORT 本体: 予測 → 対応付け → 更新/生成/破棄
# ----------------------------------------------------------------------------


class Sort:
    """最小構成の SORT トラッカ。

    引数:
        iou_threshold: これ未満の IoU の組は対応付けない（足切り）
        max_age      : 検出に当たらないまま生かしておく最大フレーム数（オクルージョン耐性）
        min_hits     : ID を出力し始めるのに必要な連続マッチ数（チラつき防止）
    """

    def __init__(self, iou_threshold=0.3, max_age=3, min_hits=2):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_hits = min_hits
        self.trackers: list[KalmanBoxTracker] = []
        self.frame_count = 0

    def update(self, detections):
        """1 フレーム分の検出 (K,4 or K,5) を受け取り、[x1,y1,x2,y2,id] の配列を返す。"""
        self.frame_count += 1
        dets = np.asarray(detections, dtype=float).reshape(-1, detections.shape[1] if len(detections) else 4)
        dets = dets[:, :4] if dets.size else np.empty((0, 4))

        # (a) 既存トラックを 1 フレーム予測
        pred_boxes = np.array([trk.predict() for trk in self.trackers]).reshape(-1, 4)

        # (b) 予測 vs 検出 を IoU コストでハンガリアン割当
        matches, unmatched_dets, unmatched_trks = self._associate(dets, pred_boxes)

        # (c) マッチしたトラックはカルマン更新
        for trk_idx, det_idx in matches:
            self.trackers[trk_idx].update(dets[det_idx])

        # (d) マッチしなかった検出は新規トラックとして生成
        for det_idx in unmatched_dets:
            self.trackers.append(KalmanBoxTracker(dets[det_idx]))

        # (e) 長く当たらないトラックは破棄
        self.trackers = [t for t in self.trackers if t.time_since_update <= self.max_age]

        # (f) 出力: 十分なヒットがある（か初期フレーム）トラックだけを ID 付きで返す
        outputs = []
        for trk in self.trackers:
            if trk.time_since_update == 0 and (
                trk.hits >= self.min_hits or self.frame_count <= self.min_hits
            ):
                b = trk.get_box()
                outputs.append([b[0], b[1], b[2], b[3], trk.id])
        return np.array(outputs, dtype=float).reshape(-1, 5)

    def _associate(self, dets, trks):
        """検出とトラック予測を IoU で対応付ける。"""
        if len(trks) == 0:
            return [], list(range(len(dets))), []
        if len(dets) == 0:
            return [], [], list(range(len(trks)))

        iou = iou_matrix(dets, trks)              # (D, T)
        # コスト = 1 - IoU を最小化（= IoU 最大化）
        pairs = linear_assignment(1.0 - iou)

        matches = []
        matched_d, matched_t = set(), set()
        for d, t in pairs:
            if iou[d, t] < self.iou_threshold:    # 重なりが薄い組は採用しない
                continue
            matches.append((int(t), int(d)))      # (trk_idx, det_idx)
            matched_d.add(int(d))
            matched_t.add(int(t))
        unmatched_dets = [d for d in range(len(dets)) if d not in matched_d]
        unmatched_trks = [t for t in range(len(trks)) if t not in matched_t]
        return matches, unmatched_dets, unmatched_trks


def main():
    out_dir = ensure_outdir()
    KalmanBoxTracker._count = 0  # 再現性のため ID カウンタを初期化
    scene = make_mot_scene(num_objects=4, num_frames=40, seed=7)
    sort = Sort(iou_threshold=0.3, max_age=3, min_hits=2)

    print("=" * 64)
    print("自前 SORT で多物体追跡（カルマン + ハンガリアン）")
    per_frame_tracks = []
    id_set = set()
    for t in range(scene.num_frames):
        tracks = sort.update(scene.det[t])
        per_frame_tracks.append(tracks)
        for row in tracks:
            id_set.add(int(row[4]))

    print(f"  フレーム数        : {scene.num_frames}")
    print(f"  真の物体数        : {len(scene.colors)}")
    print(f"  発行した ID 総数  : {len(id_set)}  （理想は真の物体数に近いほど良い）")
    avg_tracks = np.mean([len(x) for x in per_frame_tracks])
    print(f"  平均トラック数/枚  : {avg_tracks:.2f}")

    # --- 可視化: ID 軌跡（中心点の軌跡）を 1 枚に重ねる ---
    import cv2

    canvas = np.full((scene.height, scene.width, 3), 30, np.uint8)
    trails: dict[int, list] = {}
    for tracks in per_frame_tracks:
        for x1, y1, x2, y2, tid in tracks:
            tid = int(tid)
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            trails.setdefault(tid, []).append((cx, cy))
    for tid, pts in trails.items():
        color = id_color(tid)
        for i in range(1, len(pts)):
            cv2.line(canvas, pts[i - 1], pts[i], color, 2)
        cv2.putText(canvas, f"ID{tid}", pts[-1], cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    color, 1, cv2.LINE_AA)
    save_path = os.path.join(out_dir, "02_sort_trajectories.png")
    cv2.imwrite(save_path, canvas)
    print(f"\nID 別軌跡を保存: {save_path}")
    print("ポイント: 検出が欠けても max_age の間カルマン予測で生存し、ID を保てる。")


if __name__ == "__main__":
    main()
