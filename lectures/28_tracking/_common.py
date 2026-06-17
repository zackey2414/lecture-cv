"""28_tracking 共有ユーティリティ。

この回の全スクリプトが使う「合成シーン生成・IoU・割当（ハンガリアン/貪欲）・描画」を
1ファイルにまとめる。各スクリプトは先頭で

    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from _common import ...

と読み込むので、`uv run python lectures/28_tracking/<file>.py` のように
どのディレクトリから実行しても import できる（カレントディレクトリに依存しない）。

設計方針:
- 入力（画像・連番フレーム・検出）は外部データに頼らず **合成生成** する。
  追跡の本質（フレーム間でどう同じ物体に同じ ID を振るか）は、
  正解（ground truth, GT）が分かっている合成データの方が学習に向く。
- 衝突しうる巨大依存（supervision/ByteTrack, deep_sort_realtime, motmetrics, TrackEval）は
  **使わず**、numpy + opencv だけで自前実装する。scipy は任意（無ければ貪欲法に自動フォールバック）。

このファイル自体も `uv run python lectures/28_tracking/_common.py` で簡単な自己テストが走り
exit 0 で終了する。
"""

from __future__ import annotations

import numpy as np

# ----------------------------------------------------------------------------
# IoU（Intersection over Union）— 追跡・検出評価の最小単位
# ----------------------------------------------------------------------------
# box は xyxy 形式（左上 x1,y1・右下 x2,y2）で統一する。
# 追跡では「フレーム t の予測ボックス」と「フレーム t の検出/GT ボックス」が
# どれだけ重なるかを IoU で測り、それを手かかりに同一物体を対応付ける。


def iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    """2 つのボックス a, b（各 [x1,y1,x2,y2]）の IoU を返す。"""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    # 交差領域の幅・高さ（負なら重なっていない → 0 にクリップ）
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """ボックス集合 A(N,4) と B(M,4) の総当たり IoU 行列 (N,M) をベクトル化で返す。

    ループで書くと O(N*M) の Python ループになり遅いので、numpy のブロードキャストで
    一気に計算する。これが多物体追跡の「コスト行列」の土台になる。
    """
    A = np.asarray(boxes_a, dtype=float).reshape(-1, 4)
    B = np.asarray(boxes_b, dtype=float).reshape(-1, 4)
    if len(A) == 0 or len(B) == 0:
        return np.zeros((len(A), len(B)), dtype=float)
    # (N,1) と (1,M) に整形してブロードキャスト
    x11, y11, x12, y12 = A[:, 0:1], A[:, 1:2], A[:, 2:3], A[:, 3:4]
    x21, y21, x22, y22 = B[:, 0], B[:, 1], B[:, 2], B[:, 3]
    iw = np.clip(np.minimum(x12, x22) - np.maximum(x11, x21), 0, None)
    ih = np.clip(np.minimum(y12, y22) - np.maximum(y11, y21), 0, None)
    inter = iw * ih
    area_a = (x12 - x11) * (y12 - y11)
    area_b = (x22 - x21) * (y22 - y21)
    union = area_a + area_b - inter
    return np.where(union > 0, inter / union, 0.0)


# ----------------------------------------------------------------------------
# 割当（assignment）— 行 i と列 j を 1 対 1 に対応付け、総コストを最小化する
# ----------------------------------------------------------------------------
# 追跡では「既存トラックの予測位置（行）」と「新しい検出（列）」を対応付ける。
# コスト = (1 - IoU) などにすれば「IoU が高い組ほど優先して結ぶ」ことになる。
# 最適解はハンガリアン法 = scipy.optimize.linear_sum_assignment。
# scipy が無い環境でも動くよう、貪欲法（最小コストから順に確定）も自前で持つ。


def _greedy_assignment(cost: np.ndarray) -> np.ndarray:
    """貪欲法による割当。最小コストのセルから順に行・列を埋めていく。

    最適ではない（局所最適）が、追跡の入門としては振る舞いが分かりやすい。
    返り値は (K,2) の (row, col) ペア配列。
    """
    cost = np.asarray(cost, dtype=float).copy()
    n, m = cost.shape
    pairs = []
    used_r, used_c = set(), set()
    # 小さいコスト順にセルを走査
    order = np.dstack(np.unravel_index(np.argsort(cost, axis=None), cost.shape))[0]
    for r, c in order:
        if r in used_r or c in used_c:
            continue
        used_r.add(int(r))
        used_c.add(int(c))
        pairs.append((int(r), int(c)))
        if len(pairs) == min(n, m):
            break
    return np.array(pairs, dtype=int).reshape(-1, 2)


def linear_assignment(cost: np.ndarray) -> np.ndarray:
    """総コスト最小の 1対1 割当を返す (K,2) の (row, col) 配列。

    scipy があればハンガリアン法（最適）、無ければ貪欲法にフォールバックする。
    どちらも「割り当てるかどうか（IoU 閾値での足切り）」は呼び出し側の責務。
    """
    cost = np.asarray(cost, dtype=float)
    if cost.size == 0:
        return np.empty((0, 2), dtype=int)
    try:
        from scipy.optimize import linear_sum_assignment

        rows, cols = linear_sum_assignment(cost)
        return np.stack([rows, cols], axis=1)
    except Exception:  # scipy 不在など → 貪欲法
        return _greedy_assignment(cost)


# ----------------------------------------------------------------------------
# 合成 MOT シーン生成 — 壁で反射しながら動く複数の長方形
# ----------------------------------------------------------------------------
# GT（正解の軌跡）を完全に把握できるので、評価指標（MOTA/IDF1）を厳密に検証できる。
# 物体同士が交差するように初期速度を散らし、ID スイッチが起きやすい状況を作る。

# ID ごとの色（BGR）。描画と外見特徴（03）に使う。
PALETTE_BGR = [
    (60, 76, 231),   # 赤
    (113, 204, 46),  # 緑
    (219, 152, 52),  # 青
    (15, 196, 241),  # 黄
    (182, 89, 155),  # 紫
    (0, 165, 255),   # 橙
]


class MOTScene:
    """合成多物体シーン。

    属性:
        gt[t]  : {obj_id: np.array([x1,y1,x2,y2])}  フレーム t の正解ボックス群
        det[t] : np.array (K,5) = [x1,y1,x2,y2,score]  フレーム t の検出（ノイズ込み）
        colors : {obj_id: (B,G,R)}
    """

    def __init__(self, gt, det, colors, width, height):
        self.gt = gt
        self.det = det
        self.colors = colors
        self.width = width
        self.height = height
        self.num_frames = len(gt)


def make_mot_scene(
    num_objects: int = 4,
    num_frames: int = 40,
    width: int = 320,
    height: int = 240,
    seed: int = 0,
    p_miss: float = 0.10,
    p_fp: float = 0.15,
    pos_noise: float = 2.5,
    size_noise: float = 1.5,
) -> MOTScene:
    """壁で反射する長方形群の GT 軌跡と、ノイズ付き検出を生成する。

    引数:
        p_miss   : 各物体を「検出し損ねる」確率（FN の原因）
        p_fp     : 各フレームに偽検出（FP）を 1 個足す確率
        pos_noise: 検出ボックス中心に乗せる正規ノイズの標準偏差[px]
        size_noise: 検出ボックスの幅高さに乗せるノイズの標準偏差[px]
    """
    rng = np.random.default_rng(seed)

    # --- 各物体の初期状態（中心・速度・サイズ・色）を決める ---
    cx = rng.uniform(40, width - 40, num_objects)
    cy = rng.uniform(40, height - 40, num_objects)
    # 速度は交差が起きるよう左右上下に散らす
    vx = rng.uniform(-6, 6, num_objects)
    vy = rng.uniform(-6, 6, num_objects)
    w = rng.uniform(22, 34, num_objects)
    h = rng.uniform(22, 34, num_objects)
    colors = {i: PALETTE_BGR[i % len(PALETTE_BGR)] for i in range(num_objects)}

    gt, det = [], []
    for _ in range(num_frames):
        # --- GT を更新（等速 + 壁反射）---
        cx += vx
        cy += vy
        # 壁にぶつかったら速度反転（中心がフレーム内に収まるよう）
        for i in range(num_objects):
            if cx[i] < w[i] / 2 or cx[i] > width - w[i] / 2:
                vx[i] *= -1
                cx[i] = np.clip(cx[i], w[i] / 2, width - w[i] / 2)
            if cy[i] < h[i] / 2 or cy[i] > height - h[i] / 2:
                vy[i] *= -1
                cy[i] = np.clip(cy[i], h[i] / 2, height - h[i] / 2)

        frame_gt = {}
        dets = []
        for i in range(num_objects):
            box = np.array(
                [cx[i] - w[i] / 2, cy[i] - h[i] / 2, cx[i] + w[i] / 2, cy[i] + h[i] / 2],
                dtype=float,
            )
            frame_gt[i] = box
            # --- 検出を作る: 一定確率で取りこぼし（FN）---
            if rng.random() < p_miss:
                continue
            dcx = cx[i] + rng.normal(0, pos_noise)
            dcy = cy[i] + rng.normal(0, pos_noise)
            dw = w[i] + rng.normal(0, size_noise)
            dh = h[i] + rng.normal(0, size_noise)
            score = float(np.clip(rng.normal(0.85, 0.08), 0.3, 0.99))
            dets.append([dcx - dw / 2, dcy - dh / 2, dcx + dw / 2, dcy + dh / 2, score])

        # --- 偽検出（FP）を一定確率で足す ---
        if rng.random() < p_fp:
            fx = rng.uniform(20, width - 20)
            fy = rng.uniform(20, height - 20)
            fw, fh = rng.uniform(20, 32), rng.uniform(20, 32)
            dets.append(
                [fx - fw / 2, fy - fh / 2, fx + fw / 2, fy + fh / 2, float(rng.uniform(0.3, 0.6))]
            )

        gt.append(frame_gt)
        det.append(np.array(dets, dtype=float).reshape(-1, 5))

    return MOTScene(gt, det, colors, width, height)


# ----------------------------------------------------------------------------
# 描画ユーティリティ（headless 前提: imshow は呼ばず画像配列を返す/保存する）
# ----------------------------------------------------------------------------


def render_boxes(width, height, boxes, colors=None, labels=None, filled=False, bg=30):
    """ボックス群を描いた BGR uint8 画像を返す。

    boxes  : (K,4) xyxy
    colors : 各ボックスの BGR（None なら白）
    labels : 各ボックスの文字列（ID 等）
    filled : True なら塗りつぶし（合成映像→色 blob 検出のデモ用）
    """
    import cv2

    img = np.full((height, width, 3), bg, dtype=np.uint8)
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    for k, b in enumerate(boxes):
        color = colors[k] if colors is not None else (255, 255, 255)
        p1 = (int(round(b[0])), int(round(b[1])))
        p2 = (int(round(b[2])), int(round(b[3])))
        cv2.rectangle(img, p1, p2, color, -1 if filled else 2)
        if labels is not None:
            cv2.putText(
                img, str(labels[k]), (p1[0], max(10, p1[1] - 3)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA,
            )
    return img


def id_color(track_id: int):
    """トラック ID から安定した BGR 色を作る（描画用）。"""
    return PALETTE_BGR[track_id % len(PALETTE_BGR)]


def ensure_outdir(module_id: str = "28_tracking") -> str:
    """lectures/28_tracking/outputs/ を作って絶対パスを返す。"""
    import pathlib

    # このファイルの隣（lectures/28_tracking/）に outputs/ を作る
    out = pathlib.Path(__file__).resolve().parent / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    return str(out)


if __name__ == "__main__":
    # 簡単な自己テスト（exit 0 で終了）
    print("[_common] self-test")
    a = np.array([0, 0, 10, 10], float)
    b = np.array([5, 5, 15, 15], float)
    print(f"  iou_xyxy = {iou_xyxy(a, b):.4f} (expect 0.1429)")
    M = iou_matrix(np.array([a, b]), np.array([a, b]))
    print(f"  iou_matrix diag = {np.diag(M)}  (expect [1,1])")
    cost = 1 - M
    pairs = linear_assignment(cost)
    print(f"  assignment = {pairs.tolist()}")
    scene = make_mot_scene(num_objects=4, num_frames=10, seed=1)
    n_det = sum(len(d) for d in scene.det)
    print(f"  scene: frames={scene.num_frames}, objects={len(scene.colors)}, total_det={n_det}")
    print("[_common] OK")
