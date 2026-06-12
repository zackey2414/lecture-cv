"""第6回 演習の模範解答（全問 PASS する完成形）。

役割:
  - exercises.py の ex1〜ex8 を「正しい実装」に差し替えて、同じ採点ロジックを実行する。
  - 採点コード(grade)は exercises.py のものをそのまま再利用するので、重複は持たない。

使い方:
  uv run python lectures/06_homography_panorama/exercises_solutions.py
  → "ALL PASS 🎉" が出れば、全演習の参照実装が採点をすべて通過したことを意味する。

学習の進め方: まずは exercises.py を自力で埋めること。ここはあくまで答え合わせ用。
各解答の考え方は本文(README)と cv_helpers.py の同等関数のコメントも参照。
"""

from __future__ import annotations

import pathlib
import sys

import cv2
import numpy as np

# 同じフォルダの exercises.py をモジュールとして読み込めるようにする。
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import exercises as ex  # noqa: E402


# =====================================================================
# 模範解答（exercises.py の各 ex*_*() に対応）
# =====================================================================

def sol_ex1(knn_matches, ratio: float = 0.75) -> list:
    """Lowe 比率テスト: 最近傍が2番目より十分近い対応だけ残す。"""
    good = []
    for pair in knn_matches:
        if len(pair) < 2:
            continue  # 近傍が1個しか無いと比率テストできない
        m, n = pair
        if m.distance < ratio * n.distance:
            good.append(m)
    return good


def sol_ex2(pts_src, pts_dst):
    """findHomography(RANSAC) で推定し (H, インライア数) を返す。4対未満は (None, 0)。"""
    if pts_src is None or len(pts_src) < 4:
        return None, 0
    H, mask = cv2.findHomography(pts_src, pts_dst, cv2.RANSAC, 3.0)
    if H is None:
        return None, 0
    return H, int(mask.sum())


def sol_ex3(H, pts_src, pts_dst) -> float:
    """再投影誤差: H で写した点と対応点の平均ユークリッド距離。"""
    proj = cv2.perspectiveTransform(pts_src.reshape(-1, 1, 2), H).reshape(-1, 2)
    dst = pts_dst.reshape(-1, 2)
    return float(np.linalg.norm(proj - dst, axis=1).mean())


def sol_ex4(H, width, height) -> np.ndarray:
    """四隅 [(0,0),(w,0),(w,h),(0,h)] を H で写した (4,2)。"""
    corners = np.float32(
        [[0, 0], [width, 0], [width, height], [0, height]]
    ).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(corners, H).reshape(-1, 2)


def sol_ex5(shape0, shape1, H):
    """2枚分の外接矩形からキャンバスサイズと平行移動量を求める。"""
    h0, w0 = shape0
    h1, w1 = shape1
    c0 = np.float32([[0, 0], [w0, 0], [w0, h0], [0, h0]])
    c1 = cv2.perspectiveTransform(
        np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]]).reshape(-1, 1, 2), H
    ).reshape(-1, 2)
    pts = np.concatenate([c0, c1], axis=0)
    x_min, y_min = np.floor(pts.min(0)).astype(int)
    x_max, y_max = np.ceil(pts.max(0)).astype(int)
    return (int(x_max - x_min), int(y_max - y_min)), (int(-x_min), int(-y_min))


def sol_ex6(pairwise_H):
    """隣接ペアの H を順に掛け、各画像を基準フレーム(画像0)へ写す行列列を作る。"""
    m_to_ref = [np.eye(3, dtype=np.float64)]
    for H in pairwise_H:
        m_to_ref.append(m_to_ref[-1] @ H)
    return m_to_ref


def sol_ex7(img, H, size):
    """画像と全面マスクを同じ H で warp し、(warp画像, 中身マスク) を返す。"""
    warped = cv2.warpPerspective(img, H, size)
    ones = np.full(img.shape[:2], 255, np.uint8)
    wm = cv2.warpPerspective(ones, H, size)
    mask = (wm > 0).astype(np.uint8) * 255
    return warped, mask


def sol_ex8(warp_a, warp_b, w_a, w_b):
    """重み付き平均でフェザーブレンド（ゼロ割り回避＋クリップ）。"""
    wsum = w_a + w_b
    wsum_safe = wsum.copy()
    wsum_safe[wsum_safe == 0] = 1.0
    acc = (warp_a.astype(np.float64) * w_a[:, :, None]
           + warp_b.astype(np.float64) * w_b[:, :, None])
    return (acc / wsum_safe[:, :, None]).clip(0, 255).astype(np.uint8)


def _install_solutions() -> None:
    """exercises モジュールの ex*_*() を模範解答で上書きする（採点側はそのまま再利用）。"""
    ex.ex1_ratio_test = sol_ex1
    ex.ex2_estimate_homography = sol_ex2
    ex.ex3_reprojection_error = sol_ex3
    ex.ex4_warp_corners = sol_ex4
    ex.ex5_canvas_offset = sol_ex5
    ex.ex6_compose_to_reference = sol_ex6
    ex.ex7_warp_and_place = sol_ex7
    ex.ex8_feather_blend = sol_ex8


def main() -> None:
    print("(模範解答モードで採点します。全 PASS が期待値です)\n")
    _install_solutions()
    all_ok = ex.grade()
    if not all_ok:
        # 参照実装が通らないのは想定外（教材バグ）。気づけるように明示する。
        print("\n[警告] 模範解答で未達があります。教材側の確認が必要です。")


if __name__ == "__main__":
    main()
