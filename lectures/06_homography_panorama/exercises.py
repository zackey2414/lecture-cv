"""第6回 演習問題（ホモグラフィ推定とパノラマ合成）。

使い方:
  1. 各 exN_*() 関数の中の TODO を自分で実装する（最初は NotImplementedError が出る）。
  2. 実装できたら自己採点を実行:
         uv run python lectures/06_homography_panorama/exercises.py
     全問が PASS すれば "ALL PASS" と表示される。
  3. どうしても分からない時は、模範解答を実行して挙動を確認する:
         SHOW_SOLUTION=1 uv run python lectures/06_homography_panorama/exercises.py
     （まずは自力で！）

ヒント: サンプルや部品は同じフォルダの cv_helpers.py にある。
  - make_two_views() : 重なりのある2視点 (A, B, 真のホモグラフィ)
  - orb_match(), points_from_matches() : 対応点づくり
"""

from __future__ import annotations

import os
import pathlib
import sys

import cv2
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from cv_helpers import (  # noqa: E402
    make_two_views,
    orb_match,
    output_dir,
    points_from_matches,
)


# =====================================================================
# 演習（ここを実装する）
# =====================================================================

def ex1_ratio_test(knn_matches: list, ratio: float = 0.75) -> list:
    """演習1: Lowe の比率テストを実装する。

    knn_matches は knnMatch(k=2) の結果（各要素は最大2個の DMatch を持つリスト）。
    各要素について「最近傍 m の距離 < ratio × 2番目 n の距離」なら m を採用してリストで返す。
    近傍が2個に満たない要素は捨てること。
    """
    # TODO: for pair in knn_matches: 要素数チェック → m,n を取り出し
    #       m.distance < ratio * n.distance なら good に m を append
    raise NotImplementedError


def ex2_estimate_homography(pts_src: np.ndarray, pts_dst: np.ndarray
                            ) -> tuple[np.ndarray | None, int]:
    """演習2: findHomography(RANSAC) で pts_src→pts_dst を推定し (H, インライア数) を返す。

    点が4対未満なら (None, 0) を返すこと（ホモグラフィは最低4対応点が必要）。
    それ以外は cv2.findHomography(..., cv2.RANSAC, 3.0) を使い、mask の合計をインライア数とする。
    """
    # TODO: len(pts_src) < 4 なら (None, 0)
    #       H, mask = cv2.findHomography(pts_src, pts_dst, cv2.RANSAC, 3.0)
    #       return H, int(mask.sum())
    raise NotImplementedError


def ex3_reprojection_error(H: np.ndarray, pts_src: np.ndarray,
                           pts_dst: np.ndarray) -> float:
    """演習3: H で pts_src を写した点と pts_dst の平均ユークリッド距離(px)を返す。

    cv2.perspectiveTransform で写し、対応点ごとの距離を計算して平均する。
    """
    # TODO: proj = cv2.perspectiveTransform(pts_src.reshape(-1,1,2), H).reshape(-1,2)
    #       dst  = pts_dst.reshape(-1,2)
    #       return float(np.linalg.norm(proj - dst, axis=1).mean())
    raise NotImplementedError


def ex4_warp_corners(H: np.ndarray, width: int, height: int) -> np.ndarray:
    """演習4: 画像(width×height)の四隅を H で写した座標 (4,2) float32 を返す。

    四隅は [(0,0),(width,0),(width,height),(0,height)] の順。perspectiveTransform を使う。
    パノラマのキャンバス計算や平面物体の枠描画の土台になる処理。
    """
    # TODO: corners = np.float32([[0,0],[width,0],[width,height],[0,height]]).reshape(-1,1,2)
    #       return cv2.perspectiveTransform(corners, H).reshape(-1,2)
    raise NotImplementedError


def ex5_canvas_offset(shape0: tuple[int, int], shape1: tuple[int, int],
                      H: np.ndarray) -> tuple[tuple[int, int], tuple[int, int]]:
    """演習5: 画像0(基準/単位行列)と画像1(Hで写す)が両方収まるキャンバスを求める。

    返り値 ((width, height), (tx, ty))。
      - 画像0の四隅と、画像1の四隅をHで写した点を全部集め、外接矩形を求める。
      - tx, ty は最小座標を 0 に押し込む平行移動量（= -x_min, -y_min を整数化）。
    ヒント: ex4_warp_corners を H=単位行列でも使えば画像0の四隅も同じ流儀で得られる。
    """
    # TODO: 画像0四隅(np.eye(3)で写す) と 画像1四隅(Hで写す) を concatenate
    #       x_min,y_min = floor(min), x_max,y_max = ceil(max)
    #       size=(x_max-x_min, y_max-y_min), offset=(-x_min, -y_min)
    raise NotImplementedError


# =====================================================================
# 自己採点ランナー
# =====================================================================

def _grade() -> None:
    cv2.setRNGSeed(0)
    out = output_dir()  # noqa: F841  （存在確認だけ。保存はしない）
    view_a, view_b, _ = make_two_views()
    kp1, kp2, good_ref = orb_match(view_a, view_b, ratio=0.75)
    pts_a, pts_b = points_from_matches(kp1, kp2, good_ref)

    # ex1 用に knnMatch をそのまま用意（比率テスト前の生データ）。
    orb = cv2.ORB_create(nfeatures=1500)
    g1 = cv2.cvtColor(view_a, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(view_b, cv2.COLOR_BGR2GRAY)
    _, des1 = orb.detectAndCompute(g1, None)
    _, des2 = orb.detectAndCompute(g2, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    knn = bf.knnMatch(des1, des2, k=2)
    ref_good_count = sum(
        1 for p in knn if len(p) == 2 and p[0].distance < 0.75 * p[1].distance
    )

    # 固定ホモグラフィ（ex4/ex5 の決定的チェック用）。
    fixed_H = np.array([[1.05, 0.02, 40.0],
                        [0.01, 0.98, 12.0],
                        [1e-4, 5e-5, 1.0]], dtype=np.float64)

    results: list[tuple[str, bool, str]] = []

    def check(name: str, fn) -> None:
        try:
            ok, detail = fn()
            results.append((name, bool(ok), detail))
        except NotImplementedError:
            results.append((name, False, "未実装（TODOを埋めてください）"))
        except Exception as e:  # noqa: BLE001
            results.append((name, False, f"例外: {type(e).__name__}: {e}"))

    def _c1():
        good = ex1_ratio_test(knn, 0.75)
        return (len(good) == ref_good_count, f"good={len(good)} (期待 {ref_good_count})")
    check("ex1_ratio_test", _c1)

    def _c2():
        H, n = ex2_estimate_homography(pts_b, pts_a)
        ok = H is not None and H.shape == (3, 3) and n >= 100
        return (ok, f"H={'(3,3)' if H is not None else None} / インライア={n}")
    check("ex2_estimate_homography", _c2)

    def _c3():
        # インライアの対応点だけを渡す（外れ値を含むと誤差は当然大きくなるため）。
        H, mask3 = cv2.findHomography(pts_b, pts_a, cv2.RANSAC, 3.0)
        keep = mask3.ravel().astype(bool)
        err = ex3_reprojection_error(H, pts_b[keep], pts_a[keep])
        return (err < 2.0, f"再投影誤差={err:.3f}px (<2 で合格)")
    check("ex3_reprojection_error", _c3)

    def _c4():
        got = np.asarray(ex4_warp_corners(fixed_H, 640, 480), dtype=np.float64)
        exp = cv2.perspectiveTransform(
            np.float32([[0, 0], [640, 0], [640, 480], [0, 480]]).reshape(-1, 1, 2),
            fixed_H,
        ).reshape(-1, 2)
        return (got.shape == (4, 2) and np.allclose(got, exp, atol=1e-3), "四隅の投影")
    check("ex4_warp_corners", _c4)

    def _c5():
        size, offset = ex5_canvas_offset((480, 640), (480, 640), fixed_H)
        c0 = np.float32([[0, 0], [640, 0], [640, 480], [0, 480]])
        c1 = cv2.perspectiveTransform(c0.reshape(-1, 1, 2), fixed_H).reshape(-1, 2)
        pts = np.concatenate([c0, c1], axis=0)
        x_min, y_min = np.floor(pts.min(0)).astype(int)
        x_max, y_max = np.ceil(pts.max(0)).astype(int)
        exp_size = (int(x_max - x_min), int(y_max - y_min))
        exp_off = (int(-x_min), int(-y_min))
        return (tuple(size) == exp_size and tuple(offset) == exp_off,
                f"size={tuple(size)} offset={tuple(offset)}")
    check("ex5_canvas_offset", _c5)

    print("=== 採点結果 ===")
    all_ok = True
    for name, ok, detail in results:
        mark = "PASS" if ok else "FAIL"
        all_ok = all_ok and ok
        print(f"  [{mark}] {name:24s} {detail}")
    print("\nALL PASS 🎉" if all_ok else "\nまだ未達の演習があります。TODO を埋めましょう。")


# =====================================================================
# 模範解答（SHOW_SOLUTION=1 のとき本体へ差し替えて実行）。まず自力で！
# =====================================================================

def _sol_ex1(knn_matches, ratio=0.75):
    good = []
    for pair in knn_matches:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            good.append(m)
    return good


def _sol_ex2(pts_src, pts_dst):
    if pts_src is None or len(pts_src) < 4:
        return None, 0
    H, mask = cv2.findHomography(pts_src, pts_dst, cv2.RANSAC, 3.0)
    if H is None:
        return None, 0
    return H, int(mask.sum())


def _sol_ex3(H, pts_src, pts_dst):
    proj = cv2.perspectiveTransform(pts_src.reshape(-1, 1, 2), H).reshape(-1, 2)
    dst = pts_dst.reshape(-1, 2)
    return float(np.linalg.norm(proj - dst, axis=1).mean())


def _sol_ex4(H, width, height):
    corners = np.float32(
        [[0, 0], [width, 0], [width, height], [0, height]]
    ).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(corners, H).reshape(-1, 2)


def _sol_ex5(shape0, shape1, H):
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


def _install_solutions() -> None:
    g = globals()
    g["ex1_ratio_test"] = _sol_ex1
    g["ex2_estimate_homography"] = _sol_ex2
    g["ex3_reprojection_error"] = _sol_ex3
    g["ex4_warp_corners"] = _sol_ex4
    g["ex5_canvas_offset"] = _sol_ex5


if __name__ == "__main__":
    if os.environ.get("SHOW_SOLUTION") == "1":
        print("(模範解答モードで実行します)\n")
        _install_solutions()
    _grade()
