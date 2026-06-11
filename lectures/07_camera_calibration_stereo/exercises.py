"""第7回 演習問題（カメラ校正・ステレオ・エピポーラ幾何）。

使い方:
  1. 各 exN_*() の中の TODO を自分で実装する（最初は NotImplementedError が出る）。
  2. 実装できたら自己採点:
         uv run python lectures/07_camera_calibration_stereo/exercises.py
     全問 PASS で "ALL PASS" と表示される（未実装は FAIL と表示されるだけで落ちない）。
  3. どうしても分からない時は模範解答で挙動を確認（まずは自力で！）:
         SHOW_SOLUTION=1 uv run python lectures/07_camera_calibration_stereo/exercises.py

ヒント: 部品は同じフォルダの cv_helpers.py にある。
"""

from __future__ import annotations

import os
import pathlib
import sys

import cv2
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from cv_helpers import (  # noqa: E402
    TRUE_DIST,
    TRUE_K,
    chessboard_object_points,
    output_dir,
    reproject_matrix,
)


# =====================================================================
# 演習（ここを実装する）
# =====================================================================

def ex1_object_points(cols: int, rows: int, square: float) -> np.ndarray:
    """演習1: チェスボード内側角の 3D 座標 (cols*rows, 3) float32 を作る。

    盤は平面なので Z=0。並びは x が先に進む行優先:
      (0,0,0),(1,0,0),...,(cols-1,0,0),(0,1,0),... を square 倍したもの。
    """
    # TODO: np.zeros((cols*rows,3), np.float32) を作り、
    #       [:, :2] に np.mgrid[0:cols,0:rows].T.reshape(-1,2)*square を入れて返す。
    raise NotImplementedError


def ex2_reprojection_error(objp: np.ndarray, imgp: np.ndarray, rvec: np.ndarray,
                           tvec: np.ndarray, K: np.ndarray, dist: np.ndarray) -> float:
    """演習2: objp を (rvec,tvec,K,dist) で再投影し、imgp との平均ユークリッド距離(px)を返す。

    cv2.projectPoints で 3D→2D 投影し、対応点ごとの距離の平均を取る（校正品質の指標）。
    """
    # TODO: proj,_ = cv2.projectPoints(objp, rvec, tvec, K, dist)
    #       proj = proj.reshape(-1,2); det = imgp.reshape(-1,2)
    #       return float(np.linalg.norm(proj-det, axis=1).mean())
    raise NotImplementedError


def ex3_disparity_to_depth(disparity: np.ndarray, f: float, baseline: float) -> np.ndarray:
    """演習3: 視差マップ → 深度マップ。Z = f*baseline/disparity。

    視差が 0 以下（無効）の画素は深度 0 にすること（ゼロ割りを避ける）。
    入力と同じ形・float32 で返す。
    """
    # TODO: depth = np.zeros_like(disparity, np.float32)
    #       valid = disparity > 0
    #       depth[valid] = f*baseline / disparity[valid]
    #       return depth
    raise NotImplementedError


def ex4_reproject_matrix(f: float, cx: float, cy: float, baseline: float) -> np.ndarray:
    """演習4: 平行化済みステレオ（cx=cx'）の再投影行列 Q（4x4 float64）を作る。

    [X,Y,Z,W]^T = Q·[u,v,d,1]^T で 実3D点 (X/W, Y/W, Z/W) が得られる。
    この Q では W=d/baseline, Z=f となり 実Z=f*baseline/d に一致する。
    """
    # TODO: 次の 4x4 を返す（最下行に 1/baseline を置くのがミソ）:
    #   [[1,0,0,-cx],[0,1,0,-cy],[0,0,0,f],[0,0,1/baseline,0]]
    raise NotImplementedError


def ex5_pixel_to_3d(u: float, v: float, disparity: float, Q: np.ndarray) -> np.ndarray:
    """演習5: 1 画素 (u,v) と視差 disparity を Q で 3D 点 (X,Y,Z) に変換する。

    同次ベクトル [u,v,disparity,1] に Q を掛け、最後の要素 W で割って (X,Y,Z) を返す。
    reprojectImageTo3D が内部でやっている計算を 1 点ぶん手で書く。
    """
    # TODO: vec = np.array([u,v,disparity,1.0])
    #       X,Y,Z,W = Q @ vec
    #       return np.array([X/W, Y/W, Z/W])
    raise NotImplementedError


# =====================================================================
# 自己採点ランナー
# =====================================================================

def _grade() -> None:
    out = output_dir()  # noqa: F841  存在確認だけ（保存はしない）。

    # ex2 用に「歪み込みで投影した正解 imgp」を用意（誤差が 0 になるはず）。
    objp = chessboard_object_points()
    rvec = np.array([[0.12], [-0.08], [0.05]], dtype=np.float64)
    tvec = np.array([[0.5], [0.3], [15.0]], dtype=np.float64)
    imgp_exact, _ = cv2.projectPoints(objp, rvec, tvec, TRUE_K, TRUE_DIST)

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
        got = np.asarray(ex1_object_points(4, 3, 2.0), dtype=np.float64)
        exp = np.asarray(chessboard_object_points((4, 3), 2.0), dtype=np.float64)
        ok = got.shape == (12, 3) and np.allclose(got, exp)
        return ok, f"shape={got.shape} 期待={exp.shape}"
    check("ex1_object_points", _c1)

    def _c2():
        err = ex2_reprojection_error(objp, imgp_exact, rvec, tvec, TRUE_K, TRUE_DIST)
        return err < 1e-3, f"誤差={err:.2e}px（正解投影なので≈0 で合格）"
    check("ex2_reprojection_error", _c2)

    def _c3():
        disp = np.array([[60.0, 30.0], [0.0, 12.0]], dtype=np.float32)
        got = np.asarray(ex3_disparity_to_depth(disp, 600.0, 0.1), dtype=np.float64)
        exp = np.array([[1.0, 2.0], [0.0, 5.0]])  # f*baseline=60 → 60/disp、無効は0
        return got.shape == (2, 2) and np.allclose(got, exp), f"got={got.tolist()}"
    check("ex3_disparity_to_depth", _c3)

    def _c4():
        got = np.asarray(ex4_reproject_matrix(600.0, 320.0, 240.0, 0.1), dtype=np.float64)
        exp = reproject_matrix(600.0, 320.0, 240.0, 0.1)
        return got.shape == (4, 4) and np.allclose(got, exp), "Q 行列"
    check("ex4_reproject_matrix", _c4)

    def _c5():
        Q = reproject_matrix(600.0, 320.0, 240.0, 0.1)
        got = np.asarray(ex5_pixel_to_3d(400.0, 260.0, 40.0, Q), dtype=np.float64)
        vec = np.array([400.0, 260.0, 40.0, 1.0])
        X, Y, Z, W = Q @ vec
        exp = np.array([X / W, Y / W, Z / W])
        return got.shape == (3,) and np.allclose(got, exp), f"Z={got[2]:.3f}（期待 {exp[2]:.3f}）"
    check("ex5_pixel_to_3d", _c5)

    print("=== 採点結果 ===")
    all_ok = True
    for name, ok, detail in results:
        mark = "PASS" if ok else "FAIL"
        all_ok = all_ok and ok
        print(f"  [{mark}] {name:24s} {detail}")
    print("\nALL PASS 🎉" if all_ok else "\nまだ未達の演習があります。TODO を埋めましょう。")


# =====================================================================
# 模範解答（SHOW_SOLUTION=1 のとき本体へ差し替える）。まず自力で！
# =====================================================================

def _sol_ex1(cols, rows, square):
    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square
    return objp


def _sol_ex2(objp, imgp, rvec, tvec, K, dist):
    proj, _ = cv2.projectPoints(objp, rvec, tvec, K, dist)
    proj = proj.reshape(-1, 2)
    det = imgp.reshape(-1, 2)
    return float(np.linalg.norm(proj - det, axis=1).mean())


def _sol_ex3(disparity, f, baseline):
    depth = np.zeros_like(disparity, np.float32)
    valid = disparity > 0
    depth[valid] = f * baseline / disparity[valid]
    return depth


def _sol_ex4(f, cx, cy, baseline):
    return np.array(
        [[1.0, 0.0, 0.0, -cx],
         [0.0, 1.0, 0.0, -cy],
         [0.0, 0.0, 0.0, f],
         [0.0, 0.0, 1.0 / baseline, 0.0]],
        dtype=np.float64,
    )


def _sol_ex5(u, v, disparity, Q):
    vec = np.array([u, v, disparity, 1.0])
    X, Y, Z, W = Q @ vec
    return np.array([X / W, Y / W, Z / W])


def _install_solutions() -> None:
    g = globals()
    g["ex1_object_points"] = _sol_ex1
    g["ex2_reprojection_error"] = _sol_ex2
    g["ex3_disparity_to_depth"] = _sol_ex3
    g["ex4_reproject_matrix"] = _sol_ex4
    g["ex5_pixel_to_3d"] = _sol_ex5


if __name__ == "__main__":
    if os.environ.get("SHOW_SOLUTION") == "1":
        print("(模範解答モードで実行します)\n")
        _install_solutions()
    _grade()
