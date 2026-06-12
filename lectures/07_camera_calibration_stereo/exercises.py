"""第7回 演習問題（カメラ校正・ステレオ・エピポーラ幾何）。

使い方:
  1. 各 exN_*() の中の TODO を自分で実装する（最初は NotImplementedError が出る）。
  2. 実装できたら自己採点:
         uv run python lectures/07_camera_calibration_stereo/exercises.py
     全問 PASS で "ALL PASS" と表示される（未実装は FAIL と表示されるだけで落ちない）。
  3. どうしても分からない時は模範解答で挙動を確認（まずは自力で！）:
         SHOW_SOLUTION=1 uv run python lectures/07_camera_calibration_stereo/exercises.py
     完全な模範解答（全問 PASS）は exercises_solutions.py でも確認できる。

難易度は易→難。ex1〜ex5 が校正・視差・3D 復元の基礎、ex6〜ex9 が内部行列の組み立て・
深度→視差の逆算・RMS 再投影誤差の厳密計算・基本行列(E=K^T F K)へと段階的に踏み込む。

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
    mean_reprojection_error,
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


def ex6_build_intrinsics(fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    """演習6: 焦点距離 (fx,fy) と主点 (cx,cy) から内部行列 K（3x3 float64）を組む。

    K = [[fx, 0, cx],
         [0, fy, cy],
         [0,  0,  1]]
    （せん断 skew は 0 とみなす。これがカメラモデルの中核 K。）
    """
    # TODO: 上の 3x3 を np.array(..., dtype=np.float64) で返す。
    raise NotImplementedError


def ex7_depth_to_disparity(depth: np.ndarray, f: float, baseline: float) -> np.ndarray:
    """演習7: 深度マップ → 視差マップ（ex3 の逆）。disparity = f*baseline/Z。

    深度が 0 以下（無効）の画素は視差 0 にすること（ゼロ割りを避ける）。
    入力と同じ形・float32 で返す。視差↔深度が反比例の関係にあることを手で確かめる。
    """
    # TODO: disp = np.zeros_like(depth, np.float32)
    #       valid = depth > 0
    #       disp[valid] = f*baseline / depth[valid]
    #       return disp
    raise NotImplementedError


def ex8_rms_reprojection_error(objpoints: list, imgpoints: list, rvecs: list,
                               tvecs: list, K: np.ndarray, dist: np.ndarray) -> float:
    """演習8: 複数視点の RMS 再投影誤差を「総二乗誤差 / 総点数 の平方根」で厳密計算する。

    calibrateCamera の戻り値（RMS）の定義そのもの。各視点を projectPoints で再投影し、
    全点の二乗距離 (dx^2+dy^2) を**全視点で合算**してから総点数で割り、最後に sqrt する。
    視点ごとに平均を取って平均し直すのは厳密には RMS と一致しない点に注意。
    """
    # TODO: total_sq, total_n = 0.0, 0
    #       各 (objp,imgp,rvec,tvec) で proj,_ = cv2.projectPoints(...)
    #         total_sq += np.sum((proj.reshape(-1,2) - imgp.reshape(-1,2))**2)
    #         total_n  += len(imgp の点数)
    #       return float(np.sqrt(total_sq/total_n))
    raise NotImplementedError


def ex9_essential_from_fundamental(F: np.ndarray, K: np.ndarray) -> np.ndarray:
    """演習9: 基礎行列 F と内部行列 K から基本行列 E = K^T · F · K を計算する。

    F は画素座標でのエピポーラ拘束 x_R^T F x_L = 0、E は正規化座標での同じ拘束。
    左右で K が共通の場合の関係 E = K^T F K を素直に行列積で書く（3x3 float64）。
    """
    # TODO: return K.T @ F @ K （float64 の 3x3 で返す）。
    raise NotImplementedError


# =====================================================================
# 自己採点ランナー
# =====================================================================

def _grade() -> None:
    out = output_dir()  # noqa: F841  存在確認だけ（保存はしない）。

    # ex2/ex8 用に「歪み込みで投影した正解 imgp」を用意（誤差が 0 になるはず）。
    objp = chessboard_object_points()
    rvec = np.array([[0.12], [-0.08], [0.05]], dtype=np.float64)
    tvec = np.array([[0.5], [0.3], [15.0]], dtype=np.float64)
    imgp_exact, _ = cv2.projectPoints(objp, rvec, tvec, TRUE_K, TRUE_DIST)

    # ex8 用に「複数視点＋既知ノイズ」を作る（総二乗ベースの RMS を検算する）。
    rng = np.random.default_rng(7)
    objpoints, imgpoints, rvecs, tvecs = [], [], [], []
    for k in range(3):
        rv = np.array([[0.1 * k], [-0.05 * k], [0.02 * k]], dtype=np.float64)
        tv = np.array([[0.4 + 0.1 * k], [0.2], [14.0 + k]], dtype=np.float64)
        proj, _ = cv2.projectPoints(objp, rv, tv, TRUE_K, TRUE_DIST)
        noisy = proj + rng.normal(0, 0.3, size=proj.shape).astype(np.float64)
        objpoints.append(objp)
        imgpoints.append(noisy.astype(np.float32))
        rvecs.append(rv)
        tvecs.append(tv)

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

    def _c6():
        got = np.asarray(ex6_build_intrinsics(600.0, 605.0, 320.0, 240.0), dtype=np.float64)
        exp = np.array([[600.0, 0.0, 320.0], [0.0, 605.0, 240.0], [0.0, 0.0, 1.0]])
        return got.shape == (3, 3) and np.allclose(got, exp), "K 行列"
    check("ex6_build_intrinsics", _c6)

    def _c7():
        depth = np.array([[1.0, 2.0], [0.0, 5.0]], dtype=np.float32)
        got = np.asarray(ex7_depth_to_disparity(depth, 600.0, 0.1), dtype=np.float64)
        exp = np.array([[60.0, 30.0], [0.0, 12.0]])  # f*baseline=60 → 60/Z、無効は0
        return got.shape == (2, 2) and np.allclose(got, exp), f"got={got.tolist()}"
    check("ex7_depth_to_disparity", _c7)

    def _c8():
        got = float(ex8_rms_reprojection_error(objpoints, imgpoints, rvecs, tvecs,
                                               TRUE_K, TRUE_DIST))
        exp = mean_reprojection_error(objpoints, imgpoints, rvecs, tvecs, TRUE_K, TRUE_DIST)
        return np.isclose(got, exp, atol=1e-4), f"RMS={got:.4f}px（期待 {exp:.4f}px）"
    check("ex8_rms_reprojection_error", _c8)

    def _c9():
        F = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
        K = np.array([[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]])
        got = np.asarray(ex9_essential_from_fundamental(F, K), dtype=np.float64)
        exp = K.T @ F @ K
        return got.shape == (3, 3) and np.allclose(got, exp), "E=K^T F K"
    check("ex9_essential_from_fundamental", _c9)

    print("=== 採点結果 ===")
    all_ok = True
    for name, ok, detail in results:
        mark = "PASS" if ok else "FAIL"
        all_ok = all_ok and ok
        print(f"  [{mark}] {name:28s} {detail}")
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


def _sol_ex6(fx, fy, cx, cy):
    return np.array(
        [[fx, 0.0, cx],
         [0.0, fy, cy],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _sol_ex7(depth, f, baseline):
    disp = np.zeros_like(depth, np.float32)
    valid = depth > 0
    disp[valid] = f * baseline / depth[valid]
    return disp


def _sol_ex8(objpoints, imgpoints, rvecs, tvecs, K, dist):
    total_sq, total_n = 0.0, 0
    for objp, imgp, rvec, tvec in zip(objpoints, imgpoints, rvecs, tvecs):
        proj, _ = cv2.projectPoints(objp, rvec, tvec, K, dist)
        total_sq += float(np.sum((proj.reshape(-1, 2) - imgp.reshape(-1, 2)) ** 2))
        total_n += len(imgp.reshape(-1, 2))
    return float(np.sqrt(total_sq / total_n)) if total_n else 0.0


def _sol_ex9(F, K):
    return (K.T @ F @ K).astype(np.float64)


def _install_solutions() -> None:
    g = globals()
    g["ex1_object_points"] = _sol_ex1
    g["ex2_reprojection_error"] = _sol_ex2
    g["ex3_disparity_to_depth"] = _sol_ex3
    g["ex4_reproject_matrix"] = _sol_ex4
    g["ex5_pixel_to_3d"] = _sol_ex5
    g["ex6_build_intrinsics"] = _sol_ex6
    g["ex7_depth_to_disparity"] = _sol_ex7
    g["ex8_rms_reprojection_error"] = _sol_ex8
    g["ex9_essential_from_fundamental"] = _sol_ex9


if __name__ == "__main__":
    if os.environ.get("SHOW_SOLUTION") == "1":
        print("(模範解答モードで実行します)\n")
        _install_solutions()
    _grade()
