"""exercises_solutions.py — 第7回 演習の完全な模範解答（実行すると全 PASS）。

    uv run python lectures/07_camera_calibration_stereo/exercises_solutions.py

exercises.py と同じ採点ハーネス（_grade）を再利用する。各 TODO 関数を模範解答で差し替え、
全 9 問が PASS することを assert で保証する。まずは自力で exercises.py を解いてから参照すること。

各解答は単一責務・早期 return・本章の定石を素直に写したもの:
  ex1 盤の 3D 点      : np.mgrid で行優先 (x先行) の (N,3)・Z=0
  ex2 平均再投影誤差  : projectPoints → 対応点ごとの距離の平均(px)
  ex3 視差→深度       : Z=f*baseline/d、無効(d<=0)は 0
  ex4 再投影行列 Q    : 最下行に 1/baseline を置く 4x4
  ex5 画素→3D         : 同次 [u,v,d,1] に Q を掛け W で割る
  ex6 内部行列 K      : fx,fy,cx,cy を並べた 3x3
  ex7 深度→視差       : d=f*baseline/Z、無効(Z<=0)は 0（ex3 の逆）
  ex8 厳密 RMS        : 総二乗誤差/総点数 の平方根（視点平均ではない）
  ex9 基本行列 E      : E = K^T F K
"""

from __future__ import annotations

import pathlib
import sys

# exercises.py の採点ハーネスと模範解答(_sol_*)・差し替え関数(_install_solutions)を再利用する。
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import exercises  # noqa: E402


def main() -> None:
    print("=" * 60)
    print("07_camera_calibration_stereo 演習 模範解答 自己採点")
    print("=" * 60)

    # 全 TODO を模範解答へ差し替えてから採点する。
    exercises._install_solutions()

    # _grade() は採点結果を表示し、全問 PASS なら "ALL PASS" を出す。
    exercises._grade()

    # --- 念のため模範解答が全 PASS することを assert で機械的に保証する ---
    _assert_all_pass()
    print("\n全 9 問の模範解答が PASS することを確認しました。")


def _assert_all_pass() -> None:
    """exercises.py の採点に使う入力を最小再現し、全問 PASS を assert で確認する。"""
    import cv2
    import numpy as np

    from cv_helpers import (  # noqa: E402
        TRUE_DIST,
        TRUE_K,
        chessboard_object_points,
        mean_reprojection_error,
        reproject_matrix,
    )

    s = exercises  # 模範解答が差し込まれた名前空間

    objp = chessboard_object_points()
    rvec = np.array([[0.12], [-0.08], [0.05]], dtype=np.float64)
    tvec = np.array([[0.5], [0.3], [15.0]], dtype=np.float64)
    imgp_exact, _ = cv2.projectPoints(objp, rvec, tvec, TRUE_K, TRUE_DIST)

    # ex8 用の複数視点＋既知ノイズ（exercises._grade と同じ作り方）。
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

    # ex1
    got1 = np.asarray(s.ex1_object_points(4, 3, 2.0), dtype=np.float64)
    assert got1.shape == (12, 3)
    assert np.allclose(got1, chessboard_object_points((4, 3), 2.0))
    # ex2
    assert s.ex2_reprojection_error(objp, imgp_exact, rvec, tvec, TRUE_K, TRUE_DIST) < 1e-3
    # ex3
    disp = np.array([[60.0, 30.0], [0.0, 12.0]], dtype=np.float32)
    assert np.allclose(s.ex3_disparity_to_depth(disp, 600.0, 0.1),
                       np.array([[1.0, 2.0], [0.0, 5.0]]))
    # ex4
    assert np.allclose(s.ex4_reproject_matrix(600.0, 320.0, 240.0, 0.1),
                       reproject_matrix(600.0, 320.0, 240.0, 0.1))
    # ex5
    Q = reproject_matrix(600.0, 320.0, 240.0, 0.1)
    X, Y, Z, W = Q @ np.array([400.0, 260.0, 40.0, 1.0])
    assert np.allclose(s.ex5_pixel_to_3d(400.0, 260.0, 40.0, Q),
                       np.array([X / W, Y / W, Z / W]))
    # ex6
    assert np.allclose(s.ex6_build_intrinsics(600.0, 605.0, 320.0, 240.0),
                       np.array([[600.0, 0.0, 320.0], [0.0, 605.0, 240.0], [0.0, 0.0, 1.0]]))
    # ex7
    depth = np.array([[1.0, 2.0], [0.0, 5.0]], dtype=np.float32)
    assert np.allclose(s.ex7_depth_to_disparity(depth, 600.0, 0.1),
                       np.array([[60.0, 30.0], [0.0, 12.0]]))
    # ex8
    got8 = s.ex8_rms_reprojection_error(objpoints, imgpoints, rvecs, tvecs, TRUE_K, TRUE_DIST)
    exp8 = mean_reprojection_error(objpoints, imgpoints, rvecs, tvecs, TRUE_K, TRUE_DIST)
    assert np.isclose(got8, exp8, atol=1e-4)
    # ex9
    F = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    K = np.array([[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]])
    assert np.allclose(s.ex9_essential_from_fundamental(F, K), K.T @ F @ K)


if __name__ == "__main__":
    main()
