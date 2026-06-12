"""exercises_solutions.py — 第5回 演習の完全な模範解答（実行すると全 PASS）。

    uv run python lectures/05_classical_features_matching/exercises_solutions.py

exercises.py と同じ採点ハーネス（_grade）を再利用する。各 TODO 関数を模範解答で差し替え、
全 9 問が PASS することを assert で保証する。まずは自力で exercises.py を解いてから参照すること。

各解答は単一責務・早期 return・本章の定石を素直に写したもの:
  ex1 比率テスト       : len>=2 を確認してから m.distance < ratio*n.distance
  ex2 インライア率     : findHomography(RANSAC) の mask.sum()/N（4点未満は 0.0）
  ex3 距離種別         : SIFT→NORM_L2 / ORB→NORM_HAMMING、未知は ValueError
  ex4 テンプレ最良位置 : TM_CCOEFF_NORMED は最大なので minMaxLoc の max_loc
  ex5 線分カウント     : HoughLinesP（None は 0 本）
  ex6 対応点配列       : queryIdx/trainIdx で float32 (N,1,2) を組む
  ex7 crossCheck 対応数: BFMatcher(crossCheck=True).match() の len
  ex8 マルチスケール   : 各倍率の最大スコアを比較し最良 scale を返す
  ex9 円カウント       : HoughCircles はグレー入力（None は 0 個）
"""

from __future__ import annotations

import pathlib
import sys

# exercises.py の採点ハーネスと模範解答(_sol_*)・差し替え関数(_install_solutions)を再利用する。
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import exercises  # noqa: E402


def main() -> None:
    print("=" * 60)
    print("05_classical_features_matching 演習 模範解答 自己採点")
    print("=" * 60)

    # 全 TODO を模範解答へ差し替えてから採点する。
    exercises._install_solutions()

    # _grade() は採点結果を表示し、全問 PASS なら "ALL PASS" を出す。
    # ここでは結果を捕捉して assert したいので、results を直接集計し直す。
    # （_grade はビューだけなので、改めて模範解答を直接検証する。）
    exercises._grade()

    # --- 念のため模範解答が全 PASS することを assert で保証する ---
    _assert_all_pass()
    print("\n全 9 問の模範解答が PASS することを確認しました。")


def _assert_all_pass() -> None:
    """exercises.py の採点ロジックを最小再現し、全問 PASS を機械的に確認する。"""
    import cv2
    import numpy as np
    from cv_helpers import (  # noqa: E402
        make_object_scene, make_scene_bgr, make_shapes_bgr, warp_to_view,
    )

    scene = make_scene_bgr()
    view, _ = warp_to_view(scene)
    g1 = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(view, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(nfeatures=1500)
    k1, d1 = orb.detectAndCompute(g1, None)
    k2, d2 = orb.detectAndCompute(g2, None)
    knn = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(d1, d2, k=2)

    obj_scene, template, true_tl, _br = make_object_scene()
    big_scene = cv2.resize(obj_scene, None, fx=1.3, fy=1.3, interpolation=cv2.INTER_CUBIC)
    scale_grid = [round(s, 2) for s in np.linspace(0.6, 1.6, 21)]
    shapes_gray = cv2.cvtColor(make_shapes_bgr(), cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(shapes_gray, 50, 150)

    s = exercises  # 模範解答が差し込まれた名前空間

    # ex1
    good = s.ex1_ratio_test(knn, 0.75)
    assert len(good) > 0
    # ex2
    src = np.float32([k1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([k2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    assert 0.6 <= s.ex2_inlier_ratio(src, dst) <= 1.0
    # ex3
    assert s.ex3_norm_for("SIFT") == cv2.NORM_L2
    assert s.ex3_norm_for("ORB") == cv2.NORM_HAMMING
    # ex4
    tl = s.ex4_template_topleft(obj_scene, template)
    assert abs(int(tl[0]) - true_tl[0]) + abs(int(tl[1]) - true_tl[1]) <= 3
    # ex5
    assert s.ex5_count_line_segments(edges, 80) > 0
    # ex6
    esrc, edst = s.ex6_matched_points(k1, k2, good)
    assert esrc.shape == (len(good), 1, 2) and esrc.dtype == np.float32
    assert edst.shape == (len(good), 1, 2) and edst.dtype == np.float32
    # ex7
    assert s.ex7_cross_check_count(d1, d2, cv2.NORM_HAMMING) > 0
    # ex8
    best = s.ex8_best_template_scale(big_scene, template, scale_grid)
    assert abs(best - 1.3) <= 0.1
    # ex9
    assert s.ex9_count_hough_circles(shapes_gray, 30) > 0


if __name__ == "__main__":
    main()
