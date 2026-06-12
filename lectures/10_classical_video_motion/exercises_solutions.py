"""exercises_solutions.py — 第10回 演習の完全な模範解答（実行すると全 PASS）。

    uv run python lectures/10_classical_video_motion/exercises_solutions.py

exercises.py と同じ採点ハーネス（_grade）を再利用する。各 TODO 関数を模範解答で差し替え、
全 9 問が PASS することを assert で機械的に保証する。まずは自力で exercises.py を解いてから参照すること。

各解答は単一責務・早期 return・本章の定石を素直に写したもの:
  ex1 フレーム差分マスク : absdiff → gray → threshold で 2 値マスク
  ex2 塊の数え上げ       : findContours(2つ返し) で面積 >= min_area を数える
  ex3 LK 追跡成功点      : calcOpticalFlowPyrLK の status==1 の新位置だけ残す
  ex4 密フローの大きさ   : np.linalg.norm(flow, axis=2)
  ex5 平均 EPE           : ||flow - gt|| の画素平均
  ex6 背景差分(MOG2)     : 全フレーム apply → 影落とし → 形態学 → 輪郭数
  ex7 LK 平均移動量      : 追跡成功点の ||p1 - p0|| の平均
  ex8 BackProject 最尤位置: 色モデルで確率マップを作り argmax の (x, y)
  ex9 CamShift 窓成長率  : 全フレーム CamShift し last_area / first_area
"""

from __future__ import annotations

import pathlib
import sys

# exercises.py の採点ハーネスと模範解答(_sol_*)・差し替え関数(_install_solutions)を再利用する。
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import exercises  # noqa: E402


def main() -> None:
    print("=" * 60)
    print("10_classical_video_motion 演習 模範解答 自己採点")
    print("=" * 60)

    # 全 TODO を模範解答へ差し替えてから採点する。
    exercises._install_solutions()

    # _grade() は採点結果を表示し、全問 PASS なら "ALL PASS" を出す。
    exercises._grade()

    # --- 念のため模範解答が全 PASS することを assert で機械的に保証する ---
    _assert_all_pass()
    print("\n全 9 問の模範解答が PASS することを確認しました。")


def _assert_all_pass() -> None:
    """exercises.py の採点ロジックを最小再現し、全問 PASS を機械的に確認する。"""
    import cv2
    import numpy as np

    from cv_helpers import green_window, make_motion_frames, make_translating_pair

    s = exercises  # 模範解答が差し込まれた名前空間

    frames = make_motion_frames()
    prev_bgr, next_bgr = frames[18], frames[19]
    g0 = cv2.cvtColor(frames[10], cv2.COLOR_BGR2GRAY)
    g1 = cv2.cvtColor(frames[11], cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(g0, maxCorners=200, qualityLevel=0.05,
                                 minDistance=7, blockSize=7)
    a, b, shift = make_translating_pair()
    flow = cv2.calcOpticalFlowFarneback(
        cv2.cvtColor(a, cv2.COLOR_BGR2GRAY),
        cv2.cvtColor(b, cv2.COLOR_BGR2GRAY), None, 0.5, 3, 15, 3, 5, 1.2, 0)
    gt = np.zeros_like(flow)
    gt[..., 0], gt[..., 1] = shift
    init_win = green_window(frames[0])
    hist = s._hue_hist(frames[0], init_win)

    # ex1: 差分マスクの shape/dtype/値域。
    m = s.ex1_frame_diff_mask(prev_bgr, next_bgr, 25)
    assert m.shape == prev_bgr.shape[:2] and m.dtype == np.uint8
    assert set(np.unique(m)).issubset({0, 255})
    # ex2: 既知の塊数（2つ）。
    blob = np.zeros((120, 200), np.uint8)
    cv2.circle(blob, (50, 60), 20, 255, -1)
    cv2.circle(blob, (150, 60), 18, 255, -1)
    assert s.ex2_count_blobs(blob, 150) == 2
    # ex3: 追跡成功点の (M, 2)。
    good = np.asarray(s.ex3_lk_good_new(g0, g1, p0))
    assert good.ndim == 2 and good.shape[1] == 2 and len(good) > 0
    # ex4: 大きさマップ shape。
    mag = s.ex4_flow_magnitude(flow)
    assert mag.shape == flow.shape[:2]
    # ex5: 既知シフトの EPE はほぼ 0。
    assert s.ex5_mean_epe(flow, gt) < 0.5
    # ex6: 背景差分の動体数（動く物体 2 つを概ね拾う）。
    assert s.ex6_bgsub_box_count(frames, 150) >= 1
    # ex7: LK 平均移動量は正の値。
    assert s.ex7_lk_mean_displacement(g0, g1, p0) > 0.0
    # ex8: 初期窓に緑円が写る frames[0] では、最尤位置は窓の近傍（円の上）に乗る。
    px, py = s.ex8_backproject_peak(frames[0], init_win)
    wx, wy, ww, wh = init_win
    assert wx - 5 <= px <= wx + ww + 5 and wy - 5 <= py <= wy + wh + 5
    # ex9: 緑円は成長するので窓成長率 > 1.0。
    assert s.ex9_camshift_grow_ratio(frames, hist, init_win) > 1.0


if __name__ == "__main__":
    main()
