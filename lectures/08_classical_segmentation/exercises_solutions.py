"""exercises_solutions.py — 第8回 演習の完全な模範解答（実行すると全 PASS）。

    uv run python lectures/08_classical_segmentation/exercises_solutions.py

exercises.py と同じ採点ハーネス（_grade）と模範解答（_sol_*）を再利用する。
各 TODO 関数を模範解答へ差し替え、全 9 問が PASS することを assert で機械的に保証する。
まずは自力で exercises.py を解いてから参照すること（採点ロジックは重複させない）。

各解答は単一責務・早期 return・本章の定石を素直に写したもの:
  ex1 sure_fg        : distanceTransform → 0.5*max でしきい値 → uint8 0/255
  ex2 マーカ作成     : connectedComponents+1、unknown==255 を 0 にする
  ex3 GrabCut(rect)  : (1,65) float64 のモデルで GC_INIT_WITH_RECT → FGD|PR_FGD
  ex4 IoU            : TP/(TP+FP+FN)（分母 0 は 0.0）
  ex5 Dice           : 2TP/(2TP+FP+FN)（分母 0 は 0.0）
  ex6 inpaint TELEA  : cv2.inpaint(broken, mask, 3, INPAINT_TELEA)
  ex7 領域計数       : np.unique から -1（境界）と 1（背景）を除いた個数
  ex8 PSNR           : mse==0 は inf、ほかは 10*log10(255^2/mse)
  ex9 計数パイプライン: 二値+原画 → 距離変換マーカ → watershed → 領域数
"""

from __future__ import annotations

import pathlib
import sys

# exercises.py の採点ハーネスと模範解答(_sol_*)・差し替え関数(_install_solutions)を再利用する。
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import exercises  # noqa: E402


def main() -> None:
    print("=" * 60)
    print("08_classical_segmentation 演習 模範解答 自己採点")
    print("=" * 60)

    # 全 TODO を模範解答へ差し替えてから採点する。
    exercises._install_solutions()

    # _grade() は採点結果を一覧表示し、全問 PASS なら "ALL PASS" を出す。
    exercises._grade()

    # --- 念のため模範解答が全 PASS することを assert で機械的に保証する ---
    _assert_all_pass()
    print("\n全 9 問の模範解答が PASS することを確認しました。")


def _assert_all_pass() -> None:
    """exercises.py のサンプルと模範解答を直接突き合わせ、全問 PASS を確認する。"""
    import numpy as np

    s = exercises  # 模範解答が差し込まれた名前空間

    # ex1 / ex2: distanceTransform の sure_fg と connectedComponents マーカ
    binary = s._sample_touching_binary()
    sure_fg, unknown = s._sample_sure_and_unknown()
    assert np.array_equal(s.ex1_sure_foreground(binary), s._sol_ex1(binary))
    assert np.array_equal(s.ex2_build_markers(sure_fg, unknown), s._sol_ex2(sure_fg, unknown))

    # ex3: GrabCut(rect) は内部 RNG で僅かに揺れるため IoU しきい値で判定
    scene, truth_scene, rect = s._sample_scene()
    assert s._iou(s.ex3_grabcut_rect(scene, rect), truth_scene) > 0.85

    # ex4 / ex5: IoU と Dice
    pred_m, truth_m = s._sample_masks()
    assert abs(float(s.ex4_iou(pred_m, truth_m)) - s._sol_ex4(pred_m, truth_m)) < 1e-9
    assert abs(float(s.ex5_dice(pred_m, truth_m)) - s._sol_ex5(pred_m, truth_m)) < 1e-9

    # ex6: inpaint(TELEA)
    broken, mask = s._sample_broken()
    assert np.array_equal(s.ex6_inpaint_telea(broken, mask), s._sol_ex6(broken, mask))

    # ex7: markers からの領域計数（3 個）
    markers = s._sample_markers()
    assert int(s.ex7_count_regions(markers)) == s._sol_ex7(markers) == 3

    # ex8: PSNR
    pa, pb = s._sample_psnr_pair()
    assert abs(float(s.ex8_psnr(pa, pb)) - s._sol_ex8(pa, pb)) < 1e-6

    # ex9: 計数パイプライン（接触 3 円 → 3 個に分離）
    cbin, ccol = s._sample_count_scene()
    assert int(s.ex9_count_watershed(cbin, ccol)) == s._sol_ex9(cbin, ccol) == 3


if __name__ == "__main__":
    main()
