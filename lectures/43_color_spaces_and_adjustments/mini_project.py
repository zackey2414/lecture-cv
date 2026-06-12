"""章末ミニプロジェクト: 色調整パイプライン + 領域抽出 + 定量評価。

ストーリー:
  「色かぶり＋露出不足＋低コントラスト」で撮れてしまった一枚を、
  ホワイトバランス → 露出(ガンマ) → 局所コントラスト(CLAHE) → 彩度 の順に補正し、
  各段で『どれだけ元の見え方に近づいたか』を 平均輝度・平均彩度・平均ΔE で定量化する。
  仕上げに、補正後の画像から色領域(青)と肌色領域を抽出して保存する。

この1本で 01〜05 の道具（色空間・明るさ/ガンマ・彩度・WB・inRange/ΔE）を総動員する。

実行:
  uv run python lectures/43_color_spaces_and_adjustments/mini_project.py
結果は outputs/43_color_spaces_and_adjustments/ に保存される（画面表示はしない）。
"""

from __future__ import annotations

import pathlib
import sys

import cv2
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from color_helpers import (  # noqa: E402
    adjust_gamma,
    apply_color_cast,
    clahe_luminance,
    gray_world_white_balance,
    luminance,
    make_photo_scene_bgr,
    mean_delta_e,
    mean_saturation,
    output_dir,
    save_luminance_hist,
    save_montage,
    scale_saturation,
)


def make_degraded(target: np.ndarray) -> np.ndarray:
    """目標画像から「色かぶり＋露出不足＋低コントラスト」の劣化版を作る。"""
    casted = apply_color_cast(target, (1.4, 1.0, 0.78))        # 寒色かぶり
    dark = cv2.convertScaleAbs(casted, alpha=0.55, beta=-18)    # 暗く・低コントラスト
    return dark


def restore_pipeline(degraded: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """劣化画像を段階的に補正し、各ステージの (名前, 画像) を順に返す。

    各段は前段の出力を入力にする（パイプライン）。順序にも意味がある:
      1. WB で色かぶりを先に除く（後段の露出/彩度がかぶり色を増幅しないように）
      2. ガンマで露出を持ち上げる
      3. CLAHE で局所コントラストを回復
      4. 彩度を軽く補う
    """
    wb = gray_world_white_balance(degraded)
    expo = adjust_gamma(wb, 1.7)
    local = clahe_luminance(expo, clip_limit=2.0, tile=8)
    vivid = scale_saturation(local, 1.2)
    return [
        ("1 white balance", wb),
        ("2 + gamma expose", expo),
        ("3 + CLAHE local", local),
        ("4 + saturation", vivid),
    ]


def evaluate(target: np.ndarray, stages: list[tuple[str, np.ndarray]]) -> None:
    """各ステージを 平均輝度・平均彩度・目標との平均ΔE で評価して表示する。"""
    print("[evaluate] 目標画像にどれだけ近づいたか（平均ΔE が小さいほど良い）")
    print(f"  {'stage':18s} {'mean_Y':>8s} {'mean_S':>8s} {'meanDeltaE':>11s}")
    for name, img in stages:
        y = float(luminance(img).mean())
        s = mean_saturation(img)
        de = mean_delta_e(target, img)
        print(f"  {name:18s} {y:8.1f} {s:8.1f} {de:11.2f}")


def segment_outputs(bgr: np.ndarray, out: pathlib.Path) -> None:
    """補正後の画像から色領域(青)と肌色領域(YCrCb)を抽出して保存する。"""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask_blue = cv2.inRange(hsv, (100, 80, 50), (130, 255, 255))

    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    mask_skin = cv2.inRange(ycrcb, (0, 133, 77), (255, 173, 127))

    print("\n[segment] 補正後画像からの領域抽出")
    print(f"  青領域 画素数={int((mask_blue > 0).sum())}  肌色領域 画素数={int((mask_skin > 0).sum())}")

    blue_only = cv2.bitwise_and(bgr, bgr, mask=mask_blue)
    skin_only = cv2.bitwise_and(bgr, bgr, mask=mask_skin)
    save_montage(
        [("restored", bgr), ("blue (HSV inRange)", blue_only),
         ("skin (YCrCb inRange)", skin_only)],
        out / "mini_segmentation.png",
        suptitle="Segmentation from the restored image", cols=3,
    )


def main() -> None:
    out = output_dir()

    target = make_photo_scene_bgr()          # 本来の見え方（評価の基準）
    degraded = make_degraded(target)         # 撮れてしまった劣化版
    stages = restore_pipeline(degraded)
    restored = stages[-1][1]

    # 入力と各段を1枚に並べて保存。
    panels = [("target (reference)", target), ("degraded input", degraded)] + stages
    save_montage(panels, out / "mini_pipeline.png",
                 suptitle="Restoration pipeline (WB -> gamma -> CLAHE -> saturation)", cols=3)
    save_luminance_hist(
        [("target", target), ("degraded", degraded), ("restored", restored)],
        out / "mini_pipeline_hist.png", suptitle="Luminance: target / degraded / restored",
    )

    # 評価: degraded を起点に、各段で ΔE がどう動くか。
    evaluate(target, [("0 degraded input", degraded)] + stages)
    segment_outputs(restored, out)

    de0 = mean_delta_e(target, degraded)
    de1 = mean_delta_e(target, restored)
    print(f"\n[summary] 目標との平均ΔE: 劣化={de0:.2f} → 補正後={de1:.2f} "
          f"（{de0 - de1:+.2f} 改善）")
    print("完了。outputs/43_color_spaces_and_adjustments/ の mini_*.png を確認してください。")


if __name__ == "__main__":
    main()
