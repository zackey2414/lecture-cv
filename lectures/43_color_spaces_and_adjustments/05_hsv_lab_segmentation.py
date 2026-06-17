"""05: HSV / Lab / YCrCb による領域抽出 — 色で「選ぶ」。

学ぶこと:
  - cv2.inRange で色空間の範囲を指定し、特定色だけの2値マスクを作る。
  - HSV は色相で選べるが照明変化に弱い。Lab の a*/b* は知覚に近く色のしきい値が直感的。
  - 肌色検出は YCrCb の色差(Cr,Cb)が定番。純粋な赤を巻き込みにくい。
    HSV で素朴にやると赤い物体を肌と誤検出しやすい（その比較も行う）。
  - ΔE による知覚的セグメンテーション: 目標色との色差 ΔE がしきい値以下の画素を抜く。
  - 評価: マスク画素数・誤検出（非肌色を拾った数）で定量確認。

実行:
  uv run python lectures/43_color_spaces_and_adjustments/05_hsv_lab_segmentation.py
結果は lectures/43_color_spaces_and_adjustments/outputs/ に保存される。
"""

from __future__ import annotations

import pathlib
import sys

import cv2
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from color_helpers import (  # noqa: E402
    delta_e_map,
    make_color_chart_bgr,
    make_skin_scene_bgr,
    output_dir,
    save_montage,
)


def _overlay_mask(bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """マスク(0/255)で抜いた領域だけ元の色を残し、他は暗くした可視化を作る。"""
    dark = (bgr * 0.25).astype(np.uint8)
    keep = cv2.bitwise_and(bgr, bgr, mask=mask)
    inv = cv2.bitwise_and(dark, dark, mask=cv2.bitwise_not(mask))
    return cv2.add(keep, inv)


def demo_hsv_vs_lab(out: pathlib.Path) -> None:
    """同じ「緑だけ抜く」を HSV と Lab で実装して比較する。"""
    print("[1] 特定色抽出: HSV(色相) と Lab(a*/b*) で『緑』を抜く")
    chart = make_color_chart_bgr()

    hsv = cv2.cvtColor(chart, cv2.COLOR_BGR2HSV)
    mask_hsv = cv2.inRange(hsv, (40, 80, 60), (80, 255, 255))  # 緑の色相帯

    lab = cv2.cvtColor(chart, cv2.COLOR_BGR2Lab)
    # Lab の a* は緑(-)〜赤(+)。8bit では 128 が中性。a*<120 で「はっきり緑側」だけ拾う
    # （無彩色の背景=128 は除外される）。
    mask_lab = cv2.inRange(lab, (0, 0, 0), (255, 120, 255))    # a* < 120 ≒ 緑側

    print(f"  HSV 緑の色相帯マスク画素数={int((mask_hsv > 0).sum())}（狭い色相だけ）")
    print(f"  Lab a*<120 マスク画素数={int((mask_lab > 0).sum())}（緑・シアン・黄など緑寄り全部）")
    print("  → HSV は色相を細く切る道具。Lab の a* は『赤(+)/緑(-)』の知覚軸そのもので、")
    print("     しきい値1つで『緑側の色』をまとめて選べる（軸の意味が直感的・背景の灰は除外）。")
    save_montage(
        [("color chart", chart),
         ("HSV inRange (green hue band)", _overlay_mask(chart, mask_hsv)),
         ("Lab inRange (a* < 120 = green side)", _overlay_mask(chart, mask_lab))],
        out / "05_hsv_vs_lab.png",
        suptitle="Color axis: HSV hue band vs Lab a* (green-red)", cols=3,
    )


def _red_circle_mask(shape: tuple[int, int]) -> np.ndarray:
    """make_skin_scene_bgr の純赤の円と同じ位置・サイズの領域マスクを作る。

    肌検出が「純赤を肌と誤検出していないか」を、この領域との重なりで数える。
    """
    h, w = shape
    m = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(m, (int(w * 0.20), int(h * 0.75)), 36, 255, -1)
    return m


def demo_skin_detection(out: pathlib.Path) -> None:
    """肌色検出: YCrCb と HSV を比較し、純赤の誤検出を見る。"""
    print("\n[2] 肌色検出: YCrCb(色差) vs HSV")
    scene = make_skin_scene_bgr()
    h, w = scene.shape[:2]
    red_region = _red_circle_mask((h, w))

    # YCrCb の定番レンジ（Cr:133-173, Cb:77-127）。輝度Yは広く取る。
    # 純赤は Cr が 246 まで跳ね上がり上限 173 を超えるので自然に除外される。
    ycrcb = cv2.cvtColor(scene, cv2.COLOR_BGR2YCrCb)
    mask_ycc = cv2.inRange(ycrcb, (0, 133, 77), (255, 173, 127))

    # HSV で素朴に「暖色なら肌」とやる（彩度上限を切らない＝純赤まで拾う失敗パターン）。
    hsv = cv2.cvtColor(scene, cv2.COLOR_BGR2HSV)
    mask_hsv = cv2.inRange(hsv, (0, 40, 50), (25, 255, 255))

    for name, mask in [("YCrCb", mask_ycc), ("HSV", mask_hsv)]:
        skin = int((mask[: h // 2] > 0).sum())              # 上半分の肌（正解）
        red_fp = int(((mask > 0) & (red_region > 0)).sum())  # 純赤の円を拾った数（誤検出）
        print(f"  {name:6s} 肌領域の検出={skin:6d}  純赤の誤検出={red_fp:6d}")
    print("  → YCrCb は純赤の誤検出=0。HSV の素朴な暖色域は純赤(彩度MAX)を肌と取り違える。")
    save_montage(
        [("skin scene", scene),
         ("YCrCb skin mask", _overlay_mask(scene, mask_ycc)),
         ("HSV warm mask (over-detects red)", _overlay_mask(scene, mask_hsv))],
        out / "05_skin_detection.png",
        suptitle="Skin detection: YCrCb vs HSV", cols=3,
    )


def demo_delta_e_segmentation(out: pathlib.Path) -> None:
    """目標色との ΔE がしきい値以下の画素を抜く知覚的セグメンテーション。"""
    print("\n[3] ΔE セグメンテーション: 目標色に近い画素を抜く")
    chart = make_color_chart_bgr()
    target_bgr = (255, 0, 0)  # 青を目標にする
    target_img = np.full_like(chart, target_bgr)

    de = delta_e_map(chart, target_img)        # 各画素の目標色との ΔE
    for thr in (30.0, 60.0):
        mask = (de <= thr).astype(np.uint8) * 255
        print(f"  ΔE<= {thr:4.0f}: 抜けた画素数={int((mask > 0).sum())}")
    mask60 = (de <= 60.0).astype(np.uint8) * 255
    print("  → しきい値を 30→60 に上げると、知覚的に青に近いマゼンタまで含まれる。")
    print("     ΔE は『どれだけ似た色まで含めるか』を知覚スケールで直接指定できる強みがある。")
    save_montage(
        [("color chart", chart),
         ("target = blue", target_img),
         ("pixels with deltaE<=60", _overlay_mask(chart, mask60))],
        out / "05_delta_e_segmentation.png",
        suptitle="Perceptual segmentation by Delta-E threshold", cols=3,
    )


def main() -> None:
    out = output_dir()
    demo_hsv_vs_lab(out)
    demo_skin_detection(out)
    demo_delta_e_segmentation(out)
    print("\n完了。lectures/43_color_spaces_and_adjustments/outputs/ の 05_*.png を確認してください。")


if __name__ == "__main__":
    main()
