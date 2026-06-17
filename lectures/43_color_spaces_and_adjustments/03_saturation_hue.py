"""03: 彩度・色相の調整 — HSV で「色だけ」を動かす。

学ぶこと:
  - 明るさ(V)・彩度(S)・色相(H) は別々の軸。HSV に分けると、互いに干渉せず1軸だけ動かせる。
  - 彩度を S チャンネルだけ factor 倍する（0でグレースケール相当, >1で鮮やか）。明るさは保たれる。
  - 色相を H に加算して回す（mod 180）。色名は変わるが彩度・明度はほぼ不変。
  - 「特定の色相だけ」を選んで彩度を上げる選択調整（背景は触らず主役だけ鮮やかに）。
  - 効果は平均彩度（mean S）と ΔE で定量確認する。

実行:
  uv run python lectures/43_color_spaces_and_adjustments/03_saturation_hue.py
結果は lectures/43_color_spaces_and_adjustments/outputs/ に保存される。
"""

from __future__ import annotations

import pathlib
import sys

import cv2
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from color_helpers import (  # noqa: E402
    make_photo_scene_bgr,
    mean_delta_e,
    mean_saturation,
    output_dir,
    save_montage,
    scale_saturation,
    shift_hue,
)


def _value_mean(bgr: np.ndarray) -> float:
    """HSV の V（明度）の平均。彩度を動かしても明るさが保たれることの確認用。"""
    return float(cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[:, :, 2].mean())


def demo_saturation(bgr: np.ndarray, out: pathlib.Path) -> None:
    """彩度を S チャンネルだけスケールする。明度 V は変わらないことを示す。"""
    print("[1] 彩度調整: HSV の S だけを factor 倍（V は不変）")
    variants = [
        ("desaturate x0.0 (gray)", scale_saturation(bgr, 0.0)),
        ("muted x0.5", scale_saturation(bgr, 0.5)),
        ("original x1.0", bgr),
        ("vivid x1.6", scale_saturation(bgr, 1.6)),
    ]
    for name, img in variants:
        print(f"  {name:24s} mean_S={mean_saturation(img):6.1f}  mean_V={_value_mean(img):6.1f}")
    print("  → S を動かしても V（明るさ）はほぼ一定。色の鮮やかさだけが変わる。")
    save_montage(variants, out / "03_saturation.png",
                 suptitle="Saturation scaling (HSV S channel)", cols=2)


def demo_hue(bgr: np.ndarray, out: pathlib.Path) -> None:
    """色相を回す。色名は変わるが彩度・明度はほぼ不変であることを示す。"""
    print("\n[2] 色相シフト: H に加算（mod 180）。色だけ回る")
    variants = [
        ("original", bgr),
        ("hue +30", shift_hue(bgr, 30)),
        ("hue +60", shift_hue(bgr, 60)),
        ("hue +90 (approx complement)", shift_hue(bgr, 90)),
    ]
    for name, img in variants:
        de = mean_delta_e(bgr, img)
        print(f"  {name:28s} mean_S={mean_saturation(img):6.1f}  "
              f"mean_V={_value_mean(img):6.1f}  meanΔE(vs orig)={de:5.1f}")
    print("  → S/V はほぼ不変なまま、色相だけが回る（ΔE は大きい＝はっきり別の色になる）。")
    save_montage(variants, out / "03_hue.png",
                 suptitle="Hue rotation (HSV H channel, mod 180)", cols=2)


def demo_selective_saturation(bgr: np.ndarray, out: pathlib.Path) -> None:
    """特定の色相（ここでは赤系）だけ彩度を上げ、他は触らない選択調整。"""
    print("\n[3] 選択的彩度アップ: 赤系の色相だけ鮮やかにする")
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    # 赤系の色相マスク（0/179 をまたぐので2区間を OR）。彩度の下限で背景の灰色を除外。
    m1 = cv2.inRange(hsv, (0, 60, 40), (12, 255, 255))
    m2 = cv2.inRange(hsv, (168, 60, 40), (179, 255, 255))
    mask = cv2.bitwise_or(m1, m2)

    boosted = scale_saturation(bgr, 1.8)
    # マスクが立っている画素だけ boosted を、それ以外は元画像を採用する。
    mask3 = mask[:, :, None] > 0
    out_img = np.where(mask3, boosted, bgr)

    print(f"  赤系マスクの画素数={int((mask > 0).sum())}")
    print(f"  全体 mean_S: 元={mean_saturation(bgr):.1f} → 選択後={mean_saturation(out_img):.1f}")
    print(f"  全体 ΔE(元 vs 選択後)={mean_delta_e(bgr, out_img):.2f}（赤の領域だけ動くので小さめ）")
    save_montage(
        [("original", bgr), ("red-only saturation +", out_img)],
        out / "03_selective_saturation.png",
        suptitle="Selective saturation on red hue only", cols=2,
    )


def main() -> None:
    out = output_dir()
    scene = make_photo_scene_bgr()

    demo_saturation(scene, out)
    demo_hue(scene, out)
    demo_selective_saturation(scene, out)

    print("\n完了。lectures/43_color_spaces_and_adjustments/outputs/ の 03_*.png を確認してください。")


if __name__ == "__main__":
    main()
