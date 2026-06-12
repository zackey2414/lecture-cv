"""02: 明るさ・コントラスト・ガンマ・露出 — トーン（輝度）を動かす。

学ぶこと:
  - 線形調整 out = alpha*in + beta（cv2.convertScaleAbs）。alpha=コントラスト, beta=明るさ。
  - uint8 の素の加算はオーバーフローでラップアラウンド。cv2.add / convertScaleAbs は飽和(255頭打ち)。
  - ガンマ補正（非線形）を cv2.LUT で一括適用。gamma>1 で暗部持ち上げ、gamma<1 で暗くする。
  - 輝度ヒストグラム平坦化は YCrCb の Y（または Lab の L）にだけ掛けて色を崩さない。
    BGR 全チャンネルに掛けると色相がずれて色が壊れる（その失敗も並べて確認）。
  - CLAHE（局所コントラスト強調）で全体平坦化の弱点（局所の潰れ）を補う。
  - 効果は平均輝度・輝度の標準偏差(=コントラスト)・輝度ヒストグラムで定量確認する。

実行:
  uv run python lectures/43_color_spaces_and_adjustments/02_brightness_contrast_gamma.py
結果は outputs/43_color_spaces_and_adjustments/ に保存される。
"""

from __future__ import annotations

import pathlib
import sys

import cv2
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from color_helpers import (  # noqa: E402
    adjust_brightness_contrast,
    adjust_gamma,
    build_gamma_lut,
    clahe_luminance,
    equalize_luminance,
    luminance,
    make_lowlight_scene_bgr,
    make_photo_scene_bgr,
    mean_delta_e,
    output_dir,
    save_luminance_hist,
    save_montage,
)


def _tone_stats(bgr: np.ndarray) -> tuple[float, float]:
    """平均輝度（明るさ）と輝度の標準偏差（コントラストの代理量）を返す。"""
    y = luminance(bgr).astype(np.float32)
    return float(y.mean()), float(y.std())


def demo_overflow_vs_saturation() -> None:
    """uint8 のラップアラウンドと飽和演算の違いを最小例で示す。"""
    print("[1] uint8 の加算: 素の + はラップ、cv2.add は飽和")
    px = np.array([[[200, 100, 50]]], dtype=np.uint8)  # B,G,R
    add = np.array([[[100, 100, 100]]], dtype=np.uint8)
    wrap = (px + add)[0, 0]                              # 200+100=300 → 44 に巻き戻る
    sat = cv2.add(px, add)[0, 0]                          # 255 で頭打ち
    print(f"  元={tuple(int(v) for v in px[0,0])} + {tuple(int(v) for v in add[0,0])}")
    print(f"  numpy '+'(ラップ) = {tuple(int(v) for v in wrap)}  ← 200+100 が 44 に！")
    print(f"  cv2.add (飽和)    = {tuple(int(v) for v in sat)}  ← 明るさ調整はこちらが正解")


def demo_brightness_contrast(bgr: np.ndarray, out: pathlib.Path) -> None:
    """beta=明るさ, alpha=コントラストの効きを並べて確認する。"""
    print("\n[2] 線形調整 out = alpha*in + beta（convertScaleAbs）")
    variants = [
        ("original", bgr),
        ("brighter (beta=+50)", adjust_brightness_contrast(bgr, 1.0, 50)),
        ("darker (beta=-50)", adjust_brightness_contrast(bgr, 1.0, -50)),
        ("high contrast (alpha=1.6)", adjust_brightness_contrast(bgr, 1.6, -60)),
        ("low contrast (alpha=0.5)", adjust_brightness_contrast(bgr, 0.5, 64)),
    ]
    for name, img in variants:
        mean, std = _tone_stats(img)
        print(f"  {name:28s} mean_Y={mean:6.1f}  std_Y(contrast)={std:5.1f}")
    save_montage(variants, out / "02_brightness_contrast.png",
                 suptitle="Linear brightness/contrast (alpha*in + beta)", cols=3)
    save_luminance_hist(variants, out / "02_brightness_contrast_hist.png",
                        suptitle="Luminance histogram: brightness/contrast")


def demo_gamma(out: pathlib.Path) -> None:
    """ガンマ補正で露出不足の暗い写真を救う。LUT の形も確認する。"""
    print("\n[3] ガンマ補正（cv2.LUT）: 暗所写真の救済")
    dark = make_lowlight_scene_bgr()
    variants = [
        ("low-light input", dark),
        ("gamma=1.8 (lift shadows)", adjust_gamma(dark, 1.8)),
        ("gamma=2.5 (lift more)", adjust_gamma(dark, 2.5)),
        ("gamma=0.6 (darken)", adjust_gamma(dark, 0.6)),
    ]
    for name, img in variants:
        mean, std = _tone_stats(img)
        print(f"  {name:28s} mean_Y={mean:6.1f}  std_Y={std:5.1f}")
    # 代表的な LUT の数値（暗部 32 と明部 200 がどこへ移るか）。
    for g in (1.8, 0.6):
        lut = build_gamma_lut(g)
        print(f"  LUT gamma={g}: 入力32→{int(lut[32])}, 128→{int(lut[128])}, 200→{int(lut[200])}")
    save_montage(variants, out / "02_gamma.png",
                 suptitle="Gamma correction via cv2.LUT", cols=2)
    save_luminance_hist(variants, out / "02_gamma_hist.png",
                        suptitle="Luminance histogram: gamma")


def demo_equalize_clahe(out: pathlib.Path) -> None:
    """平坦化/CLAHE は輝度だけに掛ける。BGR 全chに掛けた失敗例と対比する。"""
    print("\n[4] ヒストグラム平坦化 / CLAHE は輝度チャンネルだけに掛ける")
    dark = make_lowlight_scene_bgr()

    # 失敗例: BGR の3チャンネルそれぞれを独立に平坦化 → 色相がずれて色が崩れる。
    bad = cv2.merge([cv2.equalizeHist(c) for c in cv2.split(dark)])
    # 正解1: YCrCb の Y だけ平坦化。
    eq_y = equalize_luminance(dark)
    # 正解2: Lab の L に CLAHE（局所）。
    cl = clahe_luminance(dark, clip_limit=3.0, tile=8)

    variants = [
        ("low-light input", dark),
        ("WRONG: equalize each BGR", bad),
        ("OK: equalize Y (YCrCb)", eq_y),
        ("OK: CLAHE on L (Lab)", cl),
    ]
    for name, img in variants:
        mean, std = _tone_stats(img)
        print(f"  {name:28s} mean_Y={mean:6.1f}  std_Y={std:5.1f}")
    # 「色が崩れたか」を ΔE で定量化する（入力に対する色差。BGR全chは色相がズレるので大きい）。
    de_bad = mean_delta_e(dark, bad)
    de_y = mean_delta_e(dark, eq_y)
    print(f"  入力に対する平均ΔE: BGR全ch={de_bad:5.1f}（色が崩れた）  Yのみ={de_y:5.1f}")
    print("  → 輝度だけ動かす方が、明るさは変えつつ色相のズレ（大きなΔE）を抑えられる。")
    save_montage(variants, out / "02_equalize_clahe.png",
                 suptitle="Equalize/CLAHE: luma-only keeps color", cols=2)
    save_luminance_hist(variants, out / "02_equalize_clahe_hist.png",
                        suptitle="Luminance histogram: equalize/CLAHE")


def main() -> None:
    out = output_dir()
    scene = make_photo_scene_bgr()

    demo_overflow_vs_saturation()
    demo_brightness_contrast(scene, out)
    demo_gamma(out)
    demo_equalize_clahe(out)

    print("\n完了。outputs/43_color_spaces_and_adjustments/ の 02_*.png を確認してください。")


if __name__ == "__main__":
    main()
