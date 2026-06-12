"""01: 色空間の地図 — RGB/HSV/HLS/Lab/YCrCb のスケールと相互変換。

学ぶこと:
  - 同じ色が色空間ごとに「どんな数値」で表されるかを表で対比する。
  - OpenCV 8bit の独自スケール: HSV は H=0-179・S/V=0-255、HLS は H=0-179・L/S=0-255、
    Lab は L/a/b すべて 0-255（L は 0-100 を 0-255 に、a/b は ±127 を 128 中心に）、
    YCrCb は Y/Cr/Cb=0-255。
  - 同じ cvtColor でも入力 dtype で出力スケールが変わる: float32(0-1) では HSV の H=0-360、
    Lab の L=0-100, a/b≒±127 という「本来の」値になる。
  - 往復変換 BGR→X→BGR の誤差（uint8 量子化で Lab は最大が大きい）。
  - matplotlib は RGB 前提。BGR をそのまま渡すと赤青が入れ替わる。

実行:
  uv run python lectures/43_color_spaces_and_adjustments/01_color_spaces_map.py
結果は outputs/43_color_spaces_and_adjustments/ に保存される（画面表示はしない）。
"""

from __future__ import annotations

import pathlib
import sys

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from color_helpers import (  # noqa: E402
    color_chart_patches,
    make_color_chart_bgr,
    make_photo_scene_bgr,
    output_dir,
    save_montage,
    to_rgb,
)


def _convert_pixel(bgr: tuple[int, int, int], code: int) -> tuple[int, ...]:
    """単一画素 BGR を指定コードで変換して整数タプルで返す（表示用）。"""
    px = np.uint8([[list(bgr)]])
    out = cv2.cvtColor(px, code)[0, 0]
    return tuple(int(v) for v in out)


def print_scale_table() -> None:
    """各パッチ色を主要色空間(uint8)に変換し、スケールの違いを表で示す。"""
    print("[1] 同じ色の色空間別の数値（OpenCV 8bit スケール）")
    header = f"  {'name':8s} {'BGR':>13s} {'RGB':>13s} {'HSV':>13s} {'Lab':>13s} {'YCrCb':>13s}"
    print(header)
    for name, bgr in color_chart_patches():
        rgb = (bgr[2], bgr[1], bgr[0])
        hsv = _convert_pixel(bgr, cv2.COLOR_BGR2HSV)
        lab = _convert_pixel(bgr, cv2.COLOR_BGR2Lab)
        ycc = _convert_pixel(bgr, cv2.COLOR_BGR2YCrCb)
        print(f"  {name:8s} {str(bgr):>13s} {str(rgb):>13s} "
              f"{str(hsv):>13s} {str(lab):>13s} {str(ycc):>13s}")
    print("  → H は 0-179（赤=0, 黄=30, 緑=60, 青=120）。Lab の L も 0-255 にスケールされている。")


def print_range_table() -> None:
    """各チャンネルの値域（8bit）をまとめて、スケールの食い違いを言語化する。"""
    print("\n[2] チャンネル値域（OpenCV 8bit, uint8 画像）")
    rows = [
        ("BGR/RGB", "B/G/R: 0-255", "各色の強さ"),
        ("HSV", "H: 0-179, S: 0-255, V: 0-255", "色相は 0-360 の半分"),
        ("HLS", "H: 0-179, L: 0-255, S: 0-255", "HSV と H 同じ・明度の定義が違う"),
        ("Lab", "L: 0-255, a: 0-255, b: 0-255", "本来 L:0-100, a/b:±127 → 128 中心へ"),
        ("YCrCb", "Y: 0-255, Cr: 0-255, Cb: 0-255", "輝度Yと色差Cr/Cbを分離"),
    ]
    for name, ranges, note in rows:
        print(f"  {name:8s} {ranges:34s} … {note}")


def demo_float_vs_uint8() -> None:
    """同じ純色でも入力 dtype でスケールが変わることを示す（最重要の罠）。"""
    print("\n[3] float32(0-1) 入力では H=0-360・Lab は本来値になる")
    for name, bgr01 in [("green", (0.0, 1.0, 0.0)), ("blue", (1.0, 0.0, 0.0))]:
        u8 = np.uint8([[[round(c * 255) for c in bgr01]]])
        f32 = np.float32([[list(bgr01)]])
        hsv_u8 = cv2.cvtColor(u8, cv2.COLOR_BGR2HSV)[0, 0]
        hsv_f = cv2.cvtColor(f32, cv2.COLOR_BGR2HSV)[0, 0]
        lab_f = cv2.cvtColor(f32, cv2.COLOR_BGR2Lab)[0, 0]
        print(f"  {name:6s} HSV(uint8)={tuple(int(v) for v in hsv_u8)}  "
              f"HSV(float)=({hsv_f[0]:.0f}, {hsv_f[1]:.2f}, {hsv_f[2]:.2f})  "
              f"Lab(float)=({lab_f[0]:.1f}, {lab_f[1]:.1f}, {lab_f[2]:.1f})")
    print("  → uint8 の H=60 は float では 120。スケールを混ぜると領域抽出も ΔE も崩れる。")


def demo_round_trip(bgr: np.ndarray) -> None:
    """BGR→X→BGR の往復で元に戻るか（量子化誤差）を測る。"""
    print("\n[4] 往復変換 BGR→X→BGR の誤差（uint8 画像・写真シーン）")
    specs = [
        ("HSV", cv2.COLOR_BGR2HSV, cv2.COLOR_HSV2BGR),
        ("Lab", cv2.COLOR_BGR2Lab, cv2.COLOR_Lab2BGR),
        ("YCrCb", cv2.COLOR_BGR2YCrCb, cv2.COLOR_YCrCb2BGR),
    ]
    for name, fwd, bwd in specs:
        back = cv2.cvtColor(cv2.cvtColor(bgr, fwd), bwd)
        diff = np.abs(bgr.astype(np.int32) - back.astype(np.int32))
        print(f"  {name:6s} max|diff|={int(diff.max()):3d}  mean|diff|={diff.mean():.3f}")
    print("  → 完全可逆ではない。とくに Lab は uint8 量子化で誤差が大きい（float Lab なら小さい）。")


def save_channel_montages(bgr: np.ndarray, out: pathlib.Path) -> None:
    """HSV/Lab/YCrCb の各チャンネルを白黒画像として並べ、意味を可視化する。"""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab)
    ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    panels = [
        ("input (BGR shown as RGB)", bgr),
        ("HSV: H (hue 0-179)", hsv[:, :, 0]),
        ("HSV: S (saturation)", hsv[:, :, 1]),
        ("HSV: V (value)", hsv[:, :, 2]),
        ("Lab: L (lightness)", lab[:, :, 0]),
        ("YCrCb: Y (luma)", ycc[:, :, 0]),
    ]
    save_montage(panels, out / "01_channels.png",
                 suptitle="Channel decomposition (HSV / Lab / YCrCb)", cols=3)


def save_bgr_rgb_caveat(bgr: np.ndarray, out: pathlib.Path) -> None:
    """matplotlib に BGR をそのまま渡すと色が崩れることを並べて示す。"""
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.4))
    axes[0].imshow(bgr)                       # わざと BGR をそのまま渡す（誤り）
    axes[0].set_title("WRONG: imshow(BGR)")
    axes[0].axis("off")
    axes[1].imshow(to_rgb(bgr))               # BGR2RGB してから渡す（正しい）
    axes[1].set_title("OK: imshow(BGR2RGB)")
    axes[1].axis("off")
    fig.suptitle("matplotlib assumes RGB")
    fig.tight_layout()
    fig.savefig(str(out / "01_bgr_rgb_caveat.png"), dpi=110)
    plt.close(fig)


def main() -> None:
    out = output_dir()

    chart = make_color_chart_bgr()
    from color_helpers import imwrite_unicode
    imwrite_unicode(out / "01_color_chart.png", chart)

    print_scale_table()
    print_range_table()
    demo_float_vs_uint8()

    scene = make_photo_scene_bgr()
    demo_round_trip(scene)

    save_channel_montages(scene, out)
    save_bgr_rgb_caveat(scene, out)

    print("\n完了。outputs/43_color_spaces_and_adjustments/ の 01_*.png を確認してください。")


if __name__ == "__main__":
    main()
