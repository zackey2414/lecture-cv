"""04: ホワイトバランス / 色恒常性 — 照明の色かぶりを打ち消す。

学ぶこと:
  - 色かぶり(color cast)とは、照明色のせいで全体が青/赤に寄る現象。
  - gray-world 仮説:「シーン全体を平均すれば灰色になるはず」→ 各chの平均を揃えるゲインで補正。
  - white-patch（perfect reflector）仮説:「一番明るい点は白のはず」→ 各chの最大で割って補正。
  - 評価: 補正後の各チャンネル平均が揃うか、元シーンとの平均ΔEが下がるか。
  - 落とし穴: シーンが1色に偏ると gray-world は仮説が崩れて逆に色がつく。

実行:
  uv run python lectures/43_color_spaces_and_adjustments/04_white_balance.py
結果は outputs/43_color_spaces_and_adjustments/ に保存される。
"""

from __future__ import annotations

import pathlib
import sys

import cv2
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from color_helpers import (  # noqa: E402
    apply_color_cast,
    gray_world_white_balance,
    make_photo_scene_bgr,
    mean_delta_e,
    output_dir,
    save_montage,
)


def white_patch_white_balance(bgr: np.ndarray, percentile: float = 99.0) -> np.ndarray:
    """white-patch 法。各チャンネルの上位パーセンタイル値を白(255)に合わせる。

    「最も明るい点は本来 白」という仮説。素の max はノイズに弱いので、
    99 パーセンタイル値を白の基準にして各チャンネルを引き伸ばす（よりロバスト）。
    """
    f = bgr.astype(np.float32)
    ref = np.array([np.percentile(f[:, :, c], percentile) for c in range(3)], dtype=np.float32)
    gains = 255.0 / np.clip(ref, 1e-6, None)
    return np.clip(f * gains[None, None, :], 0, 255).astype(np.uint8)


def _channel_means(bgr: np.ndarray) -> tuple[float, float, float]:
    """(B, G, R) の平均を返す。色かぶりは平均の偏りとして現れる。"""
    m = bgr.reshape(-1, 3).mean(axis=0)
    return float(m[0]), float(m[1]), float(m[2])


def _fmt_means(bgr: np.ndarray) -> str:
    b, g, r = _channel_means(bgr)
    return f"B={b:6.1f} G={g:6.1f} R={r:6.1f}"


def demo_correction(out: pathlib.Path) -> None:
    """寒色かぶりを作り、gray-world と white-patch で補正して比較する。"""
    print("[1] 色かぶりの補正（gray-world vs white-patch）")
    scene = make_photo_scene_bgr()
    # 青を強め赤を弱めた寒色かぶりを作る（青い照明をシミュレート）。
    casted = apply_color_cast(scene, (1.45, 1.0, 0.8))

    gw = gray_world_white_balance(casted)
    wp = white_patch_white_balance(casted)

    rows = [
        ("original (target)", scene),
        ("blue color cast", casted),
        ("gray-world WB", gw),
        ("white-patch WB", wp),
    ]
    for name, img in rows:
        de = mean_delta_e(scene, img)
        print(f"  {name:20s} {_fmt_means(img)}  meanΔE(vs target)={de:6.2f}")
    print("  → かぶり画像は ΔE が大きい。WB 後は各ch平均が揃い、ΔE が下がる（元の色に近づく）。")
    save_montage(rows, out / "04_white_balance.png",
                 suptitle="White balance: gray-world vs white-patch", cols=2)


def demo_gray_world_failure(out: pathlib.Path) -> None:
    """シーンが1色に偏ると gray-world が破綻することを示す（仮説の限界）。"""
    print("\n[2] 落とし穴: 単色に偏ったシーンでは gray-world が破綻する")
    # ほぼ全面が緑の草原のような画像（小さな赤い花が少しだけ）。
    field = np.full((240, 360, 3), (40, 150, 40), dtype=np.uint8)
    cv2.circle(field, (300, 60), 18, (0, 0, 220), -1)  # わずかな赤
    fixed = gray_world_white_balance(field)
    print(f"  緑原画像        {_fmt_means(field)}")
    print(f"  gray-world後    {_fmt_means(fixed)}")
    print("  → 平均を無理に灰へ揃えるため、本来緑のシーンがくすんでマゼンタ寄りに転ぶ。")
    print("     gray-world は『色が満遍なく散っている』前提。偏ったシーンには使えない。")
    save_montage(
        [("mostly-green scene", field), ("gray-world (over-corrected)", fixed)],
        out / "04_grayworld_failure.png",
        suptitle="Gray-world fails on single-color-dominated scenes", cols=2,
    )


def main() -> None:
    out = output_dir()
    demo_correction(out)
    demo_gray_world_failure(out)
    print("\n完了。outputs/43_color_spaces_and_adjustments/ の 04_*.png を確認してください。")


if __name__ == "__main__":
    main()
