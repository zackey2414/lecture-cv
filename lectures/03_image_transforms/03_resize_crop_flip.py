"""03: 幾何変換の基礎 — リサイズ・反転・クロップ。

学ぶこと:
  - cv2.resize の dsize は (幅W, 高さH) 順！ numpy の shape=(H, W) とは逆という最大の罠。
  - 補間の使い分け: 縮小は INTER_AREA、拡大は INTER_CUBIC / INTER_LINEAR。
  - fx/fy による倍率指定（dsize=None と併用）。
  - cv2.flip（左右/上下/両方）と numpy スライスによるクロップ。
  - 完成物: アスペクト比保持リサイズ・正方形強制（レターボックス）・中央正方形クロップ。

実行:
  uv run python lectures/03_image_transforms/03_resize_crop_flip.py
結果は outputs/03_image_transforms/ に保存される（画面表示はしない）。
"""

from __future__ import annotations

import pathlib
import sys

import cv2
import numpy as np

# matplotlib は import より前に Agg バックエンドへ固定する（画面不要・headless 対応）。
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from cv_helpers import make_color_scene_bgr, make_detail_pattern_bgr, output_dir  # noqa: E402
from preprocess import center_crop_square, resize_keep_aspect, resize_to_square  # noqa: E402


def main() -> None:
    out = output_dir()

    # === 1. dsize は (W, H) 順という罠 ====================================
    # 横長(300x400)の画像を 200x100 にリサイズする。dsize には (幅, 高さ)=(200,100) を渡す。
    bgr = make_color_scene_bgr(height=300, width=400)
    print("[1] dsize は (W, H) 順、shape は (H, W) 順")
    print("  元 shape:", bgr.shape, "（H=300, W=400）")
    resized = cv2.resize(bgr, (200, 100))  # ← (W=200, H=100)。shape とは逆順！
    print("  cv2.resize(img, (200, 100)) → shape:", resized.shape, "（H=100, W=200 になる）")
    cv2.imwrite(str(out / "03_resized_200x100.png"), resized)

    # fx/fy 指定なら dsize=None にして倍率で書ける（縦横別倍率も可）。
    half = cv2.resize(bgr, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    print("  fx=fy=0.5 → shape:", half.shape)

    # === 2. 補間方法の違い（縮小→拡大で比較） =============================
    # 高周波の同心円パターンを 1/4 に縮小し、また4倍に拡大して、補間ごとの差を作る。
    detail = make_detail_pattern_bgr(240, 240)
    small_area = cv2.resize(detail, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    small_near = cv2.resize(detail, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_NEAREST)
    # 拡大は INTER_NEAREST（カクカク）と INTER_CUBIC（滑らか）を比較。
    up_near = cv2.resize(small_area, (240, 240), interpolation=cv2.INTER_NEAREST)
    up_cubic = cv2.resize(small_area, (240, 240), interpolation=cv2.INTER_CUBIC)
    print("\n[2] 補間: 縮小は INTER_AREA、拡大は INTER_CUBIC/LINEAR が定石")

    # matplotlib で4枚を並べて比較図にする。cv2(BGR) → RGB を忘れない。
    titles = ["shrink NEAREST", "shrink AREA", "enlarge NEAREST", "enlarge CUBIC"]
    imgs = [small_near, small_area, up_near, up_cubic]
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    for ax, im, t in zip(axes, imgs, titles):
        ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))  # ← BGR→RGB 変換
        ax.set_title(t)
        ax.axis("off")
    fig.suptitle("Interpolation: AREA for shrink, CUBIC for enlarge")
    fig.tight_layout()
    fig.savefig(str(out / "03_interpolation_compare.png"), dpi=100)
    plt.close(fig)
    print("  → 03_interpolation_compare.png に4手法の比較を保存")

    # === 3. 反転（flip）==================================================
    # flipCode: 1=左右反転, 0=上下反転, -1=上下左右。データ拡張で頻出。
    flip_h = cv2.flip(bgr, 1)
    flip_v = cv2.flip(bgr, 0)
    flip_both = cv2.flip(bgr, -1)
    cv2.imwrite(str(out / "03_flip_h.png"), flip_h)
    cv2.imwrite(str(out / "03_flip_v.png"), flip_v)
    cv2.imwrite(str(out / "03_flip_both.png"), flip_both)
    print("\n[3] flip: 1=左右, 0=上下, -1=両方")

    # === 4. クロップは numpy スライス [y0:y1, x0:x1] ======================
    # 専用APIは不要。スライスは [行(y), 列(x)] の順で、(x, y) ではない点に注意。
    crop = bgr[50:200, 100:300]  # y: 50-200, x: 100-300
    print("\n[4] クロップ bgr[50:200, 100:300] → shape:", crop.shape, "（H=150, W=200）")
    cv2.imwrite(str(out / "03_crop.png"), crop)

    # === 5. 完成物の前処理関数（アスペクト比保持 / 正方形強制 / 中央クロップ）
    keep = resize_keep_aspect(bgr, long_side=256)             # 長辺256・比率保持
    square_pad = resize_to_square(bgr, size=256, pad_color=(0, 0, 0))  # 余白埋めで正方形
    square_crop = center_crop_square(bgr)                    # 中央を正方形に切り出し
    cv2.imwrite(str(out / "03_keep_aspect.png"), keep)
    cv2.imwrite(str(out / "03_square_letterbox.png"), square_pad)
    cv2.imwrite(str(out / "03_square_centercrop.png"), square_crop)
    print("\n[5] 前処理関数（完成物）")
    print("  resize_keep_aspect(long=256) → shape:", keep.shape, "（歪まない・比率保持）")
    print("  resize_to_square(256)        → shape:", square_pad.shape, "（余白で正方形・全体が残る）")
    print("  center_crop_square()         → shape:", square_crop.shape, "（端を捨てて正方形）")

    print("\n完了。outputs/03_image_transforms/ を確認してください。")


if __name__ == "__main__":
    main()
