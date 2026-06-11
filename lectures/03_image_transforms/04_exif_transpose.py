"""04: Pillow の幾何変換と EXIF Orientation の正規化。

学ぶこと:
  - PIL の size は (幅W, 高さH)。numpy/cv2 の shape=(高さH, 幅W, ch) と軸順が逆。
  - Image.resize / crop / rotate / thumbnail と、高品質縮小の Resampling.LANCZOS。
  - スマホ写真の EXIF Orientation 問題: 画素は回らず向きタグだけ記録される。
  - ImageOps.exif_transpose で画素を実際に回して向きを正規化する。
  - PIL(RGB) ↔ numpy ↔ cv2(BGR) の相互変換（境界で必ず色順変換）。

実行:
  uv run python lectures/03_image_transforms/04_exif_transpose.py
結果は outputs/03_image_transforms/ に保存される（画面表示はしない）。
"""

from __future__ import annotations

import pathlib
import sys

import cv2
import numpy as np
from PIL import Image, ImageOps

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from cv_helpers import make_color_scene_bgr, output_dir  # noqa: E402
from preprocess import load_image_oriented  # noqa: E402

# EXIF の Orientation タグ番号。6 = 「表示時に時計回り90度回す必要あり」。
_ORIENTATION_TAG = 0x0112


def main() -> None:
    out = output_dir()

    # === 1. size(W,H) と shape(H,W) の軸順 ================================
    # 同じ画像を PIL と numpy で見ると、幅と高さの並びが逆になる。
    bgr = make_color_scene_bgr(height=300, width=400)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)  # PIL は RGB 前提。境界で変換する。
    pil = Image.fromarray(rgb)
    print("[1] 軸順の違い（同じ画像）")
    print("  numpy shape :", bgr.shape, "→ (H=300, W=400, ch=3)")
    print("  PIL size    :", pil.size, "→ (W=400, H=300)  ※ 幅が先！")

    # === 2. PIL の基本変換: resize / crop / rotate / thumbnail ============
    # resize の引数も (幅, 高さ)。縮小は Resampling.LANCZOS が高品質（OpenCV の INTER_AREA 相当）。
    resized = pil.resize((200, 150), Image.Resampling.LANCZOS)
    # crop の箱は (left, upper, right, lower) = (x0, y0, x1, y1)。
    cropped = pil.crop((100, 50, 300, 200))
    # rotate は反時計回りが正。expand=True で回転後に全体が収まるよう外接箱を広げる。
    rotated = pil.rotate(30, expand=True, fillcolor=(0, 0, 0))
    # thumbnail は「与えた箱に収まる最大」へアスペクト比保持で縮小し、元オブジェクトを書き換える。
    thumb = pil.copy()
    thumb.thumbnail((120, 120), Image.Resampling.LANCZOS)
    print("\n[2] PIL 変換後の size")
    print("  resize((200,150)) :", resized.size)
    print("  crop              :", cropped.size, "（200x150）")
    print("  rotate(30,expand) :", rotated.size, "（外接箱が広がる）")
    print("  thumbnail((120,120)):", thumb.size, "（長辺120・比率保持）")
    for name, im in [("resize", resized), ("crop", cropped),
                     ("rotate", rotated), ("thumb", thumb)]:
        im.save(out / f"04_pil_{name}.png")  # PIL は save 時に拡張子で形式を決める

    # === 3. EXIF Orientation を持つ画像を作って再現する ====================
    # 「横長の画像に Orientation=6 を付けて JPEG 保存」する。
    # 6 は「画素は回っていないが、表示時に時計回り90度回せ」という意味。
    exif = pil.getexif()
    exif[_ORIENTATION_TAG] = 6
    oriented_path = out / "04_with_exif_orientation.jpg"
    pil.save(oriented_path, format="JPEG", exif=exif)
    print("\n[3] EXIF Orientation=6 を付けた JPEG を保存:", oriented_path.name)

    # === 4. 素朴に読むと向きが直らない / exif_transpose で正規化 ===========
    naive = Image.open(oriented_path)                 # 画素はそのまま（横倒しのまま）
    fixed = ImageOps.exif_transpose(Image.open(oriented_path))  # 画素を実際に回す
    print("\n[4] EXIF の扱い")
    print("  素朴に open した size      :", naive.size, " Orientation:",
          naive.getexif().get(_ORIENTATION_TAG))
    print("  exif_transpose 後の size   :", fixed.size, " Orientation:",
          fixed.getexif().get(_ORIENTATION_TAG), "（タグが消え、画素が回っている）")
    naive.convert("RGB").save(out / "04_exif_naive.png")
    fixed.convert("RGB").save(out / "04_exif_fixed.png")

    # 完成物 load_image_oriented は、この正規化を内側で行い BGR を返す。
    bgr_fixed = load_image_oriented(oriented_path)
    print("  load_image_oriented → BGR shape:",
          None if bgr_fixed is None else bgr_fixed.shape)

    # === 5. PIL ↔ numpy ↔ cv2 のラウンドトリップ ==========================
    # cv2(BGR) → RGB → PIL → numpy → cv2(BGR) と1周し、元に戻るか検証する。
    back_rgb = np.asarray(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
    back_bgr = cv2.cvtColor(back_rgb, cv2.COLOR_RGB2BGR)
    print("\n[5] cv2→PIL→numpy→cv2 のラウンドトリップ一致:",
          bool(np.array_equal(bgr, back_bgr)))

    print("\n完了。outputs/03_image_transforms/ を確認してください。")
    print("  04_exif_naive.png（横倒し）と 04_exif_fixed.png（正しい向き）を見比べてください。")


if __name__ == "__main__":
    main()
