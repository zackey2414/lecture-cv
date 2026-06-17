"""03: Pillow 入門と PIL ↔ numpy ↔ cv2 の相互変換。

学ぶこと:
  - PIL.Image.open / save / size / mode の基本。size は (幅W, 高さH) で numpy の shape と順序が逆。
  - np.asarray(pil) で numpy へ、Image.fromarray(arr) で numpy から戻す（コピー有無も解説）。
  - PIL は RGB、cv2 は BGR。橋渡しは必ず cvtColor で。3者を1周する変換チェーンを書く。
  - mode 変換: "RGB" / "L"(グレー) / "RGBA"(アルファ)。
  - Pillow の基本操作: resize / crop / rotate / ImageDraw / filter を一通り触る。

実行:
  uv run python lectures/01_image_basics/03_pillow_numpy_interop.py
結果は lectures/01_image_basics/outputs/ に保存される。
"""

from __future__ import annotations

import pathlib
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from cv_helpers import get_sample_bgr, output_dir  # noqa: E402


def main() -> None:
    out = output_dir()

    # 出発点は cv2 の BGR 画像。Pillow に渡すにはまず RGB へ。
    bgr = get_sample_bgr()
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # === 1. numpy → PIL（Image.fromarray）と size/mode ====================
    pil = Image.fromarray(rgb)  # RGB配列から PIL Image を作る
    print("[1] PIL Image の素性")
    print("  size :", pil.size, "← (幅W, 高さH) の順。numpy shape とは順序が逆！")
    print("  mode :", pil.mode, "← 'RGB' なら3ch, 'L' ならグレー, 'RGBA' ならアルファ付き")
    print("  shape(参考):", rgb.shape, "← numpy は (H, W, C)")
    pil.save(out / "03_from_numpy.png")

    # === 2. PIL → numpy（np.asarray）======================================
    # np.asarray は可能ならコピーを作らず PIL の内部バッファを「見る」だけ（省メモリ）。
    # 書き換えたい場合は np.array(pil)（コピー）を使うか、.copy() する。
    arr = np.asarray(pil)
    print("\n[2] PIL → numpy")
    print("  np.asarray(pil).shape:", arr.shape, "dtype:", arr.dtype)
    print("  書き込み可能か(writeable):", arr.flags.writeable, "← asarray は読み取り専用になりがち")

    # === 3. 3者を1周する変換チェーン cv2(BGR) → PIL(RGB) → numpy → cv2(BGR) =
    # 実務では「cv2で読んだ画像をPILの機能で加工し、またcv2へ戻す」場面が頻繁にある。
    back_rgb = np.asarray(Image.fromarray(rgb))   # PIL を経由しても中身はRGBのまま
    back_bgr = cv2.cvtColor(back_rgb, cv2.COLOR_RGB2BGR)  # cv2 用に BGR へ戻す
    # 1周して元の BGR と完全一致すれば、変換チェーンが正しく閉じている証拠。
    print("\n[3] cv2→PIL→numpy→cv2 のラウンドトリップ")
    print("  元のBGRと一致:", np.array_equal(bgr, back_bgr))

    # === 4. mode 変換（RGB / L / RGBA） ====================================
    gray = pil.convert("L")     # グレースケール（1ch）。.size は変わらないが mode が "L" に。
    rgba = pil.convert("RGBA")  # アルファ追加（4ch）。
    gray.save(out / "03_mode_L.png")
    rgba.save(out / "03_mode_RGBA.png")
    print("\n[4] mode 変換")
    print("  L   :", gray.mode, gray.size, "→ numpy にすると", np.asarray(gray).shape, "(H,W)で2次元")
    print("  RGBA:", rgba.mode, rgba.size, "→ numpy にすると", np.asarray(rgba).shape, "(H,W,4)")

    # === 5. Pillow の基本操作: resize / crop / rotate ======================
    # resize は (幅, 高さ) を渡す。size と同じく (W, H) の順である点に注意。
    small = pil.resize((160, 120))                 # 320x240 → 160x120 に縮小
    # crop は (left, upper, right, lower) のボックス。右下は「含まない」半開区間。
    cropped = pil.crop((40, 30, 200, 150))
    # rotate は反時計回りの角度。expand=True で回転後にはみ出ない大きさに拡張する。
    rotated = pil.rotate(30, expand=True)
    small.save(out / "03_resize_160x120.png")
    cropped.save(out / "03_crop.png")
    rotated.save(out / "03_rotate30.png")
    print("\n[5] resize/crop/rotate")
    print("  resize :", small.size)
    print("  crop   :", cropped.size, "← (right-left, lower-upper)")
    print("  rotate :", rotated.size, "← expand=True で画像全体が収まる大きさに")

    # === 6. ImageDraw で文字・図形を描く ===================================
    canvas = pil.copy()             # 元を壊さないようコピーしてから描く
    draw = ImageDraw.Draw(canvas)
    # PIL の座標も (x, y)。色は RGB タプル（cv2 の BGR とは順序が逆なので混同注意）。
    draw.rectangle((20, 20, 120, 80), outline=(255, 255, 255), width=3)
    draw.line((0, 0, pil.width, pil.height), fill=(0, 0, 0), width=2)
    draw.text((25, 25), "PIL", fill=(0, 0, 0))  # 既定フォント。日本語は別途フォント指定が必要。
    canvas.save(out / "03_imagedraw.png")

    # === 7. フィルタ（ぼかし）==============================================
    blurred = pil.filter(ImageFilter.GaussianBlur(radius=3))
    blurred.save(out / "03_blur.png")
    print("\n[6][7] ImageDraw と GaussianBlur を保存しました。")

    print("\n完了。lectures/01_image_basics/outputs/ を確認してください。")


if __name__ == "__main__":
    main()
