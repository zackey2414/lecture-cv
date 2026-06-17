"""01: 画像はndarray / cv2.imread・imwrite の正準形と罠。

学ぶこと:
  - 画像 = (H, W, 3) の uint8 numpy 配列。shape / dtype / 値域 / 画素アクセス(y, x)。
  - cv2.imread の IMREAD_* フラグでチャンネル数が変わる（カラー3ch / グレー1ch / アルファ4ch）。
  - 読み込み失敗時に cv2.imread は「例外ではなく None」を返す、という超重要仕様。
  - 日本語(非ASCII)パスは np.fromfile + cv2.imdecode / cv2.imencode + tofile で安全に扱う。
  - 保存形式と品質: PNG(可逆) と JPEG(非可逆・品質パラメータ)。

実行:
  uv run python lectures/01_image_basics/01_imread_imwrite.py
結果は lectures/01_image_basics/outputs/ に保存される（画面表示はしない）。
"""

from __future__ import annotations

import pathlib
import sys

import cv2
import numpy as np

# このスクリプトと同じディレクトリにある cv_helpers.py を import できるようにする。
# （スクリプトを直接実行すると、そのファイルのあるフォルダが sys.path に入るため通常は不要だが、
#   どこから呼ばれても確実に動くよう明示的に追加しておく。）
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from cv_helpers import get_sample_bgr, imread_unicode, imwrite_unicode, output_dir  # noqa: E402


def main() -> None:
    out = output_dir()

    # === 1. 画像は ndarray である ==========================================
    # サンプル画像（data/ に無ければ合成画像）を BGR で取得する。
    img = get_sample_bgr()

    # 画像の素性を必ず確認する癖をつける。実務のデバッグはまずこの3点を見る。
    print("[1] 画像の素性")
    print("  shape :", img.shape)          # (H, W, 3) 高さ・幅・チャンネル
    print("  dtype :", img.dtype)          # uint8（0〜255の整数）
    print("  min/max:", int(img.min()), int(img.max()))  # 値域は通常 0〜255

    # 画素アクセスは img[y, x]。numpy は「行(=y), 列(=x)」の順で、(x, y) ではない点に注意。
    y, x = 10, 20
    bgr_pixel = img[y, x]
    # 返ってくるのは [B, G, R] の3要素。OpenCV は RGB ではなく BGR 並びである。
    print(f"  画素(y={y}, x={x}) のBGR値:", bgr_pixel.tolist())

    # === 2. PNG と JPEG で保存（形式と品質） ================================
    # 拡張子で保存形式が決まる。PNG は可逆圧縮（劣化なし・サイズ大）、JPEG は非可逆（小さいが劣化）。
    png_path = out / "01_sample.png"
    jpg_q95 = out / "01_sample_q95.jpg"
    jpg_q30 = out / "01_sample_q30.jpg"

    # cv2.imwrite は「BGR の ndarray」を受け取る前提。imread と対称なので変換不要。
    cv2.imwrite(str(png_path), img)
    # JPEG 品質は IMWRITE_JPEG_QUALITY（0〜100, 既定95付近）。低いほど小さく劣化する。
    cv2.imwrite(str(jpg_q95), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    cv2.imwrite(str(jpg_q30), img, [cv2.IMWRITE_JPEG_QUALITY, 30])

    print("\n[2] 保存形式と品質によるファイルサイズの違い")
    for p in (png_path, jpg_q95, jpg_q30):
        print(f"  {p.name:24s}: {p.stat().st_size:7d} bytes")

    # === 3. IMREAD_* フラグでチャンネル数が変わる ==========================
    # 同じ PNG を3通りのフラグで読み、shape の違いを観察する。
    color = imread_unicode(png_path, cv2.IMREAD_COLOR)       # 必ず3ch（BGR）
    gray = imread_unicode(png_path, cv2.IMREAD_GRAYSCALE)    # 1ch（H, W）— 次元が1つ減る！
    unchanged = imread_unicode(png_path, cv2.IMREAD_UNCHANGED)  # 元のまま（アルファがあれば4ch）

    print("\n[3] IMREAD_* フラグと shape")
    print("  IMREAD_COLOR    :", color.shape, "← 常に3チャンネル")
    print("  IMREAD_GRAYSCALE:", gray.shape, "← (H, W) で次元が1つ少ない")
    print("  IMREAD_UNCHANGED:", unchanged.shape)
    # グレースケールは2次元なので、img[y, x] が3要素配列ではなく1個の輝度値になる。
    print("  グレー画素(y=10,x=20):", int(gray[10, 20]))

    # === 4. 読み込み失敗は None（例外ではない！） ==========================
    # 実在しないパスを読むと None。ここで気づかず使うと後段で AttributeError になる。
    missing = imread_unicode(out / "does_not_exist.png", cv2.IMREAD_COLOR)
    print("\n[4] 存在しないファイルの imread 結果:", missing, "← None。必ずチェックする")
    if missing is None:
        print("  → None ガードで早期return/早期raiseするのが定石。")

    # === 5. 日本語パス対応 =================================================
    # 非ASCIIパスは環境によって cv2.imread/imwrite が直接扱えない。
    # np.fromfile + cv2.imdecode / cv2.imencode + tofile のラッパなら安全。
    jp_path = out / "サンプル_日本語パス.png"
    ok = imwrite_unicode(jp_path, img)            # 安全な書き込み
    reread = imread_unicode(jp_path)              # 安全な読み込み
    print("\n[5] 日本語パスの保存/読み込み")
    print("  imwrite_unicode 成功:", ok)
    print("  読み戻し shape     :", None if reread is None else reread.shape)

    # === 6. アルファチャンネル付き PNG（4ch） ==============================
    # 透過を持つPNGは (H, W, 4) で、4つ目が alpha（0=透明, 255=不透明）。JPEGはアルファ非対応。
    h, w = img.shape[:2]
    bgra = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)  # 4ch化。alpha は初期値255（不透明）。
    # 左から右へ透明度を変えるグラデーション alpha を作る。
    alpha = np.linspace(0, 255, w, dtype=np.uint8)
    bgra[:, :, 3] = np.repeat(alpha[None, :], h, axis=0)
    rgba_path = out / "01_alpha.png"
    cv2.imwrite(str(rgba_path), bgra)             # PNG ならアルファごと保存できる
    print("\n[6] アルファPNG を保存:", rgba_path.name, "shape:", bgra.shape)

    print("\n完了。lectures/01_image_basics/outputs/ を確認してください。")


if __name__ == "__main__":
    main()
