"""第1回モジュール共通ヘルパ。

このファイルは「演習で何度も書くことになる定型処理」をまとめた小さな道具箱です。
本講座の方針として、各スクリプト（01〜04）はこのヘルパを import して使います。

ここに置く責務は次の3つだけに絞っています。
  1. サンプル画像の用意（data/ にあればそれを読む。無ければ合成画像を作る）
  2. 日本語パス・空白パスでも壊れない imread / imwrite ラッパ
  3. 出力先 outputs/01_image_basics/ の場所を返す

なぜヘルパにまとめるのか:
  - cv2.imread は「読めなかったら例外ではなく None を返す」という独特の仕様で、
    毎回 None チェックを書くのは冗長。ここで checked 版を用意しておく。
  - Windows などでは日本語(非ASCII)パスを cv2.imread/imwrite が直接扱えないことがある。
    np.fromfile + cv2.imdecode / cv2.imencode + ndarray.tofile という定石をここに固める。
  - 合成画像生成をヘルパ化することで、サンプル画像が無い環境でも全スクリプトが動く。

注意（コーディング原則との関係）:
  道具箱とはいえ「賢すぎる抽象化」は避け、各関数は1つのことだけを行います。
  スクリプト本体側ではこのヘルパを呼びつつ、学習上重要な処理（cvtColor 等）は
  あえて生のまま書いて「読んで学べる」ようにしています。
"""

from __future__ import annotations

import pathlib

import cv2
import numpy as np

# --- パス定義 ------------------------------------------------------------
# このファイルは lectures/01_image_basics/ にある。
#   parents[0] = 01_image_basics
#   parents[1] = lectures
#   parents[2] = プロジェクトルート（lecture-cv/）
# __file__ を基準にするので、どのディレクトリから実行しても場所がぶれない。
_HERE = pathlib.Path(__file__).resolve()
PROJECT_ROOT = _HERE.parents[2]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_ROOT = PROJECT_ROOT / "outputs"


def output_dir() -> pathlib.Path:
    """このモジュール専用の出力ディレクトリを作って返す。

    headless 環境では「画面に出す」代わりに「ファイルに保存して後で見る」のが基本。
    その保存先をここで一元管理する。存在しなければ自動で作成する。
    """
    d = OUTPUT_ROOT / "01_image_basics"
    d.mkdir(parents=True, exist_ok=True)
    return d


def make_sample_bgr(height: int = 240, width: int = 320) -> np.ndarray:
    """学習用の合成画像を BGR uint8 で作る（ネット非依存・サンプル不要のための要）。

    BGR/RGB の取り違えが「目で見て」分かるように、原色をはっきり配置する。
      - 左上: 赤    右上: 緑
      - 左下: 青    右下: 黄(=赤+緑)
    さらに下部に横方向グラデーション帯と白い円を描き、色変換や閾値の練習にも使える。

    OpenCV の画像は (高さ H, 幅 W, チャンネル3) の numpy 配列で、チャンネル順は B, G, R。
    つまり「赤」を作るには R チャンネル(index 2)だけ 255 にする点に注目。
    """
    # まず黒(0,0,0)で初期化。dtype=uint8 が画像の基本。
    img = np.zeros((height, width, 3), dtype=np.uint8)

    hh, hw = height // 2, width // 2
    # ndarray のスライスは [行(y), 列(x), チャンネル]。チャンネルは B=0, G=1, R=2。
    img[0:hh, 0:hw] = (0, 0, 255)      # 左上: 赤 (B=0,G=0,R=255)
    img[0:hh, hw:width] = (0, 255, 0)  # 右上: 緑
    img[hh:height, 0:hw] = (255, 0, 0)  # 左下: 青
    img[hh:height, hw:width] = (0, 255, 255)  # 右下: 黄(緑+赤)

    # 下端に横グラデーション帯を描く。x が増えるほど明るくする。
    band = np.linspace(0, 255, width, dtype=np.uint8)        # (W,) の 0→255
    band = np.repeat(band[None, :], 20, axis=0)              # (20, W) に縦コピー
    img[height - 20:height, :, :] = band[:, :, None]         # 3チャンネルへブロードキャスト

    # 中央に白い円。cv2.circle は (x, y) 中心・BGR 色で描く点に注意（x,y の順）。
    cv2.circle(img, (hw, hh), min(hh, hw) // 3, (255, 255, 255), thickness=-1)
    return img


def imread_unicode(
    path: str | pathlib.Path,
    flags: int = cv2.IMREAD_COLOR,
) -> np.ndarray | None:
    """日本語・空白を含むパスでも安全に読む imread の代替。

    cv2.imread は内部で C のファイル API を使うため、環境によっては非ASCIIパスを開けない。
    そこで「バイト列として読み込み(np.fromfile) → メモリ上でデコード(cv2.imdecode)」に分解する。
    読めなければ cv2.imread と同じく None を返す（呼び出し側で None チェックする前提）。
    """
    p = pathlib.Path(path)
    if not p.exists():
        return None
    # np.fromfile は Python の open を使うので非ASCIIパスでも確実に読める。
    buf = np.fromfile(str(p), dtype=np.uint8)
    if buf.size == 0:
        return None
    # imdecode はファイルではなく「メモリ上のバイト列」を画像に変換する。
    return cv2.imdecode(buf, flags)


def imwrite_unicode(
    path: str | pathlib.Path,
    img: np.ndarray,
    params: list[int] | None = None,
) -> bool:
    """日本語・空白を含むパスでも安全に書く imwrite の代替。

    読みの逆で「メモリ上にエンコード(cv2.imencode) → tofile でバイト列を書き出す」。
    拡張子(.png/.jpg)で形式が決まるのは cv2.imwrite と同じ。成功したら True を返す。
    """
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    ext = p.suffix  # 例: ".png"。これでエンコード形式が決まる。
    ok, buf = cv2.imencode(ext, img, params if params is not None else [])
    if not ok:
        return False
    buf.tofile(str(p))  # ndarray.tofile も Python の I/O 経由なので非ASCIIパスに強い。
    return True


def load_bgr_checked(path: str | pathlib.Path) -> np.ndarray:
    """読み込み失敗(None)を握りつぶさず、原因が分かる例外にして投げる。

    実務で最も多い事故が「imread が None を返したのに気づかず次の処理に渡して
    意味不明な AttributeError で落ちる」パターン。ここで早期に潰す。
    """
    img = imread_unicode(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(
            f"画像を読み込めませんでした: {path}\n"
            "  - パスが正しいか / ファイルが壊れていないかを確認してください。\n"
            "  - cv2.imread は失敗時に例外ではなく None を返す点に注意。"
        )
    return img


def get_sample_bgr(filename: str = "sample.jpg") -> np.ndarray:
    """data/ に画像があればそれを、無ければ合成画像を返す（BGR uint8）。

    これにより「サンプル画像を用意していない人」でも全スクリプトがそのまま動く。
    自分の画像で試したい場合は data/sample.jpg を置けば自動的にそちらが使われる。
    """
    candidate = DATA_DIR / filename
    img = imread_unicode(candidate, cv2.IMREAD_COLOR)
    if img is not None:
        return img
    return make_sample_bgr()
