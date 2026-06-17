"""第3回モジュール共通ヘルパ（サンプル画像生成・I/O・出力先）。

このモジュールは「色空間・描画・幾何変換」を学ぶための小さな道具箱です。
本講座の方針として、各スクリプト（01〜04）と exercises.py はこのファイルを import して使います。

ここに置く責務は次の3つに絞っています（単一責務）。
  1. 学習用の合成画像を作る（data/ にあればそれを読む。無くても必ず動く）
  2. 日本語・空白を含むパスでも壊れない imread / imwrite ラッパ
  3. 出力先 lectures/03_image_transforms/outputs/ の場所を返す

なぜ「他モジュールから import しない」のか:
  各モジュールを自己完結させるためです。第1回にも似た cv_helpers.py がありますが、
  あえて重複させて、このモジュールだけを取り出しても動くようにしています。
  教材は「読んで学ぶ」ものなので、依存を減らして追いやすさを優先します。
"""

from __future__ import annotations

import pathlib

import cv2
import numpy as np

# --- パス定義 ------------------------------------------------------------
# このファイルは lectures/03_image_transforms/ にある。
#   _HERE.parent = このモジュールのディレクトリ（lectures/03_image_transforms/）
#   parents[2]   = プロジェクトルート（lecture-cv/）
# __file__ 基準なので、どのディレクトリから実行しても場所がぶれない。
# 出力はスクリプト隣の outputs/ に置き、入力素材 data/ はプロジェクト直下を参照する。
_HERE = pathlib.Path(__file__).resolve()
PROJECT_ROOT = _HERE.parents[2]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_ROOT = _HERE.parent / "outputs"
MODULE_ID = "03_image_transforms"


def output_dir() -> pathlib.Path:
    """このモジュール専用の出力ディレクトリを作って返す。

    headless 環境では「画面に出す」代わりに「ファイルに保存して後で見る」のが基本。
    その保存先をここで一元管理する。存在しなければ自動で作成する。
    """
    d = OUTPUT_ROOT
    d.mkdir(parents=True, exist_ok=True)
    return d


def imread_unicode(
    path: str | pathlib.Path,
    flags: int = cv2.IMREAD_COLOR,
) -> np.ndarray | None:
    """日本語・空白を含むパスでも安全に読む imread の代替。

    cv2.imread は内部で C のファイル API を使うため、環境によっては非ASCIIパスを開けない。
    「バイト列として読む(np.fromfile) → メモリ上でデコード(cv2.imdecode)」に分解して回避する。
    読めなければ cv2.imread と同じく None を返す（呼び出し側で None チェックする前提）。
    """
    p = pathlib.Path(path)
    if not p.exists():
        return None
    buf = np.fromfile(str(p), dtype=np.uint8)
    if buf.size == 0:
        return None
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
    ext = p.suffix
    ok, buf = cv2.imencode(ext, img, params if params is not None else [])
    if not ok:
        return False
    buf.tofile(str(p))
    return True


def make_color_scene_bgr(height: int = 300, width: int = 400) -> np.ndarray:
    """色マスク・描画・リサイズすべてに使える合成シーンを BGR uint8 で作る。

    薄いグレー背景の上に、純色の物体を4つ配置する。
    HSV の inRange で「特定の色だけ」を抜き出す練習に都合がよいよう、
    背景は彩度(S)が低いグレー、物体は彩度の高い純色にしてある。
      - 赤い円        （左上）  BGR=(0,0,255)   HSV H=0 付近（0/179 をまたぐ）
      - 緑の長方形    （右上）  BGR=(0,255,0)   HSV H=60
      - 青い円        （左下）  BGR=(255,0,0)   HSV H=120
      - 黄色の長方形  （右下）  BGR=(0,255,255) HSV H=30
    """
    # 背景は薄いグレー（B=G=R なので彩度ゼロ）。物体の色と区別しやすい。
    img = np.full((height, width, 3), 200, dtype=np.uint8)

    qh, qw = height // 2, width // 2
    # cv2.circle / cv2.rectangle の座標は (x, y)、色は BGR タプル。thickness=-1 で塗りつぶし。
    cv2.circle(img, (qw // 2, qh // 2), min(qh, qw) // 3, (0, 0, 255), thickness=-1)      # 赤円
    cv2.rectangle(img, (qw + qw // 4, qh // 4),
                  (width - qw // 4, qh - qh // 4), (0, 255, 0), thickness=-1)             # 緑矩形
    cv2.circle(img, (qw // 2, qh + qh // 2), min(qh, qw) // 3, (255, 0, 0), thickness=-1)  # 青円
    cv2.rectangle(img, (qw + qw // 4, qh + qh // 4),
                  (width - qw // 4, height - qh // 4), (0, 255, 255), thickness=-1)        # 黄矩形
    return img


def make_detail_pattern_bgr(height: int = 240, width: int = 240) -> np.ndarray:
    """補間方法の違いが目で見えるよう、高周波の同心円＋斜線パターンを作る。

    縮小（INTER_AREA）と拡大（INTER_CUBIC/LINEAR）で結果がどう変わるかは、
    細かい縞のような高周波成分があるほど分かりやすい。学習用にあえて作り込む。
    """
    img = np.zeros((height, width, 3), dtype=np.uint8)
    cy, cx = height // 2, width // 2
    yy, xx = np.mgrid[0:height, 0:width]
    # 中心からの距離で白黒が細かく交互に変わる同心円（高周波）。
    dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    rings = ((dist // 6).astype(np.int32) % 2) * 255
    img[:] = rings[:, :, None].astype(np.uint8)
    # 斜めの細い赤線を重ねて、方向性のあるディテールも加える。
    for k in range(-height, width, 20):
        cv2.line(img, (k, 0), (k + height, height), (0, 0, 255), thickness=1)
    return img


def get_sample_bgr(filename: str = "sample.jpg") -> np.ndarray:
    """data/ に画像があればそれを、無ければ合成シーンを返す（BGR uint8）。

    サンプル画像を用意していなくても全スクリプトがそのまま動くようにするための要。
    自分の画像で試したい場合は data/sample.jpg を置けば自動的にそちらが使われる。
    """
    candidate = DATA_DIR / filename
    img = imread_unicode(candidate, cv2.IMREAD_COLOR)
    if img is not None:
        return img
    return make_color_scene_bgr()
