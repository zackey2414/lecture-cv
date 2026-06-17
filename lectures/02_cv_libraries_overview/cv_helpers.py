"""第2回モジュール共通ヘルパ（ライブラリの地図）。

この回は「概念回」です。新しい画像処理アルゴリズムを学ぶというより、
**画像・動画処理で使う主要ライブラリの役割分担と選び方を地図として掴む**のが目的です。
そのため、このヘルパには次の3つの責務だけを置きます。

  1. サンプル画像の用意（data/ にあればそれを読む。無ければ合成画像を作る）
  2. 出力先 lectures/02_cv_libraries_overview/outputs/ の場所を返す
  3. 任意ライブラリ（scikit-image / albumentations / torchvision / kornia / imageio / av）の
     「入っていれば使う・無ければ案内してスキップ」を一手に引き受ける probe()

なぜ probe() を用意するのか:
  本講座の実行コードは main 依存（cv2 / PIL / numpy / matplotlib）だけで必ず完走します。
  scikit-image などの「比較用ライブラリ」は未導入でもスクリプトが落ちてはいけません。
  そこで import を try/except で包み、結果を (モジュール, バージョン, 導入コマンド) として
  返すヘルパに集約します。各スクリプトは probe() の戻り値を見て分岐するだけで済みます。

注意（コーディング原則との関係）:
  道具箱とはいえ「賢すぎる抽象化」は避け、各関数は1つのことだけを行います。
  学習上重要な処理（cvtColor やリサイズの書き比べ）は、あえて各スクリプト本体に
  生のまま書いて「読んで学べる」ようにしています。
"""

from __future__ import annotations

import importlib
import pathlib
from dataclasses import dataclass

import cv2
import numpy as np

# --- パス定義 ------------------------------------------------------------
# このファイルは lectures/02_cv_libraries_overview/ にある。
#   parents[0] = 02_cv_libraries_overview
#   parents[1] = lectures
#   parents[2] = プロジェクトルート（lecture-cv/）
# __file__ を基準にするので、どのディレクトリから実行しても場所がぶれない。
_HERE = pathlib.Path(__file__).resolve()
PROJECT_ROOT = _HERE.parents[2]
DATA_DIR = PROJECT_ROOT / "data"
MODULE_ID = "02_cv_libraries_overview"


def output_dir() -> pathlib.Path:
    """このモジュール専用の出力ディレクトリを作って返す。

    headless 環境では「画面に出す」代わりに「ファイルに保存して後で見る」のが基本。
    その保存先をここで一元管理する。存在しなければ自動で作成する。
    """
    d = _HERE.parent / "outputs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def make_sample_bgr(height: int = 240, width: int = 320) -> np.ndarray:
    """学習用の合成画像を BGR uint8 で作る（ネット非依存・サンプル不要のための要）。

    この回では「同じ処理（リサイズ・ぼかし・回転）を各ライブラリで書き比べる」ので、
    補間や平滑化の差が\"目で見て\"分かる素材を盛り込む:
      - 細かい市松模様（高周波）… 縮小時のエイリアシング/補間差が出やすい
      - 斜めの直線         … 回転やアンチエイリアスの差が出やすい
      - 原色のブロック     … BGR/RGB 取り違えが一目で分かる
      - 横方向グラデーション… 連続階調での平滑化の効き方が分かる
      - 白い円             … 形状のなめらかさ比較用

    OpenCV の画像は (高さ H, 幅 W, チャンネル3) の numpy 配列で、チャンネル順は B, G, R。
    """
    img = np.zeros((height, width, 3), dtype=np.uint8)

    # (1) 背景に横方向グラデーションを敷く（x が増えるほど明るいグレー）。
    band = np.linspace(30, 220, width, dtype=np.uint8)      # (W,)
    img[:, :, :] = band[None, :, None]                      # 3ch にブロードキャスト

    # (2) 左上に細かい市松模様（8px 角）を置く。高周波なので縮小で差が出る。
    cell = 8
    yy, xx = np.mgrid[0:height // 2, 0:width // 2]
    checmark = (((yy // cell) + (xx // cell)) % 2).astype(np.uint8) * 255
    img[0:height // 2, 0:width // 2] = checmark[:, :, None]

    # (3) 右上に原色ブロック（赤・緑・青）。BGR 並びを体感する用。
    hw = width // 2
    third = (width - hw) // 3
    img[0:60, hw:hw + third] = (0, 0, 255)             # 赤 (B=0,G=0,R=255)
    img[0:60, hw + third:hw + 2 * third] = (0, 255, 0)  # 緑
    img[0:60, hw + 2 * third:width] = (255, 0, 0)       # 青

    # (4) 斜めの白線を数本（回転・アンチエイリアスの比較用）。
    for offset in range(-height, width, 40):
        cv2.line(img, (offset, 0), (offset + height, height), (255, 255, 255), 1, cv2.LINE_AA)

    # (5) 中央やや下に白い塗りつぶし円。cv2.circle は (x, y) 中心・BGR 色。
    cv2.circle(img, (width // 2, height * 3 // 4), 40, (255, 255, 255), thickness=-1)

    return img


def get_sample_bgr(filename: str = "sample.jpg") -> np.ndarray:
    """data/ に画像があればそれを、無ければ合成画像を返す（BGR uint8）。

    これにより「サンプル画像を用意していない人」でも全スクリプトがそのまま動く。
    自分の画像で試したい場合は data/sample.jpg を置けば自動的にそちらが使われる。
    cv2.imread は失敗時に None を返す仕様なので、None なら合成画像にフォールバックする。
    """
    candidate = DATA_DIR / filename
    if candidate.exists():
        # np.fromfile + imdecode は日本語パスにも強い読み込みの定石。
        buf = np.fromfile(str(candidate), dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is not None:
            return img
    return make_sample_bgr()


@dataclass
class Probe:
    """任意ライブラリの導入状況をまとめた小さなレコード。

    available が False のとき module は None。hint には「どう入れるか」を入れておき、
    スクリプト側がそのまま案内文として print できるようにする。
    """

    name: str            # 表示名（例: "scikit-image"）
    import_name: str     # import 名（例: "skimage"）
    available: bool
    version: str | None
    module: object | None
    hint: str            # 未導入時の導入コマンド案内


def probe(name: str, import_name: str, hint: str) -> Probe:
    """ライブラリを import してみて、結果を Probe にして返す（落とさない）。

    本講座の実行コードは main 依存だけで完走するのが鉄則。比較用の任意ライブラリは
    「入っていれば使う、無ければ案内してスキップ」を徹底する。その判定をここに集約する。
    """
    try:
        mod = importlib.import_module(import_name)
    except Exception:  # noqa: BLE001  ImportError 以外（依存の連鎖失敗など）も握る
        return Probe(name, import_name, False, None, None, hint)
    version = getattr(mod, "__version__", "?")
    return Probe(name, import_name, True, version, mod, hint)


def to_rgb(bgr: np.ndarray) -> np.ndarray:
    """BGR(OpenCV) → RGB(PIL/matplotlib/学習系) への変換を一語で。境界をまたぐ前に必ず通す。"""
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


if __name__ == "__main__":
    # 直接実行したときの自己テスト（exit 0 で終わる）。
    out = output_dir()
    img = get_sample_bgr()
    print("[cv_helpers] サンプル画像 shape:", img.shape, "dtype:", img.dtype)
    print("[cv_helpers] 出力先:", out)
    for p in (
        probe("scikit-image", "skimage", "uv add --group aug scikit-image"),
        probe("albumentations", "albumentations", "uv add --group aug albumentations"),
    ):
        state = f"あり (v{p.version})" if p.available else "なし"
        print(f"[cv_helpers] probe {p.name:14s}: {state}")
    print("[cv_helpers] OK")
