"""第3回の「完成物」: 再利用できる前処理関数群（読んで学べる実装）。

このモジュールが、本章の deliverable（成果物）です。各スクリプト 01〜04 で学ぶ要素を、
そのまま実務で使える関数の形にまとめてあります。中身は OpenCV / Pillow / numpy だけ。

含まれるもの:
  - HSV 色域でのオブジェクト抽出マスク生成器（extract_color / hsv_mask）
  - アスペクト比保持リサイズ（resize_keep_aspect）
  - 正方形強制リサイズ＝レターボックス（resize_to_square）
  - 中央正方形クロップ（center_crop_square）
  - EXIF Orientation を正規化して BGR で読む（load_image_oriented）

設計方針（コーディング原則）:
  各関数は1つのことだけを行い、早期 return で分岐を浅く保つ。
  「賢い」抽象化は避け、初学者が上から読めば処理を追えることを優先する。
"""

from __future__ import annotations

import pathlib

import cv2
import numpy as np
from PIL import Image, ImageOps

# ---------------------------------------------------------------------------
# HSV 色域でのオブジェクト抽出マスク
# ---------------------------------------------------------------------------
# OpenCV の HSV は H=0-179（0-360 ではない）、S/V=0-255。
# 純色のおおよその色相(H)中心は: 赤=0(179もまたぐ), 黄=30, 緑=60, 青=120。
# 各色について (H下限, H上限) を、彩度(S)・明度(V)の下限とともに定義する。
# 赤だけは 0 と 179 の両端にまたがるため、2区間を持たせて bitwise_or で合成する。
_COLOR_HUE_RANGES: dict[str, list[tuple[int, int]]] = {
    "red": [(0, 10), (170, 179)],   # 0/179 をまたぐので2区間
    "yellow": [(20, 35)],
    "green": [(40, 80)],
    "blue": [(100, 130)],
}


def hsv_mask(
    bgr: np.ndarray,
    lower: tuple[int, int, int],
    upper: tuple[int, int, int],
) -> np.ndarray:
    """BGR画像を HSV に変換し、[lower, upper] の範囲を 255、外を 0 とする2値マスクを返す。

    返り値は (H, W) の uint8（0 か 255）。cv2.inRange の薄いラッパだが、
    BGR→HSV 変換を内側に隠すことで「呼ぶ側は色順を気にしなくてよい」状態にしている。
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, np.array(lower, np.uint8), np.array(upper, np.uint8))


def extract_color(
    bgr: np.ndarray,
    color: str,
    s_min: int = 80,
    v_min: int = 60,
    clean: bool = True,
) -> np.ndarray:
    """プリセット色名（red/yellow/green/blue）でオブジェクト抽出マスクを作る。

    彩度(S)・明度(V)の下限を設けることで、背景のグレー（低彩度）を除外する。
    赤は色相環の両端にまたがるため、複数区間のマスクを bitwise_or で合成する。
    clean=True なら軽いモルフォロジー開閉でノイズと小穴を整える（詳細は第4回）。
    """
    if color not in _COLOR_HUE_RANGES:
        raise ValueError(f"未知の色名です: {color}（{list(_COLOR_HUE_RANGES)} から選ぶ）")

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = np.zeros(bgr.shape[:2], dtype=np.uint8)
    for h_lo, h_hi in _COLOR_HUE_RANGES[color]:
        part = cv2.inRange(hsv, (h_lo, s_min, v_min), (h_hi, 255, 255))
        mask = cv2.bitwise_or(mask, part)

    if clean:
        # 3x3 の開閉で、ぽつぽつノイズ除去と小さな穴埋めをする（モルフォロジーは第4回で詳解）。
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def apply_mask(bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """マスク(255の部分)だけ残し、それ以外を黒にした画像を返す。"""
    return cv2.bitwise_and(bgr, bgr, mask=mask)


# ---------------------------------------------------------------------------
# リサイズ（アスペクト比保持 / 正方形強制）
# ---------------------------------------------------------------------------
def _interp_for(scale: float) -> int:
    """拡大か縮小かで補間方法を選ぶ。縮小は INTER_AREA、拡大は INTER_CUBIC が定石。"""
    return cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC


def resize_keep_aspect(bgr: np.ndarray, long_side: int) -> np.ndarray:
    """長辺が long_side になるよう、アスペクト比を保ってリサイズする。

    cv2.resize の dsize は (幅W, 高さH) 順で、numpy の shape=(H, W) とは逆。
    ここを取り違えると縦横が入れ替わるので、計算した (new_w, new_h) を必ずこの順で渡す。
    """
    h, w = bgr.shape[:2]
    scale = long_side / max(h, w)
    new_w = max(1, round(w * scale))
    new_h = max(1, round(h * scale))
    # dsize は (W, H) 順！ shape は (H, W) なので毎回ここで意識的に並べ替える。
    return cv2.resize(bgr, (new_w, new_h), interpolation=_interp_for(scale))


def resize_to_square(
    bgr: np.ndarray,
    size: int,
    pad_color: tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    """アスペクト比を保ったまま size×size の正方形に収める（レターボックス）。

    画像を歪ませずに正方形入力へ揃える定番の前処理。検出・分類モデルの入力作りで多用する。
    まず長辺が size になるよう縮小し、足りない余白を pad_color で埋めて中央に置く。
    """
    fitted = resize_keep_aspect(bgr, size)
    fh, fw = fitted.shape[:2]
    # size×size のキャンバスを pad_color で塗り、中央に貼り付ける。
    canvas = np.full((size, size, 3), pad_color, dtype=np.uint8)
    top = (size - fh) // 2
    left = (size - fw) // 2
    canvas[top:top + fh, left:left + fw] = fitted
    return canvas


def center_crop_square(bgr: np.ndarray) -> np.ndarray:
    """画像中央から「内接する最大の正方形」を切り出す（numpy スライス）。

    余白を作らず正方形にしたいときに使う。端の情報は捨てられる点がレターボックスと違う。
    """
    h, w = bgr.shape[:2]
    side = min(h, w)
    top = (h - side) // 2
    left = (w - side) // 2
    # スライスは [行(y), 列(x)]。ここでも (y, x) の順を崩さない。
    return bgr[top:top + side, left:left + side]


# ---------------------------------------------------------------------------
# EXIF Orientation を正規化して読む
# ---------------------------------------------------------------------------
def load_image_oriented(path: str | pathlib.Path) -> np.ndarray | None:
    """画像を読み、EXIF Orientation を反映した「正しい向き」の BGR 配列で返す。

    スマホ写真は画素を回さず EXIF に向きだけ記録することが多い。素朴に読むと横倒しになる。
    Pillow の ImageOps.exif_transpose で画素を実際に回し、向きを正規化してから BGR にする。
    読めなければ None を返す（呼び出し側で None チェックする前提）。
    """
    p = pathlib.Path(path)
    if not p.exists():
        return None
    with Image.open(p) as im:
        # exif_transpose: EXIF の向き情報に従って画素を回転/反転し、向きタグを消す。
        fixed = ImageOps.exif_transpose(im)
        rgb = fixed.convert("RGB")          # アルファ等を落として3chに揃える
        arr = np.asarray(rgb)               # PIL(RGB) → numpy
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)  # RGB → OpenCV の BGR へ
