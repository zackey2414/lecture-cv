"""第4回 演習の模範解答（全10問）。

このファイルは exercises.py の全演習に対する完成済みの解答です。
そのまま実行すると採点ランナーが回り、全問 PASS することを確認できます:

  uv run python lectures/04_filtering_edges_morphology/exercises_solutions.py

学習の流れとしては、まず exercises.py の TODO を自力で埋めてから、
答え合わせとしてこのファイルを読む/動かすのが効果的です。

採点ロジックとサンプル生成は exercises.py のものを再利用し（DRY）、
本ファイルの解答関数を差し込んで採点します。
"""

from __future__ import annotations

import os
import sys

import cv2
import numpy as np

# 同ディレクトリの exercises.py を import できるようにパスを通す
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exercises as ex  # noqa: E402


# =====================================================================
# 模範解答（各問の TODO を埋めたもの）
# =====================================================================

def ex1_denoise_saltpepper(noisy: np.ndarray) -> np.ndarray:
    """ごま塩ノイズは中央値フィルタが圧倒的に効く（外れ値を無視できる）。"""
    return cv2.medianBlur(noisy, 3)


def ex2_gradient_magnitude(gray: np.ndarray) -> np.ndarray:
    """負の勾配を潰さないよう CV_64F で計算し、magnitude→convertScaleAbs で 8bit 化。"""
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)  # 横方向の勾配（縦エッジ）
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)  # 縦方向の勾配（横エッジ）
    mag = cv2.magnitude(gx, gy)  # sqrt(gx^2 + gy^2)
    return cv2.convertScaleAbs(mag)  # |値| を 0..255 にスケール


def ex3_otsu_binarize(gray: np.ndarray) -> tuple[float, np.ndarray]:
    """OTSU 併用時はしきい値引数を 0 にすると自動決定。戻り値の第1要素が選ばれた値。"""
    t, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return (t, binary)


def ex4_remove_specks(binary: np.ndarray) -> np.ndarray:
    """オープニング（収縮→膨張）で背景の白い粒ノイズを除去する。"""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)


def ex5_count_objects(binary: np.ndarray) -> int:
    """OpenCV 4 系は (contours, hierarchy) の【2つ】返し。本数を数える。"""
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return len(contours)


def ex6_order_corners(pts: np.ndarray) -> np.ndarray:
    """和(x+y)と差(x-y)の符号で 4隅を一意に [TL,TR,BR,BL] へ並べ替える。"""
    pts = pts.reshape(4, 2).astype(np.float32)
    s = pts.sum(axis=1)  # x+y: 最小=左上, 最大=右下
    d = pts[:, 0] - pts[:, 1]  # x-y: 最大=右上, 最小=左下
    return np.float32([pts[np.argmin(s)], pts[np.argmax(d)], pts[np.argmax(s)], pts[np.argmin(d)]])


def ex7_adaptive_binarize(gray: np.ndarray) -> np.ndarray:
    """照明ムラには局所しきい値。THRESH_BINARY_INV で暗いインクを白(255)にする。"""
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, blockSize=21, C=8
    )


def ex8_canny_after_blur(gray: np.ndarray) -> np.ndarray:
    """「Canny の前にぼかす」でノイズ由来の偽エッジを抑える。"""
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    return cv2.Canny(blur, 50, 150)


def ex9_clahe_value_channel(bgr: np.ndarray) -> np.ndarray:
    """色相 H・彩度 S を保ち、明度 V だけ CLAHE すると色を壊さずコントラストが上がる。"""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return cv2.cvtColor(cv2.merge([h, s, clahe.apply(v)]), cv2.COLOR_HSV2BGR)


def ex10_warp_to_rect(img: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """四隅整列→出力サイズ算出→getPerspectiveTransform→warpPerspective の統合。"""
    quad = ex6_order_corners(pts)  # ex6 の並べ替えを再利用
    tl, tr, br, bl = quad
    width = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    height = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    dst = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
    m = cv2.getPerspectiveTransform(quad, dst)
    return cv2.warpPerspective(img, m, (width, height))


# =====================================================================
# 採点: exercises.py のランナーへ本ファイルの解答を差し込んで実行
# =====================================================================

def _install_into_exercises() -> None:
    """exercises モジュールの本番関数を、この解答群で上書きする。"""
    ex.ex1_denoise_saltpepper = ex1_denoise_saltpepper
    ex.ex2_gradient_magnitude = ex2_gradient_magnitude
    ex.ex3_otsu_binarize = ex3_otsu_binarize
    ex.ex4_remove_specks = ex4_remove_specks
    ex.ex5_count_objects = ex5_count_objects
    ex.ex6_order_corners = ex6_order_corners
    ex.ex7_adaptive_binarize = ex7_adaptive_binarize
    ex.ex8_canny_after_blur = ex8_canny_after_blur
    ex.ex9_clahe_value_channel = ex9_clahe_value_channel
    ex.ex10_warp_to_rect = ex10_warp_to_rect


if __name__ == "__main__":
    print("(模範解答 exercises_solutions.py で採点します)\n")
    _install_into_exercises()
    ex._grade()
