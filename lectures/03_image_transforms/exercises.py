"""第3回 演習問題（色空間・描画・幾何変換）。

使い方:
  1. 各 exN_*() 関数の中の TODO を自分で実装する（最初は NotImplementedError で FAIL になる）。
  2. 実装できたら自己採点を実行:
         uv run python lectures/03_image_transforms/exercises.py
     全問 pass すれば "ALL PASS" と表示される。
  3. どうしても分からない時は、模範解答の挙動を見る:
         SHOW_SOLUTION=1 uv run python lectures/03_image_transforms/exercises.py
     （まずは自力で！）

ヒント:
  - サンプル画像は cv_helpers.make_color_scene_bgr() で得られる（BGR uint8, 300x400）。
  - OpenCV の HSV は H=0-179、S/V=0-255。cv2.resize の dsize は (幅W, 高さH) 順。
  - クロップは numpy スライス [y0:y1, x0:x1]（行=y, 列=x の順）。
"""

from __future__ import annotations

import os
import pathlib
import sys

import cv2
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from cv_helpers import make_color_scene_bgr  # noqa: E402
from preprocess import resize_to_square  # noqa: E402（採点の参照実装に使う）


# =====================================================================
# 演習（ここを実装する）
# =====================================================================

def ex1_color_mask(
    bgr: np.ndarray,
    lower_hsv: tuple[int, int, int],
    upper_hsv: tuple[int, int, int],
) -> np.ndarray:
    """演習1: BGR画像を HSV に変換し、[lower_hsv, upper_hsv] の色域を抜く2値マスクを返す。

    返り値は (H, W) の uint8（0 か 255）。
    ヒント: cv2.cvtColor(..., cv2.COLOR_BGR2HSV) → cv2.inRange。
    """
    # TODO: BGR→HSV に変換し、cv2.inRange(hsv, lower, upper) を返す
    raise NotImplementedError


def ex2_resize_to(bgr: np.ndarray, width: int, height: int) -> np.ndarray:
    """演習2: 画像を「幅 width × 高さ height」にリサイズして返す。

    最大の罠: cv2.resize の dsize は (幅, 高さ) 順！ shape の (高さ, 幅) とは逆。
    正しく実装できれば、返り値の shape は (height, width, 3) になる。
    """
    # TODO: cv2.resize(bgr, (width, height)) を返す（dsize は (W, H) 順）
    raise NotImplementedError


def ex3_crop(bgr: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
    """演習3: 矩形 (x0, y0)-(x1, y1) の領域を numpy スライスで切り出して返す。

    スライスは [行(y), 列(x)] の順。img[x0:x1, y0:y1] ではない点に注意。
    """
    # TODO: bgr[y0:y1, x0:x1] を返す
    raise NotImplementedError


def ex4_flip(bgr: np.ndarray, mode: int) -> np.ndarray:
    """演習4: cv2.flip で画像を反転して返す。mode は 1=左右, 0=上下, -1=両方。"""
    # TODO: cv2.flip(bgr, mode) を返す
    raise NotImplementedError


def ex5_letterbox_square(bgr: np.ndarray, size: int) -> np.ndarray:
    """演習5: アスペクト比を保ったまま size×size の正方形に収める（黒余白・中央配置）。

    歪ませない前処理。手順:
      1. 長辺が size になる倍率 scale = size / max(H, W) を求める。
      2. 縮小なら INTER_AREA で (new_w, new_h) にリサイズ（dsize は (W, H) 順！）。
      3. 黒(0,0,0)の size×size キャンバスを作り、中央に貼り付ける。
    返り値の shape は (size, size, 3)。
    """
    # TODO: 上の手順1〜3を実装して、size×size の正方形画像を返す
    raise NotImplementedError


# =====================================================================
# 自己採点ランナー
# =====================================================================

def _grade() -> None:
    bgr = make_color_scene_bgr()  # (300, 400, 3)
    results: list[tuple[str, bool, str]] = []

    def check(name: str, fn) -> None:
        try:
            ok, detail = fn()
            results.append((name, ok, detail))
        except NotImplementedError:
            results.append((name, False, "未実装（TODOを埋めてください）"))
        except Exception as e:  # noqa: BLE001
            results.append((name, False, f"例外: {type(e).__name__}: {e}"))

    def _check_ex1():
        lower, upper = (100, 80, 60), (130, 255, 255)  # 青
        got = ex1_color_mask(bgr, lower, upper)
        ref = cv2.inRange(cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV),
                          np.array(lower, np.uint8), np.array(upper, np.uint8))
        return (got is not None and np.array_equal(got, ref), "HSV inRange マスク")
    check("ex1_color_mask", _check_ex1)

    def _check_ex2():
        got = ex2_resize_to(bgr, width=200, height=100)
        return (got is not None and got.shape == (100, 200, 3),
                "dsize=(W,H) で shape=(H,W,3)")
    check("ex2_resize_to", _check_ex2)

    def _check_ex3():
        got = ex3_crop(bgr, x0=100, y0=50, x1=300, y1=200)
        return (got is not None and np.array_equal(got, bgr[50:200, 100:300]),
                "スライス [y0:y1, x0:x1]")
    check("ex3_crop", _check_ex3)

    def _check_ex4():
        ok = all(
            ex4_flip(bgr, m) is not None and np.array_equal(ex4_flip(bgr, m), cv2.flip(bgr, m))
            for m in (1, 0, -1)
        )
        return (ok, "左右/上下/両方の反転")
    check("ex4_flip", _check_ex4)

    def _check_ex5():
        got = ex5_letterbox_square(bgr, size=256)
        ref = resize_to_square(bgr, 256, (0, 0, 0))
        return (got is not None and got.shape == (256, 256, 3) and np.array_equal(got, ref),
                "正方形レターボックス")
    check("ex5_letterbox_square", _check_ex5)

    print("=== 採点結果 ===")
    all_ok = True
    for name, ok, detail in results:
        mark = "PASS" if ok else "FAIL"
        all_ok = all_ok and ok
        print(f"  [{mark}] {name:20s} {detail}")
    print("\nALL PASS 🎉" if all_ok else "\nまだ未達の演習があります。TODO を埋めましょう。")


# =====================================================================
# 模範解答（SHOW_SOLUTION=1 のときに本体へ差し替えて実行）
# まずは自力で解いてから見ること。
# =====================================================================

def _sol_ex1(bgr, lower_hsv, upper_hsv):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, np.array(lower_hsv, np.uint8), np.array(upper_hsv, np.uint8))


def _sol_ex2(bgr, width, height):
    return cv2.resize(bgr, (width, height))  # dsize は (W, H) 順


def _sol_ex3(bgr, x0, y0, x1, y1):
    return bgr[y0:y1, x0:x1]


def _sol_ex4(bgr, mode):
    return cv2.flip(bgr, mode)


def _sol_ex5(bgr, size):
    h, w = bgr.shape[:2]
    scale = size / max(h, w)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    fitted = cv2.resize(bgr, (new_w, new_h), interpolation=interp)
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    top, left = (size - new_h) // 2, (size - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = fitted
    return canvas


def _install_solutions() -> None:
    """模範解答で TODO 関数を差し替える（教材検証・答え合わせ用）。"""
    g = globals()
    g["ex1_color_mask"] = _sol_ex1
    g["ex2_resize_to"] = _sol_ex2
    g["ex3_crop"] = _sol_ex3
    g["ex4_flip"] = _sol_ex4
    g["ex5_letterbox_square"] = _sol_ex5


if __name__ == "__main__":
    if os.environ.get("SHOW_SOLUTION") == "1":
        print("(模範解答モードで実行します)\n")
        _install_solutions()
    _grade()
