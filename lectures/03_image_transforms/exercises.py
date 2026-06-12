"""第3回 演習問題（色空間・描画・幾何変換）。

使い方:
  1. 各 exN_*() 関数の中の TODO を自分で実装する（最初は NotImplementedError で FAIL になる）。
  2. 実装できたら自己採点を実行:
         uv run python lectures/03_image_transforms/exercises.py
     全問 pass すれば "ALL PASS" と表示される。
  3. どうしても分からない時は、模範解答の挙動を見る:
         SHOW_SOLUTION=1 uv run python lectures/03_image_transforms/exercises.py
     模範解答そのものは exercises_solutions.py にもある（実行すると全 PASS）。
     （まずは自力で！）

難易度は易 → 難の順に並べてある（ex1〜ex4 が基礎、ex5〜ex7 が応用、ex8〜ex10 が総合）。
未実装の関数があっても自己採点は例外で落ちず、その問だけ FAIL 表示で最後まで走る（exit 0）。

ヒント:
  - サンプル画像は cv_helpers.make_color_scene_bgr() で得られる（BGR uint8, 300x400）。
  - OpenCV の HSV は H=0-179、S/V=0-255。cv2.resize の dsize は (幅W, 高さH) 順。
  - クロップは numpy スライス [y0:y1, x0:x1]（行=y, 列=x の順）。
  - グレースケールは shape が (H, W) で次元が1つ減る。赤は色相環の 0/179 をまたぐ。
"""

from __future__ import annotations

import os
import pathlib
import sys

import cv2
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from cv_helpers import make_color_scene_bgr  # noqa: E402
from preprocess import (  # noqa: E402（採点の参照実装に使う）
    center_crop_square,
    resize_keep_aspect,
    resize_to_square,
)


# =====================================================================
# 演習（ここを実装する）
# =====================================================================

def ex1_color_mask(
    bgr: np.ndarray,
    lower_hsv: tuple[int, int, int],
    upper_hsv: tuple[int, int, int],
) -> np.ndarray:
    """演習1【基礎】BGR画像を HSV に変換し、[lower_hsv, upper_hsv] の色域を抜く2値マスクを返す。

    返り値は (H, W) の uint8（0 か 255）。
    ヒント: cv2.cvtColor(..., cv2.COLOR_BGR2HSV) → cv2.inRange。
    """
    # TODO: BGR→HSV に変換し、cv2.inRange(hsv, lower, upper) を返す
    raise NotImplementedError


def ex2_resize_to(bgr: np.ndarray, width: int, height: int) -> np.ndarray:
    """演習2【基礎】画像を「幅 width × 高さ height」にリサイズして返す。

    最大の罠: cv2.resize の dsize は (幅, 高さ) 順！ shape の (高さ, 幅) とは逆。
    正しく実装できれば、返り値の shape は (height, width, 3) になる。
    """
    # TODO: cv2.resize(bgr, (width, height)) を返す（dsize は (W, H) 順）
    raise NotImplementedError


def ex3_crop(bgr: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
    """演習3【基礎】矩形 (x0, y0)-(x1, y1) の領域を numpy スライスで切り出して返す。

    スライスは [行(y), 列(x)] の順。img[x0:x1, y0:y1] ではない点に注意。
    """
    # TODO: bgr[y0:y1, x0:x1] を返す
    raise NotImplementedError


def ex4_flip(bgr: np.ndarray, mode: int) -> np.ndarray:
    """演習4【基礎】cv2.flip で画像を反転して返す。mode は 1=左右, 0=上下, -1=両方。"""
    # TODO: cv2.flip(bgr, mode) を返す
    raise NotImplementedError


def ex5_to_gray(bgr: np.ndarray) -> np.ndarray:
    """演習5【基礎】BGR をグレースケール化して返す。返り値の shape は (H, W) の2次元になる。

    「色を1つに落とすと次元が1つ減る」を体で覚える問題。
    ヒント: cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)。
    """
    # TODO: cv2.cvtColor で BGR→GRAY に変換して返す
    raise NotImplementedError


def ex6_red_mask_wraparound(
    bgr: np.ndarray,
    s_min: int = 80,
    v_min: int = 60,
) -> np.ndarray:
    """演習6【応用】色相環の 0/179 をまたぐ「赤」の2値マスクを作って返す。

    赤は H=0 付近と H=179 付近の両方にあるため、1区間では半分取りこぼす。
    手順:
      1. BGR→HSV に変換。
      2. [0, 10] と [170, 179] の2つの inRange マスクを作る（S/V の下限は s_min, v_min）。
      3. cv2.bitwise_or で2区間を合成して返す。
    返り値は (H, W) の uint8（0 か 255）。
    """
    # TODO: 2区間の inRange を bitwise_or で合成した赤マスクを返す
    raise NotImplementedError


def ex7_letterbox_square(bgr: np.ndarray, size: int) -> np.ndarray:
    """演習7【応用】アスペクト比を保ったまま size×size の正方形に収める（黒余白・中央配置）。

    歪ませない前処理。手順:
      1. 長辺が size になる倍率 scale = size / max(H, W) を求める。
      2. 縮小なら INTER_AREA で (new_w, new_h) にリサイズ（dsize は (W, H) 順！）。
      3. 黒(0,0,0)の size×size キャンバスを作り、中央に貼り付ける。
    返り値の shape は (size, size, 3)。
    """
    # TODO: 上の手順1〜3を実装して、size×size の正方形画像を返す
    raise NotImplementedError


def ex8_keep_aspect(bgr: np.ndarray, long_side: int) -> np.ndarray:
    """演習8【応用】長辺が long_side になるよう、アスペクト比を保ってリサイズして返す。

    画像を歪ませないリサイズ。手順:
      1. scale = long_side / max(H, W) を求める。
      2. new_w = round(W*scale), new_h = round(H*scale)（どちらも最低1）。
      3. 縮小は INTER_AREA、拡大は INTER_CUBIC を選んで cv2.resize（dsize は (W, H) 順）。
    返り値の長辺は long_side、アスペクト比は元画像と（丸め誤差の範囲で）一致する。
    """
    # TODO: アスペクト比保持リサイズを実装して返す
    raise NotImplementedError


def ex9_center_crop_square(bgr: np.ndarray) -> np.ndarray:
    """演習9【総合】画像中央から「内接する最大の正方形」を切り出して返す。

    余白を作らず正方形にしたいときの定番。手順:
      1. side = min(H, W)。
      2. top = (H - side)//2, left = (W - side)//2。
      3. スライス bgr[top:top+side, left:left+side] を返す（[y, x] の順）。
    返り値の shape は (side, side, 3)。
    """
    # TODO: 中央正方形クロップを実装して返す
    raise NotImplementedError


def ex10_bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """演習10【総合】2値マスク(0/255)から白画素の外接矩形 (x0, y0, x1, y1) を返す。

    色マスク → bbox は、色検出（章末ミニプロジェクト）の心臓部。軸順の総点検でもある。
    手順:
      1. np.where(mask == 255) で白画素の (ys, xs) を得る（返り値は y, x の順！）。
      2. 白画素が無ければ None を返す。
      3. (xs.min(), ys.min(), xs.max(), ys.max()) を int で返す。
    """
    # TODO: マスクから外接矩形 (x0, y0, x1, y1) を求めて返す（無ければ None）
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
        got = ex5_to_gray(bgr)
        ref = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        return (got is not None and got.shape == (300, 400) and np.array_equal(got, ref),
                "グレースケールは (H,W) で次元が1つ減る")
    check("ex5_to_gray", _check_ex5)

    def _check_ex6():
        got = ex6_red_mask_wraparound(bgr)
        red1 = cv2.inRange(cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV), (0, 80, 60), (10, 255, 255))
        red2 = cv2.inRange(cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV), (170, 80, 60), (179, 255, 255))
        ref = cv2.bitwise_or(red1, red2)
        return (got is not None and np.array_equal(got, ref) and int((got == 255).sum()) > 0,
                "赤は [0,10]∪[170,179] を bitwise_or")
    check("ex6_red_mask_wraparound", _check_ex6)

    def _check_ex7():
        got = ex7_letterbox_square(bgr, size=256)
        ref = resize_to_square(bgr, 256, (0, 0, 0))
        return (got is not None and got.shape == (256, 256, 3) and np.array_equal(got, ref),
                "正方形レターボックス")
    check("ex7_letterbox_square", _check_ex7)

    def _check_ex8():
        got = ex8_keep_aspect(bgr, long_side=256)
        ref = resize_keep_aspect(bgr, 256)
        # 長辺が 256・アスペクト比保持（縦横比の誤差が小さい）かを確認。
        ratio_ok = abs((got.shape[1] / got.shape[0]) - (bgr.shape[1] / bgr.shape[0])) < 0.05
        return (got is not None and max(got.shape[:2]) == 256 and ratio_ok
                and got.shape == ref.shape, "長辺256・比率保持")
    check("ex8_keep_aspect", _check_ex8)

    def _check_ex9():
        got = ex9_center_crop_square(bgr)
        ref = center_crop_square(bgr)
        return (got is not None and got.shape == (300, 300, 3) and np.array_equal(got, ref),
                "中央正方形クロップ side=min(H,W)")
    check("ex9_center_crop_square", _check_ex9)

    def _check_ex10():
        # 既知の矩形を白で塗ったマスクを作り、外接矩形がそれと一致するか確かめる。
        mask = np.zeros((300, 400), dtype=np.uint8)
        mask[40:121, 100:201] = 255   # y:40-120, x:100-200 を白
        got = ex10_bbox_from_mask(mask)
        empty = ex10_bbox_from_mask(np.zeros((10, 10), dtype=np.uint8))
        return (got == (100, 40, 200, 120) and empty is None,
                "マスク→外接矩形 (x0,y0,x1,y1)、空は None")
    check("ex10_bbox_from_mask", _check_ex10)

    print("=== 採点結果 ===")
    all_ok = True
    for name, ok, detail in results:
        mark = "PASS" if ok else "FAIL"
        all_ok = all_ok and ok
        print(f"  [{mark}] {name:24s} {detail}")
    n_pass = sum(1 for _, ok, _ in results if ok)
    print(f"\n  {n_pass}/{len(results)} 問 PASS")
    print("ALL PASS 🎉" if all_ok else "まだ未達の演習があります。TODO を埋めましょう。")


# =====================================================================
# 模範解答（SHOW_SOLUTION=1 のときに本体へ差し替えて実行）
# 完全な解答は exercises_solutions.py にもある。まずは自力で解いてから見ること。
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


def _sol_ex5(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def _sol_ex6(bgr, s_min=80, v_min=60):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    red1 = cv2.inRange(hsv, (0, s_min, v_min), (10, 255, 255))
    red2 = cv2.inRange(hsv, (170, s_min, v_min), (179, 255, 255))
    return cv2.bitwise_or(red1, red2)


def _sol_ex7(bgr, size):
    h, w = bgr.shape[:2]
    scale = size / max(h, w)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    fitted = cv2.resize(bgr, (new_w, new_h), interpolation=interp)
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    top, left = (size - new_h) // 2, (size - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = fitted
    return canvas


def _sol_ex8(bgr, long_side):
    h, w = bgr.shape[:2]
    scale = long_side / max(h, w)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    return cv2.resize(bgr, (new_w, new_h), interpolation=interp)


def _sol_ex9(bgr):
    h, w = bgr.shape[:2]
    side = min(h, w)
    top, left = (h - side) // 2, (w - side) // 2
    return bgr[top:top + side, left:left + side]


def _sol_ex10(mask):
    ys, xs = np.where(mask == 255)
    if ys.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _install_solutions() -> None:
    """模範解答で TODO 関数を差し替える（教材検証・答え合わせ用）。"""
    g = globals()
    g["ex1_color_mask"] = _sol_ex1
    g["ex2_resize_to"] = _sol_ex2
    g["ex3_crop"] = _sol_ex3
    g["ex4_flip"] = _sol_ex4
    g["ex5_to_gray"] = _sol_ex5
    g["ex6_red_mask_wraparound"] = _sol_ex6
    g["ex7_letterbox_square"] = _sol_ex7
    g["ex8_keep_aspect"] = _sol_ex8
    g["ex9_center_crop_square"] = _sol_ex9
    g["ex10_bbox_from_mask"] = _sol_ex10


if __name__ == "__main__":
    if os.environ.get("SHOW_SOLUTION") == "1":
        print("(模範解答モードで実行します)\n")
        _install_solutions()
    _grade()
