"""第4回 演習問題（フィルタ・エッジ・閾値・モルフォロジー・輪郭・ワーピング）。

使い方:
  1. 各 exN_*() 関数の中の TODO を自分で実装する（最初は未実装で FAIL になる）。
  2. 実装できたら自己採点を実行:
         uv run python lectures/04_filtering_edges_morphology/exercises.py
     全問 pass すれば "ALL PASS" と表示される。
  3. どうしても分からない時は、模範解答の挙動を見る:
         SHOW_SOLUTION=1 uv run python lectures/04_filtering_edges_morphology/exercises.py
     （まずは自力で！）

サンプル画像は各演習へ自動で渡される（このファイル内で合成生成。外部依存なし）。
"""

from __future__ import annotations

import os

import cv2
import numpy as np


# =====================================================================
# 演習（ここを実装する）
# =====================================================================

def ex1_denoise_saltpepper(noisy: np.ndarray) -> np.ndarray:
    """演習1: ごま塩ノイズ画像を、最も適したフィルタで除去する。

    白黒の突発的な粒（salt&pepper）には平均・ガウスより「中央値」が効く。
    ksize=3 の medianBlur を使って結果（uint8 グレー画像）を返すこと。
    """
    # TODO: cv2.medianBlur(noisy, 3) を返す
    raise NotImplementedError


def ex2_gradient_magnitude(gray: np.ndarray) -> np.ndarray:
    """演習2: Sobel で勾配の大きさ画像（uint8）を作る。

    手順: ddepth=cv2.CV_64F で x 方向・y 方向の Sobel(ksize=3) を計算し、
    cv2.magnitude で大きさを取り、cv2.convertScaleAbs で 0..255 の uint8 にして返す。
    （CV_8U で計算すると負の勾配が潰れるので必ず CV_64F で）
    """
    # TODO: gx=Sobel(...CV_64F,1,0,3), gy=Sobel(...CV_64F,0,1,3),
    #       mag=cv2.magnitude(gx,gy) を convertScaleAbs して返す
    raise NotImplementedError


def ex3_otsu_binarize(gray: np.ndarray) -> tuple[float, np.ndarray]:
    """演習3: 大津の手法でしきい値を自動決定して二値化する。

    cv2.threshold に THRESH_BINARY + THRESH_OTSU を渡し、(選ばれたしきい値, 二値画像)
    のタプルを返すこと。OTSU を使うときの第2引数（しきい値）は 0 でよい。
    """
    # TODO: t, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #       return (t, binary)
    raise NotImplementedError


def ex4_remove_specks(binary: np.ndarray) -> np.ndarray:
    """演習4: 二値画像から白い粒ノイズをモルフォロジーのオープニングで除去する。

    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)) を構造要素に、
    cv2.morphologyEx(..., cv2.MORPH_OPEN, ...) を適用した結果を返すこと。
    """
    # TODO: kernel を作り MORPH_OPEN を適用して返す
    raise NotImplementedError


def ex5_count_objects(binary: np.ndarray) -> int:
    """演習5: 二値画像中の外側輪郭の数を数える。

    cv2.findContours は OpenCV 4 系で返り値が【2つ】(contours, hierarchy)。
    RETR_EXTERNAL / CHAIN_APPROX_SIMPLE で輪郭を取り、その本数(int)を返すこと。
    """
    # TODO: contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #       return len(contours)
    raise NotImplementedError


def ex6_order_corners(pts: np.ndarray) -> np.ndarray:
    """演習6: バラバラな4点を [左上, 右上, 右下, 左下] の順に並べ替える。

    透視変換の前処理で必須。座標の和(x+y)と差(x-y)を使う:
      左上=和が最小 / 右下=和が最大 / 右上=(x-y)が最大 / 左下=(x-y)が最小。
    返り値は shape (4, 2) の float32 配列。
    """
    # TODO: pts を (4,2) float32 にし、和と差で4隅を選んで並べて返す
    raise NotImplementedError


# =====================================================================
# サンプル生成（採点用。乱数シード固定で毎回同じ入力）
# =====================================================================

def _sample_gray() -> np.ndarray:
    img = np.full((120, 160), 90, dtype=np.uint8)
    cv2.rectangle(img, (20, 20), (80, 95), 220, -1)
    cv2.circle(img, (120, 60), 30, 30, -1)
    return img


def _sample_saltpepper() -> np.ndarray:
    clean = _sample_gray()
    out = clean.copy()
    m = np.random.default_rng(1).random(clean.shape)
    out[m < 0.03] = 0
    out[m > 0.97] = 255
    return out


def _sample_two_shapes() -> np.ndarray:
    img = np.zeros((120, 160), dtype=np.uint8)
    cv2.circle(img, (45, 60), 26, 255, -1)
    cv2.rectangle(img, (100, 40), (140, 90), 255, -1)
    return img


def _sample_noisy_binary() -> np.ndarray:
    img = _sample_two_shapes()
    sp = np.random.default_rng(7).random(img.shape)
    img[(img == 0) & (sp < 0.008)] = 255  # 背景に白い粒
    return img


def _sample_corners() -> np.ndarray:
    # 並べ替え前のバラバラな4点（TL,TR,BR,BL の答えが一意になるよう離して配置）
    return np.array([[210, 300], [60, 55], [255, 80], [80, 330]], dtype=np.float32)


# =====================================================================
# 模範解答
# =====================================================================

def _sol_ex1(noisy):
    return cv2.medianBlur(noisy, 3)


def _sol_ex2(gray):
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    return cv2.convertScaleAbs(cv2.magnitude(gx, gy))


def _sol_ex3(gray):
    t, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return (t, binary)


def _sol_ex4(binary):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)


def _sol_ex5(binary):
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return len(contours)


def _sol_ex6(pts):
    pts = pts.reshape(4, 2).astype(np.float32)
    s = pts.sum(axis=1)
    d = pts[:, 0] - pts[:, 1]
    return np.float32([pts[np.argmin(s)], pts[np.argmax(d)], pts[np.argmax(s)], pts[np.argmin(d)]])


def _install_solutions() -> None:
    g = globals()
    g["ex1_denoise_saltpepper"] = _sol_ex1
    g["ex2_gradient_magnitude"] = _sol_ex2
    g["ex3_otsu_binarize"] = _sol_ex3
    g["ex4_remove_specks"] = _sol_ex4
    g["ex5_count_objects"] = _sol_ex5
    g["ex6_order_corners"] = _sol_ex6


# =====================================================================
# 自己採点ランナー
# =====================================================================

def _grade() -> None:
    gray = _sample_gray()
    sp = _sample_saltpepper()
    clean_bin = _sample_two_shapes()
    noisy_bin = _sample_noisy_binary()
    corners = _sample_corners()
    results: list[tuple[str, bool, str]] = []

    def check(name: str, fn) -> None:
        try:
            ok, detail = fn()
            results.append((name, bool(ok), detail))
        except NotImplementedError:
            results.append((name, False, "未実装（TODO を埋めてください）"))
        except Exception as e:  # noqa: BLE001
            results.append((name, False, f"例外: {type(e).__name__}: {e}"))

    check("ex1_denoise_saltpepper", lambda: (
        np.array_equal(ex1_denoise_saltpepper(sp), _sol_ex1(sp)),
        "medianBlur でごま塩除去",
    ))
    check("ex2_gradient_magnitude", lambda: (
        np.array_equal(ex2_gradient_magnitude(gray), _sol_ex2(gray)),
        "Sobel→magnitude→uint8",
    ))

    def _check_ex3():
        t, binary = ex3_otsu_binarize(gray)
        et, eb = _sol_ex3(gray)
        return (abs(float(t) - float(et)) < 1e-6 and np.array_equal(binary, eb), "Otsu 自動二値化")
    check("ex3_otsu_binarize", _check_ex3)

    check("ex4_remove_specks", lambda: (
        np.array_equal(ex4_remove_specks(noisy_bin), _sol_ex4(noisy_bin)),
        "opening で粒除去",
    ))
    check("ex5_count_objects", lambda: (
        ex5_count_objects(clean_bin) == _sol_ex5(clean_bin) == 2,
        "findContours は2返し / 物体数=2",
    ))
    check("ex6_order_corners", lambda: (
        np.array_equal(np.asarray(ex6_order_corners(corners), dtype=np.float32), _sol_ex6(corners)),
        "4隅を TL,TR,BR,BL に整列",
    ))

    print("=== 採点結果 ===")
    all_ok = True
    for name, ok, detail in results:
        mark = "PASS" if ok else "FAIL"
        all_ok = all_ok and ok
        print(f"  [{mark}] {name:24s} {detail}")
    print("\nALL PASS 🎉" if all_ok else "\nまだ未達の演習があります。TODO を埋めましょう。")


if __name__ == "__main__":
    if os.environ.get("SHOW_SOLUTION") == "1":
        print("(模範解答モードで実行します)\n")
        _install_solutions()
    _grade()
