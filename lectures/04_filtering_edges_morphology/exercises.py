"""第4回 演習問題（フィルタ・エッジ・閾値・モルフォロジー・輪郭・ワーピング）。

全10問。易→難で並んでいます（ex1〜ex6 が基礎、ex7〜ex10 が統合・応用）。

使い方:
  1. 各 exN_*() 関数の中の TODO を自分で実装する（最初は未実装で FAIL になる）。
  2. 実装できたら自己採点を実行:
         uv run python lectures/04_filtering_edges_morphology/exercises.py
     全問 pass すれば "ALL PASS" と表示される。
  3. どうしても分からない時は、模範解答の挙動を見る:
         SHOW_SOLUTION=1 uv run python lectures/04_filtering_edges_morphology/exercises.py
     模範解答の全文は exercises_solutions.py にもあります（まずは自力で！）。

サンプル画像は各演習へ自動で渡される（このファイル内で合成生成。外部依存なし）。
未実装でも例外で落とさず、採点結果を表示して正常終了（exit 0）する。
"""

from __future__ import annotations

import os

import cv2
import numpy as np


# =====================================================================
# 演習（ここを実装する）
# =====================================================================

def ex1_denoise_saltpepper(noisy: np.ndarray) -> np.ndarray:
    """演習1[易]: ごま塩ノイズ画像を、最も適したフィルタで除去する。

    白黒の突発的な粒（salt&pepper）には平均・ガウスより「中央値」が効く。
    ksize=3 の medianBlur を使って結果（uint8 グレー画像）を返すこと。
    """
    # TODO: cv2.medianBlur(noisy, 3) を返す
    raise NotImplementedError


def ex2_gradient_magnitude(gray: np.ndarray) -> np.ndarray:
    """演習2[易]: Sobel で勾配の大きさ画像（uint8）を作る。

    手順: ddepth=cv2.CV_64F で x 方向・y 方向の Sobel(ksize=3) を計算し、
    cv2.magnitude で大きさを取り、cv2.convertScaleAbs で 0..255 の uint8 にして返す。
    （CV_8U で計算すると負の勾配が潰れるので必ず CV_64F で）
    """
    # TODO: gx=Sobel(...CV_64F,1,0,3), gy=Sobel(...CV_64F,0,1,3),
    #       mag=cv2.magnitude(gx,gy) を convertScaleAbs して返す
    raise NotImplementedError


def ex3_otsu_binarize(gray: np.ndarray) -> tuple[float, np.ndarray]:
    """演習3[易]: 大津の手法でしきい値を自動決定して二値化する。

    cv2.threshold に THRESH_BINARY + THRESH_OTSU を渡し、(選ばれたしきい値, 二値画像)
    のタプルを返すこと。OTSU を使うときの第2引数（しきい値）は 0 でよい。
    """
    # TODO: t, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #       return (t, binary)
    raise NotImplementedError


def ex4_remove_specks(binary: np.ndarray) -> np.ndarray:
    """演習4[中]: 二値画像から白い粒ノイズをモルフォロジーのオープニングで除去する。

    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)) を構造要素に、
    cv2.morphologyEx(..., cv2.MORPH_OPEN, ...) を適用した結果を返すこと。
    """
    # TODO: kernel を作り MORPH_OPEN を適用して返す
    raise NotImplementedError


def ex5_count_objects(binary: np.ndarray) -> int:
    """演習5[中]: 二値画像中の外側輪郭の数を数える。

    cv2.findContours は OpenCV 4 系で返り値が【2つ】(contours, hierarchy)。
    RETR_EXTERNAL / CHAIN_APPROX_SIMPLE で輪郭を取り、その本数(int)を返すこと。
    """
    # TODO: contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #       return len(contours)
    raise NotImplementedError


def ex6_order_corners(pts: np.ndarray) -> np.ndarray:
    """演習6[中]: バラバラな4点を [左上, 右上, 右下, 左下] の順に並べ替える。

    透視変換の前処理で必須。座標の和(x+y)と差(x-y)を使う:
      左上=和が最小 / 右下=和が最大 / 右上=(x-y)が最大 / 左下=(x-y)が最小。
    返り値は shape (4, 2) の float32 配列。
    """
    # TODO: pts を (4,2) float32 にし、和と差で4隅を選んで並べて返す
    raise NotImplementedError


def ex7_adaptive_binarize(gray: np.ndarray) -> np.ndarray:
    """演習7[中]: 照明ムラのある文書を適応的閾値で二値化する。

    固定/Otsu の大域的しきい値は照明ムラで破綻する。adaptiveThreshold で局所的に判断する。
    cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                          cv2.THRESH_BINARY_INV, blockSize=21, C=8) を返すこと。
    （blockSize は奇数。THRESH_BINARY_INV で暗いインクを白=255 にする）
    """
    # TODO: cv2.adaptiveThreshold(...) を返す
    raise NotImplementedError


def ex8_canny_after_blur(gray: np.ndarray) -> np.ndarray:
    """演習8[中]: 前段ぼかし付きの Canny でノイズ由来の偽エッジを抑える。

    手順: cv2.GaussianBlur(gray, (5, 5), 0) でぼかしてから cv2.Canny(blur, 50, 150)。
    「Canny の前にぼかす」は実務の鉄則。エッジ画像(uint8, 0/255)を返すこと。
    """
    # TODO: blur=GaussianBlur((5,5),0) → Canny(50,150) を返す
    raise NotImplementedError


def ex9_clahe_value_channel(bgr: np.ndarray) -> np.ndarray:
    """演習9[難]: カラー画像の色を壊さずにコントラストを上げる（HSV の V だけ CLAHE）。

    BGR各チャンネルを別々に平坦化すると色相がずれて色が崩れる。正しくは:
      1) BGR→HSV に変換し H,S,V に split
      2) clipLimit=2.0, tileGridSize=(8,8) の CLAHE を V チャンネルだけに apply
      3) merge して HSV→BGR に戻す
    返り値は BGR(uint8)。
    """
    # TODO: HSV に変換 → V に CLAHE → 戻す
    raise NotImplementedError


def ex10_warp_to_rect(img: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """演習10[難]: バラバラな4隅を与え、透視変換で正面の長方形へ起こす（統合課題）。

    手順:
      1) ex6 と同じ要領で pts を [TL,TR,BR,BL] に並べ替える
      2) 出力サイズ width/height を対辺の長い方から決める
            width  = max(|BR-BL|, |TR-TL|)
            height = max(|TR-BR|, |TL-BL|)   （いずれも int() で丸める）
      3) dst=[[0,0],[w-1,0],[w-1,h-1],[0,h-1]] へ getPerspectiveTransform → warpPerspective
    返り値は正面化した画像。ex6 の並べ替えを再利用してよい。
    """
    # TODO: 並べ替え → サイズ決定 → getPerspectiveTransform → warpPerspective
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


def _sample_uneven_doc() -> np.ndarray:
    """照明ムラのある文書グレー（右へ行くほど暗い）。ex7 用。"""
    img = np.full((140, 200), 235, dtype=np.uint8)
    f = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, "ADAPT", (15, 60), f, 1.2, 20, 3, cv2.LINE_AA)
    cv2.putText(img, "thresh", (15, 110), f, 1.0, 20, 2, cv2.LINE_AA)
    xx = np.linspace(1.0, 0.35, 200, dtype=np.float32)  # 右が暗い照明ムラ
    illum = np.repeat(xx[None, :], 140, axis=0)
    return np.clip(img.astype(np.float32) * illum, 0, 255).astype(np.uint8)


def _sample_color() -> np.ndarray:
    """低コントラストのカラー画像。ex9 用。"""
    img = np.zeros((120, 160, 3), dtype=np.uint8)
    g = np.linspace(60, 180, 160, dtype=np.uint8)
    img[:] = np.repeat(g[None, :], 120, axis=0)[:, :, None]
    cv2.rectangle(img, (20, 20, ), (70, 90), (90, 60, 50), -1)
    cv2.circle(img, (120, 60), 28, (60, 110, 190), -1)
    # 値域を圧縮して低コントラスト化
    return np.clip(img.astype(np.float32) * 0.4 + 70, 0, 255).astype(np.uint8)


def _sample_warp() -> tuple[np.ndarray, np.ndarray]:
    """傾いた小書類と、その四隅(バラバラ順)を返す。ex10 用。"""
    doc = np.full((200, 140, 3), 240, dtype=np.uint8)
    cv2.rectangle(doc, (8, 8), (131, 40), (170, 110, 40), -1)
    cv2.putText(doc, "WARP", (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(doc, "ME", (40, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (30, 30, 30), 3, cv2.LINE_AA)
    h, w = doc.shape[:2]
    src = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    dst = np.float32([[55, 35], [225, 60], [205, 250], [35, 215]])  # 傾いた四隅
    m = cv2.getPerspectiveTransform(src, dst)
    photo = cv2.warpPerspective(doc, m, (260, 280), borderValue=(50, 48, 46))
    pts = dst[[2, 0, 3, 1]].copy()  # わざと順番をシャッフルして渡す
    return photo, pts


# =====================================================================
# 模範解答（採点の基準。SHOW_SOLUTION=1 で本番関数に差し込む）
# 全文は exercises_solutions.py にも掲載。
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


def _sol_ex7(gray):
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, blockSize=21, C=8
    )


def _sol_ex8(gray):
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    return cv2.Canny(blur, 50, 150)


def _sol_ex9(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return cv2.cvtColor(cv2.merge([h, s, clahe.apply(v)]), cv2.COLOR_HSV2BGR)


def _sol_ex10(img, pts):
    quad = _sol_ex6(pts)
    tl, tr, br, bl = quad
    width = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    height = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    dst = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
    m = cv2.getPerspectiveTransform(quad, dst)
    return cv2.warpPerspective(img, m, (width, height))


def _install_solutions() -> None:
    g = globals()
    g["ex1_denoise_saltpepper"] = _sol_ex1
    g["ex2_gradient_magnitude"] = _sol_ex2
    g["ex3_otsu_binarize"] = _sol_ex3
    g["ex4_remove_specks"] = _sol_ex4
    g["ex5_count_objects"] = _sol_ex5
    g["ex6_order_corners"] = _sol_ex6
    g["ex7_adaptive_binarize"] = _sol_ex7
    g["ex8_canny_after_blur"] = _sol_ex8
    g["ex9_clahe_value_channel"] = _sol_ex9
    g["ex10_warp_to_rect"] = _sol_ex10


# =====================================================================
# 自己採点ランナー（未実装/例外でも落とさず、結果を表示して終了）
# =====================================================================

def _grade() -> bool:
    gray = _sample_gray()
    sp = _sample_saltpepper()
    clean_bin = _sample_two_shapes()
    noisy_bin = _sample_noisy_binary()
    corners = _sample_corners()
    uneven = _sample_uneven_doc()
    color = _sample_color()
    warp_img, warp_pts = _sample_warp()
    results: list[tuple[str, bool, str]] = []

    def check(name: str, fn) -> None:
        try:
            ok, detail = fn()
            results.append((name, bool(ok), detail))
        except NotImplementedError:
            results.append((name, False, "未実装（TODO を埋めてください）"))
        except Exception as e:  # noqa: BLE001  どんな例外でも落とさず FAIL 扱い
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
    check("ex7_adaptive_binarize", lambda: (
        np.array_equal(ex7_adaptive_binarize(uneven), _sol_ex7(uneven)),
        "適応的閾値で照明ムラに強く二値化",
    ))
    check("ex8_canny_after_blur", lambda: (
        np.array_equal(ex8_canny_after_blur(gray), _sol_ex8(gray)),
        "前段ぼかし付き Canny",
    ))
    check("ex9_clahe_value_channel", lambda: (
        np.array_equal(ex9_clahe_value_channel(color), _sol_ex9(color)),
        "HSV の V だけ CLAHE（色を壊さない）",
    ))

    def _check_ex10():
        got = ex10_warp_to_rect(warp_img, warp_pts)
        exp = _sol_ex10(warp_img, warp_pts)
        return (got is not None and got.shape == exp.shape and np.array_equal(got, exp),
                "四隅整列→透視変換で正面化")
    check("ex10_warp_to_rect", _check_ex10)

    print("=== 採点結果（全10問）===")
    all_ok = True
    for name, ok, detail in results:
        mark = "PASS" if ok else "FAIL"
        all_ok = all_ok and ok
        print(f"  [{mark}] {name:24s} {detail}")
    n_pass = sum(1 for _, ok, _ in results if ok)
    print(f"\n{n_pass}/{len(results)} 問 PASS")
    print("ALL PASS 🎉" if all_ok else "まだ未達の演習があります。TODO を埋めましょう。")
    return all_ok


if __name__ == "__main__":
    if os.environ.get("SHOW_SOLUTION") == "1":
        print("(模範解答モードで実行します)\n")
        _install_solutions()
    _grade()  # 返り値で分岐せず、常に正常終了（exit 0）する
