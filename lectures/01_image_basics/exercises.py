"""第1回 演習問題（画像の基礎 / OpenCV + Pillow 入門）。

使い方:
  1. 各 exN_*() 関数の中の TODO を自分で実装する（最初は NotImplementedError が出る）。
  2. 実装できたら自己採点を実行:
         uv run python lectures/01_image_basics/exercises.py
     全問が pass すれば "ALL PASS" と表示される。
  3. どうしても分からない時は、模範解答を実行して挙動を確認する:
         SHOW_SOLUTION=1 uv run python lectures/01_image_basics/exercises.py
     もしくは完全な模範解答ファイルを実行する:
         uv run python lectures/01_image_basics/exercises_solutions.py
     （まずは自力で！）

ヒント: サンプル画像は cv_helpers.get_sample_bgr() で得られる（BGR uint8）。
出力の保存先は cv_helpers.output_dir() を使うと lectures/01_image_basics/outputs/ になる。

10問（易→難）:
  ex1 : 画像の素性（shape / dtype / min / max）        … 配列としての画像
  ex2 : 画素を RGB 並びで取り出す                       … BGR/RGB の基本
  ex3 : 明るさ +value（numpy + と cv2.add の2通り）     … uint8 オーバーフロー/飽和
  ex4 : cv2→PIL→numpy→cv2 ラウンドトリップ一致         … 3者の相互変換
  ex5 : 日本語パスで保存→読み戻し                       … 非ASCIIパスの定石
  ex6 : グレースケール化して shape を返す               … 次元が1つ減る感覚
  ex7 : ROI（矩形領域）をスライスで取り出す              … numpy スライス = 画像操作
  ex8 : R チャンネルの最大値を返す                       … BGR のチャンネル順の理解
  ex9 : 安全な明るさ加算（int16 + clip = cv2.add）       … 値域を意識した演算
  ex10: グレーを3chへ戻して元カラーと横連結              … チャンネル軸の取り回し
"""

from __future__ import annotations

import os
import pathlib
import sys

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from cv_helpers import get_sample_bgr, output_dir  # noqa: E402


# =====================================================================
# 演習（ここを実装する）
# =====================================================================

def ex1_inspect(bgr: np.ndarray) -> tuple[tuple[int, ...], str, int, int]:
    """演習1: 画像の素性を返す。

    引数 bgr（BGR画像）について、(shape, dtype名, 最小値, 最大値) のタプルを返すこと。
    返り値の例: ((240, 320, 3), "uint8", 0, 255)
    """
    # TODO: img.shape / str(img.dtype) / int(img.min()) / int(img.max()) を組み立てて返す
    raise NotImplementedError


def ex2_to_rgb_pixel(bgr: np.ndarray, y: int, x: int) -> list[int]:
    """演習2: 画素(y, x)を RGB 並び [R, G, B] のリストで返す。

    OpenCV の画素は [B, G, R]。これを cvtColor もしくは並べ替えで [R, G, B] にする。
    """
    # TODO: cv2.cvtColor(..., cv2.COLOR_BGR2RGB) してから [y, x] を取り出し list 化して返す
    raise NotImplementedError


def ex3_brightness(bgr: np.ndarray, value: int) -> tuple[np.ndarray, np.ndarray]:
    """演習3: 明るさ +value を「numpyの+」と「cv2.add」の2通りで作り、(naive, saturated) を返す。

    naive は 255 を超えると巻き戻り、saturated は 255 で頭打ちになるはず。
    """
    # TODO: add = np.full_like(bgr, value) を作り、bgr + add と cv2.add(bgr, add) を返す
    raise NotImplementedError


def ex4_roundtrip(bgr: np.ndarray) -> bool:
    """演習4: cv2(BGR) → PIL(RGB) → numpy → cv2(BGR) と1周し、元の画像と一致するか返す。

    変換チェーンが正しく閉じていれば True。
    """
    # TODO: cvtColor で RGB 化 → Image.fromarray → np.asarray → cvtColor で BGR へ戻す
    #       np.array_equal(元のbgr, 戻したbgr) を返す
    raise NotImplementedError


def ex5_save_unicode(bgr: np.ndarray, out_dir: pathlib.Path) -> bool:
    """演習5: 日本語ファイル名 '演習_出力.png' で保存し、読み戻せたら True を返す。

    非ASCIIパスでも安全な方法（cv2.imencode + tofile / np.fromfile + cv2.imdecode）を使うこと。
    """
    # TODO: imencode(".png", bgr) → tofile で保存し、np.fromfile + imdecode で読み戻して
    #       None でないことを確認して True/False を返す
    raise NotImplementedError


def ex6_gray_shape(bgr: np.ndarray) -> tuple[int, ...]:
    """演習6: BGR画像をグレースケール化し、その shape（2次元 (H, W)）を返す。

    カラーは (H, W, 3)、グレーは (H, W)。色を落とすと次元が1つ減ることを確認する。
    """
    # TODO: cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) の .shape を返す
    raise NotImplementedError


def ex7_crop(bgr: np.ndarray, y0: int, x0: int, y1: int, x1: int) -> np.ndarray:
    """演習7: 矩形 ROI（y0:y1, x0:x1）をスライスで切り出して返す。

    画像 = numpy 配列なので、専用 API ではなくスライスで領域を取り出せる。
    返り値の shape は (y1-y0, x1-x0, 3) になるはず。
    """
    # TODO: numpy スライス bgr[y0:y1, x0:x1] を返す（.copy() でもビューでも可）
    raise NotImplementedError


def ex8_red_channel_max(bgr: np.ndarray) -> int:
    """演習8: R（赤）チャンネルの最大値を int で返す。

    OpenCV は BGR 並びなので、R は index 2（B=0, G=1, R=2）。
    img[:, :, 0] を返すと B の最大になってしまう点に注意。
    """
    # TODO: bgr[:, :, 2] の max を int にして返す
    raise NotImplementedError


def ex9_safe_brightness(bgr: np.ndarray, value: int) -> np.ndarray:
    """演習9: オーバーフローしない安全な明るさ加算を、広い型 + clip で実装する。

    一旦 int16（または float）に広げて value を足し、np.clip(..., 0, 255) で値域に収め、
    uint8 に戻す。結果は cv2.add(bgr, value) と完全一致するはず（= 飽和演算の手作り）。
    """
    # TODO: bgr.astype(np.int16) + value → np.clip(..., 0, 255) → astype(np.uint8) を返す
    raise NotImplementedError


def ex10_gray_to_bgr_hconcat(bgr: np.ndarray) -> np.ndarray:
    """演習10: グレースケールを3chに戻し、元のカラー画像と横に連結して返す。

    グレー画像は (H, W) の2次元なので、そのままでは3chのカラーと連結できない。
    cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) で「見た目は白黒のまま3ch」にしてから np.hstack する。
    返り値の shape は (H, 2W, 3) になるはず。
    """
    # TODO: gray = BGR2GRAY → gray3 = GRAY2BGR → np.hstack([bgr, gray3]) を返す
    raise NotImplementedError


# =====================================================================
# 自己採点ランナー
# =====================================================================

def _grade() -> None:
    bgr = get_sample_bgr()
    out = output_dir()
    results: list[tuple[str, bool, str]] = []

    def check(name: str, fn) -> None:
        try:
            ok, detail = fn()
            results.append((name, ok, detail))
        except NotImplementedError:
            results.append((name, False, "未実装（TODOを埋めてください）"))
        except Exception as e:  # noqa: BLE001  どんな例外でもプロセスは落とさず FAIL 扱い
            results.append((name, False, f"例外: {type(e).__name__}: {e}"))

    check("ex1_inspect", lambda: (
        ex1_inspect(bgr) == (bgr.shape, str(bgr.dtype), int(bgr.min()), int(bgr.max())),
        "shape/dtype/min/max",
    ))
    check("ex2_to_rgb_pixel", lambda: (
        ex2_to_rgb_pixel(bgr, 10, 20) == cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)[10, 20].tolist(),
        "RGB画素",
    ))

    def _check_ex3():
        naive, sat = ex3_brightness(bgr, 100)
        expect_naive = bgr + np.full_like(bgr, 100)
        expect_sat = cv2.add(bgr, np.full_like(bgr, 100))
        return (np.array_equal(naive, expect_naive) and np.array_equal(sat, expect_sat),
                "オーバーフロー/飽和")
    check("ex3_brightness", _check_ex3)
    check("ex4_roundtrip", lambda: (ex4_roundtrip(bgr) is True, "ラウンドトリップ一致"))
    check("ex5_save_unicode", lambda: (ex5_save_unicode(bgr, out) is True, "日本語パス保存"))

    def _check_ex6():
        h, w = bgr.shape[:2]
        return (tuple(ex6_gray_shape(bgr)) == (h, w), "グレーは(H,W)の2次元")
    check("ex6_gray_shape", _check_ex6)

    def _check_ex7():
        got = ex7_crop(bgr, 30, 40, 150, 200)
        return (np.array_equal(np.asarray(got), bgr[30:150, 40:200]), "ROIスライス")
    check("ex7_crop", _check_ex7)

    check("ex8_red_channel_max", lambda: (
        ex8_red_channel_max(bgr) == int(bgr[:, :, 2].max()),
        "Rチャンネルの最大値",
    ))

    def _check_ex9():
        got = ex9_safe_brightness(bgr, 100)
        expect = cv2.add(bgr, np.full_like(bgr, 100))  # 飽和演算と一致するのが正解
        got = np.asarray(got)
        return (got.dtype == np.uint8 and np.array_equal(got, expect), "安全な明るさ加算=飽和")
    check("ex9_safe_brightness", _check_ex9)

    def _check_ex10():
        got = np.asarray(ex10_gray_to_bgr_hconcat(bgr))
        h, w = bgr.shape[:2]
        ok_shape = got.shape == (h, 2 * w, 3)
        ok_left = ok_shape and np.array_equal(got[:, :w], bgr)  # 左半分は元カラー
        return (bool(ok_left), "グレー3ch化して横連結 (H,2W,3)")
    check("ex10_gray_to_bgr_hconcat", _check_ex10)

    print("=== 採点結果 ===")
    all_ok = True
    for name, ok, detail in results:
        mark = "PASS" if ok else "FAIL"
        all_ok = all_ok and ok
        print(f"  [{mark}] {name:24s} {detail}")
    print("\nALL PASS 🎉" if all_ok else "\nまだ未達の演習があります。TODO を埋めましょう。")


# =====================================================================
# 模範解答（SHOW_SOLUTION=1 のときに本体へ差し替えて実行）
# まずは自力で解いてから見ること。
# =====================================================================

def _sol_ex1(bgr):
    return (bgr.shape, str(bgr.dtype), int(bgr.min()), int(bgr.max()))


def _sol_ex2(bgr, y, x):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb[y, x].tolist()


def _sol_ex3(bgr, value):
    add = np.full_like(bgr, value)
    return bgr + add, cv2.add(bgr, add)


def _sol_ex4(bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    back = cv2.cvtColor(np.asarray(Image.fromarray(rgb)), cv2.COLOR_RGB2BGR)
    return np.array_equal(bgr, back)


def _sol_ex5(bgr, out_dir):
    path = pathlib.Path(out_dir) / "演習_出力.png"
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        return False
    buf.tofile(str(path))
    reread = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    return reread is not None


def _sol_ex6(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).shape


def _sol_ex7(bgr, y0, x0, y1, x1):
    return bgr[y0:y1, x0:x1]


def _sol_ex8(bgr):
    return int(bgr[:, :, 2].max())


def _sol_ex9(bgr, value):
    wide = bgr.astype(np.int16) + value
    return np.clip(wide, 0, 255).astype(np.uint8)


def _sol_ex10(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray3 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return np.hstack([bgr, gray3])


def _install_solutions() -> None:
    """模範解答で TODO 関数を差し替える（教材検証・答え合わせ用）。"""
    g = globals()
    g["ex1_inspect"] = _sol_ex1
    g["ex2_to_rgb_pixel"] = _sol_ex2
    g["ex3_brightness"] = _sol_ex3
    g["ex4_roundtrip"] = _sol_ex4
    g["ex5_save_unicode"] = _sol_ex5
    g["ex6_gray_shape"] = _sol_ex6
    g["ex7_crop"] = _sol_ex7
    g["ex8_red_channel_max"] = _sol_ex8
    g["ex9_safe_brightness"] = _sol_ex9
    g["ex10_gray_to_bgr_hconcat"] = _sol_ex10


if __name__ == "__main__":
    if os.environ.get("SHOW_SOLUTION") == "1":
        print("(模範解答モードで実行します)\n")
        _install_solutions()
    _grade()
