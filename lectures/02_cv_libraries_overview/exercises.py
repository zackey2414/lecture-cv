"""第2回 演習問題（画像・動画処理ライブラリの地図）。

この回は「概念回」なので、演習も「ライブラリ間の橋渡し」と「選び方」を中心に置きます。
新しい画像処理アルゴリズムを覚えるより、

  - cv2(BGR) ↔ PIL(RGB) ↔ numpy の相互変換を淀みなく書けること
  - 「幅・高さ」と「shape の (H, W)」の軸順を取り違えないこと
  - 課題に応じて適切なライブラリを選べること（地図の暗記）

を体に入れるのが目的です。

使い方:
  1. 各 exN_*() 関数の中の TODO を自分で実装する（最初は NotImplementedError が出る）。
  2. 実装できたら自己採点を実行:
         uv run python lectures/02_cv_libraries_overview/exercises.py
     全問が pass すれば "ALL PASS" と表示される。
  3. どうしても分からない時は、模範解答の挙動を確認する:
         SHOW_SOLUTION=1 uv run python lectures/02_cv_libraries_overview/exercises.py
     （まずは自力で！）

ヒント: サンプル画像は cv_helpers.get_sample_bgr() で得られる（BGR uint8）。
出力の保存先は cv_helpers.output_dir() を使うと outputs/02_cv_libraries_overview/ になる。
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

def ex1_bgr_to_pil(bgr: np.ndarray) -> Image.Image:
    """演習1: OpenCV の BGR 配列を、色が正しい PIL.Image（RGB）に変換して返す。

    ポイント: cv2 は BGR、PIL は RGB。fromarray に渡す前に BGR→RGB へ変換する。
    変換を忘れると赤と青が入れ替わった画像になる。
    """
    # TODO: cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) してから Image.fromarray で PIL 化して返す
    raise NotImplementedError


def ex2_pil_size(rgb_array: np.ndarray) -> tuple[int, int]:
    """演習2: RGB の numpy 配列 (H, W, 3) から、PIL の size タプル (幅 W, 高さ H) を返す。

    ポイント: numpy の shape は (H, W, 3)、PIL の size は (W, H) で軸順が逆。
    実際に Image.fromarray して .size を返しても、shape から組み立ててもよい。
    """
    # TODO: (幅, 高さ) すなわち (shape[1], shape[0]) を返す（PIL.size と一致させる）
    raise NotImplementedError


def ex3_resize_cv2(bgr: np.ndarray, width: int, height: int) -> np.ndarray:
    """演習3: cv2.resize で (width, height) のサイズへ縮小し、結果を返す。

    ポイント: cv2.resize の dsize は (幅, 高さ) の順。shape の (高さ, 幅) と逆なので、
    返り値の shape は必ず (height, width, 3) になること。縮小なので INTER_AREA を使う。
    """
    # TODO: cv2.resize(bgr, (width, height), interpolation=cv2.INTER_AREA) を返す
    raise NotImplementedError


def ex4_hflip_brighten(rgb: np.ndarray, beta: int) -> np.ndarray:
    """演習4: 最小の自作拡張。左右反転してから明るさを +beta する（飽和させる）。

    ポイント: 反転は numpy スライス [:, ::-1]。明るさ加算は uint8 のままだと
    オーバーフローするので float で計算 → np.clip(..,0,255) → uint8 に戻す。
    返り値の shape は入力と同じであること。
    """
    # TODO: flipped = rgb[:, ::-1]; out = clip(flipped.astype(float32)+beta,0,255).astype(uint8)
    raise NotImplementedError


def ex5_pick_library(task: str) -> str:
    """演習5: 課題キーワードから「まず選ぶライブラリ名」を返す（地図の暗記）。

    対応表（この通りに返すこと）:
      "differentiable_gpu_aug" -> "kornia"          # 学習ループ内・GPU・微分可能な拡張
      "bbox_mask_aug"          -> "albumentations"  # 検出/セグメ用に bbox/mask ごと拡張
      "torch_training_aug"     -> "torchvision"     # PyTorch 学習の前処理/拡張(v2)
      "intuitive_edit"         -> "Pillow"          # 直感的な画像編集・フォント描画
      "fast_classic_cv"        -> "OpenCV"          # 高速な古典CV・動画I/O
    上記以外のキーは "OpenCV"（迷ったらまず OpenCV）を返す。
    """
    # TODO: 上の対応表を dict で持ち、task に対応する値を返す（未知キーは "OpenCV"）
    raise NotImplementedError


# =====================================================================
# 自己採点ランナー
# =====================================================================

def _grade() -> None:
    bgr = get_sample_bgr()
    output_dir()  # 出力先を作っておく（演習で保存したくなった時のため）
    results: list[tuple[str, bool, str]] = []

    def check(name: str, fn) -> None:
        try:
            ok, detail = fn()
            results.append((name, bool(ok), detail))
        except NotImplementedError:
            results.append((name, False, "未実装（TODOを埋めてください）"))
        except Exception as e:  # noqa: BLE001
            results.append((name, False, f"例外: {type(e).__name__}: {e}"))

    # ex1: BGR→PIL(RGB)。PIL 画像を numpy に戻すと cvtColor の結果と一致するはず。
    def _check_ex1():
        pil = ex1_bgr_to_pil(bgr)
        if not isinstance(pil, Image.Image):
            return False, "PIL.Image を返すこと"
        got = np.asarray(pil)
        want = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return np.array_equal(got, want), "BGR→RGB→PIL の色一致"
    check("ex1_bgr_to_pil", _check_ex1)

    # ex2: PIL の size (W, H) を返せているか。
    def _check_ex2():
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        got = ex2_pil_size(rgb)
        want = Image.fromarray(rgb).size  # (W, H)
        return tuple(got) == tuple(want), f"size {want} を返す"
    check("ex2_pil_size", _check_ex2)

    # ex3: cv2.resize の dsize 順。出力 shape が (H, W, 3) になるか。
    def _check_ex3():
        out = ex3_resize_cv2(bgr, 100, 60)  # (幅100, 高さ60)
        return out.shape == (60, 100, 3), f"shape={out.shape} 期待(60,100,3)"
    check("ex3_resize_cv2", _check_ex3)

    # ex4: 反転＋明るさ加算（飽和）。
    def _check_ex4():
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        got = ex4_hflip_brighten(rgb, 50)
        flipped = rgb[:, ::-1]
        want = np.clip(flipped.astype(np.float32) + 50, 0, 255).astype(np.uint8)
        return (got.shape == rgb.shape and np.array_equal(got, want)), "反転+明るさ飽和"
    check("ex4_hflip_brighten", _check_ex4)

    # ex5: ライブラリ選択（地図の暗記）。
    def _check_ex5():
        cases = {
            "differentiable_gpu_aug": "kornia",
            "bbox_mask_aug": "albumentations",
            "torch_training_aug": "torchvision",
            "intuitive_edit": "Pillow",
            "fast_classic_cv": "OpenCV",
            "something_unknown": "OpenCV",
        }
        ok = all(ex5_pick_library(k) == v for k, v in cases.items())
        return ok, "課題→ライブラリの対応"
    check("ex5_pick_library", _check_ex5)

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

def _sol_ex1(bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def _sol_ex2(rgb_array):
    h, w = rgb_array.shape[:2]
    return (w, h)


def _sol_ex3(bgr, width, height):
    return cv2.resize(bgr, (width, height), interpolation=cv2.INTER_AREA)


def _sol_ex4(rgb, beta):
    flipped = rgb[:, ::-1]
    out = flipped.astype(np.float32) + beta
    return np.clip(out, 0, 255).astype(np.uint8)


def _sol_ex5(task):
    table = {
        "differentiable_gpu_aug": "kornia",
        "bbox_mask_aug": "albumentations",
        "torch_training_aug": "torchvision",
        "intuitive_edit": "Pillow",
        "fast_classic_cv": "OpenCV",
    }
    return table.get(task, "OpenCV")


def _install_solutions() -> None:
    """模範解答で TODO 関数を差し替える（教材検証・答え合わせ用）。"""
    g = globals()
    g["ex1_bgr_to_pil"] = _sol_ex1
    g["ex2_pil_size"] = _sol_ex2
    g["ex3_resize_cv2"] = _sol_ex3
    g["ex4_hflip_brighten"] = _sol_ex4
    g["ex5_pick_library"] = _sol_ex5


if __name__ == "__main__":
    if os.environ.get("SHOW_SOLUTION") == "1":
        print("(模範解答モードで実行します)\n")
        _install_solutions()
    _grade()
