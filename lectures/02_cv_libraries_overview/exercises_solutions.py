"""exercises_solutions.py — 第2回 演習の模範解答（実行すると全 PASS）。

    uv run python lectures/02_cv_libraries_overview/exercises_solutions.py

exercises.py と同じ採点ハーネス（_grade）を再利用し、全 10 問が PASS することを確認します。
各解答は単一責務・早期 return を意識し、相互変換（色順・軸順）と uint8 の扱い、補間の
使い分け、ライブラリ選択の対応表を素直に写しています。まずは exercises.py を自力で
解いてから、答え合わせとして読んでください。

設計メモ:
  解答コードはこのファイルだけに置き（重複排除）、SOLUTIONS 辞書で公開します。
  exercises.py の _install_solutions() がこの SOLUTIONS を import して TODO を差し替えます。
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


# --- 各問の模範解答 --------------------------------------------------------

def sol_ex1_bgr_to_pil(bgr: np.ndarray) -> Image.Image:
    # BGR→RGB に直してから PIL 化（cv2 は BGR、PIL は RGB）。
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def sol_ex2_pil_size(rgb_array: np.ndarray) -> tuple[int, int]:
    # numpy の shape (H, W, ...) に対し、PIL の size は (W, H) と逆順。
    h, w = rgb_array.shape[:2]
    return (w, h)


def sol_ex3_resize_cv2(bgr: np.ndarray, width: int, height: int) -> np.ndarray:
    # dsize は (幅, 高さ) 順。縮小は INTER_AREA。
    return cv2.resize(bgr, (width, height), interpolation=cv2.INTER_AREA)


def sol_ex4_hflip_brighten(rgb: np.ndarray, beta: int) -> np.ndarray:
    # スライスで左右反転 → float で加算して飽和（clip）→ uint8 へ。
    flipped = rgb[:, ::-1]
    out = flipped.astype(np.float32) + beta
    return np.clip(out, 0, 255).astype(np.uint8)


def sol_ex5_pick_library(task: str) -> str:
    table = {
        "differentiable_gpu_aug": "kornia",
        "bbox_mask_aug": "albumentations",
        "torch_training_aug": "torchvision",
        "intuitive_edit": "Pillow",
        "fast_classic_cv": "OpenCV",
    }
    return table.get(task, "OpenCV")  # 迷ったらまず OpenCV


def sol_ex6_to_chw(rgb: np.ndarray) -> np.ndarray:
    # (H, W, C) → (C, H, W)。値は変えず軸だけ (2,0,1) に並べ替える。
    return np.transpose(rgb, (2, 0, 1))


def sol_ex7_saturating_add(rgb: np.ndarray, value: int) -> np.ndarray:
    # int16 へ広げて加算 → 0〜255 に clip → uint8（cv2.add と同じ飽和演算）。
    out = rgb.astype(np.int16) + value
    return np.clip(out, 0, 255).astype(np.uint8)


def sol_ex8_resize_long_side(bgr: np.ndarray, long_side: int) -> np.ndarray:
    h, w = bgr.shape[:2]
    longest = max(h, w)
    if longest <= long_side:
        return bgr  # 拡大はしない（早期 return）
    scale = long_side / longest
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    return cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)


def sol_ex9_roundtrip_lossless(bgr: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)   # 行き: BGR→RGB
    pil = Image.fromarray(rgb)
    back = np.asarray(pil)
    return cv2.cvtColor(back, cv2.COLOR_RGB2BGR)  # 帰り: RGB→BGR


def sol_ex10_reproducible_aug(rgb: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)             # seed 固定で再現可能
    out = rgb[:, ::-1].copy() if rng.random() < 0.5 else rgb  # coin で反転
    beta = int(rng.integers(-50, 51))             # 明るさは毎回適用（seed 依存）
    return np.clip(out.astype(np.int16) + beta, 0, 255).astype(np.uint8)


# exercises.py 側の TODO 関数名 → 模範解答のマッピング（_install_solutions が使う）。
SOLUTIONS = {
    "ex1_bgr_to_pil": sol_ex1_bgr_to_pil,
    "ex2_pil_size": sol_ex2_pil_size,
    "ex3_resize_cv2": sol_ex3_resize_cv2,
    "ex4_hflip_brighten": sol_ex4_hflip_brighten,
    "ex5_pick_library": sol_ex5_pick_library,
    "ex6_to_chw": sol_ex6_to_chw,
    "ex7_saturating_add": sol_ex7_saturating_add,
    "ex8_resize_long_side": sol_ex8_resize_long_side,
    "ex9_roundtrip_lossless": sol_ex9_roundtrip_lossless,
    "ex10_reproducible_aug": sol_ex10_reproducible_aug,
}


def main() -> None:
    """模範解答を exercises.py へ注入し、その採点ハーネスで全 PASS を確認する。"""
    import pathlib
    import sys

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    import exercises  # 同ディレクトリの演習本体

    exercises._install_solutions()  # SOLUTIONS で TODO を差し替え
    print("(模範解答で全問を採点します)\n")
    exercises._grade()


if __name__ == "__main__":
    main()
