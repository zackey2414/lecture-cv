"""第8回 演習問題（古典セグメンテーションと復元: Watershed / GrabCut / inpaint）。

使い方:
  1. 各 exN_*() 関数の中の TODO を自分で実装する（最初は未実装で FAIL になる）。
  2. 実装できたら自己採点を実行:
         uv run python lectures/08_classical_segmentation/exercises.py
     全問 pass すれば "ALL PASS" と表示される。
  3. どうしても分からない時は、模範解答の挙動を見る:
         SHOW_SOLUTION=1 uv run python lectures/08_classical_segmentation/exercises.py
     （まずは自力で！）

サンプル画像は各演習へ自動で渡される（このファイル内で合成生成。外部依存なし）。
未実装でも例外で落ちず PASS/FAIL を表示して必ず正常終了する。
"""

from __future__ import annotations

import os

import cv2
import numpy as np

# 距離変換の sure_fg を取り出すしきい値係数（ex1/ex2 で共通）。
DIST_FACTOR = 0.5


# =====================================================================
# 演習（ここを実装する）
# =====================================================================

def ex1_sure_foreground(binary: np.ndarray) -> np.ndarray:
    """演習1: 二値画像から「確実な前景(sure_fg)」を距離変換で作る。

    手順: cv2.distanceTransform(binary, cv2.DIST_L2, 5) で各白画素の距離を求め、
    その最大値の DIST_FACTOR 倍をしきい値に cv2.threshold で二値化し、
    uint8 の 0/255 マスクにして返す。
    （触れ合う物体でも、距離のピーク付近だけ残すと別々の塊になる＝watershed の種）
    """
    # TODO: dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    #       _, sfg = cv2.threshold(dist, DIST_FACTOR*dist.max(), 255, cv2.THRESH_BINARY)
    #       return sfg.astype(np.uint8)
    raise NotImplementedError


def ex2_build_markers(sure_fg: np.ndarray, unknown: np.ndarray) -> np.ndarray:
    """演習2: watershed に渡すマーカ配列を作る。

    手順: cv2.connectedComponents(sure_fg) でラベル付け(0=背景, 1,2,...=各芯)。
    全体に +1 して背景を 1 にずらし、unknown==255 の画素を 0（未確定）にして返す。
    （背景 0 と区別するため +1 する。これが watershed のお約束）
    """
    # TODO: _, markers = cv2.connectedComponents(sure_fg)
    #       markers = markers + 1
    #       markers[unknown == 255] = 0
    #       return markers
    raise NotImplementedError


def ex3_grabcut_rect(img: np.ndarray, rect: tuple[int, int, int, int]) -> np.ndarray:
    """演習3: 矩形初期化の GrabCut で前景マスク(0/255)を返す。

    手順: 空の mask と (1,65) float64 の bgdModel/fgdModel を用意し、
    cv2.grabCut(img, mask, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT) を実行。
    mask が GC_FGD か GC_PR_FGD の画素を 255、それ以外を 0 にして返す。
    """
    # TODO: mask=np.zeros(img.shape[:2],np.uint8); bgd=np.zeros((1,65),np.float64); fgd=...
    #       cv2.grabCut(img, mask, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
    #       return np.where((mask==cv2.GC_FGD)|(mask==cv2.GC_PR_FGD),255,0).astype(np.uint8)
    raise NotImplementedError


def ex4_iou(pred: np.ndarray, truth: np.ndarray) -> float:
    """演習4: 2 つの二値マスクの IoU = TP/(TP+FP+FN) を返す（float）。

    pred>0, truth>0 を前景とみなす。TP=両方前景, FP=predだけ, FN=truthだけ。
    分母が 0 のときは 0.0 を返すこと。
    """
    # TODO: p=pred>0; t=truth>0; TP/FP/FN を数えて IoU を返す
    raise NotImplementedError


def ex5_dice(pred: np.ndarray, truth: np.ndarray) -> float:
    """演習5: 2 つの二値マスクの Dice = 2*TP/(2*TP+FP+FN) を返す（float）。

    Dice は F1 スコアと同値。IoU より少しだけ甘い（小領域に優しい）指標。
    分母が 0 のときは 0.0 を返すこと。
    """
    # TODO: p=pred>0; t=truth>0; TP/FP/FN を数えて Dice を返す
    raise NotImplementedError


def ex6_inpaint_telea(broken: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """演習6: cv2.inpaint の TELEA 法で破損領域(mask>0)を復元して返す。

    cv2.inpaint(broken, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA) を返すこと。
    mask は uint8 の 0/255（>0 が修復対象）。
    """
    # TODO: return cv2.inpaint(broken, mask, 3, cv2.INPAINT_TELEA)
    raise NotImplementedError


# =====================================================================
# サンプル生成（採点用。乱数シード固定で毎回同じ入力）
# =====================================================================

def _sample_touching_binary() -> np.ndarray:
    """触れ合う2円の二値画像(0/255)。"""
    img = np.zeros((140, 200), np.uint8)
    cv2.circle(img, (75, 70), 40, 255, -1)
    cv2.circle(img, (135, 70), 40, 255, -1)  # 2円が重なって1塊に見える
    return img


def _sample_sure_and_unknown() -> tuple[np.ndarray, np.ndarray]:
    """ex2 用の sure_fg と unknown を作る。"""
    binary = _sample_touching_binary()
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist, DIST_FACTOR * dist.max(), 255, cv2.THRESH_BINARY)
    sure_fg = sure_fg.astype(np.uint8)
    sure_bg = cv2.dilate(binary, np.ones((3, 3), np.uint8), iterations=3)
    unknown = cv2.subtract(sure_bg, sure_fg)
    return sure_fg, unknown


def _sample_scene() -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]:
    """ex3 用の色付きシーンと真マスクと外接矩形。色がよく分離するので grabCut が安定。"""
    h, w = 200, 260
    rng = np.random.default_rng(5)
    img = np.full((h, w, 3), 0, np.uint8)
    img[:] = (150, 80, 40)  # 青系の背景（BGR）
    img = np.clip(img.astype(np.float32) + rng.normal(0, 6, img.shape), 0, 255).astype(np.uint8)
    truth = np.zeros((h, w), np.uint8)
    cv2.ellipse(truth, (130, 100), (60, 45), 0, 0, 360, 255, -1)
    fg = np.clip(np.array([30, 110, 230], np.float32) + rng.normal(0, 10, (h, w, 3)),
                 0, 255).astype(np.uint8)
    img[truth == 255] = fg[truth == 255]
    x, y, bw, bh = cv2.boundingRect(truth)
    return img, truth, (x - 8, y - 8, bw + 16, bh + 16)


def _sample_masks() -> tuple[np.ndarray, np.ndarray]:
    """ex4/ex5 用に、ずれた2つの矩形マスク（重なり既知）。"""
    pred = np.zeros((100, 100), np.uint8)
    truth = np.zeros((100, 100), np.uint8)
    cv2.rectangle(pred, (20, 20), (60, 60), 255, -1)   # 41x41
    cv2.rectangle(truth, (40, 40), (80, 80), 255, -1)  # 41x41, 一部重なる
    return pred, truth


def _sample_broken() -> tuple[np.ndarray, np.ndarray]:
    """ex6 用の破損画像とマスク。"""
    img = np.full((120, 160, 3), 0, np.uint8)
    img[:] = (60, 160, 90)
    cv2.circle(img, (80, 60), 35, (40, 80, 220), -1)
    mask = np.zeros((120, 160), np.uint8)
    cv2.line(mask, (10, 10), (150, 110), 255, 3)  # 斜めの傷
    broken = img.copy()
    broken[mask > 0] = (255, 255, 255)
    return broken, mask


# =====================================================================
# 模範解答
# =====================================================================

def _sol_ex1(binary):
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    _, sfg = cv2.threshold(dist, DIST_FACTOR * dist.max(), 255, cv2.THRESH_BINARY)
    return sfg.astype(np.uint8)


def _sol_ex2(sure_fg, unknown):
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    return markers


def _sol_ex3(img, rect):
    mask = np.zeros(img.shape[:2], np.uint8)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    cv2.grabCut(img, mask, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
    return np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)


def _confusion(pred, truth) -> tuple[int, int, int]:
    p, t = pred > 0, truth > 0
    tp = int(np.sum(p & t))
    fp = int(np.sum(p & ~t))
    fn = int(np.sum(~p & t))
    return tp, fp, fn


def _sol_ex4(pred, truth):
    tp, fp, fn = _confusion(pred, truth)
    return tp / (tp + fp + fn) if (tp + fp + fn) else 0.0


def _sol_ex5(pred, truth):
    tp, fp, fn = _confusion(pred, truth)
    return 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0


def _sol_ex6(broken, mask):
    return cv2.inpaint(broken, mask, 3, cv2.INPAINT_TELEA)


def _install_solutions() -> None:
    g = globals()
    g["ex1_sure_foreground"] = _sol_ex1
    g["ex2_build_markers"] = _sol_ex2
    g["ex3_grabcut_rect"] = _sol_ex3
    g["ex4_iou"] = _sol_ex4
    g["ex5_dice"] = _sol_ex5
    g["ex6_inpaint_telea"] = _sol_ex6


# =====================================================================
# 自己採点ランナー
# =====================================================================

def _iou(pred, truth) -> float:
    tp, fp, fn = _confusion(pred, truth)
    return tp / (tp + fp + fn) if (tp + fp + fn) else 0.0


def _grade() -> None:
    binary = _sample_touching_binary()
    sure_fg, unknown = _sample_sure_and_unknown()
    scene, truth_scene, rect = _sample_scene()
    pred_m, truth_m = _sample_masks()
    broken, mask = _sample_broken()
    results: list[tuple[str, bool, str]] = []

    def check(name: str, fn) -> None:
        try:
            ok, detail = fn()
            results.append((name, bool(ok), detail))
        except NotImplementedError:
            results.append((name, False, "未実装（TODO を埋めてください）"))
        except Exception as e:  # noqa: BLE001
            results.append((name, False, f"例外: {type(e).__name__}: {e}"))

    check("ex1_sure_foreground", lambda: (
        np.array_equal(ex1_sure_foreground(binary), _sol_ex1(binary)),
        "distanceTransform で sure_fg",
    ))
    check("ex2_build_markers", lambda: (
        np.array_equal(ex2_build_markers(sure_fg, unknown), _sol_ex2(sure_fg, unknown)),
        "connectedComponents +1 / unknown=0",
    ))
    # GrabCut は内部 RNG でわずかに揺れるため、真マスクとの IoU が高いかで採点する。
    check("ex3_grabcut_rect", lambda: (
        _iou(ex3_grabcut_rect(scene, rect), truth_scene) > 0.85,
        "GrabCut(rect) で前景抽出 / IoU>0.85",
    ))
    check("ex4_iou", lambda: (
        abs(float(ex4_iou(pred_m, truth_m)) - _sol_ex4(pred_m, truth_m)) < 1e-9,
        "IoU = TP/(TP+FP+FN)",
    ))
    check("ex5_dice", lambda: (
        abs(float(ex5_dice(pred_m, truth_m)) - _sol_ex5(pred_m, truth_m)) < 1e-9,
        "Dice = 2TP/(2TP+FP+FN)",
    ))
    check("ex6_inpaint_telea", lambda: (
        np.array_equal(ex6_inpaint_telea(broken, mask), _sol_ex6(broken, mask)),
        "cv2.inpaint(TELEA) で傷消し",
    ))

    print("=== 採点結果 ===")
    all_ok = True
    for name, ok, detail in results:
        mark = "PASS" if ok else "FAIL"
        all_ok = all_ok and ok
        print(f"  [{mark}] {name:22s} {detail}")
    print("\nALL PASS 🎉" if all_ok else "\nまだ未達の演習があります。TODO を埋めましょう。")


if __name__ == "__main__":
    if os.environ.get("SHOW_SOLUTION") == "1":
        print("(模範解答モードで実行します)\n")
        _install_solutions()
    _grade()
