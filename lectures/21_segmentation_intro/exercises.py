"""第21回 演習問題（セマンティックセグメンテーションの評価指標）。

使い方:
  1. 各 exN_*() の TODO を自分で実装する（最初は NotImplementedError だが、
     採点ランナーが拾うのでプロセスは落ちず、FAIL と表示されるだけ）。
  2. 自己採点:  uv run python lectures/21_segmentation_intro/exercises.py
     全問 pass すれば "ALL PASS" と表示される。
  3. 詰まったら模範解答の挙動を見る:
        SHOW_SOLUTION=1 uv run python lectures/21_segmentation_intro/exercises.py

この5問は本モジュールの核「画素混同行列 → 各指標」を1段ずつ分解したもの
（モデルDL不要・純粋な numpy 計算）:
  ex1: 画素混同行列 cm[g,p] を作る                       … 03
  ex2: per-class IoU（未出現クラスは NaN）               … 03
  ex3: mIoU（per-class IoU の nanmean）                  … 03
  ex4: per-class Dice（=F1, 未出現クラスは NaN）         … 03
  ex5: pixel accuracy（ΣTP / 全画素）                    … 03
"""

from __future__ import annotations

import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))


# =====================================================================
# 演習（ここを実装する）
# =====================================================================
def ex1_confusion_matrix(gt: np.ndarray, pred: np.ndarray, num_classes: int) -> np.ndarray:
    """演習1: GT と予測のクラスマップから KxK の画素混同行列 cm[g, p] を作る。

    - gt, pred: 同じ形 (H, W) の int 配列。値は 0..num_classes-1。
    - cm[g, p] = 「正解が g で予測が p の画素数」。形は (num_classes, num_classes)。
    - ヒント: idx = gt*num_classes + pred を平坦化し np.bincount(idx, minlength=K*K) → reshape。
    """
    # TODO: 二重ループでも良いが、bincount を使うと速い
    raise NotImplementedError


def ex2_per_class_iou(cm: np.ndarray) -> np.ndarray:
    """演習2: 混同行列から per-class IoU を返す（未出現クラスは np.nan）。

    - TP_c = cm[c, c], FP_c = 列和 - TP, FN_c = 行和 - TP。
    - IoU_c = TP / (TP + FP + FN)。
    - GT にも予測にも出ないクラス（TP+FP+FN == 0）は 0/0 なので np.nan にする。
    - 返り値は形 (num_classes,) の float 配列。
    """
    # TODO: tp/fp/fn を出し、分母0のクラスを nan にして IoU を返す
    raise NotImplementedError


def ex3_mean_iou(cm: np.ndarray) -> float:
    """演習3: mIoU（per-class IoU のクラス平均）を返す。

    - 未出現クラスの NaN は平均から除外する（np.nanmean を使う）。
    - ex2 を再利用してよい。
    """
    # TODO: np.nanmean(ex2_per_class_iou(cm)) を float で返す
    raise NotImplementedError


def ex4_per_class_dice(cm: np.ndarray) -> np.ndarray:
    """演習4: per-class Dice係数（= F1）を返す（未出現クラスは np.nan）。

    - Dice_c = 2*TP / (2*TP + FP + FN)。
    - 未出現クラス（分母0）は np.nan。
    - 返り値は形 (num_classes,) の float 配列。
    """
    # TODO: ex2 と同じ要領で 2TP/(2TP+FP+FN) を計算（分母0は nan）
    raise NotImplementedError


def ex5_pixel_accuracy(cm: np.ndarray) -> float:
    """演習5: pixel accuracy（正しく分類された画素の割合）を返す。

    - pixel acc = 対角の総和(ΣTP) / 全画素(cm.sum())。
    - cm.sum() == 0 のときは 0.0 を返す（ゼロ割回避）。
    """
    # TODO: np.diag(cm).sum() / cm.sum() を float で返す
    raise NotImplementedError


# =====================================================================
# 自己採点ランナー
# =====================================================================
# 採点用の固定混同行列（ex2〜5 を ex1 と独立に検証するため直接与える）。
#   gt   = [[0,0,1],[2,2,1]], pred = [[0,1,1],[2,2,1]], num_classes=4（class3 は未出現）
#   cm   = [[1,1,0,0],[0,2,0,0],[0,0,2,0],[0,0,0,0]]
#   IoU  = [0.5, 2/3, 1.0, NaN]   Dice = [2/3, 0.8, 1.0, NaN]
#   mIoU = nanmean = 0.7222...    pixel acc = 5/6 = 0.8333...
REF_CM = np.array([[1, 1, 0, 0], [0, 2, 0, 0], [0, 0, 2, 0], [0, 0, 0, 0]], dtype=np.int64)
REF_IOU = np.array([0.5, 2 / 3, 1.0, np.nan])
REF_DICE = np.array([2 / 3, 0.8, 1.0, np.nan])


def _grade() -> None:
    results: list[tuple[str, bool, str]] = []

    def check(name: str, fn) -> None:
        try:
            ok, detail = fn()
            results.append((name, bool(ok), detail))
        except NotImplementedError:
            results.append((name, False, "未実装（TODOを埋めてください）"))
        except Exception as e:  # noqa: BLE001
            results.append((name, False, f"例外: {type(e).__name__}: {e}"))

    def _c1():
        gt = np.array([[0, 0, 1], [2, 2, 1]])
        pred = np.array([[0, 1, 1], [2, 2, 1]])
        got = ex1_confusion_matrix(gt, pred, 4)
        return np.array_equal(got, REF_CM), f"cm=\n{got}"
    check("ex1_confusion_matrix", _c1)

    def _c2():
        got = ex2_per_class_iou(REF_CM)
        return np.allclose(got, REF_IOU, atol=1e-6, equal_nan=True), f"iou={np.round(got, 3)}"
    check("ex2_per_class_iou", _c2)

    def _c3():
        got = ex3_mean_iou(REF_CM)
        return abs(got - float(np.nanmean(REF_IOU))) < 1e-6, f"mIoU={got:.4f}"
    check("ex3_mean_iou", _c3)

    def _c4():
        got = ex4_per_class_dice(REF_CM)
        return np.allclose(got, REF_DICE, atol=1e-6, equal_nan=True), f"dice={np.round(got, 3)}"
    check("ex4_per_class_dice", _c4)

    def _c5():
        got = ex5_pixel_accuracy(REF_CM)
        zero = ex5_pixel_accuracy(np.zeros((4, 4), dtype=np.int64))  # ゼロ割回避
        return abs(got - 5 / 6) < 1e-6 and zero == 0.0, f"pixel_acc={got:.4f}"
    check("ex5_pixel_accuracy", _c5)

    print("=== 採点結果 ===")
    all_ok = True
    for name, ok, detail in results:
        mark = "PASS" if ok else "FAIL"
        all_ok = all_ok and ok
        print(f"  [{mark}] {name:22s} {detail}")
    print("\nALL PASS 🎉" if all_ok else "\nまだ未達の演習があります。TODO を埋めましょう。")


# =====================================================================
# 模範解答（SHOW_SOLUTION=1 のときに本体へ差し替えて実行）
# まずは自力で解いてから見ること。
# =====================================================================
def _sol_ex1(gt: np.ndarray, pred: np.ndarray, num_classes: int) -> np.ndarray:
    idx = gt.ravel() * num_classes + pred.ravel()
    cm = np.bincount(idx, minlength=num_classes * num_classes)
    return cm.reshape(num_classes, num_classes).astype(np.int64)


def _sol_ex2(cm: np.ndarray) -> np.ndarray:
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    denom = tp + fp + fn
    iou = np.full(cm.shape[0], np.nan, dtype=np.float64)
    present = denom > 0
    iou[present] = tp[present] / denom[present]
    return iou


def _sol_ex3(cm: np.ndarray) -> float:
    return float(np.nanmean(_sol_ex2(cm)))


def _sol_ex4(cm: np.ndarray) -> np.ndarray:
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    denom = 2 * tp + fp + fn
    dice = np.full(cm.shape[0], np.nan, dtype=np.float64)
    present = denom > 0
    dice[present] = 2 * tp[present] / denom[present]
    return dice


def _sol_ex5(cm: np.ndarray) -> float:
    total = cm.sum()
    if total == 0:
        return 0.0
    return float(np.diag(cm).sum() / total)


def _install_solutions() -> None:
    g = globals()
    g["ex1_confusion_matrix"] = _sol_ex1
    g["ex2_per_class_iou"] = _sol_ex2
    g["ex3_mean_iou"] = _sol_ex3
    g["ex4_per_class_dice"] = _sol_ex4
    g["ex5_pixel_accuracy"] = _sol_ex5


if __name__ == "__main__":
    if os.environ.get("SHOW_SOLUTION") == "1":
        print("(模範解答モードで実行します)\n")
        _install_solutions()
    _grade()
