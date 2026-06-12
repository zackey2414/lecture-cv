"""第21回 演習の模範解答（全 PASS）。

このファイルは exercises.py の演習 ex1〜ex10 の“正解実装”を集め、
exercises.py 側の grade()（採点ロジック）をそのまま再利用して全問 PASS を確認する。
採点の二重定義を避けるため、ここには「解答」だけを置き、判定は exercises.grade に委譲する。

実行: uv run python lectures/21_segmentation_intro/exercises_solutions.py
  → 10 問すべて PASS することを確認できる（まずは自力で exercises.py を解いてから見ること）。
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from exercises import EX_NAMES, grade  # noqa: E402  採点ロジックは exercises 側を再利用


def ex1_confusion_matrix(gt: np.ndarray, pred: np.ndarray, num_classes: int) -> np.ndarray:
    """(g,p) を一意な整数に符号化し bincount で一気に数える定石。"""
    idx = gt.ravel() * num_classes + pred.ravel()
    cm = np.bincount(idx, minlength=num_classes * num_classes)
    return cm.reshape(num_classes, num_classes).astype(np.int64)


def _tp_fp_fn(cm: np.ndarray):
    """混同行列から TP/FP/FN ベクトルを取り出す共通処理。"""
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0) - tp  # 列和 - TP（そのクラスへ誤検出した画素）
    fn = cm.sum(axis=1) - tp  # 行和 - TP（そのクラスを取りこぼした画素）
    return tp, fp, fn


def ex2_per_class_iou(cm: np.ndarray) -> np.ndarray:
    """IoU = TP/(TP+FP+FN)。分母0（未出現）は NaN にする。"""
    tp, fp, fn = _tp_fp_fn(cm)
    denom = tp + fp + fn
    iou = np.full(cm.shape[0], np.nan, dtype=np.float64)
    present = denom > 0
    iou[present] = tp[present] / denom[present]
    return iou


def ex3_mean_iou(cm: np.ndarray) -> float:
    """per-class IoU の nanmean（未出現クラスを除外）。"""
    return float(np.nanmean(ex2_per_class_iou(cm)))


def ex4_per_class_dice(cm: np.ndarray) -> np.ndarray:
    """Dice = 2TP/(2TP+FP+FN)。分母0（未出現）は NaN。"""
    tp, fp, fn = _tp_fp_fn(cm)
    denom = 2 * tp + fp + fn
    dice = np.full(cm.shape[0], np.nan, dtype=np.float64)
    present = denom > 0
    dice[present] = 2 * tp[present] / denom[present]
    return dice


def ex5_pixel_accuracy(cm: np.ndarray) -> float:
    """ΣTP / 全画素。空行列は 0.0。"""
    total = cm.sum()
    if total == 0:
        return 0.0
    return float(np.diag(cm).sum() / total)


def ex6_fwiou(cm: np.ndarray) -> float:
    """FWIoU = Σ freq_c * IoU_c（freq は GT 画素割合）。NaN は nansum で除外。"""
    total = cm.sum()
    if total == 0:
        return 0.0
    freq = cm.sum(axis=1).astype(np.float64) / total  # 行和 = GT のクラス画素数
    iou = ex2_per_class_iou(cm)
    return float(np.nansum(freq * iou))


def ex7_confusion_matrix_ignore(
    gt: np.ndarray, pred: np.ndarray, num_classes: int, ignore_index: int
) -> np.ndarray:
    """gt==ignore_index の画素を捨ててから混同行列を作る。"""
    g = gt.ravel()
    p = pred.ravel()
    keep = g != ignore_index
    g, p = g[keep], p[keep]
    idx = g * num_classes + p
    cm = np.bincount(idx, minlength=num_classes * num_classes)
    return cm.reshape(num_classes, num_classes).astype(np.int64)


def ex8_binary_mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """|A∩B| / |A∪B|。union==0（両方空）は完全一致とみなして 1.0。"""
    a = np.asarray(mask_a, dtype=bool)
    b = np.asarray(mask_b, dtype=bool)
    inter = float(np.logical_and(a, b).sum())
    union = float(np.logical_or(a, b).sum())
    if union == 0:
        return 1.0
    return inter / union


def ex9_dice_from_iou(iou):
    """Dice = 2·IoU/(1+IoU)。スカラ/配列のどちらでも同じ式で動く。"""
    return 2 * iou / (1 + iou)


def ex10_global_vs_samplewise_miou(pairs: list, num_classes: int) -> tuple[float, float]:
    """global（全画素を1つの cm に積む）と samplewise（画像ごと mIoU の平均）。"""
    cm_global = np.zeros((num_classes, num_classes), dtype=np.int64)
    per_image = []
    for gt, pred in pairs:
        cm = ex1_confusion_matrix(np.asarray(gt), np.asarray(pred), num_classes)
        cm_global += cm
        per_image.append(ex3_mean_iou(cm))
    global_miou = ex3_mean_iou(cm_global)
    samplewise_miou = float(np.mean(per_image))
    return global_miou, samplewise_miou


def solution_functions() -> dict:
    """exercises.grade に渡す「名前 -> 解答実装」の辞書。"""
    g = globals()
    return {name: g[name] for name in EX_NAMES}


if __name__ == "__main__":
    print("(模範解答を exercises.grade で採点します)\n")
    ok = grade(solution_functions())
    # 模範解答は全 PASS が期待値。念のため確認（失敗してもプロセスは exit 0 のまま）。
    print("\n模範解答チェック:", "全問 PASS を確認" if ok else "想定外: 一部 FAIL")
