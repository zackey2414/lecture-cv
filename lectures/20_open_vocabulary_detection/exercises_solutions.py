"""第20回 演習の模範解答（全 PASS）。

このファイルは exercises.py の演習 ex1〜ex8 の“正解実装”を集め、
exercises.py 側の grade()（採点ロジック）をそのまま再利用して全問 PASS を確認する。
採点の二重定義を避けるため、ここには「解答」だけを置き、判定は exercises.grade に委譲する。

実行: uv run python lectures/20_open_vocabulary_detection/exercises_solutions.py
  → 8問すべて PASS することを確認できる（まずは自力で exercises.py を解いてから見ること）。
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from exercises import EX_NAMES, grade  # noqa: E402  採点ロジックは exercises 側を再利用


def _iou(box_a, box_b) -> float:
    """2 box [x1,y1,x2,y2] の IoU。ex1 の素直な実装で、他の解答からも使い回す。"""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return float(inter / union) if union > 0 else 0.0


def ex1_iou(box_a: list[float], box_b: list[float]) -> float:
    return _iou(box_a, box_b)


def ex2_cxcywh_norm_to_xyxy(box_norm: list[float], height: int, width: int) -> list[float]:
    cx, cy, w, h = box_norm
    cx_a, w_a = cx * width, w * width  # x 方向は width
    cy_a, h_a = cy * height, h * height  # y 方向は height
    return [cx_a - w_a / 2, cy_a - h_a / 2, cx_a + w_a / 2, cy_a + h_a / 2]


def ex3_greedy_match(pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels, iou_thr=0.5):
    order = np.argsort(-np.asarray(pred_scores))
    matched: set[int] = set()
    tp = fp = 0
    for i in order:
        cand = [
            g for g in range(len(gt_boxes)) if g not in matched and gt_labels[g] == pred_labels[i]
        ]
        if not cand:
            fp += 1
            continue
        ious = [_iou(list(pred_boxes[i]), list(gt_boxes[g])) for g in cand]
        j = int(np.argmax(ious))
        if ious[j] >= iou_thr:
            matched.add(cand[j])
            tp += 1
        else:
            fp += 1
    fn = len(gt_boxes) - len(matched)
    return tp, fp, fn


def ex4_prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1


def ex5_labels_to_caption(labels: list[str]) -> str:
    parts = [lab.strip().rstrip(".").strip().lower() for lab in labels]
    parts = [p for p in parts if p]
    return " ".join(f"{p}." for p in parts)


def ex6_iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    a = np.asarray(boxes_a, dtype=float)
    b = np.asarray(boxes_b, dtype=float)
    out = np.zeros((len(a), len(b)), dtype=float)
    for i in range(len(a)):
        for j in range(len(b)):
            out[i, j] = _iou(a[i], b[j])
    return out


def ex7_nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float = 0.5) -> list[int]:
    boxes = np.asarray(boxes, dtype=float)
    order = list(np.argsort(-np.asarray(scores)))
    keep: list[int] = []
    while order:
        i = int(order.pop(0))  # 残りのうち最高スコア
        keep.append(i)
        # 採用 box と重なりが大きい候補を捨てる
        order = [j for j in order if _iou(boxes[i], boxes[j]) < iou_thr]
    return keep


def ex8_average_precision(scores: np.ndarray, is_tp: np.ndarray, n_gt: int) -> float:
    scores = np.asarray(scores, dtype=float)
    is_tp = np.asarray(is_tp, dtype=float)
    if n_gt == 0 or is_tp.sum() == 0:
        return 0.0
    order = np.argsort(-scores)
    tp = is_tp[order]
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(1.0 - tp)
    recall = tp_cum / n_gt
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)
    # 全点（連続）補間: 番兵を付けて precision を右から単調非増加に均す。
    mrec = np.concatenate(([0.0], recall, [recall[-1]]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]  # recall が増えた点だけ面積を足す
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def solution_functions() -> dict:
    """exercises.grade に渡す「名前 -> 解答実装」の辞書。"""
    g = globals()
    return {name: g[name] for name in EX_NAMES}


if __name__ == "__main__":
    print("(模範解答を exercises.grade で採点します)\n")
    ok = grade(solution_functions())
    # 模範解答は全 PASS が期待値（失敗してもプロセスは exit 0 のまま）。
    print("\n模範解答チェック:", "全問 PASS を確認" if ok else "想定外: 一部 FAIL")
