"""第14回 演習の模範解答（全 PASS）。

このファイルは exercises.py の演習 ex1〜ex8 の“正解実装”を集め、
exercises.py 側の grade()（採点ロジック）をそのまま再利用して全問 PASS を確認する。
採点の二重定義を避けるため、ここには「解答」だけを置き、判定は exercises.grade に委譲する。

実行: uv run python lectures/14_eval_classification/exercises_solutions.py
  → 8問すべて PASS することを確認できる（まずは自力で exercises.py を解いてから見ること）。
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from exercises import EX_NAMES, grade  # noqa: E402  採点ロジックは exercises 側を再利用


def _safe_div(num: float, den: float) -> float:
    """0除算を 0.0 に倒すスカラ割り算。"""
    return num / den if den > 0 else 0.0


def ex1_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> np.ndarray:
    """混同行列 C[i, j] = 正解 i を j と予測した件数。np.add.at で散布加算。"""
    cm = np.zeros((k, k), dtype=int)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


def ex2_precision_recall_f1(cm: np.ndarray, c: int) -> tuple[float, float, float]:
    """クラス c の precision/recall/f1。TP=対角, FP=列-対角, FN=行-対角。"""
    tp = float(cm[c, c])
    fp = float(cm[:, c].sum() - tp)
    fn = float(cm[c, :].sum() - tp)
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    return precision, recall, f1


def ex3_macro_f1(cm: np.ndarray) -> float:
    """各クラスの f1 の単純平均（クラス対等）。"""
    f1s = [ex2_precision_recall_f1(cm, c)[2] for c in range(cm.shape[0])]
    return float(np.mean(f1s))


def ex4_top_k_accuracy(proba: np.ndarray, y_true: np.ndarray, k: int) -> float:
    """上位 k 予測に正解が入っている割合。"""
    topk = np.argsort(-proba, axis=1)[:, :k]
    return float((topk == y_true[:, None]).any(axis=1).mean())


def ex5_roc_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    """スコア降順に (FPR, TPR) を打点し、台形則で ROC-AUC。"""
    order = np.argsort(-score)
    t = y_true[order]
    n_pos = t.sum()
    n_neg = len(t) - n_pos
    tp = np.cumsum(t)
    fp = np.cumsum(1 - t)
    tpr = np.r_[0.0, tp / n_pos]
    fpr = np.r_[0.0, fp / n_neg]
    return float(np.trapezoid(tpr, fpr))


def ex6_weighted_f1(cm: np.ndarray) -> float:
    """各クラスの f1 を support（行和）で重み付け平均。"""
    k = cm.shape[0]
    f1s = np.array([ex2_precision_recall_f1(cm, c)[2] for c in range(k)])
    support = cm.sum(axis=1).astype(float)
    return float(np.average(f1s, weights=support))


def ex7_average_precision(y_true: np.ndarray, score: np.ndarray) -> float:
    """AP=ステップ和: Σ (recall_i - recall_{i-1}) * precision_i。"""
    order = np.argsort(-score)
    t = y_true[order]
    n_pos = t.sum()
    tp = np.cumsum(t)
    fp = np.cumsum(1 - t)
    recall = tp / n_pos
    precision = tp / (tp + fp)
    recall_prev = np.r_[0.0, recall[:-1]]
    return float(np.sum((recall - recall_prev) * precision))


def ex8_best_f1_threshold(y_true: np.ndarray, score: np.ndarray) -> tuple[float, float]:
    """スコア降順に切り、F1 最大の (しきい値, F1) を返す。"""
    order = np.argsort(-score)
    s = score[order]
    t = y_true[order].astype(float)
    tp = np.cumsum(t)
    fp = np.cumsum(1 - t)
    precision = tp / (tp + fp)
    recall = tp / t.sum()
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(precision), where=denom > 0)
    i = int(np.argmax(f1))
    return float(s[i]), float(f1[i])


def solution_functions() -> dict:
    """exercises.grade に渡す「名前 -> 解答実装」の辞書。"""
    g = globals()
    return {name: g[name] for name in EX_NAMES}


if __name__ == "__main__":
    print("(模範解答を exercises.grade で採点します)\n")
    ok = grade(solution_functions())
    # 模範解答は全 PASS が期待値。念のため確認（失敗してもプロセスは exit 0 のまま）。
    print("\n模範解答チェック:", "全問 PASS を確認" if ok else "想定外: 一部 FAIL")
