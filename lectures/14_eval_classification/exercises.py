"""第14回 演習問題（分類の評価指標 — 混同行列・PRF・top-k・ROC-AUC）。

使い方:
  1. 各 exN_*() の中の TODO を自分で実装する（最初は NotImplementedError で FAIL になる）。
  2. 自己採点を実行（未実装でも例外で落ちず、PASS/FAIL を一覧表示して必ず正常終了する）:
         uv run python lectures/14_eval_classification/exercises.py
  3. どうしても分からない時は、模範解答の挙動を見る:
         SHOW_SOLUTION=1 uv run python lectures/14_eval_classification/exercises.py

狙い: 指標は「ライブラリを呼ぶ」前に「自分で式を書ける」ことが大事。
      ここでは混同行列・precision/recall/F1・top-k・ROC-AUC を numpy だけで組み立てる。
"""

from __future__ import annotations

import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from eval_helpers import make_binary_scores, make_multiclass_scores  # noqa: E402


# =====================================================================
# 演習（ここを実装する）
# =====================================================================

def ex1_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> np.ndarray:
    """演習1: 混同行列 C[i, j] =「正解 i を j と予測した件数」を (k, k) int 配列で返す。

    ヒント: cm = np.zeros((k, k), int) を作り、np.add.at(cm, (y_true, y_pred), 1)。
    """
    # TODO: 上のヒントに従って混同行列を組み立てて返す
    raise NotImplementedError


def ex2_precision_recall_f1(cm: np.ndarray, c: int) -> tuple[float, float, float]:
    """演習2: 混同行列 cm からクラス c の (precision, recall, f1) を返す。

    TP=cm[c,c], FP=列cの合計-TP, FN=行cの合計-TP。
    precision=TP/(TP+FP), recall=TP/(TP+FN), f1=2PR/(P+R)。分母0のときは 0.0 を返す。
    """
    # TODO: TP/FP/FN を出し、precision/recall/f1 を計算して float タプルで返す（0除算は 0.0）
    raise NotImplementedError


def ex3_macro_f1(cm: np.ndarray) -> float:
    """演習3: 混同行列 cm から macro-F1（全クラスの F1 の単純平均）を返す。

    ヒント: ex2 を各クラスに適用して f1 を集め、その平均を返す。
    """
    # TODO: 各クラスの f1 を集めて平均を返す
    raise NotImplementedError


def ex4_top_k_accuracy(proba: np.ndarray, y_true: np.ndarray, k: int) -> float:
    """演習4: top-k accuracy（各行の上位 k 予測に正解ラベルが含まれる割合）を返す。

    ヒント: np.argsort(-proba, axis=1)[:, :k] が上位 k のクラス番号。
            それが y_true[:, None] と一致する行があるかを any で見て平均する。
    """
    # TODO: 上位 k 予測に正解が入っている割合（float）を返す
    raise NotImplementedError


def ex5_roc_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    """演習5: 二値の ROC-AUC を自作で返す（sklearn.roc_auc_score と一致させる）。

    やり方は2通り。どちらでもよい:
      (A) スコア降順に並べて (FPR, TPR) を打点し、台形則 np.trapezoid(tpr, fpr)。
      (B) Mann-Whitney 流: 全スコアに順位を付け、陽性の順位和から
          AUC = (R_pos - n_pos*(n_pos+1)/2) / (n_pos*n_neg)。
    連続スコアなら同点はほぼ無いので (A) の素朴版で sklearn とほぼ一致する。
    """
    # TODO: ROC-AUC を自作で計算して float で返す
    raise NotImplementedError


# =====================================================================
# 自己採点ランナー（未実装でも例外で落とさず PASS/FAIL を表示。必ず exit 0）
# =====================================================================

def _grade() -> None:
    from sklearn.metrics import (
        confusion_matrix,
        f1_score,
        precision_recall_fscore_support,
        roc_auc_score,
        top_k_accuracy_score,
    )

    k = 4
    y_true, proba = make_multiclass_scores(n=400, n_classes=k, signal=2.0, seed=7)
    y_pred = proba.argmax(axis=1)
    yb, sb = make_binary_scores(n=1500, pos_ratio=0.2, gap=1.3, seed=7)
    cm_ref = confusion_matrix(y_true, y_pred, labels=list(range(k)))

    results: list[tuple[str, bool, str]] = []

    def check(name: str, fn) -> None:
        try:
            ok, detail = fn()
            results.append((name, bool(ok), detail))
        except NotImplementedError:
            results.append((name, False, "未実装（TODO を埋めてください）"))
        except Exception as e:  # noqa: BLE001
            results.append((name, False, f"例外: {type(e).__name__}: {e}"))

    check("ex1_confusion_matrix", lambda: (
        np.array_equal(np.asarray(ex1_confusion_matrix(y_true, y_pred, k)), cm_ref),
        "混同行列が sklearn と一致",
    ))

    def _c2():
        P, R, F, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=list(range(k)), average=None, zero_division=0
        )
        p, r, f = ex2_precision_recall_f1(cm_ref, 1)
        return (np.isclose(p, P[1]) and np.isclose(r, R[1]) and np.isclose(f, F[1]),
                "クラス1の P/R/F1 が sklearn と一致")
    check("ex2_precision_recall_f1", _c2)

    check("ex3_macro_f1", lambda: (
        np.isclose(ex3_macro_f1(cm_ref), f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro-F1 が sklearn と一致",
    ))
    check("ex4_top_k_accuracy", lambda: (
        np.isclose(ex4_top_k_accuracy(proba, y_true, 2),
                   top_k_accuracy_score(y_true, proba, k=2, labels=list(range(k)))),
        "top-2 accuracy が sklearn と一致",
    ))
    check("ex5_roc_auc", lambda: (
        np.isclose(ex5_roc_auc(yb, sb), roc_auc_score(yb, sb), atol=1e-6),
        "ROC-AUC が sklearn と一致",
    ))

    print("=== 採点結果 ===")
    all_ok = True
    for name, ok, detail in results:
        mark = "PASS" if ok else "FAIL"
        all_ok = all_ok and ok
        print(f"  [{mark}] {name:24s} {detail}")
    print("\nALL PASS 🎉" if all_ok else "\nまだ未達の演習があります。TODO を埋めましょう。")


# =====================================================================
# 模範解答（SHOW_SOLUTION=1 のときに本体へ差し替える）。まずは自力で！
# =====================================================================

def _sol_ex1(y_true, y_pred, k):
    cm = np.zeros((k, k), dtype=int)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


def _sol_ex2(cm, c):
    tp = float(cm[c, c])
    fp = float(cm[:, c].sum() - tp)
    fn = float(cm[c, :].sum() - tp)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def _sol_ex3(cm):
    f1s = [_sol_ex2(cm, c)[2] for c in range(cm.shape[0])]
    return float(np.mean(f1s))


def _sol_ex4(proba, y_true, k):
    topk = np.argsort(-proba, axis=1)[:, :k]
    return float((topk == y_true[:, None]).any(axis=1).mean())


def _sol_ex5(y_true, score):
    order = np.argsort(-score)
    t = y_true[order]
    n_pos = t.sum()
    n_neg = len(t) - n_pos
    tp = np.cumsum(t)
    fp = np.cumsum(1 - t)
    tpr = np.r_[0.0, tp / n_pos]
    fpr = np.r_[0.0, fp / n_neg]
    return float(np.trapezoid(tpr, fpr))


def _install_solutions() -> None:
    g = globals()
    g["ex1_confusion_matrix"] = _sol_ex1
    g["ex2_precision_recall_f1"] = _sol_ex2
    g["ex3_macro_f1"] = _sol_ex3
    g["ex4_top_k_accuracy"] = _sol_ex4
    g["ex5_roc_auc"] = _sol_ex5


if __name__ == "__main__":
    if os.environ.get("SHOW_SOLUTION") == "1":
        print("(模範解答モードで実行します)\n")
        _install_solutions()
    _grade()
