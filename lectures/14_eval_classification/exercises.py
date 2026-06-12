"""第14回 演習問題（分類の評価指標 — 混同行列・PRF・top-k・ROC-AUC・AP・しきい値）。

使い方:
  1. 各 exN_*() の中の TODO を自分で実装する（最初は NotImplementedError で FAIL になる）。
  2. 自己採点を実行（未実装でも例外で落ちず、PASS/FAIL を一覧表示して必ず正常終了する）:
         uv run python lectures/14_eval_classification/exercises.py
  3. どうしても分からない時は、模範解答（全 PASS）を見る:
         uv run python lectures/14_eval_classification/exercises_solutions.py
     もしくは exercises.py を模範解答で採点（採点ロジックを共有して確認）:
         SHOW_SOLUTION=1 uv run python lectures/14_eval_classification/exercises.py

狙い: 指標は「ライブラリを呼ぶ」前に「自分で式を書ける」ことが大事。
      ここでは混同行列・precision/recall/F1・top-k・ROC-AUC・AP・しきい値最適化を
      numpy だけで組み立て、scikit-learn の値と一致させて検証する。

難易度の目安:
  ex1〜ex3 : 易（混同行列と PRF の定義そのもの）
  ex4〜ex6 : 中（top-k・ROC-AUC・weighted 平均）
  ex7〜ex8 : 難（AP=ステップ和・F1 を最大化するしきい値の探索）
"""

from __future__ import annotations

import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from eval_helpers import make_binary_scores, make_multiclass_scores  # noqa: E402


# =====================================================================
# 演習（ここを実装する）。各関数は自己完結で、ヒントの式どおり numpy で書けばよい。
# =====================================================================

def ex1_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> np.ndarray:
    """演習1（易）: 混同行列 C[i, j] =「正解 i を j と予測した件数」を (k, k) int 配列で返す。

    ヒント: cm = np.zeros((k, k), int) を作り、np.add.at(cm, (y_true, y_pred), 1)。
    """
    # TODO: 上のヒントに従って混同行列を組み立てて返す
    raise NotImplementedError


def ex2_precision_recall_f1(cm: np.ndarray, c: int) -> tuple[float, float, float]:
    """演習2（易）: 混同行列 cm からクラス c の (precision, recall, f1) を返す。

    TP=cm[c,c], FP=列cの合計-TP, FN=行cの合計-TP。
    precision=TP/(TP+FP), recall=TP/(TP+FN), f1=2PR/(P+R)。分母0のときは 0.0 を返す。
    """
    # TODO: TP/FP/FN を出し、precision/recall/f1 を計算して float タプルで返す（0除算は 0.0）
    raise NotImplementedError


def ex3_macro_f1(cm: np.ndarray) -> float:
    """演習3（易）: 混同行列 cm から macro-F1（全クラスの F1 の単純平均）を返す。

    ヒント: ex2 を各クラスに適用して f1 を集め、その平均を返す。
    """
    # TODO: 各クラスの f1 を集めて平均を返す
    raise NotImplementedError


def ex4_top_k_accuracy(proba: np.ndarray, y_true: np.ndarray, k: int) -> float:
    """演習4（中）: top-k accuracy（各行の上位 k 予測に正解ラベルが含まれる割合）を返す。

    ヒント: np.argsort(-proba, axis=1)[:, :k] が上位 k のクラス番号。
            それが y_true[:, None] と一致する行があるかを any で見て平均する。
    """
    # TODO: 上位 k 予測に正解が入っている割合（float）を返す
    raise NotImplementedError


def ex5_roc_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    """演習5（中）: 二値の ROC-AUC を自作で返す（sklearn.roc_auc_score と一致させる）。

    やり方は2通り。どちらでもよい:
      (A) スコア降順に並べて (FPR, TPR) を打点し、台形則 np.trapezoid(tpr, fpr)。
      (B) Mann-Whitney 流: 全スコアに順位を付け、陽性の順位和から
          AUC = (R_pos - n_pos*(n_pos+1)/2) / (n_pos*n_neg)。
    連続スコアなら同点はほぼ無いので (A) の素朴版で sklearn とほぼ一致する。
    """
    # TODO: ROC-AUC を自作で計算して float で返す
    raise NotImplementedError


def ex6_weighted_f1(cm: np.ndarray) -> float:
    """演習6（中）: 混同行列 cm から weighted-F1 を返す。

    weighted は「各クラスの F1 を、そのクラスの support（=正解件数=行和）で重み付け平均」したもの。
    ヒント: support_c = cm[c, :].sum()。weighted_f1 = Σ_c support_c * f1_c / Σ_c support_c。
            f1_c は ex2 と同じ式で出せる（0除算は 0.0）。
    """
    # TODO: 各クラスの f1 と support を出し、support 重み付き平均を返す
    raise NotImplementedError


def ex7_average_precision(y_true: np.ndarray, score: np.ndarray) -> float:
    """演習7（難）: 二値の AP（Average Precision=PR-AUC）を自作で返す。

    重要: AP は PR 曲線を“台形”ではなく“短冊（ステップ）和”で積む:
        AP = Σ_i (recall_i - recall_{i-1}) * precision_i
    手順: スコア降順に並べ、TP/FP を累積して recall=TP/n_pos, precision=TP/(TP+FP) を作り、
          recall の増分 × その時点の precision を足し合わせる（sklearn.average_precision_score と一致）。
    """
    # TODO: ステップ和で AP を計算して float で返す
    raise NotImplementedError


def ex8_best_f1_threshold(y_true: np.ndarray, score: np.ndarray) -> tuple[float, float]:
    """演習8（難）: F1 を最大化するしきい値とそのときの F1 を (threshold, f1) で返す。

    手順: スコア降順に並べ「上位 i 件を陽性」と切っていくと、TP/FP の累積から
          各切り位置の precision/recall→F1 が一気に出せる。F1 最大の位置の
          (そのときのスコア値, F1) を返す。pred = (score >= threshold) で再現できる切り方にする。
    ヒント: order=np.argsort(-score); s=score[order]; t=y_true[order]
            tp=np.cumsum(t); fp=np.cumsum(1-t); precision=tp/(tp+fp); recall=tp/t.sum()
            i=np.argmax(f1) として (float(s[i]), float(f1[i]))。
    """
    # TODO: F1 最大のしきい値と F1 を返す
    raise NotImplementedError


# 採点対象の関数名（exercises_solutions.py からも参照する）。
EX_NAMES = [
    "ex1_confusion_matrix",
    "ex2_precision_recall_f1",
    "ex3_macro_f1",
    "ex4_top_k_accuracy",
    "ex5_roc_auc",
    "ex6_weighted_f1",
    "ex7_average_precision",
    "ex8_best_f1_threshold",
]


# =====================================================================
# 自己採点ランダ（未実装でも例外で落とさず PASS/FAIL を表示。必ず exit 0）
#   grade(funcs) は「関数名 -> 実装」の辞書を受け取って採点する。
#   exercises.py（TODO 版）も exercises_solutions.py（模範解答）も、この同じ grade を呼ぶ。
# =====================================================================

def _make_dataset() -> dict:
    """採点用の固定データ（seed 固定で再現する）。"""
    k = 4
    y_true, proba = make_multiclass_scores(n=400, n_classes=k, signal=2.0, seed=7)
    y_pred = proba.argmax(axis=1)
    yb, sb = make_binary_scores(n=1500, pos_ratio=0.2, gap=1.3, seed=7)
    return {"k": k, "y_true": y_true, "proba": proba, "y_pred": y_pred, "yb": yb, "sb": sb}


def grade(funcs: dict) -> bool:
    """funcs（名前->実装）を採点し、PASS/FAIL を表示して all_ok を返す。

    grading は exercises 側にのみ存在し、模範解答ファイルもこれを再利用する（重複なし）。
    """
    from sklearn.metrics import (
        average_precision_score,
        confusion_matrix,
        f1_score,
        precision_recall_curve,
        precision_recall_fscore_support,
        roc_auc_score,
        top_k_accuracy_score,
    )

    d = _make_dataset()
    k, y_true, proba, y_pred, yb, sb = d["k"], d["y_true"], d["proba"], d["y_pred"], d["yb"], d["sb"]
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
        np.array_equal(np.asarray(funcs["ex1_confusion_matrix"](y_true, y_pred, k)), cm_ref),
        "混同行列が sklearn と一致",
    ))

    def _c2():
        P, R, F, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=list(range(k)), average=None, zero_division=0
        )
        p, r, f = funcs["ex2_precision_recall_f1"](cm_ref, 1)
        return (np.isclose(p, P[1]) and np.isclose(r, R[1]) and np.isclose(f, F[1]),
                "クラス1の P/R/F1 が sklearn と一致")
    check("ex2_precision_recall_f1", _c2)

    check("ex3_macro_f1", lambda: (
        np.isclose(funcs["ex3_macro_f1"](cm_ref),
                   f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro-F1 が sklearn と一致",
    ))
    check("ex4_top_k_accuracy", lambda: (
        np.isclose(funcs["ex4_top_k_accuracy"](proba, y_true, 2),
                   top_k_accuracy_score(y_true, proba, k=2, labels=list(range(k)))),
        "top-2 accuracy が sklearn と一致",
    ))
    check("ex5_roc_auc", lambda: (
        np.isclose(funcs["ex5_roc_auc"](yb, sb), roc_auc_score(yb, sb), atol=1e-6),
        "ROC-AUC が sklearn と一致",
    ))
    check("ex6_weighted_f1", lambda: (
        np.isclose(funcs["ex6_weighted_f1"](cm_ref),
                   f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "weighted-F1 が sklearn と一致",
    ))
    check("ex7_average_precision", lambda: (
        np.isclose(funcs["ex7_average_precision"](yb, sb),
                   average_precision_score(yb, sb), atol=1e-6),
        "AP(PR-AUC) が sklearn と一致",
    ))

    def _c8():
        thr, f1 = funcs["ex8_best_f1_threshold"](yb, sb)
        # 参照: sklearn の PR 曲線から得た最良 F1
        p_sk, r_sk, _ = precision_recall_curve(yb, sb)
        f1_sk = np.divide(2 * p_sk * r_sk, p_sk + r_sk,
                          out=np.zeros_like(p_sk), where=(p_sk + r_sk) > 0)
        # 返した f1 が最良 F1 と一致し、かつ threshold を当てはめて再現できるか
        pred = (sb >= thr).astype(int)
        f1_at_thr = f1_score(yb, pred, zero_division=0)
        return (np.isclose(f1, f1_sk.max(), atol=1e-6) and np.isclose(f1, f1_at_thr, atol=1e-6),
                "F1 最大のしきい値と F1 が sklearn と一致")
    check("ex8_best_f1_threshold", _c8)

    print("=== 採点結果 ===")
    all_ok = True
    for name, ok, detail in results:
        mark = "PASS" if ok else "FAIL"
        all_ok = all_ok and ok
        print(f"  [{mark}] {name:26s} {detail}")
    print("\nALL PASS 🎉" if all_ok else "\nまだ未達の演習があります。TODO を埋めましょう。")
    return all_ok


def current_functions() -> dict:
    """このファイル内の exN 実装（最初は TODO）を辞書で集める。"""
    g = globals()
    return {name: g[name] for name in EX_NAMES}


if __name__ == "__main__":
    if os.environ.get("SHOW_SOLUTION") == "1":
        # 模範解答を別ファイルから読み込み、同じ grade で採点する（採点ロジックを共有）。
        print("(模範解答モードで実行します — exercises_solutions.py の実装を採点)\n")
        import exercises_solutions  # noqa: E402

        grade(exercises_solutions.solution_functions())
    else:
        grade(current_functions())
