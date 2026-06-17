"""02: しきい値非依存の指標 — ROC 曲線/ROC-AUC と PR 曲線/PR-AUC(=AP) を自作し sklearn と突き合わせる。

ここまでの precision/recall は「あるしきい値で予測を 0/1 に固めた後」の指標だった。
だが分類器が出すのは連続スコアで、しきい値をどこに置くかで precision/recall は動く。
そこで「しきい値を全部揃いしたときの挙動」を1つの曲線・1つの面積にまとめるのが ROC と PR。

学ぶこと:
  - ROC 曲線: しきい値を下げながら (FPR, TPR) を打点。下の面積 = ROC-AUC。台形則で自作。
        TPR = TP/(TP+FN)（＝recall）, FPR = FP/(FP+TN)。
  - PR 曲線: 同じ揃いで (recall, precision) を打点。AP（Average Precision）= ステップ和で自作。
        sklearn の average_precision_score は台形ではなく「短冊（ステップ）和」である点が重要。
  - 不均衡の効き方: 同じ“分離度”でも、陽性が珍しくなると ROC-AUC はほぼ不変なのに
        PR-AUC(AP) は下がる。PR のベースライン（無情報な分類器の AP）は陽性割合そのもの。

実行: uv run python lectures/14_eval_classification/02_roc_pr_auc.py
結果（ROC/PR 曲線の図・AUC/AP の JSON）は lectures/14_eval_classification/outputs/ に保存される。
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from sklearn.metrics import (  # noqa: E402
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from eval_helpers import make_binary_scores, output_dir  # noqa: E402


def roc_curve_manual(y_true: np.ndarray, score: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """ROC 曲線 (fpr, tpr) と ROC-AUC を自作する。

    手順:
      1. スコア降順に並べる（しきい値を高→低に下げていくのに相当）。
      2. 上から1件ずつ「陽性と見なす」を増やし、TP/FP を累積する。
      3. しきい値が変わる位置（同点でない境目）だけ打点して階段を作る。
      4. (0,0) と (1,1) を端点に足し、台形則で面積を取れば ROC-AUC。
    roc_auc_score は台形則（auc(fpr, tpr)）なので、自作の台形則とほぼ完全一致する。
    """
    order = np.argsort(-score)  # スコア降順
    s = score[order]
    t = y_true[order]
    n_pos = t.sum()
    n_neg = len(t) - n_pos

    tp_cum = np.cumsum(t)  # ここまでで陽性を正しく拾った数
    fp_cum = np.cumsum(1 - t)  # ここまでで陰性を誤って陽性にした数

    # 同じスコアの塊は同時にしきい値を跨ぐので、スコアが変わる位置だけ残す
    distinct = np.where(np.diff(s) != 0)[0]
    idx = np.r_[distinct, len(s) - 1]
    tpr = np.r_[0.0, tp_cum[idx] / n_pos]
    fpr = np.r_[0.0, fp_cum[idx] / n_neg]
    auc = float(np.trapezoid(tpr, fpr))  # 台形則（numpy 2.x は trapz 廃止 → trapezoid）
    return fpr, tpr, auc


def pr_curve_manual(y_true: np.ndarray, score: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """PR 曲線 (precision, recall) と AP（Average Precision）を自作する。

    AP は「recall が増えた分 × その時点の precision」を足し合わせたもの（短冊＝ステップ和）:
        AP = Σ_i (recall_i - recall_{i-1}) * precision_i
    これは PR 曲線の下を“台形”ではなく“長方形（右端の高さ）”で積む方式で、
    sklearn.metrics.average_precision_score と一致する（台形の auc(recall, precision) とは別物）。
    """
    order = np.argsort(-score)
    t = y_true[order]
    n_pos = t.sum()

    tp_cum = np.cumsum(t)
    fp_cum = np.cumsum(1 - t)
    recall = tp_cum / n_pos
    precision = tp_cum / (tp_cum + fp_cum)

    # ステップ和（recall の増分 × その時点の precision）。連続スコアなら同点が無く sklearn と一致。
    recall_prev = np.r_[0.0, recall[:-1]]
    ap = float(np.sum((recall - recall_prev) * precision))
    return precision, recall, ap


def evaluate(name: str, y_true: np.ndarray, score: np.ndarray) -> dict:
    """1つのデータセットについて、自作とライブラリの ROC-AUC / AP を計算して突き合わせる。"""
    fpr_m, tpr_m, auc_m = roc_curve_manual(y_true, score)
    prec_m, rec_m, ap_m = pr_curve_manual(y_true, score)

    auc_sk = float(roc_auc_score(y_true, score))
    ap_sk = float(average_precision_score(y_true, score))

    # 自作とライブラリがほぼ一致することを確認（連続スコアなので 1e-6 で十分）
    assert np.isclose(auc_m, auc_sk, atol=1e-6), (auc_m, auc_sk)
    assert np.isclose(ap_m, ap_sk, atol=1e-6), (ap_m, ap_sk)

    prevalence = float(y_true.mean())  # 無情報な分類器（常に同じスコア）の AP はこの値になる
    print(f"[{name}] 陽性割合={prevalence:.1%}  "
          f"ROC-AUC: 自作={auc_m:.4f} / sklearn={auc_sk:.4f}  "
          f"AP: 自作={ap_m:.4f} / sklearn={ap_sk:.4f}（ベースラインAP≈{prevalence:.3f}）")

    # sklearn の曲線（描画用）。roc_curve/precision_recall_curve は同点をまとめて点数を減らす。
    fpr_sk, tpr_sk, _ = roc_curve(y_true, score)
    prec_sk, rec_sk, _ = precision_recall_curve(y_true, score)
    return {
        "prevalence": prevalence,
        "roc_auc_manual": auc_m, "roc_auc_sklearn": auc_sk,
        "ap_manual": ap_m, "ap_sklearn": ap_sk,
        "roc": (fpr_sk, tpr_sk), "pr": (rec_sk, prec_sk),
    }


def main() -> None:
    out = output_dir()

    # 同じ“分離度”(gap=1.4)のまま、陽性割合だけを変える2つのデータを作る。
    # → ROC-AUC はほぼ同じ（順位の指標なので頻度に鈍感）、AP は不均衡で下がる、を観察する。
    y_imb, s_imb = make_binary_scores(n=3000, pos_ratio=0.10, gap=1.4, seed=0)
    y_bal, s_bal = make_binary_scores(n=3000, pos_ratio=0.50, gap=1.4, seed=0)

    print("=== ROC-AUC と PR-AUC(AP): 自作とライブラリの一致、不均衡の効き方 ===")
    res_imb = evaluate("不均衡", y_imb, s_imb)
    res_bal = evaluate("均衡", y_bal, s_bal)
    print("\n要点: ROC-AUC は不均衡でもほぼ不変。一方 AP は陽性が珍しいほど下がり、"
          "そのベースライン（無情報分類器の AP）は陽性割合そのもの。"
          "不均衡タスクで“実感に合う”のは PR-AUC/AP の方。")

    # === ROC 曲線（不均衡 vs 均衡）======================================
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    for res, label in [(res_imb, "imbalanced (10%)"), (res_bal, "balanced (50%)")]:
        fpr, tpr = res["roc"]
        ax.plot(fpr, tpr, label=f"{label}  AUC={res['roc_auc_sklearn']:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="random (AUC=0.5)")
    ax.set_xlabel("FPR = FP/(FP+TN)")
    ax.set_ylabel("TPR = TP/(TP+FN) = recall")
    ax.set_title("ROC curve")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "02_roc_curve.png", dpi=120)
    plt.close(fig)

    # === PR 曲線（不均衡 vs 均衡。ベースライン＝陽性割合の水平線）===========
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    for res, label, color in [(res_imb, "imbalanced (10%)", "C0"), (res_bal, "balanced (50%)", "C1")]:
        rec, prec = res["pr"]
        ax.plot(rec, prec, color=color, label=f"{label}  AP={res['ap_sklearn']:.3f}")
        # 無情報分類器のベースライン（その割合だけ正解する水平線）
        ax.axhline(res["prevalence"], color=color, ls=":", lw=1)
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_title("PR curve (dotted line = baseline AP = prevalence)")
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "02_pr_curve.png", dpi=120)
    plt.close(fig)

    # === JSON 保存 =====================================================
    summary = {
        "imbalanced": {k: res_imb[k] for k in
                       ["prevalence", "roc_auc_manual", "roc_auc_sklearn", "ap_manual", "ap_sklearn"]},
        "balanced": {k: res_bal[k] for k in
                     ["prevalence", "roc_auc_manual", "roc_auc_sklearn", "ap_manual", "ap_sklearn"]},
    }
    (out / "02_roc_pr_auc.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\n完了。02_roc_curve.png / 02_pr_curve.png / 02_roc_pr_auc.json を保存しました。")


if __name__ == "__main__":
    main()
