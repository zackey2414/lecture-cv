"""第14回 章末ミニプロジェクト — 「分類モデル評価レポート」を一枚で作る総合課題。

この回で学んだ評価指標（混同行列・precision/recall/F1・macro/micro/weighted・top-k・
ROC-AUC・PR-AUC=AP）を一つに統合し、実務でそのまま使える「モデル評価レポート」を生成する。
さらに本講座の他スクリプトには無い“マスター要素”を3つ足してある:

  (1) モデル選択   : 同じ正解データに対する「強い分類器」と「弱い分類器」を、
                     accuracy だけで選ぶと不均衡で誤る → macro-F1 / AP で選ぶ、を数値で示す。
  (2) しきい値最適化: 二値サブタスクで F1 最大のしきい値と Youden-J 最大のしきい値を求め、
                     「目的が違えば最適しきい値も違う」を体感する（既定 0.5 と比較）。
  (3) 信頼区間     : ブートストラップ法で macro-F1 の 95% 信頼区間を出し、
                     「点推定の 1 値だけで優劣を断じない」評価の作法を学ぶ。

実装方針:
  - digit 始まりの既存スクリプト（01_*.py 等）は import できないので、必要な計算は
    本ファイル内に自己完結で書く（共通ヘルパ eval_helpers だけは借用してよい）。
  - 自作した numpy の値は必ず scikit-learn と突き合わせて一致を assert する。
  - 完全 CPU・合成データ・数秒で完走。matplotlib は Agg、図のタイトルは ASCII。

実行: uv run python lectures/14_eval_classification/mini_project.py
結果（ダッシュボード画像・レポート JSON）は lectures/14_eval_classification/outputs/ に保存される。
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

# headless 環境でも図を保存できるよう、pyplot より前に Agg を固定する。
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from sklearn.metrics import (  # noqa: E402
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from eval_helpers import (  # noqa: E402
    CLASS_NAMES,
    make_binary_scores,
    make_multiclass_scores,
    output_dir,
)


# =====================================================================
# 自己完結の評価コア（numpy だけで指標を組み立てる。01/02 の再実装だが import 不可なので独立）
# =====================================================================

def safe_div(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """0除算を 0 に倒す割り算。予測ゼロのクラスで precision が 0/0 になるのを 0 に揃える。"""
    return np.divide(num, den, out=np.zeros_like(num, dtype=float), where=den > 0)


def confusion_matrix_manual(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> np.ndarray:
    """混同行列 C[i, j] = 正解 i を j と予測した件数。np.add.at で散布加算する。"""
    cm = np.zeros((k, k), dtype=int)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


def per_class_prf(cm: np.ndarray) -> dict[str, np.ndarray]:
    """混同行列から one-vs-rest の TP/FP/FN と precision/recall/F1 をクラス別に出す。"""
    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0) - tp  # 列 - 対角 = 偽陽性
    fn = cm.sum(axis=1) - tp  # 行 - 対角 = 偽陰性
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> float:
    """macro-F1（クラス別 F1 の単純平均）。ブートストラップで何度も呼ぶので軽量に書く。"""
    cm = confusion_matrix_manual(y_true, y_pred, k)
    return float(per_class_prf(cm)["f1"].mean())


def evaluate_classifier(name: str, y_true: np.ndarray, proba: np.ndarray) -> dict:
    """1つの多クラス分類器について、評価指標一式を自作で計算し sklearn と突き合わせる。"""
    k = proba.shape[1]
    y_pred = proba.argmax(axis=1)
    cm = confusion_matrix_manual(y_true, y_pred, k)
    m = per_class_prf(cm)

    accuracy = float((y_true == y_pred).mean())
    f1 = m["f1"]
    support = cm.sum(axis=1).astype(float)  # 各クラスの正解件数（行和）

    macro = float(f1.mean())                          # クラス対等
    weighted = float(np.average(f1, weights=support))  # 頻度で重み付け
    # micro-F1（多クラス単一ラベルでは accuracy と一致する）
    tp_all, fp_all, fn_all = m["tp"].sum(), m["fp"].sum(), m["fn"].sum()
    micro = float(2 * tp_all / (2 * tp_all + fp_all + fn_all))

    # top-2 accuracy（上位2予測に正解が含まれる割合）
    top2 = float((np.argsort(-proba, axis=1)[:, :2] == y_true[:, None]).any(axis=1).mean())

    # しきい値非依存の指標（OvR を macro 平均）。one-hot 正解 × 確率で AP を出す。
    onehot = np.eye(k)[y_true]
    macro_auroc = float(roc_auc_score(y_true, proba, multi_class="ovr", average="macro",
                                      labels=list(range(k))))
    macro_ap = float(average_precision_score(onehot, proba, average="macro"))

    # --- 自作値を sklearn と突き合わせ（式を理解している証拠）---
    assert np.isclose(macro, f1_score(y_true, y_pred, average="macro", zero_division=0))
    assert np.isclose(weighted, f1_score(y_true, y_pred, average="weighted", zero_division=0))
    assert np.isclose(micro, f1_score(y_true, y_pred, average="micro", zero_division=0))

    return {
        "name": name,
        "confusion_matrix": cm,
        "per_class_f1": f1,
        "support": support,
        "accuracy": accuracy,
        "macro_f1": macro,
        "micro_f1": micro,
        "weighted_f1": weighted,
        "top2_accuracy": top2,
        "macro_auroc": macro_auroc,
        "macro_ap": macro_ap,
    }


# =====================================================================
# (1) モデル選択 — accuracy で選ぶと不均衡で誤る
# =====================================================================

def compare_models(strong: dict, weak: dict) -> dict:
    """強い分類器と弱い分類器を比較し、「何を基準に選ぶか」で結論が変わることを示す。

    accuracy は多数派に張るだけで上がるため、不均衡では信頼しにくい。
    少数クラスも対等に見る macro-F1 / macro-AP こそ、この種のデータでの正しい物差し。
    """
    winner_by_acc = strong["name"] if strong["accuracy"] >= weak["accuracy"] else weak["name"]
    winner_by_f1 = strong["name"] if strong["macro_f1"] >= weak["macro_f1"] else weak["name"]

    print("=== (1) モデル選択: 何を基準にするかで結論が変わる ===")
    print(f"  {'metric':14s} {strong['name']:>10s} {weak['name']:>10s}")
    for key, label in [("accuracy", "accuracy"), ("macro_f1", "macro-F1"),
                       ("weighted_f1", "weighted-F1"), ("macro_ap", "macro-AP")]:
        print(f"  {label:14s} {strong[key]:>10.4f} {weak[key]:>10.4f}")
    print(f"  → accuracy で選ぶと: {winner_by_acc}")
    print(f"  → macro-F1 で選ぶと: {winner_by_f1}")
    print("  不均衡データでは accuracy 単独で優劣を断じない。少数クラスを対等に見る "
          "macro-F1 / macro-AP を主指標にする。\n")
    return {"winner_by_accuracy": winner_by_acc, "winner_by_macro_f1": winner_by_f1}


# =====================================================================
# (2) しきい値最適化 — 目的が違えば最適しきい値も違う
# =====================================================================

def optimize_threshold(y_true: np.ndarray, score: np.ndarray) -> dict:
    """二値スコアを掃引し、F1 最大のしきい値と Youden-J 最大のしきい値を求める。

    スコア降順に「上位 i 件を陽性」と切っていき、TP/FP を累積すれば全しきい値を一気に評価できる。
      - F1 最適   : precision と recall のバランスが最良の点（陽性を当てたい運用向け）。
      - Youden-J  : J = TPR - FPR が最大の点（ROC 上で対角線から最も離れる点）。
    両者は一般に一致しない。既定の 0.5 とも比べて「しきい値は目的で決める」ことを掴む。
    """
    order = np.argsort(-score)
    s = score[order]
    t = y_true[order].astype(float)
    n_pos = t.sum()
    n_neg = len(t) - n_pos

    tp = np.cumsum(t)        # 上位 i 件を陽性としたときの TP
    fp = np.cumsum(1 - t)    # 同 FP
    precision = tp / (tp + fp)
    recall = tp / n_pos
    f1 = safe_div(2 * precision * recall, precision + recall)
    tpr = tp / n_pos
    fpr = fp / n_neg
    youden = tpr - fpr

    i_f1 = int(np.argmax(f1))     # F1 最大の切り位置
    i_j = int(np.argmax(youden))  # Youden-J 最大の切り位置

    # 自作の最良 F1 が sklearn の PR 曲線から得た最良 F1 と一致することを確認する。
    p_sk, r_sk, _ = precision_recall_curve(y_true, score)
    f1_sk = safe_div(2 * p_sk * r_sk, p_sk + r_sk)
    assert np.isclose(f1[i_f1], f1_sk.max(), atol=1e-6), (f1[i_f1], f1_sk.max())

    # 既定 0.5 での参考値
    pred_half = (score >= 0.5).astype(int)
    f1_half = float(f1_score(y_true, pred_half, zero_division=0))

    result = {
        "f1_optimal": {"threshold": float(s[i_f1]), "f1": float(f1[i_f1]),
                       "precision": float(precision[i_f1]), "recall": float(recall[i_f1])},
        "youden_optimal": {"threshold": float(s[i_j]), "youden_j": float(youden[i_j]),
                           "tpr": float(tpr[i_j]), "fpr": float(fpr[i_j])},
        "default_half": {"threshold": 0.5, "f1": f1_half},
        "_curve": {"recall": recall, "precision": precision, "f1": f1,
                   "thresholds": s, "fpr": fpr, "tpr": tpr},
    }
    print("=== (2) しきい値最適化: 目的が違えば最適しきい値も違う ===")
    fo, jo = result["f1_optimal"], result["youden_optimal"]
    print(f"  F1 最適      : thr={fo['threshold']:.3f}  F1={fo['f1']:.3f} "
          f"(P={fo['precision']:.3f}, R={fo['recall']:.3f})")
    print(f"  Youden-J 最適: thr={jo['threshold']:.3f}  J={jo['youden_j']:.3f} "
          f"(TPR={jo['tpr']:.3f}, FPR={jo['fpr']:.3f})")
    print(f"  既定 0.5     : F1={f1_half:.3f}  ← 最適とは限らない")
    print("  F1 を上げたいのか、誤受入(FPR)を抑えたいのか——運用目的でしきい値を選ぶ。\n")
    return result


# =====================================================================
# (3) ブートストラップ信頼区間 — 点推定の1値で優劣を断じない
# =====================================================================

def bootstrap_macro_f1_ci(y_true: np.ndarray, y_pred: np.ndarray, k: int,
                          n_boot: int = 500, seed: int = 123) -> dict:
    """サンプルを復元抽出して macro-F1 を n_boot 回計算し、95% 信頼区間を求める。

    評価セットが有限である以上、指標値には“ブレ”がある。点推定 1 個だけ見て
    0.001 の差で勝った負けたを言わないために、区間で実力を語る作法を身につける。
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    stats = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)  # 復元抽出（同じサイズ・重複あり）
        stats[b] = macro_f1(y_true[idx], y_pred[idx], k)
    lo, hi = np.percentile(stats, [2.5, 97.5])
    point = macro_f1(y_true, y_pred, k)
    print("=== (3) ブートストラップ 95% 信頼区間（macro-F1）===")
    print(f"  点推定 macro-F1 = {point:.4f}   95% CI = [{lo:.4f}, {hi:.4f}]  (n_boot={n_boot})")
    print("  CI が重なる2モデルは“誤差の範囲で同等”。区間で語るのがマスターの作法。\n")
    return {"point": float(point), "ci_low": float(lo), "ci_high": float(hi),
            "samples": stats}


# =====================================================================
# ダッシュボード（6 パネル）
# =====================================================================

def plot_dashboard(strong: dict, weak: dict, thr: dict, boot: dict,
                   y_bin: np.ndarray, s_bin: np.ndarray, path: pathlib.Path) -> None:
    """評価レポートを 1 枚に集約する。タイトルは ASCII のみ。"""
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.0))

    # (a) 強モデルの混同行列
    ax = axes[0, 0]
    cm = strong["confusion_matrix"]
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(CLASS_NAMES)), CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title("Confusion matrix (strong)")
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)

    # (b) クラス別 F1（強 vs 弱）
    ax = axes[0, 1]
    x = np.arange(len(CLASS_NAMES))
    ax.bar(x - 0.2, strong["per_class_f1"], width=0.4, label="strong")
    ax.bar(x + 0.2, weak["per_class_f1"], width=0.4, label="weak")
    ax.set_xticks(x, CLASS_NAMES, rotation=20)
    ax.set_ylim(0, 1.05); ax.set_ylabel("F1")
    ax.set_title("Per-class F1: strong vs weak"); ax.legend(fontsize=8)

    # (c) ROC（二値サブタスク）+ Youden-J 点
    ax = axes[0, 2]
    fpr, tpr, _ = roc_curve(y_bin, s_bin)
    ax.plot(fpr, tpr, label=f"AUC={roc_auc_score(y_bin, s_bin):.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    jo = thr["youden_optimal"]
    ax.scatter([jo["fpr"]], [jo["tpr"]], color="red", zorder=5, label="Youden-J opt")
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title("ROC (binary subtask)"); ax.legend(loc="lower right", fontsize=8)

    # (d) PR（二値サブタスク）+ F1 最適点 + ベースライン
    ax = axes[1, 0]
    prec, rec, _ = precision_recall_curve(y_bin, s_bin)
    ax.plot(rec, prec, label=f"AP={average_precision_score(y_bin, s_bin):.3f}")
    base = float(y_bin.mean())
    ax.axhline(base, color="gray", ls=":", lw=1, label=f"baseline={base:.2f}")
    fo = thr["f1_optimal"]
    ax.scatter([fo["recall"]], [fo["precision"]], color="red", zorder=5, label="F1 opt")
    ax.set_xlabel("recall"); ax.set_ylabel("precision")
    ax.set_title("PR (binary subtask)"); ax.legend(loc="lower left", fontsize=8)

    # (e) しきい値 vs F1
    ax = axes[1, 1]
    cv = thr["_curve"]
    ax.plot(cv["thresholds"], cv["f1"])
    ax.axvline(fo["threshold"], color="red", ls="--", lw=1,
               label=f"F1 opt thr={fo['threshold']:.2f}")
    ax.axvline(0.5, color="gray", ls=":", lw=1, label="default 0.5")
    ax.set_xlabel("threshold"); ax.set_ylabel("F1")
    ax.set_title("Threshold vs F1"); ax.legend(fontsize=8)

    # (f) ブートストラップ分布 + 95% CI
    ax = axes[1, 2]
    ax.hist(boot["samples"], bins=30, color="C0", alpha=0.8)
    ax.axvline(boot["point"], color="black", lw=1.5, label="point est.")
    ax.axvline(boot["ci_low"], color="red", ls="--", lw=1, label="95% CI")
    ax.axvline(boot["ci_high"], color="red", ls="--", lw=1)
    ax.set_xlabel("macro-F1"); ax.set_ylabel("count")
    ax.set_title("Bootstrap macro-F1 (strong)"); ax.legend(fontsize=8)

    fig.suptitle("Module 14 mini project: classification evaluation report", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=120)
    plt.close(fig)


# =====================================================================
# main
# =====================================================================

def main() -> None:
    out = output_dir()
    k = len(CLASS_NAMES)

    # 同じ正解 y_true に対し、信号の強さだけ変えた2モデル（seed 固定なので y_true は共通）。
    y_true, proba_strong = make_multiclass_scores(n=600, n_classes=k, signal=2.2, seed=0)
    _, proba_weak = make_multiclass_scores(n=600, n_classes=k, signal=0.7, seed=0)

    strong = evaluate_classifier("strong", y_true, proba_strong)
    weak = evaluate_classifier("weak", y_true, proba_weak)

    selection = compare_models(strong, weak)

    # 二値サブタスク（しきい値最適化用）。不均衡（陽性 20%）。
    y_bin, s_bin = make_binary_scores(n=2000, pos_ratio=0.2, gap=1.3, seed=3)
    thr = optimize_threshold(y_bin, s_bin)

    # 強モデルの macro-F1 をブートストラップで区間推定。
    boot = bootstrap_macro_f1_ci(y_true, proba_strong.argmax(axis=1), k)

    # ダッシュボード保存
    plot_dashboard(strong, weak, thr, boot, y_bin, s_bin, out / "mini_project_dashboard.png")

    # レポート JSON（_curve や生サンプルなど描画専用データは落として保存）
    def clean(d: dict) -> dict:
        return {kk: (vv.tolist() if isinstance(vv, np.ndarray) else vv)
                for kk, vv in d.items() if not kk.startswith("_")}

    report = {
        "models": {
            "strong": {kk: clean({kk: vv})[kk] for kk, vv in strong.items()},
            "weak": {kk: clean({kk: vv})[kk] for kk, vv in weak.items()},
        },
        "selection": selection,
        "threshold_optimization": clean(thr),
        "bootstrap_macro_f1": {kk: vv for kk, vv in boot.items() if kk != "samples"},
    }
    (out / "mini_project_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2)
    )
    print("完了。mini_project_dashboard.png と mini_project_report.json を "
          "lectures/14_eval_classification/outputs/ に保存しました。")


if __name__ == "__main__":
    main()
