"""01: 混同行列と TP/FP/FN/TN から precision/recall/F1/accuracy を自作し、sklearn と突き合わせる。

学ぶこと:
  - 多クラスの混同行列 C[i, j] =「正解が i で、予測が j だった件数」を numpy で組み立てる。
  - 各クラスを one-vs-rest（そのクラス vs それ以外）に分解すると TP/FP/FN/TN が読める:
        TP = 対角 C[c,c]、FP = 列c の対角以外、FN = 行c の対角以外、TN = 残り全部。
  - precision = TP/(TP+FP), recall = TP/(TP+FN), F1 = 2PR/(P+R) を定義どおり計算。
  - macro（クラス単純平均）/ micro（全件まとめ＝多クラスでは accuracy）/ weighted（頻度重み）平均の違い。
  - top-k accuracy（上位 k 予測の中に正解が入る割合）。
  - クラス不均衡だと accuracy が「多数派に張るだけ」で高く出てしまう罠。

実行: uv run python lectures/14_eval_classification/01_confusion_matrix_prf.py
結果（混同行列の図・指標 JSON）は lectures/14_eval_classification/outputs/ に保存される。
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

# headless でも図を保存できるよう、pyplot を import する前に Agg バックエンドに固定する。
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from sklearn.metrics import (  # noqa: E402
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    top_k_accuracy_score,
)

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from eval_helpers import CLASS_NAMES, make_multiclass_scores, output_dir  # noqa: E402


def safe_div(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """0除算を 0 に倒す割り算。

    予測が1件も出なかったクラスは TP+FP=0 になり precision が 0/0 になる。
    sklearn も zero_division=0 で 0 を返すので、ここも 0 に揃えておくと値が一致する。
    """
    return np.divide(num, den, out=np.zeros_like(num, dtype=float), where=den > 0)


def confusion_matrix_manual(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> np.ndarray:
    """混同行列を自作する。C[i, j] = 正解 i を j と予測した件数。

    np.add.at は (行, 列) の組ごとに +1 する“散布加算”。ループより速く、意図も明快。
    """
    cm = np.zeros((k, k), dtype=int)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


def per_class_prf(cm: np.ndarray) -> dict[str, np.ndarray]:
    """混同行列から one-vs-rest の TP/FP/FN/TN と precision/recall/F1 をクラスごとに出す。"""
    tp = np.diag(cm).astype(float)  # 対角＝正しく当てた件数
    fp = cm.sum(axis=0) - tp  # 列方向の合計 - 対角 ＝ そのクラスと誤って予測した件数
    fn = cm.sum(axis=1) - tp  # 行方向の合計 - 対角 ＝ そのクラスを取りこぼした件数
    tn = cm.sum() - tp - fp - fn  # 残り全部
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": precision, "recall": recall, "f1": f1}


def plot_confusion(cm: np.ndarray, names: list[str], path: pathlib.Path) -> None:
    """混同行列をヒートマップとして保存する（対角が濃いほど良い分類器）。"""
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(names)), names, rotation=45, ha="right")
    ax.set_yticks(range(len(names)), names)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title("Confusion matrix")
    # 各マスに件数を書き込む（背景が濃いマスは白字にして読めるようにする）
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def demo_accuracy_trap() -> dict[str, float]:
    """クラス不均衡で accuracy が誤誘導する例を数値で見せる。

    陽性がたった10%しかないデータで「常に陰性(0)と予測する」だけの“何もしない分類器”は、
    accuracy 0.90 を叩き出す。だが陽性の recall は 0、F1 も 0。accuracy だけ見ると騙される。
    """
    from eval_helpers import make_binary_scores

    y_true, score = make_binary_scores(n=2000, pos_ratio=0.1, gap=1.2, seed=1)
    prevalence = float(y_true.mean())  # 陽性の割合（=多数派に張ったときの誤差の余地）

    # (A) 常に陰性と予測する“怠け者”分類器
    pred_lazy = np.zeros_like(y_true)
    acc_lazy = float((pred_lazy == y_true).mean())
    # 陽性クラスの recall（取りこぼし。怠け者は1件も当てないので 0）
    rec_lazy = float(((pred_lazy == 1) & (y_true == 1)).sum() / max((y_true == 1).sum(), 1))

    # (B) スコア0.5でしきい値を切る、ちゃんと中身のある分類器
    pred_real = (score >= 0.5).astype(int)
    acc_real = float((pred_real == y_true).mean())
    rec_real = float(((pred_real == 1) & (y_true == 1)).sum() / max((y_true == 1).sum(), 1))

    print("\n[不均衡での accuracy の罠] 陽性割合 = {:.1%}".format(prevalence))
    print(f"  常に陰性と予測  : accuracy={acc_lazy:.3f}  陽性recall={rec_lazy:.3f}  ← 高精度に見えるが無能")
    print(f"  しきい値0.5の分類: accuracy={acc_real:.3f}  陽性recall={rec_real:.3f}  ← recall を見て初めて差が出る")
    print("  教訓: 不均衡では accuracy 単独を信じない。per-class の recall / F1 を必ず併記する。")
    return {"prevalence": prevalence, "acc_lazy": acc_lazy, "rec_lazy": rec_lazy,
            "acc_real": acc_real, "rec_real": rec_real}


def main() -> None:
    out = output_dir()
    k = len(CLASS_NAMES)
    y_true, proba = make_multiclass_scores(n=600, n_classes=k, signal=2.0, seed=0)
    y_pred = proba.argmax(axis=1)  # 予測ラベルは確率最大のクラス

    # === 1. 混同行列: 自作 vs sklearn ===================================
    cm_manual = confusion_matrix_manual(y_true, y_pred, k)
    cm_sklearn = confusion_matrix(y_true, y_pred, labels=list(range(k)))
    assert np.array_equal(cm_manual, cm_sklearn), "混同行列が sklearn と一致しません"
    print("[1] 混同行列（自作と sklearn は一致）:\n", cm_manual)

    # === 2. 混同行列から precision/recall/F1 を自作 → sklearn と一致確認 ====
    m = per_class_prf(cm_manual)
    P, R, F, S = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(k)), average=None, zero_division=0
    )
    assert np.allclose(m["precision"], P) and np.allclose(m["recall"], R) and np.allclose(m["f1"], F)
    print("\n[2] クラス別 precision/recall/F1（自作＝sklearn）:")
    for c, name in enumerate(CLASS_NAMES):
        print(f"  {name:5s} P={m['precision'][c]:.3f} R={m['recall'][c]:.3f} "
              f"F1={m['f1'][c]:.3f} (support={int(S[c])})")

    # === 3. macro / micro / weighted 平均の違い ==========================
    # macro: クラスを単純平均（少数クラスも対等に効く）
    macro_f1 = float(m["f1"].mean())
    # micro: 全クラスの TP/FP/FN をまとめてから1つの式に入れる。多クラス単一ラベルでは accuracy と一致。
    tp_all, fp_all, fn_all = m["tp"].sum(), m["fp"].sum(), m["fn"].sum()
    micro_p = tp_all / (tp_all + fp_all)
    micro_r = tp_all / (tp_all + fn_all)
    micro_f1 = float(2 * micro_p * micro_r / (micro_p + micro_r))
    accuracy = float((y_true == y_pred).mean())
    # weighted: 各クラスの F1 を support（出現頻度）で重み付け平均
    weighted_f1 = float(np.average(m["f1"], weights=S))

    sk_macro = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)[2]
    sk_micro = precision_recall_fscore_support(y_true, y_pred, average="micro", zero_division=0)[2]
    sk_weighted = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)[2]
    assert np.isclose(macro_f1, sk_macro) and np.isclose(micro_f1, sk_micro) and np.isclose(weighted_f1, sk_weighted)
    print("\n[3] F1 の平均化（自作＝sklearn）:")
    print(f"  macro   F1 = {macro_f1:.4f} （クラス単純平均: 少数クラスも対等）")
    print(f"  micro   F1 = {micro_f1:.4f} （全件まとめ＝accuracy {accuracy:.4f}）")
    print(f"  weighted F1 = {weighted_f1:.4f} （頻度で重み付け）")

    # === 4. top-k accuracy（上位k予測に正解が含まれる割合）==================
    # argsort(-proba) で確率降順に並べ、上位 k 列に正解ラベルが入っているか判定する。
    topk = np.argsort(-proba, axis=1)[:, :2]
    top2_manual = float((topk == y_true[:, None]).any(axis=1).mean())
    top2_sklearn = float(top_k_accuracy_score(y_true, proba, k=2, labels=list(range(k))))
    assert np.isclose(top2_manual, top2_sklearn)
    print(f"\n[4] top-2 accuracy = {top2_manual:.4f}（自作＝sklearn）。top-1 = {accuracy:.4f}")

    # === 5. sklearn の classification_report（実務での定番サマリ）==========
    report = classification_report(
        y_true, y_pred, labels=list(range(k)), target_names=CLASS_NAMES, zero_division=0
    )
    print("\n[5] classification_report:\n", report)

    # === 6. クラス不均衡での accuracy の罠 ================================
    trap = demo_accuracy_trap()

    # === 7. 図と JSON を保存 ============================================
    plot_confusion(cm_manual, CLASS_NAMES, out / "01_confusion_matrix.png")
    summary = {
        "confusion_matrix": cm_manual.tolist(),
        "per_class": {
            CLASS_NAMES[c]: {
                "precision": float(m["precision"][c]),
                "recall": float(m["recall"][c]),
                "f1": float(m["f1"][c]),
                "support": int(S[c]),
            }
            for c in range(k)
        },
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "weighted_f1": weighted_f1,
        "top2_accuracy": top2_manual,
        "imbalance_trap": trap,
    }
    (out / "01_metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\n完了。lectures/14_eval_classification/outputs/ に 01_confusion_matrix.png と 01_metrics.json を保存しました。")


if __name__ == "__main__":
    main()
