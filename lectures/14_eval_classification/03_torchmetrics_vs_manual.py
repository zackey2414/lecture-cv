"""03: torchmetrics の update → compute → reset サイクルを学び、自作 / sklearn と三者一致を確認する。

torchmetrics は「学習ループの途中でバッチごとに指標を貯めていき、最後にまとめて確定する」用途に向く。
そのための作法が update → compute → reset:
  - update(preds, target) … ミニバッチを与えて内部状態（TP/FP/… や全スコア）に足し込む。
  - compute()             … 貯めた状態から最終的な指標値を1つ計算して返す。
  - reset()               … 内部状態を空にする。次のエポックの頭で必ず呼ぶ（忘れると累積し続ける）。

学ぶこと:
  - 多クラスの Accuracy / F1 / AUROC / AveragePrecision / ConfusionMatrix を torchmetrics で出す。
  - 2バッチに分けて update しても、まとめて compute すれば全件で計算した値と一致すること。
  - その値が、01・02 で書いた自作実装や sklearn とも（許容誤差内で）一致すること。
  - 完全CPU。preds と target は同じ device に揃える（ここでは両方 cpu）だけ。

実行: uv run python lectures/14_eval_classification/03_torchmetrics_vs_manual.py
結果（三者比較の JSON・棒グラフ）は outputs/14_eval_classification/ に保存される。
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    average_precision_score,
    f1_score,
    roc_auc_score,
)
from torchmetrics.classification import (  # noqa: E402
    BinaryAUROC,
    BinaryAveragePrecision,
    MulticlassAccuracy,
    MulticlassAUROC,
    MulticlassAveragePrecision,
    MulticlassConfusionMatrix,
    MulticlassF1Score,
)

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from eval_helpers import CLASS_NAMES, make_binary_scores, make_multiclass_scores, output_dir  # noqa: E402

# CPU 前提だが、将来 GPU でも動くよう device は自動判定する（本講座の定石）。
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")


def multiclass_block(out: pathlib.Path) -> dict:
    """多クラス: torchmetrics をバッチ更新で回し、自作 / sklearn と突き合わせる。"""
    k = len(CLASS_NAMES)
    y_true, proba = make_multiclass_scores(n=600, n_classes=k, signal=2.0, seed=0)
    y_pred = proba.argmax(axis=1)

    # numpy → torch。preds は確率 (n, K)、target は整数ラベル (n,)。device を揃える。
    preds = torch.tensor(proba, dtype=torch.float32, device=DEVICE)
    target = torch.tensor(y_true, dtype=torch.long, device=DEVICE)

    # torchmetrics のメトリックを用意（しきい値非依存の AUROC/AP は確率を、Accuracy/F1 も確率でOK）。
    acc = MulticlassAccuracy(num_classes=k, average="micro").to(DEVICE)  # micro=全体正解率
    f1 = MulticlassF1Score(num_classes=k, average="macro").to(DEVICE)  # macro=クラス平均
    auroc = MulticlassAUROC(num_classes=k, average="macro").to(DEVICE)  # OvR の AUROC 平均
    ap = MulticlassAveragePrecision(num_classes=k, average="macro").to(DEVICE)
    cm = MulticlassConfusionMatrix(num_classes=k).to(DEVICE)

    # --- update を「2バッチに分けて」呼ぶ。実運用の学習ループを模す ---
    half = len(target) // 2
    for sl in (slice(0, half), slice(half, None)):
        for metric in (acc, f1, auroc, ap, cm):
            metric.update(preds[sl], target[sl])

    # --- compute でまとめて確定 ---
    tm = {
        "accuracy": float(acc.compute()),
        "macro_f1": float(f1.compute()),
        "macro_auroc": float(auroc.compute()),
        "macro_ap": float(ap.compute()),
    }
    cm_tm = cm.compute().cpu().numpy()

    # --- 自作 / sklearn と突き合わせ ---
    onehot = np.eye(k)[y_true]  # AP(macro) は one-hot 正解 × 確率で計算する
    ref = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_auroc": float(roc_auc_score(y_true, proba, multi_class="ovr",
                                           average="macro", labels=list(range(k)))),
        "macro_ap": float(average_precision_score(onehot, proba, average="macro")),
    }
    print("[多クラス] torchmetrics vs sklearn（バッチ更新→compute）:")
    for key in tm:
        same = "OK" if np.isclose(tm[key], ref[key], atol=1e-3) else "DIFF"
        print(f"  {key:12s} tm={tm[key]:.4f}  sklearn={ref[key]:.4f}  [{same}]")
        assert np.isclose(tm[key], ref[key], atol=1e-3), key

    # --- reset の効果を見せる: reset 後に compute すると状態が空でエラー → 再 update で復活 ---
    acc.reset()
    acc.update(preds, target)  # 全件を一度に update しても結果は同じ
    assert np.isclose(float(acc.compute()), ref["accuracy"], atol=1e-6)
    print("  reset→全件 update でも accuracy は同値（更新順・分割に依らない）")

    # 棒グラフ（torchmetrics と sklearn が重なる＝一致）を保存
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    keys = list(tm.keys())
    x = np.arange(len(keys))
    ax.bar(x - 0.2, [tm[k_] for k_ in keys], width=0.4, label="torchmetrics")
    ax.bar(x + 0.2, [ref[k_] for k_ in keys], width=0.4, label="sklearn")
    ax.set_xticks(x, keys, rotation=20)
    ax.set_ylim(0, 1.05)
    ax.set_title("multiclass: torchmetrics vs sklearn")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "03_multiclass_compare.png", dpi=120)
    plt.close(fig)

    return {"torchmetrics": tm, "sklearn": ref, "confusion_matrix": cm_tm.tolist()}


def binary_block() -> dict:
    """二値: しきい値非依存の AUROC / AP を torchmetrics と sklearn で突き合わせる。"""
    y_true, score = make_binary_scores(n=3000, pos_ratio=0.10, gap=1.4, seed=0)
    # 二値はスコアを 1次元で渡す。float64 にしておくと sklearn との差が float32 由来でブレない。
    preds = torch.tensor(score, dtype=torch.float64, device=DEVICE)
    target = torch.tensor(y_true, dtype=torch.long, device=DEVICE)

    auroc = BinaryAUROC().to(DEVICE)
    ap = BinaryAveragePrecision().to(DEVICE)
    # こちらもバッチ更新で貯める
    for i in range(0, len(target), 1000):
        auroc.update(preds[i:i + 1000], target[i:i + 1000])
        ap.update(preds[i:i + 1000], target[i:i + 1000])

    tm = {"auroc": float(auroc.compute()), "ap": float(ap.compute())}
    ref = {"auroc": float(roc_auc_score(y_true, score)), "ap": float(average_precision_score(y_true, score))}
    print("\n[二値] torchmetrics vs sklearn:")
    for key in tm:
        same = "OK" if np.isclose(tm[key], ref[key], atol=1e-4) else "DIFF"
        print(f"  {key:6s} tm={tm[key]:.4f}  sklearn={ref[key]:.4f}  [{same}]")
        assert np.isclose(tm[key], ref[key], atol=1e-4), key
    return {"torchmetrics": tm, "sklearn": ref}


def main() -> None:
    out = output_dir()
    print(f"device = {DEVICE}（評価指標は CPU で完結。preds と target を同じ device に揃えるだけ）\n")
    mc = multiclass_block(out)
    bn = binary_block()
    (out / "03_torchmetrics_vs_manual.json").write_text(
        json.dumps({"multiclass": mc, "binary": bn}, ensure_ascii=False, indent=2)
    )
    print("\n完了。03_multiclass_compare.png と 03_torchmetrics_vs_manual.json を保存しました。")


if __name__ == "__main__":
    main()
