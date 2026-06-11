"""02: 異常検知の評価 — image/pixel の AUROC・AUPR・PRO と、PaDiM vs PatchCore 比較。

学ぶこと:
  - **AUROC**(ROC 下面積) と **AUPR**(PR 下面積=average precision) の違い。
      * AUROC は正例/負例のバランスに鈍感で、画素評価のように欠陥画素がごく少数だと
        「高く出やすい」。AUPR は正例が希少なほど厳しくなり、実態（小欠陥の取りこぼし）を映す。
      * だから異常検知では image/pixel の AUROC と AUPR を **併用**する。
  - **PRO**(Per-Region Overlap): 欠陥領域ごとの被覆率を「領域の大きさに依らず平等」に平均。
    大きな欠陥に引きずられず、小さな欠陥も同じ重みで測れる pixel-AUROC の補完。
  - 2 つのメモリバンク型を比較:
      * **PaDiM**   … 位置ごとに正常ガウシアンを持ち、マハラノビス距離で採点（位置合わせ前提）。
      * **PatchCore** … 正常パッチを coreset で記憶し、最近傍距離で採点（位置に縛られにくい）。

実行:
  uv run python lectures/32_anomaly_iqa/02_anomaly_eval.py
出力:
  outputs/32_anomaly_iqa/02_roc_pr.png   … image-level の ROC 曲線と PR 曲線
  outputs/32_anomaly_iqa/02_summary.json … 両手法の各指標
"""

from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve

import anomaly_iqa_lab as lab


def run_padim(ds: dict) -> tuple[np.ndarray, np.ndarray]:
    """PaDiM の画素マップ(N,H,W) と image スコア(N,) を返す。"""
    ext = lab.FeatureExtractor(seed=0)
    etr, ete = ext.embed(ds["train"]), ext.embed(ds["test"])
    idx = lab.select_dims(etr.shape[-1], 100, seed=0)
    model = lab.padim_fit(etr[..., idx])
    maps = lab.upsample_maps(lab.padim_score(ete[..., idx], model))
    return maps, lab.image_scores(maps)


def run_patchcore(ds: dict) -> tuple[np.ndarray, np.ndarray]:
    """PatchCore の画素マップ と image スコアを返す（layer2/3 の粗い格子で軽量化）。"""
    ext = lab.FeatureExtractor(layers=("layer2", "layer3"), seed=0)
    etr, ete = ext.embed(ds["train"]), ext.embed(ds["test"])
    model = lab.patchcore_fit(etr, n_keep=800, seed=0)  # coreset でメモリバンク圧縮
    maps = lab.upsample_maps(lab.patchcore_score(ete, model))
    return maps, lab.image_scores(maps)


def evaluate(name: str, maps: np.ndarray, scores: np.ndarray, ds: dict) -> dict:
    """1 手法ぶんの全指標をまとめて辞書化し、コンソールにも出す。"""
    res = {
        "image_auroc": lab.image_auroc(scores, ds["labels"]),
        "image_aupr": lab.image_aupr(scores, ds["labels"]),
        "pixel_auroc": lab.pixel_auroc(maps, ds["masks"]),
        "pixel_aupr": lab.pixel_aupr(maps, ds["masks"]),
        "pro_auc": lab.pro_auc(maps, ds["masks"]),
    }
    print(
        f"  [{name:9s}] img AUROC={res['image_auroc']:.3f} AUPR={res['image_aupr']:.3f} | "
        f"px AUROC={res['pixel_auroc']:.3f} AUPR={res['pixel_aupr']:.3f} | PRO={res['pro_auc']:.3f}"
    )
    return res


def plot_curves(results: dict, ds: dict, scores_map: dict, path: str) -> None:
    """image-level の ROC と PR を 2 手法ぶん重ねて描く。"""
    y = ds["labels"]
    fig, (axr, axp) = plt.subplots(1, 2, figsize=(11, 4.5))
    for name, scores in scores_map.items():
        fpr, tpr, _ = roc_curve(y, scores)
        prec, rec, _ = precision_recall_curve(y, scores)
        axr.plot(fpr, tpr, label=f"{name} (AUROC={results[name]['image_auroc']:.3f})")
        axp.plot(rec, prec, label=f"{name} (AUPR={results[name]['image_aupr']:.3f})")
    axr.plot([0, 1], [0, 1], "k--", lw=0.8)
    axr.set_xlabel("FPR")
    axr.set_ylabel("TPR")
    axr.set_title("ROC (image-level)")
    axr.legend(loc="lower right")
    axp.axhline(float(y.mean()), color="k", ls="--", lw=0.8, label="chance (base rate)")
    axp.set_xlabel("Recall")
    axp.set_ylabel("Precision")
    axp.set_title("Precision-Recall (image-level)")
    axp.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> None:
    out = lab.ensure_out_dir()
    print("=" * 64)
    print("02 異常検知の評価（AUROC vs AUPR vs PRO / PaDiM vs PatchCore）")

    ds = lab.make_dataset()
    print(f"  評価 {len(ds['test'])} 枚（image 異常率 {ds['labels'].mean():.2f}）")
    px_rate = float(ds["masks"].mean())
    print(f"  画素の欠陥率 = {px_rate:.4f} ← 非常に不均衡。AUPR が AUROC より厳しく出る理由。")

    p_maps, p_scores = run_padim(ds)
    c_maps, c_scores = run_patchcore(ds)
    results = {
        "PaDiM": evaluate("PaDiM", p_maps, p_scores, ds),
        "PatchCore": evaluate("PatchCore", c_maps, c_scores, ds),
    }

    curve_path = os.path.join(out, "02_roc_pr.png")
    plot_curves(results, ds, {"PaDiM": p_scores, "PatchCore": c_scores}, curve_path)
    json_path = os.path.join(out, "02_summary.json")
    with open(json_path, "w") as f:
        json.dump(
            {"pixel_defect_rate": px_rate, "metrics": results}, f, ensure_ascii=False, indent=2
        )

    print(f"  保存: {curve_path}")
    print(f"  保存: {json_path}")
    print(
        "  読み方: pixel-AUROC は高く見えても pixel-AUPR は低い → 小欠陥の取りこぼしは AUPR/PRO で点検する。"
    )


if __name__ == "__main__":
    main()
