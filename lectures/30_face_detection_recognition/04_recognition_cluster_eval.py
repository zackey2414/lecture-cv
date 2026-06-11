"""04: 認識とクラスタリングの評価 — 指標を一望し、しきい値を定量的に選ぶ。

学ぶこと:
  - 認識(1:1)の指標: ROC-AUC / EER / TAR@FAR をまとめて算出。1:N の rank-1 も併記。
  - クラスタリングの指標: purity / NMI / homogeneity / completeness の意味と相補性。
      * homogeneity: 各クラスタが単一人物で占められているか（混ざってないか）。
      * completeness: 同一人物が 1 クラスタにまとまっているか（割れてないか）。
      * NMI: 上 2 つのバランス（情報量ベース）。クラスタ数に左右されにくい総合指標。
      * purity: 直感的だがクラスタを増やすほど上がる（過分割に甘い）ので単独では使わない。
  - DBSCAN の eps（と Agglomerative のしきい値）を **掃引** し、NMI を最大化する動作点を
    データから選ぶ。1 点 1 クラスタ（過分割）も全部 1 クラスタ（過併合）も NMI が下がる谷で
    挟まれた山が「ちょうど良い」しきい値、という読み方を身につける。
  - 結果は JSON にまとめて保存（再現・比較用）。

実行:
  uv run python lectures/30_face_detection_recognition/04_recognition_cluster_eval.py
"""

from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import roc_auc_score

import face_lab as fl


def recognition_metrics(emb: np.ndarray, labels: np.ndarray) -> dict:
    """1:1 照合 + 1:N 識別の主要指標をまとめて返す。"""
    scores, is_genuine = fl.verification_scores(emb, labels)
    thrs, far, tar = fl.roc_far_tar(scores, is_genuine)
    eer, eer_thr = fl.compute_eer(far, tar, thrs)
    tar1, _ = fl.tar_at_far(far, tar, thrs, 0.01)
    tar3, _ = fl.tar_at_far(far, tar, thrs, 0.001)
    auc = float(roc_auc_score(is_genuine.astype(int), scores))  # genuine を正例に AUC

    # 1:N: 各人物の先頭 1 枚を登録、残りをプローブに
    is_gallery = np.zeros(len(labels), dtype=bool)
    for pid in np.unique(labels):
        is_gallery[np.where(labels == pid)[0][0]] = True
    cmc = fl.identification_cmc(
        emb[is_gallery], labels[is_gallery], emb[~is_gallery], labels[~is_gallery], max_rank=5
    )
    return {
        "roc_auc": auc,
        "eer": eer,
        "eer_threshold": eer_thr,
        "tar_at_far_1e-2": tar1,
        "tar_at_far_1e-3": tar3,
        "rank1": float(cmc[0]),
        "rank5": float(cmc[-1]),
    }


def sweep_dbscan_eps(
    emb_norm: np.ndarray, labels: np.ndarray, eps_grid: np.ndarray
) -> tuple[list[dict], dict]:
    """eps を掃引し、各点の (eps, NMI, purity, homogeneity, n_clusters) を集める。

    返り値は (全結果リスト, NMI 最大の点)。NMI 同点なら推定人数が真値に近い方を優先。
    """
    results = []
    for eps in eps_grid:
        pred = DBSCAN(eps=float(eps), min_samples=2, metric="cosine").fit_predict(emb_norm)
        rep = fl.clustering_report(labels, pred)
        rep["eps"] = float(eps)
        results.append(rep)
    n_true = len(np.unique(labels))
    best = max(results, key=lambda r: (round(r["nmi"], 4), -abs(r["n_clusters_found"] - n_true)))
    return results, best


def plot_eps_sweep(results: list[dict], n_true: int, best_eps: float, path: str) -> None:
    """eps に対する NMI / purity / homogeneity と推定人数の変化を 2 軸で描く。"""
    eps = [r["eps"] for r in results]
    fig, ax1 = plt.subplots(figsize=(6.2, 4.4))
    ax1.plot(eps, [r["nmi"] for r in results], "-o", color="tab:blue", label="NMI")
    ax1.plot(eps, [r["purity"] for r in results], "-s", color="tab:green", label="purity")
    ax1.plot(
        eps, [r["homogeneity"] for r in results], "-^", color="tab:orange", label="homogeneity"
    )
    ax1.axvline(best_eps, ls="--", color="tab:red", label=f"best eps={best_eps:.3f}")
    ax1.set_xlabel("DBSCAN eps (cosine distance)")
    ax1.set_ylabel("score")
    ax1.set_ylim(0, 1.05)
    ax1.legend(loc="lower center", fontsize=8)

    ax2 = ax1.twinx()  # 推定人数は別軸
    ax2.plot(eps, [r["n_clusters_found"] for r in results], ":x", color="gray", label="#clusters")
    ax2.axhline(n_true, ls=":", color="black", alpha=0.5)
    ax2.set_ylabel("#clusters found (gray)")
    ax1.set_title("DBSCAN eps sweep (NMI peak = good threshold)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> None:
    os.makedirs(fl.OUT_DIR, exist_ok=True)
    device = fl.get_device()

    images, labels, names = fl.make_face_dataset(n_ids=8, n_per_id=6, seed=0)
    n_true = len(names)
    enc = fl.FaceEncoder(device)
    emb_raw = enc.encode(images)
    emb_norm = fl.l2_normalize(emb_raw)
    print(f"合成顔: {len(images)} 枚 / {n_true} 人  (埋め込み mode={enc.mode})\n")

    print("[認識の指標]")
    rec = recognition_metrics(emb_raw, labels)
    print(f"  ROC-AUC      = {rec['roc_auc']:.4f}")
    print(f"  EER          = {rec['eer']:.4f}")
    print(f"  TAR@FAR=1e-2 = {rec['tar_at_far_1e-2']:.4f}")
    print(f"  TAR@FAR=1e-3 = {rec['tar_at_far_1e-3']:.4f}")
    print(f"  rank-1 / rank-5 = {rec['rank1']:.4f} / {rec['rank5']:.4f}")

    print("\n[クラスタリングの指標: DBSCAN eps 掃引]")
    eps_grid = np.round(np.linspace(0.02, 0.15, 14), 3)
    results, best = sweep_dbscan_eps(emb_norm, labels, eps_grid)
    print("   eps   #clu  noise  purity   NMI    homo")
    for r in results:
        mark = "  <- best" if r["eps"] == best["eps"] else ""
        print(
            f"  {r['eps']:.3f}  {r['n_clusters_found']:>3}  {r['n_noise']:>4}   "
            f"{r['purity']:.3f}  {r['nmi']:.3f}  {r['homogeneity']:.3f}{mark}"
        )
    print(
        f"  → NMI 最大の eps={best['eps']:.3f} で 推定人数={best['n_clusters_found']}（真値 {n_true}）"
    )
    plot_eps_sweep(results, n_true, best["eps"], os.path.join(fl.OUT_DIR, "04_eps_sweep.png"))

    # まとめて JSON 保存（再現・比較用）
    out = {
        "encoder_mode": enc.mode,
        "n_identities": n_true,
        "n_images": len(images),
        "recognition": rec,
        "clustering_best_dbscan": best,
        "clustering_sweep": results,
    }
    json_path = os.path.join(fl.OUT_DIR, "04_metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"  指標を {json_path} に保存しました。")

    print("\n[プライバシ・バイアスの注意]")
    print("  - 顔データは最も機微な個人情報。収集は同意・目的限定・保持期間・削除権を前提に。")
    print("    本講座は合成顔のみを使い、実在人物のデータは扱わない。")
    print("  - 顔認識の精度は肌色・年齢・性別で偏りやすい（学習データの不均衡が主因）。")
    print("    集計を 1 つの精度に丸めず、属性ごとに TAR@FAR/EER を分解して公平性を点検する。")
    print("  - しきい値は『社会的コスト』で決まる: 誤受入(FAR)が危険な入退管理は FAR を極小に、")
    print("    アルバム整理は多少の誤りより快適さ（TAR）を優先、と用途で動作点を変える。")
    print(
        f"\n完了。outputs/{fl.MODULE_ID}/04_eps_sweep.png と 04_metrics.json を確認してください。"
    )


if __name__ == "__main__":
    main()
