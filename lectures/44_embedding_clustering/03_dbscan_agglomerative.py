"""03: k を決めない仲間たち — DBSCAN と AgglomerativeClustering。

学ぶこと:
  - k-means は k を先に決める必要があった。**DBSCAN** と **Agglomerative（凝集型）** は
    k を指定せず、「どこまで近ければ同じ束か」というしきい値で自動的にクラスタ数を決める。
  - **DBSCAN**: 密度ベース。eps 近傍に min_samples 点以上あれば芯（core）とみなして束を広げ、
    どの束にも入らない点を **ノイズ(-1)** にする。外れ値に強いが、**eps が全体で 1 つ**しかなく、
    密度がグループ間で違うと途端に難しくなる（本章でその限界を実演する）。
  - **Agglomerative**: 近いペアから順に併合する木（デンドログラム）を作り、distance_threshold で
    切る。average linkage + cosine 距離は埋め込みで扱いやすく、密度の偏りに DBSCAN より強い。
  - 距離構造（グループ内距離 vs グループ間距離）を覗くと、なぜ DBSCAN が苦戦し Agglomerative が
    綺麗に決まるのかが腑に落ちる。

実行:
  uv run python lectures/44_embedding_clustering/03_dbscan_agglomerative.py
"""

from __future__ import annotations

import json

import numpy as np
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances

import cluster_lab as cl


def distance_structure(emb: np.ndarray, labels: np.ndarray) -> dict:
    """グループ内 / グループ間のコサイン距離の要約を返す（eps を決める勘所）。"""
    d = pairwise_distances(emb, metric="cosine")
    iu, ju = np.triu_indices(len(labels), k=1)
    same = labels[iu] == labels[ju]
    intra, inter = d[iu, ju][same], d[iu, ju][~same]
    return {
        "intra_max": float(intra.max()),
        "intra_mean": float(intra.mean()),
        "inter_min": float(inter.min()),
        "inter_mean": float(inter.mean()),
    }


def dbscan_sweep(emb: np.ndarray, labels: np.ndarray, eps_values: list[float]) -> list[dict]:
    """eps を振って DBSCAN（cosine）を回し、各設定の要約を集める。"""
    rows = []
    for eps in eps_values:
        pred = DBSCAN(eps=eps, min_samples=3, metric="cosine").fit_predict(emb)
        rep = cl.clustering_report(pred, emb, labels)
        rows.append({"eps": eps, **rep})
    return rows


def main() -> None:
    out = cl.ensure_out_dir()
    device = cl.get_device()

    images, labels, names = cl.make_image_collection(n_per_group=6, seed=0)
    emb = cl.embed_images(images, device)
    true_k = len(names)

    # --- 距離構造を覗く: eps をどこに置けば良いかの直感 ---
    ds = distance_structure(emb, labels)
    print("コサイン距離の構造（L2 正規化済み埋め込み）:")
    print(f"  グループ内: max={ds['intra_max']:.3f}  mean={ds['intra_mean']:.3f}")
    print(f"  グループ間: min={ds['inter_min']:.3f}  mean={ds['inter_mean']:.3f}")
    print(f"  → 内側の最大({ds['intra_max']:.3f}) と 外側の最小({ds['inter_min']:.3f}) が重なるほど")
    print("    単一 eps での分離は難しい（DBSCAN の弱点が出る場面）\n")

    # --- DBSCAN: eps 掃引。小さいとノイズだらけ、大きいと全部 1 つに溶ける ---
    print("[DBSCAN] eps 掃引（min_samples=3, cosine）")
    print("  eps  | clusters | noise | NMI")
    rows = dbscan_sweep(emb, labels, [0.03, 0.05, 0.06, 0.07, 0.08, 0.10])
    for r in rows:
        print(f"  {r['eps']:.2f} |    {r['n_clusters_found']}     |   {r['n_noise']:2d}  | {r['nmi']:.3f}")
    best_db = max(rows, key=lambda r: r["nmi"])
    print(f"  最良 eps={best_db['eps']:.2f}: clusters={best_db['n_clusters_found']} "
          f"noise={best_db['n_noise']} NMI={best_db['nmi']:.3f}")
    print(f"  → 真の {true_k} に届く eps はごく狭い窓だけ（0.03→6, 0.05→5, 0.07→4 と急変）。")
    print("    しかもその eps でも 1 点がノイズに落ちる＝単一 eps の調整がシビア\n")

    # --- Agglomerative: distance_threshold で切る。密度の偏りに強い ---
    dt = 0.06
    pred_ag = AgglomerativeClustering(
        n_clusters=None, distance_threshold=dt, metric="cosine", linkage="average"
    ).fit_predict(emb)
    rep_ag = cl.clustering_report(pred_ag, emb, labels)
    print(f"[Agglomerative] distance_threshold={dt}（cosine, average linkage）")
    print(f"  clusters={rep_ag['n_clusters_found']}  silhouette={rep_ag['silhouette']:.3f}  "
          f"NMI={rep_ag['nmi']:.3f}  purity={rep_ag['purity']:.3f}")
    print("  → 同じデータでも threshold で切るだけで真のグループ数に綺麗に届く\n")

    # --- 参考: k=true_k の k-means と並べる ---
    pred_km = KMeans(n_clusters=true_k, n_init=10, random_state=0).fit_predict(emb)
    rep_km = cl.clustering_report(pred_km, emb, labels)

    # --- 可視化: 3 手法の PCA 2D を並べて保存 + Agglomerative のアルバム ---
    xy = PCA(n_components=2).fit_transform(emb)
    cl.scatter_2d(xy, pred_ag, labels,
                  f"Agglomerative (dt={dt}): color=cluster, marker=true group",
                  out / "03_agglomerative_scatter.png")
    cl.save_cluster_album(images, pred_ag, out / "03_album_agglomerative.png")

    summary = {
        "distance_structure": ds,
        "dbscan_sweep": rows,
        "agglomerative": {"distance_threshold": dt, **rep_ag},
        "kmeans_reference": {"k": true_k, **rep_km},
    }
    (out / "03_dbscan_agglo.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))

    print("使い分けのまとめ:")
    print("  - k が事前に分かる/球状で素直 → k-means（速くて安定）")
    print("  - 外れ値・ノイズを弾きたい/任意形状 → DBSCAN（ただし eps 調整が難しい）")
    print("  - k を決めず、密度の偏りに強く、しきい値で切りたい → Agglomerative（埋め込みで定番）")
    print(f"\n完了。lectures/{cl.MODULE_ID}/outputs/03_*.png と 03_dbscan_agglo.json を確認してください。")


if __name__ == "__main__":
    main()
