"""01: 画像コレクションを CLIP 埋め込み → k-means で自動グルーピング。

学ぶこと:
  - 検索（16/17）は「クエリに近いものを並べる」。クラスタリングは「**全体を束に分ける**」。
    クエリも正解ラベルも無いまま、似た埋め込み同士をまとめるのが k-means。
  - 手順は 3 つだけ: (1) 画像を CLIP で埋め込み、(2) **L2 正規化**、(3) KMeans でグループ ID を振る。
  - 評価は 2 系統。教師なしの **silhouette**（ラベル不要）と、答え合わせ用の **NMI / purity /
    homogeneity**（既知グループがあるとき）。本章は色×形の既知グループを“評価だけ”に使う。

実行:
  uv run python lectures/44_embedding_clustering/01_kmeans_image_embeddings.py
結果（アルバム・2D 散布図・json）は lectures/44_embedding_clustering/outputs/ に保存。
"""

from __future__ import annotations

import json

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import cluster_lab as cl


def run_kmeans(emb: np.ndarray, k: int, seed: int = 0) -> np.ndarray:
    """L2 正規化済み埋め込みに k-means を当て、各点のクラスタ ID 配列を返す。

    n_init=10: 初期中心をランダムに 10 回振り直し、inertia 最小の結果を採用する
    （k-means は初期値依存なので複数回試すのが定石。sklearn の既定は "auto"）。
    """
    return KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(emb)


def main() -> None:
    out = cl.ensure_out_dir()
    device = cl.get_device()

    # --- 1. 合成画像コレクション（色×形で 6 グループ）。ラベルは評価専用 ---
    images, labels, names = cl.make_image_collection(n_per_group=6, seed=0)
    print(f"画像コレクション: {len(images)} 枚 / 既知 {len(names)} グループ")
    print(f"  グループ: {names}")
    print("  （クラスタリングには『6 グループある』とも『どれがどれ』とも教えない）\n")

    # --- 2. CLIP 埋め込み（L2 正規化込み）---
    emb = cl.embed_images(images, device)
    print(f"CLIP 埋め込み: shape={emb.shape}  各行ノルム≈{np.linalg.norm(emb, axis=1).mean():.3f}")
    print("  ★ L2 正規化済みなので、以降の距離計算は実質コサイン距離になる\n")

    # --- 3. k-means（本当は k を知らないが、ここでは真の数 6 を与える。k 選びは 02 で）---
    k = len(names)
    pred = run_kmeans(emb, k)
    rep = cl.clustering_report(pred, emb, labels)
    print(f"[KMeans k={k}]")
    print(f"  silhouette={rep['silhouette']:.3f}（教師なし。1 に近いほど良い分離）")
    print(f"  NMI={rep['nmi']:.3f}  purity={rep['purity']:.3f}  homogeneity={rep['homogeneity']:.3f}")
    print("  （NMI/purity は既知グループとの一致度。1.0 なら完全に当てている）\n")

    # --- 4. 可視化: クラスタ別アルバム & PCA 2D 散布図 ---
    cl.save_cluster_album(images, pred, out / "01_album_kmeans.png")
    xy = PCA(n_components=2).fit_transform(emb)
    cl.scatter_2d(
        xy, pred, labels, "kmeans on CLIP embeddings (PCA 2D): color=cluster, marker=true group",
        out / "01_pca_scatter.png",
    )

    # --- 5. 数値を json に保存（再現性・後続の比較用）---
    (out / "01_kmeans_report.json").write_text(
        json.dumps({"k": k, **rep, "group_names": names}, ensure_ascii=False, indent=2)
    )

    print("読み取り方:")
    print("  - アルバムの各行が 1 クラスタ。1 行が 1 グループで揃えば自動グルーピング成功。")
    print("  - 散布図で同じ色（予測クラスタ）がまとまり、別の形（真グループ）が分かれていれば理想。")
    print("  - ここでは k=6 を与えたが、実務では k は未知。次の 02 で k の選び方を学ぶ。")
    print(f"\n完了。lectures/{cl.MODULE_ID}/outputs/01_*.png と 01_kmeans_report.json を確認してください。")


if __name__ == "__main__":
    main()
