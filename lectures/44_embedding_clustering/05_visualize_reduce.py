"""05: 高次元埋め込みを 2D で見る — PCA と t-SNE（UMAP/HDBSCAN は任意）。

学ぶこと:
  - 512 次元の埋め込みは目で見えない。**次元削減**で 2D に落として散布図にすると、クラスタが
    本当に分離しているか・どこが混ざりやすいかが一目で分かる（クラスタリングの“答え合わせ”の目）。
  - **PCA**: 分散最大の軸へ線形射影。速くて決定的、全体構造（大局）を保つ。explained_variance_
    で「2 軸で何 % を説明できたか」が読める。距離の素直な縮約。
  - **t-SNE**: 非線形。近傍関係（局所構造）を保つよう確率的に配置する。クラスタの“島”を見せるのが
    得意だが、**島同士の距離や島の大きさには意味が無い**・perplexity と乱数に敏感、という作法を
    必ず押さえる。可視化専用で、クラスタリングを t-SNE 座標の上で“やり直す”のは避ける。
  - **UMAP / HDBSCAN** は強力だが本リポジトリ既定環境には未導入。import をガードし、入れれば
    使える導線だけ示す（uv add で追加）。

実行:
  uv run python lectures/44_embedding_clustering/05_visualize_reduce.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import cluster_lab as cl


def reduce_pca(emb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """PCA で 2D に落とし、(座標, 各軸の説明分散比) を返す。"""
    pca = PCA(n_components=2)
    xy = pca.fit_transform(emb)
    return xy, pca.explained_variance_ratio_


def reduce_tsne(emb: np.ndarray, seed: int = 0) -> np.ndarray:
    """t-SNE で 2D に落とす。perplexity は点数に応じて自動調整（n より十分小さく）。

    init="pca" で初期化を安定させ、learning_rate="auto" は sklearn 推奨の既定挙動。
    """
    n = len(emb)
    perplexity = float(max(5, min(30, (n - 1) // 3)))  # n が小さくても破綻しない範囲に
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=seed,
    )
    return tsne.fit_transform(emb)


def try_umap(emb: np.ndarray) -> np.ndarray | None:
    """UMAP があれば 2D 座標を返す。未導入なら None（ガード）。"""
    try:
        import umap  # type: ignore
    except ImportError:
        return None
    return umap.UMAP(n_components=2, random_state=0).fit_transform(emb)


def main() -> None:
    out = cl.ensure_out_dir()
    device = cl.get_device()

    images, labels, names = cl.make_image_collection(n_per_group=6, seed=0)
    emb = cl.embed_images(images, device)
    pred = KMeans(n_clusters=len(names), n_init=10, random_state=0).fit_predict(emb)

    # --- PCA ---
    xy_pca, evr = reduce_pca(emb)
    print(f"PCA: 2 軸の説明分散比 = {evr[0]:.3f}, {evr[1]:.3f}（合計 {evr.sum():.3f}）")
    print("  → 2 軸でこれだけの分散を説明。残りは捨てているので“見えていない構造”もある点に注意")
    cl.scatter_2d(xy_pca, pred, labels,
                  f"PCA 2D (EVR={evr.sum():.2f}): color=cluster, marker=true group",
                  out / "05_pca.png")

    # --- t-SNE ---
    xy_tsne = reduce_tsne(emb)
    print("t-SNE: 局所構造を保つ非線形埋め込み（島の距離・大きさには意味が無い）")
    cl.scatter_2d(xy_tsne, pred, labels,
                  "t-SNE 2D: color=cluster, marker=true group (distances NOT meaningful)",
                  out / "05_tsne.png")

    # --- PCA と t-SNE を 1 枚に並べて対比 ---
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, xy, ttl in [(a1, xy_pca, "PCA (linear, global)"), (a2, xy_tsne, "t-SNE (nonlinear, local)")]:
        ax.scatter(xy[:, 0], xy[:, 1], c=pred, cmap="tab10", vmin=0, vmax=9,
                   s=60, edgecolors="k", linewidths=0.4)
        ax.set_title(ttl)
        ax.set_xlabel("dim 1")
        ax.set_ylabel("dim 2")
    fig.suptitle("Dimensionality reduction for cluster inspection (color = kmeans cluster)")
    fig.tight_layout()
    fig.savefig(out / "05_pca_vs_tsne.png", dpi=110)
    plt.close(fig)

    # --- UMAP（任意・ガード）---
    xy_umap = try_umap(emb)
    if xy_umap is None:
        print("\nUMAP は未導入のためスキップ（任意）。試すには:")
        print("  uv add umap-learn   # その後この 05 を再実行すると 05_umap.png も出ます")
    else:
        cl.scatter_2d(xy_umap, pred, labels, "UMAP 2D: color=cluster, marker=true group",
                      out / "05_umap.png")
        print("\nUMAP 検出: 05_umap.png も保存しました。")

    # --- HDBSCAN（任意・ガード）。sklearn 1.3+ なら sklearn.cluster.HDBSCAN も使える ---
    try:
        from sklearn.cluster import HDBSCAN

        hpred = HDBSCAN(min_cluster_size=3, metric="euclidean", copy=True).fit_predict(emb)
        rep = cl.clustering_report(hpred, emb, labels)
        print(f"\n(任意) sklearn HDBSCAN: clusters={rep['n_clusters_found']} "
              f"noise={rep['n_noise']} NMI={rep['nmi']:.3f}（k 不要・密度ベースの強化版）")
    except Exception as e:  # noqa: BLE001
        print(f"\nHDBSCAN はスキップ: {type(e).__name__}")

    print("\n読み取り方:")
    print("  - PCA は速く決定的。explained_variance で“どれだけ見えているか”を必ず確認する。")
    print("  - t-SNE は島の分離を見せるのが得意だが、島間距離・島の大きさを読みすぎない。")
    print("  - 可視化はクラスタリングの“検証の目”。2D 座標の上で clustering をやり直さないこと。")
    print(f"\n完了。outputs/{cl.MODULE_ID}/05_pca_vs_tsne.png を確認してください。")


if __name__ == "__main__":
    main()
