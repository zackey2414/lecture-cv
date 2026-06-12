"""章末ミニプロジェクト: 埋め込みクラスタリングの一気通貫ツール。

画像コレクションとテキスト集合を CLIP で埋め込み →（k を自動選択）→ クラスタリング →
評価（教師なし silhouette ＋ 答え合わせ NMI/purity/homogeneity）→ 2D 可視化（PCA + t-SNE）
までを 1 本に通す『完成形』。実運用の「ラベル無しデータを自動で束ねて中身を把握する」流れを
そのまま体験する。CPU で数十秒、ネットは初回のモデル重み取得のみ。

4 ステージ:
  Stage A: 画像コレクションを埋め込み → silhouette で k を自動選択 → k-means で確定
  Stage B: 同じデータに Agglomerative（k 不要）を当て、k-means と評価を比較
  Stage C: テキスト集合を独立にクラスタリング（同じ枠組みが別モダリティでも動く確認）
  Stage D: PCA + t-SNE で 2D 可視化し、6 パネル要約画像と全数値 json を保存

実行:
  uv run python lectures/44_embedding_clustering/mini_project.py
出力: outputs/44_embedding_clustering/mini_project_summary.png, mini_project_report.json
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

import cluster_lab as cl


def choose_k(emb: np.ndarray, k_range: range) -> tuple[int, list[int], list[float]]:
    """各 k の silhouette を測り、最大の k と (k 列, silhouette 列) を返す（自動 k 選択）。"""
    ks, sils = [], []
    for k in k_range:
        labels = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(emb)
        ks.append(k)
        sils.append(float(silhouette_score(emb, labels, metric="cosine")))
    best_k = int(ks[int(np.argmax(sils))])
    return best_k, ks, sils


def tsne_2d(emb: np.ndarray) -> np.ndarray:
    """点数に応じた perplexity で t-SNE 2D を返す。"""
    n = len(emb)
    perplexity = float(max(5, min(30, (n - 1) // 3)))
    return TSNE(
        n_components=2, perplexity=perplexity, init="pca",
        learning_rate="auto", random_state=0,
    ).fit_transform(emb)


def _scatter(ax, xy, color, title, markers_by=None, cmap="tab10") -> None:
    """1 つの軸に散布図を描く小道具（色＝color、必要なら marker を markers_by で分ける）。"""
    if markers_by is None:
        ax.scatter(xy[:, 0], xy[:, 1], c=color, cmap=cmap, vmin=0, vmax=9, s=55,
                   edgecolors="k", linewidths=0.4)
    else:
        marks = ["o", "s", "^", "D", "v", "P", "X", "*"]
        for g in np.unique(markers_by):
            m = markers_by == g
            ax.scatter(xy[m, 0], xy[m, 1], c=color[m], cmap=cmap, vmin=0, vmax=9,
                       marker=marks[int(g) % len(marks)], s=55, edgecolors="k", linewidths=0.4)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")


def main() -> None:
    out = cl.ensure_out_dir()
    device = cl.get_device()
    report: dict = {}

    # === データ: 画像コレクション（既知 6 グループ）＋ テキスト集合（既知 4 トピック）===
    images, img_labels, img_names = cl.make_image_collection(n_per_group=6, seed=0)
    texts, txt_labels, txt_names = cl.make_text_collection()
    img_emb = cl.embed_images(images, device)
    txt_emb = cl.embed_texts(texts, device)
    print(f"画像 {len(images)} 枚 / {len(img_names)} グループ、テキスト {len(texts)} 文 / {len(txt_names)} トピック")
    print("（グループ/トピックの正解はクラスタリングには渡さず、評価だけに使う）\n")

    # === Stage A: 画像 — silhouette で k 自動選択 → k-means ===
    best_k, ks, sils = choose_k(img_emb, range(2, 11))
    km_pred = KMeans(n_clusters=best_k, n_init=10, random_state=0).fit_predict(img_emb)
    km_rep = cl.clustering_report(km_pred, img_emb, img_labels)
    print(f"[Stage A] 自動選択 k={best_k}（真は {len(img_names)}） → k-means")
    print(f"  silhouette={km_rep['silhouette']:.3f} NMI={km_rep['nmi']:.3f} "
          f"purity={km_rep['purity']:.3f} homogeneity={km_rep['homogeneity']:.3f}")
    report["stageA_kmeans"] = {"best_k": best_k, "k_sweep": ks, "silhouette_sweep": sils, **km_rep}

    # === Stage B: 画像 — Agglomerative（k 不要）で比較 ===
    ag_pred = AgglomerativeClustering(
        n_clusters=None, distance_threshold=0.06, metric="cosine", linkage="average"
    ).fit_predict(img_emb)
    ag_rep = cl.clustering_report(ag_pred, img_emb, img_labels)
    print(f"[Stage B] Agglomerative(dt=0.06) → clusters={ag_rep['n_clusters_found']} "
          f"NMI={ag_rep['nmi']:.3f} purity={ag_rep['purity']:.3f}")
    report["stageB_agglomerative"] = {"distance_threshold": 0.06, **ag_rep}

    # === Stage C: テキスト — 同じ枠組みを別モダリティに適用 ===
    txt_pred = KMeans(n_clusters=len(txt_names), n_init=10, random_state=0).fit_predict(txt_emb)
    txt_rep = cl.clustering_report(txt_pred, txt_emb, txt_labels)
    print(f"[Stage C] テキスト k-means(k={len(txt_names)}) → silhouette={txt_rep['silhouette']:.3f} "
          f"NMI={txt_rep['nmi']:.3f} purity={txt_rep['purity']:.3f}")
    report["stageC_text"] = {"k": len(txt_names), **txt_rep}

    # === Stage D: 可視化（6 パネル）===
    xy_pca = PCA(n_components=2).fit_transform(img_emb)
    xy_tsne = tsne_2d(img_emb)
    xy_txt = PCA(n_components=2).fit_transform(txt_emb)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    # 上段: k 選択曲線 / 画像 PCA(予測) / 画像 PCA(真グループ)
    axes[0, 0].plot(ks, sils, "o-", color="tab:green")
    axes[0, 0].axvline(best_k, color="tab:green", ls=":", label=f"picked k={best_k}")
    axes[0, 0].set_title("Stage A: silhouette vs k (auto k-selection)", fontsize=10)
    axes[0, 0].set_xlabel("k")
    axes[0, 0].set_ylabel("silhouette")
    axes[0, 0].legend()
    _scatter(axes[0, 1], xy_pca, km_pred, "images PCA: color = kmeans cluster", markers_by=img_labels)
    _scatter(axes[0, 2], xy_pca, img_labels, "images PCA: color = TRUE group", cmap="tab10")
    # 下段: 画像 t-SNE(予測) / 画像 Agglomerative / テキスト PCA(予測)
    _scatter(axes[1, 0], xy_tsne, km_pred, "images t-SNE: color = kmeans cluster", markers_by=img_labels)
    _scatter(axes[1, 1], xy_pca, ag_pred, "images PCA: color = agglomerative cluster", markers_by=img_labels)
    _scatter(axes[1, 2], xy_txt, txt_pred, "texts PCA: color = kmeans cluster", markers_by=txt_labels)
    fig.suptitle("Mini-project: embedding clustering (images + texts) — color=cluster, marker=true label",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out / "mini_project_summary.png", dpi=110)
    plt.close(fig)

    (out / "mini_project_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))

    print("\n=== まとめ ===")
    print(f"  画像: 自動 k={best_k} の k-means で NMI={km_rep['nmi']:.2f}、"
          f"Agglomerative でも NMI={ag_rep['nmi']:.2f}（k 不要で同等以上）。")
    print(f"  テキスト: 同じ枠組みで NMI={txt_rep['nmi']:.2f}（境界が曖昧な分だけ画像より低い）。")
    print("  顔(30)・他の埋め込みも“同じパイプライン”で束ねられる、が本章の到達点。")
    print(f"\n完了。outputs/{cl.MODULE_ID}/mini_project_summary.png と "
          "mini_project_report.json を確認してください。")


if __name__ == "__main__":
    main()
