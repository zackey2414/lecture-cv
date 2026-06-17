"""03: 顔埋め込みのクラスタリング — 写真アルバムの「人物」自動仕分け。

学ぶこと:
  - L2 正規化した顔埋め込みに `DBSCAN` / `AgglomerativeClustering` を当て、ラベル無しで
    『同一人物ごとの束（クラスタ）』を自動生成する。スマホの写真アプリの「人物」機能の核。
  - DBSCAN は **クラスタ数を指定せず**、密度（eps 近傍に min_samples 件以上）でまとまりを作り、
    どのまとまりにも入らない点を **ノイズ(-1)** にする。未知の人物・外れ値に強い。
  - Agglomerative(凝集型) は近いものから併合し、**距離しきい値**で「ここまで近ければ同一人物」
    と切る。cosine + average linkage が顔では扱いやすい。
  - どちらも距離は **コサイン**（metric="cosine"）。L2 正規化済みなのでコサイン距離 = 1 - 内積。
  - クラスタ品質は purity / NMI / homogeneity / completeness で測る（詳細は 04 で深掘り）。

実行:
  uv run python lectures/30_face_detection_recognition/03_face_clustering.py
結果（アルバム・2D 散布図）は lectures/30_.../outputs/ に保存。
"""

from __future__ import annotations

import os

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA

import face_lab as fl


def cluster_dbscan(emb_norm: np.ndarray, eps: float = 0.05, min_samples: int = 2) -> np.ndarray:
    """L2 正規化済み埋め込みに DBSCAN（コサイン距離）。返り値はクラスタ ID 配列（-1=ノイズ）。"""
    return DBSCAN(eps=eps, min_samples=min_samples, metric="cosine").fit_predict(emb_norm)


def cluster_agglomerative(emb_norm: np.ndarray, distance_threshold: float = 0.06) -> np.ndarray:
    """凝集型クラスタリング（コサイン距離・average linkage・しきい値で自動分割）。"""
    model = AgglomerativeClustering(
        n_clusters=None,  # クラスタ数は指定せず、距離しきい値で決める
        distance_threshold=distance_threshold,
        metric="cosine",
        linkage="average",
    )
    return model.fit_predict(emb_norm)


def save_album(
    images: list[Image.Image], pred: np.ndarray, path: str, max_per_row: int = 8
) -> None:
    """予測クラスタごとに顔を 1 行に並べた「アルバム」画像を作る（行=自動仕分けされた人物）。

    ノイズ(-1) は最終行にまとめる。各行の左端に cluster ID を書く。
    """
    fs = fl.FACE_SIZE
    pad = 6
    label_w = 60
    clusters = sorted(set(pred), key=lambda c: (c == -1, c))  # -1 は最後に
    rows = []
    for c in clusters:
        idxs = np.where(pred == c)[0][:max_per_row]
        row = np.full((fs, label_w, 3), 245, dtype=np.uint8)
        tag = "noise" if c == -1 else f"#{c}"
        cv2.putText(row, tag, (4, fs // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        for i in idxs:
            bgr = cv2.cvtColor(np.array(images[i]), cv2.COLOR_RGB2BGR)
            row = np.hstack([row, np.full((fs, pad, 3), 245, np.uint8), bgr])
        rows.append(row)

    width = max(r.shape[1] for r in rows)
    rows = [np.hstack([r, np.full((fs, width - r.shape[1], 3), 245, np.uint8)]) for r in rows]
    canvas = rows[0]
    for r in rows[1:]:
        canvas = np.vstack([canvas, np.full((pad, width, 3), 200, np.uint8), r])
    cv2.imwrite(path, canvas)


def save_pca_scatter(
    emb_norm: np.ndarray, labels_true: np.ndarray, pred: np.ndarray, path: str
) -> None:
    """埋め込みを PCA で 2D に落とし、色=予測クラスタ / 形=真の人物 で散布図にする。

    同じ色（予測クラスタ）の点がまとまり、別人物が離れていれば、埋め込み空間で人物が
    きれいに分離していることが目で分かる。
    """
    xy = PCA(n_components=2).fit_transform(emb_norm)
    fig, ax = plt.subplots(figsize=(6, 5))
    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]
    for pid in np.unique(labels_true):
        m = labels_true == pid
        ax.scatter(
            xy[m, 0],
            xy[m, 1],
            c=pred[m],
            cmap="tab10",
            vmin=0,
            vmax=9,
            marker=markers[pid % len(markers)],
            s=70,
            edgecolors="k",
            linewidths=0.4,
        )
    ax.set_title("face embeddings (PCA 2D): color=predicted cluster, marker=true id")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> None:
    os.makedirs(fl.OUT_DIR, exist_ok=True)
    device = fl.get_device()

    images, labels, names = fl.make_face_dataset(n_ids=8, n_per_id=6, seed=0)
    print(
        f"合成顔: {len(images)} 枚 / {len(names)} 人（正解は 8 人だが、クラスタリングには教えない）"
    )

    enc = fl.FaceEncoder(device)
    emb = fl.l2_normalize(enc.encode(images))  # ★ クラスタリング前に必ず L2 正規化
    print(f"埋め込み: mode={enc.mode}, shape={emb.shape}\n")

    print("[DBSCAN] 密度ベース（eps=0.05, min_samples=2, cosine）")
    pred_db = cluster_dbscan(emb)
    rep_db = fl.clustering_report(labels, pred_db)
    print(
        f"  推定人数={rep_db['n_clusters_found']}  ノイズ={rep_db['n_noise']}  "
        f"purity={rep_db['purity']:.3f}  NMI={rep_db['nmi']:.3f}  homo={rep_db['homogeneity']:.3f}"
    )
    save_album(images, pred_db, os.path.join(fl.OUT_DIR, "03_album_dbscan.png"))

    print("\n[Agglomerative] 凝集型（distance_threshold=0.06, cosine, average）")
    pred_ag = cluster_agglomerative(emb)
    rep_ag = fl.clustering_report(labels, pred_ag)
    print(
        f"  推定人数={rep_ag['n_clusters_found']}  "
        f"purity={rep_ag['purity']:.3f}  NMI={rep_ag['nmi']:.3f}  homo={rep_ag['homogeneity']:.3f}"
    )
    save_album(images, pred_ag, os.path.join(fl.OUT_DIR, "03_album_agglo.png"))

    save_pca_scatter(emb, labels, pred_db, os.path.join(fl.OUT_DIR, "03_pca_scatter.png"))

    print("\n読み取り方:")
    print("  - 推定人数が真の 8 人に一致し、各クラスタが 1 人で揃えば自動仕分け成功。")
    print("  - DBSCAN は eps を大きくし過ぎると別人が混ざり、小さ過ぎるとノイズ(-1)だらけになる。")
    print("  - Agglomerative は distance_threshold が同じ役割。04 でしきい値掃引して最適点を探す。")
    print(
        f"\n完了。lectures/{fl.MODULE_ID}/outputs/03_album_*.png と 03_pca_scatter.png を確認してください。"
    )


if __name__ == "__main__":
    main()
