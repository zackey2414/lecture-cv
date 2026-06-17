"""04: テキストのクラスタリングとクロスモーダルの落とし穴（モダリティギャップ）。

学ぶこと:
  - クラスタリングは画像専用ではない。CLIP の **get_text_features** で文も同じ 512 次元空間に
    乗るので、キャプション/ラベル集合をそのまま k-means でトピック分けできる。
  - 画像とテキストが“同じ空間”なら、両方を混ぜて一緒にクラスタリングできそう…が、ここに
    有名な落とし穴 **モダリティギャップ** がある。CLIP の画像ベクトル群とテキストベクトル群は
    別々の“円錐”に偏って分布し、素朴に混ぜて k=2 で割ると **概念ではなくモダリティ（画像か
    テキストか）で割れて**しまう。
  - 一方、**クロスモーダル検索**（画像→対応キャプション）は相対比較なので、ギャップがあっても
    正しく動く。「絶対位置で束ねる（クラスタリング）」と「相対順位で引く（検索）」の差が出る。
  - 対策: モダリティごとに平均を引いて中心を揃え（centering）再正規化すると、概念で束ねられる。
  - 顔クラスタリング（30）は、この枠組みの“埋め込みが顔ベクトルになっただけ”の特殊例である。

実行:
  uv run python lectures/44_embedding_clustering/04_text_and_crossmodal.py
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import normalized_mutual_info_score, silhouette_score

import cluster_lab as cl


def cluster_texts(temb: np.ndarray, labels: np.ndarray, k: int) -> dict:
    """テキスト埋め込みを k-means でクラスタリングし silhouette / NMI を返す。"""
    pred = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(temb)
    return {
        "k": k,
        "silhouette": float(silhouette_score(temb, pred, metric="cosine")),
        "nmi": float(normalized_mutual_info_score(labels, pred)),
    }


def crossmodal_demo(device, names: list[str], images, labels) -> dict:
    """画像（各グループ代表 1 枚）とその英語キャプションを同じ空間に置いて挙動を比べる。

    返す指標:
      - retrieval_top1: 画像→キャプションの最近傍が正解の割合（相対比較＝ギャップに強い）
      - gap_cosine: 画像平均ベクトルとテキスト平均ベクトルのコサイン（小さいほどギャップ大）
      - joint_k2_*: 素朴に混ぜて k=2 → モダリティで割れる（NMI vs modality が高い）
      - centered_k6_*: モダリティ中心を引いて再正規化 → 概念で割れる（NMI vs concept が高い）
    """
    reps = [images[int(np.where(labels == g)[0][0])] for g in range(len(names))]
    caps = [f"a photo of a {n}" for n in names]
    ie = cl.embed_images(reps, device)
    te = cl.embed_texts(caps, device)

    # クロスモーダル検索: 各画像に最も近いキャプション
    sim = ie @ te.T
    top1 = float((sim.argmax(axis=1) == np.arange(len(names))).mean())

    both = np.vstack([ie, te])
    modality = np.array([0] * len(names) + [1] * len(names))  # 0=image, 1=text
    concept = np.array(list(range(len(names))) * 2)  # 同じ概念 → 同じ ID

    # 素朴に混ぜて k=2: 概念ではなくモダリティで割れることを示す
    p2 = KMeans(n_clusters=2, n_init=10, random_state=0).fit_predict(both)

    # モダリティごとに中心を引いて再正規化 → 概念で割れる
    centered = both.copy()
    centered[: len(names)] -= ie.mean(axis=0)
    centered[len(names) :] -= te.mean(axis=0)
    centered = cl.l2_normalize(centered)
    p6 = KMeans(n_clusters=len(names), n_init=10, random_state=0).fit_predict(centered)

    img_c, txt_c = ie.mean(0), te.mean(0)
    gap = float(img_c @ txt_c / (np.linalg.norm(img_c) * np.linalg.norm(txt_c)))
    return {
        "retrieval_top1": top1,
        "gap_cosine": gap,
        "joint_k2_nmi_vs_modality": float(normalized_mutual_info_score(modality, p2)),
        "joint_k2_nmi_vs_concept": float(normalized_mutual_info_score(concept, p2)),
        "centered_k6_nmi_vs_concept": float(normalized_mutual_info_score(concept, p6)),
        "centered_k6_nmi_vs_modality": float(normalized_mutual_info_score(modality, p6)),
        "_both": both,
        "_modality": modality,
    }


def plot_modality_gap(both: np.ndarray, modality: np.ndarray, path) -> None:
    """画像群とテキスト群を PCA 2D に落とし、2 つの“島”に分かれる様子を描く。"""
    xy = PCA(n_components=2).fit_transform(both)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(xy[modality == 0, 0], xy[modality == 0, 1], c="tab:blue", marker="o", s=80,
               edgecolors="k", label="image embeddings")
    ax.scatter(xy[modality == 1, 0], xy[modality == 1, 1], c="tab:red", marker="^", s=80,
               edgecolors="k", label="text embeddings")
    ax.set_title("CLIP modality gap (PCA 2D): images and texts form separate islands")
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> None:
    out = cl.ensure_out_dir()
    device = cl.get_device()

    # --- 1. テキストのクラスタリング ---
    texts, tlabels, tnames = cl.make_text_collection()
    temb = cl.embed_texts(texts, device)
    print(f"テキスト {len(texts)} 文 / 既知 {len(tnames)} トピック {tnames}")
    print("k 別の評価（真トピック数は 4。画像より概念の境界が曖昧で silhouette は低め）")
    print("  k | silhouette | NMI")
    text_rows = [cluster_texts(temb, tlabels, k) for k in (3, 4, 5)]
    for r in text_rows:
        print(f"  {r['k']} | {r['silhouette']:.3f}      | {r['nmi']:.3f}")
    print()

    # --- 2. クロスモーダル: モダリティギャップ ---
    images, labels, names = cl.make_image_collection(n_per_group=6, seed=0)
    cm = crossmodal_demo(device, names, images, labels)
    print("クロスモーダル（画像代表 6 枚 + 対応英語キャプション 6 文）:")
    print(f"  画像→キャプション 検索 top1 正解率 = {cm['retrieval_top1']:.2f}"
          "（相対比較なのでギャップがあっても効く）")
    print(f"  画像平均と文平均のコサイン = {cm['gap_cosine']:.3f}（低い＝別々の島に偏っている）")
    print("  素朴に混ぜて k=2 でクラスタリング:")
    print(f"    NMI vs モダリティ = {cm['joint_k2_nmi_vs_modality']:.3f}（高い＝画像/文で割れた）")
    print(f"    NMI vs 概念       = {cm['joint_k2_nmi_vs_concept']:.3f}（低い＝概念では割れていない）")
    print("  モダリティ中心を引いて再正規化 → k=6:")
    print(f"    NMI vs 概念       = {cm['centered_k6_nmi_vs_concept']:.3f}（高い＝概念で束ねられた）")
    print(f"    NMI vs モダリティ = {cm['centered_k6_nmi_vs_modality']:.3f}（低い＝モダリティは無視できた）\n")

    plot_modality_gap(cm.pop("_both"), cm.pop("_modality"), out / "04_modality_gap.png")
    (out / "04_text_crossmodal.json").write_text(
        json.dumps({"text_clustering": text_rows, "crossmodal": cm}, ensure_ascii=False, indent=2)
    )

    print("まとめ:")
    print("  - テキストも画像と同じ枠組み（埋め込み→正規化→クラスタリング→評価）で束ねられる。")
    print("  - ただしクロスモーダルを“素朴に混ぜて”束ねるとモダリティで割れる（モダリティギャップ）。")
    print("    検索（相対順位）は無事だが、クラスタリング（絶対位置）はギャップに弱い。")
    print("  - 対策はモダリティごとの centering 等。顔クラスタリング(30)も埋め込みが変わるだけの同じ話。")
    print(f"\n完了。lectures/{cl.MODULE_ID}/outputs/04_modality_gap.png を確認してください。")


if __name__ == "__main__":
    main()
