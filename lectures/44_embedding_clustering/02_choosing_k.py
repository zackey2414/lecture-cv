"""02: クラスタ数 k の選び方 — エルボー法とシルエット法。

学ぶこと:
  - k-means は **k を自分で決めねばならない**。多すぎると同じグループを刻みすぎ、少なすぎると
    別グループを混ぜる。では「ちょうど良い k」をどう決めるか。
  - **エルボー法**: k を増やすと inertia（各点と所属中心の二乗距離和）は単調に下がる。下がり方が
    急に緩む“肘（elbow）”が目安。ただし主観的で、肘がはっきり出ないこともある。
  - **シルエット法**: 各 k で silhouette_score（自分のクラスタに近く隣から遠いほど 1）を計算し、
    **最大の k** を選ぶ。ラベル不要で、肘より自動化しやすい。本章ではこちらを主役にする。
  - 答え合わせ: 既知グループがあるので NMI も重ねて描く。silhouette のピークが真の k と一致する
    ことを目で確認する（本来 NMI は使えない＝ラベルが無いから k を選ぶ、という状況を思い出す）。

実行:
  uv run python lectures/44_embedding_clustering/02_choosing_k.py
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, silhouette_score

import cluster_lab as cl


def sweep_k(emb: np.ndarray, k_values: range, labels_true: np.ndarray) -> dict:
    """各 k で k-means を回し、inertia / silhouette / NMI を集める。

    silhouette は cosine 距離で測る（埋め込みは L2 正規化済み＝コサインが自然）。
    NMI は『答え合わせ用』で、実運用では使えない（ラベルが無いから k を選ぶ）。
    """
    inertia, silhouette, nmi = [], [], []
    for k in k_values:
        km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(emb)
        inertia.append(float(km.inertia_))
        silhouette.append(float(silhouette_score(emb, km.labels_, metric="cosine")))
        nmi.append(float(normalized_mutual_info_score(labels_true, km.labels_)))
    return {"k": list(k_values), "inertia": inertia, "silhouette": silhouette, "nmi": nmi}


def best_k_by_silhouette(k_values: list[int], silhouette: list[float]) -> int:
    """silhouette が最大の k を返す（自動 k 選択の定石）。"""
    return int(k_values[int(np.argmax(silhouette))])


def plot_curves(res: dict, best_k: int, true_k: int, path) -> None:
    """左: エルボー（inertia）／右: シルエット & NMI を 1 枚に描く。図中文字は ASCII。"""
    k = res["k"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))

    ax1.plot(k, res["inertia"], "o-", color="tab:blue")
    ax1.axvline(true_k, color="gray", ls="--", lw=1, label=f"true k={true_k}")
    ax1.set_title("Elbow method (inertia vs k)")
    ax1.set_xlabel("k")
    ax1.set_ylabel("inertia (lower=tighter)")
    ax1.legend()

    ax2.plot(k, res["silhouette"], "o-", color="tab:green", label="silhouette (no labels)")
    ax2.plot(k, res["nmi"], "s--", color="tab:orange", label="NMI (needs labels)")
    ax2.axvline(best_k, color="tab:green", ls=":", lw=1.5, label=f"picked k={best_k}")
    ax2.set_title("Silhouette picks k; NMI confirms")
    ax2.set_xlabel("k")
    ax2.set_ylabel("score (higher=better)")
    ax2.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> None:
    out = cl.ensure_out_dir()
    device = cl.get_device()

    images, labels, names = cl.make_image_collection(n_per_group=6, seed=0)
    emb = cl.embed_images(images, device)
    true_k = len(names)
    print(f"画像 {len(images)} 枚（真のグループ数は {true_k} だが、k 選択には教えない）\n")

    k_values = range(2, 11)
    res = sweep_k(emb, k_values, labels)
    print(" k | inertia | silhouette | NMI(答え合わせ)")
    for i, k in enumerate(res["k"]):
        mark = "  <- silhouette max" if k == best_k_by_silhouette(res["k"], res["silhouette"]) else ""
        print(f" {k:2d} | {res['inertia'][i]:7.3f} | {res['silhouette'][i]:.3f}      | {res['nmi'][i]:.3f}{mark}")

    best_k = best_k_by_silhouette(res["k"], res["silhouette"])
    print(f"\nシルエット最大の k = {best_k}（真の {true_k} と一致すれば k 選択成功）")

    plot_curves(res, best_k, true_k, out / "02_choosing_k.png")
    (out / "02_choosing_k.json").write_text(
        json.dumps({"best_k": best_k, "true_k": true_k, **res}, ensure_ascii=False, indent=2)
    )

    print("\n読み取り方:")
    print("  - 左図 inertia は単調減少。傾きが急に緩む“肘”あたりが目安（やや主観的）。")
    print("  - 右図 silhouette のピークが自動選択した k。NMI のピークと一致していれば筋が良い。")
    print("  - 教訓: ラベルが無くても silhouette だけで k を選べる。NMI は本来見えない確認用。")
    print(f"\n完了。lectures/{cl.MODULE_ID}/outputs/02_choosing_k.png を確認してください。")


if __name__ == "__main__":
    main()
