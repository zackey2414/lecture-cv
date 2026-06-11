"""02: 埋め込みの「品質」を測る — kNN 分類精度 と Recall@k。

学ぶこと:
  - 埋め込みの良し悪しは、それ自体では見えない。下流タスク（ここでは検索/分類）の
    精度で測るのが定石。代表的な 2 指標を「定義 → どう計算して結果を得るか」で実装する:
      * kNN 分類精度: クエリのコサイン最近傍 k 件を多数決し、正解ラベルと一致した割合。
      * Recall@k    : クエリの上位 k 件に「同クラス」が 1 件でも入る割合（検索向け）。
  - ギャラリー（検索対象＝既知）とクエリ（検索する側＝評価）に分けて測る。
  - 同じ埋め込みでも L2 正規化（コサイン）の有無で結果が変わることを確認する。
  - ViT(CLS / mean) と ResNet(GAP) と timm を横並びで比較し、用途で選ぶ感覚をつける。

実行:
  uv run python lectures/15_image_embeddings_metric_learning/02_knn_recall_eval.py
初回はモデル重みをダウンロード（以後キャッシュ）。結果は outputs/ に保存。
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import timm
import torch
from transformers import AutoImageProcessor, ResNetModel, ViTModel

import embed_helpers as eh

VIT_ID = "google/vit-base-patch16-224"
RESNET_ID = "microsoft/resnet-18"


def _batched(seq: list, size: int):
    """大きすぎる入力を一度に流すとメモリを食うので、ミニバッチに区切って回す。"""
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


@torch.inference_mode()
def extract_all(images: list, device: torch.device) -> dict[str, np.ndarray]:
    """4 通りの埋め込みをまとめて抽出する: vit_cls / vit_mean / resnet_gap / timm。"""
    feats: dict[str, list[np.ndarray]] = {"vit_cls": [], "vit_mean": [], "resnet_gap": [], "timm": []}

    # --- ViT ---
    vit_proc = AutoImageProcessor.from_pretrained(VIT_ID)
    vit = ViTModel.from_pretrained(VIT_ID).to(device).eval()
    for batch in _batched(images, 16):
        out = vit(**vit_proc(images=batch, return_tensors="pt").to(device))
        last = out.last_hidden_state
        feats["vit_cls"].append(last[:, 0].cpu().numpy())
        feats["vit_mean"].append(last[:, 1:].mean(dim=1).cpu().numpy())

    # --- ResNet(HF) ---
    res_proc = AutoImageProcessor.from_pretrained(RESNET_ID)
    res = ResNetModel.from_pretrained(RESNET_ID).to(device).eval()
    for batch in _batched(images, 16):
        out = res(**res_proc(images=batch, return_tensors="pt").to(device))
        feats["resnet_gap"].append(out.pooler_output.flatten(1).cpu().numpy())

    # --- timm ---
    tm = timm.create_model("resnet18", pretrained=True, num_classes=0).to(device).eval()
    cfg = timm.data.resolve_data_config({}, model=tm)
    transform = timm.data.create_transform(**cfg)
    for batch in _batched(images, 16):
        x = torch.stack([transform(im) for im in batch]).to(device)
        feats["timm"].append(tm(x).cpu().numpy())

    return {k: np.concatenate(v, axis=0) for k, v in feats.items()}


def split_gallery_query(labels: np.ndarray, n_query_per_class: int = 6) -> tuple[np.ndarray, np.ndarray]:
    """各クラスの末尾 n_query 枚をクエリ、残りをギャラリーにする添字を返す。"""
    gallery_idx, query_idx = [], []
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        query_idx.extend(idx[-n_query_per_class:])
        gallery_idx.extend(idx[:-n_query_per_class])
    return np.array(gallery_idx), np.array(query_idx)


def euclidean_knn_accuracy(g, gl, q, ql, k=5) -> float:
    """比較用: 正規化せず「生のユークリッド距離」で kNN 分類した精度。"""
    # (nq, ng) の二乗距離。小さいほど近い。
    d = ((q[:, None, :] - g[None, :, :]) ** 2).sum(-1)
    topk = np.argsort(d, axis=1)[:, :k]
    preds = np.array([np.bincount(gl[row]).argmax() for row in topk])
    return float((preds == ql).mean())


def main() -> None:
    os.makedirs(eh.OUT_DIR, exist_ok=True)
    device = eh.get_device()
    print(f"device = {device}")

    images, labels, names = eh.make_dataset(n_per_class=18, seed=0)
    print(f"データセット: {len(images)} 枚 / {len(names)} クラス {names}")

    g_idx, q_idx = split_gallery_query(labels, n_query_per_class=6)
    print(f"ギャラリー {len(g_idx)} 枚 / クエリ {len(q_idx)} 枚\n")

    print("埋め込みを抽出中（初回はモデルDLあり）…")
    emb = extract_all(images, device)

    K = 5
    print(f"\n[評価] cosine(L2正規化) での kNN 分類精度 と Recall@{K}")
    results = {}
    for name, vecs in emb.items():
        g, q = vecs[g_idx], vecs[q_idx]
        gl, ql = labels[g_idx], labels[q_idx]
        acc = eh.knn_accuracy(g, gl, q, ql, k=K)
        rec = eh.recall_at_k(g, gl, q, ql, k=K)
        results[name] = (acc, rec)
        print(f"  {name:11s} kNN-acc={acc:.3f}  Recall@{K}={rec:.3f}")

    # 正規化の効果: ベクトルの「長さ」がバラつく状況で、コサインと生ユークリッドを比較する。
    # 合成のキレイなデータでは長さがほぼ揃っているので、まず各ベクトルに
    # ランダムな正のスケールを掛けて「露出/明るさで長さが変わった」状況を作る（向きは不変）。
    g0, q0 = emb["resnet_gap"][g_idx], emb["resnet_gap"][q_idx]
    gl, ql = labels[g_idx], labels[q_idx]
    rng = np.random.default_rng(0)
    gs = g0 * rng.uniform(0.2, 3.0, size=(len(g0), 1))  # 1枚ごとに長さだけ変える
    qs = q0 * rng.uniform(0.2, 3.0, size=(len(q0), 1))
    cos_acc = eh.knn_accuracy(gs, gl, qs, ql, k=K)  # 内部で L2 正規化 → 長さの影響を消す
    euc_acc = euclidean_knn_accuracy(gs, gl, qs, ql, k=K)  # 長さの違いに距離が支配される
    print("\n[正規化の効果] resnet_gap に『長さのバラつき』を注入して比較")
    print(f"  cosine(L2正規化) kNN-acc = {cos_acc:.3f}  ← 向きだけ見るので長さ変動に強い")
    print(f"  raw Euclidean    kNN-acc = {euc_acc:.3f}  ← 長さ(明るさ等)に振り回されて低下")

    # 棒グラフで横並び比較
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    xs = np.arange(len(results))
    accs = [results[k][0] for k in results]
    recs = [results[k][1] for k in results]
    ax.bar(xs - 0.2, accs, width=0.4, label="kNN accuracy")
    ax.bar(xs + 0.2, recs, width=0.4, label=f"Recall@{K}")
    ax.set_xticks(xs)
    ax.set_xticklabels(list(results.keys()), rotation=15)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("score")
    ax.set_title("Embedding quality: kNN accuracy vs Recall@k")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(eh.OUT_DIR, "02_knn_recall_compare.png"), dpi=110)
    plt.close(fig)

    # 一番良かった埋め込みを PCA で 2D に落として散布図（クラスが分かれていれば良い埋め込み）
    best = max(results, key=lambda k: results[k][0])
    from sklearn.decomposition import PCA

    xy = PCA(n_components=2).fit_transform(eh.l2_normalize(emb[best]))
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    for c, cname in enumerate(names):
        m = labels == c
        ax.scatter(xy[m, 0], xy[m, 1], s=14, label=cname)
    ax.set_title(f"PCA(2D) of L2-normalized embeddings: {best}")
    ax.legend(fontsize=7, markerscale=1.2)
    fig.tight_layout()
    fig.savefig(os.path.join(eh.OUT_DIR, "02_pca_scatter.png"), dpi=110)
    plt.close(fig)

    print(f"\n最良の埋め込み: {best}")
    print(f"完了。{eh.OUT_DIR}/02_knn_recall_compare.png と 02_pca_scatter.png を確認してください。")


if __name__ == "__main__":
    main()
