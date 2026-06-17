"""章末ミニプロジェクト: コンパクト埋め込み検索エンジン — メトリック学習で「小さくても効く」。

この回の学び（01 取り出す → 02 測る → 03 作り変える）を 1 本に統合する総合課題。
実務の検索では、埋め込みは小さいほどメモリ・速度で有利（FAISS の index も軽くなる）。
そこで本課題のテーマは「どこまで次元を圧縮しても検索が壊れないか。メトリック学習で
どこまで攻めた圧縮が許されるか」を定量化する。

完成形がやること:
  1. 合成データ（6 クラス）を 2 つのバックボーン（timm resnet18 / HF ViT CLS）で埋め込み、
     ギャラリー/クエリに分けて「素の高次元埋め込み」の検索性能（kNN / Recall@k）を測る。
  2. 凍結した resnet18 の 512 次元特徴を、いろいろな小次元 d ∈ {2,4,8,16,32} へ射影して比較:
       * ランダム射影（学習なし）  … 強い圧縮で同クラスが潰れ、Recall が落ちる。
       * InfoNCE 学習済み射影（メトリック学習）… 同じ d でも「同クラスを近く」並べ直すので
         Recall を保つ。これが圧縮スイープ曲線の主役。
  3. 「Recall@k を 0.95 以上に保てる最小次元」を、ランダム射影と学習済み射影で比較し、
     メトリック学習がどれだけ攻めた圧縮を許すかを示す。最小次元の検索結果グリッド・
     学習前後の 2D 可視化・スイープ曲線・metrics.json を保存する。

これは 01〜03 の集大成: 埋め込みの取り出し（形の違い）・コサインでの評価・損失で空間を
作り変える、を「コンパクト検索エンジン」として 1 つに組み上げる。

実行:
  uv run python lectures/15_image_embeddings_metric_learning/mini_project.py
初回はモデル重みを HuggingFace / timm からダウンロード（以後キャッシュ）。
学習は小次元ベクトル上の数百反復 × 数次元なので CPU でも数十秒で完走する。
結果は lectures/15_image_embeddings_metric_learning/outputs/ に保存（画面表示はしない）。
"""

from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")  # 画面の無い環境でも図を保存できるよう pyplot より前に設定
import matplotlib.pyplot as plt
import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from transformers import AutoImageProcessor, ViTModel

import embed_helpers as eh

VIT_ID = "google/vit-base-patch16-224"
DIMS = [2, 4, 8, 16, 32]  # 圧縮スイープで試す射影次元
TARGET_RECALL = 0.95  # 「これ以上を保てる最小次元」を探す閾値
EPOCHS = 400
LR = 1e-2
TEMPERATURE = 0.1
K = 5
SEED = 0


# ---------------------------------------------------------------------------
# 1. 特徴抽出（digit 始まりスクリプトは import できないので自己完結で書く）
# ---------------------------------------------------------------------------
@torch.inference_mode()
def extract_timm_resnet18(images: list, device: torch.device) -> np.ndarray:
    """timm resnet18(num_classes=0) のプール済み 512 次元特徴を取り出す（未正規化で返す）。"""
    model = timm.create_model("resnet18", pretrained=True, num_classes=0).to(device).eval()
    # 前処理は手打ちせず、モデル固有の平均/分散/サイズを自動生成する（ズレ事故を防ぐ）
    cfg = timm.data.resolve_data_config({}, model=model)
    transform = timm.data.create_transform(**cfg)
    feats = []
    for i in range(0, len(images), 16):  # ミニバッチでメモリを抑える
        x = torch.stack([transform(im) for im in images[i : i + 16]]).to(device)
        feats.append(model(x).cpu().numpy())
    return np.concatenate(feats, axis=0)


@torch.inference_mode()
def extract_vit_cls(images: list, device: torch.device) -> np.ndarray:
    """HF ViTModel の CLS トークン（last_hidden_state[:,0]）を 768 次元埋め込みとして取り出す。"""
    proc = AutoImageProcessor.from_pretrained(VIT_ID)  # transformers v5 は fast(torchvision)
    model = ViTModel.from_pretrained(VIT_ID).to(device).eval()
    feats = []
    for i in range(0, len(images), 16):
        batch = images[i : i + 16]
        out = model(**proc(images=batch, return_tensors="pt").to(device))
        feats.append(out.last_hidden_state[:, 0].cpu().numpy())  # CLS（pooler は使わない）
    return np.concatenate(feats, axis=0)


# ---------------------------------------------------------------------------
# 2. メトリック学習（凍結特徴の上の射影ヘッドだけを学習）
# ---------------------------------------------------------------------------
def supervised_infonce(z: torch.Tensor, labels: torch.Tensor, temperature: float) -> torch.Tensor:
    """教師あり InfoNCE。L2 正規化済み z について、同クラスを正例集合として引き寄せる。

    CLIP の画像-テキスト対照学習はこれの 2 モーダル版。03 で手書きした損失をそのまま使う。
    """
    sim = (z @ z.T) / temperature  # 正規化後の内積 = コサイン類似度 / 温度
    sim.fill_diagonal_(-1e9)  # 自分自身は対象外
    same = (labels[:, None] == labels[None, :]).float()
    same.fill_diagonal_(0)
    logp = F.log_softmax(sim, dim=1)
    loss = -(same * logp).sum(1) / same.sum(1).clamp(min=1)  # 各行で正例位置の log 確率を上げる
    return loss.mean()


def make_head(in_dim: int, out_dim: int) -> nn.Linear:
    """毎回同じ初期化（公平な比較のため SEED 固定）の線形射影ヘッド。"""
    torch.manual_seed(SEED)
    return nn.Linear(in_dim, out_dim)


def train_projection_head(
    feats: torch.Tensor, g_idx: np.ndarray, labels: np.ndarray, out_dim: int
) -> nn.Linear:
    """ギャラリー特徴だけを使い、Linear(in_dim -> out_dim) を InfoNCE で学習して返す。"""
    head = make_head(feats.shape[1], out_dim)
    opt = torch.optim.Adam(head.parameters(), lr=LR)
    gi = torch.from_numpy(g_idx)
    glab = torch.from_numpy(labels[g_idx])
    for _ in range(EPOCHS):
        z = F.normalize(head(feats[gi]), p=2, dim=1)  # 射影後に L2 正規化（コサイン空間）
        loss = supervised_infonce(z, glab, TEMPERATURE)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return head


def project(head: nn.Linear, feats: torch.Tensor) -> np.ndarray:
    """射影ヘッドを通して L2 正規化した埋め込みを numpy で返す（評価・可視化用）。"""
    with torch.inference_mode():
        return F.normalize(head(feats), p=2, dim=1).cpu().numpy()


# ---------------------------------------------------------------------------
# 3. 評価ヘルパ（embed_helpers の kNN / Recall を使い回す）
# ---------------------------------------------------------------------------
def evaluate(emb: np.ndarray, g_idx: np.ndarray, q_idx: np.ndarray, labels: np.ndarray) -> dict:
    """1 つの埋め込みについて kNN 分類精度 と Recall@K を返す。"""
    g, q = emb[g_idx], emb[q_idx]
    gl, ql = labels[g_idx], labels[q_idx]
    return {
        "knn_acc": eh.knn_accuracy(g, gl, q, ql, k=K),
        "recall_at_k": eh.recall_at_k(g, gl, q, ql, k=K),
    }


def split_gallery_query(labels: np.ndarray, n_query_per_class: int = 7) -> tuple[np.ndarray, np.ndarray]:
    """各クラスの末尾 n_query 枚をクエリ、残りをギャラリーにする添字を返す。"""
    g_idx, q_idx = [], []
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        q_idx.extend(idx[-n_query_per_class:])
        g_idx.extend(idx[:-n_query_per_class])
    return np.array(g_idx), np.array(q_idx)


def smallest_dim_keeping(recall_by_dim: dict[int, float], target: float) -> int | None:
    """Recall@K >= target を満たす最小の射影次元を返す（無ければ None）。"""
    ok = [d for d in sorted(recall_by_dim) if recall_by_dim[d] >= target]
    return ok[0] if ok else None


# ---------------------------------------------------------------------------
# 4. 可視化
# ---------------------------------------------------------------------------
def save_sweep_curve(
    rand_recall: dict[int, float], learn_recall: dict[int, float], raw_recall: float, path: str
) -> None:
    """射影次元 d に対する Recall@K を、ランダム射影と学習済み射影で重ねた曲線。"""
    dims = sorted(rand_recall)
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    ax.plot(dims, [rand_recall[d] for d in dims], marker="o", label="random projection")
    ax.plot(dims, [learn_recall[d] for d in dims], marker="s", label="InfoNCE metric learning")
    ax.axhline(raw_recall, ls="--", color="gray", label="raw 512-D (no compression)")
    ax.set_xscale("log", base=2)
    ax.set_xticks(dims)
    ax.set_xticklabels([str(d) for d in dims])
    ax.set_xlabel("projection dim (compressed embedding size)")
    ax.set_ylabel(f"Recall@{K}")
    ax.set_ylim(0, 1.05)
    ax.set_title("Compression sweep: metric learning keeps Recall at tiny dims")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def save_pca_before_after(
    emb_before: np.ndarray, emb_after: np.ndarray, labels: np.ndarray, names: list[str], dim: int, path: str
) -> None:
    """最小次元での埋め込みを 2D に落として学習前後を左右に並べる（重なり→分離が見える）。"""
    def to2d(emb: np.ndarray) -> np.ndarray:
        if emb.shape[1] == 2:
            return emb  # 既に 2D ならそのまま
        return PCA(n_components=2).fit_transform(eh.l2_normalize(emb))

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.0))
    for ax, emb, title in [
        (axes[0], emb_before, "before (random projection)"),
        (axes[1], emb_after, "after InfoNCE metric learning"),
    ]:
        xy = to2d(emb)
        for c, cname in enumerate(names):
            m = labels == c
            ax.scatter(xy[m, 0], xy[m, 1], s=14, label=cname)
        ax.set_title(title, fontsize=10)
    axes[1].legend(fontsize=7, loc="best")
    fig.suptitle(f"Metric learning reshapes the {dim}-D embedding space")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def save_retrieval_grid(
    images: list,
    emb: np.ndarray,
    g_idx: np.ndarray,
    q_idx: np.ndarray,
    labels: np.ndarray,
    names: list[str],
    path: str,
) -> None:
    """代表クエリ 3 件について、コサイン上位 K 件のギャラリー画像を並べる検索グリッド。

    正解クラスと一致した結果は緑枠、外れは赤枠で囲み、検索の当たり外れを目で確認できる。
    """
    sim = eh.cosine_sim_matrix(emb[q_idx], emb[g_idx])  # (nq, ng) 大きいほど近い
    rng = np.random.default_rng(SEED)
    # クラスがばらけるよう、別クラスのクエリを 3 件選ぶ
    pick = []
    for c in rng.permutation(np.unique(labels)):
        cand = np.where(labels[q_idx] == c)[0]
        if len(cand):
            pick.append(int(cand[0]))
        if len(pick) == 3:
            break

    fig, axes = plt.subplots(len(pick), K + 1, figsize=(1.5 * (K + 1), 1.6 * len(pick)))
    for r, qi in enumerate(pick):
        q_label = labels[q_idx][qi]
        # クエリ画像（先頭列）
        ax = axes[r][0]
        ax.imshow(np.asarray(images[q_idx[qi]]))
        ax.set_title("query\n" + names[q_label], fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color("black")
            sp.set_linewidth(1.5)
        # 上位 K 件の検索結果
        topk = np.argsort(-sim[qi])[:K]
        for j, gi in enumerate(topk):
            ax = axes[r][j + 1]
            ax.imshow(np.asarray(images[g_idx[gi]]))
            hit = labels[g_idx][gi] == q_label
            ax.set_title(f"#{j + 1} " + ("OK" if hit else "NG"), fontsize=7)
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_color("green" if hit else "red")
                sp.set_linewidth(2.5)
    fig.suptitle("Top-K retrieval with the compact metric-learned embedding")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 5. パイプライン本体
# ---------------------------------------------------------------------------
def main() -> None:
    os.makedirs(eh.OUT_DIR, exist_ok=True)
    device = eh.get_device()
    print(f"device = {device}")

    images, labels, names = eh.make_dataset(n_per_class=22, seed=0)
    g_idx, q_idx = split_gallery_query(labels, n_query_per_class=7)
    print(f"データセット: {len(images)} 枚 / {len(names)} クラス")
    print(f"ギャラリー {len(g_idx)} 枚 / クエリ {len(q_idx)} 枚\n")

    # --- (1) 2 つのバックボーンで素の埋め込みを抽出して比較 -----------------
    print("[1] バックボーンで特徴抽出（初回はモデルDLあり）…")
    resnet_emb = extract_timm_resnet18(images, device)  # (N, 512)
    vit_emb = extract_vit_cls(images, device)  # (N, 768)
    print(f"    timm resnet18 : {resnet_emb.shape}")
    print(f"    ViT CLS       : {vit_emb.shape}")

    raw = {
        "resnet18_gap_512d": evaluate(resnet_emb, g_idx, q_idx, labels),
        "vit_cls_768d": evaluate(vit_emb, g_idx, q_idx, labels),
    }
    print(f"\n[1] 素の高次元埋め込みの検索性能（kNN / Recall@{K}）")
    for name, m in raw.items():
        print(f"    {name:18s} kNN-acc={m['knn_acc']:.3f}  Recall@{K}={m['recall_at_k']:.3f}")
    raw_recall = raw["resnet18_gap_512d"]["recall_at_k"]

    # --- (2) 圧縮スイープ: 各次元でランダム射影 vs InfoNCE 学習済み射影 ------
    print(f"\n[2] resnet18 の 512 次元を {DIMS} へ圧縮し、ランダム射影 vs メトリック学習を比較")
    feats = F.normalize(torch.from_numpy(resnet_emb).float(), p=2, dim=1)  # 入力特徴も正規化
    rand_recall: dict[int, float] = {}
    learn_recall: dict[int, float] = {}
    sweep: dict[str, dict] = {}
    emb_before_small = emb_after_small = None
    for d in DIMS:
        # ランダム射影（学習なし）: 強い圧縮で同クラスが潰れやすい
        head0 = make_head(feats.shape[1], d)
        emb_rand = project(head0, feats)
        m_rand = evaluate(emb_rand, g_idx, q_idx, labels)
        # InfoNCE 学習済み射影: 同じ次元でも「同クラスを近く」並べ直す
        head = train_projection_head(feats, g_idx, labels, d)
        emb_learn = project(head, feats)
        m_learn = evaluate(emb_learn, g_idx, q_idx, labels)

        rand_recall[d] = m_rand["recall_at_k"]
        learn_recall[d] = m_learn["recall_at_k"]
        sweep[str(d)] = {"random": m_rand, "metric_learned": m_learn}
        print(
            f"    d={d:2d}  random: kNN={m_rand['knn_acc']:.3f} R@{K}={m_rand['recall_at_k']:.3f}"
            f"   | learned: kNN={m_learn['knn_acc']:.3f} R@{K}={m_learn['recall_at_k']:.3f}"
        )
        if d == DIMS[0]:  # 最小次元の埋め込みを可視化用に保持
            emb_before_small, emb_after_small = emb_rand, emb_learn

    # --- (3) 「Recall を保てる最小次元」でメトリック学習の効果をまとめる -----
    d_rand = smallest_dim_keeping(rand_recall, TARGET_RECALL)
    d_learn = smallest_dim_keeping(learn_recall, TARGET_RECALL)
    print(f"\n[3] Recall@{K} >= {TARGET_RECALL} を保てる最小次元")
    print(f"    ランダム射影    : {d_rand}")
    print(f"    メトリック学習  : {d_learn}  ← 同じ品質をより小さい埋め込みで達成（圧縮に強い）")

    # --- (4) 図と成果物（JSON）を保存 ---------------------------------------
    save_sweep_curve(
        rand_recall, learn_recall, raw_recall, os.path.join(eh.OUT_DIR, "mini_compression_sweep.png")
    )
    save_pca_before_after(
        emb_before_small, emb_after_small, labels, names, DIMS[0],
        os.path.join(eh.OUT_DIR, "mini_pca_before_after.png"),
    )
    # 検索グリッドは「Recall を保てる最小次元の学習済み埋め込み」で作る（実用の縮図）
    best_dim = d_learn if d_learn is not None else DIMS[-1]
    best_head = train_projection_head(feats, g_idx, labels, best_dim)
    best_emb = project(best_head, feats)
    save_retrieval_grid(
        images, best_emb, g_idx, q_idx, labels, names,
        os.path.join(eh.OUT_DIR, "mini_retrieval_grid.png"),
    )

    payload = {
        "engine": "timm resnet18 (512d) -> InfoNCE projection -> cosine kNN search",
        "n_images": int(len(images)),
        "n_gallery": int(len(g_idx)),
        "n_query": int(len(q_idx)),
        "k": K,
        "raw_baselines": {k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in raw.items()},
        "compression_sweep": {
            d: {
                method: {kk: round(vv, 4) for kk, vv in mv.items()}
                for method, mv in dv.items()
            }
            for d, dv in sweep.items()
        },
        "target_recall": TARGET_RECALL,
        "min_dim_keeping_recall": {"random_projection": d_rand, "metric_learned": d_learn},
        "retrieval_grid_dim": int(best_dim),
    }
    out_json = os.path.join(eh.OUT_DIR, "mini_metrics.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print("\n[4] 成果物を保存:")
    for fn in ["mini_compression_sweep.png", "mini_pca_before_after.png",
               "mini_retrieval_grid.png", "mini_metrics.json"]:
        print(f"    {eh.OUT_DIR}/{fn}")
    print("\n完了。抽出 → 評価 → メトリック学習を『コンパクト検索エンジン』として統合できました。")


if __name__ == "__main__":
    main()
