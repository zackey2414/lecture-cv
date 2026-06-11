"""03: メトリック学習 — Triplet / InfoNCE で「似たものを近く」に並べ替える。

学ぶこと:
  - メトリック学習 = 「ラベルが示す similarity に合うように埋め込み空間を学習する」こと。
    分類のように決定境界を引くのではなく、同クラスを近く・異クラスを遠くに「配置」する。
  - 凍結したバックボーン(timm resnet18)の 512 次元特徴を、学習可能な射影ヘッドで 2 次元に
    落とす。2 次元なのは (a) 直接プロットして空間を目で見るため (b) 圧縮で『良い配置』の
    効きが顕著に出るため。ランダムな 2 次元射影は 6 クラスが重なって kNN がボロボロだが、
    メトリック学習で並べ替えると分離する。
  - 損失を 3 つ実装して比較する:
      * TripletMarginLoss（ランダム負例）: アンカー/正例(同クラス)/負例(異クラス)の三つ組。
      * Triplet バッチハード（ハードネガティブ）: バッチ内で「最も遠い正例」と「最も近い負例」
        を選ぶ。難しい例ほど勾配が大きく、少ない反復で先に高精度へ到達しやすい（サンプル効率）。
      * InfoNCE（教師ありコントラスト）: 温度付き softmax で同クラスを引き寄せる。CLIP の
        画像-テキスト対照学習はこれの 2 モーダル版＝メトリック学習の一種。
  - 学習前後で kNN 分類精度 / Recall@k を比較し、収束曲線でハードネガティブの効果を定量化する
    （小規模・簡単なデータでは最終精度は拮抗するが、ハードネガティブは収束が速い）。

実行:
  uv run python lectures/15_image_embeddings_metric_learning/03_triplet_infonce.py
初回はモデル重みをダウンロード。学習は 2 次元ベクトル上の数百反復で CPU 数秒。
結果は outputs/ に保存。
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

import embed_helpers as eh

OUT_DIM = 2  # 射影先の次元。2 にして「空間そのもの」を可視化する
MARGIN = 0.3
EPOCHS = 300
LR = 1e-2
SEED = 0


# ---------------------------------------------------------------------------
# 1. 凍結バックボーンで特徴を一度だけ抽出（以降の学習はこの 512 次元ベクトル上で行う）
# ---------------------------------------------------------------------------
@torch.inference_mode()
def extract_frozen_features(images: list, device: torch.device) -> torch.Tensor:
    """timm resnet18(num_classes=0) でプール済み 512 次元特徴を取り、L2 正規化して返す。

    バックボーンは凍結（学習しない）。重い CNN を毎反復回さず、特徴を 1 回計算して
    使い回すことで、CPU でも数秒で済む。これは実務でも定番の高速化（特徴キャッシュ）。
    """
    model = timm.create_model("resnet18", pretrained=True, num_classes=0).to(device).eval()
    cfg = timm.data.resolve_data_config({}, model=model)
    transform = timm.data.create_transform(**cfg)
    feats = []
    for i in range(0, len(images), 16):
        x = torch.stack([transform(im) for im in images[i : i + 16]]).to(device)
        feats.append(model(x).cpu())
    return F.normalize(torch.cat(feats, 0), p=2, dim=1)  # 入力特徴も正規化しておく


# ---------------------------------------------------------------------------
# 2. 3 つの損失（メトリック学習の核）
# ---------------------------------------------------------------------------
def random_triplet_loss(z: torch.Tensor, labels: torch.Tensor, margin: float, rng) -> torch.Tensor:
    """ランダムに (anchor, positive=同クラス, negative=異クラス) を組んで TripletMarginLoss。

    triplet の狙い: d(a, p) + margin < d(a, n) になるよう、同クラスを近く・異クラスを遠くへ。
    ただし負例をランダムに選ぶと「すでに十分遠い簡単な負例」ばかりになり学びが鈍い。
    """
    labels_np = labels.numpy()
    a_idx, p_idx, n_idx = [], [], []
    for i in range(len(z)):
        c = int(labels_np[i])
        pos = np.where((labels_np == c) & (np.arange(len(z)) != i))[0]
        neg = np.where(labels_np != c)[0]
        a_idx.append(i)
        p_idx.append(int(rng.choice(pos)))
        n_idx.append(int(rng.choice(neg)))
    trip = nn.TripletMarginLoss(margin=margin)  # 既定は L2 距離
    return trip(z[a_idx], z[p_idx], z[n_idx])


def batch_hard_triplet_loss(z: torch.Tensor, labels: torch.Tensor, margin: float) -> torch.Tensor:
    """バッチハード Triplet: 各アンカーで「最も遠い正例」と「最も近い負例」を選ぶ。

    ハードネガティブ・マイニング = わざと難しい三つ組で学ぶ。難しい例ほど勾配が大きいので
    境界付近が早く整理され、ランダム負例より少ない反復で立ち上がりやすい（サンプル効率）。
    全ペア距離行列を作って一気に選ぶ（ベクトル化）。
    """
    dist = torch.cdist(z, z)  # (G, G) のユークリッド距離
    same = labels[:, None] == labels[None, :]  # 同クラス真偽
    # 最も遠い正例: 同クラスのうち距離最大（自分自身は同クラスだが距離0なので max には影響しない）
    pos_dist = (dist * same).max(dim=1).values
    # 最も近い負例: 異クラスのうち距離最小（同クラス位置を +inf にして min から除外）
    neg = dist.clone()
    neg[same] = float("inf")
    neg_dist = neg.min(dim=1).values
    # hinge: max(0, d(a,p) - d(a,n) + margin)
    return torch.relu(pos_dist - neg_dist + margin).mean()


def infonce_loss(z: torch.Tensor, labels: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """教師ありコントラスト(InfoNCE)。同クラスを正例集合とした温度付き softmax。

    各サンプルについて、同クラス（正例）の類似度を上げ、それ以外（負例）を下げる。
    CLIP の対照学習は「画像とそのキャプションを正例ペア、他を負例」とした InfoNCE で、
    まさにこのメトリック学習をモーダル間で行ったもの。
    """
    sim = (z @ z.T) / temperature  # 正規化後の内積=コサイン類似度 / 温度
    sim.fill_diagonal_(-1e9)  # 自分自身は対象外
    same = (labels[:, None] == labels[None, :]).float()
    same.fill_diagonal_(0)
    logp = F.log_softmax(sim, dim=1)
    # 各行で「正例位置の log 確率」の平均を最大化（= 負の平均を最小化）
    loss = -(same * logp).sum(1) / same.sum(1).clamp(min=1)
    return loss.mean()


# ---------------------------------------------------------------------------
# 3. 学習ループ（射影ヘッドだけを学習。バックボーンは凍結済み）
# ---------------------------------------------------------------------------
def make_head() -> nn.Linear:
    torch.manual_seed(SEED)  # 毎回同じ初期化にして、損失間の比較を公平にする
    return nn.Linear(512, OUT_DIM)


def project(head: nn.Linear, feats: torch.Tensor) -> np.ndarray:
    """射影ヘッドを通して L2 正規化した埋め込みを numpy で返す（評価・可視化用）。"""
    with torch.inference_mode():
        return F.normalize(head(feats), p=2, dim=1).cpu().numpy()


def train_head(
    mode: str, feats: torch.Tensor, g_idx: np.ndarray, labels: np.ndarray, eval_fn=None
) -> tuple[nn.Linear, list[tuple[int, float]]]:
    """指定モード(rand/hard/infonce)でギャラリーだけを使って射影ヘッドを学習する。

    eval_fn を渡すと 15 反復ごとにクエリ kNN を測り、収束曲線 [(epoch, knn), ...] を返す。
    """
    head = make_head()
    opt = torch.optim.Adam(head.parameters(), lr=LR)
    gi = torch.from_numpy(g_idx)
    glab = torch.from_numpy(labels[g_idx])
    rng = np.random.default_rng(0)
    curve: list[tuple[int, float]] = []
    for ep in range(EPOCHS):
        z = F.normalize(head(feats[gi]), p=2, dim=1)  # ギャラリーの埋め込み
        if mode == "rand":
            loss = random_triplet_loss(z, glab, MARGIN, rng)
        elif mode == "hard":
            loss = batch_hard_triplet_loss(z, glab, MARGIN)
        else:
            loss = infonce_loss(z, glab)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if eval_fn is not None and (ep % 15 == 0 or ep == EPOCHS - 1):
            curve.append((ep, eval_fn(project(head, feats))[0]))
    return head, curve


def scatter_2d(emb: np.ndarray, labels: np.ndarray, names: list[str], title: str, ax) -> None:
    for c, cname in enumerate(names):
        m = labels == c
        ax.scatter(emb[m, 0], emb[m, 1], s=14, label=cname)
    ax.set_title(title, fontsize=10)
    ax.set_aspect("equal")


def main() -> None:
    os.makedirs(eh.OUT_DIR, exist_ok=True)
    device = eh.get_device()
    print(f"device = {device}")

    images, labels, names = eh.make_dataset(n_per_class=20, seed=0)
    print(f"データセット: {len(images)} 枚 / {len(names)} クラス")
    print("凍結バックボーンで特徴抽出中（初回はモデルDLあり）…")
    feats = extract_frozen_features(images, device)

    # ギャラリー/クエリ分割（学習はギャラリーのみ、評価はクエリで）
    g_idx, q_idx = [], []
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        q_idx.extend(idx[-7:])
        g_idx.extend(idx[:-7])
    g_idx, q_idx = np.array(g_idx), np.array(q_idx)
    gl, ql = labels[g_idx], labels[q_idx]

    def evaluate(emb: np.ndarray) -> tuple[float, float]:
        acc = eh.knn_accuracy(emb[g_idx], gl, emb[q_idx], ql, k=5)
        rec = eh.recall_at_k(emb[g_idx], gl, emb[q_idx], ql, k=5)
        return acc, rec

    # 参考: 凍結 512 次元のまま（圧縮なし）の実力
    raw_acc, raw_rec = evaluate(feats.numpy())
    print(f"\n[参考] 凍結 512 次元そのまま : kNN-acc={raw_acc:.3f}  Recall@5={raw_rec:.3f}")

    # 学習前: ランダム初期化の 2 次元射影（6 クラスが重なって性能が出ない）
    head0 = make_head()
    emb_init = project(head0, feats)
    init_acc, init_rec = evaluate(emb_init)
    print(f"[学習前] ランダム 2D 射影     : kNN-acc={init_acc:.3f}  Recall@5={init_rec:.3f}  ← 出発点")

    # 3 つの損失で学習して比較（rand/hard は収束曲線も記録）
    print("\n[学習後] 2D 射影ヘッドをメトリック学習（ギャラリーのみで学習→クエリで評価）")
    results = {}
    emb_after = {}
    curves = {}
    for mode, label in [("rand", "Triplet (random neg)"), ("hard", "Triplet (hard neg)"), ("infonce", "InfoNCE")]:
        head, curve = train_head(mode, feats, g_idx, labels, eval_fn=evaluate)
        emb = project(head, feats)
        acc, rec = evaluate(emb)
        results[label] = (acc, rec)
        emb_after[mode] = emb
        curves[mode] = curve
        print(f"  {label:22s}: kNN-acc={acc:.3f}  Recall@5={rec:.3f}")

    # ハードネガティブの効果は「最終精度」ではなく「収束の速さ（サンプル効率）」で見るのが正しい。
    # 早い反復予算（30 反復時点）での kNN を比べる。
    def knn_at(curve, ep):
        return next((a for e, a in curve if e >= ep), curve[-1][1])

    rand_early, hard_early = knn_at(curves["rand"], 30), knn_at(curves["hard"], 30)
    print("\n[ハードネガティブの効果] 早い予算（30反復）での kNN")
    print(f"  random neg = {rand_early:.3f}   hard neg = {hard_early:.3f}   (差 {hard_early - rand_early:+.3f})")
    print("  ※ 小規模・簡単なデータでは最終精度は拮抗するが、ハードネガティブは先に立ち上がる。")

    # 学習前後の 2D 空間を並べて保存（最終 kNN が最良のモードを after に使う）
    best_mode = max(emb_after, key=lambda m: results[{"rand": "Triplet (random neg)", "hard": "Triplet (hard neg)", "infonce": "InfoNCE"}[m]][0])
    best_label = {"rand": "Triplet random", "hard": "Triplet hard", "infonce": "InfoNCE"}[best_mode]
    best_acc = evaluate(emb_after[best_mode])[0]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.0))
    scatter_2d(emb_init, labels, names, f"before (random 2D): kNN={init_acc:.2f}", axes[0])
    scatter_2d(emb_after[best_mode], labels, names, f"after {best_label}: kNN={best_acc:.2f}", axes[1])
    axes[1].legend(fontsize=7, loc="upper right")
    fig.suptitle("Metric learning reshapes the 2D embedding space")
    fig.tight_layout()
    fig.savefig(os.path.join(eh.OUT_DIR, "03_embedding_before_after.png"), dpi=110)
    plt.close(fig)

    # 収束曲線（random neg vs hard neg）
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for mode, lab in [("rand", "random neg"), ("hard", "hard neg")]:
        ep, ac = zip(*curves[mode])
        ax.plot(ep, ac, marker="o", ms=3, label=lab)
    ax.set_xlabel("epoch")
    ax.set_ylabel("query kNN accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("Convergence: hard-negative mining rises faster early")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(eh.OUT_DIR, "03_hardneg_convergence.png"), dpi=110)
    plt.close(fig)

    print("\nまとめ:")
    print("  - ランダム 2D 射影では 6 クラスが重なり kNN が低い（圧縮で情報が潰れる）。")
    print("  - Triplet/InfoNCE で『同クラスを近く』に学習すると 2D でも分離し、kNN/Recall が上がる。")
    print("  - ハードネガティブは難しい例から学ぶため、少ない反復で先に高精度へ（サンプル効率）。")
    print("    ただし難例に過集中して不安定化することもあるので、実務では semi-hard 等で和らげる。")
    print("  - CLIP の画像-テキスト対照学習は、この InfoNCE をモーダル間に広げたもの（16回で詳説）。")
    print(f"\n完了。{eh.OUT_DIR}/03_embedding_before_after.png と 03_hardneg_convergence.png を確認してください。")


if __name__ == "__main__":
    main()
