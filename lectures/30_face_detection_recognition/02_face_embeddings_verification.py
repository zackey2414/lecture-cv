"""02: 顔埋め込み（認識）と 1:1 照合 / 1:N 識別。

学ぶこと:
  - 顔を固定長ベクトル（埋め込み）に変換し、**コサイン類似度**で「同一人物か」を測る。
    本講座は依存衝突を避けるため insightface(ArcFace) ではなく torchvision ResNet18 の
    512 次元特徴を「顔埋め込みの代用」に使う（`FaceEncoder` を差し替えれば ArcFace になる）。
  - 埋め込みは必ず **L2 正規化** してから比較する（向きだけ見る＝明るさ等に頑健）。
  - **1:1 照合(verification)**: 2 枚が同一人物か? ＝ コサイン類似度をしきい値で二値判定。
    全ペアを genuine/impostor に分け、ROC・EER・TAR@FAR で品質を測る。
  - **1:N 識別(identification)**: プローブが登録済み N 人の誰か? ＝ ギャラリー全員と照合し、
    類似度上位に正解が来るかを CMC（rank-1, rank-5）で測る。
  - 整列(alignment)の話: ArcFace 等は顔ランドマークで目・口を定位置に揃えてから埋め込む。
    本章の合成顔はすでに中央寄せ済み（＝整列済み相当）。整列が崩れると埋め込みが劣化する。

実行:
  uv run python lectures/30_face_detection_recognition/02_face_embeddings_verification.py
初回は ResNet18 重みを DL（以後キャッシュ）。結果は lectures/30_.../outputs/ に保存。
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")  # headless でも図を保存できるよう pyplot より前に設定
import matplotlib.pyplot as plt
import numpy as np

import face_lab as fl


def plot_score_distributions(scores: np.ndarray, is_genuine: np.ndarray, path: str) -> None:
    """genuine と impostor のスコア分布ヒストグラムを重ね描きして保存。

    2 つの山が離れているほど『どこかで線を引けば綺麗に分かれる』＝良い埋め込み。
    重なっている領域が、しきい値をどう選んでも避けられない誤り（EER 付近）に対応する。
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    bins = np.linspace(scores.min(), scores.max(), 40)
    ax.hist(
        scores[is_genuine], bins=bins, alpha=0.6, label="genuine (same person)", color="tab:green"
    )
    ax.hist(
        scores[~is_genuine], bins=bins, alpha=0.6, label="impostor (different)", color="tab:red"
    )
    ax.set_xlabel("cosine similarity")
    ax.set_ylabel("count")
    ax.set_title("verification score distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_roc(far: np.ndarray, tar: np.ndarray, eer: float, path: str) -> None:
    """ROC 曲線（横 FAR・縦 TAR）と EER 点・対角線を描いて保存。"""
    fig, ax = plt.subplots(figsize=(5.2, 5))
    ax.plot(far, tar, color="tab:blue", lw=2, label="ROC")
    ax.plot([0, 1], [1, 0], ls=":", color="gray", label="FAR=FRR (EER line)")
    ax.scatter([eer], [1 - eer], color="tab:red", zorder=5, label=f"EER={eer:.3f}")
    ax.set_xlabel("FAR (False Accept Rate)")
    ax.set_ylabel("TAR (True Accept Rate)")
    ax.set_title("ROC curve (1:1 verification)")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_cmc(cmc: np.ndarray, path: str) -> None:
    """CMC 曲線（rank-k 累積識別率）を保存。"""
    fig, ax = plt.subplots(figsize=(5.2, 4))
    ranks = np.arange(1, len(cmc) + 1)
    ax.plot(ranks, cmc, marker="o", color="tab:purple")
    ax.set_xlabel("rank k")
    ax.set_ylabel("identification rate")
    ax.set_title("CMC curve (1:N identification)")
    ax.set_ylim(0, 1.02)
    ax.set_xticks(ranks)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def verification_demo(emb: np.ndarray, labels: np.ndarray) -> None:
    """1:1 照合の評価（ROC / EER / TAR@FAR）を計算・可視化・印字する。"""
    scores, is_genuine = fl.verification_scores(emb, labels)
    n_gen, n_imp = int(is_genuine.sum()), int((~is_genuine).sum())
    print(f"  全ペア {len(scores)} 組 = genuine {n_gen} + impostor {n_imp}")

    thrs, far, tar = fl.roc_far_tar(scores, is_genuine)
    eer, eer_thr = fl.compute_eer(far, tar, thrs)
    tar1, thr1 = fl.tar_at_far(far, tar, thrs, 0.01)
    tar3, thr3 = fl.tar_at_far(far, tar, thrs, 0.001)
    print(f"  EER          = {eer:.3f}  (しきい値 {eer_thr:+.3f})")
    print(f"  TAR@FAR=1e-2 = {tar1:.3f}  (しきい値 {thr1:+.3f})")
    print(f"  TAR@FAR=1e-3 = {tar3:.3f}  (しきい値 {thr3:+.3f})")
    print("  → 認証要件として FAR を固定し、その下で TAR を最大化するのが顔認証の基本。")

    plot_score_distributions(scores, is_genuine, os.path.join(fl.OUT_DIR, "02_score_dist.png"))
    plot_roc(far, tar, eer, os.path.join(fl.OUT_DIR, "02_roc.png"))

    # EER しきい値での判定例（accept/reject が直感に合うか）
    nrm = fl.l2_normalize(emb)
    same = float(nrm[0] @ nrm[1])  # 同一人物の 0,1 枚目
    diff = float(nrm[0] @ nrm[-1])  # 別人（最後の人）の 1 枚
    print(
        f"  判定例(しきい値 {eer_thr:+.3f}): 同一人物ペア sim={same:+.3f} → "
        f"{'ACCEPT' if same >= eer_thr else 'REJECT'} / "
        f"別人ペア sim={diff:+.3f} → {'ACCEPT' if diff >= eer_thr else 'REJECT'}"
    )


def identification_demo(emb: np.ndarray, labels: np.ndarray) -> None:
    """1:N 識別の評価（gallery=登録1枚/人, probe=残り）。CMC と rank-1 を出す。"""
    # 各人物の先頭 1 枚を「登録(ギャラリー)」、残りを「照会(プローブ)」にする
    is_gallery = np.zeros(len(labels), dtype=bool)
    for pid in np.unique(labels):
        is_gallery[np.where(labels == pid)[0][0]] = True

    g_emb, g_lab = emb[is_gallery], labels[is_gallery]
    p_emb, p_lab = emb[~is_gallery], labels[~is_gallery]
    cmc = fl.identification_cmc(g_emb, g_lab, p_emb, p_lab, max_rank=5)
    print(f"  ギャラリー {len(g_lab)} 人 / プローブ {len(p_lab)} 枚")
    print(f"  rank-1 = {cmc[0]:.3f}   rank-5 = {cmc[-1]:.3f}")
    print("  → rank-1 は『最も似た 1 人が本人だった割合』。CMC は rank を緩めた累積識別率。")
    plot_cmc(cmc, os.path.join(fl.OUT_DIR, "02_cmc.png"))


def main() -> None:
    os.makedirs(fl.OUT_DIR, exist_ok=True)
    device = fl.get_device()
    print(f"device = {device}")

    images, labels, names = fl.make_face_dataset(n_ids=8, n_per_id=6, seed=0)
    print(f"合成顔: {len(images)} 枚 / {len(names)} 人（各 6 枚）")

    enc = fl.FaceEncoder(device)
    emb = enc.encode(images)
    print(f"埋め込み: mode={enc.mode}, shape={emb.shape}（L2 正規化して比較）\n")

    print("[1:1 照合 verification]")
    verification_demo(emb, labels)

    print("\n[1:N 識別 identification]")
    identification_demo(emb, labels)

    print(f"\n完了。lectures/{fl.MODULE_ID}/outputs/02_*.png（分布/ROC/CMC）を確認してください。")


if __name__ == "__main__":
    main()
