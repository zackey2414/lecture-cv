"""02: 埋め込み回帰による CLIP 蒸留（この章の中心）。

teacher CLIP の画像埋め込みを、小型 student CNN に「真似させる」蒸留を行う。
損失は (1 - コサイン類似度) + MSE。ラベルは一切使わず、teacher の埋め込みだけを
教師信号にする点が CLIP 蒸留の肝（= response/embedding distillation の VLM 版）。

ポイント:
  - teacher は eval() + inference_mode() で凍結し、埋め込みを一度だけ前計算してキャッシュ
  - student は 64px の安い入力から teacher(224px) の 512 次元埋め込みを再現する
  - 学習前後で teacher との整合度（コサイン）がどれだけ上がるかを測る

実行:
    uv run python lectures/39_clip_distillation/02_embedding_distillation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import clip_distill_helpers as H  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402


def precompute_teacher_targets(model, processor, images, device):
    """teacher の画像埋め込みを前計算（凍結なので 1 回でよい）。"""
    return H.teacher_image_embeds(model, processor, images, device)


def train_student(student, X, targets, epochs: int = 40, bs: int = 16, lr: float = 2e-3):
    """student を埋め込み蒸留損失で学習し、エポックごとの損失/整合度を記録する。"""
    opt = torch.optim.AdamW(student.parameters(), lr=lr)
    n = X.shape[0]
    history = {"loss": [], "cos": [], "mse": [], "align": []}

    for _ in range(epochs):
        student.train()
        perm = torch.randperm(n)
        ep_loss = ep_cos = ep_mse = 0.0
        nb = 0
        for i in range(0, n, bs):
            idx = perm[i : i + bs]
            out = student(X[idx])
            loss, cos, mse = H.embedding_distill_loss(out, targets[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += float(loss.detach())
            ep_cos += float(cos)
            ep_mse += float(mse)
            nb += 1

        # エポック末に train 全体での teacher 整合度を測る
        student.eval()
        with torch.inference_mode():
            align = H.mean_cosine_alignment(student(X), targets)
        history["loss"].append(ep_loss / nb)
        history["cos"].append(ep_cos / nb)
        history["mse"].append(ep_mse / nb)
        history["align"].append(align)
    return history


def plot_history(history) -> None:
    """損失と teacher 整合度の推移を保存。"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.5))
    ax1.plot(history["loss"], label="total")
    ax1.plot(history["cos"], label="1-cos")
    ax1.plot(history["mse"], label="mse")
    ax1.set_title("Distillation loss")
    ax1.set_xlabel("epoch")
    ax1.legend(fontsize=8)

    ax2.plot(history["align"], color="tab:green")
    ax2.set_title("Student-Teacher cosine alignment")
    ax2.set_xlabel("epoch")
    ax2.set_ylim(0, 1.02)
    saved = H.save_fig(fig, "02_distillation_curves.png")
    print(f"[plot] saved -> {saved}")


def main() -> int:
    H.set_seed(0)
    device = H.pick_device()

    # データ: 学習用と評価用を別シードで生成（汎化を見る）
    tr_imgs, tr_lab = H.make_dataset(n_per_class=30, seed=1)
    te_imgs, te_lab = H.make_dataset(n_per_class=10, seed=2)
    print(f"[data] train={len(tr_imgs)} test={len(te_imgs)}")

    print("[load] teacher CLIP ...")
    model, processor, device = H.load_teacher(device)

    # teacher 埋め込み（教師信号）を前計算してキャッシュ
    tr_targets = precompute_teacher_targets(model, processor, tr_imgs, device)
    te_targets = precompute_teacher_targets(model, processor, te_imgs, device)

    # student 入力テンソル
    Xtr = H.student_preprocess(tr_imgs, device)
    Xte = H.student_preprocess(te_imgs, device)

    student = H.StudentCNN().to(device)
    t_params = H.count_params(model.vision_model) + model.visual_projection.weight.numel()
    s_params = H.count_params(student)
    print(f"[size] teacher image tower={t_params:,}  student={s_params:,}  "
          f"compression x{t_params / s_params:.0f}")

    # 学習前の整合度（ランダム初期化なのでほぼ 0 付近）
    student.eval()
    with torch.inference_mode():
        before = H.mean_cosine_alignment(student(Xte), te_targets)
    print(f"[before] test alignment={before:.3f}")

    print("[train] embedding distillation (cosine + mse) ...")
    history = train_student(student, Xtr, tr_targets)

    student.eval()
    with torch.inference_mode():
        after = H.mean_cosine_alignment(student(Xte), te_targets)
    print(f"[after]  test alignment={after:.3f}  (train final={history['align'][-1]:.3f})")

    plot_history(history)

    # student の重みを保存（03 / mini_project から再利用可能にする）
    ckpt = H.out_dir() / "student_distilled.pt"
    torch.save(student.state_dict(), ckpt)
    print(f"[save] student weights -> {ckpt}")

    # 自己検証: 蒸留で整合度が明確に上がっていること
    assert after > before + 0.3, f"蒸留で整合度が十分上がっていません: {before:.3f} -> {after:.3f}"
    assert after >= 0.8, f"最終整合度が低すぎます: {after:.3f}"
    print("[done] 小型 student が teacher の埋め込み空間を再現できた")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
