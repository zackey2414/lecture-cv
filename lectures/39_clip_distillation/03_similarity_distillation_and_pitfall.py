"""03: 類似度行列の蒸留（contrastive distillation）と「スケールずれ」の落とし穴。

02 は teacher の画像埋め込みを直接回帰した。実際の CLIP/TinyCLIP/MobileCLIP 系では、
teacher の *画像-テキスト類似度行列*（logit_scale で温度を付けたもの）を student に
KL ダイバージェンスで模倣させる「親和性(affinity)蒸留」がよく使われる。

ここでは 2 つを学ぶ:
  (A) 類似度行列蒸留: soft = KL(softmax(teacher行列) || log_softmax(student行列))
  (B) 落とし穴: 埋め込みを L2 正規化しない / logit_scale を揃えないと、行列のスケールが
      teacher と student で全く合わず、KL が破綻する。これを数値で見せる。

実行:
    uv run python lectures/39_clip_distillation/03_similarity_distillation_and_pitfall.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import clip_distill_helpers as H  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402


def teacher_affinity(img_emb, txt_emb, logit_scale: float) -> torch.Tensor:
    """teacher の画像-テキスト類似度行列（温度付き logits）。正準形。"""
    return H.zeroshot_logits(img_emb, txt_emb, logit_scale)  # [N, C]


def kl_affinity_loss(student_logits, teacher_logits) -> torch.Tensor:
    """親和性蒸留の損失。

    入力 = log_softmax(student)、ターゲット = softmax(teacher)、reduction='batchmean'。
    38 章で学んだ Hinton 蒸留の KL と同じ向き・同じ約束（順序を逆にすると壊れる）。
    """
    return F.kl_div(
        F.log_softmax(student_logits, dim=1),
        F.softmax(teacher_logits, dim=1),
        reduction="batchmean",
    )


def demo_scale_mismatch(img_emb, txt_emb, logit_scale: float) -> None:
    """(B) 正規化・温度を揃えないと行列スケールがどれだけずれるかを数値で示す。"""
    print("\n=== 落とし穴: 正規化 / logit_scale を揃えないとスケールがずれる ===")

    # 正しい行列（正規化済み x logit_scale）
    correct = teacher_affinity(img_emb, txt_emb, logit_scale)

    # 誤り 1: 正規化を忘れて生の内積（埋め込みのノルムが効いてしまう）
    raw_img = img_emb * 11.0  # 正規化前は長さ 10〜12 程度だったことを再現
    raw_txt = txt_emb * 11.0
    wrong_norm = raw_img @ raw_txt.t()  # logit_scale も付けない

    # 誤り 2: 正規化はしたが logit_scale を付け忘れ（コサインそのまま [-1,1]）
    wrong_scale = F.normalize(img_emb, dim=-1) @ F.normalize(txt_emb, dim=-1).t()

    print(f"  正しい行列   : 範囲 [{correct.min():.2f}, {correct.max():.2f}]  (logit_scale={logit_scale:.0f})")
    print(f"  正規化忘れ   : 範囲 [{wrong_norm.min():.2f}, {wrong_norm.max():.2f}]  (ノルムに依存しスケール暴走)")
    print(f"  logit_scale忘れ: 範囲 [{wrong_scale.min():.2f}, {wrong_scale.max():.2f}]  (温度が無く softmax がほぼ一様)")

    # softmax の鋭さの違い: 温度を付けないと確率が平坦になり「教師信号」が薄まる
    p_correct = correct.softmax(1).max(1).values.mean()
    p_flat = wrong_scale.softmax(1).max(1).values.mean()
    print(f"  最大確率の平均: 正しい={p_correct:.3f} / logit_scale忘れ={p_flat:.3f}")
    print("  -> logit_scale を揃えないと teacher のソフトターゲットが平坦化し蒸留が効かない")


def train_affinity(student, X, img_targets, txt_emb, logit_scale, epochs=40, bs=16, lr=2e-3):
    """類似度行列蒸留で student を学習する。"""
    opt = torch.optim.AdamW(student.parameters(), lr=lr)
    n = X.shape[0]
    txt_n = F.normalize(txt_emb, dim=-1)
    hist = []
    for _ in range(epochs):
        student.train()
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i : i + bs]
            s_emb = F.normalize(student(X[idx]), dim=-1)
            s_logits = logit_scale * s_emb @ txt_n.t()
            t_logits = logit_scale * F.normalize(img_targets[idx], dim=-1) @ txt_n.t()
            loss = kl_affinity_loss(s_logits, t_logits)
            opt.zero_grad()
            loss.backward()
            opt.step()
        student.eval()
        with torch.inference_mode():
            s_all = logit_scale * F.normalize(student(X), dim=-1) @ txt_n.t()
            t_all = logit_scale * F.normalize(img_targets, dim=-1) @ txt_n.t()
            hist.append(float(kl_affinity_loss(s_all, t_all)))
    return hist


def main() -> int:
    H.set_seed(0)
    device = H.pick_device()

    tr_imgs, tr_lab = H.make_dataset(n_per_class=30, seed=1)
    te_imgs, te_lab = H.make_dataset(n_per_class=10, seed=2)

    print("[load] teacher CLIP ...")
    model, processor, device = H.load_teacher(device)
    logit_scale = H.teacher_logit_scale(model)

    tr_img = H.teacher_image_embeds(model, processor, tr_imgs, device)
    te_img = H.teacher_image_embeds(model, processor, te_imgs, device)
    txt_emb = H.teacher_text_embeds(model, processor, H.CLASS_PROMPTS, device)

    # (B) まず落とし穴を体感
    demo_scale_mismatch(te_img, txt_emb, logit_scale)

    # (A) 親和性蒸留で student を学習
    print("\n=== 親和性(類似度行列)蒸留 ===")
    Xtr = H.student_preprocess(tr_imgs, device)
    Xte = H.student_preprocess(te_imgs, device)
    student = H.StudentCNN().to(device)
    hist = train_affinity(student, Xtr, tr_img, txt_emb, logit_scale)

    student.eval()
    with torch.inference_mode():
        se = student(Xte)
    sacc = H.zeroshot_accuracy(se, txt_emb, te_lab, logit_scale)
    tacc = H.zeroshot_accuracy(te_img, txt_emb, te_lab, logit_scale)
    spred = H.zeroshot_predict(se, txt_emb, logit_scale)
    tpred = H.zeroshot_predict(te_img, txt_emb, logit_scale)
    print(f"[affinity] KL {hist[0]:.4f} -> {hist[-1]:.4f}")
    print(f"[affinity] student acc={sacc:.3f}  teacher acc={tacc:.3f}  agreement={H.agreement(spred, tpred):.3f}")

    # 図: KL の推移
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(hist, color="tab:purple")
    ax.set_title("Affinity (KL) distillation loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("KL divergence")
    saved = H.save_fig(fig, "03_affinity_kl.png")
    print(f"[plot] saved -> {saved}")

    assert hist[-1] < hist[0], "KL が下がっていません（蒸留が進んでいない）"
    assert H.agreement(spred, tpred) >= 0.8, "student が teacher と十分一致していません"
    print("[done] 類似度行列蒸留と、正規化/logit_scale を揃える重要性を確認した")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
