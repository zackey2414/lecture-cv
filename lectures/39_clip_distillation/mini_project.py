"""章末ミニプロジェクト: 小型 student への CLIP 蒸留と「保持率」評価。

統合課題:
  teacher CLIP を 551 倍小さい student CNN に埋め込み蒸留し、
  「ゼロショット分類精度・画像検索 Recall@k・teacher との埋め込み整合度」が
  どれだけ保たれるか（保持率, retention）を一気通貫で評価する。

出力:
  - コンソールに評価テーブル（teacher vs student, before vs after）
  - lectures/39_clip_distillation/outputs/mini_project_summary.png（メトリクス棒グラフ）

実行:
    uv run python lectures/39_clip_distillation/mini_project.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import clip_distill_helpers as H  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402


def evaluate(img_emb, txt_emb, labels, logit_scale, teacher_pred=None):
    """1 つの埋め込み集合に対する評価指標をまとめて返す。"""
    acc = H.zeroshot_accuracy(img_emb, txt_emb, labels, logit_scale)
    r1 = H.retrieval_recall_at_k(img_emb, labels, k=1)
    r5 = H.retrieval_recall_at_k(img_emb, labels, k=5)
    pred = H.zeroshot_predict(img_emb, txt_emb, logit_scale)
    agree = None if teacher_pred is None else H.agreement(pred, teacher_pred)
    return {"acc": acc, "r1": r1, "r5": r5, "pred": pred, "agree": agree}


def train_distilled_student(Xtr, tr_targets, epochs=45, bs=16, lr=2e-3):
    """埋め込み蒸留で student を学習して返す。"""
    student = H.StudentCNN()
    opt = torch.optim.AdamW(student.parameters(), lr=lr)
    n = Xtr.shape[0]
    for _ in range(epochs):
        student.train()
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i : i + bs]
            loss, _, _ = H.embedding_distill_loss(student(Xtr[idx]), tr_targets[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    student.eval()
    return student


def plot_summary(teacher_m, before_m, after_m, align_before, align_after) -> None:
    """teacher / 蒸留前 / 蒸留後 のメトリクスを棒グラフで比較。"""
    labels = ["zero-shot acc", "retrieval R@1", "retrieval R@5", "align w/ teacher"]
    teacher_vals = [teacher_m["acc"], teacher_m["r1"], teacher_m["r5"], 1.0]
    before_vals = [before_m["acc"], before_m["r1"], before_m["r5"], align_before]
    after_vals = [after_m["acc"], after_m["r1"], after_m["r5"], align_after]

    x = np.arange(len(labels))
    w = 0.26
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - w, teacher_vals, w, label="teacher (151M)", color="tab:gray")
    ax.bar(x, before_vals, w, label="student before KD", color="tab:red")
    ax.bar(x + w, after_vals, w, label="student after KD (160K)", color="tab:green")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("score")
    ax.set_title("CLIP distillation: metric retention (toy synthetic data)")
    ax.legend(fontsize=8)
    saved = H.save_fig(fig, "mini_project_summary.png")
    print(f"[plot] saved -> {saved}")


def main() -> int:
    H.set_seed(0)
    device = H.pick_device()

    # 1) データ
    tr_imgs, tr_lab = H.make_dataset(n_per_class=30, seed=1)
    te_imgs, te_lab = H.make_dataset(n_per_class=12, seed=2)
    print(f"[data] train={len(tr_imgs)} test={len(te_imgs)} classes={H.CLASS_NAMES}")

    # 2) teacher 埋め込み（教師信号）とテキスト埋め込み
    print("[load] teacher CLIP (openai/clip-vit-base-patch32) ...")
    model, processor, device = H.load_teacher(device)
    ls = H.teacher_logit_scale(model)
    tr_targets = H.teacher_image_embeds(model, processor, tr_imgs, device)
    te_teacher = H.teacher_image_embeds(model, processor, te_imgs, device)
    txt_emb = H.teacher_text_embeds(model, processor, H.CLASS_PROMPTS, device)

    Xtr = H.student_preprocess(tr_imgs, device)
    Xte = H.student_preprocess(te_imgs, device)

    # 3) 蒸留前(ランダム初期化) student の評価
    rnd = H.StudentCNN().eval()
    with torch.inference_mode():
        before_emb = rnd(Xte)
    align_before = H.mean_cosine_alignment(before_emb, te_teacher)

    # 4) 蒸留学習
    print("[train] embedding distillation ...")
    student = train_distilled_student(Xtr, tr_targets)
    with torch.inference_mode():
        after_emb = student(Xte)
    align_after = H.mean_cosine_alignment(after_emb, te_teacher)

    # 5) 評価
    teacher_m = evaluate(te_teacher, txt_emb, te_lab, ls)
    before_m = evaluate(before_emb, txt_emb, te_lab, ls, teacher_pred=teacher_m["pred"])
    after_m = evaluate(after_emb, txt_emb, te_lab, ls, teacher_pred=teacher_m["pred"])

    retention = after_m["acc"] / max(teacher_m["acc"], 1e-9)

    # 6) レポート
    print("\n================ 評価テーブル ================")
    print(f"{'metric':<22}{'teacher':>10}{'before KD':>12}{'after KD':>12}")
    print(f"{'zero-shot acc':<22}{teacher_m['acc']:>10.3f}{before_m['acc']:>12.3f}{after_m['acc']:>12.3f}")
    print(f"{'retrieval R@1':<22}{teacher_m['r1']:>10.3f}{before_m['r1']:>12.3f}{after_m['r1']:>12.3f}")
    print(f"{'retrieval R@5':<22}{teacher_m['r5']:>10.3f}{before_m['r5']:>12.3f}{after_m['r5']:>12.3f}")
    print(f"{'cosine align':<22}{1.0:>10.3f}{align_before:>12.3f}{align_after:>12.3f}")
    print(f"{'agreement w/ teacher':<22}{'-':>10}{before_m['agree']:>12.3f}{after_m['agree']:>12.3f}")
    print("---------------------------------------------")
    print(f"zero-shot 保持率 (student/teacher) = {retention:.1%}")
    print(f"パラメータ圧縮: teacher画像タワー 87.8M -> student {H.count_params(student)/1e3:.0f}K "
          f"(x{87_849_216 / H.count_params(student):.0f})")
    print("=============================================\n")

    # 7) 検索の定性例: テキストクエリ -> 最も似た test 画像のクラス
    logits = H.zeroshot_logits(after_emb, txt_emb, ls)  # [N, C]
    for c in range(H.NUM_CLASSES):
        top_img = int(logits[:, c].argmax())
        print(f"[retrieval] query='{H.CLASS_PROMPTS[c]}' -> top image is class '{H.CLASS_NAMES[te_lab[top_img]]}'")

    plot_summary(teacher_m, before_m, after_m, align_before, align_after)

    # 8) 自己検証
    assert align_after > align_before + 0.3, "蒸留で整合度が上がっていない"
    assert retention >= 0.9, f"ゼロショット保持率が低い: {retention:.2f}"
    assert after_m["r5"] >= 0.9, f"検索 R@5 が低い: {after_m['r5']:.2f}"
    print("[done] 小型 student が teacher の分類・検索・埋め込みを高い保持率で再現した")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
