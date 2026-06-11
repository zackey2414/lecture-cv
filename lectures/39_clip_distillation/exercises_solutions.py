"""39_clip_distillation 演習 — 模範解答（すべて PASS）。

exercises.py の TODO を埋めた完成版。自己採点ハーネスで全問 PASS を確認できる。

実行:
    uv run python lectures/39_clip_distillation/exercises_solutions.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import clip_distill_helpers as H  # noqa: E402


# ===========================================================================
# 解答（学習者は exercises.py でこれらを実装する）
# ===========================================================================


def ex1_l2_normalize(x: torch.Tensor) -> torch.Tensor:
    """Q1(易): 行ごとに L2 正規化する。各行のノルムが 1 になる。"""
    return F.normalize(x, dim=-1)


def ex2_cosine_logits(img: torch.Tensor, txt: torch.Tensor, logit_scale: float) -> torch.Tensor:
    """Q2(易): 画像-テキストの温度付き類似度行列を返す。

    必ず両者を L2 正規化してから内積し、logit_scale を掛ける（CLIP の正準形）。
    """
    s = F.normalize(img, dim=-1)
    t = F.normalize(txt, dim=-1)
    return logit_scale * s @ t.t()


def ex3_predict(logits: torch.Tensor) -> np.ndarray:
    """Q3(易): クラス logits から予測クラス index を返す。"""
    return logits.argmax(dim=1).cpu().numpy()


def ex4_distill_loss(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    """Q4(中): 埋め込み蒸留損失 = (1 - cos) + mse（正規化してから）。

    student == teacher のとき 0 になる。
    """
    s = F.normalize(student, dim=-1)
    t = F.normalize(teacher, dim=-1)
    cos_loss = (1.0 - (s * t).sum(dim=-1)).mean()
    mse_loss = F.mse_loss(s, t)
    return cos_loss + mse_loss


def ex5_kl_affinity(student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
    """Q5(中): 親和性蒸留の KL。

    入力 = log_softmax(student)、ターゲット = softmax(teacher)、reduction='batchmean'。
    順序を逆にしたり log を付け忘れると壊れる。
    """
    return F.kl_div(
        F.log_softmax(student_logits, dim=1),
        F.softmax(teacher_logits, dim=1),
        reduction="batchmean",
    )


def ex6_retention(student_acc: float, teacher_acc: float) -> float:
    """Q6(中): 保持率 = student_acc / teacher_acc（teacher が 0 のときは 0）。"""
    if teacher_acc <= 0:
        return 0.0
    return student_acc / teacher_acc


def ex7_distill_step(student, optimizer, X, targets) -> float:
    """Q7(難): 蒸留の 1 ステップ（forward -> loss -> backward -> step）。loss を返す。"""
    student.train()
    out = student(X)
    loss = ex4_distill_loss(out, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return float(loss.detach())


def ex8_recall_at_k(img_emb: torch.Tensor, labels: np.ndarray, k: int) -> float:
    """Q8(難): 画像->画像検索の Recall@k（leave-one-out, 同一クラスが上位 k に入る割合）。"""
    s = F.normalize(img_emb, dim=-1)
    sim = (s @ s.t()).cpu().numpy()
    np.fill_diagonal(sim, -np.inf)
    n = sim.shape[0]
    hits = 0
    for i in range(n):
        topk = np.argsort(-sim[i])[:k]
        if np.any(labels[topk] == labels[i]):
            hits += 1
    return hits / n


# ===========================================================================
# 自己採点ハーネス（exercises.py と共通）
# ===========================================================================


def _build_context():
    """採点に使う teacher 埋め込み等を一度だけ用意する。"""
    H.set_seed(0)
    device = H.pick_device()
    imgs, labels = H.make_dataset(n_per_class=8, seed=1)
    model, processor, device = H.load_teacher(device)
    ctx = {
        "device": device,
        "labels": labels,
        "logit_scale": H.teacher_logit_scale(model),
        "img_emb": H.teacher_image_embeds(model, processor, imgs, device),
        "txt_emb": H.teacher_text_embeds(model, processor, H.CLASS_PROMPTS, device),
        "X": H.student_preprocess(imgs, device),
    }
    return ctx


def _check1(ctx):
    x = torch.randn(5, 16)
    out = ex1_l2_normalize(x)
    norms = out.norm(dim=-1)
    assert torch.allclose(norms, torch.ones(5), atol=1e-5), f"ノルムが 1 でない: {norms}"


def _check2(ctx):
    logits = ex2_cosine_logits(ctx["img_emb"], ctx["txt_emb"], ctx["logit_scale"])
    assert logits.shape == (ctx["img_emb"].shape[0], H.NUM_CLASSES), f"shape 不正: {logits.shape}"
    # logit_scale=100 で正規化済みコサインを掛けるので最大値は ~100 以下
    assert 1.0 < float(logits.max()) <= ctx["logit_scale"] + 1e-3, "スケールが logit_scale と不整合"


def _check3(ctx):
    logits = ex2_cosine_logits(ctx["img_emb"], ctx["txt_emb"], ctx["logit_scale"])
    pred = ex3_predict(logits)
    acc = float((pred == ctx["labels"]).mean())
    assert acc >= 0.9, f"teacher のゼロショット精度が低い: {acc}"


def _check4(ctx):
    t = ctx["img_emb"]
    assert float(ex4_distill_loss(t, t)) < 1e-5, "同一埋め込みで損失が 0 でない"
    noisy = t + 0.5 * torch.randn_like(t)
    assert float(ex4_distill_loss(noisy, t)) > float(ex4_distill_loss(t, t)), "ノイズで損失が増えない"


def _check5(ctx):
    tl = ex2_cosine_logits(ctx["img_emb"], ctx["txt_emb"], ctx["logit_scale"])
    assert float(ex5_kl_affinity(tl, tl)) < 1e-5, "同一行列で KL が 0 でない"
    wrong = tl + 5.0 * torch.randn_like(tl)
    assert float(ex5_kl_affinity(wrong, tl)) > 1e-3, "異なる行列で KL が増えない"


def _check6(ctx):
    assert abs(ex6_retention(0.8, 1.0) - 0.8) < 1e-9
    assert ex6_retention(0.5, 0.0) == 0.0
    assert abs(ex6_retention(0.9, 0.9) - 1.0) < 1e-9


def _check7(ctx):
    torch.manual_seed(0)
    student = H.StudentCNN()
    opt = torch.optim.AdamW(student.parameters(), lr=2e-3)
    X, targets = ctx["X"], ctx["img_emb"]
    student.eval()
    with torch.inference_mode():
        before = H.mean_cosine_alignment(student(X), targets)
    for _ in range(30):
        ex7_distill_step(student, opt, X, targets)
    student.eval()
    with torch.inference_mode():
        after = H.mean_cosine_alignment(student(X), targets)
    assert after > before + 0.3, f"蒸留で整合度が上がらない: {before:.3f} -> {after:.3f}"


def _check8(ctx):
    r5 = ex8_recall_at_k(ctx["img_emb"], ctx["labels"], k=5)
    assert r5 >= 0.9, f"teacher 埋め込みの R@5 が低い: {r5}"
    # k を増やしても Recall は単調非減少
    r1 = ex8_recall_at_k(ctx["img_emb"], ctx["labels"], k=1)
    assert r5 >= r1 - 1e-9, "Recall@5 < Recall@1 はおかしい"


CHECKS = [
    ("Q1 l2_normalize", _check1),
    ("Q2 cosine_logits", _check2),
    ("Q3 predict", _check3),
    ("Q4 distill_loss", _check4),
    ("Q5 kl_affinity", _check5),
    ("Q6 retention", _check6),
    ("Q7 distill_step", _check7),
    ("Q8 recall_at_k", _check8),
]


def run() -> int:
    print("[setup] teacher と合成データを準備中 ...")
    ctx = _build_context()
    passed = 0
    for name, check in CHECKS:
        try:
            check(ctx)
            print(f"  PASS  {name}")
            passed += 1
        except NotImplementedError:
            print(f"  TODO  {name}  (未実装)")
        except Exception as e:  # noqa: BLE001
            print(f"  FAIL  {name}: {type(e).__name__}: {e}")
    print(f"\n採点: {passed}/{len(CHECKS)} PASS")
    # 演習ファイルは未完成でも exit 0（学習者が繰り返し回せるように）
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
