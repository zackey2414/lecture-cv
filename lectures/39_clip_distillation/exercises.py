"""39_clip_distillation 演習 — 自己採点ワークシート。

各 ex*_ 関数の TODO を埋めて `uv run python lectures/39_clip_distillation/exercises.py` を回すと
PASS / FAIL / TODO が表示される。全 8 問が PASS すれば合格。
模範解答は exercises_solutions.py。

難易度: Q1-3 易 / Q4-6 中 / Q7-8 難。
ヒント: ほぼ全問で「埋め込みは L2 正規化してから比較・温度を掛ける」が鍵。

実行:
    uv run python lectures/39_clip_distillation/exercises.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F  # noqa: F401  (解答で使う)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import clip_distill_helpers as H  # noqa: E402


# ===========================================================================
# ここを実装する（各関数の中身を TODO から置き換える）
# ===========================================================================


def ex1_l2_normalize(x: torch.Tensor) -> torch.Tensor:
    """Q1(易): 行ごとに L2 正規化する。各行のノルムが 1 になるように。

    ヒント: torch.nn.functional.normalize(x, dim=-1)
    """
    raise NotImplementedError("Q1: 行ごとの L2 正規化を実装してください")


def ex2_cosine_logits(img: torch.Tensor, txt: torch.Tensor, logit_scale: float) -> torch.Tensor:
    """Q2(易): 画像-テキストの温度付き類似度行列 [N, C] を返す。

    手順: img と txt をそれぞれ L2 正規化 -> 内積 (img @ txt^T) -> logit_scale を掛ける。
    """
    raise NotImplementedError("Q2: 正規化 -> 内積 -> logit_scale を実装してください")


def ex3_predict(logits: torch.Tensor) -> np.ndarray:
    """Q3(易): クラス logits [N, C] から予測クラス index の np.ndarray を返す。

    ヒント: logits.argmax(dim=1) を numpy に。
    """
    raise NotImplementedError("Q3: argmax による予測を実装してください")


def ex4_distill_loss(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    """Q4(中): 埋め込み蒸留損失 = (1 - cos) + mse。

    student, teacher を L2 正規化してから、
    (1 - 行ごとコサイン).mean() と F.mse_loss(s, t) を足す。
    student == teacher のとき 0 になるはず。
    """
    raise NotImplementedError("Q4: コサイン + MSE の蒸留損失を実装してください")


def ex5_kl_affinity(student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
    """Q5(中): 親和性蒸留の KL ダイバージェンス。

    入力 = log_softmax(student)、ターゲット = softmax(teacher)、reduction='batchmean'。
    向き(順序)と log の有無、reduction に注意（38 章の Hinton 蒸留と同じ約束）。
    """
    raise NotImplementedError("Q5: F.kl_div を正しい向き/reduction で実装してください")


def ex6_retention(student_acc: float, teacher_acc: float) -> float:
    """Q6(中): 保持率 = student_acc / teacher_acc。teacher_acc <= 0 のときは 0.0。"""
    raise NotImplementedError("Q6: 保持率を実装してください")


def ex7_distill_step(student, optimizer, X, targets) -> float:
    """Q7(難): 蒸留の 1 ステップ。forward -> ex4_distill_loss -> backward -> step。

    student.train() にして、loss の float 値を返す（loss.detach() を使う）。
    """
    raise NotImplementedError("Q7: 1 ステップの学習更新を実装してください")


def ex8_recall_at_k(img_emb: torch.Tensor, labels: np.ndarray, k: int) -> float:
    """Q8(難): 画像->画像検索の Recall@k（leave-one-out）。

    手順: L2 正規化 -> 類似度行列 -> 対角(自分自身)を -inf -> 各行の上位 k に
    同一クラスが 1 つでもあれば hit。hit 数 / N を返す。
    """
    raise NotImplementedError("Q8: leave-one-out の Recall@k を実装してください")


# ===========================================================================
# 自己採点ハーネス（編集不要）
# ===========================================================================


def _build_context():
    H.set_seed(0)
    device = H.pick_device()
    imgs, labels = H.make_dataset(n_per_class=8, seed=1)
    model, processor, device = H.load_teacher(device)
    return {
        "device": device,
        "labels": labels,
        "logit_scale": H.teacher_logit_scale(model),
        "img_emb": H.teacher_image_embeds(model, processor, imgs, device),
        "txt_emb": H.teacher_text_embeds(model, processor, H.CLASS_PROMPTS, device),
        "X": H.student_preprocess(imgs, device),
    }


def _check1(ctx):
    x = torch.randn(5, 16)
    out = ex1_l2_normalize(x)
    norms = out.norm(dim=-1)
    assert torch.allclose(norms, torch.ones(5), atol=1e-5), f"ノルムが 1 でない: {norms}"


def _check2(ctx):
    logits = ex2_cosine_logits(ctx["img_emb"], ctx["txt_emb"], ctx["logit_scale"])
    assert logits.shape == (ctx["img_emb"].shape[0], H.NUM_CLASSES), f"shape 不正: {logits.shape}"
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
    return 0  # 未完成でも exit 0（繰り返し挑戦できるように）


if __name__ == "__main__":
    raise SystemExit(run())
