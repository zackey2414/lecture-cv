"""01: teacher CLIP の正準 API と「類似度行列」を体に入れる。

CLIP 蒸留の出発点は teacher を正しく扱うこと。ここでは
  1) 画像 / テキストを埋め込みにする（encode 相当）
  2) L2 正規化する（しないとスケールがバラバラ）
  3) logit_scale（温度）を掛けて画像-テキスト類似度行列にする
という teacher 側の「正準フロー」を最小コードで確認する。

このフローこそ student に模倣させる対象なので、まず teacher で完全に理解しておく。

実行:
    uv run python lectures/39_clip_distillation/01_teacher_similarity_matrix.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import clip_distill_helpers as H  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402


def show_normalization_matters(model, processor, images, device) -> None:
    """『正規化前は内積がスケールでバラつく / 正規化後はコサインで揃う』を数値で見る。"""
    from PIL import Image

    inputs = processor(images=[Image.fromarray(a) for a in images[:6]], return_tensors="pt").to(device)
    with torch.inference_mode():
        raw = model.get_image_features(**inputs).pooler_output  # 未正規化

    norms = raw.norm(dim=-1)
    print("[normalize] 生の埋め込みノルム:", np.round(norms.numpy(), 2))
    print("            -> 画像ごとに 10〜12 程度でバラつく（内積は長さに引きずられる）")

    normed = F.normalize(raw, dim=-1)
    print("[normalize] 正規化後ノルム:", np.round(normed.norm(dim=-1).numpy(), 3))
    print("            -> すべて 1.0。これで内積 = コサイン類似度になり比較が公平になる")


def build_similarity_matrix(model, processor, images, labels, device):
    """teacher の画像-テキスト類似度行列（温度付き softmax 確率）を作る。"""
    logit_scale = H.teacher_logit_scale(model)
    img_emb = H.teacher_image_embeds(model, processor, images, device)
    txt_emb = H.teacher_text_embeds(model, processor, H.CLASS_PROMPTS, device)

    # logits = logit_scale * (正規化画像) @ (正規化テキスト)^T
    logits = H.zeroshot_logits(img_emb, txt_emb, logit_scale)  # [N, C]
    probs = logits.softmax(dim=1)  # CLIP は softmax で相互排他クラスの確率に変換
    acc = H.zeroshot_accuracy(img_emb, txt_emb, labels, logit_scale)
    print(f"[similarity] logit_scale(exp)={logit_scale:.1f}  zero-shot acc={acc:.3f}")
    return logits, probs, acc


def plot_matrix(probs: torch.Tensor, labels: np.ndarray, path_name: str) -> None:
    """確率行列をヒートマップ保存（タイトルは ASCII でフォント警告回避）。"""
    fig, ax = plt.subplots(figsize=(4.5, 6))
    im = ax.imshow(probs.cpu().numpy(), aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(H.NUM_CLASSES))
    ax.set_xticklabels([n.replace(" ", "\n") for n in H.CLASS_NAMES], fontsize=8)
    ax.set_xlabel("text prompt (class)")
    ax.set_ylabel("image index")
    ax.set_title("Teacher CLIP image-text probabilities")
    # 正解クラスを赤枠で示し、対角的に高確率になることを見せる
    for i, y in enumerate(labels):
        ax.add_patch(plt.Rectangle((y - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="red", lw=0.8))
    fig.colorbar(im, ax=ax, fraction=0.046, label="softmax prob")
    saved = H.save_fig(fig, path_name)
    print(f"[plot] saved -> {saved}")


def main() -> int:
    H.set_seed(0)
    device = H.pick_device()

    # 少量の合成データ（色 + 形の 3 クラス）。teacher の挙動確認には十分。
    images, labels = H.make_dataset(n_per_class=6, seed=1)
    print(f"[data] {len(images)} images / classes={H.CLASS_NAMES}")

    print("[load] teacher = openai/clip-vit-base-patch32 (transformers v5) ...")
    model, processor, device = H.load_teacher(device)

    # (1) 正規化の重要性を数値で確認
    show_normalization_matters(model, processor, images, device)

    # (2) 温度付き類似度行列 -> ゼロショット確率
    _, probs, acc = build_similarity_matrix(model, processor, images, labels, device)

    # (3) 可視化
    plot_matrix(probs, labels, "01_teacher_similarity.png")

    # しきい値チェック（教材の自己検証）。明確に分離したデータなので高精度のはず。
    assert acc >= 0.9, f"teacher のゼロショット精度が低すぎます: {acc}"
    print("[done] teacher の正準フロー(encode -> L2 normalize -> logit_scale)を確認した")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
