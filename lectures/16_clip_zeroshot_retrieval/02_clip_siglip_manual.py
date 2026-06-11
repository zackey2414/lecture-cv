"""02: pipeline を分解する — CLIP/SigLIP を手書きし、softmax と sigmoid の違いを体験する。

01 の pipeline はブラックボックスだった。ここでは processor と model を分離して
『前処理 → forward → 後処理』を自分の手で書く。学ぶ核は3つ:

  (A) CLIP のゼロショット forward:
      processor(text=labels, images=imgs) で同時前処理 → model(**inputs)
      → outputs.logits_per_image → softmax で確率。
  (B) 埋め込みの非対称性（最重要の落とし穴）:
      get_image_features / get_text_features は『未正規化』。コサイン類似度には
      F.normalize が必須。一方 forward の logits_per_image は内部で正規化済み。
      → 手書きの (正規化埋め込みの内積 × logit_scale) が forward と一致することを確認する。
  (C) CLIP(softmax) と SigLIP(sigmoid) の確率解釈の違い:
      CLIP は softmax = 候補が相互排他（合耈1.0、必ずどれかを選ぶ）。
      SigLIP は sigmoid = 各ラベル独立（0〜1を別々に判定、全部低い＝該当なしも表現できる）。

実行: uv run python lectures/16_clip_zeroshot_retrieval/02_clip_siglip_manual.py
出力: outputs/16_clip_zeroshot_retrieval/02_*.png / 02_softmax_vs_sigmoid.json
"""

from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import clip_helpers as H  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402


def clip_zeroshot(model, processor, images, labels, device) -> dict:
    """CLIP の手書きゼロショット。logits_per_image と softmax 確率を返す。

    あわせて『非対称性』を実証する:
      forward の logits_per_image == (F.normalize した画像埋め込み・テキスト埋め込みの内積)
                                     × logit_scale.exp()
    """
    inputs = processor(text=labels, images=images, return_tensors="pt", padding=True).to(device)
    with torch.inference_mode():
        outputs = model(**inputs)  # forward 全体（CLIP の本来の使い方）
        logits = outputs.logits_per_image  # (n_images, n_texts) 大きいほど似ている

        # --- 非対称性の検証: get_*_features は未正規化 ---
        img_emb = H.clip_image_embeds(model, inputs["pixel_values"])  # (Ni, D) 未正規化
        txt_emb = H.clip_text_embeds(model, inputs["input_ids"], inputs["attention_mask"])  # (Nt, D)
        img_norms = img_emb.norm(dim=-1)  # 1.0 ではない（未正規化の証拠）
        # 正規化してから内積を取り、温度 logit_scale を掛けると forward と一致する。
        img_n = F.normalize(img_emb, p=2, dim=-1)
        txt_n = F.normalize(txt_emb, p=2, dim=-1)
        manual_logits = (img_n @ txt_n.t()) * model.logit_scale.exp()

    probs = logits.softmax(dim=1)  # 行ごとに合耈1.0（相互排他）
    matches = torch.allclose(logits, manual_logits, atol=1e-3)
    return {
        "labels": labels,
        "logits": logits.cpu().numpy(),
        "probs": probs.cpu().numpy(),
        "image_embed_norms": img_norms.cpu().numpy(),  # ≇10〜12（≠1）
        "forward_equals_manual": bool(matches),
    }


def siglip_zeroshot(model, processor, images, labels, device) -> dict:
    """SigLIP の手書きゼロショット。sigmoid 確率（各ラベル独立）を返す。

    注意: SigLIP の processor は padding='max_length' を期待する（CLIP は True でよい）。
    sigmoid は softmax と違い、各ラベルを独立に 0〜1 で評価するので、
    『どれも該当しない』画像では全部低い値になりうる（softmax は必ず合耈1.0で誰かを選ぶ）。
    """
    inputs = processor(text=labels, images=images, return_tensors="pt", padding="max_length").to(device)
    with torch.inference_mode():
        outputs = model(**inputs)
        logits = outputs.logits_per_image  # (n_images, n_texts)
    probs = logits.sigmoid()  # 行の合計は1.0にならない（独立判定）
    return {"labels": labels, "logits": logits.cpu().numpy(), "probs": probs.cpu().numpy()}


def plot_heatmaps(clip_res: dict, siglip_res: dict, row_names: list[str], path: pathlib.Path) -> None:
    """CLIP(softmax) と SigLIP(sigmoid) の確率行列をヒートマップで並べる。"""
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for ax, res, title in (
        (axes[0], clip_res, "CLIP: softmax (rows sum to 1, mutually exclusive)"),
        (axes[1], siglip_res, "SigLIP: sigmoid (independent per label)"),
    ):
        p = res["probs"]
        im = ax.imshow(p, vmin=0, vmax=1, cmap="viridis", aspect="auto")
        ax.set_xticks(range(len(res["labels"])))
        ax.set_xticklabels(res["labels"], rotation=30, ha="right", fontsize=7)
        ax.set_yticks(range(len(row_names)))
        ax.set_yticklabels(row_names, fontsize=8)
        for i in range(p.shape[0]):
            for j in range(p.shape[1]):
                ax.text(j, i, f"{p[i, j]:.2f}", ha="center", va="center", color="white", fontsize=7)
        ax.set_title(title, fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> None:
    out = H.ensure_output_dir()
    device = H.pick_device()

    # 入力画像（2枚）と、2種類のラベル集合を用意する。
    images = [H.make_shape_image("red", "circle"), H.make_shape_image("blue", "square")]
    row_names = ["red circle", "blue square"]

    # ケース1: 正解が候補に含まれる素直なラベル集合。
    labels_match = ["a red circle", "a blue square", "a green triangle"]
    # ケース2: 正解がどれも含まれない『該当なし』ラベル集合（softmax と sigmoid の差が出る）。
    labels_nomatch = ["a cat", "a dog", "a car"]

    clip_model, clip_proc = H.load_clip(device)
    siglip_model, siglip_proc = H.load_siglip(device)

    # --- ケース1 ---
    clip1 = clip_zeroshot(clip_model, clip_proc, images, labels_match, device)
    sig1 = siglip_zeroshot(siglip_model, siglip_proc, images, labels_match, device)
    plot_heatmaps(clip1, sig1, row_names, out / "02_softmax_vs_sigmoid_match.png")

    # --- ケース2（該当なし）---
    clip2 = clip_zeroshot(clip_model, clip_proc, images, labels_nomatch, device)
    sig2 = siglip_zeroshot(siglip_model, siglip_proc, images, labels_nomatch, device)
    plot_heatmaps(clip2, sig2, row_names, out / "02_softmax_vs_sigmoid_nomatch.png")

    # --- コンソール出力（理解の要点） ---
    print(f"device = {device}")
    print("[非対称性] CLIP の画像埋め込みノルム（未正規化なので≠1）:",
          np.round(clip1["image_embed_norms"], 2).tolist())
    print("[非対称性] forward の logits == 正規化埋め込みの内積×logit_scale ?:",
          clip1["forward_equals_manual"])
    print("  → get_*_features は未正規化。コサイン類似度の前に F.normalize が必須。")
    print("\n[ケース1: 正解あり] CLIP softmax 確率（行合計=1）:")
    print("  labels:", labels_match)
    print("  ", np.round(clip1["probs"], 3).tolist())
    print("  SigLIP sigmoid 確率（独立）:")
    print("  ", np.round(sig1["probs"], 3).tolist())
    print("\n[ケース2: 該当なし labels=cat/dog/car]")
    print("  CLIP softmax（無理にどれかを選ぶので合耈1.0、最大値も高め）:")
    print("  ", np.round(clip2["probs"], 3).tolist())
    print("  SigLIP sigmoid（全ラベル低い＝『該当なし』を表現できる）:")
    print("  ", np.round(sig2["probs"], 3).tolist())

    # --- json 保存 ---
    payload = {
        "asymmetry": {
            "image_embed_norms": np.round(clip1["image_embed_norms"], 4).tolist(),
            "forward_equals_manual": clip1["forward_equals_manual"],
        },
        "case_match": {
            "labels": labels_match,
            "clip_softmax": np.round(clip1["probs"], 4).tolist(),
            "siglip_sigmoid": np.round(sig1["probs"], 4).tolist(),
        },
        "case_nomatch": {
            "labels": labels_nomatch,
            "clip_softmax": np.round(clip2["probs"], 4).tolist(),
            "siglip_sigmoid": np.round(sig2["probs"], 4).tolist(),
        },
    }
    (out / "02_softmax_vs_sigmoid.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nsaved: {out/'02_softmax_vs_sigmoid_match.png'}, {out/'02_softmax_vs_sigmoid_nomatch.png'}, "
          f"{out/'02_softmax_vs_sigmoid.json'}")


if __name__ == "__main__":
    main()
