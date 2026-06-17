"""03: CLIP 埋め込みで画像テキスト検索（text->image / image->image）と Recall@k/mAP/MRR。

ゼロショット『分類』は候補ラベルとの照合だった。ここでは候補が『画像コレクション』になる。
手順は CLIP の原理そのもの:
  1. コレクションの全画像を get_image_features で埋め込み → L2 正規化（必須）。
  2. クエリ（テキスト or 画像）も同じ手順で埋め込み → L2 正規化。
  3. コサイン類似度（正規化済みベクトルの内積）でランキング → torch.topk。

評価:
  - text->image を『各クエリに対応する1枚が正解』として、Recall@1/5/10・mAP・MRR を
    torchmetrics.functional.retrieval で算出（クエリ別に出して平均）。
  - 『正規化を忘れるとなぜ壊れるか』を、ノルムを意図的に膛らませた1枚（高ノルム distractor）で実証。
    コサインは大きさを無視するので頽健、生の内積は大きさに引きずられて検索が崩壊する。

実行: uv run python lectures/16_clip_zeroshot_retrieval/03_text_image_retrieval.py
出力: lectures/16_clip_zeroshot_retrieval/outputs/03_*.png / 03_retrieval_metrics.json
"""

from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torchmetrics.functional.retrieval import (  # noqa: E402
    retrieval_average_precision,
    retrieval_recall,
    retrieval_reciprocal_rank,
)

import clip_helpers as H  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402


def embed_images(model, processor, images, device) -> torch.Tensor:
    """画像コレクションを CLIP で埋め込む（射影後・未正規化、(N, D)）。"""
    with torch.inference_mode():
        inputs = processor(images=images, return_tensors="pt").to(device)
        return H.clip_image_embeds(model, inputs["pixel_values"])


def embed_texts(model, processor, texts, device) -> torch.Tensor:
    """テキスト群を CLIP で埋め込む（射影後・未正規化、(Q, D)）。padding=True を忘れない。"""
    with torch.inference_mode():
        inputs = processor(text=texts, return_tensors="pt", padding=True).to(device)
        return H.clip_text_embeds(model, inputs["input_ids"], inputs["attention_mask"])


def retrieval_metrics(sims: torch.Tensor, relevant: list[set[int]], ks=(1, 5, 10)) -> dict:
    """類似度行列 (Q, N) と各クエリの正解集合から Recall@k・mAP・MRR を平均で返す。

    torchmetrics の functional 版はクエリ1本ずつ (preds, target) で計算するので、
    全クエリ分を回して numpy 平均する（クエリ別 AP を平均したのが retrieval mAP）。
    """
    n = sims.shape[1]
    rec = {k: [] for k in ks}
    aps, rrs = [], []
    for q in range(sims.shape[0]):
        preds = sims[q]
        target = torch.zeros(n, dtype=torch.bool)
        for idx in relevant[q]:
            target[idx] = True
        for k in ks:
            rec[k].append(retrieval_recall(preds, target, top_k=k).item())
        aps.append(retrieval_average_precision(preds, target).item())
        rrs.append(retrieval_reciprocal_rank(preds, target).item())
    out = {f"Recall@{k}": round(float(np.mean(rec[k])), 4) for k in ks}
    out["mAP"] = round(float(np.mean(aps)), 4)
    out["MRR"] = round(float(np.mean(rrs)), 4)
    return out


def plot_text_to_image(
    images, meta, queries, sims: torch.Tensor, query_ids: list[int], path: pathlib.Path, topk: int = 3
) -> None:
    """選んだいくつかのテキストクエリについて、上位 topk 検索結果を並べて保存する。"""
    nrow = len(query_ids)
    fig, axes = plt.subplots(nrow, topk, figsize=(topk * 2.4, nrow * 2.6))
    axes = np.array(axes).reshape(nrow, topk)
    for r, q in enumerate(query_ids):
        order = torch.argsort(sims[q], descending=True).tolist()
        for c in range(topk):
            idx = order[c]
            ax = axes[r, c]
            ax.imshow(images[idx])
            ax.axis("off")
            correct = idx == q  # exact-match の正解は同じ並びの画像
            mark = "OK" if correct else "  "
            ax.set_title(f"#{c+1} {meta[idx][0]} {meta[idx][1]}\ncos={sims[q, idx]:.3f} {mark}", fontsize=7)
        axes[r, 0].set_ylabel(queries[q], fontsize=8, rotation=0, ha="right", va="center")
    fig.suptitle("text -> image retrieval (top-3, cosine similarity)", fontsize=11)
    fig.tight_layout(rect=(0.06, 0, 1, 0.96))
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> None:
    out = H.ensure_output_dir()
    device = H.pick_device()

    # 12枚（色×形）の画像コレクションと、その人間可読メタ。
    images, meta = H.build_collection()
    model, processor = H.load_clip(device)

    # --- 1) 画像を埋め込み（未正規化）→ 正規化 ---
    img_raw = embed_images(model, processor, images, device)
    img_emb = F.normalize(img_raw, p=2, dim=-1)  # コサイン類似度のための L2 正規化（必須）

    # --- 2) text->image（exact-match: クエリ q の正解は画像 q）---
    queries = [f"a photo of a {c} {s}" for c, s in meta]
    txt_raw = embed_texts(model, processor, queries, device)
    txt_emb = F.normalize(txt_raw, p=2, dim=-1)
    sims = txt_emb @ img_emb.t()  # (Q, N) コサイン類似度（-1〜1）
    relevant = [{q} for q in range(len(queries))]
    t2i = retrieval_metrics(sims, relevant)

    # 代表4クエリの検索結果パネルを保存。
    plot_text_to_image(images, meta, queries, sims, query_ids=[0, 4, 7, 11],
                       path=out / "03_text_to_image.png")

    # --- 3) image->image（あるクエリ画像に似た画像。自分自身は除外）---
    q_img = 0  # red circle
    self_excluded = (img_emb @ img_emb[q_img]).clone()
    self_excluded[q_img] = -2.0  # 自分を最下位に追いやる
    nn_order = torch.argsort(self_excluded, descending=True).tolist()[:3]
    i2i = [
        {"rank": r + 1, "image": f"{meta[i][0]} {meta[i][1]}", "cos": round(float(img_emb[q_img] @ img_emb[i]), 4)}
        for r, i in enumerate(nn_order)
    ]

    # --- 4) 正規化の意味: スコアの値域と『高ノルム distractor』実験 ---
    sims_unnorm = txt_raw @ img_raw.t()  # 未正規化のまま内積（=コサインではない）
    changed = sum(
        int(torch.argsort(sims[q], descending=True).tolist()
            != torch.argsort(sims_unnorm[q], descending=True).tolist())
        for q in range(len(queries))
    )
    # 1枚だけ埋め込みのノルムを6倍に膛らませる（現実の「やたら自信のある／外れ値」ベクトルの模擬）。
    img_raw_distractor = img_raw.clone()
    img_raw_distractor[0] = img_raw_distractor[0] * 6.0
    sims_norm_d = txt_emb @ F.normalize(img_raw_distractor, p=2, dim=-1).t()  # コサイン: 大きさを無視
    sims_unnorm_d = txt_raw @ img_raw_distractor.t()                          # 生内積: 大きさに釣られる
    metrics_norm_d = retrieval_metrics(sims_norm_d, relevant)
    metrics_unnorm_d = retrieval_metrics(sims_unnorm_d, relevant)

    # --- コンソール出力 ---
    print(f"device = {device}  collection = {len(images)} images")
    print(f"[text->image exact-match] {t2i}")
    print(f"[image->image] query='{meta[q_img][0]} {meta[q_img][1]}' nearest:")
    for d in i2i:
        print(f"   #{d['rank']} {d['image']:16s} cos={d['cos']:.3f}")
    print("\n[正規化の意味]")
    print(f"  コサイン(正規化)のスコア範囲: {sims.min():.3f} 〜 {sims.max():.3f}（-1〜1で解釈可能）")
    print(f"  生内積(未正規化)のスコア範囲: {sims_unnorm.min():.1f} 〜 {sims_unnorm.max():.1f}（大きさ次第で無制限）")
    print(f"  正規化有無でフルランキングが変わったクエリ数: {changed}/{len(queries)}")
    print("  高ノルム distractor 実験（1枚のノルムを6倍）:")
    print(f"    コサイン(正規化) : {metrics_norm_d}  ← 大きさを無視するので頽健")
    print(f"    生内積(未正規化) : {metrics_unnorm_d}  ← 高ノルム画像が全クエリを乗っ取り崩壊")

    # --- json 保存 ---
    payload = {
        "text_to_image_exact_match": t2i,
        "image_to_image_query": f"{meta[q_img][0]} {meta[q_img][1]}",
        "image_to_image_neighbors": i2i,
        "normalization": {
            "cosine_score_range": [round(float(sims.min()), 4), round(float(sims.max()), 4)],
            "rawdot_score_range": [round(float(sims_unnorm.min()), 4), round(float(sims_unnorm.max()), 4)],
            "full_ranking_changed_queries": changed,
            "high_norm_distractor": {
                "cosine_normalized": metrics_norm_d,
                "raw_innerproduct": metrics_unnorm_d,
            },
        },
    }
    (out / "03_retrieval_metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nsaved: {out/'03_text_to_image.png'}, {out/'03_retrieval_metrics.json'}")


if __name__ == "__main__":
    main()
