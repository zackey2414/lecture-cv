"""章末ミニプロジェクト: ゼロショット検索＆タグ付けエンジン（棄却つき）。

本章で学んだ要素を 1 本に統合した「完成形」。CLIP（と任意で SigLIP）を使い、
合成画像コレクションに対して以下 4 ステージを通しで実行し、図と JSON を出力する。

  Stage A. プロンプト・アンサンブル — 単一テンプレ vs 複数テンプレ平均で
           ゼロショット分類の精度とマージン（top1-top2）がどう変わるか（§2/§3 の発展）。
  Stage B. text->image 検索エンジン — 正規化＋コサイン＋topk で Recall@k/mAP/MRR（§6/§7）。
  Stage C. 正規化アブレーション — 高ノルム distractor で「生内積」が崩壊することの再現（§4/§8）。
  Stage D. 開集合と棄却 — 語彙に無いクエリ（cat / chair / star）を、
           CLIP のコサイン閾値 と SigLIP の sigmoid 独立スコアで「該当なし」と判定する（§5）。

すべて合成画像で完走（ネットは初回のモデル重み DL のみ）。CPU・eval+inference_mode 前提。
digit 始まりのスクリプトは import できないため、必要処理は自己完結させ、
共通の道具箱 clip_helpers だけを借用する。

実行: uv run python lectures/16_clip_zeroshot_retrieval/mini_project.py
出力: lectures/16_clip_zeroshot_retrieval/outputs/mini_project_*.png / mini_project_report.json
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

import clip_helpers as H  # noqa: E402  共通の道具箱（合成画像・device・CLIP/SigLIP ロード）
import matplotlib.pyplot as plt  # noqa: E402

# プロンプト・アンサンブルのテンプレート集（CLIP 論文は数十個を平均する）。
# {} に「red circle」などのクラス名を差し込む。テンプレを増やすほどノイズが平均で消える。
TEMPLATES: tuple[str, ...] = (
    "a photo of a {}",
    "an image of a {}",
    "a {}",
    "a picture of a {}",
    "a {} on a plain background",
)


# =====================================================================
# 埋め込みの取り出し（射影後・未正規化）。正規化は呼び出し側で必ず行う。
# =====================================================================

def embed_images_raw(model, proc, images, device) -> torch.Tensor:
    """画像コレクションを CLIP で埋め込む（射影後・未正規化、(N, D)）。"""
    with torch.inference_mode():
        inp = proc(images=images, return_tensors="pt").to(device)
        return H.clip_image_embeds(model, inp["pixel_values"])


def embed_texts_raw(model, proc, texts, device) -> torch.Tensor:
    """テキスト群を CLIP で埋め込む（射影後・未正規化、(T, D)）。padding=True を忘れない。"""
    with torch.inference_mode():
        inp = proc(text=list(texts), return_tensors="pt", padding=True).to(device)
        return H.clip_text_embeds(model, inp["input_ids"], inp["attention_mask"])


def single_class_weights(model, proc, class_names, device) -> torch.Tensor:
    """単一テンプレ "a photo of a {name}" のクラス重み（正規化済み (C, D)）。"""
    prompts = [f"a photo of a {n}" for n in class_names]
    return F.normalize(embed_texts_raw(model, proc, prompts, device), p=2, dim=-1)


def ensemble_class_weights(model, proc, class_names, templates, device) -> torch.Tensor:
    """複数テンプレを平均したクラス重み（正規化済み (C, D)）。

    各テンプレ埋め込みを個別に L2 正規化 → 平均 → 再正規化、が CLIP 流の定石。
    正規化してから平均することで、テンプレ間のノルム差に引きずられないようにする。
    """
    cols = []
    for name in class_names:
        raw = embed_texts_raw(model, proc, [t.format(name) for t in templates], device)  # (T, D)
        mean = F.normalize(raw, p=2, dim=-1).mean(dim=0)  # 個別正規化 → 平均
        cols.append(F.normalize(mean, p=2, dim=0))        # 再正規化
    return torch.stack(cols, dim=0)


# =====================================================================
# Stage A: プロンプト・アンサンブル（単一 vs 平均）
# =====================================================================

def classify_stats(img_emb: torch.Tensor, weights: torch.Tensor, gold: list[int]) -> dict:
    """正規化済み画像埋め込みとクラス重みから、精度と平均マージンを出す。

    margin = (top1 ロジット - top2 ロジット) の平均。大きいほど「自信を持って正解と外せている」。
    """
    logits = img_emb @ weights.t()  # (N, C) コサイン（両者正規化済み）
    top2 = logits.topk(2, dim=1).values
    margin = float((top2[:, 0] - top2[:, 1]).mean())
    pred = logits.argmax(dim=1)
    acc = float((pred == torch.tensor(gold)).float().mean())
    return {"accuracy": round(acc, 4), "mean_margin": round(margin, 4)}


# =====================================================================
# Stage B: text->image 検索の評価（Recall@k / mAP / MRR）
# =====================================================================

def retrieval_metrics(sims: torch.Tensor, relevant: list[set[int]], ks=(1, 5)) -> dict:
    """類似度行列 (Q, N) と各クエリの正解集合から Recall@k・mAP・MRR を平均で返す。"""
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


# =====================================================================
# Stage C: 正規化アブレーション（高ノルム distractor）
# =====================================================================

def normalization_ablation(img_raw: torch.Tensor, txt_raw: torch.Tensor,
                           relevant: list[set[int]], scale: float = 6.0) -> dict:
    """1 枚だけノルムを scale 倍した distractor を入れ、コサイン vs 生内積で mAP 等を比較。

    コサインは大きさを無視するので頑健、生内積は大きさに釣られて崩壊する（§8 の再現）。
    """
    img_raw_d = img_raw.clone()
    img_raw_d[0] = img_raw_d[0] * scale  # 「やたら自信のある外れ値」ベクトルの模擬
    txt_n = F.normalize(txt_raw, p=2, dim=-1)
    sims_cos = txt_n @ F.normalize(img_raw_d, p=2, dim=-1).t()  # コサイン: 大きさ無視
    sims_dot = txt_raw @ img_raw_d.t()                           # 生内積: 大きさに釣られる
    return {
        "cosine_normalized": retrieval_metrics(sims_cos, relevant),
        "raw_innerproduct": retrieval_metrics(sims_dot, relevant),
    }


# =====================================================================
# Stage D: 開集合と棄却（語彙に無いクエリの検出）
# =====================================================================

def clip_abstention(model, proc, img_emb: torch.Tensor, queries_in, queries_oov, device) -> dict:
    """CLIP のコサイン最大値で「該当あり/なし」を分離できるか調べる。

    in-vocab（語彙にある）クエリの最大コサインと oov（語彙にない）クエリの最大コサインが
    分離していれば、その中点を閾値にして棄却できる。分離していなければ閾値での棄却は難しい。
    """
    def max_cos(qs):
        q_emb = F.normalize(embed_texts_raw(model, proc, qs, device), p=2, dim=-1)
        return (q_emb @ img_emb.t()).max(dim=1).values  # 各クエリの最大コサイン

    cos_in = max_cos(queries_in).tolist()
    cos_oov = max_cos(queries_oov).tolist()
    gap = min(cos_in) - max(cos_oov)            # 正なら綺麗に分離している
    thr = (min(cos_in) + max(cos_oov)) / 2.0    # 分離していれば中点が良い閾値
    return {
        "max_cos_in_vocab": [round(v, 4) for v in cos_in],
        "max_cos_oov": [round(v, 4) for v in cos_oov],
        "separation_gap": round(float(gap), 4),
        "threshold": round(float(thr), 4),
        "separable": bool(gap > 0),
    }


def siglip_abstention(images, queries_in, queries_oov, device) -> dict | None:
    """SigLIP の sigmoid（独立判定）で「全ラベル低い＝該当なし」を表現できるか調べる。

    SigLIP は重いわけではないが、未導入や読み込み失敗でも本体が完走するようにガードする。
    各クエリについて 12 枚に対する logits_per_text を sigmoid し、最大値を取る。
    oov なら全画像で低い（=最大値も低い）はず。閾値 0.5 で accept/abstain を決める。
    """
    try:
        model, proc = H.load_siglip(device)
    except Exception as e:  # noqa: BLE001  依存欠如・DL 失敗などでも完走させる
        return {"skipped": True, "reason": f"{type(e).__name__}: {e}"}

    queries = list(queries_in) + list(queries_oov)
    # SigLIP の processor は padding='max_length' を期待する（CLIP は True でよい）。
    inputs = proc(text=queries, images=images, return_tensors="pt", padding="max_length").to(device)
    with torch.inference_mode():
        logits = model(**inputs).logits_per_image  # (n_images, n_queries)
    # クエリ視点に転置 → 各クエリの「最も合う画像」の sigmoid 値。
    max_sig = logits.t().sigmoid().max(dim=1).values
    n_in = len(queries_in)
    return {
        "skipped": False,
        "max_sigmoid_in_vocab": [round(float(v), 4) for v in max_sig[:n_in]],
        "max_sigmoid_oov": [round(float(v), 4) for v in max_sig[n_in:]],
        "threshold": 0.5,
    }


# =====================================================================
# 図の保存（4 パネル要約）
# =====================================================================

def plot_summary(report: dict, path: pathlib.Path) -> None:
    """4 ステージの要点を 1 枚に要約する（タイトルは ASCII 固定）。"""
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    # Panel A: プロンプト・アンサンブル（精度とマージン）。
    ax = axes[0, 0]
    a = report["stage_a_prompt_ensemble"]
    x = np.arange(2)
    ax.bar(x - 0.18, [a["single"]["accuracy"], a["ensemble"]["accuracy"]], width=0.36, label="accuracy", color="#4c78a8")
    ax.bar(x + 0.18, [a["single"]["mean_margin"], a["ensemble"]["mean_margin"]], width=0.36, label="mean margin", color="#f58518")
    ax.set_xticks(x)
    ax.set_xticklabels(["single", "ensemble"])
    ax.set_ylim(0, 1.05)
    ax.set_title("A. prompt ensembling (acc & top1-top2 margin)")
    ax.legend(fontsize=8)

    # Panel B: 検索メトリクス。
    ax = axes[0, 1]
    b = report["stage_b_retrieval"]
    names = list(b.keys())
    ax.bar(np.arange(len(names)), [b[k] for k in names], color="#54a24b")
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_title("B. text->image retrieval metrics")

    # Panel C: 正規化アブレーション（mAP の比較）。
    ax = axes[1, 0]
    c = report["stage_c_normalization"]
    vals = [c["cosine_normalized"]["mAP"], c["raw_innerproduct"]["mAP"]]
    ax.bar(["cosine\n(normalized)", "raw dot\n(unnormalized)"], vals, color=["#54a24b", "#e45756"])
    ax.set_ylim(0, 1.05)
    ax.set_title("C. high-norm distractor: mAP collapse")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)

    # Panel D: 棄却（CLIP コサイン最大値の in-vocab vs oov 分布 + 閾値）。
    ax = axes[1, 1]
    d = report["stage_d_abstention"]["clip"]
    ax.scatter(np.zeros(len(d["max_cos_in_vocab"])), d["max_cos_in_vocab"], color="#4c78a8", label="in-vocab", zorder=3)
    ax.scatter(np.ones(len(d["max_cos_oov"])), d["max_cos_oov"], color="#e45756", label="oov", zorder=3)
    ax.axhline(d["threshold"], color="gray", ls="--", label=f"threshold={d['threshold']:.2f}")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["in-vocab", "oov (no match)"])
    ax.set_title("D. abstention by CLIP cosine threshold")
    ax.legend(fontsize=8)

    fig.suptitle("Mini project: zero-shot retrieval & tagging engine with abstention", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=110)
    plt.close(fig)


# =====================================================================
# 実行本体
# =====================================================================

def main() -> None:
    out = H.ensure_output_dir()
    device = H.pick_device()

    # 12 枚（色×形）の合成コレクションと人間可読メタ。クラス名は "red circle" 等。
    images, meta = H.build_collection()
    class_names = [f"{c} {s}" for c, s in meta]
    gold = list(range(len(images)))  # 画像 i の正解クラスは i（exact-match）。

    model, proc = H.load_clip(device)

    # 画像埋め込み（未正規化）→ 正規化。生ベクトルは Stage C で使う。
    img_raw = embed_images_raw(model, proc, images, device)
    img_emb = F.normalize(img_raw, p=2, dim=-1)

    # --- Stage A: プロンプト・アンサンブル ---
    w_single = single_class_weights(model, proc, class_names, device)
    w_ensemble = ensemble_class_weights(model, proc, class_names, TEMPLATES, device)
    stage_a = {
        "single": classify_stats(img_emb, w_single, gold),
        "ensemble": classify_stats(img_emb, w_ensemble, gold),
        "templates": list(TEMPLATES),
    }

    # --- Stage B: text->image 検索（アンサンブル重みをクエリに使う）---
    sims = img_emb @ w_ensemble.t()        # (N_img, C) を検索行列にする
    sims_q = sims.t()                       # (C=Q, N_img) クエリ視点
    relevant = [{i} for i in range(len(class_names))]
    stage_b = retrieval_metrics(sims_q, relevant)

    # --- Stage C: 正規化アブレーション（単一テンプレの生ベクトルで比較）---
    txt_raw_single = embed_texts_raw(model, proc, [f"a photo of a {n}" for n in class_names], device)
    stage_c = normalization_ablation(img_raw, txt_raw_single, relevant)

    # --- Stage D: 開集合と棄却 ---
    queries_in = ["a photo of a red circle", "a photo of a blue square", "a photo of a green triangle"]
    queries_oov = ["a photo of a cat", "a photo of a wooden chair", "a photo of a purple star"]
    stage_d = {
        "clip": clip_abstention(model, proc, img_emb, queries_in, queries_oov, device),
        "siglip": siglip_abstention(images, queries_in, queries_oov, device),
        "queries_in_vocab": queries_in,
        "queries_oov": queries_oov,
    }

    report = {
        "device": str(device),
        "model": H.CLIP_MODEL_ID,
        "collection_size": len(images),
        "stage_a_prompt_ensemble": stage_a,
        "stage_b_retrieval": stage_b,
        "stage_c_normalization": stage_c,
        "stage_d_abstention": stage_d,
    }

    plot_summary(report, out / "mini_project_summary.png")
    (out / "mini_project_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # --- コンソール要約 ---
    print(f"device = {device}  collection = {len(images)} images  model = {H.CLIP_MODEL_ID}")
    print("\n[A] prompt ensembling (single vs ensemble):")
    print(f"    single  : acc={stage_a['single']['accuracy']:.3f}  margin={stage_a['single']['mean_margin']:.4f}")
    print(f"    ensemble: acc={stage_a['ensemble']['accuracy']:.3f}  margin={stage_a['ensemble']['mean_margin']:.4f}")
    print("    → この合成セットは易しく単一でも十分高い。アンサンブルの効果は")
    print("      実データ・曖昧なクラス・タイポの多い語彙ほど顕著になる（分散を均す）。")
    print(f"\n[B] text->image retrieval: {stage_b}")
    print("\n[C] high-norm distractor (mAP): "
          f"cosine={stage_c['cosine_normalized']['mAP']:.3f}  raw-dot={stage_c['raw_innerproduct']['mAP']:.3f}")
    print("    → 生内積は 1 枚の高ノルム画像に検索を乗っ取られ崩壊。コサインは無関心。")
    clip_d = stage_d["clip"]
    print("\n[D] abstention (CLIP cosine):")
    print(f"    in-vocab max-cos = {clip_d['max_cos_in_vocab']}")
    print(f"    oov      max-cos = {clip_d['max_cos_oov']}")
    print(f"    separable={clip_d['separable']}  gap={clip_d['separation_gap']:.4f}  threshold={clip_d['threshold']:.4f}")
    sig_d = stage_d["siglip"]
    if sig_d and not sig_d.get("skipped"):
        print("    SigLIP sigmoid (independent, threshold 0.5):")
        print(f"      in-vocab = {sig_d['max_sigmoid_in_vocab']}  oov = {sig_d['max_sigmoid_oov']}")
        print("    → oov は sigmoid が全画像で低く、0.5 閾値で自然に『該当なし』を棄却できる。")
    else:
        print(f"    SigLIP: skipped ({sig_d.get('reason') if sig_d else 'n/a'})")

    print(f"\nsaved: {out/'mini_project_summary.png'}, {out/'mini_project_report.json'}")


if __name__ == "__main__":
    main()
