"""03: キャプションを評価する — BLEU / ROUGE-L / CIDEr / CLIPScore。

評価には2系統ある:
  参照あり（人手キャプションと比較）:  BLEU・ROUGE-L・CIDEr（主指標）
  参照なし（画像との整合だけを見る）:  CLIPScore

この回でやること:
  1. メトリクスの「気持ち」を crafted な例で掛む（完全一致→言い換え→無関係 で
     スコアがどう動くか）。値の大小の感覚を先に作る。
  2. 実モデル（BLIP）の出力を合成シーンの参照キャプションで採点する。
  3. 生成パラメータ（num_beams）を変えてスコアの変化を観察する。
  4. 参照不要の CLIPScore を併記し、参照ありメトリクスとの違いを見る。

実行: uv run python lectures/24_image_captioning/03_caption_metrics.py
出力: outputs/24_image_captioning/03_*.png / 03_metrics.json
"""

from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import caption_helpers as H  # noqa: E402
import caption_metrics as M  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402


def demo_metric_intuition() -> dict:
    """crafted な候補で、メトリクスの大小の感覚を作る（モデル不要・純計算）。"""
    target = "a red car on the road"  # 共通の参照キャプション
    # 質の異なる3候補を、同じ参照に対して採点する
    candidates = [
        "a red car on the road",  # 完全一致 → 最高
        "a red vehicle driving on a street",  # 言い換え（語は違うが意味は近い）
        "two people playing soccer",  # 無関係 → 低い
    ]
    labels = ["exact", "paraphrase", "wrong"]
    references_corpus = [[target] for _ in candidates]
    # CIDEr の IDF は「データセット全体」から求めるのが本来。多様な文を IDF 用に渡す
    # （こうしないと参照が同一なため IDF が 0 に潰れ、CIDEr=0 になってしまう）。
    idf_corpus = [
        [target],
        ["a green tree in a grassy field"],
        ["a sunset over the ocean"],
        ["a dog running in a park"],
        ["a blue boat on the sea"],
        ["a red vehicle on a street"],
    ]
    _, cider_each = M.compute_cider(candidates, references_corpus, idf_corpus=idf_corpus)
    rows = []
    for i, (lab, cand) in enumerate(zip(labels, candidates)):
        rows.append(
            {
                "label": lab,
                "candidate": cand,
                "BLEU": round(M.corpus_bleu([cand], [references_corpus[i]]), 4),
                "ROUGE_L": round(M.rouge_l([cand], [references_corpus[i]]), 4),
                "CIDEr": round(cider_each[i], 4),
            }
        )
    return {
        "reference": target,
        "note": "exact が最高。言い換えは BLEU では厳しく 0 になりがちだが ROUGE/CIDEr では中間。無関係は全て低い。",
        "rows": rows,
    }


def evaluate_model(num_beams: int) -> dict:
    """BLIP の出力を参照キャプションで採点（参照あり）＋ CLIPScore（参照なし）。"""
    device = H.pick_device()
    images, names, references = H.build_gallery()

    cap = H.load_captioner("blip", device)
    candidates = H.caption_images(cap, images, num_beams=num_beams, max_new_tokens=20)

    # 参照あり（コーパス全体）
    bleu = M.corpus_bleu(candidates, references)
    rouge = M.rouge_l(candidates, references)
    cider_mean, cider_each = M.compute_cider(candidates, references)
    # 参照なし: 生成キャプションと画像の CLIPScore（1対1）
    clip_scores = M.clip_score_pairs(images, candidates, device)

    per_image = []
    for i, name in enumerate(names):
        per_image.append(
            {
                "image": name,
                "caption": candidates[i],
                "CIDEr": round(cider_each[i], 4),
                "CLIPScore": clip_scores[i],
                "references": references[i],
            }
        )
    return {
        "num_beams": num_beams,
        "corpus_BLEU": round(bleu, 4),
        "ROUGE_L": round(rouge, 4),
        "CIDEr_mean": round(cider_mean, 4),
        "CLIPScore_mean": round(float(np.mean(clip_scores)), 3),
        "per_image": per_image,
    }


def plot_beam_sweep(sweep: list[dict], path: pathlib.Path) -> None:
    """num_beams を変えたときの各指標の変化を棒グラフで保存する。"""
    beams = [s["num_beams"] for s in sweep]
    metrics = ["corpus_BLEU", "ROUGE_L", "CIDEr_mean"]
    x = np.arange(len(beams))
    width = 0.25
    fig, ax = plt.subplots(figsize=(7, 4))
    for j, m in enumerate(metrics):
        ax.bar(x + (j - 1) * width, [s[m] for s in sweep], width, label=m)
    ax.set_xticks(x)
    ax.set_xticklabels([f"num_beams={b}" for b in beams])
    ax.set_ylabel("score")
    ax.set_title("Caption metrics vs. num_beams (BLIP, synthetic scenes)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> None:
    out = H.ensure_output_dir()

    intuition = demo_metric_intuition()
    sweep = [evaluate_model(num_beams=b) for b in (1, 3, 5)]

    plot_beam_sweep(sweep, out / "03_beam_sweep.png")
    (out / "03_metrics.json").write_text(
        json.dumps({"intuition": intuition, "beam_sweep": sweep}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # コンソール: 直感づくり
    print(f"=== メトリクスの直感（同じ参照 '{intuition['reference']}' に対して） ===")
    print(f"{'label':12s} | {'BLEU':>6s} {'ROUGE_L':>8s} {'CIDEr':>7s} | candidate")
    for r in intuition["rows"]:
        print(
            f"{r['label']:12s} | {r['BLEU']:6.3f} {r['ROUGE_L']:8.3f} {r['CIDEr']:7.3f} | {r['candidate']}"
        )
    print(f"  → {intuition['note']}")

    # コンソール: ビーム数スイープ
    print("\n=== BLIP を参照キャプションで採点（num_beams を変えて） ===")
    print(f"{'num_beams':>9s} | {'BLEU':>6s} {'ROUGE_L':>8s} {'CIDEr':>7s} | CLIPScore(mean)")
    for s in sweep:
        print(
            f"{s['num_beams']:9d} | {s['corpus_BLEU']:6.3f} {s['ROUGE_L']:8.3f} "
            f"{s['CIDEr_mean']:7.3f} | {s['CLIPScore_mean']:.2f}"
        )
    print("\n参照あり(BLEU/ROUGE/CIDEr)は『人手キャプションにどれだけ近いか』、")
    print("参照なし(CLIPScore)は『画像と合っているか』を測る。両者は別物なので併記する。")
    print(f"saved: {out / '03_beam_sweep.png'}, {out / '03_metrics.json'}")


if __name__ == "__main__":
    main()
