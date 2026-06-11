"""01: pipeline('zero-shot-image-classification') でゼロショット分類を最短体験する。

この回のねらいは「まず動かして成功体験を得る」こと。前処理〜推論〜後処理を
高レベル API pipeline が一手に引き受けるので、CLIP の中身を知らなくても
「任意のテキストラベル候補（candidate_labels）で画像を分類」できる。

ゼロショット分類とは: モデルを再学習せず、推論時に渡したラベル候補だけで分類すること。
通常の分類器は学習時に決めた固定クラス（ImageNet の1000種など）しか答えられないが、
CLIP は画像とテキストを同じ空間に埋め込むので、その場で渡した任意の語と照合できる。
だから "a red circle" でも "a traffic sign" でも、候補に入れさえすれば分類対象になる。

実行: uv run python lectures/16_clip_zeroshot_retrieval/01_zeroshot_pipeline.py
出力: outputs/16_clip_zeroshot_retrieval/01_*.png / 01_zeroshot_results.json
"""

from __future__ import annotations

import json
import pathlib
import sys

# 同フォルダの helper を import できるよう、最初にパスを通す（スクリプト単体実行のため）。
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import clip_helpers as H  # noqa: E402  同じフォルダの道具箱（合成画像・device・出力先）
import matplotlib.pyplot as plt  # noqa: E402


def run_pipeline_zeroshot() -> dict:
    """pipeline で各画像を candidate_labels に対して分類し、結果を辞書で返す。"""
    from transformers import pipeline
    from transformers.utils import logging as hf_logging

    hf_logging.set_verbosity_error()

    device = H.pick_device()

    # 評価用に「正解の分かっている」小さな合成セットを作る（色×形の代表4枚）。
    samples = [
        ("red", "circle"),
        ("blue", "square"),
        ("green", "triangle"),
        ("yellow", "circle"),
    ]
    images = [H.make_shape_image(c, s) for c, s in samples]
    gold = [f"a {c} {s}" for c, s in samples]  # 正解ラベル（top-1 accuracy 用）

    # candidate_labels は「その場で決める」任意の語の集合。これがゼロショットの本質。
    # 正解4つ＋ダミー2つを混ぜ、紛れがあっても正しく当てられるか見る。
    candidate_labels = [
        "a red circle",
        "a blue square",
        "a green triangle",
        "a yellow circle",
        "a purple star",
        "a black hexagon",
    ]

    # task 名と model を指定するだけ。前処理・推論・softmax 後処理まで pipeline が行う。
    # device: CPU は -1（または "cpu"）。GPU があれば 0 を渡す。
    clf = pipeline(
        task="zero-shot-image-classification",
        model=H.CLIP_MODEL_ID,
        device=0 if device.type == "cuda" else -1,
    )

    # まとめて推論。返り値は画像ごとに [{'score':..,'label':..}, ...]（score 降順）。
    outputs = clf(images, candidate_labels=candidate_labels)

    results = []
    correct = 0
    for (color, shape), gold_label, per_image in zip(samples, gold, outputs):
        top = per_image[0]  # 最上位（最も似ているラベル）
        is_correct = top["label"] == gold_label
        correct += int(is_correct)
        results.append(
            {
                "image": f"{color} {shape}",
                "gold": gold_label,
                "pred": top["label"],
                "pred_score": round(float(top["score"]), 4),
                "correct": is_correct,
                # 全候補のスコア（softmax 後。合計は約1.0 = 相互排他の確率）
                "scores": {d["label"]: round(float(d["score"]), 4) for d in per_image},
            }
        )

    top1_acc = correct / len(samples)
    return {"top1_accuracy": top1_acc, "candidate_labels": candidate_labels, "results": results}


def plot_scores(summary: dict, path: pathlib.Path) -> None:
    """各画像の候補ラベル別スコアを横棒グラフ（4パネル）で保存する。"""
    results = summary["results"]
    labels = summary["candidate_labels"]
    fig, axes = plt.subplots(1, len(results), figsize=(4.2 * len(results), 3.4))
    axes = np.atleast_1d(axes)
    y = np.arange(len(labels))
    for ax, r in zip(axes, results):
        scores = [r["scores"][lab] for lab in labels]
        colors = ["#d62728" if lab == r["pred"] else "#9ecae1" for lab in labels]
        ax.barh(y, scores, color=colors)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=7)
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        mark = "OK" if r["correct"] else "NG"
        ax.set_title(f"input: {r['image']}  [{mark}]", fontsize=9)
        ax.set_xlabel("softmax score")
    fig.suptitle("Zero-shot image classification (CLIP, mutually-exclusive softmax)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> None:
    out = H.ensure_output_dir()
    summary = run_pipeline_zeroshot()

    # 結果の保存（図 + json）。
    plot_scores(summary, out / "01_zeroshot_scores.png")
    (out / "01_zeroshot_results.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # コンソールにも要点を出す（読みながら理解できるように）。
    print(f"device = {H.pick_device()}  model = {H.CLIP_MODEL_ID}")
    print(f"top-1 accuracy = {summary['top1_accuracy']:.2f}  (合成4枚)")
    for r in summary["results"]:
        mark = "OK" if r["correct"] else "NG"
        print(f"  [{mark}] {r['image']:16s} -> {r['pred']:18s} (score={r['pred_score']:.3f})")
    print("候補ラベルは推論時に自由に決められる（=ゼロショット）。")
    print("scores の合計は約1.0 — softmax は『候補のどれか1つ』を選ぶ相互排他の確率。")
    print(f"saved: {out/'01_zeroshot_scores.png'}, {out/'01_zeroshot_results.json'}")


if __name__ == "__main__":
    main()
