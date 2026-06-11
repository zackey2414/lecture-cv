"""03: VQA をどう「採点」するか — VQAv2 accuracy の仕組みを最後まで実装する。

VQA の難しさは「正解が1つに決まらない」こと。同じ画像・質問でも人は
"red" / "crimson" / "a red one" と答える。だから VQAv2 ベンチマークは
「人間10人の回答」を正解として持ち、次の式でモデルを採点する:

    accuracy(予測) = min( 予測と一致した人間の数 / 3, 1.0 )

3人以上が同じ答えなら満点（1.0）。1人だけなら 0.33、誰とも一致しなければ 0。
比較の前に必ず answer を正規化する（小文字化・句読点除去・数詞→数字・冠詞除去）。
正規化を忘れると "Red." ≠ "red" となり、正しい答えを誤判定する典型バグになる。

この回では:
  1. 正規化が無いと何が起きるかを対比で見る
  2. 簡易版 min(#agree/3,1) と 公式 leave-one-out 平均版の両方を実装・比較
  3. 実モデル(ViLT)の予測でベンチマークを採点（モデルが無ければ固定予測で代用）

実行: uv run python lectures/25_vqa_vlm/03_vqa_accuracy.py
出力: outputs/25_vqa_vlm/03_*.png / 03_vqa_accuracy.json
"""

from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import vqa_helpers as H  # noqa: E402


def demo_normalization() -> list[dict]:
    """正規化の有無で一致判定がどう変わるかを並べて返す（教材デモ）。"""
    pairs = [("Red.", "red"), ("A red one", "red"), ("two", "2"), ("YES!", "yes"), ("blue", "red")]
    rows = []
    for pred, human in pairs:
        raw_eq = pred == human
        norm_eq = H.normalize_answer(pred) == H.normalize_answer(human)
        rows.append(
            {
                "pred": pred,
                "human": human,
                "raw_equal": raw_eq,
                "norm_pred": H.normalize_answer(pred),
                "norm_equal": norm_eq,
            }
        )
    return rows


def predict_with_vilt(bench: list[dict], device: torch.device) -> list[str] | None:
    """ViLT で各設問に予測を出す。失敗したら None（呼び出し側が代替予測へ）。"""
    try:
        processor, model = H.load_vilt(device)
    except Exception as e:  # noqa: BLE001
        print(f"  ViLT スキップ: {H.hint_for_model_error(e)}")
        return None
    preds = []
    for item in bench:
        enc = processor(item["image"], item["question"], return_tensors="pt").to(device)
        with torch.inference_mode():
            logits = model(**enc).logits
        preds.append(model.config.id2label[int(logits.argmax(-1))])
    return preds


def evaluate(bench: list[dict], preds: list[str]) -> dict:
    """予測列をベンチマークに対して採点し、各指標をまとめて返す。"""
    rows = []
    simple_scores, vqav2_scores, exact_scores = [], [], []
    for item, pred in zip(bench, preds):
        simple = H.vqa_accuracy_simple(pred, item["human_answers"])
        vqav2 = H.vqa_accuracy_vqav2(pred, item["human_answers"])
        exact = float(H.normalize_answer(pred) == H.normalize_answer(item["gt_answer"]))
        simple_scores.append(simple)
        vqav2_scores.append(vqav2)
        exact_scores.append(exact)
        rows.append(
            {
                "scene": item["scene_id"],
                "question": item["question"],
                "gt": item["gt_answer"],
                "pred": pred,
                "vqa_simple": round(simple, 3),
                "vqa_v2": round(vqav2, 3),
                "exact_match": exact,
            }
        )
    return {
        "rows": rows,
        "mean_vqa_simple": round(float(np.mean(simple_scores)), 3),
        "mean_vqa_v2": round(float(np.mean(vqav2_scores)), 3),
        "mean_exact_match": round(float(np.mean(exact_scores)), 3),
    }


def plot_scores(result: dict, path: pathlib.Path) -> None:
    """設問ごとの VQAv2 accuracy を棒グラフで保存する。"""
    rows = result["rows"]
    labels = [f"{r['scene']}:{r['question'][:18]}…" for r in rows]
    scores = [r["vqa_v2"] for r in rows]
    colors = ["#2ca02c" if s >= 0.99 else ("#ff7f0e" if s > 0 else "#d62728") for s in scores]
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    y = np.arange(len(rows))
    ax.barh(y, scores, color=colors)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.05)
    ax.axvline(result["mean_vqa_v2"], color="black", ls="--", lw=1)
    ax.set_xlabel("VQAv2 accuracy = min(#agree/3, 1)")
    ax.set_title(f"VQA accuracy per question (mean={result['mean_vqa_v2']:.2f}, dashed=mean)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> None:
    out = H.ensure_output_dir()
    device = H.pick_device()
    bench = H.build_benchmark()

    # --- 1) 正規化の効果 ---
    print("[正規化の有無で一致判定はどう変わるか]")
    norm_rows = demo_normalization()
    for r in norm_rows:
        print(
            f"  pred={r['pred']:12s} human={r['human']:6s}"
            f"  raw== {str(r['raw_equal']):5s} -> norm('{r['norm_pred']}')== {r['norm_equal']}"
        )
    print("  → 正規化しないと 'Red.' と 'red' すら別物。比較前の normalize は必須。")

    # --- 2) モデル予測（無ければ固定予測で採点の流れだけ見せる） ---
    print("\n[ベンチマークを採点]")
    preds = predict_with_vilt(bench, device)
    used = "ViLT"
    if preds is None:
        # モデルが使えない環境でも採点の仕組みを最後まで動かすための代替予測。
        preds = ["red", "2", "yes", "3", "green", "red"]
        used = "固定予測（モデル代替）"
    print(f"  予測ソース: {used}")

    result = evaluate(bench, preds)
    result["prediction_source"] = used
    for r in result["rows"]:
        print(
            f"  [{r['scene']}] {r['question'][:30]:30s} gt={r['gt']:5s} pred={r['pred']:8s}"
            f"  v2={r['vqa_v2']:.2f} simple={r['vqa_simple']:.2f} EM={r['exact_match']:.0f}"
        )
    print(
        f"\n  mean VQAv2={result['mean_vqa_v2']:.3f}"
        f"  mean simple={result['mean_vqa_simple']:.3f}"
        f"  mean exact-match={result['mean_exact_match']:.3f}"
    )

    plot_scores(result, out / "03_vqa_accuracy.png")
    (out / "03_vqa_accuracy.json").write_text(
        json.dumps({"normalization": norm_rows, "evaluation": result}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("\nポイント:")
    print("  - VQAv2 accuracy は『3人以上が同意なら満点』。表記ゆれに頑健な設計。")
    print("  - exact-match は厳しすぎ（1つの正解と完全一致のみ）。人手の多様性を捨てている。")
    print("  - 簡易版と公式 leave-one-out 版はだいたい近いが、後者の方がやや辛い。")
    print(f"saved: {out/'03_vqa_accuracy.png'}, {out/'03_vqa_accuracy.json'}")


if __name__ == "__main__":
    main()
