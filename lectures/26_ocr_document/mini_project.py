"""章末ミニプロジェクト: 合成帳票の「読み取り → 構造化 → 評価」を一気通貫で行う。

この回の要素（TrOCR の生成 OCR ／Donut の文書理解 ／CER・WER・ANLS の評価）を
1 本のパイプラインに統合する。題材は合成請求書。次の2段を通し、最後に総合レポートを出す。

  ステージA【テキスト認識】 印字テキスト行を TrOCR で読み、GT と比べて CER/WER を算出。
  ステージB【文書理解】     請求書を Donut/DocVQA に渡し、項目に関する質問へ回答 →
                            exact-match と ANLS（DocVQA 標準指標）で採点。

どちらのモデルが落ちても（ネット不通など）パイプラインは止めず、できた範囲でレポートを出す
（必ず exit 0）。実画像で試したい場合は data/26_ocr_document/ に請求書画像を置く。

実行: uv run python lectures/26_ocr_document/mini_project.py
出力: outputs/26_ocr_document/mini_*.png / mini_report.json
"""

from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import matplotlib.pyplot as plt  # noqa: E402

import ocr_helpers as H  # noqa: E402


def stage_a_recognition(device) -> dict:
    """ステージA: TrOCR で印字行を読み、CER/WER を測る。"""
    print("--- ステージA: テキスト認識（TrOCR）---")
    proc, model = H.load_trocr(device)
    if model is None:
        return {"available": False}

    refs = [s["text"] for s in H.TEXT_LINES]
    records, preds = [], []
    for spec, ref in zip(H.TEXT_LINES, refs):
        img = H.render_text_line(ref, size=spec["size"])
        pred = H.trocr_read(proc, model, img, device)
        preds.append(pred)
        records.append({"image": img, "gt": ref, "pred": pred, "cer": H.cer(ref, pred)})
        print(f"  {ref!r:24s} -> {pred!r:26s} CER={H.cer(ref, pred):.3f}")

    corpus = H.corpus_cer(refs, preds)
    wer_macro = sum(H.wer(r, p) for r, p in zip(refs, preds)) / len(refs)
    print(f"  コーパス CER = {corpus:.4f} / 平均 WER = {wer_macro:.4f}")
    H.save_lines_panel(records, H.OUTPUT_DIR / "mini_recognition.png")
    return {
        "available": True,
        "corpus_cer": round(corpus, 4),
        "mean_wer": round(wer_macro, 4),
        "lines": [{"gt": r["gt"], "pred": r["pred"], "cer": round(r["cer"], 4)} for r in records],
    }


def stage_b_understanding(image, qa, device) -> dict:
    """ステージB: Donut/DocVQA で質問応答し、exact-match と ANLS を測る。"""
    print("--- ステージB: 文書理解（Donut/DocVQA）---")
    proc, model = H.load_donut(device)
    if model is None:
        return {"available": False}

    rows, em_hits, anls_sum, n_gt = [], 0, 0.0, 0
    for q, gt in qa:
        answer, _ = H.donut_ask(proc, model, image, q, device)
        has_gt = bool(gt)
        em = int(H.normalize_text(answer) == H.normalize_text(gt)) if has_gt else None
        score = H.anls(gt, answer) if has_gt else None
        if has_gt:
            em_hits += em
            anls_sum += score
            n_gt += 1
        rows.append({"question": q, "gt": gt, "answer": answer,
                     "exact_match": em, "anls": None if score is None else round(score, 3)})
        print(f"  Q: {q} -> {answer!r}"
              + (f"  exact={em} anls={score:.2f}" if has_gt else "  (no GT)"))

    em_acc = (em_hits / n_gt) if n_gt else None
    mean_anls = (anls_sum / n_gt) if n_gt else None
    if n_gt:
        print(f"  Exact-match = {em_hits}/{n_gt} = {em_acc:.2f} / Mean ANLS = {mean_anls:.3f}")
    return {"available": True, "exact_match_accuracy": em_acc, "mean_anls": mean_anls, "qa": rows}


def save_summary(stage_a: dict, stage_b: dict, path: pathlib.Path) -> None:
    """総合スコアを1枚の棒グラフにまとめる（CER は『1-CER』にして高いほど良いに揃える）。"""
    labels, values, colors = [], [], []
    if stage_a.get("available"):
        labels.append("recognition\n(1 - CER)")
        values.append(max(0.0, 1.0 - stage_a["corpus_cer"]))
        colors.append("#4C78A8")
    if stage_b.get("available") and stage_b.get("exact_match_accuracy") is not None:
        labels.append("DocVQA\nexact-match")
        values.append(stage_b["exact_match_accuracy"])
        colors.append("#54A24B")
        labels.append("DocVQA\nmean ANLS")
        values.append(stage_b["mean_anls"])
        colors.append("#F58518")
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    if labels:
        bars = ax.bar(labels, values, color=colors)
        for b, v in zip(bars, values):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}", ha="center", va="bottom")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("score (higher is better)")
    ax.set_title("Mini-project: document OCR + understanding")
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    out = H.ensure_output_dir()
    device = H.pick_device()
    print(f"=== 章末ミニプロジェクト / device = {device} ===")

    user = H.user_image_or_none()
    if user is not None:
        image = user
        qa = [("What is the total?", ""), ("What is the date?", "")]
        print("data/26_ocr_document/ の実画像を使用（GT 無し → 回答のみ）。")
    else:
        image, _fields, qa = H.build_invoice()
    image.save(out / "mini_input_invoice.png")

    stage_a = stage_a_recognition(device)
    stage_b = stage_b_understanding(image, qa, device)
    save_summary(stage_a, stage_b, out / "mini_summary.png")

    report = {"device": str(device), "stage_a_recognition": stage_a, "stage_b_understanding": stage_b}
    (out / "mini_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nsaved: {out / 'mini_summary.png'}, {out / 'mini_report.json'}")
    print("完了。outputs/26_ocr_document/ の図とレポートで、認識精度(CER)と理解精度(ANLS)を確認。")


if __name__ == "__main__":
    main()
