"""03: Donut ― OCR フリー文書理解（task_prompt → generate → token2json）。

Donut（Document understanding transformer）は『Swin（画像）エンコーダ + BART デコーダ』で、
**OCR を一切介さず**、画像から直接「構造化トークン列」を生成する。DocVQA 版は
  task_prompt = "<s_docvqa><s_question>{Q}</s_question><s_answer>"
を decoder の初期入力に与え、続き（答え）を生成 → token2json で {question, answer} に整形する。

このスクリプトで確認すること:
  1. 合成請求書に対し、Donut/DocVQA で複数の質問に回答する（OCR 段が無いのに答えられる）。
  2. exact-match（正規化後の完全一致）と ANLS（綴り揺れに寛容な DocVQA 標準指標）で評価。
  3. 高レベル API pipeline('document-question-answering') でも同じ答えが出ることを確認。

実行: uv run python lectures/26_ocr_document/03_donut_docvqa.py
出力: outputs/26_ocr_document/03_*.png / 03_docvqa.json
"""

from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from PIL import ImageDraw  # noqa: E402

import ocr_helpers as H  # noqa: E402


def try_pipeline(image, question: str) -> str | None:
    """高レベル API pipeline('document-question-answering') での1問。失敗時 None。

    Donut 系モデルを渡すと OCR フリーで動く（LayoutLM 系は別途 OCR の words/boxes が要る）。
    """
    try:
        from transformers import pipeline

        pipe = pipeline("document-question-answering", model=H.DONUT_MODEL_ID)
        res = pipe(image=image, question=question)
        return str(res[0]["answer"]).strip() if res else ""
    except Exception as e:  # noqa: BLE001
        print(f"[pipeline] スキップ: {type(e).__name__}: {e}")
        return None


def main() -> None:
    out = H.ensure_output_dir()
    device = H.pick_device()
    print(f"=== 03: Donut/DocVQA ({H.DONUT_MODEL_ID}) / device = {device} ===")

    # 入力（実画像があれば優先。無ければ合成請求書＋既知の QA）
    user = H.user_image_or_none()
    if user is not None:
        image = user
        qa = [("What is the total?", ""), ("What is the date?", "")]  # GT 不明 → 回答のみ
        fields = {"note": "user image (no ground truth)"}
        print("data/26_ocr_document/ の実画像を使用します（GT 無しのため回答のみ表示）。")
    else:
        image, fields, qa = H.build_invoice()
    image.save(out / "03_input_invoice.png")

    processor, model = H.load_donut(device)
    if model is None:
        print("Donut を読み込めませんでした。ネット接続を確認のうえ再実行してください。")
        (out / "03_docvqa.json").write_text(
            json.dumps({"available": False}, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return

    # --- DocVQA: 質問ごとに回答し、ANLS / exact-match を測る ---
    rows = []
    em_hits, anls_sum = 0, 0.0
    annotated = image.convert("RGB").copy()
    for q, ans_gt in qa:
        answer, parsed = H.donut_ask(processor, model, image, q, device)
        has_gt = bool(ans_gt)
        em = int(H.normalize_text(answer) == H.normalize_text(ans_gt)) if has_gt else None
        score = H.anls(ans_gt, answer) if has_gt else None
        if has_gt:
            em_hits += em
            anls_sum += score
        rows.append({"question": q, "gt": ans_gt, "answer": answer,
                     "exact_match": em, "anls": None if score is None else round(score, 3)})
        tag = f"exact={em} anls={score:.2f}" if has_gt else "(no GT)"
        print(f"  Q: {q}\n     -> {answer!r}   {tag}")

    n_gt = sum(1 for _, a in qa if a)
    if n_gt:
        print(f"\nExact-match accuracy = {em_hits}/{n_gt} = {em_hits / n_gt:.2f}")
        print(f"Mean ANLS           = {anls_sum / n_gt:.3f}（1.0 が満点。0.5 未満は 0 に丸める指標）")

    # 注釈画像（Q/A を画像下に焼き込む簡易版）
    note = " | ".join(f"{r['question'].split()[-1].rstrip('?')}={r['answer']}" for r in rows[:4])
    ImageDraw.Draw(annotated).text((20, image.height - 24), note[:90], fill=(180, 30, 30))
    annotated.save(out / "03_docvqa_annotated.png")

    # --- 高レベル pipeline でも同じ答えが出ることを1問だけ確認 ---
    pipe_q = qa[0][0]
    pipe_ans = try_pipeline(image, pipe_q)
    if pipe_ans is not None:
        print(f"\n[pipeline] Q: {pipe_q} -> {pipe_ans!r}（直接 generate と整合）")

    payload = {
        "model": H.DONUT_MODEL_ID,
        "fields": fields,
        "qa": rows,
        "exact_match_accuracy": (em_hits / n_gt) if n_gt else None,
        "mean_anls": (anls_sum / n_gt) if n_gt else None,
        "pipeline_check": {"question": pipe_q, "answer": pipe_ans},
    }
    (out / "03_docvqa.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nsaved: {out / '03_docvqa_annotated.png'}, {out / '03_docvqa.json'}")


if __name__ == "__main__":
    main()
