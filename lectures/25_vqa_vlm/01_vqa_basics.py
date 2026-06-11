"""01: VQA の2つの定番アプローチを最短で動かして比べる。

VQA（Visual Question Answering）= 画像 + 自然言語の質問 → 回答 のタスク。
回答の出し方には大きく2つの流派があり、まずこの違いを体で覚えるのがこの回の目的。

  (1) 分類型（discriminative）: ViLT
      あらかじめ決めた「3129語の回答候補」から1つを選ぶ。出力は logits で、
      argmax → id2label で答えになる。生成しないので速く・決定的。
      欠点: 候補に無い答えは絶対に出ない（語彙の外は答えられない）。

  (2) 生成型（generative）: BLIP-VQA
      Encoder-Decoder が回答テキストを「生成」する。model.generate を使う。
      語彙に縛られず柔軟だが、デコードのぶん少し遅く、表記ゆれも出る。

どちらも processor(image, question) で前処理する点は共通。v5 では
pipeline("visual-question-answering") は廃止されたので、専用クラスを直接使う。

実行: uv run python lectures/25_vqa_vlm/01_vqa_basics.py
出力: outputs/25_vqa_vlm/01_*.png / 01_vqa_basics.json
"""

from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import torch  # noqa: E402

import vqa_helpers as H  # noqa: E402


# 評価に使う共通の質問セット（合成シーンAに対して）。
QUESTIONS = [
    "What color is the circle?",
    "What color is the square?",
    "How many shapes are in the image?",
    "Is there a square in the image?",
]


def run_vilt(image, questions: list[str], device: torch.device) -> list[dict]:
    """ViLT（分類型）で各質問に答える。logits → argmax → id2label。"""
    processor, model = H.load_vilt(device)
    results = []
    for q in questions:
        # processor が画像とテキストを同時に前処理（pixel_values + input_ids）。
        enc = processor(image, q, return_tensors="pt").to(device)
        with torch.inference_mode():
            logits = model(**enc).logits  # (1, 3129) 回答候補ぶんのスコア
        idx = int(logits.argmax(-1))  # 最もスコアの高い候補
        # softmax で「どれくらい自信があるか」も覓いておく（相互排他の確率）。
        conf = float(torch.softmax(logits, dim=-1)[0, idx])
        answer = model.config.id2label[idx]
        results.append({"question": q, "answer": answer, "confidence": round(conf, 3)})
    return results


def run_blip(image, questions: list[str], device: torch.device) -> list[dict]:
    """BLIP-VQA（生成型）で各質問に答える。model.generate で回答テキストを生成。"""
    processor, model = H.load_blip_vqa(device)
    results = []
    for q in questions:
        enc = processor(image, q, return_tensors="pt").to(device)
        with torch.inference_mode():
            # CPU 前提なので max_new_tokens は小さく（VQA の答えは数語で十分）。
            out = model.generate(**enc, max_new_tokens=10, num_beams=1)
        answer = processor.decode(out[0], skip_special_tokens=True).strip()
        results.append({"question": q, "answer": answer})
    return results


def main() -> None:
    out = H.ensure_output_dir()
    device = H.pick_device()
    print(f"device = {device}")

    # 評価対象は合成シーンA（左に赤い円・右に青い四角）。
    bench = H.build_benchmark()
    image = bench[0]["image"]  # シーンA の画像

    summary: dict = {"questions": QUESTIONS, "vilt": None, "blip": None, "note": ""}

    # --- 分類型 ViLT ---
    try:
        vilt = run_vilt(image, QUESTIONS, device)
        summary["vilt"] = vilt
        print("\n[ViLT 分類型] 3129語の候補から選ぶ（速い・決定的）")
        for r in vilt:
            print(f"  Q: {r['question']:34s} A: {r['answer']:10s} (conf={r['confidence']:.2f})")
    except Exception as e:  # noqa: BLE001  モデルDL失敗等でも exit 0 を守る
        summary["note"] += f"ViLT skip: {H.hint_for_model_error(e)} "
        print(f"\n[ViLT] スキップ: {H.hint_for_model_error(e)}")

    # --- 生成型 BLIP-VQA ---
    try:
        blip = run_blip(image, QUESTIONS, device)
        summary["blip"] = blip
        print("\n[BLIP-VQA 生成型] 回答テキストを generate（柔軟・少し遅い）")
        for r in blip:
            print(f"  Q: {r['question']:34s} A: {r['answer']}")
    except Exception as e:  # noqa: BLE001
        summary["note"] += f"BLIP skip: {H.hint_for_model_error(e)} "
        print(f"\n[BLIP-VQA] スキップ: {H.hint_for_model_error(e)}")

    # --- 結果を1枚絵に（左:画像 / 右:両モデルの回答） ---
    lines = ["[ViLT vs BLIP-VQA]", ""]
    for i, q in enumerate(QUESTIONS):
        lines.append(f"Q{i + 1}: {q}")
        if summary["vilt"]:
            lines.append(f"   ViLT: {summary['vilt'][i]['answer']}")
        if summary["blip"]:
            lines.append(f"   BLIP: {summary['blip'][i]['answer']}")
        lines.append("")
    H.save_qa_panel(image, lines, out / "01_vqa_basics.png", title="Scene A (red circle / blue square)")
    (out / "01_vqa_basics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\nまとめ:")
    print("  - 分類型(ViLT)は語彙内なら高速・安定だが、候補に無い答えは出せない。")
    print("  - 生成型(BLIP)は柔軟だが表記ゆれ・数え間違いが出やすい（合成図形は学習分布外）。")
    print(f"saved: {out/'01_vqa_basics.png'}, {out/'01_vqa_basics.json'}")


if __name__ == "__main__":
    main()
