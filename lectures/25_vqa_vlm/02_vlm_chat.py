"""02: チャット形式 VLM の正準フロー apply_chat_template → generate → batch_decode。

第24回の BLIP は「画像→キャプション」専用だったが、近年の VLM（Vision-Language Model）は
LLM と同じ「チャット」インターフェースを持つ。ユーザの発話（role: user）の content に
"画像" と "テキスト" を並べて入れ、モデルが assistant として返答する。命令に従えるので
VQA・説明・要約・比較などを1つのモデルで何でもこなせる。

この回の核は、どの VLM でも共通の3ステップ:

  1. messages を組み立てる（★画像は別引数ではなく content の中に埋め込む）
  2. processor.apply_chat_template(..., add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt") で入力テンソル化
  3. model.generate(...) → processor.batch_decode(..., skip_special_tokens=True)
     生成プロンプト部分を切り落として「新しく生成された答え」だけを取り出す

ここでは CPU でも数秒で動く超小型 SmolVLM2-256M を使う。大型（Qwen2.5-VL-7B 等）は
同じ流儀だがメモリ十数GB・数分かかるため概念のみ（README 参照）。

注意: SmolVLM のプロセッサは num2words に依存する。未導入なら案内を出して
グレースフルにスキップする（このスクリプトは常に exit 0）。
  有効化: uv add num2words

実行: uv run python lectures/25_vqa_vlm/02_vlm_chat.py
出力: lectures/25_vqa_vlm/outputs/02_*.png / 02_vlm_chat.json
"""

from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import torch  # noqa: E402

import vqa_helpers as H  # noqa: E402


def show_template(processor) -> str:
    """教材用: 画像プレースホルダ付きメッセージが、どんなテキストに展開されるか覓く。

    tokenize=False にすると「実際にモデルへ渡る文字列」が見える。<image> という特殊
    トークンの位置に、後で画像パッチ埋め込みが差し込まれる、という仕組みが分かる。
    """
    placeholder = [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "What color is the circle?"}]}
    ]
    return processor.apply_chat_template(placeholder, add_generation_prompt=True, tokenize=False)


def ask(processor, model, image, question: str, device: torch.device, max_new_tokens: int = 24) -> str:
    """1問だけ VLM に質問し、生成された回答テキストを返す（正準3ステップ）。"""
    # 1) messages（画像は content に埋め込む）
    messages = H.build_chat_messages(image, question)
    # 2) テンソル化（add_generation_prompt=True で「assistant の番」までを用意）
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(device)
    prompt_len = inputs["input_ids"].shape[1]  # 入力（プロンプト）の長さを覚えておく
    # 3) 生成 → デコード（プロンプト部分を切り落とし、新規生成だけを取り出す）
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    answer = processor.batch_decode(generated[:, prompt_len:], skip_special_tokens=True)[0]
    return answer.strip()


def main() -> None:
    out = H.ensure_output_dir()
    device = H.pick_device()
    print(f"device = {device}  model = {H.SMOLVLM_MODEL_ID}")

    # 評価対象は合成シーンB（緑の三角・黄の円・赤の四角）。複数物体で「比較」も試せる。
    bench = H.build_benchmark()
    image = next(b["image"] for b in bench if b["scene_id"] == "B")

    questions = [
        "What color is the triangle? Answer in one word.",
        "How many shapes are in the image?",
        "Which shape is yellow?",
        "Describe the image in one short sentence.",
    ]

    summary: dict = {"model": H.SMOLVLM_MODEL_ID, "template": None, "qa": [], "note": ""}

    try:
        processor, model = H.load_smolvlm(device)
    except Exception as e:  # noqa: BLE001  num2words 未導入 / DL 失敗でも exit 0
        hint = H.hint_for_model_error(e)
        summary["note"] = hint
        print(f"\nSmolVLM をロードできませんでした → {hint}")
        print("チャット VLM の『メッセージ構造』だけ確認して終了します（generate はスキップ）:")
        # モデルが無くても、画像を content に埋め込む“正準形”は見せられる。
        msgs = H.build_chat_messages(image, questions[0])
        # 画像オブジェクトは長いので type だけ表示する。
        shown = [{"role": m["role"], "content": [c.get("type") for c in m["content"]]} for m in msgs]
        print("  messages =", shown)
        summary["qa"] = [{"question": q, "answer": "(skipped)"} for q in questions]
        (out / "02_vlm_chat.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return

    # 教材: チャットテンプレートの展開結果を表示。
    template = show_template(processor)
    summary["template"] = template
    print("\n[apply_chat_template が作る文字列（tokenize=False）]")
    print("  " + template.replace("\n", "\\n"))
    print("  <image> の位置に画像パッチ埋め込みが差し込まれる、という構造に注目。")

    print("\n[VQA / 対話]")
    for q in questions:
        ans = ask(processor, model, image, q, device)
        summary["qa"].append({"question": q, "answer": ans})
        print(f"  Q: {q}\n    A: {ans}")

    # 結果を1枚絵に。
    lines = ["[SmolVLM2-256M chat]", ""]
    for item in summary["qa"]:
        lines.append("Q: " + item["question"])
        lines.append("A: " + item["answer"])
        lines.append("")
    H.save_qa_panel(image, lines, out / "02_vlm_chat.png", title="Scene B (green tri / yellow circ / red sq)")
    (out / "02_vlm_chat.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\nまとめ:")
    print("  - VLM チャットは『画像を content に入れる → apply_chat_template → generate → batch_decode』。")
    print("  - generate 後は prompt 長で切り、新規トークンだけ decode するのが定石。")
    print("  - 大型 VLM(Qwen2.5-VL 等)も同じ流儀。CPU では小型 + max_new_tokens 小 が現実的。")
    print(f"saved: {out/'02_vlm_chat.png'}, {out/'02_vlm_chat.json'}")


if __name__ == "__main__":
    main()
