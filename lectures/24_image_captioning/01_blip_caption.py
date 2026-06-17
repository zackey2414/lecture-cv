"""01: BLIP で画像キャプションの最小実装を身につける。

この回の主役は Salesforce/blip-image-captioning-base（小型で CPU でも数秒）。
学ぶ正準フローはたった3手:

    inputs = processor(images=image, return_tensors="pt")   # 前処理
    out    = model.generate(**inputs)                        # 生成
    text   = processor.decode(out[0], skip_special_tokens=True)  # 復号

さらに3つを体験する:
  1. 無条件キャプション（画像だけ渡す）
  2. 条件付きキャプション（text="a photo of" のような接頭辞を与える）
  3. 高レベル API pipeline("image-text-to-text") で同じことを1行で

実行: uv run python lectures/24_image_captioning/01_blip_caption.py
出力: lectures/24_image_captioning/outputs/01_*.png / 01_blip_results.json
"""

from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import torch  # noqa: E402

import caption_helpers as H  # noqa: E402  同じフォルダの道具箱


def run_unconditional_and_conditional() -> dict:
    """processor→generate→decode の3手で、無条件/条件付きキャプションを作る。"""
    device = H.pick_device()

    # AutoModelForImageTextToText で BLIP をロード（v5 の正準。旧 Vision2Seq は廃止）。
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from transformers.utils import logging as hf_logging

    hf_logging.set_verbosity_error()
    model_id = H.MODELS["blip"]["id"]
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForImageTextToText.from_pretrained(model_id).to(device).eval()

    images, names, _ = H.build_gallery()

    results = []
    for image, name in zip(images, names):
        # --- 無条件キャプション: 画像だけを前処理して渡す ---
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.inference_mode():  # eval() ＋ inference_mode が CPU 推論の作法
            uncond_ids = model.generate(**inputs, max_new_tokens=20, num_beams=3)
        uncond = processor.decode(uncond_ids[0], skip_special_tokens=True).strip()

        # --- 条件付きキャプション: 接頭辞 text を与えると続きを生成する ---
        prompt = "a photo of"
        cond_inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
        with torch.inference_mode():
            cond_ids = model.generate(**cond_inputs, max_new_tokens=20, num_beams=3)
        cond = processor.decode(cond_ids[0], skip_special_tokens=True).strip()

        results.append(
            {
                "image": name,
                "unconditional": uncond,
                "conditional_prompt": prompt,
                "conditional": cond,
            }
        )

    return {
        "model": model_id,
        "device": str(device),
        "results": results,
        "images": images,
        "names": names,
    }


def run_pipeline(images: list, names: list) -> list[str]:
    """pipeline("image-text-to-text") で無条件キャプションを1行生成する。

    ★v5 の注意: image-text-to-text は text 引数が必須。BLIP の無条件キャプションは
      text="" を渡すと得られる（チャットテンプレートは持たないモデルなので
      messages 形式ではなく images+text を直接渡す）。
    """
    from transformers import pipeline
    from transformers.utils import logging as hf_logging

    hf_logging.set_verbosity_error()
    device = H.pick_device()
    pipe = pipeline(
        task="image-text-to-text",
        model=H.MODELS["blip"]["id"],
        device=0 if device.type == "cuda" else -1,
    )
    outs = []
    for image in images:
        out = pipe(images=image, text="", max_new_tokens=20)
        outs.append(out[0]["generated_text"].strip())
    return outs


def main() -> None:
    out = H.ensure_output_dir()
    summary = run_unconditional_and_conditional()
    images, names = summary.pop("images"), summary.pop("names")

    # pipeline 版も走らせて、手書きと同じ結果が得られることを確認する。
    pipe_caps = run_pipeline(images, names)
    for r, pc in zip(summary["results"], pipe_caps):
        r["pipeline"] = pc

    # 図と json に保存。
    grid_caps = [f"{r['image']}:\n{r['unconditional']}" for r in summary["results"]]
    H.save_caption_grid(
        images, grid_caps, out / "01_blip_unconditional.png", title="BLIP unconditional captions"
    )
    (out / "01_blip_results.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # コンソール出力（読みながら理解できるように）。
    print(f"model = {summary['model']}  device = {summary['device']}")
    for r in summary["results"]:
        print(f"\n[{r['image']}]")
        print(f"  無条件      : {r['unconditional']}")
        print(f"  条件付き    : '{r['conditional_prompt']}' -> {r['conditional']}")
        print(f"  pipeline    : {r['pipeline']}")
    print("\n手書き(processor→generate→decode) と pipeline は実質同じ処理。")
    print(f"saved: {out / '01_blip_unconditional.png'}, {out / '01_blip_results.json'}")


if __name__ == "__main__":
    main()
