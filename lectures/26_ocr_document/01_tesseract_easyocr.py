"""01: 古典 Tesseract と 深層 EasyOCR ― 座標付き OCR の2系統を体験する。

OCR は本来「テキスト検出（どこに文字があるか＝box）」と「テキスト認識（その box の文字列）」
の2段で成り立つ。Tesseract も EasyOCR も、この2段をまとめて担い、**文字列だけでなく
単語ごとの座標（bbox）と確信度（conf）** を返せるのが TrOCR（生成のみ）との大きな違い。

このスクリプトの方針:
  - pytesseract / easyocr は本講座の既定依存に**含めない**（OS パッケージや巨大モデルを
    伴うため）。import は try/except でガードし、未導入なら「導入方法」と「出力フォーマット」
    を案内するだけにする（＝必ず exit 0）。
  - 導入済みなら、合成テキスト行に対して実際に推論し、単語 box と conf を可視化・保存する。

実行: uv run python lectures/26_ocr_document/01_tesseract_easyocr.py
出力: lectures/26_ocr_document/outputs/01_*.png / 01_engines.json
"""

from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

import ocr_helpers as H  # noqa: E402


def try_tesseract(image: Image.Image) -> dict:
    """pytesseract で OCR。未導入/未設定なら案内だけ返す（落ちない）。"""
    try:
        import pytesseract  # ラッパー本体（別途 OS の tesseract-ocr が必要）
    except Exception:
        return {
            "available": False,
            "reason": "pytesseract 未導入",
            "install": "uv add --group ocr pytesseract  +  OS 側: apt-get install -y tesseract-ocr tesseract-ocr-jpn",
            "api": "image_to_string(img) で全文、image_to_data(img, output_type=DICT) で "
            "level/left/top/width/height/conf/text の列（単語ごとの box と確信度）",
        }
    try:
        # image_to_string: 全文を1つの文字列で。image_to_data: 単語ごとの座標+conf。
        text = pytesseract.image_to_string(image).strip()
        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        words = []
        for i in range(len(data["text"])):
            w = data["text"][i].strip()
            if w and int(float(data["conf"][i])) >= 0:
                words.append(
                    {
                        "text": w,
                        "conf": float(data["conf"][i]),
                        "box": [data["left"][i], data["top"][i], data["width"][i], data["height"][i]],
                    }
                )
        return {"available": True, "text": text, "words": words}
    except Exception as e:  # noqa: BLE001  (TesseractNotFoundError 等)
        return {
            "available": False,
            "reason": f"{type(e).__name__}: {e}",
            "install": "OS に tesseract 本体が無い可能性: apt-get install -y tesseract-ocr",
        }


def try_easyocr(image: Image.Image) -> dict:
    """easyocr で OCR。未導入なら案内だけ返す（落ちない）。"""
    try:
        import easyocr  # 初回に検出/認識モデルを自動 DL する
    except Exception:
        return {
            "available": False,
            "reason": "easyocr 未導入",
            "install": "uv add --group ocr easyocr",
            "api": "Reader(['en'], gpu=False).readtext(np_image) → [(bbox4点, text, conf), ...]",
        }
    try:
        reader = easyocr.Reader(["en"], gpu=False)  # gpu=False を明示（CPU 前提）
        results = reader.readtext(np.array(image))
        words = [
            {"text": text, "conf": float(conf), "quad": [[float(x), float(y)] for x, y in box]}
            for (box, text, conf) in results
        ]
        return {"available": True, "words": words}
    except Exception as e:  # noqa: BLE001
        return {"available": False, "reason": f"{type(e).__name__}: {e}"}


def draw_word_boxes(image: Image.Image, words: list[dict]) -> Image.Image:
    """単語 box を画像に重ね描き（Tesseract: [l,t,w,h] / EasyOCR: quad の両対応）。"""
    canvas = image.convert("RGB").copy()
    d = ImageDraw.Draw(canvas)
    for w in words:
        if "box" in w:
            x, y, ww, hh = w["box"]
            d.rectangle([x, y, x + ww, y + hh], outline=(220, 30, 30), width=2)
        elif "quad" in w:
            pts = [tuple(p) for p in w["quad"]]
            d.line(pts + [pts[0]], fill=(30, 120, 220), width=2)
    return canvas


def main() -> None:
    out = H.ensure_output_dir()

    # 入力: 1 行テキストではレイアウトが分かりにくいので、複数行を縦に積んだ「ミニ文書」を作る。
    lines = ["Invoice 2026", "Total: 980 USD", "Serial AB-12345"]
    line_imgs = [H.render_text_line(t, size=30) for t in lines]
    W = max(im.width for im in line_imgs)
    Himg = sum(im.height for im in line_imgs)
    doc = Image.new("RGB", (W, Himg), "white")
    y = 0
    for im in line_imgs:
        doc.paste(im, (0, y))
        y += im.height
    doc.save(out / "01_input_doc.png")

    print("=== 01: Tesseract / EasyOCR（座標付き OCR）===")
    tess = try_tesseract(doc)
    easy = try_easyocr(doc)

    # --- Tesseract ---
    if tess.get("available"):
        print(f"[Tesseract] text = {tess['text']!r}")
        print(f"[Tesseract] 単語数 = {len(tess['words'])}（各 box と conf あり）")
        draw_word_boxes(doc, tess["words"]).save(out / "01_tesseract_boxes.png")
        print(f"  saved: {out / '01_tesseract_boxes.png'}")
    else:
        print(f"[Tesseract] 利用不可: {tess['reason']}")
        print(f"  導入: {tess.get('install', '')}")
        print(f"  出力フォーマット: {tess.get('api', '')}")

    # --- EasyOCR ---
    if easy.get("available"):
        joined = " ".join(w["text"] for w in easy["words"])
        print(f"[EasyOCR] text = {joined!r}")
        print(f"[EasyOCR] 検出語数 = {len(easy['words'])}（各 quad と conf あり）")
        draw_word_boxes(doc, easy["words"]).save(out / "01_easyocr_boxes.png")
        print(f"  saved: {out / '01_easyocr_boxes.png'}")
    else:
        print(f"[EasyOCR] 利用不可: {easy['reason']}")
        print(f"  導入: {easy.get('install', '')}")
        print(f"  出力フォーマット: {easy.get('api', '')}")

    if not tess.get("available") and not easy.get("available"):
        print("\nどちらも未導入のため概念説明のみ。TrOCR は 02、CER/WER 評価は 04 で動かせます。")

    (out / "01_engines.json").write_text(
        json.dumps({"tesseract": tess, "easyocr": easy}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nsaved: {out / '01_input_doc.png'}, {out / '01_engines.json'}")


if __name__ == "__main__":
    main()
