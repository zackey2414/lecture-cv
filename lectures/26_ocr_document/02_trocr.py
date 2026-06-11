"""02: TrOCR ― Transformer 系の生成 OCR（VisionEncoderDecoder + generate）。

TrOCR は『ViT（画像）エンコーダ + テキストデコーダ』の seq2seq モデル。Tesseract/EasyOCR の
ように box を出すのではなく、**画像を見て１文字ずつテキストを“生成”** する（generate）。
1 行（行検出済みのクロップ）の認識に強く、microsoft/trocr-base-printed は印字用の既定。

このスクリプトで確認すること:
  1. processor で画像 → pixel_values、model.generate で系列生成、batch_decode で文字列化。
  2. 合成テキスト行に対する予測を GT と比べ、CER（文字誤り率）を測る。
  3. greedy(num_beams=1) と beam search(num_beams=4) の違いを見る（生成パラメータの制御）。
  4. 印字 TrOCR は出力が大文字寄りになりがち → 正規化（小文字化）の効果を実感する。

実行: uv run python lectures/26_ocr_document/02_trocr.py
出力: outputs/26_ocr_document/02_*.png / 02_trocr.json
"""

from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import ocr_helpers as H  # noqa: E402


def main() -> None:
    out = H.ensure_output_dir()
    device = H.pick_device()
    print(f"=== 02: TrOCR ({H.TROCR_MODEL_ID}) / device = {device} ===")

    processor, model = H.load_trocr(device)
    if model is None:
        # ロード失敗（ネット不通など）でも exit 0。後続の学習に支障が出ないよう案内のみ。
        print("TrOCR を読み込めませんでした。ネット接続を確認のうえ再実行してください。")
        (out / "02_trocr.json").write_text(
            json.dumps({"available": False}, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return

    records = []
    refs, preds = [], []
    for spec in H.TEXT_LINES:
        gt = spec["text"]
        img = H.render_text_line(gt, size=spec["size"])
        pred = H.trocr_read(processor, model, img, device, num_beams=4)
        c_raw = H.cer(gt, pred, lower=False)   # 大文字小文字を区別したまま
        c_norm = H.cer(gt, pred, lower=True)   # NFKC + 小文字化で正規化
        refs.append(gt)
        preds.append(pred)
        records.append(
            {"id": spec["id"], "gt": gt, "pred": pred,
             "CER_raw": round(c_raw, 4), "CER_norm": round(c_norm, 4),
             "image": img, "cer": c_norm}
        )
        print(f"  {gt!r:24s} -> {pred!r:26s}  CER(raw)={c_raw:.3f}  CER(norm)={c_norm:.3f}")

    # コーパス全体のマイクロ平均 CER（正規化後）
    micro = H.corpus_cer(refs, preds, lower=True)
    micro_raw = H.corpus_cer(refs, preds, lower=False)
    print(f"\nコーパス CER(raw)  = {micro_raw:.4f}  ← 大文字寄りの出力で見かけ上 悪化")
    print(f"コーパス CER(norm) = {micro:.4f}  ← 小文字化で揃えると本来の認識力が見える")

    # greedy と beam の比較（最初の難しめの行 line6 で）
    hard = next(s for s in H.TEXT_LINES if s["id"] == "line6")
    hard_img = H.render_text_line(hard["text"], size=hard["size"])
    greedy = H.trocr_read(processor, model, hard_img, device, num_beams=1)
    beam = H.trocr_read(processor, model, hard_img, device, num_beams=4)
    print(f"\n[生成パラメータ] '{hard['text']}'（小さめ文字で難しめ）")
    print(f"  greedy(num_beams=1) -> {greedy!r}  CER(norm)={H.cer(hard['text'], greedy):.3f}")
    print(f"  beam  (num_beams=4) -> {beam!r}  CER(norm)={H.cer(hard['text'], beam):.3f}")

    # 図と json を保存
    H.save_lines_panel(records, out / "02_trocr_lines.png")
    payload = {
        "model": H.TROCR_MODEL_ID,
        "corpus_CER_raw": round(micro_raw, 4),
        "corpus_CER_norm": round(micro, 4),
        "lines": [{k: v for k, v in r.items() if k != "image"} for r in records],
        "beam_compare": {"text": hard["text"], "greedy": greedy, "beam4": beam},
    }
    (out / "02_trocr.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nsaved: {out / '02_trocr_lines.png'}, {out / '02_trocr.json'}")


if __name__ == "__main__":
    main()
