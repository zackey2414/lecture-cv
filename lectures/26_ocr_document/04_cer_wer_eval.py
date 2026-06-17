"""04: OCR の評価 ― CER / WER を「定義から」理解し、エンジンを横並び比較する。

CER（Character Error Rate）= 編集距離(文字) / 参照文字数 = (置換 S + 削除 D + 挿入 I) / N。
WER（Word Error Rate）   = 編集距離(単語) / 参照単語数。いずれも 0 が完璧で、1 を超え得る
（挿入が多いと参照長を超える）。日本語は分かち書きが曖昧なため、WER より CER を主指標にする。

このスクリプトで確認すること:
  1. 編集距離 → CER/WER の定義を、手作り関数 と torchmetrics で一致させて検算する。
  2. 前処理（NFKC・小文字化）を入れる/入れないで CER が大きく変わることを数値で見る。
  3. 同じ合成テキスト行を、利用可能な OCR エンジン（TrOCR は必ず／Tesseract・EasyOCR は
     導入時のみ）で読み、コーパス CER を棒グラフで横並び比較する。
  4. jiwer があれば jiwer.cer とも一致を確認（無ければ案内のみ）。

実行: uv run python lectures/26_ocr_document/04_cer_wer_eval.py
出力: lectures/26_ocr_document/outputs/04_*.png / 04_eval.json
"""

from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import ocr_helpers as H  # noqa: E402


def demo_definition() -> dict:
    """CER/WER の定義を最小例で確認（手作り vs torchmetrics の一致検算）。"""
    ref = "the quick brown fox"
    hyp = "the quikc brown box"  # quick->quikc(置換2), fox->box(置換1)
    mine = {"CER": round(H.cer(ref, hyp), 4), "WER": round(H.wer(ref, hyp), 4)}
    tm = H.torchmetrics_cer_wer([ref], [hyp])
    print("=== CER/WER の定義（最小例）===")
    print(f"  ref = {ref!r}")
    print(f"  hyp = {hyp!r}")
    print(f"  手作り : CER={mine['CER']:.4f}  WER={mine['WER']:.4f}")
    if tm:
        print(f"  torchmetrics: CER={tm['CER']:.4f}  WER={tm['WER']:.4f}  ← 手作りと一致")
    return {"ref": ref, "hyp": hyp, "mine": mine, "torchmetrics": tm}


def demo_normalization() -> dict:
    """正規化（NFKC・小文字化）の有無で CER が変わることを示す。"""
    ref = "Invoice No.123"
    hyp = "ＩＮＶＯＩＣＥ　Ｎｏ．１２３"  # 全角・大文字（OCR が出しがちな揺れを再現）
    raw = round(H.cer(ref, hyp, lower=False), 4)
    # lower=False でも NFKC は効くので、ここでは「NFKC も切った」生の編集距離も見せる
    raw_no_nfkc = round(H.edit_distance(ref, hyp) / len(ref), 4)
    norm = round(H.cer(ref, hyp, lower=True), 4)
    print("\n=== 正規化の効果 ===")
    print(f"  ref = {ref!r}")
    print(f"  hyp = {hyp!r}（全角・大文字）")
    print(f"  CER(正規化なし・生)      = {raw_no_nfkc:.4f}")
    print(f"  CER(NFKC のみ・大小区別) = {raw:.4f}")
    print(f"  CER(NFKC + 小文字化)     = {norm:.4f}  ← 同じ土俵に乗せると一致(=0)")
    return {"ref": ref, "hyp": hyp, "cer_raw": raw_no_nfkc, "cer_nfkc": raw, "cer_norm": norm}


def collect_engines(device) -> dict[str, list[str]]:
    """合成テキスト行を、利用可能な各エンジンで読む。返り値: {engine: [pred,...]}。"""
    engines: dict[str, list[str]] = {}

    # --- TrOCR（必ず試す）---
    proc, model = H.load_trocr(device)
    if model is not None:
        preds = [H.trocr_read(proc, model, H.render_text_line(s["text"], size=s["size"]), device)
                 for s in H.TEXT_LINES]
        engines["TrOCR"] = preds

    # --- Tesseract（導入時のみ）---
    try:
        import pytesseract

        engines["Tesseract"] = [
            pytesseract.image_to_string(H.render_text_line(s["text"], size=s["size"])).strip()
            for s in H.TEXT_LINES
        ]
    except Exception:
        pass

    # --- EasyOCR（導入時のみ）---
    try:
        import easyocr

        reader = easyocr.Reader(["en"], gpu=False)
        easy = []
        for s in H.TEXT_LINES:
            res = reader.readtext(np.array(H.render_text_line(s["text"], size=s["size"])))
            easy.append(" ".join(t for _, t, _ in res))
        engines["EasyOCR"] = easy
    except Exception:
        pass

    return engines


def main() -> None:
    out = H.ensure_output_dir()
    device = H.pick_device()

    defn = demo_definition()
    norm = demo_normalization()

    # --- エンジン横並び比較（コーパス CER）---
    refs = [s["text"] for s in H.TEXT_LINES]
    engines = collect_engines(device)
    engine_cer: dict[str, float] = {}
    per_line: dict[str, list[dict]] = {}
    print("\n=== エンジン別 コーパス CER（合成印字行）===")
    for name, preds in engines.items():
        engine_cer[name] = round(H.corpus_cer(refs, preds), 4)
        per_line[name] = [
            {"gt": r, "pred": p, "CER": round(H.cer(r, p), 4)} for r, p in zip(refs, preds)
        ]
        print(f"  {name:10s} corpus CER = {engine_cer[name]:.4f}")

    if engine_cer:
        H.save_engine_bar(engine_cer, out / "04_engine_cer.png")

    # --- jiwer との一致確認（あれば）---
    jiwer_check = None
    if engines:
        first = next(iter(engines))
        try:
            import jiwer

            hyps = [H.normalize_text(p) for p in engines[first]]
            rns = [H.normalize_text(r) for r in refs]
            jiwer_check = {"engine": first, "jiwer_cer": round(float(jiwer.cer(rns, hyps)), 4),
                           "ours": engine_cer[first]}
            print(f"\n[jiwer] {first}: jiwer.cer={jiwer_check['jiwer_cer']:.4f}  ours={engine_cer[first]:.4f}")
        except Exception:
            print("\n[jiwer] 未導入のためスキップ（uv add --group metrics jiwer で導入可）。"
                  " 本講座は torchmetrics と自前編集距離で代替しています。")

    payload = {
        "definition": defn,
        "normalization": norm,
        "engine_corpus_cer": engine_cer,
        "per_line": per_line,
        "jiwer_check": jiwer_check,
    }
    (out / "04_eval.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    saved = f"{out / '04_eval.json'}"
    if engine_cer:
        saved = f"{out / '04_engine_cer.png'}, " + saved
    print(f"\nsaved: {saved}")


if __name__ == "__main__":
    main()
