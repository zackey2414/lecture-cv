"""第26回 共通の道具箱（OCRと文書理解）。

このモジュールは「読み物」ではなく、各スクリプトが import して使う再利用部品です。
責務を次の6つに絞ってあります（単一責務）:

  1. device の自動判定（cuda > mps > cpu。本講座は cpu 前提）と出力ディレクトリの用意
  2. 合成データの生成（印字テキスト行の画像＋正解文字列、合成請求書＋項目GT＋QA）
  3. テキスト前処理（NFKC 正規化・小文字化・空白圧縮）― CER/WER の土台
  4. 評価指標（編集距離 S+D+I → CER / WER / ANLS）を標準ライブラリだけで自前実装
  5. TrOCR（VisionEncoderDecoder）と Donut（DocVQA）のロード＋推論（失敗しても落ちない）
  6. 図の保存（matplotlib Agg。文字は ASCII のみで 豆腐 を避ける）

なぜ合成データか:
  OCR/文書理解の教材本体は「ネットからデータセットを取得しない」方針で組んでいます。
  入力（印字テキスト行・請求書）は PIL でその場で描画するため、正解文字列・正解項目を
  画素単位で厳密に把握でき、CER/WER や DocVQA の評価が再現可能になります。実画像で
  試したい場合は data/26_ocr_document/ に画像を置けば 03 / mini_project が拾います。
"""

from __future__ import annotations

import os
import pathlib
import re
import unicodedata

# --- HuggingFace / tokenizers のログを静かにして教材の出力を読みやすく ---
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from PIL import Image, ImageDraw, ImageFont

import matplotlib  # noqa: E402

matplotlib.use("Agg")  # headless（GUI なし）でも図を保存できるバックエンド
import matplotlib.pyplot as plt  # noqa: E402

# プロジェクト規約: 結果は outputs/<module_id>/ に集約する。
MODULE_ID = "26_ocr_document"
OUTPUT_DIR = pathlib.Path("outputs") / MODULE_ID
DATA_DIR = pathlib.Path("data") / MODULE_ID  # 実画像があればここから（任意）

# 使用モデル（いずれも CPU で現実的な小型／既定）。環境変数で差し替え可能。
TROCR_MODEL_ID = os.environ.get("TROCR_MODEL", "microsoft/trocr-base-printed")
DONUT_MODEL_ID = os.environ.get("DONUT_MODEL", "naver-clova-ix/donut-base-finetuned-docvqa")

# 印字テキスト行のデータセット（02_trocr.py / 04_cer_wer_eval.py が共有）。
# size を小さくした行（Contract No）は「低解像度＝難しめ」で、認識誤りが出やすい。
TEXT_LINES: list[dict] = [
    {"id": "line1", "text": "Invoice 2026", "size": 32},
    {"id": "line2", "text": "Total: 980 USD", "size": 32},
    {"id": "line3", "text": "Serial AB-12345", "size": 32},
    {"id": "line4", "text": "Date 2026-06-11", "size": 32},
    {"id": "line5", "text": "Hello World", "size": 32},
    {"id": "line6", "text": "Contract No: Z9-8471", "size": 16},  # 小さめ＝難しめ
]

# DejaVu（matplotlib 同梱で必ず存在）を既定フォントにする。OS 側にあればそれも探す。
_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    str(pathlib.Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans-Bold.ttf"),
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]


# =====================================================================
# 1) device / 出力ディレクトリ
# =====================================================================

def pick_device():
    """cuda > mps > cpu の優先で torch.device を返す（全回共通の定石）。

    本講座は CPU 前提だが、GPU/Apple Silicon があれば自動で使う。
    重要: モデルと入力テンソルの device は必ず揃える（.to(device) を両方に）。
    torch は重いのでこの関数内で遅延 import する（純計算のスクリプトは torch 不要）。
    """
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def ensure_output_dir() -> pathlib.Path:
    """outputs/<module_id>/ を作って返す（既にあってもエラーにしない）。"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


# =====================================================================
# 2) 合成データ（印字テキスト行・請求書）
# =====================================================================

def _font(size: int) -> ImageFont.FreeTypeFont:
    """指定サイズの TrueType フォントを返す（候補を順に探し、最後は PIL 既定）。"""
    for path in _FONT_CANDIDATES:
        if pathlib.Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()  # 最終フォールバック（小さいビットマップ）


def render_text_line(text: str, size: int = 32, pad: int = 12) -> Image.Image:
    """1 行の印字テキストを白地・黒文字で描いた RGB 画像を返す（OCR の入力）。

    幅は文字列の見た目幅に合わせて自動調整する。背景は白・文字は黒で、
    実務の「スキャン文書の一行」を模した最小の入力にしている。
    """
    font = _font(size)
    bbox = font.getbbox(text)
    w = max(360, bbox[2] - bbox[0] + 2 * pad)
    h = max(48, size + 2 * pad)
    img = Image.new("RGB", (w, h), "white")
    ImageDraw.Draw(img).text((pad, pad), text, font=font, fill="black")
    return img


def build_invoice() -> tuple[Image.Image, dict[str, str], list[tuple[str, str]]]:
    """合成請求書の画像・正解項目(GT)・想定 QA を返す（Donut/DocVQA 用）。

    返り値:
      image  : 640x420 の請求書画像（白地・黒文字、項目をレイアウトして配置）
      fields : 正解項目 {company, invoice_no, date, total}（構造化抽出の答え合わせ用）
      qa     : [(質問, 正解答え), ...]（DocVQA の答え合わせ用）
    """
    fields = {
        "company": "Acme Corp",
        "invoice_no": "12345",
        "date": "2026-06-11",
        "total": "980 USD",
    }
    img = Image.new("RGB", (640, 420), "white")
    d = ImageDraw.Draw(img)
    d.text((30, 26), "INVOICE", font=_font(40), fill="black")
    d.line((30, 86, 610, 86), fill="black", width=2)
    rows = [
        f"Company: {fields['company']}",
        f"Invoice No: {fields['invoice_no']}",
        f"Date: {fields['date']}",
        "Item: Widget x 4",
        f"Total: {fields['total']}",
    ]
    for i, line in enumerate(rows):
        d.text((30, 120 + i * 52), line, font=_font(26), fill="black")
    qa = [
        ("What is the invoice number?", fields["invoice_no"]),
        ("What is the total?", fields["total"]),
        ("What is the company name?", fields["company"]),
        ("What is the date?", fields["date"]),
    ]
    return img, fields, qa


def user_image_or_none() -> Image.Image | None:
    """data/26_ocr_document/ に画像があれば最初の1枚を RGB で返す（無ければ None）。"""
    if not DATA_DIR.exists():
        return None
    for p in sorted(DATA_DIR.iterdir()):
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
            return Image.open(p).convert("RGB")
    return None


# =====================================================================
# 3) テキスト前処理（CER/WER の土台）
# =====================================================================

def normalize_text(s: str, lower: bool = True) -> str:
    """OCR 評価のための正規化: NFKC → (小文字化) → 前後空白除去 → 連続空白を1つに。

    NFKC は全角英数・全角空白・互換文字を半角の正準形へ畳む（"Ｉｎｖｏｉｃｅ"→"Invoice",
    "１２３"→"123", 全角空白→半角空白）。OCR は全角/半角や大小文字を取り違えやすいため、
    予測と正解を同じ土俵に乗せてから CER/WER を測るのが鉄則。日本語では分かち書きに
    依存する WER より、文字単位の CER を主指標にする。
    """
    s = unicodedata.normalize("NFKC", s)
    if lower:
        s = s.lower()
    return re.sub(r"\s+", " ", s.strip())


# =====================================================================
# 4) 評価指標（編集距離 → CER / WER）
# =====================================================================

def edit_distance(a, b) -> int:
    """レーベンシュタイン距離（置換 S + 削除 D + 挿入 I の最小回数）。

    a, b は任意のシーケンス（文字列でも単語リストでも可）。要素の等価判定しか
    使わないので、文字単位（CER）にも単語単位（WER）にも同じ実装が使える。
    1 行ぶんの DP（メモリ O(n)）で計算する。
    """
    m, n = len(a), len(b)
    if m == 0:
        return n
    if n == 0:
        return m
    prev_row = list(range(n + 1))
    for i in range(1, m + 1):
        cur_row = [i] + [0] * n
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur_row[j] = min(
                cur_row[j - 1] + 1,      # 挿入
                prev_row[j] + 1,         # 削除
                prev_row[j - 1] + cost,  # 置換 or 一致
            )
        prev_row = cur_row
    return prev_row[n]


def cer(reference: str, hypothesis: str, lower: bool = True) -> float:
    """CER = 編集距離(文字) / 参照文字数。前処理(NFKC・小文字化)を内部で揃える。

    参照が空のときは、予測も空なら 0.0、そうでなければ 1.0 を返す（ゼロ割回避）。
    """
    r = normalize_text(reference, lower=lower)
    h = normalize_text(hypothesis, lower=lower)
    if len(r) == 0:
        return 0.0 if len(h) == 0 else 1.0
    return edit_distance(r, h) / len(r)


def wer(reference: str, hypothesis: str, lower: bool = True) -> float:
    """WER = 編集距離(単語) / 参照単語数。空白でトークン化して語の列で編集距離を測る。"""
    r = normalize_text(reference, lower=lower).split()
    h = normalize_text(hypothesis, lower=lower).split()
    if len(r) == 0:
        return 0.0 if len(h) == 0 else 1.0
    return edit_distance(r, h) / len(r)


def corpus_cer(references: list[str], hypotheses: list[str], lower: bool = True) -> float:
    """コーパス全体の CER（マイクロ平均 = 全誤り数 / 全参照文字数）。

    サンプルごとの CER を単純平均（マクロ平均）すると短い行の誤りが過大に効く。
    総編集距離 / 総参照文字数の「マイクロ平均」が標準。torchmetrics も同方式。
    """
    tot_err = tot_chars = 0
    for r, h in zip(references, hypotheses):
        rn = normalize_text(r, lower=lower)
        hn = normalize_text(h, lower=lower)
        tot_err += edit_distance(rn, hn)
        tot_chars += len(rn)
    return 0.0 if tot_chars == 0 else tot_err / tot_chars


def torchmetrics_cer_wer(references: list[str], hypotheses: list[str]) -> dict | None:
    """torchmetrics の CharErrorRate / WordErrorRate で CER/WER を計算（クロスチェック用）。

    予測・正解は同じ正規化を施してから渡す。torchmetrics が無ければ None を返す。
    """
    try:
        from torchmetrics.text import CharErrorRate, WordErrorRate
    except Exception:
        return None
    refs = [normalize_text(r) for r in references]
    hyps = [normalize_text(h) for h in hypotheses]
    return {
        "CER": float(CharErrorRate()(hyps, refs)),
        "WER": float(WordErrorRate()(hyps, refs)),
    }


def anls(reference: str, hypothesis: str, threshold: float = 0.5) -> float:
    """ANLS（Average Normalized Levenshtein Similarity）の1ペア分。DocVQA の標準指標。

    正規化レーベンシュタイン類似度 NL_sim = 1 - 編集距離/max(len)。これが threshold(=0.5)
    以上ならその値を、未満なら 0.0 をスコアにする（綴り揺れには寛容、別物には 0）。
    """
    r = normalize_text(reference)
    h = normalize_text(hypothesis)
    if len(r) == 0 and len(h) == 0:
        return 1.0
    denom = max(len(r), len(h))
    sim = 1.0 - edit_distance(r, h) / denom if denom > 0 else 0.0
    return sim if sim >= threshold else 0.0


# =====================================================================
# 5) モデルのロード＋推論（TrOCR / Donut）― 失敗しても落ちない
# =====================================================================

def load_trocr(device):
    """TrOCR(VisionEncoderDecoder) のプロセッサとモデルを返す。失敗時は (None, None)。

    TrOCR は『ViT エンコーダ + テキストデコーダ』の seq2seq。画像を pixel_values にして
    model.generate で1文字ずつ生成する（分類ではなく系列生成）。
    """
    try:
        import torch
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel

        processor = TrOCRProcessor.from_pretrained(TROCR_MODEL_ID)
        model = VisionEncoderDecoderModel.from_pretrained(TROCR_MODEL_ID).to(device)
        model.eval()
        torch.set_grad_enabled(False)
        return processor, model
    except Exception as e:  # noqa: BLE001
        print(f"[TrOCR] ロード失敗のためスキップします: {type(e).__name__}: {e}")
        return None, None


def trocr_read(processor, model, image: Image.Image, device, num_beams: int = 4) -> str:
    """1 枚の画像から TrOCR で文字列を読み取る（generate → batch_decode）。"""
    import torch

    with torch.inference_mode():
        pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
        ids = model.generate(pixel_values, max_new_tokens=32, num_beams=num_beams)
    return processor.batch_decode(ids, skip_special_tokens=True)[0].strip()


def load_donut(device):
    """Donut(DocVQA) のプロセッサとモデルを返す。失敗時は (None, None)。

    Donut は『Swin エンコーダ + BART デコーダ』の OCR フリー文書理解モデル。OCR を介さず
    画像から直接、構造化トークン列（task_prompt 付き）を生成する。
    """
    try:
        import torch
        from transformers import DonutProcessor, VisionEncoderDecoderModel

        processor = DonutProcessor.from_pretrained(DONUT_MODEL_ID)
        model = VisionEncoderDecoderModel.from_pretrained(DONUT_MODEL_ID).to(device)
        model.eval()
        torch.set_grad_enabled(False)
        return processor, model
    except Exception as e:  # noqa: BLE001
        print(f"[Donut] ロード失敗のためスキップします: {type(e).__name__}: {e}")
        return None, None


def donut_ask(processor, model, image: Image.Image, question: str, device) -> tuple[str, dict]:
    """Donut/DocVQA に質問して (answer 文字列, token2json の dict) を返す。

    task_prompt = "<s_docvqa><s_question>{Q}</s_question><s_answer>" を decoder_input_ids
    に与え、続きを生成させる。生成列を token2json で {question, answer} に構造化する。
    """
    import torch

    task_prompt = f"<s_docvqa><s_question>{question}</s_question><s_answer>"
    decoder_input_ids = processor.tokenizer(
        task_prompt, add_special_tokens=False, return_tensors="pt"
    ).input_ids.to(device)
    pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)
    with torch.inference_mode():
        out = model.generate(
            pixel_values,
            decoder_input_ids=decoder_input_ids,
            max_length=64,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            use_cache=True,
            num_beams=1,
        )
    seq = processor.batch_decode(out)[0]
    seq = seq.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
    parsed = processor.token2json(seq)
    answer = parsed.get("answer", "") if isinstance(parsed, dict) else ""
    return str(answer).strip(), parsed if isinstance(parsed, dict) else {"raw": seq}


# =====================================================================
# 6) 可視化（図の文字は ASCII のみ）
# =====================================================================

def save_lines_panel(records: list[dict], path: pathlib.Path) -> None:
    """テキスト行・正解・予測・CER を縦に並べた図を保存（02 / mini_project 用）。

    records: [{"image": PIL, "gt": str, "pred": str, "cer": float}, ...]
    """
    n = len(records)
    fig, axes = plt.subplots(n, 1, figsize=(7, 1.4 * n + 0.5))
    if n == 1:
        axes = [axes]
    for ax, rec in zip(axes, records):
        ax.imshow(rec["image"])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(
            f"GT: {rec['gt']!r}   PRED: {rec['pred']!r}   CER={rec['cer']:.3f}",
            fontsize=9, loc="left",
        )
    fig.suptitle("TrOCR: synthetic printed lines (GT vs prediction)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def save_engine_bar(engine_cer: dict[str, float], path: pathlib.Path) -> None:
    """エンジン別の平均 CER（マイクロ平均）を棒グラフで保存（04 用）。"""
    names = list(engine_cer.keys())
    vals = [engine_cer[k] for k in names]
    fig, ax = plt.subplots(figsize=(5, 3.2))
    bars = ax.bar(names, vals, color=["#4C78A8", "#F58518", "#54A24B", "#B279A2"][: len(names)])
    ax.set_ylabel("corpus CER (lower is better)")
    ax.set_ylim(0, max(0.05, max(vals) * 1.3) if vals else 1.0)
    ax.set_title("OCR engines: corpus CER on synthetic printed lines")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _smoke_test() -> None:
    """道具箱の自己点検（モデル不要の純計算だけ）。直接実行で動作確認できる。"""
    out = ensure_output_dir()
    # 前処理: 全角・大文字混じりが正準形へ畳まる
    assert normalize_text("  Ｉｎｖｏｉｃｅ　No.１２３ ") == "invoice no.123"
    # 編集距離 / CER / WER の定義確認
    assert edit_distance("kitten", "sitting") == 3
    assert abs(cer("abc", "abd") - 1 / 3) < 1e-9
    assert abs(wer("the cat sat", "the dog sat") - 1 / 3) < 1e-9
    assert abs(corpus_cer(["abc", "de"], ["abd", "de"]) - 0.2) < 1e-9
    assert anls("12345", "12345") == 1.0 and anls("yes", "no") == 0.0
    # 合成データが生成できる
    line = render_text_line("Hello World")
    inv, fields, qa = build_invoice()
    print(f"[smoke] output_dir = {out}")
    print(f"[smoke] line size = {line.size}, invoice size = {inv.size}")
    print(f"[smoke] fields = {fields}")
    print(f"[smoke] qa[0] = {qa[0]}")
    print("[smoke] normalize/edit_distance/cer/wer/anls OK")
    # 合成行と請求書を保存しておく（他スクリプトの入力イメージの確認用）
    line.save(out / "00_sample_line.png")
    inv.save(out / "00_sample_invoice.png")
    print(f"[smoke] saved: {out / '00_sample_line.png'}, {out / '00_sample_invoice.png'}")


if __name__ == "__main__":
    _smoke_test()
