"""use_case.py — 画像の自動 alt-text / 説明文ジェネレータ（アクセシビリティ向け実用最小版）。

【何をするツールか】
  フォルダ内の画像に BLIP でキャプションを付け、それを Web の `<img alt="...">` 用の
  alt-text に整形して書き出す、アクセシビリティ支援の小ツールです。
  CMS やブログに大量の画像を載せるとき、alt 属性（スクリーンリーダーが読み上げる代替テキスト）
  を1枚ずつ手で書くのは大変です。そこを「下書きを自動生成 → 人が確認・修正」に変えるのが狙い。
  本章 §3 の正準フロー（processor → generate → decode）と §6.2 の CLIPScore を、
  そのまま「実運用の品質スクリーニング付き alt-text 生成」へ落とし込んだものです。

【実データで使うには（ここが本題）】
  data/24_image_captioning/ に自分の .png/.jpg を置くだけで、そのフォルダが対象になります
  （H.load_user_or_synthetic がそれを拾います）。1枚も無いときは合成シーン（夕焼け/赤い車/木）
  に自動フォールバックして必ず完走します（exit 0）。
  まず合成で挙動を掴み、次に自分の画像で実運用に切り替える、という流れを想定しています。

【mini_project.py との違い】
  mini_project.py は章の総合課題＝「複数モデル×生成設定をベンチマークし CIDEr/CLIP の
  リーダーボードを作る」評価寄りの完成形です。
  こちらは現実の小ツールとして完結します。BLIP 1本に絞り、出力は人が読む順位表ではなく
  「Web にそのまま貼れる alt-text・HTML ギャラリー・サイドカー .txt」という成果物です。
  コピーして自分の alt-text 自動付与パイプラインの出発点にしてください。

【alt-text の作法（このツールが内部でやっている整形）】
  - スクリーンリーダーは要素を「画像」と先に読み上げるため、"a photo of" / "an image of"
    のような冗長な前置きは削る（重複読み上げを避ける）。
  - 文頭を大文字化し、末尾にピリオドを付けて自然な区切りにする。
  - 長すぎる alt は読み上げが冗長になるため、目安（既定 125 文字）超過を flag する。
  - CLIPScore（画像とテキストの整合度）が低い説明は hallucination（画像にない記述）の
    疑いとして flag し、人手レビューに回す。←アクセシビリティでは「誤った alt」が最悪なので重要。

【使い方】
  # 既定（BLIP・ビームサーチで安定した1文）で全画像に alt-text を付ける
  uv run python lectures/24_image_captioning/use_case.py
  # モデルを変えて試す（blip / git / vitgpt2 のいずれか）
  uv run python lectures/24_image_captioning/use_case.py git

【出力（すべて outputs/24_image_captioning/ に保存）】
  use_case_alt_text.json   画像ごとの {file, caption, alt_text, clip_score, flags}
  use_case_gallery.html    画像を base64 で埋め込んだ自己完結 HTML（ブラウザ/スクリーンリーダーで確認可）
  use_case_preview.png     画像＋alt-text＋CLIPScore のプレビューグリッド
  alt_text/<name>.alt.txt  画像ごとのサイドカー .txt（多くの CMS/静的サイトが読む形式）

【拡張アイデア（練習）】
  - 言語対応: alt を多言語化する（生成は英語のまま、翻訳 API/モデルで日本語 alt を併記）。
  - キーワード強調: 条件付きキャプション（text="a photo of a"）で語り口を固定し、商品名やブランドを差し込む。
  - 長文版の併設: alt（簡潔）と longdesc（詳細＝max_new_tokens を増やした説明）を出し分ける。
  - 品質ゲート: CLIPScore がしきい値未満の画像だけを「要レビュー」キューに JSON 出力し、人手に回す。
  - CMS 連携: WordPress/microCMS の API に alt を一括 PUT する薄いアダプタを足す。
  - バッチ最適化: 大量画像なら caption_images をミニバッチで回し、CLIP 埋め込みもバッチ化する。
"""

from __future__ import annotations

import base64
import html
import io
import json
import pathlib
import sys

# digit 始まりの番号付きスクリプト（01_*.py 等）は import 不可だが、
# caption_helpers / caption_metrics は import できる。同じフォルダを import パスに足して借りる。
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import caption_helpers as H  # noqa: E402  device 判定・モデルロード・合成画像・図保存
import caption_metrics as M  # noqa: E402  CLIPScore（参照不要の品質指標）

# alt-text が長すぎると読み上げが冗長になる。実務でよく使う目安（厳密な規格値ではない）。
MAX_ALT_CHARS = 125
# CLIPScore がこの値未満なら「画像と説明が合っていない疑い」として要レビュー扱いにする。
# ※本来は手元データで較正すべき閾値。本章 §6.3 の実測（CLIPScore は概ね 25〜32）を踏まえた目安。
WEAK_CLIP_THRESHOLD = 26.0

# スクリーンリーダーが「画像」と先に読み上げるため、alt から削る冗長な前置き。
REDUNDANT_PREFIXES = (
    "a photo of ",
    "an image of ",
    "a picture of ",
    "a close up of ",
    "there is ",
    "this is ",
)


def to_alt_text(caption: str) -> tuple[str, list[str]]:
    """生のキャプションを alt 属性向けに整形し、(alt_text, 警告フラグ) を返す。

    やること（alt-text の作法）:
      1. 前後空白を除去
      2. 冗長な前置き（"a photo of" 等）を削る
      3. 文頭を大文字化し、末尾にピリオドを付ける
      4. 空なら安全なプレースホルダにフォールバック
      5. 長すぎる場合は "too_long" フラグを立てる（切り詰めはせず人手判断に委ねる）
    """
    flags: list[str] = []
    text = caption.strip()

    if not text:
        # 生成に失敗しても落とさない。空の alt="" は「装飾画像」を意味してしまい誤りなので明示する。
        return "(no description generated)", ["empty_caption"]

    # 冗長な前置きを（小文字比較で）1つだけ削る。
    low = text.lower()
    for prefix in REDUNDANT_PREFIXES:
        if low.startswith(prefix):
            text = text[len(prefix) :]
            break

    text = text.strip()
    if text:
        text = text[0].upper() + text[1:]  # 文頭を大文字化
    if text and text[-1] not in ".!?":
        text += "."  # 末尾にピリオド（読み上げの自然な区切り）

    if len(text) > MAX_ALT_CHARS:
        flags.append("too_long")  # 切り詰めず、レビューで縮めてもらう

    return text, flags


def image_to_data_uri(image, size: int = 256) -> str:
    """PIL 画像を base64 の data URI（PNG）に変換する。

    HTML に画像を「埋め込む」ことで、出力 HTML 単体をブラウザやスクリーンリーダーで
    そのまま開いて alt の読み上げを確認できる（外部ファイル参照に依存しない自己完結成果物）。
    プレビュー用途なので長辺 size px に縮小してファイルを軽くする。
    """
    im = image.copy()
    im.thumbnail((size, size))  # 縦横比を保って縮小
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def build_records(images, names, captions, clip_scores) -> list[dict]:
    """1画像 = 1レコードの alt-text 台帳を作る（JSON/HTML/サイドカーの共通データ源）。"""
    records = []
    for name, cap, score in zip(names, captions, clip_scores):
        alt, flags = to_alt_text(cap)
        # CLIPScore が低い＝画像と説明が噛み合っていない疑い → 要レビュー。
        if score < WEAK_CLIP_THRESHOLD:
            flags.append("low_clip_score")
        records.append(
            {
                "file": name,
                "caption": cap,           # モデルの生出力（監査用に残す）
                "alt_text": alt,          # 実際に使う整形済み alt
                "char_count": len(alt),
                "clip_score": round(float(score), 2),
                "flags": flags,           # 空なら自動採用OK、非空なら人手確認推奨
                "needs_review": bool(flags),
            }
        )
    return records


def write_sidecar_txt(records: list[dict], alt_dir: pathlib.Path) -> None:
    """画像ごとのサイドカー .txt（<name>.alt.txt）を書き出す。

    「image.jpg の隣に image.alt.txt を置く」のは多くの CMS / 静的サイトジェネレータが
    読む素直な受け渡し形式。実運用ではこのファイルを画像と一緒に配布する。
    """
    alt_dir.mkdir(parents=True, exist_ok=True)
    for r in records:
        # ファイル名に使えない文字を避けて安全なステムにする。
        safe = "".join(c if (c.isalnum() or c in "-_") else "_" for c in r["file"])
        (alt_dir / f"{safe}.alt.txt").write_text(r["alt_text"] + "\n", encoding="utf-8")


def write_gallery_html(images, records, path: pathlib.Path) -> None:
    """画像を base64 で埋め込んだ自己完結 HTML ギャラリーを書き出す。

    各画像は本物の <img alt="..."> として出力するので、ブラウザやスクリーンリーダーで
    開けば「生成した alt がどう読み上げられるか」を実物で確認できる。
    要レビュー（flags 有り）の画像にはバッジを付けて目立たせる。
    """
    cards = []
    for image, r in zip(images, records):
        uri = image_to_data_uri(image)
        alt_attr = html.escape(r["alt_text"], quote=True)  # alt 属性に安全に埋め込む
        badge = ""
        if r["needs_review"]:
            reasons = html.escape(", ".join(r["flags"]))
            badge = f'<span class="badge">要レビュー: {reasons}</span>'
        cards.append(
            "<figure class='card'>"
            f"<img src='{uri}' alt='{alt_attr}'>"
            f"<figcaption><code>alt=\"{alt_attr}\"</code>"
            f"<div class='meta'>file: {html.escape(r['file'])} / "
            f"CLIPScore: {r['clip_score']} / {r['char_count']} chars {badge}</div>"
            "</figcaption></figure>"
        )

    doc = (
        "<!doctype html><html lang='ja'><head><meta charset='utf-8'>"
        "<title>Auto alt-text gallery</title><style>"
        "body{font-family:sans-serif;margin:24px;background:#fafafa}"
        "h1{font-size:18px}"
        ".grid{display:flex;flex-wrap:wrap;gap:16px}"
        ".card{background:#fff;border:1px solid #ddd;border-radius:8px;padding:8px;width:280px}"
        ".card img{width:100%;height:auto;border-radius:4px;background:#eee}"
        ".meta{color:#555;font-size:12px;margin-top:6px}"
        ".badge{display:inline-block;background:#ffe0e0;color:#a00;border-radius:4px;"
        "padding:1px 6px;margin-left:4px;font-weight:bold}"
        "code{font-size:12px;word-break:break-all}"
        "</style></head><body>"
        "<h1>Auto alt-text gallery (BLIP caption -> alt attribute)</h1>"
        "<p>各画像は実際の &lt;img alt=...&gt; として埋め込んでいます。"
        "スクリーンリーダーで alt の読み上げを確認できます。</p>"
        f"<div class='grid'>{''.join(cards)}</div>"
        "</body></html>"
    )
    path.write_text(doc, encoding="utf-8")


def plot_preview(images, records, path: pathlib.Path) -> None:
    """画像＋alt-text＋CLIPScore のプレビューグリッドを保存する（既存ヘルパを借用）。

    図タイトル/キャプションは matplotlib の豆腐（□）回避のため ASCII 中心にする。
    alt 本文（日本語を含みうる）はコンソール・JSON・HTML 側で確認できる。
    """
    caps = []
    for r in records:
        mark = "  [REVIEW]" if r["needs_review"] else ""
        caps.append(f"{r['file']} (CLIP={r['clip_score']}){mark}\n{r['alt_text']}")
    H.save_caption_grid(images, caps, path, title="Auto alt-text (BLIP)")


def main(argv: list[str]) -> None:
    out = H.ensure_output_dir()
    device = H.pick_device()

    # --- モデル選択（任意の第1引数で blip/git/vitgpt2 を切替。既定は素直に強い BLIP）---
    model_key = "blip"
    if len(argv) > 1:
        if argv[1] in H.MODELS:
            model_key = argv[1]
        else:
            print(f"未知のモデル '{argv[1]}'（候補: {list(H.MODELS)}）。blip を使います。")

    # --- 1) 対象画像の読み込み（実データ優先・合成フォールバック）---
    images, names, _refs = H.load_user_or_synthetic()
    data_dir = pathlib.Path("data") / H.MODULE_ID
    using_real = data_dir.is_dir() and any(
        p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"} for p in data_dir.iterdir()
    )
    source = f"data/{H.MODULE_ID}/（実画像）" if using_real else "合成シーン（夕焼け/赤い車/木）"

    # --- 2) キャプション生成（alt-text は1つに確定したいので安定志向のビームサーチ）---
    cap = H.load_captioner(model_key, device)
    captions = H.caption_images(cap, images, num_beams=3, max_new_tokens=20)

    # --- 3) 品質スクリーニング: CLIPScore（参照不要）で画像と説明の整合度を測る ---
    clip_scores = M.clip_score_pairs(images, captions, device)

    # --- 4) alt-text 台帳を組み立てる（JSON/HTML/サイドカーの共通データ源）---
    records = build_records(images, names, captions, clip_scores)

    # --- 5) 成果物を保存 ---
    write_sidecar_txt(records, out / "alt_text")
    write_gallery_html(images, records, out / "use_case_gallery.html")
    plot_preview(images, records, out / "use_case_preview.png")
    payload = {
        "source": source,
        "model_id": H.MODELS[model_key]["id"],
        "device": str(device),
        "n_images": len(images),
        "max_alt_chars": MAX_ALT_CHARS,
        "weak_clip_threshold": WEAK_CLIP_THRESHOLD,
        "n_needs_review": sum(r["needs_review"] for r in records),
        "records": records,
    }
    (out / "use_case_alt_text.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # --- 6) コンソール要約 ---
    print(f"model={H.MODELS[model_key]['id']}  device={device}  source={source}")
    print(f"画像 {len(images)} 枚に alt-text を生成（要レビュー {payload['n_needs_review']} 枚）\n")
    for r in records:
        flag = f"  [要レビュー: {', '.join(r['flags'])}]" if r["needs_review"] else ""
        print(f"  [{r['file']}] CLIP={r['clip_score']:.1f}")
        print(f"      caption : {r['caption']}")
        print(f"      alt     : {r['alt_text']}{flag}")
    print(
        f"\nsaved: {out/'use_case_alt_text.json'}, {out/'use_case_gallery.html'},"
        f"\n       {out/'use_case_preview.png'}, {out/'alt_text'}/<name>.alt.txt"
    )


if __name__ == "__main__":
    main(sys.argv)
