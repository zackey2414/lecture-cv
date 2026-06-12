"""実践ユースケース: レシート/書類リーダー（行を切って読み、JSON に構造化する小ツール）。

このスクリプトは「文字画像 → テキスト抽出 → 構造化（行/フィールド）→ JSON 化」までを
1 本でこなす、現実の小ツールの“動く出発点”です。経費精算アプリや帳簿入力の前段で
「レシート写真を 1 枚渡すと、店名・日付・明細（品名と金額）・合計を構造化レコードにして返す」
ような処理を、本章で学んだ TrOCR（生成 OCR）と Donut（OCR フリー文書理解）で実装します。

【mini_project.py との違い（重要）】
  - mini_project.py は「すでに 1 行ずつに切り出された合成行」を TrOCR で読み、CER/WER/ANLS で
    “精度を測る（評価）”ことが主目的でした。教材的な採点パイプラインです。
  - こちらの use_case.py は採点ではなく“成果物を作る”ツールです。複数行が 1 枚に並んだ
    レシート画像を、まず**自前の行検出（横方向の射影プロファイル）で 1 行ずつに分割**し
    （TrOCR は認識専用＝検出は別途必要、という第2節・第5節の要点の実演）、各行を TrOCR で
    読み、**正規表現で店名・日付・明細・合計に構造化**して `receipt.json` に書き出します。
    さらに Donut/DocVQA で合計金額をOCRフリーに**クロスチェック**します。GT を持たない
    “現場の 1 枚”を、そのまま下流システムが使える JSON に変えるのがゴールです。

【データの置き方（実データ優先・合成フォールバック）】
  - data/26_ocr_document/ に画像（.png/.jpg/.jpeg/.bmp/.tif）を置くと、その最初の 1 枚を
    実レシートとして読み取ります（ocr_helpers.user_image_or_none が拾います）。
  - 置かなければ、複数行のレシートを PIL でその場に合成して必ず完走します（exit 0）。
    ネット接続は TrOCR/Donut の重み初回ダウンロードにのみ使います。モデルが落ちても
    （ネット不通など）行検出と構造化の枠組みは動かし、空テキストでも JSON を出します。

【拡張アイデア（ここから練習を広げる）】
  - 行検出を高度化: 二値化（大津法）→ 連結成分や傾き補正を足し、斜めのレシートに対応する。
  - 明細パーサを強化: 数量×単価、軽減税率(*)マーク、通貨記号、小計/税/おつりの抽出を足す。
  - Donut を CORD 版（naver-clova-ix/donut-base-finetuned-cord-v2）に差し替え、token2json が
    返すネストした {menu, total} 構造を直接使う（正規表現を置き換える）。
  - 出力 JSON を会計ソフトの取り込み形式（CSV/freee/MFクラウド風）に整形して書き出す。
  - 信頼度の低い行（TrOCR の繰り返し・空など）を検出して「要確認」フラグを立てる。

実行: uv run python lectures/26_ocr_document/use_case.py
出力: outputs/26_ocr_document/use_case_receipt.png（行検出の可視化）/ use_case_receipt.json（構造化レコード）
"""

from __future__ import annotations

import json
import pathlib
import re
import sys

# 番号付きスクリプトは import 不可だが、道具箱 ocr_helpers.py は再利用してよい。
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import matplotlib.pyplot as plt  # noqa: E402  （ocr_helpers が Agg を設定済み）

import ocr_helpers as H  # noqa: E402


# =====================================================================
# 1) 入力レシート（実画像が無ければ複数行のレシートを合成）
# =====================================================================

# 合成レシートの中身（金額は末尾の整数で、明細パーサの答え合わせがしやすい形）。
# 実画像を使う場合はこの定義は使われない。
_RECEIPT_LINES: list[str] = [
    "MARUI MART",          # 店名（先頭行）
    "2026-06-12 14:32",    # 日付＋時刻
    "Coffee Beans   980",  # 明細: 品名 + 金額
    "Milk 1L        210",
    "Bread          350",
    "TOTAL         1540",  # 合計
]


def build_receipt():
    """複数行のレシート画像を白地・黒文字で 1 枚に描いて返す（合成フォールバック）。

    mini_project が使う build_invoice() は「項目をラベル付きで並べた請求書」だが、
    こちらは実レシートに近い「店名→日付→明細→合計」の縦並びにして、後段の
    行検出（射影プロファイル）と明細パースの練習材料にする。
    """
    line_h = 46           # 1 行ぶんの送り（行間の空白が射影プロファイルの谷になる）
    top, left = 24, 28
    width = 360
    height = top * 2 + line_h * len(_RECEIPT_LINES)
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    for i, text in enumerate(_RECEIPT_LINES):
        # 先頭（店名）だけ少し大きく、他は等倍。フォントは道具箱の内部関数を借用。
        font = H._font(30 if i == 0 else 24)
        draw.text((left, top + i * line_h), text, font=font, fill="black")
    return img


# =====================================================================
# 2) 行検出（横方向の射影プロファイル）― TrOCR は 1 行入力前提
# =====================================================================

def segment_lines(image, min_ink: int = 2, gap: int = 8, pad: int = 4):
    """画像を「文字行」ごとに切り、[(box, 行画像), ...] を返す。

    TrOCR は認識専用で検出を持たない（第5節）。複数行の文書を読むには、まず行を
    切り出す必要がある。ここでは最も単純で頑健な**横方向の射影プロファイル**を使う:
      - グレースケール化し、各行(y)に黒画素（インク）が一定数以上あるかを調べる。
      - インクのある y が連続する区間を 1 つの「行バンド」とみなす。
      - 文字内部の細い隙間で切れすぎないよう、近接バンドは gap 行まで結合する。
    斜め・湾曲したレシートには弱いが、整ったスキャン/合成にはこれで十分。発展課題で
    二値化（大津法）や傾き補正に差し替えるとよい。
    """
    import numpy as np

    gray = np.asarray(image.convert("L"))
    h, w = gray.shape
    row_has_ink = (gray < 128).sum(axis=1) > min_ink  # 各行にインクがあるか（True/False）

    # 連続する True 区間を (y0, y1) のバンドに変換する。
    bands: list[list[int]] = []
    start: int | None = None
    for y, has in enumerate(row_has_ink):
        if has and start is None:
            start = y
        elif not has and start is not None:
            bands.append([start, y])
            start = None
    if start is not None:
        bands.append([start, h])

    # 近接バンドを結合（行内の空白で過分割されるのを防ぐ）。
    merged: list[list[int]] = []
    for b in bands:
        if merged and b[0] - merged[-1][1] <= gap:
            merged[-1][1] = b[1]
        else:
            merged.append(b)

    # 各バンドを上下に少し広げて全幅で切り出す（box = (left, top, right, bottom)）。
    results = []
    for y0, y1 in merged:
        box = (0, max(0, y0 - pad), w, min(h, y1 + pad))
        results.append((box, image.crop(box)))
    return results


# =====================================================================
# 3) 構造化（読み取った行 → 店名/日付/明細/合計の JSON レコード）
# =====================================================================

_DATE_RE = re.compile(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}")  # 2026-06-12 や 2026/6/12
_TRAILING_AMOUNT_RE = re.compile(r"(.*?)[\s:]*([0-9][0-9,]*)\s*$")  # 「品名 ... 金額」


def _to_int_amount(raw: str) -> int | None:
    """"1,540" のような金額文字列を整数に。失敗したら None。"""
    digits = raw.replace(",", "")
    return int(digits) if digits.isdigit() else None


def structure_receipt(lines: list[str]) -> dict:
    """OCR で読んだ行リストを構造化レコードにする（正規表現ベースの素朴なパーサ）。

    返り値の例:
      {"store": "MARUI MART", "date": "2026-06-12",
       "items": [{"name": "Coffee Beans", "price": 980}, ...],
       "total": 1540, "raw_lines": [...]}
    ルールはシンプル: 先頭の非空行＝店名、日付パターン＝date、"total" を含む行＝合計、
    末尾に数字を持つ残りの行＝明細。実レシートに合わせてここを育てるのが練習の肝。
    """
    record: dict = {"store": None, "date": None, "items": [], "total": None, "raw_lines": lines}
    for idx, line in enumerate(lines):
        text = line.strip()
        if not text:
            continue

        # 店名: 最初に現れた非空行（まだ未設定のとき）。
        if record["store"] is None:
            record["store"] = text
            continue

        # 日付: パターンに一致すれば採用（最初の 1 件）。
        m_date = _DATE_RE.search(text)
        if m_date and record["date"] is None:
            record["date"] = m_date.group(0).replace("/", "-")
            continue

        # 合計: "total" を含む行の末尾金額。
        if "total" in text.lower():
            m = _TRAILING_AMOUNT_RE.match(text)
            if m:
                record["total"] = _to_int_amount(m.group(2))
            continue

        # 明細: 末尾に金額を持つ行を {name, price} に分解。
        m = _TRAILING_AMOUNT_RE.match(text)
        if m and m.group(1).strip():
            price = _to_int_amount(m.group(2))
            if price is not None:
                record["items"].append({"name": m.group(1).strip(), "price": price})

    return record


# =====================================================================
# 4) 可視化（行検出の box ＋ 構造化結果を 1 枚に）
# =====================================================================

def save_visualization(image, segments, record: dict, donut_total, path: pathlib.Path) -> None:
    """左に行検出を重ねたレシート、右に構造化 JSON の要約テキストを描いて保存。

    図タイトル・ラベルは CJK 豆腐（□）を避けるため ASCII のみにする（規約）。
    """
    fig, (ax_img, ax_txt) = plt.subplots(1, 2, figsize=(9, 5.2))

    # 左: 検出された各行を矩形で囲んで番号を振る。
    ax_img.imshow(image)
    for i, (box, _crop) in enumerate(segments):
        left, top, right, bottom = box
        ax_img.add_patch(
            plt.Rectangle((left, top), right - left, bottom - top,
                          fill=False, edgecolor="#E4572E", linewidth=1.5)
        )
        ax_img.text(left + 2, top - 2, f"L{i}", color="#E4572E", fontsize=8, va="bottom")
    ax_img.set_xticks([])
    ax_img.set_yticks([])
    ax_img.set_title("line detection (projection profile)")

    # 右: 構造化レコードを読みやすいテキストで（明細は最大 8 行まで）。
    lines = ["Structured record (receipt.json)", ""]
    lines.append(f"store : {record.get('store')}")
    lines.append(f"date  : {record.get('date')}")
    lines.append("items :")
    for it in record.get("items", [])[:8]:
        lines.append(f"   - {it['name']}: {it['price']}")
    lines.append(f"total : {record.get('total')}")
    if donut_total is not None:
        lines.append("")
        lines.append(f"Donut total (cross-check): {donut_total}")
    ax_txt.axis("off")
    ax_txt.text(0.0, 1.0, "\n".join(lines), va="top", ha="left",
                family="monospace", fontsize=10, transform=ax_txt.transAxes)
    ax_txt.set_title("TrOCR read -> regex parse -> JSON")

    fig.suptitle("Use case: receipt reader (TrOCR + Donut)", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# =====================================================================
# 5) パイプライン本体
# =====================================================================

def main() -> None:
    out = H.ensure_output_dir()
    device = H.pick_device()
    print(f"=== 実践ユースケース: レシート/書類リーダー / device = {device} ===")

    # --- 入力: 実画像優先、無ければ合成レシート ---
    user = H.user_image_or_none()
    if user is not None:
        image = user
        print("data/26_ocr_document/ の実画像をレシートとして使用します。")
    else:
        image = build_receipt()
        print("実画像が無いため合成レシートを使用します（data/26_ocr_document/ に画像を置くと差し替え）。")
    image.save(out / "use_case_input.png")

    # --- 行検出（TrOCR は 1 行入力前提なので、まず行に切る）---
    segments = segment_lines(image)
    print(f"行検出: {len(segments)} 行を検出しました。")

    # --- 各行を TrOCR で読む（モデルが無ければ空テキストで継続＝exit 0 を死守）---
    proc, model = H.load_trocr(device)
    line_texts: list[str] = []
    if model is None:
        print("TrOCR を読み込めませんでした。行検出のみ行い、テキストは空で構造化します。")
        line_texts = ["" for _ in segments]
    else:
        for i, (_box, crop) in enumerate(segments):
            text = H.trocr_read(proc, model, crop, device)
            line_texts.append(text)
            print(f"  L{i}: {text!r}")

    # --- 構造化（行 → 店名/日付/明細/合計の JSON レコード）---
    record = structure_receipt(line_texts)
    print("\n構造化レコード:")
    print(f"  store={record['store']!r} date={record['date']!r} "
          f"items={len(record['items'])}件 total={record['total']!r}")

    # --- Donut/DocVQA で合計をクロスチェック（OCR フリー。落ちても続行）---
    donut_total = None
    d_proc, d_model = H.load_donut(device)
    if d_model is not None:
        answer, _parsed = H.donut_ask(d_proc, d_model, image, "What is the total?", device)
        donut_total = answer
        print(f"\n[Donut] What is the total? -> {answer!r}（OCRフリーのクロスチェック）")
    else:
        print("\n[Donut] 読み込めなかったためクロスチェックはスキップします。")

    # --- 成果物の保存（構造化 JSON ＋ 行検出の可視化）---
    save_visualization(image, segments, record, donut_total, out / "use_case_receipt.png")
    payload = {
        "device": str(device),
        "n_lines_detected": len(segments),
        "trocr_available": model is not None,
        "record": record,
        "donut_total_crosscheck": donut_total,
    }
    (out / "use_case_receipt.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nsaved: {out / 'use_case_receipt.png'}, {out / 'use_case_receipt.json'}")
    print("完了。receipt.json をそのまま経費精算/帳簿入力の下流に渡せる形にしてあります。")


if __name__ == "__main__":
    main()
