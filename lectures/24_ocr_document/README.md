# 24_ocr_document: OCRと文書理解 — Tesseract/EasyOCR/TrOCR・Donut/LayoutLM・CER/WER

> トラック: **マルチモーダル** ／ レベル: **中級** ／ 必要な依存グループ: `dl` `hf` `ocr` `metrics`

## 🎯 この章のゴール
印字/手書き文字抽出を古典Tesseract・深層EasyOCR・Transformer系TrOCRの3系統で比較し、座標付き出力とgenerateの違いを理解、OCRフリーのDonut(task_prompt→token2json)やLayoutLMv3、document-question-answering pipelineで帳票の構造化/質問応答を実装し、CER/WERで評価できる。

## 扱うトピック
- pytesseract(image_to_string/image_to_data、要OSパッケージtesseract-ocr-jpn)
- easyocr.Reader(['ja','en'],gpu=False).readtext
- TrOCR(VisionEncoderDecoderModel、microsoft/trocr-base-printed)
- Donut(naver-clova-ix/donut-base-finetuned-docvqa、token2json)
- LayoutLMv3とdocument-question-answering pipeline
- 評価指標CER=(S+D+I)/参照文字数・WER、jiwerと前処理(NFKC正規化)

## 主要API
`pytesseract.image_to_string` / `easyocr.Reader` / `TrOCRProcessor` / `VisionEncoderDecoderModel` / `DonutProcessor` / `processor.token2json` / `pipeline('document-question-answering')` / `jiwer.cer`

## 評価方法
認識文字列と正解の編集距離からCER=(置換+削除+挿入)/参照文字数とWERをjiwerで算出(日本語は分かち書き依存のためCERを主指標)。文書QAはDocVQAのANLS概念を紹介しexact-match/編集距離で近似評価する。前処理(小文字化・NFKC)を揃える。

## 完成物
同一文書画像をTesseract/EasyOCR/TrOCRで読み比べCER/WERを出す比較スクリプトと、Donut/DocVQAで帳票から項目抽出・質問応答するコード。

## CPU / GPU メモ
CPUで全系統動作。pytesseractは別途OSにtesseract-ocr(+jpn)が必要でDockerfileにapt-get記述、EasyOCRはgpu=Falseで初回モデルDL、TrOCR/Donutはtrocr-base/donut-baseを既定。

## 予定スクリプト
- `01_tesseract_easyocr.py`
- `02_trocr.py`
- `03_donut_docvqa.py`
- `04_cer_wer_eval.py`

---
> ⚠️ この回はロードマップ上の**プレースホルダ**です。教材本体（解説＋実行コード＋演習）は順次作成します。

> 依存追加の例: `uv add --group dl <packages>`（必要グループ: `dl` `hf` `ocr` `metrics`）
