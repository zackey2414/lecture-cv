# 22_image_captioning: 画像キャプション生成 入門 — BLIP/GIT/ViT-GPT2と生成パラメータ・評価

> トラック: **マルチモーダル** ／ レベル: **入門** ／ 必要な依存グループ: `dl` `hf` `metrics`

## 🎯 この章のゴール
Encoder-Decoderキャプショニングを、BLIP/GIT/ViT-GPT2の小型モデルでpipeline('image-text-to-text')とAutoModelForVision2Seq手書きの両方で実装し、model.generateのnum_beams/max_new_tokens/repetition_penaltyで出力を制御、無条件/条件付きキャプションを書け、BLEU/CIDEr/CLIPScoreで品質評価できる。

## 扱うトピック
- BlipForConditionalGeneration/VisionEncoderDecoderModel(vit-gpt2)/GITの違い
- processor(image,return_tensors='pt')→generate→decode(skip_special_tokens=True)
- 無条件キャプションと条件付き(text=...)、transformers v5のimage-text-to-text
- generate制御(num_beams/max_new_tokens/do_sample/repetition_penalty)、バッチbatch_decode
- 評価指標BLEU/METEOR/ROUGE-L/CIDEr(主指標)/SPICE
- 参照不要のCLIPScore

## 主要API
`BlipForConditionalGeneration` / `VisionEncoderDecoderModel` / `nlpconnect/vit-gpt2-image-captioning` / `pipeline('image-text-to-text')` / `model.generate` / `processor.batch_decode` / `torchmetrics.multimodal.CLIPScore`

## 評価方法
生成キャプションを参照キャプションと比較しBLEU(sacrebleu)・ROUGE-L・CIDEr(TF-IDF重みn-gramコサイン、主指標)で評価し、参照不要のCLIPScore(画像-テキスト整合度、torchmetrics)も併記。num_beams/max_new_tokensを変えてスコア変化を観察する。

## 完成物
複数小型モデルで画像にキャプションを付け、生成パラメータを変えながらBLEU/CIDEr/CLIPScoreを出力する比較スクリプト。

## CPU / GPU メモ
CPUはblip-image-captioning-base/vit-gpt2/git-baseを既定、BLIP-2(2.7b)はCPU非現実的でGPU推奨と明記。CIDErは自前ortorchmetrics、SPICE/METEORはJava必須で任意扱い。

## 予定スクリプト
- `01_blip_caption.py`
- `02_vitgpt2_git.py`
- `03_caption_metrics.py`

---
> ⚠️ この回はロードマップ上の**プレースホルダ**です。教材本体（解説＋実行コード＋演習）は順次作成します。

> 依存追加の例: `uv add --group dl <packages>`（必要グループ: `dl` `hf` `metrics`）
