# 14_clip_zeroshot_retrieval: CLIP/SigLIPによるゼロショット分類と画像テキスト検索

> トラック: **マルチモーダル** ／ レベル: **中級** ／ 必要な依存グループ: `dl` `hf` `embed`

## 🎯 この章のゴール
画像と言語を共有潜在空間に射影する原理を理解し、CLIPProcessorで同時前処理→logits_per_image→softmax(SigLIPはsigmoid)のゼロショット分類、get_image_features/get_text_featuresを正規化必須(forward内のlogitsは正規化済みという非対称)で取り出し、コサイン類似度でtext→image/image→text検索を書ける。

## 扱うトピック
- 共有埋め込み空間と対照学習の直感、candidate_labelsでのゼロショット
- pipeline('zero-shot-image-classification')とAutoModel手書きの対比
- get_image_features/get_text_featuresは未正規化→F.normalizeが必須
- CLIPはsoftmax(相互排他)・SigLIPはsigmoid(独立)という損失/確率解釈の違い
- 正規化+内積/コサインによるtext→image・image→image検索とtorch.topk
- sentence-transformers(clip-ViT-B-32)/open-clipの高レベルAPI対比

## 主要API
`CLIPModel` / `CLIPProcessor` / `AutoModel` / `model.get_image_features` / `model.get_text_features` / `outputs.logits_per_image` / `torch.softmax` / `torch.sigmoid` / `torch.topk`

## 評価方法
ゼロショット分類はラベル付き小データでtop-1 accuracyを測る。画像テキスト検索はクエリごとのRecall@1/5/10とretrieval mAP(クエリ別APの平均)、MRRをtorchmetrics.retrievalで算出し、正規化忘れ時との結果差も実験で示す。

## 完成物
任意ラベルでゼロショット分類し、画像コレクションへのテキスト検索ランキング(Recall@k/mAP付き)を返すスクリプト(CLIPとSigLIPの確率解釈の違いを並記)。

## CPU / GPU メモ
CPUはopenai/clip-vit-base-patch32が最速の定番、SigLIPはsiglip2-base。SigLIPはsentencepiece/protobufが必要。padding=True忘れに注意。

## 予定スクリプト
- `01_zeroshot_pipeline.py`
- `02_clip_siglip_manual.py`
- `03_text_image_retrieval.py`

---
> ⚠️ この回はロードマップ上の**プレースホルダ**です。教材本体（解説＋実行コード＋演習）は順次作成します。

> 依存追加の例: `uv add --group dl <packages>`（必要グループ: `dl` `hf` `embed`）
