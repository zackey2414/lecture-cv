# 31_multimodal_embeddings: マルチモーダル埋め込みの拡張 — SigLIP2多言語・ImageBind(音声+画像+テキスト)

> トラック: **マルチモーダル** ／ レベル: **上級** ／ 必要な依存グループ: `dl` `hf` `embed` `vector`

## 🎯 この章のゴール
画像-テキストを越える埋め込みを理解し、SigLIP2の多言語・高精度埋め込みで検索を強化し、ImageBindで音声・画像・テキストを1つの空間に束ねるクロスモーダル検索(音→画像など)を体験でき、共有空間でのRecall@kでクロスモーダル検索精度を評価できる。

## 扱うトピック
- SigLIP2(多言語)のget_text_features/get_image_featuresと検索強化
- ImageBindで音声/画像/テキストを単一空間に(ModalityType.VISION/TEXT/AUDIO)
- data.load_and_transform_*とクロスモーダル検索(音→画像)
- FAISS(15回)との接続で大規模化
- ImageBindは実験的・概念デモに留める
- SigLIP/SigLIP2のsigmoid損失と多言語対応

## 主要API
`AutoModel` / `google/siglip2-base-patch16-224` / `model.get_text_features` / `model.get_image_features` / `imagebind_model.imagebind_huge` / `data.load_and_transform_audio_data` / `ModalityType.AUDIO` / `faiss.IndexFlatIP`

## 評価方法
クロスモーダル検索をRecall@k(例: 音声クエリで対応画像が上位kに入る割合)とretrieval mAPで評価し、共有埋め込み空間の品質を定量化する。SigLIP2は多言語クエリでの検索Recall@kをCLIPと比較する。

## 完成物
SigLIP2多言語で画像テキスト検索を強化し、ImageBindで音→画像のクロスモーダル検索を行う小デモと、Recall@k評価コード。

## CPU / GPU メモ
SigLIP2はCPUで動作(sentencepiece必須)。ImageBindは公式PyPIなしでgit+torchaudio依存が重く音声前処理にffmpeg必要、教材は小デモ+概念に留める。

## 予定スクリプト
- `01_siglip2_retrieval.py`
- `02_imagebind_crossmodal.py`
- `03_crossmodal_recall_eval.py`

---
> ⚠️ この回はロードマップ上の**プレースホルダ**です。教材本体（解説＋実行コード＋演習）は順次作成します。

> 依存追加の例: `uv add --group dl <packages>`（必要グループ: `dl` `hf` `embed` `vector`）
