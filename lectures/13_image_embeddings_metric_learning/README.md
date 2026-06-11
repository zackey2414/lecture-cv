# 13_image_embeddings_metric_learning: 画像埋め込みとメトリック学習 — ViT/ResNet特徴・対照/triplet学習

> トラック: **埋め込み・検索** ／ レベル: **中級** ／ 必要な依存グループ: `dl` `hf`

## 🎯 この章のゴール
分類ヘッド無しのViTModel/ResNetModelからlast_hidden_state/pooler_output/CLSトークン/mean poolingで埋め込みを取り出し、L2正規化とコサイン類似度を理解し、contrastive/triplet/InfoNCEで『似たものを近く』に配置するメトリック学習を実装でき、良い埋め込みが検索/分類/クラスタリングを支えることを統合的に理解する。

## 扱うトピック
- ViTのlast_hidden_state[:,0](CLS)/mean poolingとResNetのpooler_output(GAP)の形の違い
- output_hidden_statesで中間層、F.normalizeによるL2正規化
- timm forward_featuresとモデル固有前処理
- コサイン類似度とkNN分類による埋め込み品質評価
- TripletMarginLoss/InfoNCEとハードネガティブ
- CLIPの対照学習がメトリック学習の一種である位置づけ

## 主要API
`ViTModel` / `ResNetModel` / `last_hidden_state` / `pooler_output` / `torch.nn.functional.normalize` / `torch.nn.TripletMarginLoss` / `torch.nn.CosineSimilarity` / `timm.create_model`

## 評価方法
埋め込み品質を、抽出ベクトルでのkNN分類精度(accuracy)と、ラベル付きクエリでのRecall@k(上位kに同クラスが入る割合)で評価する。triplet学習の前後で同指標を比較し、ハードネガティブ採用の効果を定量化する。

## 完成物
ViT/ResNet埋め込みを抽出・L2正規化してkNN分類とRecall@kを測るスクリプトと、小データでtriplet/InfoNCEを回して埋め込み空間を改善する学習コード。

## CPU / GPU メモ
CPUで小モデル・小バッチ。pooler_outputとlast_hidden_stateの形の違い(ResNetは特徴マップ)を取り違えない。

## 予定スクリプト
- `01_vit_resnet_embeddings.py`
- `02_knn_recall_eval.py`
- `03_triplet_infonce.py`

---
> ⚠️ この回はロードマップ上の**プレースホルダ**です。教材本体（解説＋実行コード＋演習）は順次作成します。

> 依存追加の例: `uv add --group dl <packages>`（必要グループ: `dl` `hf`）
