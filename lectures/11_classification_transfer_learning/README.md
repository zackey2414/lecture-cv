# 11_classification_transfer_learning: 画像分類と転移学習 — ResNet/ViT(torchvision/timm/HuggingFace)

> トラック: **深層CV(分類)** ／ レベル: **中級** ／ 必要な依存グループ: `dl` `hf`

## 🎯 この章のゴール
CNN(残差接続)とViT(パッチ埋め込み+CLS)の仕組みを概念と実装で理解し、torchvision/timm/HFから事前学習重みをロードして推論、最終層付け替え+特徴抽出器凍結で小データをfine-tuneでき、pipelineと手書き(processor+model分離)の両方を書ける。

## 扱うトピック
- ResNetの畳み込み+残差、ViTのパッチ/位置埋め込み/CLS
- torchvision.models(weights API)とtimm.create_model(pretrained)
- HF: pipeline('image-classification')とAutoImageProcessor+ResNet/ViTForImageClassificationの手書き
- 転移学習: requires_grad_(False)凍結とnn.Linearヘッド付け替え、学習率の段差
- id2labelでのラベル変換、torch.no_grad/inference_mode
- timmのforward_features/num_classes=0とcreate_transformでモデル固有前処理

## 主要API
`torchvision.models.resnet50` / `timm.create_model` / `AutoImageProcessor` / `ViTForImageClassification` / `model.config.id2label` / `param.requires_grad_(False)` / `nn.Linear` / `torch.inference_mode`

## 評価方法
小分類データセット(CIFAR-10部分集合等)でtop-1/top-5 accuracyと混同行列をtorchmetrics(Accuracy, ConfusionMatrix)で算出し、素の事前学習モデルとヘッド付け替えfine-tune後でaccuracyを比較してmacro-F1も併記する。

## 完成物
事前学習ResNet/ViTを特徴抽出器として小データに転移学習し、accuracy/混同行列を出力する学習・評価スクリプト(pipeline版と手書き版の両方)。

## CPU / GPU メモ
CPUで現実的な小モデル(resnet18/vit_tiny/mobilenetv3_small)・小バッチ・少エポックに限定。transformers 5.xはAutoImageProcessor必須(AutoFeatureExtractor廃止)。

## 予定スクリプト
- `01_pipeline_classify.py`
- `02_resnet_vit_manual.py`
- `03_transfer_finetune.py`

---
> ⚠️ この回はロードマップ上の**プレースホルダ**です。教材本体（解説＋実行コード＋演習）は順次作成します。

> 依存追加の例: `uv add --group dl <packages>`（必要グループ: `dl` `hf`）
