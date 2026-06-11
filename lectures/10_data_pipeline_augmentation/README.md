# 10_data_pipeline_augmentation: PyTorch画像テンソルとデータ拡張 — transforms v2 / albumentations / DataLoader

> トラック: **深層CV(分類)** ／ レベル: **初級** ／ 必要な依存グループ: `dl` `aug`

## 🎯 この章のゴール
HWC↔CHW変換・ToImage/ToDtype/Normalizeのスケーリング、Dataset/DataLoaderでのバッチ化、torchvision transforms v2とalbumentationsによるデータ拡張、学習時のみ拡張・推論は決定論的という原則、CLIP/ImageNetでmean/stdが異なる点を自力で書ける。

## 扱うトピック
- 画像テンソルのHWC↔CHWとToImage/ToDtype(0-1スケール)
- Normalize(ImageNet統計とCLIP専用統計の違い)
- torch.utils.data.Dataset/DataLoaderとnum_workers
- transforms v2(RandomResizedCrop/Flip/ColorJitter)
- albumentations.Composeと検出/セグメ用のbbox/mask同時変換
- 学習時拡張・推論時決定論(eval/no_grad)の原則

## 主要API
`torchvision.transforms.v2` / `v2.ToImage` / `v2.ToDtype` / `v2.Normalize` / `torch.utils.data.Dataset` / `DataLoader` / `albumentations.Compose`

## 評価方法
—

## 完成物
小画像フォルダを読むカスタムDatasetと、学習用(拡張あり)/推論用(決定論)2系統のtransformを切り替えるパイプライン、拡張結果を格子状に保存する確認ツール。

## CPU / GPU メモ
完全CPU。num_workersはCPUコア数に応じて調整。transforms v1とv2の混在(二重スケーリング)を避ける。

## 予定スクリプト
- `01_tensor_layout_normalize.py`
- `02_dataset_dataloader.py`
- `03_augment_v2_albumentations.py`

---
> ⚠️ この回はロードマップ上の**プレースホルダ**です。教材本体（解説＋実行コード＋演習）は順次作成します。

> 依存追加の例: `uv add --group dl <packages>`（必要グループ: `dl` `aug`）
