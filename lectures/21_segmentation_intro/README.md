# 19_segmentation_intro: セマンティックセグメンテーション 入門 — DeepLab/FCN/LR-ASPP・SegFormer・mIoU/Dice

> トラック: **セグメンテーション** ／ レベル: **入門** ／ 必要な依存グループ: `dl` `hf` `metrics`

## 🎯 この章のゴール
分類との違い(ピクセル単位ラベル)を理解し、torchvision(deeplabv3/fcn/lraspp_mobilenet軽量)の出力out['out']をargmaxでクラスマップ化、HF pipeline('image-segmentation')のSegFormerで即推論し、クラスID→パレット色で可視化、評価指標mIoU/Dice/pixel accを自力とtorchmetricsで計算できる。

## 扱うトピック
- torchvision: deeplabv3/lraspp、出力dict out['out']のargmax(1)
- HF pipeline('image-segmentation')でSegFormer(nvidia/segformer-b0)、返り値[{label,mask}]
- クラスID→パレット色の可視化とnn.functional.interpolate
- pixel accuracy・per-class IoU・mIoU・Dice(=F1)・FWIoUの定義
- background/ignore_indexと未出現クラスの扱い
- ピクセル混同行列からの指標算出

## 主要API
`torchvision.models.segmentation.lraspp_mobilenet_v3_large` / `output['out']` / `logits.argmax(dim=1)` / `pipeline('image-segmentation')` / `nvidia/segformer-b0-finetuned-ade-512-512` / `torchmetrics.segmentation.MeanIoU` / `torchmetrics.classification.Dice`

## 評価方法
セグメンテーション精度を、予測マスクとGTマスクのピクセル混同行列からpixel acc・per-class IoU・mIoU=平均IoU・Dice=2TP/(2TP+FP+FN)を自作実装で計算し、torchmetrics.segmentation.MeanIoU/Diceと一致確認する。ignore_index・未出現クラスのNaN扱いも明示。

## 完成物
画像をセマンティックセグメンテーションして色付きマスクを保存し、mIoU/Dice/pixel accを自作とtorchmetricsで出す評価スクリプト。

## CPU / GPU メモ
CPU向けにlraspp_mobilenet_v3_largeやsegformer-b0/b1を既定。出力dictをそのままargmaxしない(out['out']を取る)。

## 予定スクリプト
- `01_torchvision_semseg.py`
- `02_segformer_pipeline.py`
- `03_miou_dice_eval.py`

---
> ⚠️ この回はロードマップ上の**プレースホルダ**です。教材本体（解説＋実行コード＋演習）は順次作成します。

> 依存追加の例: `uv add --group dl <packages>`（必要グループ: `dl` `hf` `metrics`）
