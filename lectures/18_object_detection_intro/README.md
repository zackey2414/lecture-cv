# 16_object_detection_intro: 物体検出 入門 — torchvision weights API・YOLO・DETR/RT-DETR

> トラック: **検出** ／ レベル: **入門** ／ 必要な依存グループ: `dl` `hf` `detect`

## 🎯 この章のゴール
torchvision検出モデル([0,1] float CHWのリスト入力・内部正規化)、Ultralytics YOLO(1行推論)、HF DETR/RT-DETR(post_process_object_detectionとtarget_sizes=(H,W))の3系統で最小推論を書き、score閾値フィルタ・NMS・draw_bounding_boxesでの可視化、COCOラベルのindex0=__background__を扱える。

## 扱うトピック
- torchvision: fasterrcnn等のweights enum・weights.transforms()・出力{boxes(xyxy),labels,scores}
- Ultralytics YOLO('yolo11n.pt')のresults[0].boxes/.plot()、YOLO11(NMS内蔵)とYOLO26(NMS-free)
- HF DETR/RT-DETR: AutoModelForObjectDetection+post_process_object_detection、target_sizesは(H,W)
- score閾値とtorchvision.ops.nms/batched_nms、box_convert/box_iou
- draw_bounding_boxes(uint8画像要求)による可視化
- COCO categoriesとid2label、CPU向け軽量モデル(mobilenet/rtdetr_r18/yolo11n)

## 主要API
`torchvision.models.detection.fasterrcnn_resnet50_fpn_v2` / `weights.transforms()` / `ultralytics.YOLO` / `results[0].boxes.xyxy` / `AutoModelForObjectDetection` / `image_processor.post_process_object_detection` / `torchvision.ops.nms` / `torchvision.utils.draw_bounding_boxes`

## 評価方法
まずscore閾値・NMS後の検出を可視化で定性確認し、定量はtorchmetrics.detection.MeanAveragePrecision(preds/targetをbox辞書で渡す)でmAP@0.5/mAP@[.5:.95]を簡易算出する(本格的な自作実装は17回)。target_sizesの(H,W)取り違えによるbbox歪みの有無も検証。

## 完成物
同一画像をtorchvision/YOLO/DETRの3系統で検出し、閾値・NMS・可視化を統一APIでまとめ、torchmetricsでmAPを出すベンチ用スクリプト。

## CPU / GPU メモ
CPU向けにfasterrcnn_mobilenet_v3_large_320_fpn/ssdlite320/yolo11n/rtdetr_v2_r18vdを既定にし、imgsz/min_sizeを下げる。HFのDETR/RT-DETRはtimmが必須。

## 予定スクリプト
- `01_torchvision_detection.py`
- `02_yolo_oneliner.py`
- `03_detr_rtdetr.py`

---
> ⚠️ この回はロードマップ上の**プレースホルダ**です。教材本体（解説＋実行コード＋演習）は順次作成します。

> 依存追加の例: `uv add --group dl <packages>`（必要グループ: `dl` `hf` `detect`）
