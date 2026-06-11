# 20_instance_panoptic_sam: インスタンス/パノプティックセグメンテーションとSAM — mask AP・PQ

> トラック: **セグメンテーション** ／ レベル: **中級** ／ 必要な依存グループ: `dl` `hf` `detect` `metrics`

## 🎯 この章のゴール
Mask R-CNN(masks(N,1,H,W)確率→>0.5でbool)、Mask2Formerのpost_process_instance/panoptic_segmentation、プロンプト型SAM(点/box入力→post_process_masks)を使い、軽量なSlimSAM/MobileSAM/SAM2-tinyでCPU動作させ、mask AP(COCOeval segm)とPQ=SQ×RQの評価を実装できる。

## 扱うトピック
- maskrcnn_resnet50_fpn_v2の{boxes,labels,scores,masks}とdraw_segmentation_masks(bool要求)
- Mask2Formerのインスタンス/パノプティック後処理({segments_info,segmentation})
- SAM/SamProcessorのinput_points(1=前景/0=背景)/input_boxesとpost_process_masks/iou_scores
- Ultralytics SAM('mobile_sam.pt')/SAM('sam2.1_t.pt')、SlimSAM
- mask AP(COCOeval iouType='segm')とRLE(pycocotools.mask)
- パノプティックPQ=SQ(平均IoU)×RQ(検出F1)、things/stuff統合

## 主要API
`torchvision.models.detection.maskrcnn_resnet50_fpn_v2` / `torchvision.utils.draw_segmentation_masks` / `SamModel` / `SamProcessor` / `processor.post_process_masks` / `pycocotools.cocoeval.COCOeval` / `torchmetrics.detection.PanopticQuality`

## 評価方法
インスタンスはmask IoUベースのmask AP(COCOeval iouType='segm'、box IoUをマスクIoUに置換)で評価。パノプティックはPQ=SQ×RQ(IoU>0.5で一意マッチ、SQ=マッチ片平均IoU、RQ=2TP/(2TP+FP+FN))をtorchmetrics.PanopticQualityで算出。SAMはマスクとGTのIoUで品質確認。

## 完成物
Mask R-CNN/Mask2Formerでインスタンス/パノプティックセグメンテーションし、点/box指定SAMでマスク生成、mask AP・PQを出す実習一式。

## CPU / GPU メモ
CPUはSlimSAM(Zigeng/SlimSAM-uniform-77)/MobileSAM(約3秒/40MB)/SAM2-tiny、swin-large Mask2FormerやViT-Huge SAMは避ける。SAMマスクは低解像→post_process_masks必須。

## 予定スクリプト
- `01_maskrcnn_instance.py`
- `02_mask2former_panoptic.py`
- `03_sam_prompt_seg.py`
- `04_maskap_pq_eval.py`

---
> ⚠️ この回はロードマップ上の**プレースホルダ**です。教材本体（解説＋実行コード＋演習）は順次作成します。

> 依存追加の例: `uv add --group dl <packages>`（必要グループ: `dl` `hf` `detect` `metrics`）
