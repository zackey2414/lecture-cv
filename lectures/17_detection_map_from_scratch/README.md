# 17_detection_map_from_scratch: ★物体検出mAPの自力実装 — IoU→マッチング→PR曲線→AP補間→mAP

> トラック: **評価指標** ／ レベル: **中級** ／ 必要な依存グループ: `dl` `metrics`

## 🎯 この章のゴール
IoUを定義し、予測をconfidence降順に並べてIoU≥閾値の未マッチGTへ貪欲に対応付けTP/FP/FNを決め、累積からprecision/recall列→AP(11点/全点/COCO101点補間)→クラス平均mAP@0.5・IoU 0.50:0.05:0.95平均mAP@[.5:.95]をnumpyで一から実装し、pycocotoolsのCOCOevalと突き合わせて検証する。

## 扱うトピック
- IoU=交差/和とtorchvision.ops.box_iouでの検算
- confidence降順ソート・クラス別評価・1GptへのTP二重カウント防止
- cumsum(tp)/cumsum(fp)→precision/recall列の構築
- AP補間方式の違い(PASCAL11点/全点monotone/COCO101点)
- IoU閾値ループでmAP@0.5とmAP@[.5:.95]、AP_S/M/L・AR
- pycocotools COCOeval(iouType='bbox')での検算、bbox形式(xywh vs xyxy)とmaxDets/areaRng

## 主要API
`np.argsort` / `np.cumsum` / `np.maximum.accumulate` / `torchvision.ops.box_iou` / `pycocotools.coco.COCO` / `pycocotools.cocoeval.COCOeval` / `coco.loadRes` / `COCOeval.summarize`

## 評価方法
本モジュール自体が評価指標の自作実装回。自作mAP(PR曲線面積・101点補間)とpycocotoolsのAP/AP50/AP75が小数点数桁まで一致することを検証データで確認し、補間方式やソート抜けによる差分を意図的に作って原因を説明できるようにする。

## 完成物
GT/予測(COCO形式)からIoUマッチング・PR・AP・mAPをnumpyで計算する自作評価器と、pycocotools COCOevalとの一致レポート。

## CPU / GPU メモ
完全CPU。pycocotools 2.0.11はcp310-cp314のwheel配布で無ビルド導入可。numpy 2.x ABI不一致に注意。

## 予定スクリプト
- `01_iou_matching.py`
- `02_pr_ap_interpolation.py`
- `03_map_vs_pycocotools.py`

---
> ⚠️ この回はロードマップ上の**プレースホルダ**です。教材本体（解説＋実行コード＋演習）は順次作成します。

> 依存追加の例: `uv add --group dl <packages>`（必要グループ: `dl` `metrics`）
