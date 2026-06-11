# 26_tracking: 物体追跡 — OpenCV CSRT/KCF・ByteTrack・DeepSORTとMOT評価

> トラック: **動画・追跡** ／ レベル: **中級** ／ 必要な依存グループ: `dl` `detect` `track` `metrics`

## 🎯 この章のゴール
検出器なしの単一物体トラッカ(CSRT/KCF)と、検出器(YOLO)出力にフレーム間で同一IDを付与する多物体追跡(ByteTrack/DeepSORT)を実装し、純CPUのByteTrackでリアルタイム追跡を成立させ、MOTA/IDF1/HOTAでトラッキング品質を評価できる。

## 扱うトピック
- 単一物体トラッカ(cv2.TrackerCSRT_create/legacy.TrackerKCF_create、selectROI)
- supervision.ByteTrack(Kalman+Hungarian、純CPU、検出器非依存)
- deep_sort_realtime.DeepSort(外見特徴)との対比
- update_with_detections/sv.DetectionsとID切替の扱い
- 評価: MOTA=1-(FN+FP+IDSW)/GT・IDF1・HOTA=√(DetA×AssA)
- フレーム毎のハンガリアン法による最適対応付け

## 主要API
`cv2.TrackerCSRT_create` / `cv2.legacy.TrackerKCF_create` / `supervision.ByteTrack` / `update_with_detections` / `deep_sort_realtime.DeepSort` / `motmetrics.MOTAccumulator` / `scipy.optimize.linear_sum_assignment`

## 評価方法
多物体追跡をmotmetricsでMOTA(検出誤りとID切替を統合、負値もとる)・MOTP・IDF1(ID単位のF1)で評価し、HOTA(検出DetAと関連付けAssAを分離、IoU閾値で平均)はTrackEvalで算出する。各フレームでGT-予測をハンガリアン法で対応付ける。

## 完成物
YOLO検出+ByteTrackで動画の多物体を追跡しID付き軌跡を描画するスクリプトと、MOTA/IDF1/HOTAを出すMOT評価コード。

## CPU / GPU メモ
ByteTrackはKalman+HungarianでGPU不要・軽量、CPU実演に最適。CSRTはmain、KCF/MOSSEはcv2.legacy(contrib)で名前空間の違いを明示。TrackEvalはPyPI安定版なしでGit pin導入。

## 予定スクリプト
- `01_single_object_tracker.py`
- `02_yolo_bytetrack.py`
- `03_deepsort.py`
- `04_mot_metrics.py`

---
> ⚠️ この回はロードマップ上の**プレースホルダ**です。教材本体（解説＋実行コード＋演習）は順次作成します。

> 依存追加の例: `uv add --group dl <packages>`（必要グループ: `dl` `detect` `track` `metrics`）
