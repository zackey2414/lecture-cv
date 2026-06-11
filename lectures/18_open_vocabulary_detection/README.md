# 18_open_vocabulary_detection: オープン語彙物体検出 — OWL-ViT/OWLv2・Grounding DINO

> トラック: **検出** ／ レベル: **中級** ／ 必要な依存グループ: `dl` `hf`

## 🎯 この章のゴール
任意のテキストラベルで物体を検出する手法を理解し、OWL-ViT/OWLv2(candidate_labels)とGrounding DINO(小文字+ピリオド区切りキャプション)を使い分け、post_process_grounded_object_detection/post_process_object_detection(target_sizes=(H,W))でbox/score/labelを取り出し、box_threshold/text_threshold調整で過検出/未検出を制御できる。

## 扱うトピック
- pipeline('zero-shot-object-detection')とAutoModelForZeroShotObjectDetection
- OWL-ViT/OWLv2のcandidate_labelsによる検出
- Grounding DINOのキャプション形式('a cat. a remote.')とbox/text閾値
- post_process_*のtarget_sizes=(H,W)と座標変換
- 閉語彙(16回)/開語彙の位置づけとCLIP系検索との関係
- Cluster-CLIP baselinesのOWLv2実装との対応

## 主要API
`pipeline('zero-shot-object-detection')` / `Owlv2ForObjectDetection` / `AutoModelForZeroShotObjectDetection` / `IDEA-Research/grounding-dino-tiny` / `processor.post_process_grounded_object_detection` / `google/owlvit-base-patch32`

## 評価方法
任意ラベル検出を、GTがある画像でprecision/recall(IoU≥0.5マッチ)とmAP(17回の自作mAPまたはtorchmetrics)で評価し、box_threshold/text_thresholdをスイープしてP-R曲線とF1最大点を求めて閾値選択の妥当性を定量化する。

## 完成物
テキストラベルを与えて未学習カテゴリを検出し、閾値スイープでP/R/F1を出すOWLv2/Grounding DINO比較スクリプト。

## CPU / GPU メモ
CPUはowlvit-base-patch32/owlv2-base/grounding-dino-tinyを既定に。Grounding DINOのキャプションは小文字+ピリオド区切り必須、timm依存。

## 予定スクリプト
- `01_owlvit_owlv2.py`
- `02_grounding_dino.py`
- `03_threshold_sweep_eval.py`

---
> ⚠️ この回はロードマップ上の**プレースホルダ**です。教材本体（解説＋実行コード＋演習）は順次作成します。

> 依存追加の例: `uv add --group dl <packages>`（必要グループ: `dl` `hf`）
