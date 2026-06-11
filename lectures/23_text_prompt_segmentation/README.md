# 21_text_prompt_segmentation: テキストプロンプト/参照セグメンテーション — CLIPSeg・Grounded-SAM

> トラック: **セグメンテーション** ／ レベル: **中級** ／ 必要な依存グループ: `dl` `hf` `detect`

## 🎯 この章のゴール
文で指定した領域をマスク化する手法を理解し、軽量なCLIPSeg(logits+sigmoid+閾値)でCPUテキスト条件付きセグメンテーションを実行し、発展としてGrounding DINOの検出boxをSAMのinput_boxesに渡すGrounded-SAMの2段構成を組め、参照領域のIoU/Diceで評価できる。

## 扱うトピック
- CLIPSegProcessor/CLIPSegForImageSegmentationのテキスト条件付きセグメ
- outputs.logits→sigmoid→閾値でマスク化
- Grounding DINO(box検出)→SAM(input_boxes)のGrounded-SAM 2段構成
- pipeline('mask-generation')との比較
- しきい値選択とマスク品質の関係
- 検出+セグメの連携設計

## 主要API
`CLIPSegProcessor` / `CLIPSegForImageSegmentation` / `CIDAS/clipseg-rd64-refined` / `torch.sigmoid` / `SamModel` / `SamProcessor` / `grounding-dino + SAM input_boxes`

## 評価方法
参照テキストで指定した領域の予測マスクとGTマスクのIoU/Diceで評価し、CLIPSegのsigmoidしきい値をスイープしてIoU最大の閾値を求める。Grounded-SAMはGDINOのbox閾値が最終マスクIoUに与える影響を比較する。

## 完成物
テキストで対象を指定してマスクを生成するCLIPSegスクリプトと、GDINO+SAMのGrounded-SAMパイプライン、IoU/Dice評価コード。

## CPU / GPU メモ
CLIPSegはCPUで軽量に動作。Grounded-SAMはgrounding-dino-tiny+MobileSAM/SAM2-tinyでCPU化する。

## 予定スクリプト
- `01_clipseg.py`
- `02_grounded_sam.py`
- `03_referring_iou_eval.py`

---
> ⚠️ この回はロードマップ上の**プレースホルダ**です。教材本体（解説＋実行コード＋演習）は順次作成します。

> 依存追加の例: `uv add --group dl <packages>`（必要グループ: `dl` `hf` `detect`）
