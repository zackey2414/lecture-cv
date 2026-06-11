# 07_classical_segmentation: 古典セグメンテーションと復元 — Watershed・GrabCut・古典inpaint

> トラック: **古典CV** ／ レベル: **初級** ／ 必要な依存グループ: （基礎のみ・追加依存なし）

## 🎯 この章のゴール
距離変換+マーカ制御のWatershedで接触物体を分離し、矩形指定のGrabCutで対話的に前景抽出、cv2.inpaintで傷消し/物体除去を行え、深層SAM(20回)やLaMa(29回)の前段として人手の事前知識を与える古典手法の感覚と限界を掴む。

## 扱うトピック
- distanceTransform/connectedComponentsとマーカ設計
- watershedによる接触物体分離
- grabCut(GC_INIT_WITH_RECT)による前景抽出
- cv2.inpaint(TELEA/NS)による復元
- パラメータ・初期化への敏感性(前処理込みの体系化)
- 深層手法(SAM/LaMa)への橋渡し

## 主要API
`cv2.distanceTransform` / `cv2.connectedComponents` / `cv2.watershed` / `cv2.grabCut` / `cv2.GC_INIT_WITH_RECT` / `cv2.inpaint` / `cv2.INPAINT_TELEA`

## 評価方法
前景抽出の精度を、GrabCut/Watershed結果と手動アノテーションマスクのIoU=交差/和とDice=2TP/(2TP+FP+FN)で評価する(IoU/Diceは自作の混同行列から算出)。復元は元画像とのPSNR/SSIMで比較。

## 完成物
矩形指定で前景を切り出すGrabCutツールと、マスクIoU/Diceを出力する評価コード、傷消しinpaintスクリプト。

## CPU / GPU メモ
完全CPU。すべてOpenCV mainで動作する。

## 予定スクリプト
- `01_watershed.py`
- `02_grabcut.py`
- `03_inpaint_classic.py`

---
> ⚠️ この回はロードマップ上の**プレースホルダ**です。教材本体（解説＋実行コード＋演習）は順次作成します。
