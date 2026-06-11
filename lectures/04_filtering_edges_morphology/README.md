# 03_filtering_edges_morphology: フィルタ・エッジ・閾値・モルフォロジー・輪郭・ワーピング

> トラック: **画像の基礎** ／ レベル: **中級** ／ 必要な依存グループ: （基礎のみ・追加依存なし）

## 🎯 この章のゴール
平滑化フィルタ、Sobel/Laplacian/Canny、Otsu/適応的閾値、収縮膨張とオープニング/クロージング、findContoursによる形状解析、ヒストグラム平坦化/CLAHE、アフィン/透視変換を一通り自力で書け、二値化→モルフォロジー→輪郭→ワーピングという古典CVの前処理連鎖を組める。

## 扱うトピック
- 平滑化(blur/GaussianBlur/medianBlur/bilateralFilter/filter2D)
- エッジ(Sobel/Laplacian/Canny、CV_64FとconvertScaleAbs)
- 閾値(threshold/THRESH_OTSU/adaptiveThreshold)
- モルフォロジー(erode/dilate/morphologyEx、getStructuringElement)
- 輪郭抽出と形状解析(findContoursは4系で2返し、contourArea/boundingRect/approxPolyDP)
- ヒストグラム/CLAHE、アフィン・透視変換(getPerspectiveTransform/warpPerspective)による書類まっすぐ化

## 主要API
`cv2.GaussianBlur` / `cv2.Canny` / `cv2.threshold` / `cv2.adaptiveThreshold` / `cv2.morphologyEx` / `cv2.findContours` / `cv2.createCLAHE` / `cv2.getPerspectiveTransform` / `cv2.warpPerspective`

## 評価方法
—

## 完成物
スキャン文書を二値化→輪郭検出→透視変換で正面化する書類補正スクリプトと、フィルタ/エッジ/閾値の効果を並べて保存する比較ツール。

## CPU / GPU メモ
完全CPU。OpenCV4系のfindContoursは(contours,hierarchy)の2返しである点を明示する。

## 予定スクリプト
- `01_smoothing.py`
- `02_edges_canny.py`
- `03_threshold_morphology.py`
- `04_contours_warp.py`

---
> ⚠️ この回はロードマップ上の**プレースホルダ**です。教材本体（解説＋実行コード＋演習）は順次作成します。
