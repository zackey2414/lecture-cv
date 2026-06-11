# 04_classical_features_matching: 特徴点検出とマッチング — SIFT/ORB・BFMatcher/FLANN・テンプレート・Hough

> トラック: **古典CV** ／ レベル: **初級** ／ 必要な依存グループ: （基礎のみ・追加依存なし）

## 🎯 この章のゴール
局所特徴量の考え方を体得し、SIFT/ORBでキーポイントと記述子を抽出、Loweの比率テスト(0.75)で誤対応を除去するマッチングを書け、テンプレートマッチングとHough変換(直線/円)の限界(回転・スケール非不変、パラメータ過敏)も実験で理解する。

## 扱うトピック
- SIFT(特許失効でmain移行)とORB(NORM_HAMMING)、detectAndCompute
- BFMatcher/FlannBasedMatcher、knnMatchと比率テスト/クロスチェック
- drawKeypoints/drawMatchesによる対応可視化
- テンプレートマッチング(matchTemplate/minMaxLoc)とマルチスケール探索
- Hough変換(Canny→HoughLinesP/HoughCircles)のパラメータ敏感性
- SIFTとORBの速度/頑健性の比較

## 主要API
`cv2.SIFT_create` / `cv2.ORB_create` / `cv2.BFMatcher` / `cv2.FlannBasedMatcher` / `knnMatch` / `cv2.matchTemplate` / `cv2.minMaxLoc` / `cv2.HoughLinesP` / `cv2.HoughCircles`

## 評価方法
マッチング品質を定量化する: 比率テスト後の良マッチ数と、後続のRANSACで得たインライア率(inliers/total)を指標とし、同一物体の異なる視点ペアで良マッチ数・インライア率を測って閾値0.75の妥当性を比較検証する。

## 完成物
2枚画像から良マッチ対応を抽出・可視化し良マッチ数とインライア率を出力するスクリプト、テンプレート/Houghのパラメータスイープ比較。

## CPU / GPU メモ
完全CPU。SIFTは現行opencv-pythonのmainに入っており特許失効でxfeatures2d不要、ORBは特にCPUで高速。

## 予定スクリプト
- `01_sift_orb_match.py`
- `02_template_matching.py`
- `03_hough_lines_circles.py`

---
> ⚠️ この回はロードマップ上の**プレースホルダ**です。教材本体（解説＋実行コード＋演習）は順次作成します。
