# 06_camera_calibration_stereo: カメラキャリブレーション・ステレオ・エピポーラ幾何

> トラック: **古典CV** ／ レベル: **中級** ／ 必要な依存グループ: （基礎のみ・追加依存なし）

## 🎯 この章のゴール
内部/外部パラメータと歪み係数の意味を理解し、チェスボードからcalibrateCameraで較正→undistortで歪み補正、さらにステレオ較正→平行化→StereoSGBMで視差マップ→reprojectImageTo3Dで点群化する一連を書け、再投影誤差で品質を評価できる。

## 扱うトピック
- findChessboardCorners/cornerSubPixとcalibrateCamera
- 歪み係数とgetOptimalNewCameraMatrix/undistort
- 基礎/基本行列とエピポーラ拘束(findFundamentalMat)
- stereoCalibrate/stereoRectifyによる平行化
- StereoSGBMによる視差マップとreprojectImageTo3D
- 視差→深度(古典)とDepth Anything(深層・25回)の対比

## 主要API
`cv2.findChessboardCorners` / `cv2.cornerSubPix` / `cv2.calibrateCamera` / `cv2.undistort` / `cv2.findFundamentalMat` / `cv2.stereoRectify` / `cv2.StereoSGBM_create` / `cv2.reprojectImageTo3D`

## 評価方法
calibrateCameraの戻り値であるRMS再投影誤差(画素単位)を品質指標とし、cv2.projectPointsで再投影した点とコーナー検出点の平均誤差を自前でも計算してライブラリ値と一致を確認する。ステレオは既知サイズ物体で視差→距離の妥当性を検証。

## 完成物
チェスボード画像群から内部行列/歪み係数を求めundistortするキャリブレーションツールと、ステレオ対から視差マップ・深度を生成するスクリプト。

## CPU / GPU メモ
完全CPU。チェスボード撮影画像はサンプルを同梱し、撮影できない環境でも完走できるようにする。

## 予定スクリプト
- `01_calibrate_camera.py`
- `02_undistort.py`
- `03_stereo_sgbm_depth.py`

---
> ⚠️ この回はロードマップ上の**プレースホルダ**です。教材本体（解説＋実行コード＋演習）は順次作成します。
