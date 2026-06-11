# 10_classical_video_motion: 古典的な動画処理 — オプティカルフロー・背景差分・動き解析

> トラック: **動画・ストリーム** ／ レベル: **中級** ／ 必要な依存グループ: （基礎のみ・追加依存なし）

## 🎯 この章のゴール
学習を使わない古典的な動画解析（フレーム差分・背景差分、疎/密オプティカルフロー、meanshift/camshift）を自力で実装し、映像中の動きを抽出・可視化できる。深層のフロー(後の27)や追跡(後の28)との違い・使い分けも説明できる。

## 扱うトピック
- フレーム差分と背景差分（MOG2/KNN）による前景抽出
- Lucas-Kanade 疎オプティカルフロー（goodFeaturesToTrack + calcOpticalFlowPyrLK）
- Farnebäck 密オプティカルフロー（calcOpticalFlowFarneback）とHSV可視化
- meanshift / camshift による色ヒストグラム追跡
- 動き検出・モーション履歴の基礎

## 主要API
`cv2.absdiff` / `cv2.createBackgroundSubtractorMOG2` / `cv2.goodFeaturesToTrack` / `cv2.calcOpticalFlowPyrLK` / `cv2.calcOpticalFlowFarneback` / `cv2.meanShift` / `cv2.CamShift` / `cv2.calcBackProject`

## 評価方法
定性評価（フロー場・前景マスク・追跡窓の可視化が妥当か）。密フローは終点誤差(EPE)の概念に触れる（定量評価は深層フローの27で深掘り）。

## 完成物
合成した動く図形の連番フレームに対し、背景差分・疎/密オプティカルフロー・meanshift追跡で動きを抽出して可視化する一連のスクリプト。

## CPU / GPU メモ
全てCPU・OpenCVで完結。ネット非依存（合成フレームを生成）。

## 予定スクリプト
- `01_frame_diff_bgsub.py`
- `02_optical_flow_lk.py`
- `03_optical_flow_farneback.py`
- `04_meanshift_camshift.py`

---
> ⚠️ この回はロードマップ上の**プレースホルダ**です。教材本体（解説＋実行コード＋演習）は順次作成します。
