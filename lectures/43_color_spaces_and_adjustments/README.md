# 43_color_spaces_and_adjustments: 色空間と画像の調整 — 明るさ・彩度・色相・コントラスト・ガンマ・ホワイトバランス

> トラック: **画像の基礎** ／ レベル: **初級** ／ 必要な依存グループ: （基礎のみ・追加依存なし）
> 前提モジュール: `03_image_transforms`

## 🎯 この章のゴール
画像が RGB(色)だけでなく 明るさ(輝度)・彩度・色相・明度・コントラスト といった軸を持つことを理解し、HSV/HSL/Lab/YCbCr など目的別の色空間で『色情報と明るさ情報を分離して』扱える。明るさ/コントラスト/ガンマ/露出・彩度/色相の調整、輝度チャンネルのヒストグラム平坦化(色を保つ)、ホワイトバランス/色恒常性、HSV/Lab ベースの領域抽出(肌色・特定色)、知覚的色差ΔE(Lab) を自力で実装できる。

## 扱うトピック
- 色空間の地図(RGB/HSV/HSL/Lab/YCbCr/XYZ)と使い分け
- 明るさ(輝度)・彩度・色相・明度・コントラストという軸の意味
- 明るさ/コントラスト/ガンマ補正/露出の調整
- 彩度・色相の調整(HSVで色だけ動かす)
- 輝度チャンネルのヒストグラム平坦化・CLAHE(色を崩さない)
- ホワイトバランス/色恒常性(gray-world 等)
- HSV/Lab ベースの領域抽出(肌色検出・特定色抽出)
- 知覚的色差 ΔE(Lab) と、色以外の per-pixel 情報(alpha/depth/勾配)の発展

## 主要API
`cv2.cvtColor` / `cv2.COLOR_BGR2HSV` / `cv2.COLOR_BGR2Lab` / `cv2.COLOR_BGR2YCrCb` / `cv2.convertScaleAbs` / `cv2.LUT` / `cv2.inRange` / `cv2.createCLAHE` / `np.clip`

## 評価方法
—（調整は定性＋ヒストグラム/平均彩度/ΔE で定量確認）

## 完成物
明るさ/彩度/コントラスト/ガンマ/ホワイトバランスを変えて結果とヒストグラムを並べる調整ツール＋HSV/Lab領域抽出デモ

## CPU / GPU メモ
全てCPU・main依存(numpy/cv2/PIL/matplotlib)のみ。ネット不要。

## 予定スクリプト
- `01_color_spaces_map.py`
- `02_brightness_contrast_gamma.py`
- `03_saturation_hue.py`
- `04_white_balance.py`
- `05_hsv_lab_segmentation.py`
- `mini_project.py`

---
> ⚠️ この回はロードマップ上の**プレースホルダ**です。教材本体（解説＋実行コード＋演習）は順次作成します。
