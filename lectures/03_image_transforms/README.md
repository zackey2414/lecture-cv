# 02_image_transforms: 色空間・描画・幾何変換 — 前処理パイプラインの土台

> トラック: **画像の基礎** ／ レベル: **初級** ／ 必要な依存グループ: （基礎のみ・追加依存なし）

## 🎯 この章のゴール
色空間変換(Gray/HSV、OpenCVのH=0-179スケール)、矩形/テキスト等の描画、リサイズ(dsizeが(W,H)で逆順・補間の使い分け)・反転・クロップ、PIL↔numpy↔cv2の軸順の違いを正準形で書け、検出結果の可視化や前処理に直結する基礎技能を身につける。

## 扱うトピック
- cv2.cvtColor(BGR2GRAY/BGR2HSV)とinRangeによる色マスク
- cv2.line/rectangle/circle/putText(座標(x,y)・BGRタプル・LINE_AA)
- cv2.resize(dsize=(W,H)・INTER_AREA縮小/INTER_CUBIC拡大)・flip・スライスクロップ
- PIL.Image.resize/crop/rotate/thumbnailとResampling.LANCZOS
- size=(W,H)とshape=(H,W)の軸順の混同解消
- EXIF Orientationとexif_transposeによる正規化

## 主要API
`cv2.cvtColor` / `cv2.inRange` / `cv2.rectangle` / `cv2.putText` / `cv2.resize` / `cv2.INTER_AREA` / `cv2.flip` / `PIL.Image.resize` / `ImageOps.exif_transpose`

## 評価方法
—

## 完成物
HSV色域でのオブジェクト抽出マスク生成器と、アスペクト比保持/正方形強制リサイズ・EXIF正規化を含む再利用可能な前処理関数群。

## CPU / GPU メモ
完全CPU。すべてOpenCV/Pillowのprebuiltホイールで動作する。

## 予定スクリプト
- `01_colorspace_hsv_mask.py`
- `02_drawing.py`
- `03_resize_crop_flip.py`
- `04_exif_transpose.py`

---
> ⚠️ この回はロードマップ上の**プレースホルダ**です。教材本体（解説＋実行コード＋演習）は順次作成します。
