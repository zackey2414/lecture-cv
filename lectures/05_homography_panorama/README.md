# 05_homography_panorama: ホモグラフィ推定とパノラマ合成

> トラック: **古典CV** ／ レベル: **初級** ／ 必要な依存グループ: （基礎のみ・追加依存なし）

## 🎯 この章のゴール
特徴点マッチングの応用として、対応点からfindHomography(RANSAC)で射影変換を推定しwarpPerspectiveで2枚を貼り合わせる手作りパノラマパイプラインを書き、最低4対応点・インライアmask検証の実務注意を理解した上で高レベルのcv2.Stitcherと比較できる。

## 扱うトピック
- 対応点→findHomography(RANSAC)とインライアmask検証
- warpPerspective/perspectiveTransformによる位置合わせと合成
- 重なり領域のブレンディングとシーム
- cv2.Stitcher_createによる自動パノラマとの比較
- 最低4点必要・外れ値除去(比率テスト+RANSAC)の必須性
- 物体位置合わせ(平面物体検出)への応用

## 主要API
`cv2.findHomography` / `cv2.RANSAC` / `cv2.warpPerspective` / `cv2.perspectiveTransform` / `cv2.Stitcher_create`

## 評価方法
推定ホモグラフィの品質を、対応点を変換した後の再投影誤差(平均ユークリッド距離)とRANSACインライア数/比で評価し、手作りパイプラインとcv2.Stitcherの合成結果を重なり領域のSSIMで比較する。

## 完成物
複数枚を順次つなぐ手作りパノラマ合成スクリプトと、再投影誤差・インライア数をレポートする検証コード。

## CPU / GPU メモ
完全CPU。Stitcherはmainに含まれ追加依存不要。

## 予定スクリプト
- `01_homography_ransac.py`
- `02_panorama_manual.py`
- `03_stitcher_compare.py`

---
> ⚠️ この回はロードマップ上の**プレースホルダ**です。教材本体（解説＋実行コード＋演習）は順次作成します。
