# 08_video_io_basics: 動画I/Oの基礎 — VideoCapture/VideoWriter・メタデータ・FPS

> トラック: **動画・ストリーム** ／ レベル: **入門** ／ 必要な依存グループ: （基礎のみ・追加依存なし）

## 🎯 この章のゴール
動画=連続フレームと理解し、VideoCaptureでファイル/カメラを開きwhileループでretフラグ判定しながらread()→処理→release()する正準パターン、CAP_PROPでのメタデータ取得とシーク、FOURCC指定のVideoWriter書き出し、ソースFPSと処理FPSの違いを書ける。

## 扱うトピック
- VideoCapture/isOpened/read/release の正準ループとretチェック
- フレームはBGR numpy配列、cvtColor/resize/ROIの基本操作
- CAP_PROP(FPS/FRAME_COUNT/WIDTH/HEIGHT)とPOS_FRAMESシーク
- VideoWriter(FOURCC・出力サイズ整合・isOpened検証、mp4v/.mp4とXVID/.avi)
- time.perf_counter+dequeでの処理FPS移動平均
- ライブではFRAME_COUNTが不正値になり得る判断

## 主要API
`cv2.VideoCapture` / `cap.read` / `cap.release` / `cap.get` / `cv2.CAP_PROP_FPS` / `cv2.VideoWriter` / `cv2.VideoWriter_fourcc` / `time.perf_counter`

## 評価方法
—

## 完成物
動画を1フレームずつ読み、Nフレーム間引きで縮小画像を連番保存しつつ処理FPSを表示し、結果をVideoWriterで再書き出しするスクリプト。

## CPU / GPU メモ
完全CPU(ソフトウェアデコード)。Docker/サーバはopencv-python-headless+ファイル/動画保存で確認。ffmpegはDockerイメージに同梱、macはbrew install ffmpeg。

## 予定スクリプト
- `01_videocapture_loop.py`
- `02_capprops_seek.py`
- `03_videowriter.py`

---
> ⚠️ この回はロードマップ上の**プレースホルダ**です。教材本体（解説＋実行コード＋演習）は順次作成します。
