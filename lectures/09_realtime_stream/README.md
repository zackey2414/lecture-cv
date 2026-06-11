# 09_realtime_stream: リアルタイム・ストリーム処理 — 背景差分・最適化・スレッド/プロセス分離・RTSP/YouTube

> トラック: **動画・ストリーム** ／ レベル: **中級** ／ 必要な依存グループ: `video`

## 🎯 この章のゴール
CPUのみで実時間に追いつかせる定石(縮小・フレームスキップ・grab/retrieve)、producer/consumerのスレッド/プロセス分離とキュー満杯時のフレームドロップ、背景差分による動体検出、RTSP/yt-dlpライブURL接続と再接続を実装し、640x480程度でCPUリアルタイム処理を成立させる。

## 扱うトピック
- 背景差分(MOG2/KNN)+モルフォロジーによる動体検出
- CPU実時間化(早期resize INTER_AREA・フレームスキップ・grab/retrieve)
- threadingのproducer/consumerとqueue.Queue(maxsize=1)+put_nowaitドロップ
- multiprocessingでCPUバウンド推論を分離(GIL回避)
- RTSP低遅延(CAP_FFMPEG・rtsp_transport;tcp・BUFFERSIZE=1)と再接続ループ
- yt-dlpでライブ配信URLを解決しVideoCaptureへ

## 主要API
`cv2.createBackgroundSubtractorMOG2` / `cap.grab` / `cap.retrieve` / `threading.Thread` / `queue.Queue` / `put_nowait` / `multiprocessing.Process` / `cv2.CAP_FFMPEG` / `cv2.CAP_PROP_BUFFERSIZE`

## 評価方法
パイプラインのリアルタイム性能を評価する: 各ステージの処理レイテンシ(p50/p99, time.perf_counter)・スループット(処理FPS, EMA)・フレームドロップ率を計測し、縮小/スキップ/スレッド分離の前後で数値比較して律速段を特定する。

## 完成物
Webカメラ/動画/YouTube URLを入力に背景差分で動体を検出し、スレッド分離+フレームドロップでリアルタイム動作するストリームアプリと性能プロファイル出力。

## CPU / GPU メモ
完全CPU。解像度低減とフレームドロップで実時間化する設計。yt-dlpはサイト変更で壊れやすく要更新。DockerでWebカメラは--device=/dev/video0、表示はheadless。

## 予定スクリプト
- `01_background_subtraction.py`
- `02_frameskip_grab_retrieve.py`
- `03_threaded_capture.py`
- `04_rtsp_youtube_stream.py`

---
> ⚠️ この回はロードマップ上の**プレースホルダ**です。教材本体（解説＋実行コード＋演習）は順次作成します。

> 依存追加の例: `uv add --group video <packages>`（必要グループ: `video`）
