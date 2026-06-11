# 27_video_action_recognition: 動画理解・行動認識 — VideoMAE / r3d_18

> トラック: **動画・追跡** ／ レベル: **上級** ／ 必要な依存グループ: `dl` `hf` `video`

## 🎯 この章のゴール
クリップ(複数フレーム)から行動クラスを推定する仕組みを理解し、transformers VideoMAE(kinetics finetuned)かtorchvision r3d_18を使ってフレームサンプリング(clip_len/frame_rate)と専用正規化を正しく行い、CPUで小モデル+短クリップの推論を成立させ、top-1/top-5で評価できる。

## 扱うトピック
- クリップ単位の行動認識とVideoMAEForVideoClassification
- VideoMAEImageProcessorのフレームサンプリング(clip_len/frame_rate)と専用正規化
- torchvision r3d_18(R3D_18_Weights、Kinetics-400)
- pipeline('video-classification')
- cv2.VideoCaptureでのフレーム抽出(torchvision内蔵デコーダ廃止のため)
- 評価: clip-levelのtop-1/top-5 accuracy

## 主要API
`VideoMAEForVideoClassification` / `VideoMAEImageProcessor` / `torchvision.models.video.r3d_18` / `R3D_18_Weights.DEFAULT` / `pipeline('video-classification')` / `cv2.VideoCapture`

## 評価方法
行動認識の正解率をclip-levelのtop-1/top-5 accuracyで評価し、小検証セットで混同行列も併記する。前処理(clip_len/frame_rate/正規化)を変えるとスコアが壊れることを実験で示し、正しいサンプリングの重要性を定量化する。

## 完成物
動画クリップから行動クラスを推定し、フレームサンプリングを変えながらtop-1/top-5を出すスクリプト。

## CPU / GPU メモ
CPUは小モデル+短クリップ(clip_len小)+低解像度に限定。動画読込はcv2.VideoCapture(torchvision 0.26で内蔵デコーダ廃止)。専用正規化を誤ると無意味な出力。

## 予定スクリプト
- `01_videomae_action.py`
- `02_r3d18_action.py`
- `03_action_topk_eval.py`

---
> ⚠️ この回はロードマップ上の**プレースホルダ**です。教材本体（解説＋実行コード＋演習）は順次作成します。

> 依存追加の例: `uv add --group dl <packages>`（必要グループ: `dl` `hf` `video`）
