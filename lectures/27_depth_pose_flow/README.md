# 25_depth_pose_flow: 単眼深度・姿勢/キーポイント・オプティカルフロー

> トラック: **深度・姿勢・動き** ／ レベル: **中級** ／ 必要な依存グループ: `dl` `hf` `pose` `metrics`

## 🎯 この章のゴール
Depth Anything V2(depth-estimation pipeline)で単眼相対深度、MediaPipe Pose Landmarkerで人体33点キーポイント、torchvision RAFT(small)で深層密オプティカルフローを推定でき、相対深度の正規化・[-1,1]正規化と8の倍数サイズ・反復出力の最後採用といった前処理の勘所と、深度/姿勢/フローの評価指標を理解する。

## 扱うトピック
- Depth Anything V2(Small)のpipeline('depth-estimation')、出力は相対(逆)深度で正規化が必要
- MediaPipe Pose Landmarker(IMAGE/VIDEO/LIVE_STREAMモード、XNNPACK CPU)
- torchvision RAFT(raft_small、[-1,1]正規化・8の倍数・flow_to_image)
- 古典オプティカルフロー(Lucas-Kanade/Farneback)との対比と輝度一定仮定
- 評価: 深度AbsRel/RMSE/δ<1.25、フローEPE、姿勢OKS/PCK
- CPU負荷対策(small/低解像度/inference_mode)

## 主要API
`transformers.pipeline('depth-estimation')` / `AutoModelForDepthEstimation` / `mediapipe PoseLandmarker` / `torchvision.models.optical_flow.raft_small` / `torchvision.utils.flow_to_image` / `cv2.calcOpticalFlowPyrLK` / `cv2.calcOpticalFlowFarneback`

## 評価方法
深度はGTがある場合AbsRel=mean(|d-d*|/d*)・RMSE・δ<1.25(threshold accuracy)を算出(相対深度はスケール合わせ後)。フローはEPE=予測と真の流れの平均端点誤差。姿勢はOKS(物体スケールとper-keypoint定数で正規化)とPCK@0.2を計算する。

## 完成物
1枚画像→深度マップ可視化、動画→姿勢キーポイント描画、画像対→フロー可視化を行うスクリプト群とAbsRel/EPE/OKSの評価コード。

## CPU / GPU メモ
CPUはDepth-Anything-V2-Small(数秒)・raft_small・低解像度を使う。MediaPipeはXNNPACKでCPU実時間だがnumpy<2/protobuf競合を起こしやすく専用グループで隔離。

## 予定スクリプト
- `01_depth_anything.py`
- `02_mediapipe_pose.py`
- `03_raft_optical_flow.py`
- `04_depth_flow_pose_metrics.py`

---
> ⚠️ この回はロードマップ上の**プレースホルダ**です。教材本体（解説＋実行コード＋演習）は順次作成します。

> 依存追加の例: `uv add --group dl <packages>`（必要グループ: `dl` `hf` `pose` `metrics`）
