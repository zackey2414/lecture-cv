# 28_face_detection_recognition: 顔検出と顔認識 — Haar→DNN/MediaPipe・insightface ArcFace

> トラック: **深度・姿勢・動き** ／ レベル: **中級** ／ 必要な依存グループ: `face` `pose` `metrics`

## 🎯 この章のゴール
古典Haar/LBPカスケードで顔検出の原理を学び、OpenCV DNN(res10 SSD)やMediaPipe Face Detectorで頑健化、検出済み顔からinsightface ArcFaceの512次元埋め込みを得てコサイン類似度で1:1照合・1:N検索を実装でき、TAR@FARで認証性能を評価できる。

## 扱うトピック
- Haar Cascade(detectMultiScale)→OpenCV DNN(res10 SSD)→MediaPipe Face Detectorの段階移行
- insightface FaceAnalysis(buffalo_l、ctx_id=-1でCPU、onnxruntime)
- face.normed_embeddingとコサイン類似度による1:1照合/1:N検索
- 閾値設計と初回モデル自動DL
- 評価: 検出のprecision/recall、認識のROC/TAR@FAR
- 顔の位置合わせ(landmark)の役割

## 主要API
`cv2.CascadeClassifier` / `detectMultiScale` / `cv2.dnn.readNetFromCaffe` / `mediapipe FaceDetector` / `insightface.app.FaceAnalysis` / `prepare(ctx_id=-1)` / `face.normed_embedding`

## 評価方法
顔認識(認証)を、同一/別人ペアのコサイン類似度分布からROC曲線を描き、誤受入率FARを固定したときの正受入率TAR@FAR(例FAR=1e-3)で評価する。閾値はFAR固定で決定。顔検出はGTとのIoUマッチでprecision/recallを算出する。

## 完成物
画像中の顔を検出してArcFace埋め込みを抽出し、本人判定(1:1)と顔検索(1:N)を行うスクリプトと、TAR@FAR/ROCを出す評価コード。

## CPU / GPU メモ
CPUはinsightfaceをctx_id=-1/providers=['CPUExecutionProvider']で実行、onnxruntime-gpuは入れない。insightface 1.0でC++ビルド不要。MediaPipeは専用グループで隔離。

## 予定スクリプト
- `01_haar_dnn_face_detect.py`
- `02_insightface_arcface.py`
- `03_tar_far_eval.py`

---
> ⚠️ この回はロードマップ上の**プレースホルダ**です。教材本体（解説＋実行コード＋演習）は順次作成します。

> 依存追加の例: `uv add --group face <packages>`（必要グループ: `face` `pose` `metrics`）
