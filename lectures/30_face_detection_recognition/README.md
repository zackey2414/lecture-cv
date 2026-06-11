# 30_face_detection_recognition: 顔検出・顔認識・人物クラスタリング

> トラック: **深度・姿勢・動き** ／ レベル: **中級** ／ 必要な依存グループ: `dl` `hf` `metrics`
> 前提モジュール: `15_image_embeddings_metric_learning`

## 🎯 この章のゴール
顔の 検出→整列→埋め込み(認識) を理解し、1:1 照合(verification)・1:N 識別、さらに顔埋め込みの**クラスタリングによる人物の自動グルーピング（人物一致）**を実装できる。評価は TAR@FAR/ROC/EER（認識）と purity/NMI/homogeneity（クラスタリング）。

## 扱うトピック
- 顔検出（OpenCV Haar / DNN）
- 顔の整列(alignment)と埋め込み（ArcFace系の考え方）
- 1:1 照合(verification)と閾値・1:N 識別
- 顔埋め込みのクラスタリング（DBSCAN/agglomerative）で人物ごとに自動グルーピング
- 評価: TAR@FAR・ROC・EER（認識）/ purity・NMI・homogeneity（クラスタリング）
- プライバシ・バイアスの注意

## 主要API
`cv2.CascadeClassifier` / `cv2.dnn.readNet` / `sklearn.cluster.DBSCAN` / `sklearn.cluster.AgglomerativeClustering` / `sklearn.metrics.normalized_mutual_info_score`

## 評価方法
認識: TAR@FAR / ROC / EER。クラスタリング: purity / NMI / homogeneity。

## 完成物
顔写真群を 検出→埋め込み→クラスタリング して『同一人物ごとのアルバム』へ自動仕分けし、認識精度とクラスタ品質を測るパイプライン

## CPU / GPU メモ
全てCPU。顔検出はOpenCV、埋め込みは軽量モデル。insightface/mediapipe は任意・ガード。

## 予定スクリプト
- `01_face_detection.py`
- `02_face_embeddings_verification.py`
- `03_face_clustering.py`
- `04_recognition_cluster_eval.py`
- `mini_project.py`

---
> ⚠️ この回はロードマップ上の**プレースホルダ**です。教材本体（解説＋実行コード＋演習）は順次作成します。
