# lecture-cv カリキュラム・ロードマップ

> Computer Vision を「AI の補助なしで自力で書ける」レベルまで叩き込むための、CPU のみ（MacBook 等 GPU 無し）で完走できるハンズオン講座の全体地図です。

## 講座の全体ゴール

Python既習者が、AIの補助なしに自力でCV関連コードを書き切れる状態を目標とする総合ハンズオン講座。OpenCV/Pillowの画像表現(BGR/RGB・ndarray)と前処理から始め、古典CV(特徴点マッチング→ホモグラフィ→パノラマ、カメラキャリブレーション/ステレオ、watershed/GrabCut/背景差分)、深層CV(ResNet/ViTの分類と転移学習、物体検出、セグメンテーション、単眼深度・姿勢・オプティカルフロー、物体追跡・行動認識、顔検出/認識)、CLIP/SigLIPによるゼロショット分類・画像テキスト検索とFAISSベクトルDB、画像キャプション/VQA/OCR/文書理解、拡散モデルによる生成・編集、異常検知・画像品質、マルチモーダル埋め込み(SigLIP2/ImageBind)までを実装する。さらに各タスクで評価指標を必ず測り(物体検出mAPはnumpyで自力実装しpycocotoolsと突き合わせ)、推論プロファイリング→量子化→枝刈り→ONNX→ランタイム最適化と知識蒸留(CLIP/VLM蒸留含む)のモデル圧縮トラックを通過し、最終的に参照リポジトリ Cluster-CLIP(dense CLIP特徴+空間連結クラスタリング+FAISS+SQLite+multiprocessingストリーム)をCPUで動く小型版として自力再構築できる到達度を目指す。全モジュールがGPU無し(MacBook等CPUのみ)で完走できるよう設計する。

## レベルの考え方（入門→初級→中級→上級）

入門=各ライブラリの正準APIとデータ表現(BGR/RGB・テンソル形状CHW/HWC・torch.deviceのCPU/MPS/CUDAフォールバック)を把握し、pipeline等の高レベルAPIで最短コードを動かして成功体験を得る段階。初級=pipelineの中身を分解し、前処理(processor/transforms)→推論(eval+inference_mode)→可視化→評価までを自分の手で書ける基礎アプリ実装段階。中級=複数モデルからCPUで現実的なものを選定し、タスク固有の評価指標(mIoU/Dice/Recall@k/CER/AbsRel等)を計算し、解像度低減・フレームスキップ・スレッド/プロセス分離・量子化などCPU最適化を調べながら実務タスクを完遂できる段階。上級=パイプライン統合・モデル圧縮(蒸留/量子化/ONNX)・dense特徴クラスタリング・ストリーム並列処理など、ライブラリ構成を理解した応用設計とCluster-CLIP再構築ができる段階。並びは依存関係順(画像基礎→古典CV→動画/ストリーム→分類/評価→埋め込み/CLIP/FAISS→検出/セグメ/キャプション等の各タスク→生成/異常/マルチモーダル→最適化/蒸留→Cluster-CLIP応用)で、要求された検出・セグメ・キャプションは独立した入門モジュールとして軽量モデルで先に置き、評価(mAP自作含む)と最適化トラックで仕上げる。

## 前提知識

本講座は **Python 既習者**を対象とします。深層 CV のトラック（分類以降）では PyTorch の基礎（テンソル / autograd / `nn.Module` / optimizer / 学習・評価ループ）を前提とします。未習なら、別講座 [`../../courses/pytorch`](../../courses) で先に基礎を押さえることを推奨します。

## 進め方（環境）

```bash
# 1) ローカル（uv）
uv sync                 # 画像基礎(00〜11)に必要な numpy/opencv/pillow/matplotlib が入る
uv run python lectures/01_image_basics/01_imread_imwrite.py

# 深層・各タスクのトラックに進むときに、必要なグループだけ足す
uv sync --group dl --group hf            # PyTorch + HuggingFace
uv sync --group vector --group metrics   # FAISS + 評価指標

# 2) Docker（CPU 既定。GPU は docker-compose.yaml のコメント参照）
docker compose up -d --build
docker compose exec lecture-cv uv run python lectures/01_image_basics/01_imread_imwrite.py
```

各モジュールの `README.md` 冒頭に必要な依存グループを明記しています。重い／衝突しやすい依存（後述）は、その回に到達してから `uv add --group <name>` で足す方針です。

## モジュール一覧（実行順・トラック別）

### 環境構築

| レベル | ID | タイトル | 依存グループ | 評価 |
| --- | --- | --- | --- | --- |
| 入門 | `00_setup` | 環境構築 — uv + Docker + CPU版PyTorch + HuggingFaceキャッシュ + device判定 | dl | — |

### 画像の基礎

| レベル | ID | タイトル | 依存グループ | 評価 |
| --- | --- | --- | --- | --- |
| 入門 | `01_image_basics` | 画像の基礎 — ndarray表現・BGR/RGB・OpenCV/Pillow I/O・headless表示 | — | — |
| 入門 | `02_cv_libraries_overview` | 画像・動画処理ライブラリの地図 — OpenCV/Pillow/scikit-image/albumentations/kornia ほか | — | 概念回（数値評価は無し） |
| 初級 | `03_image_transforms` | 色空間・描画・幾何変換 — 前処理パイプラインの土台 | — | — |
| 中級 | `04_filtering_edges_morphology` | フィルタ・エッジ・閾値・モルフォロジー・輪郭・ワーピング | — | — |

### 古典CV

| レベル | ID | タイトル | 依存グループ | 評価 |
| --- | --- | --- | --- | --- |
| 初級 | `05_classical_features_matching` | 特徴点検出とマッチング — SIFT/ORB・BFMatcher/FLANN・テンプレート・Hough | — | マッチング品質を定量化する: 比率テスト後の良マッチ数と、後続のRA |
| 初級 | `06_homography_panorama` | ホモグラフィ推定とパノラマ合成 | — | 推定ホモグラフィの品質を、対応点を変換した後の再投影誤差(平均ユーク |
| 中級 | `07_camera_calibration_stereo` | カメラキャリブレーション・ステレオ・エピポーラ幾何 | — | calibrateCameraの戻り値であるRMS再投影誤差(画素単 |
| 初級 | `08_classical_segmentation` | 古典セグメンテーションと復元 — Watershed・GrabCut・古典inpaint | — | 前景抽出の精度を、GrabCut/Watershed結果と手動アノテ |

### 動画・ストリーム

| レベル | ID | タイトル | 依存グループ | 評価 |
| --- | --- | --- | --- | --- |
| 入門 | `09_video_io_basics` | 動画I/Oの基礎 — VideoCapture/VideoWriter・メタデータ・FPS | — | — |
| 中級 | `10_classical_video_motion` | 古典的な動画処理 — オプティカルフロー・背景差分・動き解析 | — | 定性評価（フロー場・前景マスク・追跡窓の可視化が妥当か） |
| 中級 | `11_realtime_stream` | リアルタイム・ストリーム処理 — 背景差分・最適化・スレッド/プロセス分離・RTSP/YouTube | video | パイプラインのリアルタイム性能を評価する: 各ステージの処理レイテン |

### 深層CV(分類)

| レベル | ID | タイトル | 依存グループ | 評価 |
| --- | --- | --- | --- | --- |
| 初級 | `12_data_pipeline_augmentation` | PyTorch画像テンソルとデータ拡張 — transforms v2 / albumentations / DataLoader | dl, aug | — |
| 中級 | `13_classification_transfer_learning` | 画像分類と転移学習 — ResNet/ViT(torchvision/timm/HuggingFace) | dl, hf | 小分類データセット(CIFAR-10部分集合等)でtop-1/top |

### 評価指標

| レベル | ID | タイトル | 依存グループ | 評価 |
| --- | --- | --- | --- | --- |
| 初級 | `14_eval_classification` | 評価指標の基礎(A) — 混同行列・precision/recall/F1・ROC/PR・AUC | dl, metrics | 本モジュール自体が評価指標の実装回 |
| 中級 | `19_detection_map_from_scratch` | ★物体検出mAPの自力実装 — IoU→マッチング→PR曲線→AP補間→mAP | dl, metrics | 本モジュール自体が評価指標の自作実装回 |

### 埋め込み・検索

| レベル | ID | タイトル | 依存グループ | 評価 |
| --- | --- | --- | --- | --- |
| 中級 | `15_image_embeddings_metric_learning` | 画像埋め込みとメトリック学習 — ViT/ResNet特徴・対照/triplet学習 | dl, hf | 埋め込み品質を、抽出ベクトルでのkNN分類精度(accuracy)と |
| 中級 | `17_faiss_image_search` | FAISSベクトルDBと画像検索システム(評価込み) | dl, hf, vector, metrics | ANN品質を評価する: IndexFlat(厳密)の結果をgroun |

### マルチモーダル

| レベル | ID | タイトル | 依存グループ | 評価 |
| --- | --- | --- | --- | --- |
| 中級 | `16_clip_zeroshot_retrieval` | CLIP/SigLIPによるゼロショット分類と画像テキスト検索 | dl, hf, embed | ゼロショット分類はラベル付き小データでtop-1 accuracyを |
| 入門 | `24_image_captioning` | 画像キャプション生成 入門 — BLIP/GIT/ViT-GPT2と生成パラメータ・評価 | dl, hf, metrics | 生成キャプションを参照キャプションと比較しBLEU(sacreble |
| 中級 | `25_vqa_vlm` | VQAと軽量VLMによる画像理解・グラウンディング | dl, hf | VQAの正答を、VQA v2方式のaccuracy=min(一致した |
| 中級 | `26_ocr_document` | OCRと文書理解 — Tesseract/EasyOCR/TrOCR・Donut/LayoutLM・CER/WER | dl, hf, ocr, metrics | 認識文字列と正解の編集距離からCER=(置換+削除+挿入)/参照文字 |
| 上級 | `33_multimodal_embeddings` | マルチモーダル埋め込みの拡張 — SigLIP2多言語・ImageBind(音声+画像+テキスト) | dl, hf, embed, vector | クロスモーダル検索をRecall@k(例: 音声クエリで対応画像が上 |

### 検出

| レベル | ID | タイトル | 依存グループ | 評価 |
| --- | --- | --- | --- | --- |
| 入門 | `18_object_detection_intro` | 物体検出 入門 — torchvision weights API・YOLO・DETR/RT-DETR | dl, hf, detect | まずscore閾値・NMS後の検出を可視化で定性確認し、定量はtor |
| 中級 | `20_open_vocabulary_detection` | オープン語彙物体検出 — OWL-ViT/OWLv2・Grounding DINO | dl, hf | 任意ラベル検出を、GTがある画像でprecision/recall( |

### セグメンテーション

| レベル | ID | タイトル | 依存グループ | 評価 |
| --- | --- | --- | --- | --- |
| 入門 | `21_segmentation_intro` | セマンティックセグメンテーション 入門 — DeepLab/FCN/LR-ASPP・SegFormer・mIoU/Dice | dl, hf, metrics | セグメンテーション精度を、予測マスクとGTマスクのピクセル混同行列か |
| 中級 | `22_instance_panoptic_sam` | インスタンス/パノプティックセグメンテーションとSAM — mask AP・PQ | dl, hf, detect, metrics | インスタンスはmask IoUベースのmask AP(COCOeva |
| 中級 | `23_text_prompt_segmentation` | テキストプロンプト/参照セグメンテーション — CLIPSeg・Grounded-SAM | dl, hf, detect | 参照テキストで指定した領域の予測マスクとGTマスクのIoU/Dice |

### 深度・姿勢・動き

| レベル | ID | タイトル | 依存グループ | 評価 |
| --- | --- | --- | --- | --- |
| 中級 | `27_depth_pose_flow` | 単眼深度・姿勢/キーポイント・オプティカルフロー | dl, hf, pose, metrics | 深度はGTがある場合AbsRel=mean(|d-d*|/d*)・R |
| 中級 | `30_face_detection_recognition` | 顔検出と顔認識 — Haar→DNN/MediaPipe・insightface ArcFace | face, pose, metrics | 顔認識(認証)を、同一/別人ペアのコサイン類似度分布からROC曲線を |

### 動画・追跡

| レベル | ID | タイトル | 依存グループ | 評価 |
| --- | --- | --- | --- | --- |
| 中級 | `28_tracking` | 物体追跡 — OpenCV CSRT/KCF・ByteTrack・DeepSORTとMOT評価 | dl, detect, track, metrics | 多物体追跡をmotmetricsでMOTA(検出誤りとID切替を統合 |
| 上級 | `29_video_action_recognition` | 動画理解・行動認識 — VideoMAE / r3d_18 | dl, hf, video | 行動認識の正解率をclip-levelのtop-1/top-5 ac |

### 生成・編集

| レベル | ID | タイトル | 依存グループ | 評価 |
| --- | --- | --- | --- | --- |
| 上級 | `31_generation_editing` | 画像生成・編集 — 拡散モデル text-to-image・インペイント・超解像・背景除去 | dl, hf, diffusion, classical, metrics | 超解像/インペイントなど参照ありタスクはPSNR=10log10(M |

### 異常検知・品質

| レベル | ID | タイトル | 依存グループ | 評価 |
| --- | --- | --- | --- | --- |
| 上級 | `32_anomaly_iqa` | 異常検知と画像品質評価 — anomalib(PaDiM/PatchCore)・pyiqa | dl, anomaly, metrics | 異常検知をimage-levelとpixel-levelのAUROC |

### 最適化・デプロイ

| レベル | ID | タイトル | 依存グループ | 評価 |
| --- | --- | --- | --- | --- |
| 初級 | `34_inference_profiling` | 推論高速化の地図 — 計測ファースト・プロファイリング・autocast・torch.compile | dl | 本モジュール自体が評価(ベンチ)回 |
| 中級 | `35_quantization_pruning` | 量子化と枝刈り — PTQ(動的/静的)・QAT・torchao・pruningの実効速度の罠 | dl, quant | 圧縮効果を三角関係で評価する: 量子化/pruning前後でaccu |
| 中級 | `36_onnx_runtime` | ONNXエクスポートとonnxruntime — グラフ最適化・動的量子化 | dl, onnx | ONNX化の正しさをtorch出力とonnxruntime出力の数値 |
| 上級 | `37_runtime_edge_optimization` | ランタイム/エッジ最適化 — OpenVINO・CoreML・LiteRT・TensorRT概要 | dl, onnx | 各ランタイム(eager torch/ONNX/OpenVINO/C |
| 中級 | `38_knowledge_distillation` | 知識蒸留の基礎(a) — 温度付きKD・特徴量蒸留(CPUトイ学習) | dl, distill | 蒸留の効果を、同一studentアーキでの『素の教師あり学習』と『K |
| 上級 | `39_clip_distillation` | VLM/CLIPの蒸留(b) — TinyCLIP/MobileCLIP・埋め込み模倣 | dl, hf, embed, distill | 蒸留した小CLIPの品質を、ラベル付き小データでのzero-shot |

### 応用(Cluster-CLIP)

| レベル | ID | タイトル | 依存グループ | 評価 |
| --- | --- | --- | --- | --- |
| 上級 | `40_cluster_clip_dense_cluster` | Cluster-CLIP中核 — dense CLIP特徴と空間連結クラスタリング | dl, hf, embed, vector, metrics | クラスタリング品質を、各クラスタ代表ベクトルとテキストクエリのコサイ |
| 上級 | `41_cluster_clip_pipeline` | Cluster-CLIPパイプライン統合(総合) — Split→Build→Search→Stream | dl, hf, embed, vector, video | 検索品質を、Cluster-CLIP本体と同じくGT BBoxとクラ |

## 設計メモ・既知の注意点（網羅性検証より）

このロードマップは多段の調査・設計の後、独立した検証エージェントでレビューしています。実装時に必ず留意する点：

- **依存衝突（最重要）**: `mediapipe`(`27`深度姿勢/`30`顔)・`anomalib`(`32`)・`ImageBind`(`33`)・`TrackEval`(`28`追跡) は、本講座の標準 `numpy 2.x` や巨大依存・git pin と衝突しやすく、**単一の uv 環境に同居させると解決に失敗し得ます**。これらの回は `[tool.uv] conflicts` で衝突宣言するか、専用の `pyproject`／venv に隔離し、各回を個別にインストールして動作確認する方針とします。**コア講座（画像基礎〜古典CV〜検出・セグメ・埋め込み・検索・評価・最適化）は単一環境で動きます。**
- **バージョンのピン留め**: `transformers` v5 系は破壊的変更が広い（`AutoImageProcessor` 必須・画像 processor は fast 実装のみ＝`torchvision` 必須・TF/Flax 全廃）。採用版を固定し、各回で実機確認します。`torch`／`torchvision` も整合する単一ペアを確定します（現状 `2.12+cpu` / `0.27+cpu`）。各教材フッターに「どの版時点か」を明記します。
- **FAISS の名称**: `faiss-gpu` という pip パッケージは**存在しません**。CPU は `faiss-cpu`、GPU は `faiss-gpu-cuvs`（Linux + NVIDIA・cpu 版と排他）。本講座は全て `faiss-cpu` で完結します。
- **Cluster-CLIP（`40`）**: dense CLIP 特徴の抽出は **`open_clip` 前提**（HF `CLIPModel` とは属性名が異なる）。`ViT-B-32` の `force_quick_gelu=True`、CenterCrop を排した正方形強制 Resize など、参照リポと同一の前処理を必修とします。
- **過密モジュールの分割**: `27`（深度＋姿勢＋フロー）と `31`（生成＋インペイント＋超解像＋背景除去）は密度が高いため、実装時に分割を検討します。
- **評価の徹底**: 各タスク回は `evaluation` を必ず実測し、特に `19` で**物体検出 mAP を numpy で自力実装**し `pycocotools` と突き合わせて理解を固めます。

---

_lecture-cv ロードマップ ／ 設計時点: 2026-06。ライブラリ版は各回の教材フッターに明記します。_
