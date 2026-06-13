# lecture-cv カリキュラム・ロードマップ

> Computer Vision を「AI の補助なしで自力で書ける」レベルまで叩き込む、CPUのみで完走するハンズオン講座の全体地図。

## 講座の全体ゴール

Python既習者が、AIの補助なしに自力でCV関連コードを書き切れる状態を目標とする総合ハンズオン講座。OpenCV/Pillowの画像表現(BGR/RGB・ndarray)と前処理から始め、古典CV(特徴点マッチング→ホモグラフィ→パノラマ、カメラキャリブレーション/ステレオ、watershed/GrabCut/背景差分)、深層CV(ResNet/ViTの分類と転移学習、物体検出、セグメンテーション、単眼深度・姿勢・オプティカルフロー、物体追跡・行動認識、顔検出/認識)、CLIP/SigLIPによるゼロショット分類・画像テキスト検索とFAISSベクトルDB、画像キャプション/VQA/OCR/文書理解、拡散モデルによる生成・編集、異常検知・画像品質、マルチモーダル埋め込み(SigLIP2/ImageBind)までを実装する。さらに各タスクで評価指標を必ず測り(物体検出mAPはnumpyで自力実装しpycocotoolsと突き合わせ)、推論プロファイリング→量子化→枝刈り→ONNX→ランタイム最適化と知識蒸留(CLIP/VLM蒸留含む)のモデル圧縮トラックを通過し、最終的にCluster-CLIP(dense CLIP特徴+空間連結クラスタリング+FAISS+SQLite+multiprocessingストリーム)をCPUで動く小型版として自力再構築できる到達度を目指す。全モジュールがGPU無し(MacBook等CPUのみ)で完走できるよう設計する。

## レベルの考え方（入門→初級→中級→上級）

入門=各ライブラリの正準APIとデータ表現(BGR/RGB・テンソル形状CHW/HWC・torch.deviceのCPU/MPS/CUDAフォールバック)を把握し、pipeline等の高レベルAPIで最短コードを動かして成功体験を得る段階。初級=pipelineの中身を分解し、前処理(processor/transforms)→推論(eval+inference_mode)→可視化→評価までを自分の手で書ける基礎アプリ実装段階。中級=複数モデルからCPUで現実的なものを選定し、タスク固有の評価指標(mIoU/Dice/Recall@k/CER/AbsRel等)を計算し、解像度低減・フレームスキップ・スレッド/プロセス分離・量子化などCPU最適化を調べながら実務タスクを完遂できる段階。上級=パイプライン統合・モデル圧縮(蒸留/量子化/ONNX)・dense特徴クラスタリング・ストリーム並列処理など、ライブラリ構成を理解した応用設計とCluster-CLIP再構築ができる段階。並びは依存関係順(画像基礎→古典CV→動画/ストリーム→分類/評価→埋め込み/CLIP/FAISS→検出/セグメ/キャプション等の各タスク→生成/異常/マルチモーダル→最適化/蒸留→Cluster-CLIP応用)で、要求された検出・セグメ・キャプションは独立した入門モジュールとして軽量モデルで先に置き、評価(mAP自作含む)と最適化トラックで仕上げる。

## 学習順序は「グラフ」で

この講座は番号順の一本道ではなく、**前提でつながった有向グラフ(DAG)**です。番号は安定IDで、学習順は各回の「前提」をたどります。可視化は公開サイトの **学習順序グラフ**（<https://zackey2414.github.io/lecture-cv/graph.html>）を参照。

## モジュール一覧（トラック別・前提つき）

### 環境構築

| レベル | ID | タイトル | 前提 | 依存グループ |
| --- | --- | --- | --- | --- |
| 入門 | `00_setup` | 環境構築 — uv + Docker + CPU版PyTorch + HuggingFaceキャッシュ + device判定 | — | dl |

### 画像の基礎

| レベル | ID | タイトル | 前提 | 依存グループ |
| --- | --- | --- | --- | --- |
| 入門 | `01_image_basics` | 画像の基礎 — ndarray表現・BGR/RGB・OpenCV/Pillow I/O・headless表示 | 00 | — |
| 入門 | `02_cv_libraries_overview` | 画像・動画処理ライブラリの地図 | 01 | — |
| 初級 | `03_image_transforms` | 色空間・描画・幾何変換 — 前処理パイプラインの土台 | 01 | — |
| 中級 | `04_filtering_edges_morphology` | フィルタ・エッジ・閾値・モルフォロジー・輪郭・ワーピング | 03 | — |
| 初級 | `43_color_spaces_and_adjustments` | 色空間と画像の調整 — 明るさ・彩度・色相・コントラスト・ガンマ・ホワイトバランス | 03 | — |

### 古典CV

| レベル | ID | タイトル | 前提 | 依存グループ |
| --- | --- | --- | --- | --- |
| 初級 | `05_classical_features_matching` | 特徴点検出とマッチング — SIFT/ORB・BFMatcher/FLANN・テンプレート・Hough | 04 | — |
| 初級 | `06_homography_panorama` | ホモグラフィ推定とパノラマ合成 | 05 | — |
| 中級 | `07_camera_calibration_stereo` | カメラキャリブレーション・ステレオ・エピポーラ幾何 | 06 | — |
| 初級 | `08_classical_segmentation` | 古典セグメンテーションと復元 — Watershed・GrabCut・古典inpaint | 04 | — |

### 動画・ストリーム

| レベル | ID | タイトル | 前提 | 依存グループ |
| --- | --- | --- | --- | --- |
| 入門 | `09_video_io_basics` | 動画I/Oの基礎 — VideoCapture/VideoWriter・メタデータ・FPS | 01 | — |
| 中級 | `10_classical_video_motion` | 古典的な動画処理 — オプティカルフロー・背景差分 | 09, 05 | — |
| 中級 | `11_realtime_stream` | リアルタイム・ストリーム処理 — 背景差分・最適化・スレッド/プロセス分離・RTSP/YouTube | 09, 10 | video |

### 深層CV(分類)

| レベル | ID | タイトル | 前提 | 依存グループ |
| --- | --- | --- | --- | --- |
| 初級 | `12_data_pipeline_augmentation` | PyTorch画像テンソルとデータ拡張 — transforms v2 / albumentations / DataLoader | 03 | dl, aug |
| 中級 | `13_classification_transfer_learning` | 画像分類と転移学習 — ResNet/ViT(torchvision/timm/HuggingFace) | 12 | dl, hf |

### 評価指標

| レベル | ID | タイトル | 前提 | 依存グループ |
| --- | --- | --- | --- | --- |
| 初級 | `14_eval_classification` | 評価指標の基礎(A) — 混同行列・precision/recall/F1・ROC/PR・AUC | 13 | dl, metrics |
| 中級 | `19_detection_map_from_scratch` | ★物体検出mAPの自力実装 — IoU→マッチング→PR曲線→AP補間→mAP | 18 | dl, metrics |

### 埋め込み・検索

| レベル | ID | タイトル | 前提 | 依存グループ |
| --- | --- | --- | --- | --- |
| 中級 | `15_image_embeddings_metric_learning` | 画像埋め込みとメトリック学習 — ViT/ResNet特徴・対照/triplet学習 | 13 | dl, hf |
| 中級 | `17_faiss_image_search` | FAISSベクトルDBと画像検索システム(評価込み) | 15, 16 | dl, hf, vector, metrics |
| 中級 | `42_multimodal_vector_search` | マルチモーダル・ベクトル検索（FAISS）— 画像・テキスト・クロスモーダル | 16, 17 | dl, hf, vector, metrics |
| 中級 | `44_embedding_clustering` | 埋め込みのクラスタリング — 画像・テキスト・クロスモーダルを教師なしで束ねる | 16, 17 | dl, hf, metrics, vector |
| 中級 | `45_sketch_emoji_search` | 実践: 手書きスケッチで絵文字を検索（CLIP＋FAISS のスケッチ画像検索 SBIR） | 16, 17 | dl, hf, vector, metrics |

### マルチモーダル

| レベル | ID | タイトル | 前提 | 依存グループ |
| --- | --- | --- | --- | --- |
| 中級 | `16_clip_zeroshot_retrieval` | CLIP/SigLIPによるゼロショット分類と画像テキスト検索 | 15 | dl, hf, embed |
| 入門 | `24_image_captioning` | 画像キャプション生成 入門 — BLIP/GIT/ViT-GPT2と生成パラメータ・評価 | 16 | dl, hf, metrics |
| 中級 | `25_vqa_vlm` | VQAと軽量VLMによる画像理解・グラウンディング | 24 | dl, hf |
| 中級 | `26_ocr_document` | OCRと文書理解 — Tesseract/EasyOCR/TrOCR・Donut/LayoutLM・CER/WER | 13 | dl, hf, ocr, metrics |
| 上級 | `33_multimodal_embeddings` | マルチモーダル埋め込みの拡張 — SigLIP2多言語・ImageBind(音声+画像+テキスト) | 16 | dl, hf, embed, vector |

### 検出

| レベル | ID | タイトル | 前提 | 依存グループ |
| --- | --- | --- | --- | --- |
| 入門 | `18_object_detection_intro` | 物体検出 入門 — torchvision weights API・YOLO・DETR/RT-DETR | 13, 14 | dl, hf, detect |
| 中級 | `20_open_vocabulary_detection` | オープン語彙物体検出 — OWL-ViT/OWLv2・Grounding DINO | 18, 16 | dl, hf |

### セグメンテーション

| レベル | ID | タイトル | 前提 | 依存グループ |
| --- | --- | --- | --- | --- |
| 入門 | `21_segmentation_intro` | セマンティックセグメンテーション 入門 — DeepLab/FCN/LR-ASPP・SegFormer・mIoU/Dice | 13, 14 | dl, hf, metrics |
| 中級 | `22_instance_panoptic_sam` | インスタンス/パノプティックセグメンテーションとSAM — mask AP・PQ | 21, 18 | dl, hf, detect, metrics |
| 中級 | `23_text_prompt_segmentation` | テキストプロンプト/参照セグメンテーション — CLIPSeg・Grounded-SAM | 21, 16 | dl, hf, detect |

### 深度・姿勢・動き

| レベル | ID | タイトル | 前提 | 依存グループ |
| --- | --- | --- | --- | --- |
| 中級 | `27_depth_pose_flow` | 単眼深度・姿勢/キーポイント・オプティカルフロー | 13, 10 | dl, hf, pose, metrics |
| 中級 | `30_face_detection_recognition` | 顔検出・顔認識・人物クラスタリング | 15 | dl, hf, metrics |

### 動画・追跡

| レベル | ID | タイトル | 前提 | 依存グループ |
| --- | --- | --- | --- | --- |
| 中級 | `28_tracking` | 物体追跡 — OpenCV CSRT/KCF・ByteTrack・DeepSORTとMOT評価 | 18, 10 | dl, detect, track, metrics |
| 上級 | `29_video_action_recognition` | 動画理解・行動認識 — VideoMAE / r3d_18 | 13, 11 | dl, hf, video |

### 生成・編集

| レベル | ID | タイトル | 前提 | 依存グループ |
| --- | --- | --- | --- | --- |
| 上級 | `31_generation_editing` | 画像生成・編集 — 拡散モデル text-to-image・インペイント・超解像・背景除去 | 13 | dl, hf, diffusion, classical, metrics |

### 異常検知・品質

| レベル | ID | タイトル | 前提 | 依存グループ |
| --- | --- | --- | --- | --- |
| 上級 | `32_anomaly_iqa` | 異常検知と画像品質評価 — anomalib(PaDiM/PatchCore)・pyiqa | 15 | dl, anomaly, metrics |

### 最適化・デプロイ

| レベル | ID | タイトル | 前提 | 依存グループ |
| --- | --- | --- | --- | --- |
| 初級 | `34_inference_profiling` | 推論高速化の地図 — 計測ファースト・プロファイリング・autocast・torch.compile | 13 | dl |
| 中級 | `35_quantization_pruning` | 量子化と枝刈り — PTQ(動的/静的)・QAT・torchao・pruningの実効速度の罠 | 34 | dl, quant |
| 中級 | `36_onnx_runtime` | ONNXエクスポートとonnxruntime — グラフ最適化・動的量子化 | 34 | dl, onnx |
| 上級 | `37_runtime_edge_optimization` | ランタイム/エッジ最適化 — OpenVINO・CoreML・LiteRT・TensorRT概要 | 36, 35 | dl, onnx |
| 中級 | `38_knowledge_distillation` | 知識蒸留の基礎(a) — 温度付きKD・特徴量蒸留(CPUトイ学習) | 13, 34 | dl, distill |
| 上級 | `39_clip_distillation` | VLM/CLIPの蒸留(b) — TinyCLIP/MobileCLIP・埋め込み模倣 | 38, 16 | dl, hf, embed, distill |

### 応用(Cluster-CLIP)

| レベル | ID | タイトル | 前提 | 依存グループ |
| --- | --- | --- | --- | --- |
| 上級 | `40_cluster_clip_dense_cluster` | Cluster-CLIP中核 — dense CLIP特徴と空間連結クラスタリング | 16, 33 | dl, hf, embed, vector, metrics |
| 上級 | `41_cluster_clip_pipeline` | Cluster-CLIPパイプライン統合(総合) — Split→Build→Search→Stream | 40, 17, 42, 11 | dl, hf, embed, vector, video |


---

_lecture-cv ロードマップ ／ 設計時点: 2026-06。学習順は学習順序グラフ（公開サイトの graph.html）が正。_
