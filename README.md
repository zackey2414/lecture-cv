# lecture-cv

Computer Vision（CV）を「**AI の補助なしでも自分一人でゴリゴリ書け、内容を熟知している**」状態まで叩き込むための、ハンズオン講座リポジトリです。OpenCV / Pillow の画像基礎から、古典 CV、深層 CV（分類・検出・セグメンテーション・深度/姿勢/追跡）、CLIP/SigLIP とベクトル検索（FAISS）、キャプション/VQA/OCR、生成・編集、各タスクの**精度評価**（物体検出 mAP は自力実装）、そして量子化・枝刈り・ONNX・知識蒸留などの**モデル圧縮/高速化**まで、全 42 モジュールを段階的に実装します。

最終的には、参照リポジトリ [`cluster-clip`](../../cluster-clip) の Cluster-CLIP（dense CLIP 特徴 + 空間連結クラスタリング + FAISS + ストリーム処理）を、CPU で動く小型版として自力再構築できる到達度を目指します。

## 特徴

- **GPU 不要**: 全モジュールが CPU のみ（MacBook 等）で完走できるよう設計。GPU は任意の高速化です。
- **docker + uv**: 環境は `uv` で管理。Docker でも同一構成を再現できます。
- **ハンズオン**: 各回は「解説（地の文）＋ 実行できる `.py` ＋ 演習」。読むだけでなく**動かして・書き換えて**学びます。
- **評価を重視**: 各タスクで評価指標を必ず実測。`19` では mAP を numpy で一から実装し pycocotools と突き合わせます。
- **ネット非依存**: 第1回はサンプル画像が無くても合成画像で完走します。

## 教材サイト（ブラウザで読む）

各回の解説・コード・演習を、`courses` と同系のデザイン（紫グラデのヒーロー・トラック別カード・コードハイライト）で一覧・通読できる**静的サイト**を `site/` に用意しています。テキストとして読み込み、内容を根から理解するのに使ってください。

```bash
# 生成（Markdown→HTML を事前レンダリング。生成物はオフラインで開けます）
uv run --group site python tools/build_site.py
# 生成後、site/index.html をブラウザで開く（macOS: open site/index.html）
```

教材を追記・更新したら同じコマンドで再生成します。

## クイックスタート

```bash
# --- ローカル（uv） ---
uv sync                                   # 画像基礎〜古典CV〜古典動画(00〜11): numpy/opencv/pillow/matplotlib
uv run python lectures/00_setup/check_env.py          # 環境スモークテスト
uv run python lectures/01_image_basics/01_imread_imwrite.py

# 深層・各タスクのトラックに進むとき、必要なグループだけ足す
uv sync --group dl --group hf             # PyTorch + HuggingFace
uv sync --group vector --group metrics    # FAISS + 評価指標

# --- Docker（CPU 既定。GPU は docker-compose.yaml のコメント参照） ---
docker compose up -d --build
docker compose exec lecture-cv uv run python lectures/01_image_basics/01_imread_imwrite.py
```

> 結果は基本的に `outputs/<モジュール>/` に保存されます（headless 環境でも後から確認できるように）。
> `cv2.imshow` を使いたい場合は、既定の `opencv-python-headless` を `opencv-python`（GUI 版）へ差し替えてください（両者は排他）。

## ディレクトリ構成

```
lecture-cv/
├── lectures/                 # 教材本体（番号_スネークケースのモジュール）
│   ├── 00_setup/             # 環境構築・スモークテスト（check_env.py）
│   ├── 01_image_basics/      # ★作成済み（解説 + 実行コード + 演習）
│   ├── 02_… 〜 11_…/         # ★作成済み（ライブラリ地図・古典CV・動画・古典動画・リアルタイム）
│   ├── 12_… 〜 17_…/         # ★作成済み（深層: 拡張・分類/転移・評価・埋め込み・CLIP・FAISS）
│   ├── 18_… 〜 23_…/         # ★作成済み（検出・mAP自作・オープン語彙・セグメンテーション）
│   ├── 24_… 〜 26_…/         # ★作成済み（マスター水準: キャプション・VQA/VLM・OCR）
│   └── 27_… 〜 41_…/         # ロードマップ上のプレースホルダ（順次作成）
├── site/                     # 教材閲覧サイト（静的HTML・生成物）→ site/index.html を開く
├── tools/build_site.py       # 閲覧サイトのビルダー（Markdown→HTML）
├── docs/roadmap.md           # 全 42 モジュールのロードマップ（必読）
├── docs/curriculum.json      # 全モジュールのメタ情報（サイト生成・教材作成に使用）
├── data/                     # 入力データ（各自で配置。.gitkeep のみ追跡）
├── outputs/                  # 実行結果の出力先
├── pyproject.toml            # uv 依存定義（main + dependency-groups）
├── Dockerfile / docker-compose.yaml
└── README.md
```

## カリキュラム（トラック概観）

詳細・実行順・各回のゴールは **[docs/roadmap.md](docs/roadmap.md)** を参照してください。

| # | トラック | 主な内容 |
| --- | --- | --- |
| 1 | 画像の基礎 | ndarray/BGR・RGB、I/O、色空間・幾何変換、フィルタ・輪郭 |
| 2 | 古典 CV | 特徴点マッチング→ホモグラフィ→パノラマ、キャリブレーション/ステレオ、Watershed/GrabCut |
| 3 | 動画・ストリーム | VideoCapture/Writer、背景差分、リアルタイム最適化、RTSP |
| 4 | 深層 CV（分類） | transforms/拡張、ResNet/ViT 分類・転移学習 |
| 5 | 評価指標 | 混同行列/PR/ROC、★物体検出 mAP の自力実装 |
| 6 | 埋め込み・検索 | メトリック学習、FAISS 類似画像検索 |
| 7 | マルチモーダル | CLIP/SigLIP ゼロショット・画像テキスト検索、キャプション/VQA/OCR、ImageBind |
| 8 | 検出 | torchvision/YOLO/DETR、オープン語彙（OWL-ViT/Grounding DINO） |
| 9 | セグメンテーション | セマンティック/インスタンス/パノプティック、SAM、テキストプロンプト |
| 10 | 深度・姿勢・動き / 追跡 | 単眼深度、姿勢、オプティカルフロー、物体追跡、行動認識、顔 |
| 11 | 生成・編集 / 異常・品質 | 拡散モデル t2i・インペイント・超解像・背景除去、異常検知・IQA |
| 12 | 最適化・デプロイ | プロファイリング、量子化、枝刈り、ONNX、ランタイム最適化、知識蒸留・CLIP蒸留 |
| 13 | 応用 | Cluster-CLIP の中核と統合パイプライン（総合） |

## 依存グループ（`uv sync --group <name>`）

| グループ | 用途 | 主なパッケージ |
| --- | --- | --- |
| （main） | 画像基礎 | numpy, opencv-python-headless, pillow, matplotlib |
| `dl` | 深層学習の土台 | torch, torchvision（Linux は CPU ホイール） |
| `hf` | HuggingFace | transformers, huggingface-hub, timm, safetensors, sentencepiece, einops |
| `vector` | ベクトル検索 | faiss-cpu |
| `metrics` | 評価指標 | torchmetrics, scikit-learn, pycocotools |
| `aug` | データ拡張 | albumentations |
| `dev` | 開発ツール（既定） | ruff, mypy, poethepoet, ipython |

> `detect / face / anomaly / onnx / distill` などの専用トラックは、依存衝突を避けるため**その回に到達してから個別に追加**します（理由は docs/roadmap.md「設計メモ・既知の注意点」）。

## 参考

- お手本の講座サイト（構成イメージ）: [`../../courses`](../../courses)（GitHub Pages: <https://zackey2414.github.io/courses/>）
- 最終応用の題材: [`../../cluster-clip`](../../cluster-clip)

---

_lecture-cv ／ 設計時点: 2026-06。Python 3.12 / uv 0.10 系。各教材のライブラリ版は各回のフッターに明記します。_
