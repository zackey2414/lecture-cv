# lecture-cv

Computer Vision（CV）を「**AI の補助なしでも自分一人でゴリゴリ書け、内容を熟知している**」状態まで叩き込むための、ハンズオン講座リポジトリです。OpenCV / Pillow の画像基礎から、古典 CV、深層 CV（分類・検出・セグメンテーション・深度/姿勢/追跡）、CLIP/SigLIP とベクトル検索（FAISS）、キャプション/VQA/OCR、生成・編集、各タスクの**精度評価**（物体検出 mAP は自力実装）、そして量子化・枝刈り・ONNX・知識蒸留などの**モデル圧縮/高速化**まで、全 46 モジュールを段階的に実装します。

最終的には、**Cluster-CLIP**（dense CLIP 特徴 + 空間連結クラスタリング + FAISS + ストリーム処理）を、CPU で動く小型版として自力再構築できる到達度を目指します。

## 特徴

- **GPU 不要**: 全モジュールが CPU のみ（MacBook 等）で完走できるよう設計。GPU は任意の高速化です。
- **docker + uv**: 環境は `uv` で管理。Docker でも同一構成を再現できます。
- **ハンズオン**: 各回は「解説（地の文）＋ 実行できる `.py` ＋ 演習」。読むだけでなく**動かして・書き換えて**学びます。
- **図解つき**: 各回に概念図（インラインSVG）を多数埋め込み、配列の形状・座標系・前処理パイプライン・評価指標などを視覚的につかめます。
- **評価を重視**: 各タスクで評価指標を必ず実測。`19` では mAP を numpy で一から実装し pycocotools と突き合わせます。
- **ネット非依存**: 第1回はサンプル画像が無くても合成画像で完走します。

## 教材サイト（ブラウザで読む）

各回の解説・コード・演習・図解を、見やすい静的サイトで一覧・通読できます。**公開サイトはこちら**:

### 📖 https://zackey2414.github.io/lecture-cv/

オレンジ基調のデザインで、**ジャンル別／難易度別に切り替えられる**カード、各回に埋め込んだ**図解（インラインSVG）**、コードハイライトを備えます。テキストとして読み込み、内容を根から理解するのに使ってください。サイト内の **「学習順序グラフ」**（`graph.html`）には、前提（prerequisite）でつながった**依存グラフ（DAG）・推奨学習順・前提一覧**があり、この講座が番号順の一本道ではなく各回の前提をたどるグラフ構造であることが分かります（後から回を足しても筋道をたどれます）。**ロードマップの一覧表からも各回ページへ飛べます。**

> はじめての方は、サイト上部の **「はじめ方」**（`getting-started.html`）を最初に開いてください。**ローカルでの始め方・最初の一歩・各回の進め方**を 1 ページにまとめています。

サイトは GitHub Actions が `main` への push 時に自動でビルドし GitHub Pages へ配信します。**生成物 `site/` はリポジトリにはコミットしません**（教材ソースだけを追跡）。手元でプレビューしたいときは次でビルドできます。

```bash
# Markdown→HTML を事前レンダリング（プレビュー用。生成物はオフラインで開けます）
uv run --group site python tools/build_site.py
# 生成後、site/index.html をブラウザで開く（macOS: open site/index.html）
```

## クイックスタート

> 🐳 **迷ったら Docker。** 母艦が **Intel Mac・Apple Silicon Mac・Windows・Linux のどれでも**、コンテナは Linux なので **全 46 回が確実に動きます**。とくに深層トラック（12 以降）は PyTorch を使い、**Intel Mac ではネイティブに torch を入れられない**（PyTorch が torch 2.3 以降の Intel Mac 向け配布を終了）ため、Docker が唯一確実な道です。手早く基礎だけ触るなら uv のネイティブ実行が軽量です。Docker と uv の役割分担（箱は Docker・中身は uv）は [docs/docker-basics.md](docs/docker-basics.md) を参照。

### A) Docker（推奨 — どのデバイスでも全回が動く）

Docker は「箱（OS＋ライブラリ＋Python＋uv）」を用意するだけ。**コンテナに入って、その中で `uv` で環境を整えて実行**します（Docker と uv の役割分担・ファイル関係は [docs/docker-basics.md](docs/docker-basics.md)）。

```bash
# ① ホストで：起動してコンテナに入る（初回だけ --build）
docker compose up -d --build
docker compose exec lecture-cv bash
```

```bash
# ② コンテナの中で：uv で整えて実行（プロンプトが container 内に変わる）
uv sync                                   # 画像基礎(00〜11)の依存をそろえる
uv run python lectures/00_setup/check_env.py
uv run python lectures/01_image_basics/01_imread_imwrite.py
uv sync --group dl --group hf             # 深層トラック(12 以降)に進むとき必要なグループを足す
```

> 各回ページの「動かし方」の `uv ...` コマンドは、Docker の場合①でコンテナに入った後そのまま実行できます。詳細は [docs/docker-basics.md](docs/docker-basics.md)。

### B) ローカル（uv — 軽量に基礎から始めたい人向け）

```bash
uv sync                                   # 画像基礎〜古典CV〜古典動画(00〜11): numpy/opencv/pillow/matplotlib
uv run python lectures/00_setup/check_env.py          # 環境スモークテスト
uv run python lectures/01_image_basics/01_imread_imwrite.py

# 深層・各タスクのトラックに進むとき、必要なグループだけ足す
uv sync --group dl --group hf             # PyTorch + HuggingFace
uv sync --group vector --group metrics    # FAISS + 評価指標
```

> ⚠️ **Intel Mac（x86_64）はローカルの深層トラックが動きません。** `uv sync --group dl` が「torch … doesn't have … wheel for … macosx … x86_64」で失敗します（PyTorch が torch 2.3 以降の Intel Mac 向け wheel を廃止したため。pyproject では作れない外部制約）。→ **深層トラックは上の A) Docker を使ってください**（Apple Silicon Mac はネイティブで動きます）。詳しい切り分けは [はじめ方ガイドの「つまずいたら」](docs/getting-started.md) を参照。

> 結果は基本的に各回の `lectures/<モジュール>/outputs/` に保存されます（headless 環境でも後から確認できるように）。
> `cv2.imshow` を使いたい場合は、既定の `opencv-python-headless` を `opencv-python`（GUI 版）へ差し替えてください（両者は排他）。

## ディレクトリ構成

```
lecture-cv/
├── lectures/                 # 教材本体（各回 = README・*.py・演習。実行結果は各回内の outputs/ に自動生成）
│   ├── 00_setup/             # 環境構築・device 判定・スモークテスト（check_env.py）
│   ├── 01_… 〜 04_…/         # 画像の基礎: ndarray・BGR/RGB・I/O・色空間/幾何変換・フィルタ/エッジ/輪郭
│   ├── 05_… 〜 08_…/         # 古典CV: 特徴点マッチング・ホモグラフィ/パノラマ・キャリブ/ステレオ・古典セグメ
│   ├── 09_… 〜 11_…/         # 動画・ストリーム: I/O・オプティカルフロー/背景差分・リアルタイム/RTSP
│   ├── 12_… 〜 14_…/         # 深層分類: データ拡張・ResNet/ViT 転移学習・分類評価指標
│   ├── 15_… 〜 17_…/         # 埋め込み・検索: メトリック学習・CLIP ゼロショット・FAISS 画像検索
│   ├── 18_… 〜 20_…/         # 検出: torchvision/YOLO/DETR・mAP の自力実装・オープン語彙
│   ├── 21_… 〜 23_…/         # セグメンテーション: セマンティック・インスタンス/SAM・テキストプロンプト
│   ├── 24_… 〜 26_…/         # キャプション生成・VQA/VLM・OCR/文書理解
│   ├── 27_… 〜 30_…/         # 深度/姿勢/フロー・物体追跡・動画行動認識・顔検出/認識
│   ├── 31_… 〜 33_…/         # 生成・編集（拡散モデル）・異常検知/IQA・マルチモーダル埋め込み
│   ├── 34_… 〜 39_…/         # 最適化: プロファイリング・量子化/枝刈り・ONNX・ランタイム・知識蒸留
│   ├── 40_…, 41_…/          # 応用 Cluster-CLIP: dense特徴クラスタリング・統合パイプライン（総仕上げ）
│   └── 42_… 〜 45_…/         # 応用: マルチモーダル検索・色空間調整・埋め込みクラスタリング・スケッチ検索
├── site/                     # 教材閲覧サイト（生成物・gitignore。CIでビルドし Pages へ配信）
├── tools/build_site.py       # 閲覧サイトのビルダー（Markdown→HTML）
├── docs/getting-started.md   # はじめ方ガイド（clone 後の進め方・どこに書くか。サイトの「はじめ方」）
├── docs/docker-basics.md     # Docker 入門（Docker と uv の責務・ファイル関係・コンテナで uv 運用）
├── docs/roadmap.md           # 全 46 モジュールのロードマップ（必読）
├── docs/curriculum.json      # 全モジュールのメタ情報（サイト生成・教材作成に使用）
├── data/                     # 入力データ（各自で配置。.gitkeep のみ追跡）
├── pyproject.toml            # uv 依存定義（main + dependency-groups）
├── Dockerfile / docker-compose.yaml
├── .github/workflows/        # GitHub Pages 自動ビルド＆配信（deploy-pages.yml）
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
| `diffusion` | 生成・編集（拡散モデル） | diffusers, accelerate |
| `onnx` | ONNX 変換・推論 | onnx, onnxruntime |
| `embed` | 効率 CLIP | open-clip-torch |
| `site` | 教材サイト生成 | markdown, pygments |
| `dev` | 開発ツール（既定） | ruff, mypy, poethepoet, ipython |

> 生成（`diffusion`）・ONNX（`onnx`）・効率CLIP（`embed`）などの重い/専用グループは、依存衝突や容量を避けるため**その回に到達してから個別に追加**します（理由は docs/roadmap.md「設計メモ・既知の注意点」）。

## 参考

- 姉妹プロジェクト（プログラミング講座サイト・GitHub Pages）: <https://zackey2414.github.io/courses/>

---

_lecture-cv ／ 設計時点: 2026-06。Python 3.12 / uv 0.10 系。各教材のライブラリ版は各回のフッターに明記します。_
