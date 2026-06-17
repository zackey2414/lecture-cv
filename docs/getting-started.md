# はじめ方 — clone してから、どこに書いて進めるか

> この 1 ページで、**手元の PC に取得して動かす**ところから、**どのディレクトリ／ファイルを読み・書き換え・自分で書くのか**、そして**各回の進め方**までを通しで案内します。GPU は不要です（CPU のみで全 46 回を完走できます）。**環境で迷う人・Mac で深層トラックに進む人は Docker が確実**です（どのデバイスでも全回動く。理由と手順は §6 と末尾の付録）。
>
> git / GitHub の使い方そのものは説明しません（clone・branch・commit は分かっている前提）。focus は **「clone した後、何をどこに書けば学習が進むのか」** です。

## 1. 取得して、最初の一歩まで

```bash
git clone https://github.com/zackey2414/lecture-cv.git
cd lecture-cv

# uv 未導入なら: curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync                                          # main 依存（numpy / opencv / pillow / matplotlib）
uv run python lectures/00_setup/check_env.py     # 環境スモークテスト（device 判定など）
uv run python lectures/01_image_basics/01_imread_imwrite.py
```

これだけで最初のトラック（00〜11）は動きます。以降は **`uv run` を頭に付ける**だけ（`.venv` の有効化は不要）。

> 🐳 **環境で迷ったら／Mac で深層トラック（12 以降）に進むなら Docker が確実です。** 母艦が **Intel Mac・Apple Silicon・Windows・Linux のどれでも**、コンテナは Linux なので **全 46 回が確実に動きます**。とくに **Intel Mac は深層トラックの PyTorch をネイティブに入れられません**（PyTorch が torch 2.3 以降の Intel Mac 向け配布を終了したため）。その場合は下の「付録: Docker で始める」へ。詰まったときの切り分けは §6。

## 2. ディレクトリの地図 — どこを「読み」、どこに「書く」か

clone 直後の構成と、**あなたが触る／触らない**の対応です。学習で**あなたが書き込むのは基本 `lectures/<id>/exercises.py` の TODO と `data/` への入力配置だけ**です。

```
lecture-cv/
├── lectures/                 # ← 教材本体。ここを順に進める
│   └── <NN_モジュール名>/      #    例: 01_image_basics, 17_faiss_image_search …
│       ├── README.md         # 📖 読む：その回の解説・図解（まずここ）
│       ├── NN_*.py           # ▶️ 動かす：番号順の実行スクリプト（読んで・書き換えて実験）
│       ├── exercises.py      # ✍️ 書く：TODO を自分で埋める“あなたのファイル”（自己採点つき）
│       ├── exercises_solutions.py  # 🔑 見る：模範解答（先に自分で解いてから開く）
│       └── outputs/          # 🆕 自動生成：その回の実行結果の保存先（触らなくてよい）
├── data/                     # ⬇️ 置く：自分の入力画像・動画はここ（.gitkeep のみ追跡）
├── docs/                     # 教材インフラ：ロードマップ等（読む用。編集不要）
├── tools/build_site.py       # 教材インフラ：閲覧サイトのビルダー（編集不要）
├── site/                     # 教材インフラ：サイト生成物（gitignore・編集不要）
├── pyproject.toml            # 依存定義（グループを足すときだけ uv が更新）
└── Dockerfile / docker-compose.yaml
```

要点:

- **読むのは** `lectures/<id>/README.md`（解説）。**動かすのは** 同じフォルダの番号付き `NN_*.py`。
- **あなたが書くのは** `lectures/<id>/exercises.py` の **TODO**。ここを埋めるのが学習の中心です。
- **自分の画像・動画は** `data/` に置く（教材は合成画像でも動くので、無くても始められます）。
- **実行結果は** 各回の `lectures/<id>/outputs/` に自動保存（headless 環境でも後から確認できる）。**ここは見るだけ**でOK。
- `docs/` `tools/` `site/` は**教材の仕組み側**。学習中に編集する必要はありません。

## 3. 1 つの回をどう進めるか（4 ステップ）

`lectures/<id>/` を 1 つ開いたら、毎回この順で進めます。

1. **解説を読む** — `README.md`（または各回ページ）の地の文と**図解**で、配列の形・座標系・前処理の流れをつかむ。
2. **スクリプトを動かす** — 番号順に実行し、`lectures/<id>/outputs/` の結果を見る。
   ```bash
   uv run python lectures/<id>/01_xxx.py
   uv run python lectures/<id>/02_yyy.py
   ```
3. **書き換えて実験** — パラメータや入力画像を変え、出力がどう変わるか観察する（ここが一番伸びる）。
4. **演習を自分で書いて採点** — `exercises.py` の **TODO** を自力で埋め、実行すると**自己採点**されます。
   ```bash
   uv run python lectures/<id>/exercises.py          # 自己採点が走る
   SHOW_SOLUTION=1 uv run python lectures/<id>/exercises.py   # 答え合わせ（または exercises_solutions.py を読む）
   ```

> コツ: **模範解答（`exercises_solutions.py`）は先に開かない**。手が止まったときだけ見ましょう。

## 4. 回ごとに依存グループを足す

深層・各タスクの回は、**到達してから**必要なグループだけ追加します（重い依存を最初から全部入れない方針）。**各回ページの「依存グループ」欄**に必要なグループが書いてあるので、その通りに足します。

```bash
uv sync --group dl --group hf             # PyTorch + HuggingFace（分類・CLIP・検出 …）
uv sync --group vector --group metrics    # FAISS + 評価指標
```

| グループ | 用途 | 例 |
| --- | --- | --- |
| （main） | 画像基礎 | numpy, opencv, pillow, matplotlib |
| `dl` | 深層学習の土台 | torch, torchvision（Linux は CPU ホイール） |
| `hf` | HuggingFace | transformers, timm, safetensors … |
| `vector` | ベクトル検索 | faiss-cpu |
| `metrics` | 評価指標 | torchmetrics, scikit-learn, pycocotools |

> 全グループ一覧は **README の「依存グループ」表**。`diffusion` / `onnx` / `embed` などの重いグループも、該当回で同じ要領で個別追加します。

## 5. どの順番で進めるか — 番号順ではなく「グラフ」

番号は安定した ID にすぎません。学ぶ順番は各回の **前提（prerequisite）** をたどる**有向グラフ（DAG）**です。

- まずは **`00_setup` → `01_image_basics`** から番号順に進め、各回ページ上部の「前提」リンクで前後関係を確認するのが安全。
- 全体像は **[学習順序グラフ](graph.html)**（依存グラフ・推奨順・前提一覧）と **[ロードマップ](roadmap.html)**（全 46 回をトラック別に一覧）で確認できます。

## 6. つまずいたら

- **ImportError（依存が無い）**: その回の「依存グループ」を `uv sync --group <name>` で足したか確認。
- **`torch ... doesn't have ... wheel for ... macosx ... x86_64`（`uv sync --group dl` が失敗）**: あなたは **Intel Mac**。PyTorch は torch 2.3 以降 Intel Mac 向け wheel を出していないため、ネイティブでは深層トラックの torch が入りません（世界的にどのプロジェクトでも同じ制約）。→ **「付録: Docker で始める」を使ってください**（全 46 回が確実に動きます）。基礎トラック(00〜11)だけなら `uv sync`（`--group dl` なし）でネイティブでも動きます。／**Apple Silicon Mac なのにこの x86_64 エラーが出る**場合は uv・Python が Rosetta(x86_64) で動いています。素の arm64 ターミナル（`uname -m` が `arm64`）で `curl -LsSf https://astral.sh/uv/install.sh | sh` → `file "$(command -v uv)"` が arm64 になったら `rm -rf .venv && uv sync --group dl`。
- **画面に何も出ない**: 既定は headless。`lectures/<id>/outputs/` に保存された画像を見る。`cv2.imshow` を使いたい時だけ `opencv-python-headless` を `opencv-python`（GUI 版）に差し替え（両者は排他）。
- **モデルDLで止まる**: 深層トラックは初回に HuggingFace からモデルを取得。ネットワークと `~/.cache/huggingface` の容量を確認。
- **遅い**: CPU 前提の設計。中級以降で解像度低減・フレームスキップ・量子化などの **CPU 最適化**を扱います。

---

## 付録: Docker で始める / サイトを手元でプレビュー

**Docker はどのデバイスでも全 46 回が確実に動く道**です（コンテナは Linux なので、母艦が Intel Mac・Apple Silicon・Windows・Linux のいずれでも PyTorch の CPU 版がそのまま動きます）。深層トラックの依存（dl/hf/vector/metrics/aug）はイメージのビルド時に導入されるので、起動後はそのまま 12 以降も実行できます。

```bash
# Docker（CPU 既定。GPU は docker-compose.yaml のコメント参照）
docker compose up -d --build
docker compose exec lecture-cv uv run python lectures/00_setup/check_env.py
docker compose exec lecture-cv uv run python lectures/01_image_basics/01_imread_imwrite.py

# この「はじめ方」を含む教材サイトを手元でビルド（生成物 site/ はコミットしない）
uv run --group site python tools/build_site.py   # 生成後 site/index.html を開く
```

準備ができたら、**[トップ（全回の一覧）](index.html)** から最初の回を開いて始めましょう。
