# 第15回 画像埋め込みとメトリック学習 — ViT/ResNet 特徴・対照/triplet 学習

> トラック: **埋め込み・検索** ／ レベル: **中級** ／ 必要な依存グループ: `dl` `hf`（torch / torchvision / transformers / timm）

## 🎯 この章のゴール

この章のゴールは、**「画像を 1 本のベクトル（埋め込み）に変換し、その良し悪しを測り、メトリック学習で空間そのものを作り変える」一連の流れを、AI 補助なしで自分の手で書ける**ようになることです。分類（13・14 回）が「この画像はどのクラスか」を当てるタスクだったのに対し、埋め込みは「画像どうしがどれくらい似ているか」を距離・角度で表せる汎用表現です。検索・クラスタリング・重複検出・few-shot 分類・推薦——下流のほとんどは、良い埋め込みさえあれば最近傍を引くだけで解けます。だからこそ「埋め込みをどう取り出し、どう評価し、どう良くするか」は埋め込み・検索トラックの土台になります。

具体的な到達点は 3 つです。1 つ目は、**分類ヘッド無しの `ViTModel` / `ResNetModel` と `timm` から埋め込みを正しく取り出せる**こと。ここで最大の落とし穴が「出力の形」です。ViT の `last_hidden_state` は `(B, 1+パッチ数, 次元)` の系列で先頭が CLS トークン、ResNet の `last_hidden_state` は `(B, C, H, W)` の特徴マップ——同じ名前でも形がまるで違い、取り違えると検索がそもそも動きません。2 つ目は、**埋め込みの品質を kNN 分類精度と Recall@k で定量化できる**こと。3 つ目は、このモジュールの完成物である、**Triplet / InfoNCE でメトリック学習を回し、埋め込み空間を「同クラスが近く」なるよう作り変える**スクリプトを書き上げることです。

すべて CPU のみで完走します。入力画像は合成生成（赤い円・緑の四角・青い三角…の 6 クラス）なので外部データは不要で、モデル重みだけ初回に HuggingFace / timm からダウンロードしてキャッシュします。深層学習が走りますが、重い学習はしません——バックボーンは凍結し、2 次元の射影ヘッドだけを数百反復学習するので、CPU でも各スクリプト数秒で終わります。

---

## 1. なぜ「埋め込み」を学ぶのか — 分類との違い

画像分類モデルは、最後に `Linear(隠れ次元 → クラス数)` の「分類ヘッド」を載せ、ロジット最大のクラスを答えます。しかし実務では「学習時に無かったクラスを後から足したい」「2 枚が同じ商品か知りたい」「似た画像を大量の中から探したい」といった、固定クラスに収まらない要求が大半です。これらは分類では解けませんが、**画像を固定長ベクトルへ写す関数（エンコーダ）**さえあれば、ベクトル間の距離・角度で一気に解けます。この「分類ヘッドの一歩手前のベクトル」こそが埋め込みです。

埋め込みの良さは「同じ意味の画像が近く、違う意味の画像が遠くに配置されているか」で決まります。ImageNet で学習済みのバックボーン（ResNet / ViT）は、何も追加学習しなくても、視覚的に似た画像を近いベクトルへ写す性質を既に持っています。だから本章の前半（`01`・`02`）は「学習済みモデルから埋め込みを取り出し、その素の実力を測る」ことに充て、後半（`03`）で「ラベルが示す similarity に合わせて空間を作り変える＝メトリック学習」へ進みます。

この章を貫く視点は「画像 → ベクトル → 距離」です。前半は良いベクトルの取り出し方、後半はそのベクトルの配置を学習で整えること。次章（16 回 CLIP ゼロショット、17 回 FAISS 検索）は、この埋め込みをテキストと結びつけたり、大規模に高速検索したりする応用なので、本章はその全ての前提になります。

## 2. 埋め込みの取り出しと「形」の違い（`01_vit_resnet_embeddings.py`）

埋め込みを取り出すには、分類ヘッド無しの「素のモデル」を使います。HuggingFace では `ViTForImageClassification` ではなく `ViTModel`、`ResNetForImageClassification` ではなく `ResNetModel` を `from_pretrained` します。前処理は `AutoImageProcessor`（transformers v5 では torchvision バックエンドの fast 実装がデフォルトで、`backend == "torchvision"`。旧 `AutoFeatureExtractor` や `use_fast=` 引数は廃止されたので、古いブログのコードをそのまま写すと動きません）。推論は必ず `model.eval()` と `torch.inference_mode()` で行い、勾配を切ってメモリ・時間を節約します（CPU では特に効きます）。

ここで最重要なのが**出力の "形"**です。`ViTModel` の `last_hidden_state` は `(B, 1+196, 768)`——画像を 16×16 のパッチ 196 個に切り、先頭に系列要約用の CLS トークンを足した系列です。埋め込みとしては `last_hidden_state[:, 0]`（CLS）か、パッチ平均 `last_hidden_state[:, 1:].mean(1)`（mean pooling）を使います。一方 `ResNetModel` の `last_hidden_state` は `(B, 512, 7, 7)` の**特徴マップ**で、これはベクトルではありません。ベクトルが欲しければ `pooler_output`（`(B, 512, 1, 1)` の Global Average Pooling 済み）を `flatten(1)` して `(B, 512)` にします。`timm` では `create_model(..., num_classes=0)` がプール済み `(B, 512)`、`forward_features` が未プール `(B, 512, 7, 7)` を返します。

```python
# ViT: 系列の先頭が CLS。pooler_output は使わない（後述の落とし穴）
out = vit_model(**proc(images=imgs, return_tensors="pt"))
cls  = out.last_hidden_state[:, 0]        # (B, 768)
mean = out.last_hidden_state[:, 1:].mean(dim=1)  # (B, 768)
# ResNet: last_hidden_state は (B,C,H,W) の特徴マップ。GAP 済みは pooler_output
gap = resnet_model(**inputs).pooler_output.flatten(1)  # (B, 512)
# timm: num_classes=0 でプール済み埋め込み、前処理はモデル固有値を自動生成
tm = timm.create_model("resnet18", pretrained=True, num_classes=0).eval()
cfg = timm.data.resolve_data_config({}, model=tm)
x = timm.data.create_transform(**cfg)(pil_image)
```

上のコードで覚えてほしいのは、`timm` の前処理を**手打ちしない**ことです。`resolve_data_config` + `create_transform` がモデル固有の入力サイズ・平均・分散を自動生成してくれます。ImageNet の平均/分散を手で書くとモデルによって微妙にズレて精度が落ちる事故が起きます。`01` を実行すると、6 クラスから 8 枚を取り、各埋め込みについて「同クラス平均コサイン類似度 − 異クラス平均」の **gap** を表示します。実測では `vit_mean ≈ +0.31` が最大、`vit_cls ≈ +0.19`、`resnet_gap ≈ +0.23`、`timm ≈ +0.21` と、どれも正の gap（＝同クラスがちゃんと近い）が出ます。

そして**この章で必ず一度は踏む落とし穴が `ViTModel.pooler_output`** です。`google/vit-base-patch16-224` のチェックポイントには学習済みの pooler が含まれず、ロード時の LOAD REPORT に `pooler.dense.* = MISSING`（新規初期化）と出ます。`pooler_output` は `tanh(W·CLS)` という CLS の変換なので形上は埋め込みが出ますが（実測 gap ≈ +0.18 と CLS に近い値が出てしまうため「壊れている」とは見えにくい）、**重みが未学習なので品質の保証がありません**。再現性・意味づけの面から、ViT の埋め込みは必ず CLS か mean pooling を使うのが定石です。`01_cosine_vit_cls.png` と `01_cosine_vit_pooler.png` を見比べ、CLS では同クラスの 2×2 ブロックがくっきり明るいことを確認してください。

## 3. L2 正規化とコサイン類似度 — なぜ「向き」で測るのか

埋め込みどうしの「近さ」をどう測るかには 2 つの代表があります。**ユークリッド距離**（ベクトルの差の長さ、小さいほど近い）と**コサイン類似度**（2 ベクトルのなす角、大きいほど近い）です。検索・メトリック学習ではコサイン類似度が定番です。理由は、画像の明るさ・コントラストが変わるとベクトルの「長さ（ノルム）」は変わりやすい一方、「向き」は意味をよく保つからです。コサインは長さを無視して向きだけを見るので、こうした撮影条件のブレに強くなります。

実装は単純で、各ベクトルを **L2 正規化**（ノルムを 1 にする）してから内積を取るだけです。`torch.nn.functional.normalize(x, p=2, dim=-1)`、または numpy で `x / np.linalg.norm(x, axis=1, keepdims=True)`。正規化後は「内積 = コサイン類似度」になります。FAISS（17 回）でも「`IndexFlatIP`（内積）＋ 事前 L2 正規化」でコサイン検索を実現するので、この習慣はそのまま次章で再利用します。ゼロ割を避けるため、ノルムには小さな下限（`1e-12`）を入れておくのが安全です。

```python
import torch.nn.functional as F
emb = F.normalize(emb, p=2, dim=-1)   # 各行を単位ベクトルに
sim = emb @ emb.T                      # 内積 = コサイン類似度（対角は 1.0）
```

`02` ではこの効果を数値で確かめます。きれいな合成データでは元々ノルムが揃っているので差が出にくいため、各ベクトルにランダムな正のスケール（0.2〜3.0 倍）を掛けて「露出差で長さがバラついた」状況を作ります。すると `resnet_gap` の kNN 精度は、コサイン（L2 正規化）が **0.97** を保つのに対し、生のユークリッド距離は **0.92** へ落ちます。向きだけを見るコサインが、長さの揺れに強いことが定量的に見て取れます。

## 4. 埋め込みの品質を測る — kNN 分類精度と Recall@k（`02_knn_recall_eval.py`）

埋め込みの良し悪しは、ベクトルを眺めても分かりません。**下流タスクの精度**で測るのが定石です。本章では検索・分類向けの 2 指標を、定義から実装します。まず**ギャラリー**（検索対象＝既知のラベル付きデータ）と**クエリ**（評価する側）に分けます。`kNN 分類精度`は「各クエリのコサイン最近傍 k 件を多数決し、予測ラベルが正解と一致した割合」。`Recall@k`は「各クエリの上位 k 件に同クラスが 1 件でも入る割合」。前者は 1 つのラベルに当てに行く厳しめの指標、後者は「欲しい仲間が上位 k に出てくるか」だけを問う検索向けの緩めの指標で、両者は別物です。

計算手順はどちらも共通で、(1) クエリ×ギャラリーのコサイン類似度行列を作り、(2) 各行で類似度の高い上位 k 件の添字を取り（`np.argsort(-sim)` の先頭 k）、(3) kNN なら k 件のラベルを多数決、Recall なら k 件に同クラスが含まれるかを判定します。実装は本モジュールの `embed_helpers.py` に `knn_accuracy` / `recall_at_k` として置いてあり、`01`〜`03` で共有します。下のように行列演算だけで書けるので、数千件程度なら FAISS を使わずとも一瞬です。

```python
sim  = normalize(query) @ normalize(gallery).T   # (nq, ng) 大きいほど近い
topk = np.argsort(-sim, axis=1)[:, :k]           # 各クエリの上位 k 添字
# kNN: 上位 k のラベルを多数決
preds = np.array([np.bincount(gallery_labels[r]).argmax() for r in topk])
acc   = (preds == query_labels).mean()
# Recall@k: 上位 k に同クラスが 1 件でもあれば hit
recall = (gallery_labels[topk] == query_labels[:, None]).any(1).mean()
```

`02` を実行すると、ViT(CLS / mean)・ResNet(GAP)・timm の 4 通りを横並びで比較します。実測では多くが kNN 精度 1.00 / Recall@5 1.00（`resnet_gap` のみ kNN 0.97）と、学習済みバックボーンが追加学習なしでこの合成タスクをほぼ完璧に解けることが分かります。結果は `02_knn_recall_compare.png`（棒グラフ）と、最良の埋め込みを PCA で 2 次元に落とした `02_pca_scatter.png`（クラスごとに分かれた島が見える）に保存されます。「素の埋め込みでここまで効く」——これが、良い表現が下流を支えるという本章のメッセージの前半です。

## 5. メトリック学習 — Triplet / InfoNCE で空間を作り変える（`03_triplet_infonce.py`）

では、埋め込みを「自分の similarity 定義に合わせて」もっと良くするには？ それが**メトリック学習**です。分類のように決定境界を引くのではなく、「同クラスは近く・異クラスは遠く」に**配置そのもの**を学習します。本章では学びを際立たせるため、凍結した `timm resnet18` の 512 次元特徴を、学習可能な線形ヘッドで**2 次元**まで圧縮します。2 次元にする狙いは 2 つ——(a) 空間を直接プロットして目で見られる、(b) 強い圧縮なので「良い配置」の効きが顕著に出る、です。実際、ランダム初期化の 2 次元射影では 6 クラスが重なり、kNN はわずか **0.33**（出発点）まで落ちます。512 次元なら 1.00 だった分離が、雑な圧縮で潰れるわけです。

代表的な 3 つの損失を実装して比較します。**Triplet（三つ組）損失**は、アンカー `a`・正例 `p`（同クラス）・負例 `n`（異クラス）について `d(a,p) + margin < d(a,n)` を満たすよう押し引きします。**ハードネガティブ・マイニング**は、バッチ内で「最も遠い正例」と「最も近い負例」をわざと選ぶ難しい三つ組で学ぶ手法で、難例ほど勾配が大きく学びが速くなります。**InfoNCE（教師ありコントラスト）**は、温度付き softmax で同クラス全体を正例集合として引き寄せます。下のコードは三つ組の押し引き（hinge）と InfoNCE の核心です。

```python
# Triplet（バッチハード）: 最も遠い正例 d_ap と 最も近い負例 d_an
loss = torch.relu(d_ap - d_an + margin).mean()
# InfoNCE: 正規化済み内積を温度で割り、同クラス位置の log 確率を上げる
sim = (z @ z.T) / temperature
sim.fill_diagonal_(-1e9)                       # 自分自身は除外
loss = -(same_mask * F.log_softmax(sim, 1)).sum(1) / same_mask.sum(1)
```

`03` を実行すると、ランダム 2 次元射影（kNN 0.33）から学習後に **Triplet(random) 0.98 / Triplet(hard) 0.95 / InfoNCE 0.98** へと、いずれも大きく改善します（`03_embedding_before_after.png` の左右で、重なっていた点群が放射状に分離する様子が見えます）。重要なのは**ハードネガティブの効き方**です。小規模・簡単なデータでは最終精度は拮抗しますが、収束は速くなります。実測では学習 30 反復時点で hard が **0.98**、random が **0.95** と、ハードネガティブが先に立ち上がります（`03_hardneg_convergence.png` の曲線）。これが「ハードネガティブはサンプル効率が良い」という主張の中身です。ただし難例に過集中すると不安定化することもあり、実務では semi-hard 等で和らげます。

最後に位置づけを 1 つ。**CLIP の画像-テキスト対照学習は、この InfoNCE をモーダル間に広げたもの**です。「画像とそのキャプションを正例ペア、バッチ内の他を負例」として InfoNCE を回し、画像エンコーダとテキストエンコーダを同一空間へ揃える——つまり CLIP はメトリック学習の一種です。本章で InfoNCE を手で書いておくと、次章（16 回）の CLIP がぐっと腑に落ちます。

## 6. このモジュールの構成（スクリプト一覧）

各スクリプトは単一責務で、上から順に「取り出す → 測る → 作り変える」と理解が積み上がるよう並べています。すべて結果を `outputs/15_image_embeddings_metric_learning/` に保存し、画面表示はしません。入力画像は `embed_helpers.py` が合成生成するので外部データ不要、モデル重みだけ初回にダウンロードしてキャッシュします。

| ファイル | 役割（単一責務） |
| --- | --- |
| `embed_helpers.py` | 共有部品。`get_device`（cpu/mps/cuda 判定）、合成ラベル付きデータセット、`l2_normalize` / `knn_accuracy` / `recall_at_k` |
| `01_vit_resnet_embeddings.py` | `ViTModel` / `ResNetModel` / `timm` から埋め込み抽出。`last_hidden_state` と `pooler_output` の形の違い、ViT pooler 未学習の罠、コサイン類似度ヒートマップ |
| `02_knn_recall_eval.py` | kNN 分類精度と Recall@k で埋め込み品質を評価。4 手法の横並び比較、L2 正規化（コサイン）の効果、PCA 散布図 |
| `03_triplet_infonce.py` | メトリック学習の核。凍結特徴 + 2D 射影ヘッドを Triplet / ハードネガティブ / InfoNCE で学習。学習前後の kNN/Recall 比較と収束曲線 |
| `mini_project.py` | **章末ミニプロジェクト（統合の完成物）**。抽出 → 評価 → メトリック学習を「コンパクト検索エンジン」に統合。圧縮スイープ・検索グリッド・metrics.json を出力 |
| `exercises.py` | TODO 形式の演習 9 問（自己採点ランナー `grade()` 付き）。numpy だけで完結しモデル DL 不要 |
| `exercises_solutions.py` | 演習 9 問の模範解答（全 PASS）。`exercises.py` の `grade()` を再利用して採点（採点ロジックは重複なし） |

表の通り `mini_project.py` が deliverable の中核（埋め込み抽出・評価・メトリック学習の統合）で、`03` がそのメトリック学習部分、`01`・`02` が前提（取り出しと評価）です。まず `01` から順に実行し、各 `outputs/15_*.png` を開きながら本文を読み返すと理解が定着します。

## 7. 動かし方

このモジュールは `dl`（torch / torchvision）と `hf`（transformers / timm / huggingface-hub）の依存グループを使います。CPU のみで完走し、合成画像なので外部データは不要です（モデル重みのみ初回にダウンロード）。プロジェクトルートで以下を順に実行してください。

```bash
# 依存をインストール（初回のみ。深層トラックのグループを足す）
uv sync --group dl --group hf

# 各スクリプトを実行（結果は outputs/15_image_embeddings_metric_learning/ に保存される）
uv run python lectures/15_image_embeddings_metric_learning/01_vit_resnet_embeddings.py
uv run python lectures/15_image_embeddings_metric_learning/02_knn_recall_eval.py
uv run python lectures/15_image_embeddings_metric_learning/03_triplet_infonce.py

# 章末ミニプロジェクト（統合の完成物）: コンパクト検索エンジン + 圧縮スイープ
uv run python lectures/15_image_embeddings_metric_learning/mini_project.py

# 演習: まずは TODO を自分で埋める（最初は全部 FAIL でも exit 0）
uv run python lectures/15_image_embeddings_metric_learning/exercises.py
# どうしても分からない時だけ、模範解答（全 PASS）の挙動を見る
uv run python lectures/15_image_embeddings_metric_learning/exercises_solutions.py
```

実画像で試したい人は、`data/` に画像を置き、`embed_helpers.make_dataset` の代わりに `PIL.Image.open(path).convert("RGB")` で読み込んで同じ抽出関数へ渡せば、そのまま kNN / Recall / メトリック学習が動きます。初回のモデルダウンロードを避けたい場合は、キャッシュ済みなら `HF_HUB_OFFLINE=1` を付けてオフライン実行できます。Docker では `HF_HOME`（既定 `~/.cache/huggingface`）をボリュームマウントすると再ダウンロードを防げます。

## 8. よくあるエラーと対処（チェックリスト）

実装中に詰まったら、まずこの表を見てください。この章の不具合の大半は、ここに挙げた数個に集約されます。とくに上の 2 つ（出力の形・正規化忘れ）は必ず一度は遭遇します。

| 症状 | ほぼ確実な原因 | 対処 |
| --- | --- | --- |
| 埋め込みのつもりが 4 次元 `(B,C,H,W)` で検索が壊れる | ResNet の `last_hidden_state`（特徴マップ）を使った | `pooler_output.flatten(1)` で `(B, C)` のベクトルにする |
| ViT 埋め込みの品質が安定しない/再現しない | `pooler_output` を使った（CKPT 由来で未学習） | CLS `last_hidden_state[:,0]` か mean pooling を使う |
| コサイン類似度が変な値になる/検索順がおかしい | L2 正規化を忘れて生の内積を取った | `F.normalize(x, p=2, dim=-1)` してから内積 |
| `AutoFeatureExtractor` が無い / `use_fast` でエラー | transformers v5 で廃止された | `AutoImageProcessor` を使う（fast/torchvision がデフォルト） |
| `RuntimeError: expected ... same device` | model と inputs の device がズレた | `inputs.to(model.device)` で揃える。CPU は float32 |
| timm の精度が出ない | 前処理の平均/分散を手打ちしてズレた | `resolve_data_config` + `create_transform` を使う |
| 図が出ない/フリーズ | matplotlib のバックエンド未設定 | `pyplot` import 前に `matplotlib.use("Agg")` |

この 7 項目を「症状を見たら原因が言える」状態にできれば、この章のゴールに到達しています。

## 9. まとめ

この章では、`ViTModel` / `ResNetModel` / `timm` からの埋め込み抽出（`last_hidden_state` と `pooler_output` の形の違い、ViT pooler 未学習の罠）→ L2 正規化とコサイン類似度（なぜ向きで測るか）→ kNN 分類精度と Recall@k での品質評価 → Triplet / InfoNCE / ハードネガティブによるメトリック学習、までを「自分で書いて・なぜそうするか説明できる」レベルで扱いました。とくに「出力の形を取り違えない」「コサインの前に必ず L2 正規化」「ViT は CLS か mean、pooler は使わない」の 3 点は、知っているだけで無駄なデバッグを確実に減らせます。

次章（16 回）では、ここで手書きした InfoNCE がそのまま `openai/clip-vit-base-patch32` の対照学習として現れ、画像とテキストを同一空間で結ぶゼロショット分類・検索へ進みます。続く 17 回では、本章の「L2 正規化 → コサイン類似度」を FAISS の `IndexFlatIP` で大規模・高速に回し、Recall@k を自前で測ります。本章の埋め込みと評価の感覚が、その全ての前提です。まずは演習 9 問を自力で全問 PASS させ、ベクトルの扱いを手に馴染ませてから次へ進んでください。

---

## 🛠 章末ミニプロジェクト — コンパクト埋め込み検索エンジン（`mini_project.py`）

この章の総まとめとして、`01`（取り出す）→ `02`（測る）→ `03`（作り変える）を **1 本の検索エンジン**へ統合します。テーマは実務直結の問い——**「埋め込みはどこまで小さくできるか。メトリック学習でどこまで攻めた圧縮が許されるか」**です。検索インデックス（17 回の FAISS）はベクトルが小さいほどメモリ・速度で有利なので、「品質を保ったまま次元を削る」ことには大きな価値があります。

`mini_project.py` がやることは 4 段です。(1) 合成 6 クラスを **timm resnet18(512 次元)** と **ViT CLS(768 次元)** の 2 バックボーンで埋め込み、ギャラリー/クエリに分けて素の検索性能（kNN 精度・Recall@5）を測る。(2) 凍結した resnet18 特徴を、いろいろな小次元 `d ∈ {2,4,8,16,32}` へ射影し、**ランダム射影（学習なし）** と **InfoNCE 学習済み射影（メトリック学習）** を比較する圧縮スイープを回す。(3) 「Recall@5 ≥ 0.95 を保てる最小次元」をランダム/学習で比べる。(4) スイープ曲線・学習前後の 2D 散布図・検索結果グリッド（緑＝正解/赤＝不正解）・`mini_metrics.json` を保存する。

実測のハイライト（CPU・数十秒）は次の通りです。素の 512/768 次元はどちらも kNN 1.00 / Recall@5 1.00。しかし **2 次元まで圧縮すると、ランダム射影は Recall@5 が 0.71 まで崩れる**のに対し、**InfoNCE で並べ替えた 2 次元は 0.98 を保ちます**。「Recall@5 ≥ 0.95 を保てる最小次元」はランダム射影が `d=4`、メトリック学習が `d=2`——**同じ検索品質を半分の次元で達成**できるわけです。これが「メトリック学習は強い圧縮に効く」という本章のメッセージの完成形です。`mini_compression_sweep.png`（2 本の曲線）と `mini_retrieval_grid.png`（2 次元埋め込みでの実検索）を開いて、圧縮と品質のトレードオフを目で確かめてください。

| 出力ファイル | 内容 |
| --- | --- |
| `mini_compression_sweep.png` | 射影次元 d に対する Recall@5（random vs InfoNCE、512 次元の水準線つき） |
| `mini_pca_before_after.png` | 最小次元での学習前後の埋め込み（重なり → 分離） |
| `mini_retrieval_grid.png` | コンパクト埋め込みでの上位 K 検索（緑＝正解クラス/赤＝不正解） |
| `mini_metrics.json` | 素の埋め込み・圧縮スイープ・最小保持次元の全数値 |

## ✅ 到達チェックリスト

次の問いに「コードを見ずに口頭で説明できる」なら、この章のゴールに到達しています。詰まった項目があれば、対応する節・スクリプトに戻って手を動かしてください。

- [ ] `ViTModel` の `last_hidden_state` と `ResNetModel` の `last_hidden_state` が**なぜ形（次元数）が違うのか**を説明できる（系列 vs 特徴マップ）。
- [ ] ViT の埋め込みに **CLS か mean pooling を使い、`pooler_output` を避ける理由**（CKPT 由来で未学習）を言える。
- [ ] ResNet/timm でベクトル埋め込みが欲しいとき、**どの出力をどう変形するか**（`pooler_output.flatten(1)` / `num_classes=0`）を即答できる。
- [ ] **L2 正規化 → 内積 = コサイン類似度**であること、なぜ「長さ」でなく「向き」で測るのかを説明できる。
- [ ] **kNN 分類精度と Recall@k の違い**（多数決で当てる vs 上位 k に仲間が出るか）を定義から言える。
- [ ] **Triplet 損失（`d(a,p)+margin < d(a,n)`）** と **ハードネガティブ・マイニング**の狙い（サンプル効率）を説明できる。
- [ ] **InfoNCE が CLIP の画像-テキスト対照学習と同型**であることを言える。
- [ ] `mini_project.py` の圧縮スイープで、**メトリック学習がより小さい次元で同じ Recall を保つ**ことを自分の言葉で要約できる。
- [ ] 演習 9 問すべてを `exercises.py` で **自力 PASS** できる（`exercises_solutions.py` を見る前に）。

## ❓ よくある落とし穴・FAQ・デバッグ

§8 のエラー表（症状 → 原因 → 対処）に加え、つまずきやすい論点を Q&A で補足します。

- **Q. ヒートマップの対角が 1.0 にならない / 同クラスブロックが暗い。** A. 正規化前に内積を取っているか、ViT の `pooler_output`（未学習）を使っている可能性が高い。`F.normalize(x, p=2, dim=-1)` を必ず先に通し、ViT は CLS/mean を使う。
- **Q. kNN 精度は 1.00 なのに Recall@k が下がる（またはその逆）。** A. 別物の指標なので不一致は正常。kNN は近傍 k 件の**多数決で 1 ラベルに当てる**厳しめの指標、Recall@k は上位 k に**同クラスが 1 件でもあれば hit** の緩い検索指標。`k` を変えると両者の差が広がる。
- **Q. メトリック学習しても精度が上がらない（`03`/mini で +0.000）。** A. 圧縮が弱い（次元が大きい）と素の埋め込みで既に満点のため伸び代が無い。`mini_project.py` のように **強く圧縮（2〜4 次元）して headroom を作る**と効果が見える。逆に過圧縮（極端な低次元）+高 LR は不安定化するので LR/温度を下げる。
- **Q. `np.bincount` でエラー / kNN の多数決がおかしい。** A. `np.bincount` は**非負整数ラベル**前提。ラベルが連続 int でない場合は `np.unique(..., return_inverse=True)` で詰めてから使う。
- **Q. バッチハード Triplet の損失が常に 0 になる。** A. クラスが完全分離していると `d(a,p) < d(a,n)` が常に成立し hinge=0 になる（これ自体は正常）。学習挙動を観察したいなら、わざと重なるデータか大きめ margin で試す（演習 7 の採点サンプルは重なりを入れてある）。
- **Q. `transformers` で `AutoFeatureExtractor` が無い / `use_fast=` でエラー。** A. v5 で廃止。`AutoImageProcessor`（torchvision バックエンドの fast がデフォルト）に置き換える。
- **デバッグの定石**: 埋め込みが壊れたら **(1) 形 `emb.shape` を print**（ベクトル `(B, D)` か、特徴マップ `(B,C,H,W)` を渡していないか）→ **(2) ノルム `np.linalg.norm(emb,axis=1)`** が極端でないか → **(3) 正規化後に同クラスのコサインが高い**か、の順で切り分ける。

## 🚀 発展トピック・参考

- **損失の発展**: ここで触れた Triplet / InfoNCE の他に、**ArcFace / CosFace**（角度マージンを分類ヘッドに埋め込む顔認証の定番）、**Proxy-Anchor / Proxy-NCA**（各クラスの代理点で計算量を削減）、**SupCon**（教師ありコントラストの一般形）がある。`pytorch-metric-learning` ライブラリにこれらが揃っている。
- **ハードネガティブの加減**: 最難例に過集中すると崩壊（collapse）しやすい。**semi-hard**（`d(a,p) < d(a,n) < d(a,p)+margin` の負例だけ使う）や、バッチ全体を使う **multi-similarity loss** が実務では安定。
- **次元削減の選択肢**: 学習で射影する以外に、**PCA / 教師あり次元削減**で軽量化する手もある。`mini_project.py` の圧縮スイープと突き合わせて、学習射影が PCA をどれだけ上回るか試すと面白い。
- **大規模検索への橋渡し**: 本章の「L2 正規化 → コサイン類似度」は、17 回の **FAISS `IndexFlatIP`** にそのまま載る。さらに `IVF`/`HNSW`/`PQ` で近似最近傍にすれば、数百万件規模でも実時間で検索できる（精度-速度のトレードオフは Recall@k で測る）。
- **CLIP への接続**: `03` と mini の InfoNCE は、16 回の `openai/clip-vit-base-patch32` の画像-テキスト対照学習そのもの。本章で手書きした損失が、次章のゼロショット分類・クロスモーダル検索の中身になる。
- 公式ドキュメント: [transformers ViT/ResNet](https://huggingface.co/docs/transformers/en/index) ／ [timm feature extraction](https://huggingface.co/docs/timm/feature_extraction) ／ [torchmetrics retrieval](https://lightning.ai/docs/torchmetrics/stable/) ／ [FAISS wiki](https://github.com/facebookresearch/faiss/wiki)。

---

> 本教材で参照・検証したライブラリとバージョン（2026-06 時点の安定版で動作確認）:
> Python 3.12 ／ torch 2.12+cpu（2.12.0+cpu）／ torchvision 0.27+cpu（0.27.0+cpu）／ transformers 5.11（5.11.0）／ timm 1.0.27 ／ huggingface-hub 1.18 ／ safetensors 0.8 ／ numpy 2.4（2.4.6）／ scikit-learn 1.9（1.9.0）。
> 関連章で使う faiss-cpu 1.14.2（17 回）も同環境で動作します。すべて CPU のみで完走し、ネット接続は初回のモデル重みダウンロードのみ（以後キャッシュ）。結果は `outputs/15_image_embeddings_metric_learning/` に保存します。