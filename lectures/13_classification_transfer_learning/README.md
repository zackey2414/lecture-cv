# 第13回 画像分類と転移学習 — ResNet/ViT(torchvision・timm・HuggingFace)

> トラック: **深層CV(分類)** ／ レベル: **中級** ／ 必要な依存グループ: `dl`（torch・torchvision）と `hf`（transformers・timm・huggingface_hub・safetensors）。評価で `metrics`（torchmetrics・scikit-learn）も使います。すべて **CPU で完走**します。

## 🎯 この章のゴール

ここまで（00〜12回）では、OpenCV/Pillow による画像表現と前処理、古典的な特徴量、そしてデータパイプラインを扱ってきました。本章からは、いよいよ**深層CV**に踏み込みます。最初のテーマは、画像分類の二大バックボーンである **CNN(ResNet)** と **ViT(Vision Transformer)** を「概念」と「実装」の両面から理解し、そのうえで**事前学習済みの重みを活かして少ないデータ・少ない計算で新しいタスクに適応させる**——すなわち**転移学習(transfer learning)**を自分の手で書けるようになることです。GPU がなくても心配はいりません。CPU だけで、数十秒のうちに「学習前は当てずっぽう、学習後はほぼ正解」という転移学習の威力を再現します。

到達点は4つです。第一に、**ResNet（畳み込み＋残差接続）と ViT（パッチ埋め込み＋CLSトークン）**の発想の違いを言葉で説明できること。第二に、HuggingFace の **`pipeline` で最短分類**を体験したうえで、その中身である **`AutoImageProcessor` + `*ForImageClassification` を手書き**で分解できること（`画像→pixel_values→logits→argmax→id2label` の一連）。第三に、**torchvision / timm / HuggingFace の3つのエコシステム**から事前学習重みをロードし、分類ヘッドを外して**埋め込み（特徴ベクトル）**を取り出せること。第四に、**バックボーンを凍結（`requires_grad_(False)`）して新しい `nn.Linear` ヘッドに付け替え**、合成データで微調整し、**top-1/top-5 accuracy・混同行列・macro-F1** で素のモデルと比較評価できることです。

本章のスクリプトはすべて、ネット接続もデータセットも不要で完走するように、**色×形の幾何図形を合成生成**して題材とします（「赤い丸」「青い三角」など、人にも機械にも区別しやすい9クラス）。モデル重み（合計150MB程度）だけは初回に HuggingFace / torchvision からダウンロードしてローカルにキャッシュしますが、それ以外はすべてオフラインで動きます。実写真で試したい人向けには、`data/13_classification_transfer_learning/` に画像を置けば自動でそちらを使う導線も用意しました。なお本章は **transformers 5.x（v5）の正準API**で統一しているため、古いブログの `AutoFeatureExtractor` 系コードとは書き方が異なります。この点には最初に注意してください（詳細は第11節）。

---

## 1. 転移学習という発想 — なぜ事前学習が効くのか

深層学習の分類モデルをゼロから学習させるには、ふつう数十万〜数百万枚のラベル付き画像と、GPU での長時間学習が要ります。しかし現実のプロジェクトでそんな贅沢は稀で、手元にあるのは「数十〜数百枚の自前データ」ということがほとんどです。そこで効くのが**転移学習**——すなわち**巨大データ(ImageNet 等)で先に鍛えたモデルの「視覚特徴」を流用し、最後の数層だけを自分のタスクに合わせて学習し直す**やり方です。本章は、この「事前学習＋微調整」という現代CVの最も実用的な定石を体得する回です。

なぜ、こうした流用が成り立つのでしょうか。CNN や ViT が ImageNet で学習する過程では、浅い層が「エッジ・色・コーナー」、中間層が「テクスチャ・模様」、深い層が「物体パーツ」といった**汎用的で再利用可能な視覚特徴**を獲得することが知られています。これらは犬や猫を見分けるためだけの特徴ではなく、「画像とはどういうものか」という一般的な知識にほかなりません。したがって、新しいタスクのクラスがその特徴空間の中で**線形に分離できる**なら、最後の線形層（分類ヘッド）をほんの少し学習するだけで高精度が出ます。本章の `03_transfer_finetune.py` は、まさにこれを数値で見せます——凍結した ResNet 特徴の上に乗せた**乱数初期化のヘッドは正解率 0.111（9クラスの偶然＝1/9）**にすぎませんが、**わずか30ステップ学習しただけで 1.000** まで跳ね上がります。

転移学習には、大きく2つの流儀があります。ひとつは本章が主に扱う**特徴抽出(feature extraction)**で、バックボーンを完全に凍結し、新しいヘッドだけを学習します。データが少ない・計算資源が乏しいときの第一選択であり、過学習しにくく高速です。もうひとつは**ファインチューニング(fine-tuning)**で、バックボーンも含めた全体を（ただし**バックボーンには小さい学習率、ヘッドには大きい学習率**という「学習率の段差」をつけて）微調整します。こちらはデータが比較的多く、タスクが ImageNet と離れているときに効きますが、壊しすぎ（事前学習特徴の破壊）には注意が要ります。迷ったら「まず特徴抽出、足りなければファインチューニング」——これが実務の順序です。

## 2. CNN(ResNet)とViT — 2つのバックボーンの仕組み

**ResNet** は CNN（畳み込みニューラルネット）の代表格です。畳み込みとは「小さなフィルタを画像上で滑らせて局所パターンを検出する」操作で、層を重ねるほど広い範囲の抽象的な特徴を捉えます。ただし、層を深くするほど勾配が消失して学習が進まなくなる——この壁を破ったのが ResNet の**残差接続(residual/skip connection)**でした。各ブロックを `出力 = F(x) + x` の形にし、「変換 `F(x)` をまるごと学ぶ代わりに、入力からの差分だけを学べばよい」ようにしたのです。これにより100層を超える深いネットも安定して学習できるようになり、ResNet は今なお最も使われる CNN バックボーンとなっています。本章では最小構成の **ResNet-18**（約1170万パラメータ、最終手前の特徴は512次元）を使います。

一方の **ViT(Vision Transformer)** は、自然言語処理の Transformer を画像に持ち込んだ発想です。まず画像を 16×16 などの**パッチ**に分割し、各パッチを1つの「トークン（単語のようなもの）」とみなして系列にします。224×224 をパッチ16で切れば 14×14＝196 個のパッチになり、その先頭に分類用の特殊トークン **[CLS]** を1つ足して、計197トークンを Transformer に通します。各トークンには、「画像のどこにあったか」を表す**位置埋め込み(position embedding)**が加わります。Transformer の**自己注意(self-attention)**は全パッチ同士の関係を一度に見るため、CNN のような局所性の制約がなく、大域的な関係を捉えやすいのが特徴です。そして最終的に **[CLS] トークンの表現**を取り出して分類します。本章では軽量な **ViT-tiny**（hidden 192次元）を使います。

では、両者をどう使い分ければよいのでしょうか。勘所はこうです。**CNN(ResNet)は局所性という強い帰納バイアスを持ち、少〜中規模データでも安定**して学習でき、推論も軽い。一方の**ViT はバイアスが弱い分、大量データ（や強い事前学習）があれば CNN を上回る**ことが多く、注意機構によって解釈もしやすい。とはいえ、CPU 推論の軽さや実装の枯れ具合では ResNet がいまだ有利な場面が多く、本章でも転移学習の主役は ResNet-18 にしています。`02_resnet_vit_manual.py` は、同じ合成画像を ResNet と ViT の両方に通し、出力（`pixel_values`/`logits` の形、予測ラベル）を並べることで、2つのバックボーンが**同じ「画像→logits→argmax」の枠組み**で動くことを体感させます。

## 3. 高レベルAPI `pipeline` で最短分類（`01_pipeline_classify.py`）

まずは成功体験から始めましょう。HuggingFace の `pipeline` は、**前処理・推論・後処理を1行にまとめた最高レベルのAPI**です。`task="image-classification"` を指定してモデルIDを渡し、あとは PIL 画像を入れるだけで、`[{"label": ..., "score": ...}, ...]` がスコア降順で返ってきます。内部では、`AutoImageProcessor` がリサイズ・正規化を行い、モデルが推論し、`softmax` と `id2label` で人が読めるラベルに直す——という一連を、すべて肩代わりしてくれます。最初は中身を気にせず、「動く」ことそのものを味わうのが目的です。

```python
from transformers import pipeline
clf = pipeline("image-classification", model="microsoft/resnet-18", device=device, top_k=5)
results = clf(pil_image)   # [{'label': 'envelope', 'score': 0.87}, ...] が score 降順で返る
```

`device` には `torch.device`（`"cpu"`/`"mps"`/`"cuda"`）をそのまま渡せ、`top_k` で上位何件を返すかを指定できます。ここで注意したいのは、本章の入力が**合成図形**である点です。ImageNet の1000クラスに「赤い丸」というラベルは存在しないため、出力は `envelope`(封筒) や `pick`(ピック) など**それっぽい別物**になります。とはいえ、これは異常ではありません——学ぶべきは「ラベルの正しさ」ではなく、「`pipeline` がどう動くか」という**仕組み**だからです。意味のある予測を見たければ、`data/13_classification_transfer_learning/` に実写真を置いて再実行してください（`load_demo_images` が自動でそちらを優先します）。

ただし `pipeline` は手軽な反面、前処理や後処理が**ブラックボックス**になりがちです。本講座のゴールは「AIの補助なしに自力でCVコードを書けること」ですから、便利さに甘えず、あえて**中身を分解**していきます。具体的には、次節から `pipeline` がやっていることを `processor` と `model` に分け、自分の手で書き下します。なお `01_pipeline_classify.py` は、4枚の画像を分類して上位5件を表示し、予測パネル（`01_pipeline_predictions.png`）と JSON（`01_pipeline_predictions.json`）を保存します。

## 4. `pipeline` を分解する — processor + model の手書き（`02_resnet_vit_manual.py`）

`pipeline` の正体は、「**前処理器(processor/transforms) ＋ モデル(model)**」という2部品です。この2つを分けて書けるようになると、入力テンソルの形を確認したり、途中の特徴を取り出したり、独自の後処理を挟んだりと、応用が一気に広がります。HuggingFace 版の正準フローは次の通りです。まず `AutoImageProcessor` が PIL 画像を `pixel_values`（正規化済みの `(1, 3, 224, 224)` テンソル）に変換し、続いて `*ForImageClassification` モデルがそれを `logits`（1000クラス分のスコア）へと変換します。

```python
from transformers import AutoImageProcessor, ViTForImageClassification
processor = AutoImageProcessor.from_pretrained("WinKawaks/vit-tiny-patch16-224")
model = ViTForImageClassification.from_pretrained("WinKawaks/vit-tiny-patch16-224").eval()

inputs = processor(images=img, return_tensors="pt")          # 前処理: resize→rescale→normalize
with torch.inference_mode():                                 # 推論は勾配を切る（CPUで必須の作法）
    logits = model(**inputs).logits                          # (1, 1000)
idx = int(logits.argmax(-1).item())
label = model.config.id2label[idx]                           # インデックス→ラベル名に変換
```

ここで `AutoImageProcessor` が内部で行っているのは、**リサイズ → 0-1 への rescale → チャンネルごとの正規化 `(x - mean)/std`** の3段階です（演習2でこの中身を手で再現します）。`return_tensors="pt"` で PyTorch テンソルを受け取り、推論は **`model.eval()` ＋ `torch.inference_mode()`** で囲むのが鉄則です。これを忘れると無駄な勾配計算でメモリと時間を浪費し、特にCPUでは致命的になります。また `argmax` で得た**生のインデックスは人には読めない**ので、必ず `model.config.id2label[idx]` で `envelope` のようなラベル名に直しましょう。`02_resnet_vit_manual.py` は、torchvision の ResNet-18 と HF の ViT-tiny で同じ画像を分類し、`pixel_values` と `logits` の形を表示することで、CNN と ViT が同じ枠組みで動くことを示します。

一方 torchvision の場合は、`processor` の代わりに**重みに紐づく前処理オブジェクト**を使います。具体的には、`ResNet18_Weights.DEFAULT.transforms()` が正準な前処理（resize/centercrop/normalize、内部は `transforms.v2`）を返し、`weights.meta["categories"]` が ImageNet-1k の1000ラベルを保持します。このように、HuggingFace では `processor` と `id2label` がモデルに、torchvision では前処理とラベルが `weights` オブジェクトに同梱されており、**「前処理とラベルは必ずモデル/重みとセットで管理する」**という発想は両者で共通です。手打ちで平均/分散をズラすと精度が静かに落ちるので、必ず付属の前処理を使ってください。

## 5. 3つのエコシステム（torchvision / timm / HuggingFace）の使い分け

事前学習モデルの入手先は主に3つあり、それぞれ流儀が違います。まず **torchvision** は PyTorch 公式で、`weights` API（`ResNet18_Weights.DEFAULT`）が「重み・前処理・ラベル」を一体で提供する堅実な選択肢です。次に **timm**（PyTorch Image Models）は、ResNet/ViT/EfficientNet/MobileNet など**膨大な画像モデルを統一API**で扱え、CPU向け軽量モデルの宝庫となっています。そして **HuggingFace transformers** はマルチモーダルまで含む巨大エコシステムで、`pipeline` や `AutoModel` の統一インターフェース、Hub での共有が強みです。それぞれの要点を下表にまとめます。

| エコシステム | モデル作成 | 前処理 | ラベル | 強み |
| --- | --- | --- | --- | --- |
| torchvision | `resnet18(weights=ResNet18_Weights.DEFAULT)` | `weights.transforms()` | `weights.meta["categories"]` | 公式・枯れている・依存が軽い |
| timm | `timm.create_model("resnet18", pretrained=True)` | `create_transform(**resolve_data_config(...))` | （別途）`num_classes` で制御 | モデル数が圧倒的・軽量モデル豊富 |
| HuggingFace | `ViTForImageClassification.from_pretrained(id)` | `AutoImageProcessor.from_pretrained(id)` | `model.config.id2label` | 統一API・Hub共有・マルチモーダル |

実務での使い分けはこうです。**「とにかく定番のCNN/ViTを手堅く」なら torchvision**、**「珍しいモデルや極小モデルを試したい」なら timm**、**「pipeline で手早く、あるいは CLIP など別タスクと統一的に」なら HuggingFace**、と考えるとよいでしょう。本章はこの3つすべてを通すことで、どれが来ても同じ「ロード→前処理→推論」の型で対応できる感覚を養います。なお timm では、前処理を**手打ちせず** `resolve_data_config` ＋ `create_transform` で**モデル固有の入力サイズ・平均/分散を自動生成**するのが事故防止の定石です（モデルごとに正しい正規化値が異なるため）。

バージョンに関する重要な注意点も挙げておきます。**transformers 5.x（v5）では、画像プロセッサが torchvision バックエンドの fast 実装のみ**になりました。そのため `AutoImageProcessor` を使うには torchvision が事実上必須であり（本章は `dl` グループで導入済み）、旧 `AutoFeatureExtractor` や `use_fast=` 引数は廃止されています。古い記事のコードをそのまま写すと動かないので、**画像は必ず `AutoImageProcessor`** と覚えてください。

## 6. 埋め込みの取り出し — penultimate / pooler / CLS / forward_features

分類モデルは「画像→logits（クラス確率）」を出力しますが、**最後の分類層を外せば「画像→特徴ベクトル（埋め込み）」**としても使えます。そしてこの埋め込みこそ、第15回のメトリック学習、第16回の CLIP 検索、第17回の FAISS ベクトル検索の土台となるものです。そこで本章のうちに、取り出し方の選択肢を体得しておきましょう。`02_resnet_vit_manual.py` は3通りの取り出し方を実演し、それぞれの形（次元）を表示します。

```python
# torchvision: 最終 fc を Identity に差し替え → penultimate(512次元)が出力になる
model.fc = nn.Identity()
emb = model(x).squeeze(0)                        # (512,)

# timm: num_classes=0 で分類ヘッドを外す → global average pooling 済みの埋め込み
emb_model = timm.create_model("resnet18", pretrained=True, num_classes=0)
pooled = emb_model(x)                            # (1, 512)  プール済みベクトル
feat_map = emb_model.forward_features(x)         # (1, 512, 7, 7)  未プーリングの特徴マップ

# HuggingFace ViT: base モデルの last_hidden_state から CLS / mean-pool を取る
out = vit_base(**inputs)                         # last_hidden_state: (1, 197, 192)
cls = out.last_hidden_state[:, 0]                # (1, 192)  先頭の CLS トークン
mean = out.last_hidden_state[:, 1:].mean(dim=1)  # (1, 192)  パッチトークンの平均
```

ただし、ここには**3つの落とし穴**があります。第一は、**`pooler_output` と `last_hidden_state` の形の違い**です。ViT の `last_hidden_state` は `(B, 1+196, hidden)` のトークン列、`pooler_output` は `(B, hidden)` の集約ベクトル、ResNet系の `last_hidden_state` は `(B, C, H, W)` の特徴マップと、中身がまったく違います。第二は、**timm で `num_classes=0` を忘れると 1000次元のロジットが返り、埋め込みと誤用**してしまう点です（必ず `num_classes=0` か `forward_features` を使います）。第三は、本章で実際に遭遇したように、**ViT-tiny の `pooler` がチェックポイントに含まれず、乱数初期化のままになっていることがある**点です（ロード時に `MISSING` 警告が出ます）。この場合 `pooler_output` は無意味なので、埋め込みには**学習済みの CLS トークンか mean-pool を使う**のが安全です。

埋め込みが意味を持つことを確かめるため、`02` の最後では**コサイン類似度**を計算します。L2 正規化してから内積を取るのがコサイン類似度で、実際に計算すると、同じクラス（赤い丸×2）の類似度が **0.97**、別クラス（赤い丸 vs 青い三角）が **0.69** となり、**同クラスの方が明確に高い**ことが確認できます（`02_embedding_cosine.png`）。この「似た画像は近く、違う画像は遠い」という性質こそ、検索・クラスタリングの根幹です。なお、埋め込みをコサイン類似度に使うときは**必ず L2 正規化**が要る点を覚えておいてください（第16回で、CLIP の `get_image_features` が未正規化という落とし穴として再登場します）。

## 7. 転移学習の実装 — 凍結 + ヘッド付け替え（`03_transfer_finetune.py`）

いよいよ本章の成果物です。やることは、次の3ステップに集約されます。**(1) ImageNet 学習済み ResNet-18 をロードし、最終 fc を `nn.Identity()` で外して特徴抽出器にする。(2) 全パラメータを `requires_grad_(False)` で凍結する。(3) 9クラス用の新しい `nn.Linear(512, 9)` ヘッドを付ける**。これで「凍結された汎用特徴 ＋ 学習可能な小さなヘッド」という転移学習モデルが完成します。コードの核心は次の通りです。凍結できたかどうかは、「学習可能パラメータ数が 0 か」で確認できます。

```python
backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
backbone.fc = nn.Identity()                       # 1000クラス分類層を外す → 512次元が出力に
for param in backbone.parameters():
    param.requires_grad_(False)                   # 特徴抽出器として完全凍結
head = nn.Linear(512, NUM_CLASSES)                # 新しい分類ヘッド（これだけ学習する）
```

CPU で高速に回すための工夫も入れてあります。**凍結バックボーンの出力は二度と変わらない**ことを利用し、全画像の特徴を**一度だけ前計算してキャッシュ**するのです。以降は軽い `nn.Linear` だけを学習するので、126枚の特徴抽出も30ステップの学習も一瞬で終わります（実務でも頻出の最適化です）。ただし、ここにひとつ実装上の罠があります——特徴抽出を `torch.inference_mode()` で行うと「推論専用テンソル」になり、**後でヘッドを学習する際の autograd に入力できずエラー**になるのです。そのため、前計算した特徴を学習に使う場合は、`torch.inference_mode()` ではなく **`torch.no_grad()`** を使ってください（no_grad のテンソルは通常テンソルとして扱えます）。本章のコードも、この点を踏まえて `no_grad` にしてあります。

```python
optimizer = torch.optim.Adam(head.parameters(), lr=1e-2)   # 学習対象は head だけ
for step in range(30):
    optimizer.zero_grad()
    loss = loss_fn(head(f_train), y_train)                 # キャッシュ特徴 → ヘッド → 損失
    loss.backward(); optimizer.step()
```

なお全体ファインチューニングをしたい場合は、`optimizer` に**パラメータグループ**を渡し、`backbone` には小さい学習率（例 1e-5）、`head` には大きい学習率（例 1e-3）を与えます。これが冒頭で触れた**「学習率の段差」**であり、事前学習特徴を壊さずに微修正するための定石です（本章はコードのコメントで触れるに留め、まずは凍結＝特徴抽出で転移学習の本質を掴みます）。

## 8. 評価 — top-1/top-5・混同行列・macro-F1（torchmetrics）

転移学習が「効いた」かどうかは、**数値で示す**——これが本講座の方針です。まずは評価指標の定義から押さえましょう。**top-1 accuracy** は最もスコアの高い予測が正解と一致した割合、**top-5 accuracy** は上位5予測のどれかに正解が含まれる割合です（クラス数が多いほど意味を持つ指標で、9クラスでは易しめになります）。**macro-F1** はクラスごとの F1（precision と recall の調和平均）を**単純平均**したもので、各クラスを平等に評価するためクラス不均衡に強い指標です。そして**混同行列**は行=正解・列=予測の集計表で、対角が大きいほど良く、どのクラスをどのクラスと取り違えたかが一目で分かります。これらはいずれも torchmetrics で計算します。

```python
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score, MulticlassConfusionMatrix
acc1 = MulticlassAccuracy(num_classes=9, average="micro")            # top-1
acc5 = MulticlassAccuracy(num_classes=9, average="micro", top_k=5)   # top-5
f1   = MulticlassF1Score(num_classes=9, average="macro")
cm   = MulticlassConfusionMatrix(num_classes=9)
top1, top5, mf1, conf = acc1(logits, y), acc5(logits, y), f1(logits, y), cm(logits, y)
```

実装上の注意点をひとつ挙げておくと、**`top_k` を使う指標には、argmax 後のラベルではなく `logits`（スコア列）を渡す**必要があります（上位kを計算するため）。さて `03_transfer_finetune.py` を実行すると、**学習前（乱数ヘッド）の top-1 = 0.111（＝1/9 の偶然）から、30ステップ学習後には top-1 = 1.000・macro-F1 = 1.000** へと跳ね上がる様子が表示され、混同行列（`03_confusion_matrix.png`）は完全な対角になります。下表がその比較で、まさに「凍結した汎用特徴の上に、わずかな線形層を学習しただけ」で分類が成立したことを示しています。

| モデル | top-1 | top-5 | macro-F1 | 解釈 |
| --- | --- | --- | --- | --- |
| 学習前（乱数ヘッド） | 約0.11 | 約0.55 | 約0.03 | 9クラスの偶然（1/9）とほぼ同じ＝何も学べていない |
| 学習後（30ステップ） | 1.00 | 1.00 | 1.00 | 凍結特徴＋線形ヘッドだけで完全分類 |

今回は合成図形の色と形がはっきり分かれているため精度が満点になりますが、**大事なのは絶対値ではなく、「偶然レベル→高精度への跳躍」というパターン**です。実データはもっと難しく、混同行列に取り違えが現れます（本章でも `noise_std` を大きくすれば誤りが出ます）。なお、クラス別の precision/recall/F1 は scikit-learn の `classification_report` でも出力しているので（第14回への布石）、「accuracy だけでなく、クラス別・複数指標で見る」習慣をここで付けておきましょう。評価指標の体系的な扱いは、第14回で深掘りします。

## 9. このモジュールの構成（スクリプト一覧）

各スクリプトは単一責務で、上から順に「最短で動かす → 中身を分解する → 転移学習で仕上げる」と、理解が積み上がるように並んでいます。いずれも `outputs/13_classification_transfer_learning/` に図と JSON を保存し、画面表示には依存しません。また共通部品（device 判定・合成データ生成・保存）は `dl_helpers.py` にまとめてあり、各スクリプトはこれを import して使います。深層CVトラックの最初の回なので、`dl_helpers.get_device()` の device 判定ロジックは、以降の回でもそのまま再利用できます。

| ファイル | 役割（単一責務） |
| --- | --- |
| `dl_helpers.py` | `get_device`・合成データ生成（色×形9クラス）・`load_demo_images`(data/優先)・図の保存。道具箱 |
| `01_pipeline_classify.py` | `pipeline("image-classification")` で最短分類、top-k 表示、予測パネル/JSON 保存 |
| `02_resnet_vit_manual.py` | processor+model 手書き（CNN vs ViT）、埋め込み取り出し（penultimate/CLS/forward_features）、コサイン類似度 |
| `03_transfer_finetune.py` | 凍結＋ヘッド付け替えで転移学習、特徴キャッシュ、top-1/top-5・混同行列・macro-F1 で評価 |
| `mini_project.py` | 章末ミニプロジェクト：3バックボーン横並びベンチ（凍結特徴→線形プローブ／重心分類／近傍検索）。比較図・混同行列・検索図・JSON を保存 |
| `use_case.py` | 実践ユースケース：**フォルダ＝クラス**の少数枚から自前分類器を学習・**.pt 保存**・推論する小ツール（凍結特徴＋線形プローブ）。`data/` に画像があれば実データ、無ければ合成で完走 |
| `exercises.py` | TODO形式の演習9問（自己採点ランナー付き。`SHOW_SOLUTION=1` で模範解答） |
| `exercises_solutions.py` | 演習の完全な模範解答（実行すると全9問 PASS。採点は `exercises.py` 側を再利用） |

表の通り、`dl_helpers.py` だけは「読み物」ではなく「再利用する道具」です。そのため最初に一読してから 01 へ進むと、各スクリプトが何を import しているかが腑に落ちます。とりわけ、合成データ生成（`make_shape_image` / `make_dataset`）が分類デモ・埋め込み・転移学習のすべての練習台になっている点に注目してください。

## 10. 動かし方

このモジュールは、`dl`（torch・torchvision）と `hf`（transformers・timm ほか）、そして評価用の `metrics`（torchmetrics・scikit-learn）という依存グループを使います。GPU は不要で、**初回のみ**モデル重み（ResNet-18×2系統・ViT-tiny、合計150MB程度）をダウンロードしてキャッシュします。準備ができたら、プロジェクトルートで以下を順に実行してください。

```bash
# 依存をインストール（初回のみ）。本章は dl / hf / metrics グループが必要。
uv sync --group dl --group hf --group metrics

# 各スクリプトを実行（結果は outputs/13_classification_transfer_learning/ に保存される）
uv run python lectures/13_classification_transfer_learning/01_pipeline_classify.py
uv run python lectures/13_classification_transfer_learning/02_resnet_vit_manual.py
uv run python lectures/13_classification_transfer_learning/03_transfer_finetune.py

# 章末ミニプロジェクト（3バックボーン横並びベンチ。CPU で十数秒）
uv run python lectures/13_classification_transfer_learning/mini_project.py

# 実践ユースケース（フォルダ＝クラスで自前分類器を学習・保存・推論。data/ が空でも合成で完走）
uv run python lectures/13_classification_transfer_learning/use_case.py

# 演習: まずは TODO を自分で埋める（最初は全部 FAIL でも exit 0）
uv run python lectures/13_classification_transfer_learning/exercises.py
# どうしても分からない時だけ、模範解答の挙動を見る
SHOW_SOLUTION=1 uv run python lectures/13_classification_transfer_learning/exercises.py
# 完全な模範解答（実行すると全9問 PASS）
uv run python lectures/13_classification_transfer_learning/exercises_solutions.py

# （任意）実写真で試す: data/13_classification_transfer_learning/ に *.jpg/*.png を置いて 01/02 を再実行
```

実行を終えたら、`outputs/13_classification_transfer_learning/` の画像を解説と照らし合わせてください。特に `02_embedding_cosine.png`（同クラスは類似度が高い）と `03_confusion_matrix.png`（学習後は完全な対角）を見ると、本章の2大テーマ（**埋め込みの意味**と**転移学習の威力**）が視覚的に腑に落ちます。なお、初回にダウンロードしたモデルは `~/.cache/huggingface`（HF）と `~/.cache/torch/hub`（torchvision）にキャッシュされ、2回目以降はオフラインで即起動します。Docker で使う場合は、このキャッシュをボリュームマウントすると毎回の再DLを防げます。

## 11. よくあるエラーと対処（チェックリスト）

最後に、本章（とくに transformers 5.x）でつまずきやすい点を、「症状 → 原因 → 対処」の形でまとめます。深層CVは環境とAPIの罠が多いので、詰まったらまずここを見てください。

| 症状 | ほぼ確実な原因 | 対処 |
| --- | --- | --- |
| `AutoFeatureExtractor` が無い/動かない | transformers 5.x で廃止された | `AutoImageProcessor` を使う（画像は常にこれ） |
| `AutoImageProcessor` がエラーで作れない | torchvision 未導入（v5 は fast 実装のみ） | `uv sync --group dl` で torchvision を入れる |
| `RuntimeError: ... device` で落ちる | model と inputs の device がズレている | `inputs.to(model.device)` で必ず揃える |
| `Inference tensors cannot be saved for backward` | `inference_mode` で作った特徴を学習に使った | 前計算は `torch.no_grad()` で行う（clone でも可） |
| 予測ラベルが整数のまま読めない | `id2label` で変換していない | `model.config.id2label[idx]` でラベル名に直す |
| 埋め込みが1000次元になる | timm で `num_classes=0` を忘れた | `num_classes=0` か `forward_features` を使う |
| `pooler_output` が無意味な値 | そのチェックポイントの pooler が未学習(MISSING) | CLS トークンか mean-pool を埋め込みに使う |
| CPU 推論が異常に遅い | float16/half を使った、勾配を切っていない | CPUは float32、推論は `eval()`+`inference_mode()` |
| 図中の日本語が豆腐(□)になる | 既定フォントにCJKグリフが無い | 図中の文字はASCIIにする（本章はそうしている） |

とりわけ上3つ（`AutoFeatureExtractor` 廃止・torchvision 必須・device 揃え）は、transformers 5.x の「あるある」です。古いブログのコードを写経して動かないときは、まずこの3点を疑ってください。

## 12. まとめ

本章では、画像分類の二大バックボーン（CNN＝ResNet、ViT）の仕組みを概念と実装の両面から押さえ、HuggingFace の `pipeline` で最短分類を体験したうえで `AutoImageProcessor` + `*ForImageClassification` に分解し、torchvision/timm/HuggingFace の3エコシステムから事前学習重みをロードして推論・埋め込み取り出しを行い、最後に**バックボーン凍結＋ヘッド付け替え**による転移学習を合成データで実装して、top-1/top-5・混同行列・macro-F1 で「学習前の偶然レベル→学習後の高精度」という跳躍を数値で確認しました。これらすべてに通底するのは、「**汎用的な事前学習特徴を流用し、少しの学習で新タスクに適応する**」という、現代CVの最も実用的な発想です。

ここで身につけた「埋め込みを取り出す」「コサイン類似度で似た画像を測る」「凍結＋ヘッド学習」という3つの道具は、第14回（評価指標）、第15回（メトリック学習）、第16回（CLIP ゼロショット）、第17回（FAISS 検索）へとそのまま続いていきます。まずは演習を全問 PASS させ、`03_confusion_matrix.png` の対角が意味すること（＝転移学習が効いたこと）を自分の言葉で説明できるようにしてから、次へ進んでください。

---

## 🛠 章末ミニプロジェクト — バックボーン比べ × 転移学習 × 画像検索

本章で別々に学んだ部品（**埋め込みの取り出し**・**凍結＋線形ヘッドの転移学習**・**複数指標での評価**・**コサイン類似度**）を、**1 本のパイプライン**に統合する総合課題です。`mini_project.py` を実行すると、合成図形9クラス(色×形)を題材に、**3 つの事前学習バックボーンを「特徴抽出器」として横並びでベンチマーク**します。

対象バックボーンは `resnet18_tv`（torchvision・penultimate 512次元）・`resnet18_timm`（timm・pooled 512次元）・`vit_tiny_cls`（HuggingFace ViT-tiny の CLS トークン 192次元）の3つ。各バックボーンについて次を一気通貫で実施します。

1. **凍結特徴の前計算**: 学習用・テスト用・「**強ノイズ**」テスト用の画像を埋め込みに変換する。ここは `inference_mode` ではなく `no_grad`（後で線形ヘッド学習に使うため。第7節の落とし穴）。
2. **線形プローブ（linear probe）**: 凍結特徴の上に `nn.Linear` だけを学習する＝転移学習の本質。
3. **評価**: `top-1` と `macro-F1` を、**クリーンなテスト**と**ノイズを強めたテスト**の両方で計測する（学習時より厳しい分布へどれだけ頑健か＝分布シフト耐性）。
4. **学習不要ベースライン**: 各クラスの特徴の重心（centroid）への**コサイン類似度で最近傍分類**した精度を出す。「埋め込みが既にクラスを分離していれば、ヘッドを学習しなくてもそこそこ当たる」を体感する。
5. **画像検索デモ**: ベストなバックボーンの埋め込みで、クエリ画像に近い学習画像を**コサイン類似度で上位3件**取り出す（第16〜17回への布石）。

実行すると、**線形プローブはどのバックボーンもクリーンでほぼ満点**になる一方、**強ノイズ下では差が開き**、また**学習ゼロの重心分類でも高い精度が出る**（＝凍結特徴がそのままクラスを分離している）という、本章の主張が数値で立ち上がります。出力は `outputs/13_classification_transfer_learning/` に保存されます。

| 生成物 | 内容 |
| --- | --- |
| `mini_project_backbone_comparison.png` | 3バックボーン × （クリーン線形プローブ／強ノイズ線形プローブ／学習不要の重心分類）の棒グラフ |
| `mini_project_confusion_best.png` | ベストバックボーンの線形プローブ混同行列（クリーンテスト） |
| `mini_project_retrieval.png` | クエリ画像と、コサイン近傍トップ3の学習画像（各近傍のクラス名・cos 値・正誤付き） |
| `mini_project_metrics.json` | 各バックボーンの埋め込み次元・抽出時間・top-1/macro-F1・強ノイズ精度・重心精度・検索 precision@1 の数値ログ |

```bash
uv run python lectures/13_classification_transfer_learning/mini_project.py
cat outputs/13_classification_transfer_learning/mini_project_metrics.json
```

**発展のヒント**: バックボーンを増やす（`mobilenetv3_small_100` などの極小モデルを timm から追加）／`HARD_NOISE` を上げて頑健性の差を強調する／`data/13_classification_transfer_learning/` に実写真を置いて「自分のデータでの少データ分類」に置き換える。これらはどれも数行の変更で試せます。

## ✅ 到達チェックリスト

この章を終えたら、次が**できる／説明できる**ことを確認してください。

- [ ] **ResNet（畳み込み＋残差接続）と ViT（パッチ埋め込み＋CLSトークン）**の発想の違いを、帰納バイアスとデータ量の観点で説明できる。
- [ ] `pipeline("image-classification")` で最短分類を動かし、`top_k` と `device` の意味を説明できる。
- [ ] `pipeline` を **`AutoImageProcessor` + `*ForImageClassification` に分解**し、`画像→pixel_values→logits→argmax→id2label` を手書きできる。
- [ ] `AutoImageProcessor` の中身（**リサイズ → 0-1 rescale → ImageNet 正規化**）を numpy で再現できる（演習2）。
- [ ] **torchvision / timm / HuggingFace** の3エコシステムから事前学習重みをロードし、「前処理とラベルは重み/モデルとセット」という共通発想を説明できる。
- [ ] 分類ヘッドを外して**埋め込み**を取り出せる（torchvision `fc=Identity`／timm `num_classes=0`・`forward_features`／ViT の `CLS`・mean-pool）。`pooler_output` と `last_hidden_state` の**形の違い**も言える。
- [ ] **バックボーン凍結（`requires_grad_(False)`）＋ 新しい `nn.Linear` ヘッド**で転移学習モデルを組み、学習対象がヘッドだけになっていることをパラメータ数で確認できる（演習3）。
- [ ] 前計算した特徴を学習に使うとき **`inference_mode` ではなく `no_grad`** を使う理由を説明できる。
- [ ] **top-1 / top-5 accuracy・macro-F1・混同行列**を計算し、`top_k` 指標には argmax 後のラベルではなく `logits` を渡すと言える（演習4・7・8）。
- [ ] **コサイン類似度＝L2正規化してから内積**を実装でき、同クラスが近く別クラスが遠いことを示せる（演習5・9）。
- [ ] ミニプロジェクトを実行し、**「凍結特徴 → 線形プローブ／重心分類／近傍検索」**を複数バックボーンで横並び評価できる。

## ❓ よくある落とし穴・FAQ・デバッグ

実装中に詰まったら、まずここを見てください（第11節の症状別表と併せて参照）。多くの不具合は、transformers 5.x のAPI変更と「device / 勾配 / 正規化」の3点に集約されます。

- **Q. `AutoFeatureExtractor` が無い／古いブログのコードが動かない。** A. transformers 5.x で**廃止**されました。画像は常に `AutoImageProcessor` を使います。`use_fast=` 引数も消滅（fast 実装のみ）。
- **Q. `AutoImageProcessor` の生成でエラーになる。** A. v5 の画像プロセッサは **torchvision バックエンドの fast 実装のみ**です。`uv sync --group dl` で torchvision を入れてください。
- **Q. `RuntimeError: ... expected ... device` で落ちる。** A. **model と inputs の device がズレ**ています。`inputs.to(model.device)` で必ず揃えます（片方だけ `.to()` が原因の筆頭）。
- **Q. `Inference tensors cannot be saved for backward` が出る。** A. **`inference_mode` で作った特徴を学習に使った**ためです。前計算は `torch.no_grad()` で行います（`mini_project.py` / `03` がこの定石）。
- **Q. 予測が整数のまま読めない。** A. `model.config.id2label[idx]`（torchvision は `weights.meta["categories"][idx]`）で**ラベル名に変換**します（演習6）。
- **Q. timm の埋め込みが 1000 次元になる。** A. `num_classes=0` を忘れて分類ロジットを取っています。`num_classes=0` か `forward_features` を使います。
- **Q. ViT の `pooler_output` が変な値／検索が崩れる。** A. vit-tiny の `pooler` は**未学習(MISSING)のことがある**ので、埋め込みには**学習済みの CLS トークンか mean-pool**を使います。CLIP の `get_*_features` 同様、コサイン類似の前に **L2 正規化**を忘れない。
- **Q. 合成図形なのに ImageNet ラベルが「封筒」など変なものになる。** A. 異常ではありません。合成図形は ImageNet の1000クラスに無いので「それっぽい別物」が出ます。学ぶのは**仕組み**であってラベルの正しさではありません。`data/` に実写真を置けば意味のある予測になります。
- **Q. CPU 推論が異常に遅い。** A. **float16/half を使った**か**勾配を切っていない**のが原因。CPU は `float32`、推論は `eval()` + `inference_mode()`（学習に再利用する特徴抽出だけ `no_grad`）。
- **Q. 図中の日本語が豆腐(□)になる。** A. 既定フォントに CJK グリフが無いためです。**図中の文字は ASCII** に保ちます（本章のスクリプトはそうしています）。
- **Q. ミニプロジェクトの精度が満点で「効いた」のか分からない。** A. 合成図形は色と形が明瞭なので満点になりがちです。本質は**絶対値ではなく「学習前の偶然レベル→学習後の高精度」「クリーン→強ノイズで差が開く」というパターン**です。`HARD_NOISE` を上げると差が見えます。

## 🚀 発展トピック・参考

- **全体ファインチューニング（learning-rate の段差）**: 本章は凍結＝特徴抽出が主役でしたが、データが多くタスクが ImageNet と離れていれば、バックボーンも含めて微調整します。`optimizer` に**パラメータグループ**を渡し、backbone に小さい lr（例 1e-5）、head に大きい lr（例 1e-3）を与えるのが定石です（事前学習特徴を壊さないため）。
- **線形プローブ vs k-NN/重心分類**: 凍結特徴の良し悪しを測る2大プロトコルです。線形プローブは表現の線形分離性を、k-NN/重心は局所構造を測ります。`mini_project.py` で両者を並べて出しているので、傾向を見比べてください。
- **データ拡張（第12回）との接続**: 少データ転移学習では、回転・色ジッタ・RandAugment 等の拡張が効きます。`albumentations` や `torchvision.transforms.v2` で学習画像を水増しすると、強ノイズ下の精度が改善することがあります。
- **EMA / 学習率スケジューラ / 早期終了**: 実データの微調整では `CosineAnnealingLR` などのスケジューラや早期終了で過学習を抑えます。本章のトイ設定では不要ですが、実務の必須要素です。
- **より新しい/軽量なバックボーン**: timm には `mobilenetv3_small_100`・`efficientnet_b0`・`vit_tiny_patch16_224`・ConvNeXt 系など CPU 向けの選択肢が豊富です。`timm.list_models(pretrained=True)` で探し、同じパイプラインに差し替えて比較できます。
- **次回以降への接続**: ここで得た「埋め込みを取り出す」「コサイン類似度で測る」「凍結＋ヘッド学習」は、第14回（評価指標）・第15回（メトリック学習）・第16回（CLIP ゼロショット）・第17回（FAISS 検索）へ直結します。
- 参考ドキュメント: HuggingFace transformers「Image classification」 https://huggingface.co/docs/transformers/tasks/image_classification ／ torchvision models https://pytorch.org/vision/stable/models.html ／ timm https://huggingface.co/docs/timm/index ／ He et al. (2015) "Deep Residual Learning"（ResNet）／ Dosovitskiy et al. (2020) "An Image is Worth 16x16 Words"（ViT）。

## 💡 実践ユースケース集

本章の「凍結特徴 + 線形プローブ」は、教材を超えて**現場でそのまま使える少データ分類の定石**です。事前学習バックボーンがすでに汎用的な視覚特徴を持っているので、最後の `nn.Linear` 1 枚を数十枚で学習するだけで、あなた固有のカテゴリに適応できます。ここでは現実の応用を 3 つ挙げ、そのうち最後の 1 つを、動く出発点 `use_case.py` として用意しました。

### ① 自分のカテゴリで画像分類器（`use_case.py` — 動く出発点）

- **何に使うか**: 「自社製品の良品/不良品」「現場写真のシーン仕分け」「手元の素材を数カテゴリに自動タグ付け」など、**自分で定義した少数クラス**を少数枚から分類したいとき。Teachable-Machine 的なツールの最小実装です。
- **作り方の要点**: `data/13_classification_transfer_learning/<クラス名>/` にクラスごとのフォルダを作り画像を入れる（**フォルダ名＝クラス名**）→ 凍結 ResNet-18 で全画像を 512 次元特徴に変換 → その上の `nn.Linear` だけを学習 → 学習済みヘッド＋クラス名を `use_case_classifier.pt` として保存 → 保存物を読み直して新規画像を推論（`softmax` の最大値を信頼度として表示）。
- **注意**: 実データが無い（クラスが 2 つ未満）ときは合成図形で必ず完走します（exit 0）。クラスは 2 つ以上・各 3〜5 枚もあれば動きますが、**枚数が極端に少ない/クラス間が似すぎる**と精度は伸びません。図のタイトルは ASCII 化してあるので、**日本語フォルダ名でも豆腐(□)になりません**（コンソール出力は日本語のまま表示）。

実行コマンドと拡張は次の通りです。

```bash
# 合成データで動作確認（data/ が空でもOK。CPUで数十秒、初回のみResNet-18重みをDL）
uv run python lectures/13_classification_transfer_learning/use_case.py

# 自分のデータで使う: クラスごとのフォルダに画像を置いてから再実行
#   data/13_classification_transfer_learning/
#     ├── cat/ img1.jpg img2.png ...   ← フォルダ名がクラス名
#     ├── dog/ ...
#     └── _inbox/ unknown1.jpg ...     ← (任意) ラベル無しの「分類したい画像」置き場
uv run python lectures/13_classification_transfer_learning/use_case.py
cat outputs/13_classification_transfer_learning/use_case_metrics.json
```

**拡張アイデア**: ①クラスを増やす＝フォルダを足すだけ（コード変更不要）／②`BACKBONE` を timm の `mobilenetv3_small_100` などに差し替えて速度・精度を比較／③学習画像に `torchvision.transforms.v2` の回転・色ジッタを足して頑健化／④`softmax` 信頼度が低い入力は `"unknown"` を返す「オープンセット」化／⑤保存した `.pt` を読み込み新クラスを足して線形層だけ再学習する増分学習／⑥`load_classifier()` + `predict()` を FastAPI から呼んで分類 API 化。

### ② 学習不要の最近傍分類（重心 / k-NN で即タグ付け）

- **何に使うか**: 線形ヘッドすら学習したくない・**1 クラス数枚しか集まらない**ような超少データのとき。各クラスの特徴の平均（重心）を作り、新画像はコサイン類似が最大の重心へ割り当てます。
- **作り方の要点**: `use_case.py` の `extract_features` で各クラスの埋め込みを集め、L2 正規化して平均＝重心を作る → クエリ特徴との内積（コサイン類似）が最大のクラスを予測。`mini_project.py` の `nearest_centroid_accuracy` がそのまま雛形になります。
- **注意**: コサイン類似の前に**必ず L2 正規化**します。重心法はクラス内のばらつきが大きいと崩れやすいので、その場合は重心ではなく全サンプルとの k-NN に切り替えます。

### ③ 画像検索・重複/近傍さがし（埋め込みインデックス）

- **何に使うか**: 「この画像に似たものを手持ちから探す」「ほぼ重複した写真を見つける」など、分類ではなく**類似検索**が欲しいとき。素材管理・モデレーション・データ整理で多用します。
- **作り方の要点**: 全画像を凍結特徴に変換して保存（インデックス化）→ クエリ画像の特徴とコサイン類似度で上位 K 件を返す。`mini_project.py` の検索デモがそのまま出発点で、規模が大きくなったら第17回の FAISS に置き換えます。
- **注意**: バックボーンと前処理は**インデックス作成時と検索時で必ず同一**にします（揃っていないと特徴空間がズレて検索が無意味になります）。次元数とメモリ量はバックボーンで決まる（ResNet-18 は 512 次元）ので、大規模では軽量モデルや次元圧縮を検討します。

---

> 本教材で参照・検証したライブラリとバージョン（2026-06 時点の安定版で動作確認）:
> Python 3.12 ／ torch 2.12.0+cpu ／ torchvision 0.27.0+cpu ／ transformers 5.11.0 ／ timm 1.0.27 ／ huggingface_hub 1.18.0 ／ safetensors 0.8.0 ／ torchmetrics 1.9.0 ／ scikit-learn 1.9.0 ／ numpy 2.4.6 ／ Pillow 12.2.0 ／ matplotlib 3.10.9 ／ faiss-cpu（第17回で使用）。使用モデル: `microsoft/resnet-18`・`WinKawaks/vit-tiny-patch16-224`・torchvision `ResNet18_Weights.DEFAULT`・timm `resnet18`（すべて CPU・初回のみ重みDL）。