# 第16回 CLIP/SigLIP によるゼロショット分類と画像テキスト検索

> トラック: **マルチモーダル** ／ レベル: **中級** ／ 依存グループ: `dl`（torch/torchvision）・`hf`（transformers/sentencepiece ほか）・`metrics`（torchmetrics）。CPU だけで完走します（初回のみモデル重みを HuggingFace からダウンロード）。

## 🎯 この章のゴール

第13回では事前学習モデルを「固定クラスの分類器」として転移学習し、第15回では画像を1本のベクトルへ埋め込む方法を学びました。本章のテーマは、その埋め込みを**画像と言語で共有**することです。CLIP（Contrastive Language–Image Pre-training）は、画像エンコーダとテキストエンコーダを「対になった画像・キャプションが近づくように」対照学習し、**画像と文章を同じ潜在空間の点**にします。同じ空間に乗っているからこそ、画像ベクトルと「赤い円」という文ベクトルの近さを測れて、**学習し直さずに任意のラベルで分類（ゼロショット）**でき、**文で画像を検索**できます。この章を終えると、その原理を腑に落とし、`CLIPProcessor`/`CLIPModel` でゼロショット分類と画像テキスト検索を、AI 補助なしで自分の手で書けるようになります。

具体的な到達点は4つです。第一に、`pipeline("zero-shot-image-classification")` で**まず動かして**から、それを `processor` と `model` に**分解して手書き**できること。第二に、`get_image_features`/`get_text_features` が返すベクトルは**未正規化**で、コサイン類似度の前に `F.normalize` が必須――一方 `forward` の `logits_per_image` は内部で正規化済み、という**非対称**を実験で確かめること。第三に、CLIP は **softmax（候補が相互排他）**、SigLIP は **sigmoid（各ラベル独立）**という**確率解釈の違い**を、「該当ラベルが無い」場面の挙動差として体験すること。第四に、正規化＋コサイン＋`torch.topk` で **text→image / image→image 検索**を実装し、**Recall@k・mAP・MRR** を `torchmetrics` で算出できることです。

本章のスクリプトはすべて、ネット接続もデータセットDLも無しで完走するよう、入力画像を**その場で合成**します（明るい背景の上に「赤い円」「青い四角」などの色×形を描いた画像）。CLIP は色や形という概念をゼロショットで強く捉えるので、合成画像でも `"a photo of a red circle"` が赤い円を正しく引き当て、教材として意味のある類似度が出ます。実画像で試したい人は `data/16_clip_zeroshot_retrieval/` に画像を置けば自動で使われます（`clip_helpers.load_user_or_synthetic` 参照）。ダウンロードが走るのは初回のモデル重み取得（CLIP と SigLIP）だけで、以降はローカルキャッシュから即起動します。

---

## 1. 共有埋め込み空間と対照学習 — CLIP はなぜゼロショットできるのか

ふつうの画像分類器は、学習時に決めた固定クラス（ImageNet なら1000種）にしか答えられません。出力層の各ノードが特定クラスに割り当てられているからです。CLIP の発想はまったく違います。**画像を1本のベクトルに、文も1本のベクトルに**変換し、両者を**同じ次元の同じ空間**へ射影します。学習では数億の「画像とそのキャプション」のペアを使い、**対になった画像・文は近く、無関係な組は遠く**なるよう対照損失（InfoNCE）で引き寄せ・引き離しを繰り返します。その結果、空間上では「犬の写真」のベクトルと「a photo of a dog」のベクトルが自然と近くに集まります。

この「同じ空間に画像も言葉も乗っている」性質が、ゼロショットの正体です。分類したいとき、候補ラベルを**文に変換**して（例: ラベル "cat" を `"a photo of a cat"` という**プロンプト**に）テキストエンコーダに通し、画像ベクトルとの**コサイン類似度**を測って一番近い文を選ぶだけ。出力ノードを持たないので、候補は推論時に自由に差し替えられます。だからモデルを再学習せずに「赤い円か青い四角か」でも「猫か犬か車か」でも、その場で問いを組み立てられるのです。検索も同じ原理で、クエリ文に近い画像を並べれば text→image 検索になります。

本章で使うモデルは2つです。**CLIP**（`openai/clip-vit-base-patch32`）は CPU でも軽快に動く最速の定番で、softmax ベースの確率解釈を持ちます。**SigLIP**（`google/siglip2-base-patch16-224`）は対照損失を sigmoid（ペアごとの二値判定）に置き換えた後継で、確率解釈が独立になります（第5節）。どちらも「画像とテキストを同じ空間に埋め込む」という骨格は共通で、違いは**損失と確率の読み方**だけ、と捉えると見通しが良くなります。まずは原理を頭に入れて、次節で実際に動かしてみましょう。

## 2. ゼロショット分類を最短で動かす（pipeline）

理屈を一度脇に置いて、まず成功体験を得るのが近道です。transformers の高レベル API `pipeline("zero-shot-image-classification")` は、前処理（画像のリサイズ・正規化、テキストのトークン化）から推論、softmax 後処理までを一手に引き受けます。`01_zeroshot_pipeline.py` はこれを使い、合成画像を `candidate_labels`（その場で渡す任意のラベル候補）に対して分類します。`candidate_labels` を自由に決められることこそが「ゼロショット」の体感ポイントです。

下が pipeline 呼び出しの核です。`task` と `model` を指定し、画像と候補ラベルを渡すだけ。`device` は CPU なら `-1`、CUDA があれば `0` を渡します（本講座は CPU 前提なので既定で `-1` 相当）。返り値は画像ごとに `[{'score':…, 'label':…}, …]` のリストで、score 降順に並んでいます。

```python
from transformers import pipeline

clf = pipeline("zero-shot-image-classification",
               model="openai/clip-vit-base-patch32", device=-1)  # CPU は -1
labels = ["a red circle", "a blue square", "a green triangle", "a yellow circle"]
outputs = clf(images, candidate_labels=labels)   # 画像ごとに score 降順のリスト
top = outputs[0][0]                              # 1枚目の最上位ラベル
```

合成4枚（red circle / blue square / green triangle / yellow circle）での実行結果は top-1 accuracy = 1.00 で、すべて正しく当たります。スコアを見ると `blue square` は 1.000 と即断する一方、`red circle` は 0.652 とやや控えめです。これは候補に `yellow circle` という「同じ円」が混ざっていて、CLIP が形（円）と色（赤）の両方を見比べて迷うためで、**候補ラベルの作り方が結果を左右する**ことが読み取れます。pipeline は手軽ですが、内部の前処理・後処理がブラックボックスのままです。学習目的では中身を知る必要があるので、次節で同じことを手書きに分解します。

## 3. pipeline を分解する — CLIPProcessor と CLIPModel を手書きする

`02_clip_siglip_manual.py` では pipeline を `processor` と `model` の2部品に開きます。`CLIPProcessor` は**画像とテキストを同時に前処理**し、`pixel_values`（正規化済み画像テンソル）と `input_ids`/`attention_mask`（トークン列）を作ります。複数ラベルをまとめて渡すときは長さを揃えるため **`padding=True` を忘れない**のが鉄則です（忘れると長さ不一致でエラーになります）。`model(**inputs)` の `forward` は両者をエンコードして同一空間に射影し、`outputs.logits_per_image`（形 `(画像数, ラベル数)`、大きいほど似ている）を返します。これを `softmax(dim=1)` すれば各画像のラベル確率です。

```python
from transformers import CLIPModel, CLIPProcessor
import torch

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
inputs = proc(text=labels, images=images, return_tensors="pt", padding=True).to(device)
with torch.inference_mode():                      # 推論は勾配を切る（CPU では特に効く）
    logits = model(**inputs).logits_per_image     # (n_images, n_labels)
probs = logits.softmax(dim=1)                      # 行ごとに合計 1.0
```

ここで強調したい作法が3つあります。ひとつは **`.to(device)` をモデルと入力の両方に**揃えること（片方だけだと `RuntimeError`）。`inputs.to(device)` は `processor` の出力（`BatchEncoding`）にそのまま効きます。ふたつめは **`model.eval()` ＋ `torch.inference_mode()`** で、推論時は勾配計算を止めてメモリと時間を節約します。みっつめは **dtype**で、CPU では `float32` 一択です（`float16`/`half` は CPU では遅い・未対応 op が多い）。GPU があるときだけ `autocast`/`bfloat16` を検討します。これで pipeline の中身が見えました。次は「埋め込みを直接取り出す」ときに必ずぶつかる落とし穴を扱います。

## 4. 埋め込みの非対称性 — `get_*_features` は未正規化、`forward` の logits は正規化済み

検索や独自の類似度計算では、`forward` 全体ではなく**射影後の埋め込みベクトルそのもの**が欲しくなります。CLIP には `model.get_image_features(...)` と `model.get_text_features(...)` が用意されています。ここで**transformers v5 の重要な変化**を押さえてください。v5 ではこれらは「テンソル」ではなく **`BaseModelOutputWithPooling` オブジェクト**を返し、欲しい射影後ベクトル `(B, 512)` は **`.pooler_output`** に入っています（`.last_hidden_state` の方は射影前の生の中間表現です）。古いブログのように戻り値を直接テンソル扱いするコードは v5 では動きません。

そして最大の落とし穴が**正規化の非対称**です。`.pooler_output` の埋め込みは**未正規化**で、`02` の実測では画像ベクトルのノルムは約 11（決して 1 ではない）です。コサイン類似度は「正規化したベクトルの内積」なので、検索の前に必ず `F.normalize(x, p=2, dim=-1)` で L2 正規化が要ります。一方、`forward` が返す `logits_per_image` は**内部で正規化済み**――事実、「正規化した画像埋め込みとテキスト埋め込みの内積に温度 `logit_scale.exp()` を掛けた値」が `logits_per_image` に一致します（`02` で `torch.allclose(...) == True` を確認）。**素朴に `get_*_features` を取ると未正規化、`forward` 経由だと正規化済み**――この差を取り違えると検索が静かに崩れます。

```python
import torch.nn.functional as F

with torch.inference_mode():
    img = model.get_image_features(pixel_values=inputs["pixel_values"]).pooler_output  # (B,512) 未正規化
    txt = model.get_text_features(input_ids=inputs["input_ids"],
                                  attention_mask=inputs["attention_mask"]).pooler_output
    img_n = F.normalize(img, p=2, dim=-1)          # ← これを忘れるとコサインにならない
    txt_n = F.normalize(txt, p=2, dim=-1)
    logits_manual = (img_n @ txt_n.t()) * model.logit_scale.exp()  # forward の logits と一致
```

本講座の `clip_helpers.clip_image_embeds` / `clip_text_embeds` は、この `.pooler_output` 取り出しを薄くラップしただけの関数です（中身は上のコードそのもの）。「未正規化のベクトルが返る」ことを前提に、呼び出し側で必ず `F.normalize` を挟む設計にしてあります。なぜ正規化がそこまで重要なのかは、第8節の「高ノルム distractor 実験」で数字とともに体感します。先に、CLIP と SigLIP のもうひとつの違い――確率の読み方――を見ておきましょう。

## 5. softmax と sigmoid — CLIP と SigLIP の確率解釈の違い

CLIP は学習時、1枚の画像に対して「正しいキャプションは1つ」という前提で **softmax** ベースの対照損失を使います。だから推論でも `logits_per_image.softmax(dim=1)` は**候補ラベル全体で合計 1.0** になり、「**候補のどれか1つ**を選ぶ」相互排他の確率になります。これは「正解が候補に必ず入っている」場面では自然ですが、**どの候補にも当てはまらない画像**でも「無理やり一番マシなものに確率を寄せる」副作用があります。`02` の実験では、赤い円・青い四角に対して候補を `cat / dog / car` にすると、CLIP softmax は `[0.296, 0.143, 0.560]` のように**それらしく分配**してしまい、「該当なし」と言えません。

SigLIP はこの点を変えます。損失を **sigmoid**（各画像・各テキストの組を独立に「合う/合わない」の二値で判定）に置き換えて学習するため、推論の `logits_per_image.sigmoid()` は**各ラベルを独立に 0〜1 で評価**します。合計が 1 になる制約はありません。同じ `cat / dog / car` のケースで SigLIP の sigmoid 確率は `[0.000, 0.000, 0.000]` ――つまり**全ラベル低い＝「どれも該当しない」と正しく表現**できます。下の表が両者の挙動差です（`02` の実測値）。

| 入力 | 候補ラベル | CLIP softmax（合計1.0） | SigLIP sigmoid（独立） |
| --- | --- | --- | --- |
| 赤い円 | red circle / blue square / green triangle | **0.99** / 0.01 / 0.00 | **0.98** / 0.00 / 0.00 |
| 赤い円 | cat / dog / car（該当なし） | 0.30 / 0.14 / **0.56** | 0.00 / 0.00 / 0.00 |

この違いは実務の選択に直結します。「候補のどれか1つに必ず分類したい」（排他的なカテゴリ分け）なら CLIP の softmax が素直です。一方、「タグ付けのように複数ラベルが同時に立ちうる」「該当なしを検出したい」「閾値でフィルタしたい」なら、各ラベルが独立に意味を持つ SigLIP の sigmoid が向きます。実装上の注意として、**SigLIP の processor は `padding="max_length"` を期待**します（CLIP は `padding=True` でよい）。`02_softmax_vs_sigmoid_match.png` / `_nomatch.png` のヒートマップで、行の合計が 1 になる CLIP と、ばらばらに点灯する SigLIP を見比べてください。

## 6. 画像テキスト検索 — 正規化＋コサイン＋topk

ゼロショット「分類」は候補が**ラベル文**でした。検索は候補が**画像コレクション**になるだけで、原理はまったく同じです。`03_text_image_retrieval.py` は次の3手順を踏みます。(1) コレクションの全画像を `get_image_features().pooler_output` で埋め込み、`F.normalize` で正規化して `(N, D)` の行列にする。(2) クエリ（テキストでも画像でも）を同じ手順で埋め込み・正規化する。(3) **正規化済みベクトルの内積＝コサイン類似度**でスコアを出し、`torch.topk`（または `argsort`）で上位を並べる。これで text→image も image→image も書けます。

```python
img_emb = F.normalize(embed_images(model, proc, images, device), p=2, dim=-1)  # (N, D)
txt_emb = F.normalize(embed_texts(model, proc, queries, device), p=2, dim=-1)  # (Q, D)
sims = txt_emb @ img_emb.t()                  # (Q, N) コサイン類似度（-1〜1）
top_scores, top_idx = torch.topk(sims[q], k=3)   # クエリ q の上位3画像
```

`03_text_to_image.png` は代表4クエリについて上位3枚を並べたパネルです。`"a photo of a red circle"` には赤い円が、`"a photo of a blue square"` には青い四角が先頭に来て、コサイン値（0.2〜0.36 程度）も添えてあります。image→image 検索では、クエリ画像（赤い円）に対し**自分自身を除外**して近傍を返します（`sims[q_self] = -2` で最下位に押しやるのが簡単な定石です）。実測では赤い円の近傍は「赤い四角（同色）」「青い円（同形）」が上位に来て、CLIP が**色と形の両方**を類似の手がかりにしていることが見て取れます。検索が「それっぽく」動いたら、次は**どれだけ正しいかを数字で測る**段階です。

## 7. 検索の評価 — Recall@k / mAP / MRR

検索の良し悪しは「上位に正解がどれだけ来たか」で測ります。本章の3指標を定義から押さえましょう。**Recall@k** は「上位 k 件に正解（関連アイテム）が入った割合」で、`(上位k に入った正解数) / (正解の総数)`。**MRR（Mean Reciprocal Rank）** は「最初の正解が出た順位の逆数」をクエリ平均したもので、1位で当たれば 1.0、3位なら 1/3。**retrieval mAP** は「クエリごとに Average Precision（適合率を正解の順位で平均した値）を出し、それを全クエリで平均」した、ランキング全体の質を見る主指標です。いずれも「クエリ1本ずつ計算して平均する」のが基本構造です。

計算は `torchmetrics` の functional 版が簡潔です。クエリ1本分の「スコア列 `preds`」と「正解フラグ列 `target`（bool）」を渡すと、各指標が1本分だけ返るので、全クエリを回して平均します。`03` の `retrieval_metrics` がまさにこれで、`relevant`（各クエリの正解インデックス集合）から `target` を組み立てています。

```python
from torchmetrics.functional.retrieval import (
    retrieval_recall, retrieval_average_precision, retrieval_reciprocal_rank,
)
target = torch.zeros(N, dtype=torch.bool); target[gt_idx] = True   # 正解の位置を True に
r_at_5 = retrieval_recall(preds, target, top_k=5)   # このクエリの Recall@5
ap     = retrieval_average_precision(preds, target) # このクエリの AP（平均すると mAP）
rr     = retrieval_reciprocal_rank(preds, target)   # このクエリの逆順位（平均すると MRR）
```

`03` の text→image（各クエリの正解は対応する1枚）での実測は **Recall@1 = 0.92、Recall@5 = 1.00、mAP = 0.96、MRR = 0.96**。12クエリ中11個で1位に正解が来て、残り1個も上位5位以内に入る、という読みです。「Recall@1 が 0.92」と「mAP が 0.96」が**別の角度から同じランキングを評価**していること――前者は1位だけ、後者は正解の順位全体を見ている――を意識すると、指標の使い分けが腑に落ちます（評価指標の体系そのものは第14回で深掘りしました）。さて、この検索は第4節の正規化を**正しくやった**結果です。もし正規化を忘れるとどうなるかを、次節で確かめます。

## 8. 正規化を忘れるとどうなるか — 高ノルム distractor 実験

第4節で「`get_*_features` は未正規化だから `F.normalize` が必須」と述べました。では、忘れたら実際どれだけ壊れるのか。まず**スコアの値域**が違います。`03` の実測で、正規化したコサインは **0.206〜0.361**（-1〜1 に収まり、閾値設定や比較ができる）なのに対し、未正規化のまま内積を取ると **19.0〜34.4**（ベクトルの大きさ次第で青天井、値の意味が読めない）。そして**ランキングも変わり**、12クエリ全部で並び順が変化しました。ただし正直に言うと、この合成セットは色の手がかりが強すぎて、**上位1位だけ見ると未正規化でも正解が生き残ってしまう**ことが多く、指標の差が出にくいのです。

そこで「なぜ大きさを無視（＝正規化）しないと危ないのか」を確実に見せるため、`03` では**1枚だけ埋め込みのノルムを6倍に膨らませた distractor**（現実でいう「やたら自信のある／外れ値ベクトル」の模擬）を入れて比較します。コサイン類似度は**ベクトルの向きだけ**を見るのでノルムを無視しますが、生の内積は**大きさに比例**するので、ノルムを膨らませた画像が**ほぼ全クエリで1位を乗っ取り**ます。結果が下表で、コサインは高ノルム化に**無関心（指標が落ちない）**、生内積は**崩壊**します。

| スコアリング | Recall@1 | Recall@5 | mAP | MRR |
| --- | --- | --- | --- | --- |
| コサイン（L2正規化）— 大きさを無視 | **0.917** | 1.000 | **0.958** | 0.958 |
| 生の内積（未正規化）— 大きさに釣られる | 0.083 | 1.000 | 0.528 | 0.528 |

数字の差は歴然です。mAP は 0.96 → 0.53、Recall@1 は 0.92 → 0.08 まで落ちました（1枚の高ノルム画像が12クエリ中ほぼ全部の1位を奪ったため）。教訓は明快で、**コサイン類似度には L2 正規化が必須**――忘れると「埋め込みのノルムが大きいだけのアイテム」が検索結果を支配します。実コレクションでは画像ごとにノルムがもっとばらつくので、この罠は合成セットよりはるかに現実的に効いてきます。「うまく動いているように見えても、正規化を省いてはいけない」と覚えてください。これは次の第17回（FAISS）で `faiss.normalize_L2` ＋ `IndexFlatIP` を使う伏線でもあります。

## 9. 高レベル API との対比（sentence-transformers / open-clip）

ここまで transformers の `CLIPModel`/`CLIPProcessor` を直接触り、前処理・正規化・コサインを自分で書いてきました。これは「中身を理解する」目的には最適ですが、実務で素早く検索を組むなら**高レベル API** も知っておくと便利です。代表が **sentence-transformers** で、`SentenceTransformer("clip-ViT-B-32")` を使うと、`encode(images, normalize_embeddings=True)` の一行で「埋め込み＋L2正規化」まで済み（第4節の落とし穴を内部で吸収してくれる）、`util.cos_sim` でコサイン行列が得られます。本章で手書きした処理が、そのまま薄いラッパに包まれている格好です。

```python
# 参考（このリポジトリの既定環境には未インストール。uv add --group embed sentence-transformers で導入）
from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer("clip-ViT-B-32")
img_emb = model.encode(images, normalize_embeddings=True)   # 正規化まで込み
txt_emb = model.encode(["a red circle"], normalize_embeddings=True)
scores = util.cos_sim(txt_emb, img_emb)                     # コサイン類似度行列
```

もうひとつの選択肢が **open-clip**（`open_clip.create_model_and_transforms` / `get_tokenizer`）で、OpenAI 版以外の大規模学習済み（LAION 学習や MobileCLIP/SigLIP 系）を同じ作法で扱えます。使い分けの指針はこうです。**学習目的・原理理解**なら本章のように transformers で手書きするのが一番。**手早く検索プロダクトを組む**なら sentence-transformers が正規化やバッチ化を肩代わりしてくれて楽。**OpenAI 以外の重みや最新アーキを試したい**なら open-clip、です。なお `embed` グループ（sentence-transformers / open-clip）は本リポジトリの既定環境には入れていないので、上のコードは「読み物」として示しています。本章の実行スクリプト3本は transformers だけで完結します。

## 10. transformers v5 の注意点（古いコードが動かない理由）

最後に、本章で踏んだ **transformers 5.x の破壊的変更**をまとめます。ネット上の CLIP チュートリアルは 4.x 時代のものが多く、そのまま写すと動かないことがあるためです。第一に、画像の前処理クラスは **`AutoImageProcessor`** に一本化され、旧 `AutoFeatureExtractor` は**廃止**されました。さらに画像プロセッサは torchvision バックエンドの **fast 実装のみ**になったので（`use_fast` 引数の概念も消滅）、画像モデルを使うなら **torchvision が事実上必須**です（入れ忘れると processor 生成でエラー）。本講座は `dl` グループで torchvision を入れているので問題ありません。

第二に、第4節で見たとおり **`get_image_features`/`get_text_features` の戻り値が `BaseModelOutputWithPooling` オブジェクト**になり、射影後の埋め込みは **`.pooler_output`** から取り出します（4.x ではテンソルが直接返っていました）。下が「古い書き方 → v5 の書き方」の対応です。

```python
# 4.x（古い・v5 では動かない/挙動が違う）
# feat = model.get_image_features(**inputs)         # かつてはテンソルが返った
# proc = AutoFeatureExtractor.from_pretrained(id)   # 廃止

# 5.x（本章の正準）
from transformers import AutoImageProcessor      # 画像 processor はこちら（fast のみ）
feat = model.get_image_features(pixel_values=pv).pooler_output  # .pooler_output で取り出す
```

第三に、**モデルキャッシュ**まわりです。初回の `from_pretrained` はネット越しに重みを取得して `~/.cache/huggingface`（環境変数 `HF_HOME`）にキャッシュし、次回以降はそこから即起動します（旧 `TRANSFORMERS_CACHE` は非推奨）。Docker ではこのディレクトリをボリュームマウントしないと、コンテナを作り直すたびに再ダウンロードが走るので注意してください。完全オフラインで回すなら `HF_HUB_OFFLINE=1` を設定します。これらを押さえておけば、古い記事に惑わされず v5 の正準コードを書けます。

## 11. このモジュールの構成（スクリプト一覧）

各スクリプトは単一責務で、上から読むと「動かす → 分解する → 検索して測る」と理解が積み上がります。すべて `outputs/16_clip_zeroshot_retrieval/` に図と json を保存し、画面表示には依存しません。合成画像生成・device 判定・モデルロード・埋め込み取り出しといった共通処理は `clip_helpers.py` にまとめ、各スクリプトはそれを import します。

| ファイル | 役割（単一責務） |
| --- | --- |
| `clip_helpers.py` | device 判定・合成コレクション生成・CLIP/SigLIP ロード・`get_*_features().pooler_output` 取り出し・図保存。道具箱 |
| `01_zeroshot_pipeline.py` | `pipeline("zero-shot-image-classification")` で最短ゼロショット。candidate_labels・top-1 accuracy・スコア棒グラフ |
| `02_clip_siglip_manual.py` | `CLIPProcessor`+`CLIPModel` 手書き、埋め込みの非対称（未正規化）検証、CLIP softmax vs SigLIP sigmoid（該当なし比較） |
| `03_text_image_retrieval.py` | 正規化＋コサイン＋topk で text→image / image→image 検索、Recall@k・mAP・MRR、高ノルム distractor 実験 |
| `mini_project.py` | 章末の統合課題。プロンプト・アンサンブル → 検索評価 → 正規化アブレーション → 開集合の棄却（CLIP コサイン閾値 / SigLIP sigmoid）を1本に通す完成形 |
| `exercises.py` | TODO 形式の演習9問（易→難、自己採点ランナー付き。`SHOW_SOLUTION=1` で模範解答に差し替え） |
| `exercises_solutions.py` | 演習の模範解答（全問 PASS）。採点ロジックは `exercises.py` を再利用し、解答実装だけを保持（重複なし） |

表のとおり `clip_helpers.py` だけは「読み物」ではなく「再利用する道具」です。とくに `clip_image_embeds`/`clip_text_embeds`（`.pooler_output` を返すだけ＝**未正規化**）と `build_collection`（色×形の12枚）が、3スクリプト全部の土台になっています。まず helper を一読してから 01 へ進むと、各スクリプトが何を import しているかが腑に落ちます。

## 🛠 章末ミニプロジェクト — ゼロショット検索＆タグ付けエンジン（棄却つき）

ここまでの学び（埋め込み → 正規化 → コサイン → 確率解釈 → 評価）を**1本に統合**するのが `mini_project.py` です。合成コレクション（色×形の12枚）に対して、実運用の検索エンジンが踏む4ステージを通しで実行し、`outputs/16_clip_zeroshot_retrieval/mini_project_summary.png`（4パネル要約）と `mini_project_report.json`（全数値）を出力します。CPU で数十秒、ネットは初回のモデル重みDLのみです。

- **Stage A: プロンプト・アンサンブル**（§2/§3 の発展）。ラベルを `"a photo of a {x}"` 1本で埋め込む「単一テンプレ」と、5種のテンプレ（`"an image of a {x}"`, `"a {x} on a plain background"` …）を**各々正規化 → 平均 → 再正規化**した「アンサンブル」を比較します。実測は単一/アンサンブルとも **accuracy = 0.917**、平均マージン（top1−top2）は **0.0355 / 0.0337** とほぼ互角。この合成セットは易しいので差は僅少ですが、CLIP 論文が80テンプレを平均するのは、**実データ・曖昧なクラス・語彙のゆらぎで分散を均す**ためだ、という勘所を体感します（「単純な手法ほど効く場面を見極める」のが master の視点）。
- **Stage B: text→image 検索エンジン**（§6/§7）。アンサンブル重みをクエリに使い、正規化＋コサイン＋topk で検索して **Recall@1 = 0.917 / Recall@5 = 1.00 / mAP = 0.958 / MRR = 0.958**。
- **Stage C: 正規化アブレーション**（§4/§8）。1枚だけ埋め込みノルムを6倍にした distractor を入れ、**コサイン（mAP 0.958）** と **生内積（mAP 0.528）** を比べて「大きさに釣られる」崩壊を再現します。
- **Stage D: 開集合と棄却**（§5）。語彙に**無い**クエリ（`cat` / `wooden chair` / `purple star`）を「該当なし」と判定できるかを2方式で検証します。**CLIP のコサイン最大値**は in-vocab `[0.30, 0.34, 0.34]` と oov `[0.23, 0.22, 0.25]` に**綺麗に分離**し（gap=0.043）、中点 0.276 を閾値に棄却できます。**SigLIP の sigmoid** は in-vocab `[0.93, 0.96, 0.99]` に対し oov は**すべて 0.00**で、固定閾値 0.5 で自然に棄却できます。softmax は必ずどれかを選ぶ＝棄却できない、という第5節の含意を「検索の棄却」という実用文脈で締めくくります。

```bash
uv run python lectures/16_clip_zeroshot_retrieval/mini_project.py
# → outputs/16_clip_zeroshot_retrieval/mini_project_summary.png, mini_project_report.json
```

このミニプロジェクトを自分の手で読み解き、4つの数字（margin・mAP・mAP崩壊・棄却の分離 gap）が**何を測っているか**を説明できれば、本章のゴールに到達しています。

## ✅ 到達チェックリスト

次の項目をすべて「コードで再現でき、理由を一言で説明できる」状態を目標にしてください。

- [ ] **ゼロショットの原理**: CLIP が画像と文を同じ空間に射影し、候補ラベルを**文に変換**してコサインで照合する流れを説明できる。
- [ ] **pipeline → 手書き**: `pipeline("zero-shot-image-classification")` の結果を、`CLIPProcessor`＋`CLIPModel`＋`softmax` で**再現**できる（§2→§3）。
- [ ] **埋め込みの非対称**: `get_*_features(...).pooler_output` は**未正規化**、`forward` の `logits_per_image` は**正規化済み**、という差を `torch.allclose` で確認できる（§4）。
- [ ] **正規化の必須性**: `F.normalize` を入れた**コサイン**と、入れない**生内積**でスコア値域・ランキングが変わることを示せる（§8）。
- [ ] **softmax vs sigmoid**: CLIP の `softmax`（合計1・相互排他）と SigLIP の `sigmoid`（独立・該当なしを表現）の違いを、「該当なし」ケースで説明できる（§5）。
- [ ] **検索の実装**: 画像コレクションを埋め込み・正規化し、テキスト/画像クエリで `topk` 検索（image→image は自己除外）を書ける（§6）。
- [ ] **評価指標**: Recall@k・mAP・MRR を `torchmetrics.functional.retrieval` で**クエリ別→平均**して算出できる（§7）。
- [ ] **温度 logit_scale**: 正規化埋め込みの内積に `logit_scale.exp()` を掛けると `forward` の logits に一致することを再構成できる（演習8）。
- [ ] **棄却**: コサイン閾値や sigmoid で「該当なし（-1）」を返せる＝**開集合**を扱える（演習9・ミニプロジェクト Stage D）。
- [ ] **v5 の作法**: `AutoImageProcessor`（fast 専用・torchvision 必須）、`.pooler_output`、`HF_HOME` キャッシュを把握している（§10）。
- [ ] **演習**: `exercises.py` を9問すべて自力で PASS させた（`exercises_solutions.py` で答え合わせ）。

## ❓ よくある落とし穴・FAQ・デバッグ

§13 に「症状 → 原因 → 対処」の早見表があります。ここではその表に載らない**判断の指針**と、つまずいたときの**切り分け手順**を補足します。

**Q1. 検索結果がそれっぽいのに、ときどき変なものが上位に来る。** まず `F.normalize` を**画像・クエリの両方**に掛けているか確認します。次に、埋め込み行列の各行ノルムを `emb.norm(dim=-1)` で出し、`1.0` に揃っているかを見ます。揃っていなければ正規化漏れです。ミニプロジェクト Stage C のように、1枚だけノルムが大きいと**そのアイテムが検索を乗っ取る**のが典型です。

**Q2. CLIP の確率が「該当なし」でも高く出る。これは壊れている?** いいえ、仕様です。CLIP の `softmax` は**候補内で相互排他**なので、正解が候補に無くても「一番マシ」に確率を寄せます（§5）。「該当なし」を扱いたいなら、SigLIP の `sigmoid` か、コサインの**閾値**（ミニプロジェクト Stage D）で棄却を実装します。

**Q3. `get_image_features(...)` の戻り値に `.norm()` を呼んだら AttributeError。** v5 では戻り値が `BaseModelOutputWithPooling` オブジェクトです。`.pooler_output` でテンソルを取り出してから演算します（§4/§10）。

**Q4. SigLIP だけ結果がおかしい / エラーになる。** SigLIP の processor は `padding="max_length"` を期待します（CLIP の `padding=True` とは別）。また確率は `sigmoid()` で読みます（`softmax` ではない）。トークナイザに `sentencepiece` が要るので `hf` グループを入れてください。

**Q5. 毎回モデルがダウンロードされて遅い（特に Docker / CI）。** `~/.cache/huggingface`（`HF_HOME`）をボリュームマウント or 永続化します。完全オフラインなら `HF_HUB_OFFLINE=1`。再現性重視なら `from_pretrained(..., revision="<commit>")` でコミット固定も検討します。

**デバッグの切り分け順**: ①`emb.shape` と `emb.norm(dim=-1)`（次元・正規化の確認）→ ②スコア行列 `sims` の値域（コサインなら −1〜1、外れていれば正規化漏れ）→ ③`sims` の `argsort` 上位を**画像で目視**（数字だけ見ない）→ ④指標は**1クエリずつ**手計算と突き合わせ（§7 の Recall@1 と mAP を1本で検算）。この順で見ると、たいていの「検索が変」は正規化か device か padding に行き着きます。

## 🚀 発展トピック・参考

本章の骨格（埋め込み → 正規化 → コサイン → 評価/棄却）は、そのまま次章以降と実務に伸びます。

- **大規模化（第17回 FAISS）**: 12枚の総当たりは行列積で済みますが、件数が増えると `faiss.IndexFlatIP` ＋ `faiss.normalize_L2`（本章のコサインそのもの）→ さらに `IVF`/`HNSW`/`PQ` で近似検索へ。本章の正規化の作法が前提知識になります。
- **プロンプト設計**: ゼロショット精度はプロンプトに敏感です。CLIP 論文の80テンプレ平均、クラス名の言い換え（`"a photo of a dog"` vs `"a dog"`）、ドメイン特化テンプレ（`"a satellite photo of {x}"`）など。ミニプロジェクト Stage A を土台に、テンプレ集を差し替えて効果を測ってみましょう。
- **他アーキ/重み**: SigLIP2・MetaCLIP・MobileCLIP（CPU 向け軽量）など。`open-clip`（`embed` グループ）で LAION 学習や蒸留版を同じ作法で扱えます。第39回（CLIP 蒸留）に接続します。
- **高レベル API**: `sentence-transformers` の `SentenceTransformer("clip-ViT-B-32").encode(..., normalize_embeddings=True)` は本章の手書きを薄く包んだもの（§9）。素早くプロダクトを組むときの選択肢です。
- **指標の深掘り（第14回）**: nDCG・Precision@k・retrieval mAP の補間方式など。`torchmetrics.retrieval` の各クラスを参照。
- **公式ドキュメント**: [transformers CLIP](https://huggingface.co/docs/transformers/en/model_doc/clip) ／ [SigLIP](https://huggingface.co/docs/transformers/en/model_doc/siglip) ／ [torchmetrics retrieval](https://lightning.ai/docs/torchmetrics/stable/) ／ [OpenAI CLIP 論文](https://arxiv.org/abs/2103.00020) ／ [SigLIP 論文](https://arxiv.org/abs/2303.15343)。

## 12. 動かし方

このモジュールは `dl`（torch/torchvision）・`hf`（transformers/sentencepiece ほか）・`metrics`（torchmetrics）グループに依存します。CPU だけで完走し、初回のみ CLIP と SigLIP の重みを HuggingFace からダウンロードします（以降はキャッシュから即起動）。プロジェクトルートで以下を順に実行してください。

```bash
# 依存グループをインストール（初回のみ）
uv sync --group dl --group hf --group metrics

# 各スクリプトを実行（結果は outputs/16_clip_zeroshot_retrieval/ に保存される）
uv run python lectures/16_clip_zeroshot_retrieval/clip_helpers.py            # 道具箱のスモークテスト＋コレクション図
uv run python lectures/16_clip_zeroshot_retrieval/01_zeroshot_pipeline.py
uv run python lectures/16_clip_zeroshot_retrieval/02_clip_siglip_manual.py
uv run python lectures/16_clip_zeroshot_retrieval/03_text_image_retrieval.py

# 章末ミニプロジェクト（4ステージを統合した完成形。図 + JSON を出力）
uv run python lectures/16_clip_zeroshot_retrieval/mini_project.py

# 演習: まずは TODO を自分で埋める（最初は全部 FAIL だが exit 0）
uv run python lectures/16_clip_zeroshot_retrieval/exercises.py
# どうしても分からない時だけ、模範解答の挙動を見る（2通りとも同じ採点ロジックを使う）
SHOW_SOLUTION=1 uv run python lectures/16_clip_zeroshot_retrieval/exercises.py
uv run python lectures/16_clip_zeroshot_retrieval/exercises_solutions.py   # 全問 PASS の確認

# （任意）実画像で試す: data/16_clip_zeroshot_retrieval/ に .png/.jpg を置くと自動で使われる
```

実行後は `outputs/16_clip_zeroshot_retrieval/` の図を解説と照らし合わせてください。とくに `02_softmax_vs_sigmoid_nomatch.png`（CLIP は無理に分配、SigLIP は全部低い）と `03_text_to_image.png`（クエリ文の上位3枚）を見ると、本章の2大テーマ（確率解釈の違い・検索の仕組み）が視覚的に腑に落ちます。図中の文字は CJK フォントの豆腐（□）を避けるため ASCII にしてあります。色が反転して見える場合は、合成画像を RGB のまま扱っているか（cv2 経由で BGR が混ざっていないか）を確認してください。

## 13. よくあるエラーと対処（チェックリスト）

最後に、本章でつまずきやすい点を「症状 → 原因 → 対処」でまとめます。CLIP/transformers 特有の罠が多いので、詰まったらまずここを見てください。

| 症状 | ほぼ確実な原因 | 対処 |
| --- | --- | --- |
| 検索結果がデタラメ／スコアが巨大 | `get_*_features` が未正規化なのに正規化していない | `F.normalize(x, p=2, dim=-1)` を画像・テキスト両方に |
| `get_image_features(...).norm()` でエラー | v5 は戻り値が `BaseModelOutputWithPooling`（テンソルでない） | `.pooler_output` を取り出してから使う |
| `AutoFeatureExtractor` が無い/動かない | v5 で廃止 | `AutoImageProcessor`（または `CLIPProcessor`）を使う |
| processor 生成で torchvision 関連エラー | v5 の画像 processor は torchvision 必須 | `dl` グループ（torchvision）を入れる |
| `RuntimeError: ... different devices` | model と inputs の device がズレた | `.to(device)` を両方に。`inputs.to(device)` を忘れない |
| 複数ラベルで長さ不一致エラー | `padding` を指定していない | CLIP は `padding=True`、SigLIP は `padding="max_length"` |
| SigLIP の確率が全部おかしい | SigLIP は sigmoid（softmax ではない） | `logits_per_image.sigmoid()` で読む。合計は1にならない |
| CPU で推論が極端に遅い | `float16`/`half` を CPU で使っている | CPU は `float32`。`inference_mode()` を付ける |
| 毎回モデルを再DLする（Docker） | キャッシュをマウントしていない | `~/.cache/huggingface`（`HF_HOME`）をボリューム化 |

この表の9項目が、本章で遭遇しがちな不具合のほぼ全てです。とくに上3つ（正規化忘れ・`.pooler_output`・`AutoImageProcessor`）は v5＋CLIP の「あるある」なので、症状を見たら原因を即座に言い当てられるようにしておきましょう。

## 14. まとめ

本章では、CLIP/SigLIP が**画像と言語を共有潜在空間に射影する**原理から出発し、`pipeline` での最短ゼロショット、`CLIPProcessor`＋`CLIPModel` の手書き分解、`get_*_features` の**未正規化**と `forward` の**正規化済み**という非対称、CLIP の **softmax（相互排他）** と SigLIP の **sigmoid（独立）** の確率解釈差、そして正規化＋コサイン＋`topk` による画像テキスト検索と **Recall@k / mAP / MRR** までを、すべて合成画像の上で「自分で再現し、数字で確認できる」レベルで扱いました。通底するのは「**コサイン類似度には L2 正規化が必須**」「**softmax と sigmoid は別物**」という2つの勘所です。

ここで身につけた「埋め込み → 正規化 → コサインで検索」という骨格は、次の第17回（FAISS によるベクトル検索）で `faiss.normalize_L2`＋`IndexFlatIP` として、また最終章（第40・41回）の Cluster-CLIP パイプラインへとそのまま繋がります。まずは演習を全問 PASS させ、`03` の「高ノルム distractor で生内積の mAP が 0.96→0.53 に崩れる」結果を自分の言葉で説明できるようにしてから、次へ進んでください。

---

> 本教材で参照・検証したライブラリとバージョン（torch 2.12+cpu / torchvision 0.27+cpu / transformers 5.11 / faiss-cpu、2026-06 時点の安定版で動作確認）:
> Python 3.12 ／ torch 2.12.0+cpu ／ torchvision 0.27.0+cpu ／ transformers 5.11.0 ／ huggingface-hub 1.18.0 ／ timm 1.0.27 ／ safetensors 0.8.0 ／ sentencepiece 0.2.1 ／ torchmetrics 1.9.0 ／ scikit-learn 1.9.0 ／ faiss-cpu 1.14.2（第17回で使用）／ numpy 2.4.6 ／ Pillow 12.2.0 ／ matplotlib 3.10.9 ／ opencv-python-headless 4.13.0（合成画像の描画）
> 使用モデル: `openai/clip-vit-base-patch32`（CLIP）／ `google/siglip2-base-patch16-224`（SigLIP2）。初回のみ HuggingFace から重みを取得しキャッシュします。