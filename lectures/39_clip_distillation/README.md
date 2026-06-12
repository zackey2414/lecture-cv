# 39_clip_distillation: VLM/CLIP の蒸留 — TinyCLIP/MobileCLIP・埋め込み模倣

> トラック: **最適化・デプロイ** ／ レベル: **上級** ／ 必要な依存グループ: `dl` `hf` `embed` `distill`
> 前提モジュール: `38_knowledge_distillation`（温度付き KD・特徴量蒸留）／ `16_clip_zeroshot_retrieval`（CLIP の正準フロー）

---

## 🎯 この章のゴール

大きな CLIP（teacher）の知識を、**桁違いに小さい画像エンコーダ（student）へ蒸留**する手法を、手を動かして理解する。具体的には次ができるようになることを目指す。

- teacher CLIP の**画像埋め込みを student に模倣させる蒸留**（埋め込み回帰: コサイン + MSE）を自前で書ける。
- teacher の**画像-テキスト類似度行列**を、埋め込みを **L2 正規化**し **`logit_scale`（温度）** を掛けて作り、それを student に **KL（親和性蒸留）** で模倣させられる。
- **正規化・温度を揃えないと類似度のスケールがずれて蒸留が壊れる**ことを、数値で説明できる。
- **TinyCLIP（重み継承 + 親和性蒸留）** と **MobileCLIP/MobileCLIP2（DataCompDR によるデータセット強化）** の考え方の違いを言える。
- `open_clip` で MobileCLIP/TinyCLIP を**CPU ロード**し、効率CLIP（MobileCLIP/TinyCLIP）の TorchScript 配布形まで接続できる。
- 蒸留結果を **ゼロショット精度・検索 Recall@k・teacher との埋め込みコサイン整合度（保持率）** で評価できる。

このモジュールのトイ実験では、**teacher（合計 1.51 億パラメータ、画像タワー 8785 万）→ student（約 16 万パラメータ、x551 圧縮）** という極端な圧縮でも、合成データ上で**ゼロショット精度を 100% 保持**し、teacher 埋め込みとのコサイン整合度 0.99 を達成する様子を観察する。

---


## 1. 直感 — 「答え」ではなく「埋め込み空間」を移す

38 章の知識蒸留では、teacher が出す**クラス確率（ソフトターゲット）**を student に真似させた。しかし、CLIP のような VLM（Vision-Language Model）には固定のクラスが無い。代わりに CLIP が持っているのは、**画像とテキストを同じ向きに並べた共有埋め込み空間**そのものである。したがって VLM 蒸留のゴールは「正解ラベルを当てる」ことではなく、**teacher が描いた埋め込み空間の幾何（どの画像がどのテキストの近くに来るか）を、小さな student へ丸ごと移植する**ことになる。

そして CLIP 蒸留の気持ちよさは、**ラベルが一切要らない**点にある。teacher にラベル無し画像をたくさん通して埋め込みを取り出せば、それがそのまま student の教師信号になるからだ。student は「この画像は埋め込み空間のこの座標に来るべきだ」とだけ教わり、teacher の世界観（赤い丸は赤い丸どうし近く、青い三角からは遠い…）を再現していく。その結果、student は **teacher のテキスト埋め込みを使ってゼロショット分類・検索ができる**ようになる——自分ではテキストを一度も見ていないのに、だ。

本章のトイ実験では、teacher として `openai/clip-vit-base-patch32`（224px 入力・ViT-B/32・埋め込み 512 次元）を使い、student として 64px 入力の小さな CNN（約 16 万パラメータ）を使う。student は「安い小さな画像」から「teacher の 512 次元埋め込み」を予測する関数を学ぶ。入力解像度もモデルサイズも段違いに小さいのに、埋め込み空間は再現できる——これが蒸留の威力だ。

## 2. 理論 — L2 正規化・logit_scale・3 種類の蒸留信号

CLIP の類似度計算は必ず次の正準形を取る。画像埋め込み `i` とテキスト埋め込み `t` を**それぞれ L2 正規化**し（`F.normalize`）、内積を取り、学習で得た**温度パラメータ `logit_scale`**（の `exp`）を掛ける。

```
logits = logit_scale * normalize(i) @ normalize(t).T     # exp(logit_scale) は CLIP では約 100
probs  = softmax(logits, dim=class)                       # CLIP は softmax（相互排他クラス）
```

では、なぜ正規化が要るのか。`get_image_features` が返す射影後の生埋め込みは、ノルムが画像ごとにバラバラ（本章の teacher では約 10〜12）であり、正規化しないと内積が「ベクトルの長さ」に引きずられて意味的な近さを測れない。逆に正規化すれば、内積はそのまま**コサイン類似度**になり、全ペアを公平に比較できる。一方 `logit_scale`（≈100）は **softmax の鋭さを決める温度**であり、これが無いとコサインは [-1, 1] に収まったまま softmax がほぼ一様になり、「どれが正解か」という teacher の確信が消えてしまう。したがって、**蒸留では teacher と student で正規化と `logit_scale` を必ず揃える**——これがこの章で最も重要な約束だ。

teacher から student へ移す信号には、大きく 3 つの粒度がある。**(a) 埋め込み回帰**: student の埋め込みを teacher の埋め込みそのものに `1 - cos`（＋補助の MSE）で寄せる。最も直接的で安定だ。**(b) 親和性（類似度行列）蒸留**: teacher の画像-テキスト類似度行列を softmax してソフトターゲットにし、student の行列を `KL` で合わせる。これは 38 章の Hinton 蒸留を「クラス確率」ではなく「画像×テキストの確率」に置き換えたもので、TinyCLIP が使う形である。**(c) 対照蒸留（contrastive distillation）**: バッチ内の画像-テキスト対応を InfoNCE で学びつつ teacher の類似度も模倣する、実データの大規模学習で使う形。本章は CPU トイ実験なので、(a) を主役に据え、(b) を `03` で実装し、(c) は概念として触れるにとどめる。

## 3. 正準 API — transformers の teacher と open_clip の効率 CLIP

teacher 側（transformers v5）の正準フローは次の通り。`CLIPModel` をロードし、`get_image_features` / `get_text_features` で埋め込みを取り、**自分で `F.normalize` してから** `logit_scale.exp()` を掛ける。

```python
from transformers import CLIPModel, CLIPProcessor
import torch, torch.nn.functional as F

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval()
proc  = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

with torch.inference_mode():
    img = model.get_image_features(**proc(images=pil_list, return_tensors="pt")).pooler_output
    txt = model.get_text_features(**proc(text=prompts, return_tensors="pt", padding=True)).pooler_output
img = F.normalize(img, dim=-1)   # ← 未正規化(ノルム約10〜12)なので必須
txt = F.normalize(txt, dim=-1)
logits = model.logit_scale.exp() * img @ txt.T
```

> ⚠️ transformers v5 では `get_image_features` は**テンソルではなく出力オブジェクト**を返す。射影後の埋め込みは `.pooler_output`（[N, 512]・**未正規化**）にある。一方 `model(**inputs).image_embeds` は **L2 正規化済み**で返る。どちらを使うにせよ「正規化されているか」を必ず確認すること（本章のヘルパは `pooler_output` を取り出して明示的に `F.normalize` する）。

効率 CLIP 側（`open_clip` 3.3）の正準フローはこう。`create_model_and_transforms` で**アーキ + 前処理**を同時に得て、`encode_image` / `encode_text` で埋め込みを取り、`get_tokenizer` でテキストをトークン化する。事前学習キーは `list_pretrained()` で確認する。

```python
import open_clip, torch.nn.functional as F
print([(n, t) for n, t in open_clip.list_pretrained() if "MobileCLIP" in n or "TinyCLIP" in n])
model, _, preprocess = open_clip.create_model_and_transforms("MobileCLIP-S1", pretrained="datacompdr")
tokenizer = open_clip.get_tokenizer("MobileCLIP-S1")
model.eval()
with torch.inference_mode():
    img = F.normalize(model.encode_image(preprocess(pil).unsqueeze(0)), dim=-1)
    txt = F.normalize(model.encode_text(tokenizer(["a dog", "a cat"])), dim=-1)
```

## 4. 実装をひとつずつ

**`01_teacher_similarity_matrix.py` — 正準フローの確認。** まず teacher だけで「encode → L2 正規化 → `logit_scale` → 類似度行列」を完走する。生埋め込みのノルムがバラバラ（約 10〜12）で、正規化すると全部 1.0 になることを数値で見せ、画像-テキスト確率行列をヒートマップに保存する。合成 3 クラス（赤い丸・緑の四角・青い三角）で teacher のゼロショット精度は 1.000 になる。

**`02_embedding_distillation.py` — この章の中心。** teacher の画像埋め込みを**一度だけ前計算**してキャッシュし（teacher は `eval()` + `inference_mode()` で凍結）、student を `(1 - cos) + MSE` で学習する。学習前は student と teacher の整合度が約 0.03（ランダム）なのが、約 40 エポック（CPU で数秒）で **0.99** まで上がる。teacher 画像タワー 8785 万 → student 約 16 万（**x551 圧縮**）。学習後の重みは `outputs/39_clip_distillation/student_distilled.pt` に保存される。

**`03_similarity_distillation_and_pitfall.py` — 親和性蒸留と落とし穴。** teacher の画像-テキスト類似度行列を softmax したソフトターゲットを、student の行列に `KL(log_softmax(student) ‖ softmax(teacher))`（`reduction='batchmean'`）で合わせる（TinyCLIP 型）。同時に**スケールずれの落とし穴**を数値で再現する: 正規化を忘れると行列がノルムに引きずられて暴走し、`logit_scale` を忘れると softmax の最大確率が 0.996 → 0.35 へと平坦化して teacher の確信が消える。

**`04_open_clip_mobileclip.py` — 効率 CLIP と参考実装接続。** `list_pretrained()` で MobileCLIP/TinyCLIP のキーを列挙し、アーキを**オフラインで（`pretrained=None`）構築**してパラメータ数と CPU レイテンシを比較する。事前学習重みのロードは `try/except` でガードし、ネットが無ければ概念に切り替える。最後に効率CLIP の TorchScript 配布形（例: MobileCLIP の `.ts`、約 600MB）に触れ、実運用での `load_clip_model()` 相当（`force_quick_gelu=True` で効率 CLIP を読み、CenterCrop を排した独自前処理で dense 特徴を取り出す）と、本章の自前 student が**「L2 正規化 + `logit_scale` で温度付与」という同じ約束を共有**していることを確認する。

## 5. 落とし穴（このモジュールで必ず踏む）

最頻の事故は**スケールずれ**だ。teacher と student で「正規化したか」「`logit_scale` を掛けたか」が食い違うと、同じ画像でも類似度の数値が桁で変わり、KL も MSE も意味をなさない。`03` の出力（正しい行列は範囲 [20, 33]、正規化忘れは [13, 21]、`logit_scale` 忘れは [0.2, 0.3]）を必ず自分の目で見ておくこと。次に多いのが **teacher を凍結し忘れる**事故で、`eval()` を呼ばないと BatchNorm/Dropout が動いて教師信号が毎回揺れ、optimizer に teacher のパラメータを渡すと誤って teacher 側を更新してしまう。

もうひとつは **MobileCLIP は x86 CPU では必ずしも速くない**という現実だ。本章の計測では、ランダム初期化の MobileCLIP-S1（FastViT 系、8497 万パラメータ）の画像エンコードは約 47ms で、teacher の ViT-B/32（約 16ms）より**遅い**。MobileCLIP の真価はモバイル GPU/ANE での再パラメータ化（reparameterization）にあり、PyTorch eager の x86 CPU ではその利得が出ない。「軽量 ＝ どの環境でも速い」ではなく、**どのハードウェアで速いのか**を必ず確認する。

最後は**用語の混同**である。まず「**知識蒸留（model distillation）**」は teacher→student へ知識を移すことを指す。これに対し「**dataset distillation**」は、学習データを少数の合成画像へ凝縮する全く別の研究だ。さらに MobileCLIP の「**dataset reinforcement（DataCompDR）**」は、teacher の埋め込みや合成キャプションを**事前計算してデータセットに焼き込む**ことで小モデルを強くする手法であり、これも別物である。MobileCLIP の論文の主張が「アーキより**データ（強化）が効く**」である点を押さえておきたい。

## 6. 実務の使い分け

**まず既製の蒸留済みモデルを試す。** 自分で蒸留する前に、`open_clip` の MobileCLIP-S0/S1/S2 や TinyCLIP が要件を満たさないか確認する。これらは大規模データで蒸留済みで、ゼロから蒸留するより圧倒的に強い。`list_pretrained()` でキーを確認し、CPU/エッジのレイテンシ要件に合うものを選ぶ。

**自前蒸留が要るのは「ドメイン特化」と「極端な小型化」のとき。** 自社の画像分布（製造ライン・医用・特定の商品画像など）でしか使わないなら、その分布のラベル無し画像で teacher 埋め込みを取り、小さな student に蒸留すると、汎用効率 CLIP より小さく・速く・そのドメインで十分な精度のモデルが作れる。本章の `02`/`mini_project` がそのレシピの最小形だ。**埋め込み回帰（a）から始め**、必要なら親和性蒸留（b）を足す、という順序が安全。

**評価は必ず 3 軸で。** ①ゼロショット精度（teacher との**保持率** = student_acc / teacher_acc）、②検索 Recall@k、③teacher 埋め込みとの**コサイン整合度**。①が落ちているのに③が高い、あるいはその逆、という乖離が出たら、正規化/`logit_scale`/テキスト埋め込みの不一致をまず疑う。デプロイ段（36/37 章）の ONNX 化・量子化と組み合わせるのが定石だ。

---

## 🛠 章末ミニプロジェクト（`mini_project.py`）

**課題:** teacher CLIP を x551 小さい student CNN に埋め込み蒸留し、**ゼロショット精度・検索 Recall@k・teacher 整合度の「保持率」**を一気通貫で評価せよ。蒸留前（ランダム student）との比較表と、棒グラフ `outputs/39_clip_distillation/mini_project_summary.png` を出力する。

完成形を実行すると次のような表が出る（CPU・合成データ・約数十秒）。

```
================ 評価テーブル ================
metric                   teacher   before KD    after KD
zero-shot acc              1.000       0.333       1.000
retrieval R@1              1.000       1.000       1.000
retrieval R@5              1.000       1.000       1.000
cosine align               1.000       0.033       0.986
agreement w/ teacher           -       0.333       1.000
---------------------------------------------
zero-shot 保持率 (student/teacher) = 100.0%
パラメータ圧縮: teacher画像タワー 87.8M -> student 160K (x551)
```

> 読み解きの勘所: **蒸留前でも検索 R@1/R@5 が 1.0** なのは、ランダム CNN でも「色」が支配的なので同一クラス画像どうしは互いに近いから。一方**ゼロショット精度と整合度は蒸留前は壊滅**（0.33 / 0.03）している——これは「画像どうしの相対的な近さ」と「teacher のテキスト空間との絶対的な整合」は別物で、後者こそ蒸留で獲得する信号だ、という重要な教訓だ。

---

## ✅ 到達チェックリスト

- [ ] `get_image_features().pooler_output` は**未正規化**だと知っており、比較前に `F.normalize` を掛けられる。
- [ ] `logits = logit_scale * normalize(i) @ normalize(t).T` の正準形を空で書ける。
- [ ] 埋め込み回帰蒸留 `(1 - cos) + MSE` を実装し、teacher を `eval()`+`inference_mode()` で凍結して前計算できる。
- [ ] 親和性蒸留の KL を `KL(log_softmax(student) ‖ softmax(teacher), batchmean)` の**正しい向き**で書ける。
- [ ] 正規化／`logit_scale` を揃えないとスケールがずれ softmax が平坦化することを数値で説明できる。
- [ ] TinyCLIP（重み継承＋親和性蒸留）と MobileCLIP（DataCompDR データセット強化）の違いを言える。
- [ ] `open_clip.list_pretrained()` で効率 CLIP のキーを確認し、CPU でロード（または概念代替）できる。
- [ ] ゼロショット精度・Recall@k・コサイン整合度の 3 軸で**保持率**を評価できる。
- [ ] 「知識蒸留／dataset distillation／dataset reinforcement」を取り違えない。

---

## ❓ 落とし穴・FAQ・デバッグ

**Q. 蒸留しても student のゼロショット精度が上がらない。**
A. まず正規化と `logit_scale` を teacher と完全に揃えているか確認。student 埋め込みを正規化し忘れている、`logit_scale` を student 側で掛けていない、teacher テキスト埋め込みを正規化していない、のいずれかが定番。`03` の `demo_scale_mismatch` を自分のコードに当てて行列のレンジを見る。

**Q. `get_image_features(...)` に `.shape` が無いと言われる。**
A. transformers v5 では戻り値が `BaseModelOutputWithPooling`。埋め込みは `.pooler_output` にある（[N, 512]、**未正規化**）。`model(**inputs).image_embeds` を使えば正規化済みで返る。

**Q. teacher の損失に勾配が流れて teacher が更新される / メモリが膨らむ。**
A. teacher は `model.eval()` の上で `torch.inference_mode()`（または `no_grad`）の中で呼ぶ。optimizer には **student のパラメータだけ**渡す（`AdamW(student.parameters())`）。埋め込みは一度前計算してキャッシュする。

**Q. MobileCLIP をロードしたら CPU で teacher より遅い。**
A. 仕様。MobileCLIP/FastViT はモバイル GPU/ANE 向けの再パラメータ化で速くなる設計で、x86 CPU の PyTorch eager では利得が出ない。CPU 最速を狙うなら、本章の小型 student のように**入力解像度とパラメータを両方小さくする**か、36/37 章の ONNX/量子化を併用する。

**Q. `open_clip.create_model('MobileCLIP-S1', pretrained='datacompdr')` がネット無しで失敗する。**
A. 重み（数百 MB）の DL が要る。オフラインなら `pretrained=None` でアーキだけ構築してパラメータ数・API を確認する（`04` はこの分岐を `try/except` で持つ）。重み未取得時は概念（TinyCLIP=重み継承＋親和性蒸留／MobileCLIP=DataCompDR）で代替する。

**Q. KL が `nan` になる / 学習が発散する。**
A. `F.kl_div` の**入力に `log_softmax`、ターゲットに `softmax`** を渡しているか（逆は壊れる）。`reduction='batchmean'`（既定の `'mean'` は要素数で割り数式とずれる）。`logit_scale` が大き過ぎる行列に追加の温度を二重掛けしていないかも確認。

**Q. student の入力前処理は teacher と同じにすべき？**
A. 必須ではない。student はもっと安い入力（本章は 64px・[-1,1] 正規化）でよく、**出力（埋め込み）を teacher に合わせる**のが蒸留。入力を小さくできることが student を速くする源泉でもある。

---

## 🚀 発展トピック・参考

- **TinyCLIP（ICCV 2023）**: 親和性蒸留（affinity mimicking）＋**重み継承（weight inheritance）**＋多段縮約で、CLIP を大幅小型化しつつゼロショットを保つ。本章 `03` の KL 蒸留がその中核。
- **MobileCLIP / MobileCLIP2（CVPR 2024 ほか）**: **DataCompDR** による dataset reinforcement（teacher のアンサンブル埋め込みと合成キャプションを事前計算しデータに焼き込む）で、アーキ更新より**データで**効率と精度を稼ぐ。`open_clip` 3.x が `MobileCLIP-S0/S1/S2/B` と `MobileCLIP2-*` を同梱。
- **対照蒸留（contrastive distillation）**: 実データ大規模学習では InfoNCE と teacher 模倣を併用する。本章 (a)/(b) の延長。
- **次の一手**: 蒸留した student を `36_onnx_runtime`/`37_runtime_edge_optimization` で ONNX 化・動的量子化し、`17_faiss_image_search`/`42_multimodal_vector_search` の検索基盤に載せると、エッジで動く CLIP 検索が完成する。効率CLIP の TorchScript 配布形（例: MobileCLIP）はまさにその実運用形。
- ドキュメント: open_clip <https://github.com/mlfoundations/open_clip> ／ transformers CLIP <https://huggingface.co/docs/transformers/model_doc/clip> ／ Apple ml-mobileclip <https://github.com/apple/ml-mobileclip>

---

## ▶ 動かし方

```bash
# 依存（CPU 前提。Linux は torch CPU ホイールが pyproject の index 経由で入る）
uv sync --group dl --group hf --group embed
# ※ distill グループは本章では自前実装のため必須ではない（torchdistill 等は発展用）

# 1) teacher の正準フロー（encode -> normalize -> logit_scale -> 類似度行列）
uv run python lectures/39_clip_distillation/01_teacher_similarity_matrix.py
# 2) 埋め込み回帰蒸留（この章の中心）
uv run python lectures/39_clip_distillation/02_embedding_distillation.py
# 3) 親和性(KL)蒸留 + スケールずれの落とし穴
uv run python lectures/39_clip_distillation/03_similarity_distillation_and_pitfall.py
# 4) open_clip で MobileCLIP/TinyCLIP・参考実装接続
uv run python lectures/39_clip_distillation/04_open_clip_mobileclip.py
# 章末ミニプロジェクト（保持率の総合評価）
uv run python lectures/39_clip_distillation/mini_project.py
# 演習（自己採点）と模範解答
uv run python lectures/39_clip_distillation/exercises.py
uv run python lectures/39_clip_distillation/exercises_solutions.py
```

図と重みは `outputs/39_clip_distillation/` に保存される（matplotlib は Agg、`imshow` は使わない）。初回のみ teacher CLIP の重みを HuggingFace から取得する（以降キャッシュ）。

---

> 版: **torch 2.12+cpu / torchvision 0.27+cpu / transformers 5.11 / open_clip 3.3**（2026-06）
> CPU 前提・`model.eval()` + `torch.inference_mode()`・学習は合成データの数十秒トイ。
> 前提: 38_knowledge_distillation（温度付き KD）／ 16_clip_zeroshot_retrieval（CLIP 正準フロー）。
