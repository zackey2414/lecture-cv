# 33_multimodal_embeddings: マルチモーダル埋め込みの拡張 — SigLIP / SigLIP2（多言語）と ImageBind の発想

> 前提: 第16回「CLIP ゼロショット & 検索」。CLIP の `get_image_features` / `get_text_features`、L2 正規化、コサイン類似度を一度触っていると理解が速い。

---

## 🎯 この章のゴール

- **共有埋め込み空間**という発想を、CLIP から **SigLIP / SigLIP2** へ広げて理解する。CLIP（softmax 対照損失）と SigLIP（sigmoid 損失）の **損失設計の違い** が、ゼロショット分類の **確率の読み方** にどう効くかを手で確かめられる。
- SigLIP2 の **多言語**埋め込みで画像テキスト検索を強化し、英語中心の SigLIP との差を **Recall@k** で定量化できる。「モデルを差し替えるだけで多言語対応が手に入る」移植性を体感する。
- **クロスモーダル検索**（テキスト→画像、そして概念として **音→画像**）を、共有空間 + FAISS で実装し、**Recall@k / retrieval mAP** で評価できる。ImageBind の「全モダリティを1つの空間に束ねる」発想を、重い git 依存なしのトイ空間で再現して理解する。
- 落とし穴（`get_*_features` は未正規化・v5 は `.pooler_output`・SigLIP は `padding="max_length"`・CLIP と SigLIP は次元が違って index を混ぜられない・sigmoid 絶対値はモデル間で比較不可）を避けられるようになる。

この章の成果物は **「多言語マルチモーダル検索エンジン」**（SigLIP2 + FAISS、Recall@k/mAP 評価、音→画像の拡張つき）です。`mini_project.py` が完成形として動きます。

---

## 1. 直感 — 「同じ空間」に写せば、別モーダルどうしを結べる

CLIP・SigLIP・SigLIP2 は、画像エンコーダとテキストエンコーダを **対照学習**で訓練したモデルです。学習のゴールはシンプルで、「対応する（画像, テキスト）の対は近く、無関係な対は遠く」なるよう、両エンコーダの出力を **同じベクトル空間**へ押し込むことにあります。訓練が終わると、画像も文も同じ次元のベクトルになり、その間の **コサイン類似度**が「どれだけ意味的に一致しているか」を表します。これが「画像を別モーダルと結びつけて扱う」一連のタスク（ゼロショット分類・画像テキスト検索・クロスモーダル検索）の唯一の土台です。

では、なぜこれがゼロショットで効くのでしょうか。分類器を学習しなくても、「`a photo of a red circle`」という**文**を埋め込み、手元の**画像**を埋め込んで、両者のコサインを測れば「この画像はこの文にどれだけ合うか」が分かります。候補ラベルを文にして並べ、最もコサインが高い文を選べば、それがゼロショット分類です。検索も同じ原理で、テキストのベクトルで画像の集合を引けば「文に合う画像」が、画像のベクトルで文の集合を引けば「画像に合う文」が出てきます。**埋め込みが同じ空間にある**という一点が、これら全部を成立させています。

この章で広げるのは2方向です。1つ目は **言語**。CLIP / SigLIP(base) は英語中心ですが、SigLIP2 は多言語トークナイザ（Gemma 系）と多言語コーパスで訓練されており、日本語・仏語・西語のクエリでも同じ概念の画像を引けます。2つ目は **モダリティ**。ImageBind は画像・テキストに加えて音声・深度などを「すべて画像（vision）を中心に束ねる」ことで、**音声クエリで画像を引く**といったクロスモーダル検索を可能にします。どちらも「共有空間を広げる」拡張だ、と捉えると見通しが良くなります。

---

## 2. 理論 — sigmoid 損失 vs softmax 対照損失、なぜ正規化するのか

CLIP の対照損失は **softmax（InfoNCE）型**です。1つのバッチに N 個の（画像, テキスト）対を集め、各画像について「N 個のテキストのうち、正しい1つを当てる」分類問題として解きます。したがって推論時も、候補ラベルのロジット `logits_per_image` に **softmax** を掛け、ラベル間で**和が 1 になる**確率にするのが正準です。これは「提示した候補の中でどれが最も近いか」を答える、**相互排他**の確率解釈になります。

SigLIP の損失は **sigmoid（pairwise binary）型**です。各（画像, テキスト）対を独立に「対応する/しない」の2値分類として扱い、バッチ全体の softmax 正規化を必要としません（だから巨大バッチでも安定して学習でき、精度・効率で有利）。推論時は `logits_per_image` に **sigmoid** を掛け、各ラベルが**独立に 0〜1**の確率になり、その和は 1 になりません。これは「**各ラベルが当てはまるか**」を独立に答える解釈で、「どれも当てはまらない（全部低い）」「複数当てはまる（多ラベル）」を自然に表現できます。`01_siglip_vs_clip.py` は、同じ画像・同じラベルに CLIP=softmax と SigLIP=sigmoid を掛け、確率の付き方の違いと、**取り違えると解釈が壊れる**様子（CLIP に sigmoid を当てると和が 1 でなくなり「候補内比較」が消える、など）を並べて見せます。

一方、検索や類似度計算では、向き（方向）だけが意味を持つので **コサイン類似度**を使います。`get_image_features` / `get_text_features` が返すベクトルは **未正規化**（ノルムが 1 ではない）なので、`F.normalize`（または `np.linalg.norm` で割る）で **L2 正規化**してから内積を取ると、それがコサインになります。FAISS では `IndexFlatIP`（内積）に**正規化済み**ベクトルを載せることで、コサイン最近傍検索を高速に行えます。評価は検索指標で行い、**Recall@k**＝「クエリの正解のうち上位 k 件に入った割合（取りこぼしの少なさ）」、**mAP@k**＝「順位まで考慮した適合率の平均（上位に正解を集める力）」を使います。正解が1クエリに複数あるとき（例: 同じ概念の画像が3枚）、Recall@1 は最大でも 1/3 にしかならない、という挙動も `04_recall_eval.py` で具体的に確認します。

---

## 3. 正準 API — SigLIP / SigLIP2 / CLIP を「同じ作法」で

3モデルとも `transformers` v5 の `AutoModel` / `AutoProcessor`（または専用クラス）で扱えます。埋め込みの取り出しも共通で、`get_image_features(pixel_values=...)` と `get_text_features(input_ids=...)` を使います。**v5 ではこれらが `BaseModelOutputWithPooling` を返すことがあり、射影後の埋め込みは `.pooler_output` に入ります**（旧版はテンソル直返し）。両対応にするヘルパを1つ用意しておくと壊れません。

```python
import torch
from transformers import AutoModel, AutoProcessor

mid = "google/siglip-base-patch16-224"           # SigLIP（英語中心, 768次元, sigmoid）
model = AutoModel.from_pretrained(mid).eval()      # 初回のみ重み DL
proc = AutoProcessor.from_pretrained(mid)

def pooled(out):                                   # v5: .pooler_output / 旧: テンソル
    return out.pooler_output if hasattr(out, "pooler_output") else out

# --- ゼロショット分類（生ロジット → sigmoid） ---
inputs = proc(images=[pil_image], text=labels,
              padding="max_length", return_tensors="pt")   # ★SigLIP は max_length 必須
with torch.inference_mode():
    logits = model(**inputs).logits_per_image              # (1, ラベル数)
    probs = torch.sigmoid(logits)                          # 各ラベル独立 0〜1（和≠1）

# --- 埋め込み（未正規化 → 自分で L2 正規化） ---
with torch.inference_mode():
    img = pooled(model.get_image_features(pixel_values=inputs["pixel_values"]))
    txt = pooled(model.get_text_features(input_ids=inputs["input_ids"]))
img = torch.nn.functional.normalize(img, p=2, dim=-1)      # ← これを忘れると罠
```

CLIP との違いは**前処理だけ**です。CLIP は `padding=True`（バッチ内最長）で詰め、テキスト特徴に `attention_mask` も渡します。SigLIP / SigLIP2 は `padding="max_length"`（既定 64 トークン）で詰めるのが正準で、テキスト特徴は `input_ids` だけで取ります。多言語にしたいときは **モデル ID を `google/siglip2-base-patch16-224` に変えるだけ**（処理コードは同一）。本章ではこの差を `siglip_lab.py` の `MMEncoder` に閉じ込め、`MMEncoder("clip" / "siglip" / "siglip2")` で切り替えられるようにしています。

FAISS への接続も最小です。正規化したベクトルを `faiss.IndexFlatIP(dim)` に `add` し、正規化したクエリで `search` するだけで済みます。これは第15/17/42回の大規模ベクトル検索とまったく同じ作法で、`02_siglip_retrieval.py` の index をディスク保存すれば、そのまま第42回の永続マルチモーダル検索に発展します。**注意: CLIP は 512 次元、SigLIP 系は 768 次元**なので、別モデルの埋め込みを同じ index に混ぜることはできません（モデルごとに index を分ける）。

---

## 4. 実装を1つずつ — スクリプトで段階的に組む

各スクリプトは独立に動き、初回のみ HuggingFace から重みを DL（以後キャッシュ）。結果は `outputs/33_multimodal_embeddings/` に保存されます。共通部品は `siglip_lab.py`（device 判定・合成データ・`MMEncoder`・FAISS 小道具・評価指標・トイ三モーダル空間）。

- **`01_siglip_vs_clip.py` — sigmoid vs softmax の核心**。赤い円1枚を、紛らわしいラベル群で CLIP と SigLIP に分類させ、softmax（和=1, 相互排他）と sigmoid（各ラベル独立）の確率の付き方を並べて表示。さらに「取り違えると解釈が壊れる」例も示す。図 `01_sigmoid_vs_softmax.png`。
- **`02_siglip_retrieval.py` — 埋め込みと検索の土台**。SigLIP で 36 枚の画像と 12 本の文を埋め込み、`get_*_features` が**未正規化**であることを確認 → L2 正規化 → `IndexFlatIP` で **text→image / image→text** 検索。図 `02_text_to_image.png`。
- **`03_multilingual_siglip2.py` — 多言語の強化**。同じ画像コレクションに対し、英・日・仏・西のクエリで Recall@1 を測り、英語中心の SigLIP と多言語の SigLIP2 を比較（日本語で SigLIP が落ち、SigLIP2 が取り戻す）。図 `03_multilingual_recall.png`。
- **`04_recall_eval.py` — 評価の正準**。text→image 検索を **Recall@k / mAP@k**（k=1,3,5）で CLIP と SigLIP について計測し、表・図・JSON に保存。正解が複数あるときの Recall の挙動も体感。`04_recall_metrics.json`。
- **`05_imagebind_concept.py` — 束ねる発想（音→画像）**。ImageBind を実行経路に入れず、概念ごとのアンカー + モダリティ別ノイズで「束ねられた共有空間」を再現し、**音声クエリ→画像**検索を Recall@k で評価。合成音は wav と波形図で具体化。本物の ImageBind を試す雛形は try/except でガード（未導入なら自動スキップ）。

```python
# MMEncoder で3モデルを同じ作法で扱う（siglip_lab.py）
from siglip_lab import MMEncoder, build_ip_index, search_index, evaluate_text_to_image

enc = MMEncoder("siglip2")                 # "clip" / "siglip" / "siglip2"
img_emb = enc.embed_images(images)          # (N, 768) 未正規化
txt_emb = enc.embed_texts(captions)         # (12, 768) 未正規化（多言語OK）
index = build_ip_index(img_emb)             # 内部で L2 正規化して IndexFlatIP に add
scores, ids = search_index(index, txt_emb, k=3)            # text -> image
ev = evaluate_text_to_image(txt_emb, img_emb, cap_metas, img_metas, k=5)
print(ev["recall_at_k"], ev["map_at_k"])
```

---

## 🛠 章末ミニプロジェクト — 多言語マルチモーダル検索エンジン

`mini_project.py` が完成形です。`MultilingualImageSearch` クラスが **SigLIP2 + FAISS** をまとめ、次を一気通貫で行います:

1. 合成画像コレクション（4色×3形×3枚＝36枚。`data/33_multimodal_embeddings/` に実画像があれば優先）を SigLIP2 で埋め込み、`IndexFlatIP` に登録。
2. **英・日・仏・西**の多言語クエリ「青い三角」で検索し、言語が違っても同じ画像が出ることを確認（上位ヒットを `mini_multilingual_query.png` に保存）。
3. 言語別に **Recall@k / mAP@k**（k=1,3,5）を計測し、`mini_metrics.json` に保存。
4. **拡張**: トイ共有空間で **音→画像** クロスモーダル検索の Recall@1 を測り、「共有空間さえあれば画像×テキストと同じ仕組みで音声も扱える」ことを示す。

```bash
uv run python lectures/33_multimodal_embeddings/mini_project.py
```

**腕試し（発展課題）**: ①`build_collection(variants=8)` に増やして Recall@k がどう動くか観察する。②`MultilingualImageSearch` を `MMEncoder("clip")` に差し替え、多言語クエリでの落差を Recall で比較する。③`search` を「画像→画像」検索に拡張する（クエリも `embed_images` にするだけ）。④`mini_metrics.json` を読み、言語×k のヒートマップを描く。

---

## ✅ 到達チェックリスト

- [ ] CLIP=softmax（相互排他, 和=1）と SigLIP=sigmoid（各ラベル独立, 和≠1）の **確率解釈の違い** を説明できる。
- [ ] `get_image_features` / `get_text_features` の出力は **未正規化**で、コサインには **L2 正規化**が要ると理解している。
- [ ] v5 では `get_*_features` が `.pooler_output` を持つこと、SigLIP は `padding="max_length"` が必要なことを知っている。
- [ ] 正規化ベクトル + `IndexFlatIP` で **text↔image** 検索ができ、第42回の永続化に繋げられる。
- [ ] **Recall@k / mAP@k** を自分で計算でき、正解が複数あるときの Recall の挙動を説明できる。
- [ ] SigLIP2 が **多言語**で SigLIP より強いことを Recall で示せる（モデル差し替えのみ）。
- [ ] ImageBind の「**全モダリティを vision に束ねる**」発想と、共有空間があれば **音→画像** 検索が回ることを説明できる。
- [ ] CLIP(512) と SigLIP(768) は **次元が違い index を混ぜられない**ことを知っている。

---

## ❓ 落とし穴・FAQ・デバッグ

- **`SiglipTokenizer requires the protobuf library`**: SigLIP / SigLIP2 のトークナイザは `sentencepiece` と **`protobuf`** が必要。`hf` グループに両方入れておく（`uv add --group hf sentencepiece protobuf`）。Donut や一部 T5 系も同じ。
- **コサインが妙に低い / ランキングが変**: `get_*_features` の出力を**正規化せず**に内積を取っているのが定番。`F.normalize(..., p=2, dim=-1)`（または FAISS 前に正規化）を必ず通す。なお SigLIP は sigmoid 学習ゆえコサインの**絶対値**は小さめに出る（0.1 前後でも正常）。**ランキングが正しいか**で判断する。
- **SigLIP でテキストがうまく入らない**: `padding=True` ではなく **`padding="max_length"`** が正準（既定 64 トークン）。テキスト特徴は `input_ids` だけで取り、`attention_mask` は渡さない（CLIP は逆に渡す）。
- **v5 で `get_*_features` がテンソルでなくオブジェクト**: `.pooler_output` を取る。`pooled = out.pooler_output if hasattr(out, "pooler_output") else out` の両対応ヘルパを使う。
- **sigmoid の絶対値でモデルを比較してしまう**: SigLIP と SigLIP2 では学習時のバイアス/温度が違い、sigmoid の**絶対値は直接比較できない**。モデル比較は **検索 Recall**（相対ランキング）で行う（`03` はこの方針）。
- **CLIP と SigLIP の埋め込みを同じ index に入れたい**: 次元が **512 vs 768** で不可能。さらに空間自体が別物なので意味的にも混ぜられない。モデルごとに index を分ける。
- **`pipeline("image-to-text")` や `pipeline("visual-question-answering")` が無い**: transformers v5 で削除。本章は埋め込み API（`get_*_features`）が中心なので影響しないが、関連タスクでは `image-text-to-text` を使う（第24/25回）。
- **ImageBind が import できない / 重い**: 公式 PyPI が無く git + torchaudio + ffmpeg 依存。本章は実行経路に入れず**概念デモ**で完結させる。`05` の本物雛形は未導入なら自動スキップ（exit 0）。

---

## 🚀 発展トピック・参考

- **高レベル API**: `sentence-transformers`（`SentenceTransformer("clip-ViT-B-32").encode(normalize_embeddings=True)` + `util.cos_sim`）や `open-clip-torch`（LAION 学習の OpenCLIP / MobileCLIP / SigLIP 系）でも同じ検索が書ける。実装比較に有用（任意・`embed` グループ）。
- **大規模化**: 本章の `IndexFlatIP` は全探索（厳密）。件数が増えたら `IndexIVFFlat` / `IndexHNSWFlat` で近似最近傍に切り替える（第17/42回）。ID とメタデータの対応表は別管理し、index と一緒に保存する。
- **本物の ImageBind（任意・実験的）**: 音声・画像・テキストを1空間に束ねる。

```python
# 実行経路には入れない概念スケッチ（git 依存・torchaudio/ffmpeg が必要）
# uv add --group imagebind "imagebind @ git+https://github.com/facebookresearch/ImageBind.git" torchaudio
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType
from imagebind import data
model = imagebind_model.imagebind_huge(pretrained=True).eval()
inputs = {
    ModalityType.VISION: data.load_and_transform_vision_data(image_paths, "cpu"),
    ModalityType.AUDIO:  data.load_and_transform_audio_data(audio_paths, "cpu"),
}
emb = model(inputs)                       # 各モダリティが同じ空間のベクトルに
sim = emb[ModalityType.AUDIO] @ emb[ModalityType.VISION].T   # 音 -> 画像
```

- **モデル選び**: 精度重視なら `siglip2-base-patch16-*` 系、軽さ重視なら CLIP base、多言語が要るなら必ず SigLIP2。検索が主目的なら埋め込み専用に蒸留した小型モデル（第39回 CLIP 蒸留）も選択肢。
- **公式ドキュメント**: [SigLIP](https://huggingface.co/docs/transformers/model_doc/siglip) / [SigLIP2](https://huggingface.co/docs/transformers/model_doc/siglip2) / [FAISS](https://github.com/facebookresearch/faiss/wiki) / [ImageBind](https://github.com/facebookresearch/ImageBind)。

---

## ▶ 動かし方

```bash
# 依存（未導入なら）: 深層学習・HuggingFace一式・FAISS
uv sync --group dl --group hf --group vector
# ※ SigLIP のトークナイザに sentencepiece / protobuf が必要（hf グループに含む）

# 本編（番号順）。初回のみ各モデルの重みを DL（以後キャッシュ）
uv run python lectures/33_multimodal_embeddings/01_siglip_vs_clip.py
uv run python lectures/33_multimodal_embeddings/02_siglip_retrieval.py
uv run python lectures/33_multimodal_embeddings/03_multilingual_siglip2.py
uv run python lectures/33_multimodal_embeddings/04_recall_eval.py
uv run python lectures/33_multimodal_embeddings/05_imagebind_concept.py

# 章末ミニプロジェクト（多言語マルチモーダル検索エンジンの完成形）
uv run python lectures/33_multimodal_embeddings/mini_project.py

# 演習（自己採点。未実装でも exit 0）と模範解答（全 PASS）
uv run python lectures/33_multimodal_embeddings/exercises.py
uv run python lectures/33_multimodal_embeddings/exercises_solutions.py
```

成果物（図・wav・JSON）は `outputs/33_multimodal_embeddings/` に保存される。
`data/33_multimodal_embeddings/` に実画像（`.png/.jpg`）を置くと、合成より優先して使われる。

---

> 参照ライブラリ: **torch 2.12+cpu** / **torchvision 0.27+cpu** / **transformers 5.11** / **faiss-cpu** / **diffusers 0.38**（生成は第31回）
> （モデル: SigLIP `google/siglip-base-patch16-224`・SigLIP2 `google/siglip2-base-patch16-224`・CLIP `openai/clip-vit-base-patch32`、headless OpenCV、matplotlib=Agg、CPU・`model.eval()`+`torch.inference_mode()`、ImageBind は概念のみ） — 2026-06