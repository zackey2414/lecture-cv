# 第24回 画像キャプション生成 入門 — BLIP / GIT / ViT-GPT2 と生成パラメータ・評価

> トラック: **マルチモーダル** ／ レベル: **入門** ／ 必要な依存グループ: `dl` `hf` `metrics`
>
> 前提知識: 第13〜16回（分類・埋め込み・CLIP）。CLIP の「画像とテキストを同じ空間に埋め込む」感覚を
> 使うと、本章の CLIPScore がすっと入ります。

---

## 🎯 この章のゴール

この章を終えると、次が「自力で再現できる」ようになります。

- 画像キャプション生成が **Vision エンコーダ + テキストデコーダ（Encoder–Decoder）** の系であることを説明できる。
- BLIP / GIT / ViT-GPT2 の **3アーキテクチャの違い**（cross-attention 型か、画像トークン接頭辞型か）を言える。
- transformers v5 の正準フロー **`processor(image) → model.generate → processor.decode`** を、`pipeline('image-text-to-text')` と `AutoModelForImageTextToText` 手書きの両方で書ける。
- **無条件キャプション**と、接頭辞を与える**条件付きキャプション**を実装できる。
- `model.generate` の **`num_beams` / `max_new_tokens` / `do_sample` / `temperature` / `repetition_penalty`** を使い分けて出力を制御できる。
- 生成文を **BLEU / ROUGE-L / CIDEr（主指標）** と、参照不要の **CLIPScore** で評価でき、各指標の長所・限界を説明できる。
- なぜ **CIDEr は珍しい n-gram を重視するのか**、なぜ **torchmetrics の CLIPScore が transformers v5 では現状動かないのか** といった落とし穴を理解している。

完成物は「複数の小型モデル × 生成設定でキャプションを付け、BLEU/CIDEr/CLIPScore のリーダーボードを出す比較ベンチマーク」（`mini_project.py`）です。

---

## 1. 直感 — キャプションとは「画像を読み、言葉で書く」こと

画像分類は「1024 個のラベルから1つ選ぶ」タスクでした。一方、キャプションはそうではありません。**任意の自然文を1語ずつ生成する**ため、出力空間は「語彙サイズ ^ 文長」という天文学的な広さに膨らみます。そのためキャプションモデルは、内部に必ず2つの部品を持ちます。

1. **Vision エンコーダ**: 画像を意味ベクトル（パッチ列の埋め込み）に変換する。ViT が定番です。
2. **テキストデコーダ**: そのベクトルを手がかりに、`<bos> a red car on …` と**自己回帰的（1語ずつ、前の語を見て次の語）**に文を綴る。GPT-2 のような言語モデルが担います。

つまりキャプションは「画像を読む CV」と「文を書く NLP」の合流点です。本章で使う3モデルも、違うのはこの2部品の**つなぎ方**だけ、と捉えると一気に整理できます。イメージとしては「**画像を見ながら文章を書く穴埋め**」であり、デコーダは毎ステップ「ここまで書いた文 ＋ 画像」を条件に、次の語を確率で選んでいきます。

<figure class="lec-fig"><svg viewBox="0 0 660 240" role="img" aria-label="画像をViTエンコーダで画像特徴に変換し、テキストデコーダが特徴を手がかりに文を1語ずつ自己回帰生成するパイプライン" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><rect x="26" y="46" width="110" height="66" fill="#dbeafe"/><rect x="26" y="112" width="110" height="44" fill="#ffedd5"/><circle cx="66" cy="88" r="15" fill="#f97316"/><line x1="62" y1="46" x2="62" y2="156" stroke="#ffffff" stroke-width="1.2" opacity="0.7"/><line x1="99" y1="46" x2="99" y2="156" stroke="#ffffff" stroke-width="1.2" opacity="0.7"/><line x1="26" y1="83" x2="136" y2="83" stroke="#ffffff" stroke-width="1.2" opacity="0.7"/><line x1="26" y1="120" x2="136" y2="120" stroke="#ffffff" stroke-width="1.2" opacity="0.7"/><rect x="26" y="46" width="110" height="110" fill="none" stroke="#71717a" stroke-width="1.5"/><text x="81" y="176" text-anchor="middle" font-size="13" fill="#52525b">入力画像</text><line x1="140" y1="101" x2="166" y2="101" stroke="#71717a" stroke-width="2"/><polygon points="174,101 164,96 164,106" fill="#71717a"/><rect x="178" y="58" width="108" height="86" rx="7" fill="#eff6ff" stroke="#2563eb" stroke-width="2"/><text x="232" y="96" text-anchor="middle" font-size="18" font-weight="700" fill="#1d4ed8">ViT</text><text x="232" y="120" text-anchor="middle" font-size="12" fill="#2563eb">エンコーダ</text><line x1="290" y1="101" x2="312" y2="101" stroke="#71717a" stroke-width="2"/><polygon points="320,101 310,96 310,106" fill="#71717a"/><rect x="326" y="52" width="34" height="15" fill="#dbeafe" stroke="#2563eb" stroke-width="1.3"/><rect x="326" y="75" width="34" height="15" fill="#dbeafe" stroke="#2563eb" stroke-width="1.3"/><rect x="326" y="98" width="34" height="15" fill="#dbeafe" stroke="#2563eb" stroke-width="1.3"/><rect x="326" y="121" width="34" height="15" fill="#dbeafe" stroke="#2563eb" stroke-width="1.3"/><rect x="326" y="144" width="34" height="15" fill="#dbeafe" stroke="#2563eb" stroke-width="1.3"/><text x="343" y="176" text-anchor="middle" font-size="13" fill="#52525b">画像特徴</text><line x1="362" y1="101" x2="396" y2="101" stroke="#ea580c" stroke-width="2"/><polygon points="404,101 394,96 394,106" fill="#ea580c"/><rect x="408" y="58" width="230" height="86" rx="8" fill="#fff7ed" stroke="#ea580c" stroke-width="2"/><text x="523" y="84" text-anchor="middle" font-size="14" font-weight="700" fill="#c2410c">テキストデコーダ</text><text x="523" y="110" text-anchor="middle" font-size="14.5" font-weight="700" fill="#18181b">a red car on the road</text><text x="523" y="132" text-anchor="middle" font-size="11" fill="#ea580c">→ 1語ずつ自己回帰で生成</text></svg><figcaption>キャプションモデルは <b>Vision エンコーダ＋テキストデコーダ</b> の2部品でできています。<b>ViT</b> が画像をパッチ列の<b>画像特徴</b>に変換し、<b>テキストデコーダ</b>がその特徴を手がかりに <code>a red car on …</code> と<b>1語ずつ自己回帰的に</b>文を綴ります。「画像を読む CV」と「文を書く NLP」の合流点です。</figcaption></figure>

この「1語ずつ確率で選ぶ」ところに、後半の主役 **生成パラメータ** が効いてきます。最も確率の高い語を貪欲に取るのか、複数候補を保持して全体の尤度を最大化するのか、わざと確率的に揺らして多様性を出すのか――同じモデルでも戦略次第で出力は大きく変わります。

---

## 2. 仕組み — 3つのアーキテクチャの違い

キャプションモデルの中身は「Vision エンコーダ」と「テキストデコーダ」の組み合わせですが、両者の**橋の架け方**で系統が分かれます。本章の3モデルは、ちょうど代表的な3方式に対応しています。

| モデル | クラス（v5 が割り当てる実体） | エンコーダ→デコーダの橋 | ひとことで |
|---|---|---|---|
| `Salesforce/blip-image-captioning-base` | `BlipForConditionalGeneration` | 画像特徴をデコーダの **cross-attention** に注入 | キャプション専用に設計・素直に強い |
| `nlpconnect/vit-gpt2-image-captioning` | `VisionEncoderDecoderModel`（ViT + GPT-2） | ViT 出力を GPT-2 の **cross-attention** に渡す | 既存の ViT と GPT-2 を「接着」した教科書的構成 |
| `microsoft/git-base` | `GitForCausalLM` | 画像トークンを**言語モデルの接頭辞**として連結 | デコーダのみ（causal LM）に画像を食わせる新しめの方式 |

BLIP と ViT-GPT2 は **Encoder–Decoder 型**で、デコーダの各層が cross-attention を通じて画像を「見ながら」書きます。一方 GIT は **デコーダのみ（decoder-only）型**で、画像をパッチトークン化して文章トークンの前に置き、ふつうの言語モデルとして続きを生成します。そして BLIP-2 や最近の VLM（第25回）は、この decoder-only 方向の発展形にあたります。

<figure class="lec-fig"><svg viewBox="0 0 620 312" role="img" aria-label="Encoder-Decoder型は画像特徴をデコーダのcross-attentionへ注入し、decoder-only型は画像トークンを接頭辞として文の前に連結する2方式の比較" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><text x="310" y="32" text-anchor="middle" font-size="13" font-weight="700" fill="#2563eb">① Encoder–Decoder 型（BLIP・ViT-GPT2）</text><text x="205" y="54" text-anchor="middle" font-size="11" fill="#ea580c">cross-attention</text><rect x="46" y="66" width="30" height="15" fill="#dbeafe" stroke="#2563eb" stroke-width="1.3"/><rect x="46" y="86" width="30" height="15" fill="#dbeafe" stroke="#2563eb" stroke-width="1.3"/><rect x="46" y="106" width="30" height="15" fill="#dbeafe" stroke="#2563eb" stroke-width="1.3"/><rect x="46" y="126" width="30" height="15" fill="#dbeafe" stroke="#2563eb" stroke-width="1.3"/><text x="61" y="156" text-anchor="middle" font-size="11.5" fill="#52525b">画像特徴</text><line x1="78" y1="74" x2="326" y2="74" stroke="#ea580c" stroke-width="1.5" stroke-dasharray="5 3"/><polygon points="334,74 324,69 324,79" fill="#ea580c"/><line x1="78" y1="100" x2="326" y2="100" stroke="#ea580c" stroke-width="1.5" stroke-dasharray="5 3"/><polygon points="334,100 324,95 324,105" fill="#ea580c"/><line x1="78" y1="126" x2="326" y2="126" stroke="#ea580c" stroke-width="1.5" stroke-dasharray="5 3"/><polygon points="334,126 324,121 324,131" fill="#ea580c"/><rect x="336" y="64" width="172" height="80" rx="8" fill="#fff7ed" stroke="#ea580c" stroke-width="2"/><text x="422" y="100" text-anchor="middle" font-size="14" font-weight="700" fill="#c2410c">テキストデコーダ</text><text x="422" y="122" text-anchor="middle" font-size="10.5" fill="#c2410c">（各層で cross-attn）</text><line x1="508" y1="104" x2="534" y2="104" stroke="#71717a" stroke-width="2"/><polygon points="542,104 532,99 532,109" fill="#71717a"/><text x="578" y="108" text-anchor="middle" font-size="11" font-weight="700" fill="#18181b">→ 文生成</text><line x1="30" y1="160" x2="590" y2="160" stroke="#e4e4e7" stroke-width="1"/><text x="310" y="186" text-anchor="middle" font-size="13" font-weight="700" fill="#c2410c">② decoder-only 型（GIT）</text><rect x="44" y="224" width="290" height="46" rx="6" fill="#fafafa" stroke="#d4d4d8" stroke-width="1.5"/><rect x="56" y="230" width="30" height="34" fill="#dbeafe" stroke="#2563eb" stroke-width="1.3"/><rect x="90" y="230" width="30" height="34" fill="#dbeafe" stroke="#2563eb" stroke-width="1.3"/><rect x="124" y="230" width="30" height="34" fill="#dbeafe" stroke="#2563eb" stroke-width="1.3"/><text x="105" y="218" text-anchor="middle" font-size="10.5" font-weight="700" fill="#2563eb">画像トークン（接頭辞）</text><rect x="166" y="230" width="150" height="34" rx="4" fill="#ffedd5" stroke="#f97316" stroke-width="1.3"/><text x="241" y="252" text-anchor="middle" font-size="12" fill="#c2410c">文トークン …</text><text x="189" y="296" text-anchor="middle" font-size="11" fill="#52525b">因果 LM が左→右に1系列を処理</text><line x1="336" y1="247" x2="470" y2="247" stroke="#71717a" stroke-width="2"/><polygon points="478,247 468,242 468,252" fill="#71717a"/><text x="540" y="251" text-anchor="middle" font-size="12" font-weight="700" fill="#18181b">→ 次の語（予測）</text></svg><figcaption>エンコーダとデコーダの<b>橋の架け方</b>で系統が分かれます。<b>① Encoder–Decoder 型</b>（BLIP・ViT-GPT2）は、別建ての画像特徴をデコーダ各層の <b>cross-attention</b> に注入して「見ながら」書きます。<b>② decoder-only 型</b>（GIT）は<b>画像トークンを接頭辞</b>として文トークンの前に連結し、1つの<b>因果言語モデル</b>が左→右に続きを生成します。</figcaption></figure>

実装上の重要な差は **前処理器（processor）**です。BLIP と GIT は `AutoProcessor` 一つで「画像＋テキストをまとめて」前処理できます。ところが ViT-GPT2 の `AutoProcessor` は**中身が GPT2Tokenizer だけ**で、画像を渡せません。そのため ViT-GPT2 では `AutoImageProcessor`（画像）＋ `AutoTokenizer`（文字）を**別々に**使う必要があります。もっとも、本章の `caption_helpers.Captioner` はこの差を `kind="combined" / "encdec"` で吸収しており、どのモデルでも `caption_images(cap, images, ...)` という同じ呼び方で使えるようにしてあります。

なぜ3つも触るのか――1つのモデルだけ覚えると「BLIP の作法」が「キャプション一般の作法」だと誤解しがちだからです。3つを並べると「**Auto クラスとパイプラインは共通、processor の組み立てだけモデル差がある**」という構造が見えてきます。

---

## 3. 正準API — transformers v5 のキャプション生成

最小フローはたった3手です。これがすべての土台になります。

```python
from transformers import AutoModelForImageTextToText, AutoProcessor
import torch

model_id = "Salesforce/blip-image-captioning-base"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForImageTextToText.from_pretrained(model_id).eval()  # ★eval を忘れない

inputs = processor(images=image, return_tensors="pt")          # 1) 前処理
with torch.inference_mode():                                    #    勾配オフ・高速
    out_ids = model.generate(**inputs, max_new_tokens=20)       # 2) 生成
caption = processor.decode(out_ids[0], skip_special_tokens=True)  # 3) 復号
```

<figure class="lec-fig"><svg viewBox="0 0 660 200" role="img" aria-label="transformers v5のキャプション生成は、processorで前処理しmodel.generateで生成しprocessor.decodeで復号する3手のフロー。3モデルとも共通" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><text x="330" y="30" text-anchor="middle" font-size="14" font-weight="700" fill="#18181b">transformers v5 キャプション生成の正準フロー（3手）</text><rect x="24" y="64" width="190" height="84" rx="8" fill="#fff7ed" stroke="#ea580c" stroke-width="2"/><rect x="235" y="64" width="190" height="84" rx="8" fill="#fff7ed" stroke="#ea580c" stroke-width="2"/><rect x="446" y="64" width="190" height="84" rx="8" fill="#fff7ed" stroke="#ea580c" stroke-width="2"/><text x="119" y="94" text-anchor="middle" font-size="15" font-weight="700" fill="#c2410c">① 前処理</text><text x="119" y="116" text-anchor="middle" font-size="12" fill="#71717a" font-family="'JetBrains Mono', monospace">processor(images)</text><text x="119" y="136" text-anchor="middle" font-size="11" fill="#52525b">画像 → テンソル</text><text x="330" y="94" text-anchor="middle" font-size="15" font-weight="700" fill="#c2410c">② 生成</text><text x="330" y="116" text-anchor="middle" font-size="12" fill="#71717a" font-family="'JetBrains Mono', monospace">model.generate</text><text x="330" y="136" text-anchor="middle" font-size="11" fill="#52525b">→ out_ids</text><text x="541" y="94" text-anchor="middle" font-size="15" font-weight="700" fill="#c2410c">③ 復号</text><text x="541" y="116" text-anchor="middle" font-size="12" fill="#71717a" font-family="'JetBrains Mono', monospace">processor.decode</text><text x="541" y="136" text-anchor="middle" font-size="11" fill="#52525b">→ キャプション文</text><line x1="214" y1="106" x2="229" y2="106" stroke="#71717a" stroke-width="2"/><polygon points="235,106 225,101 225,111" fill="#71717a"/><line x1="425" y1="106" x2="440" y2="106" stroke="#71717a" stroke-width="2"/><polygon points="446,106 436,101 436,111" fill="#71717a"/><text x="330" y="182" text-anchor="middle" font-size="12" fill="#3f3f46">★ BLIP / GIT / ViT-GPT2 すべて同じ3手（Auto クラスで共通）</text></svg><figcaption>transformers v5 のキャプション生成は、どのモデルでも <b>3手</b>で書けます。<code>processor(images)</code> で画像を<b>テンソルへ前処理</b> → <code>model.generate</code> で <b>out_ids を生成</b> → <code>processor.decode</code> で <b>文字列へ復号</b>、の順です。<b>BLIP・GIT・ViT-GPT2</b> はすべて <code>AutoModelForImageTextToText</code> ＋この同じ3手で扱え、生成は <code>torch.inference_mode()</code> 内で行います。</figcaption></figure>

ポイントは `AutoModelForImageTextToText` です。これが **v5 の正準クラス**で、BLIP も GIT も ViT-GPT2 も**同じ Auto クラス**で読み込めます（内部で各実体クラスに解決される）。`skip_special_tokens=True` を付けると `<bos>` や `<eos>` が落ちて読める文になります。複数画像なら `processor.batch_decode(out_ids, skip_special_tokens=True)` を使います。

⚠️ **transformers v5 の破壊的変更**（古いチュートリアルとの差）を押さえておきましょう。これを知らないと import の時点で詰みます。

- 旧 **`AutoModelForVision2Seq` は削除**されました（`import` するとエラー）。→ `AutoModelForImageTextToText` を使う。
- 旧 **`AutoFeatureExtractor` も廃止**。画像前処理は **`AutoImageProcessor`**（torchvision バックエンドの fast 実装のみ）。
- `pipeline('image-to-text')` に prompt を渡す旧用法は非推奨。**条件付き・対話は `pipeline('image-text-to-text')`** を使う。

`pipeline` を使えば前処理〜生成〜復号を1行に畳めます。ただし `image-text-to-text` は **`text` 引数が必須**で、BLIP の無条件キャプションは `text=""` を渡して得ます（BLIP はチャットテンプレートを持たないので、第25回の `messages` 形式ではなく `images=..., text=...` を直接渡します）。

```python
from transformers import pipeline
pipe = pipeline("image-text-to-text", model=model_id, device=-1)  # CPU は device=-1
print(pipe(images=image, text="", max_new_tokens=20)[0]["generated_text"])
```

`01_blip_caption.py` では、この**手書き3手**と**pipeline 1行**が実質同じ結果になることを並べて確認します。手書きで中身を理解してから pipeline で短く書く、という順序が定着の近道です。

---

## 4. 実装1 — 無条件と条件付きキャプション（`01_blip_caption.py`）

キャプションには2モードあります。

- **無条件**: 画像だけを渡し、モデルに自由に説明させる（`processor(images=image)`）。
- **条件付き**: 接頭辞 `text="a photo of"` を与え、その**続き**を書かせる（`processor(images=image, text="a photo of")`）。プロンプトで語り口を誘導したり、`"a photo of a"` のように冠詞まで固定して安定化させたりできます。

合成シーン（夕焼けの海・赤い車・木）に対する BLIP の実出力は次の通りで、合成画像でも驚くほど妥当です。

```
[sunset]  無条件: the sun is setting over the ocean
          条件付き 'a photo of' -> a photo of the sun setting over the ocean
[red_car] 無条件: a red truck driving down the road
[tree]    無条件: a tree in the middle of a field
```

条件付きは「出力フォーマットを揃えたい」「特定の観点（色・個数・場所）を言わせたい」ときに有効です。一方で**プロンプトに引きずられて画像にない要素を書く**こともある（hallucination）ので、後段の CLIPScore で「画像と本当に合っているか」を必ず確認します。なお ViT-GPT2 は条件付けに対応しない（text を渡しても無視される）ため、本章のヘルパは ViT-GPT2 では text を黙って無視します。

---

## 5. 実装2 — モデル比較と生成パラメータ制御（`02_vitgpt2_git.py`）

ここが本章のヤマです。**同じモデル・同じ画像でも、生成戦略（探索方法）で出力は大きく変わります。** 主要パラメータを整理します。

| パラメータ | 役割 | 直感 |
|---|---|---|
| `num_beams=1`, `do_sample=False` | **貪欲法 (greedy)** | 各ステップで最尤の1語。速い・決定的・**反復しがち** |
| `num_beams=4` | **ビームサーチ** | 上位 k 文を保持し、文全体の尤度を最大化。安定して質が高い |
| `do_sample=True`, `temperature`, `top_p` | **サンプリング** | 確率分布から抽選。多様だが品質が揺れる。`temperature` 大で奔放に |
| `repetition_penalty=1.3` | 反復抑制 | 既出語の確率を割り引く。`no_repeat_ngram_size=2` も有効 |
| `max_new_tokens` | 長さ上限 | 短すぎると尻切れ、長すぎると冗長・反復 |

<figure class="lec-fig"><svg viewBox="0 0 640 270" role="img" aria-label="次の語の確率分布に対し、貪欲は最尤1語、ビームは上位k文を保持、サンプリングは確率で抽選という探索戦略の違い" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><text x="150" y="32" text-anchor="middle" font-size="13" font-weight="700" fill="#18181b">次の語 確率分布</text><line x1="98" y1="46" x2="98" y2="170" stroke="#d4d4d8" stroke-width="1.5"/><rect x="98" y="52" width="170" height="22" fill="#ea580c"/><text x="90" y="68" text-anchor="end" font-size="13" font-weight="700" fill="#c2410c">a</text><rect x="98" y="82" width="98" height="22" fill="#ffedd5" stroke="#f97316" stroke-width="1.3"/><text x="90" y="98" text-anchor="end" font-size="12" fill="#52525b">the</text><rect x="98" y="112" width="58" height="22" fill="#ffedd5" stroke="#f97316" stroke-width="1.3"/><text x="90" y="128" text-anchor="end" font-size="12" fill="#52525b">red</text><rect x="98" y="142" width="24" height="22" fill="#ffedd5" stroke="#f97316" stroke-width="1.3"/><text x="90" y="158" text-anchor="end" font-size="12" fill="#52525b">blue</text><line x1="300" y1="24" x2="300" y2="250" stroke="#e4e4e7" stroke-width="1"/><rect x="320" y="46" width="300" height="56" rx="6" fill="#fff7ed" stroke="#ea580c" stroke-width="1.5"/><text x="336" y="70" font-size="13" font-weight="700" fill="#c2410c">貪欲 greedy</text><text x="336" y="92" font-size="11" fill="#3f3f46">最尤の1語・速い・反復しがち</text><rect x="320" y="110" width="300" height="56" rx="6" fill="#eff6ff" stroke="#2563eb" stroke-width="1.5"/><text x="336" y="134" font-size="13" font-weight="700" fill="#1d4ed8">ビーム beam=k</text><text x="336" y="156" font-size="11" fill="#3f3f46">上位 k 文を保持・安定して高品質</text><rect x="320" y="174" width="300" height="56" rx="6" fill="#fafafa" stroke="#71717a" stroke-width="1.5"/><text x="336" y="198" font-size="13" font-weight="700" fill="#18181b">サンプリング</text><text x="336" y="220" font-size="11" fill="#3f3f46">確率で抽選・多様（要 seed 固定）</text></svg><figcaption>同じモデル・同じ確率分布でも、<b>次の語の選び方</b>で出力は変わります。<b>貪欲(greedy)</b> は確率最大の1語を取り速いが反復しがち、<b>ビーム(beam)</b> は上位 k 文を保持して文全体の尤度を上げ安定、<b>サンプリング</b> は確率に従って抽選するので多様だが揺れます（再現には <code>torch.manual_seed</code> を固定）。</figcaption></figure>

実際に ViT-GPT2（GPT-2 デコーダは反復が出やすく教材向き）で観察すると、次のように戦略差がはっきり出ます。

```
[greedy            ] a blurry photo of a blue and yellow sun
[beam4             ] a close up picture of an orange and blue sky
[beam4_reppen      ] a close up picture of an orange and blue sky
[max_new_tokens=8  ] a blurry photo of a blue and yellow   ← 尻切れ
[sampling t=1.3] seed0: black and white image shows a blue sky
               seed1: a close up picture of a bright blue vase
               seed2: a green cloudy blue cloud hangs on a yellow backdrop
```

サンプリングは乱数を使うので、**再現性のために必ず `torch.manual_seed(...)` を固定**します（seed を変えると毎回違う文＝多様性が出る）。実務では「説明文を1つ確定したい」なら beam、「候補を複数出して人に選ばせたい／データ拡張したい」ならサンプリング、という使い分けになります。

バッチ推論も覚えましょう。`processor(images=[img1, img2, ...])` で複数画像をまとめて前処理し、`model.generate` 後に `processor.batch_decode(...)` で一括復号します。CPU でも画像数ぶんの for ループより速く、`02` のモデル比較はこのバッチ経路で動いています。

---

## 6. 実装3 — 評価（`03_caption_metrics.py`）

「良いキャプション」を数値化するのが評価です。2系統あります。

### 6.1 参照あり指標 — 人手キャプションにどれだけ近いか

- **BLEU**: 候補と参照の **n-gram 適合率**（precision）の幾何平均に、短すぎる文を罰する **Brevity Penalty** を掛ける。`modified precision` は同じ語の水増しを参照側の最大数で**クリップ**して防ぐのがミソ。機械翻訳由来で、**言い換えに厳しく**短文では 0 になりがち。
- **ROUGE-L**: **最長共通部分列 (LCS)** ベースの F値。語順を保った重なりを見る。recall 寄りで BLEU より緩い。
- **CIDEr（本章の主指標）**: 各 n-gram を **TF-IDF で重み付け**し、候補ベクトルと参照ベクトルの**コサイン類似度**を n=1..4 で平均する。**「誰でも言う一般語（a, photo, of）は軽く、その画像特有の珍しい語の一致を重く」**評価するのが核心。キャプション専用に設計され、人間評価との相関が高い。

CIDEr の直感は「珍しさ＝情報量」です。`a photo of` が一致しても情報はほぼゼロ、`red car` や `sunset` が一致したら本質を当てている、という重み付けを **IDF（参照コーパス全体での出現の少なさ）**で実現します。

```python
import caption_metrics as M
# 同じ参照 "a red car on the road" に対する3候補
#   exact      : BLEU=1.000 ROUGE=1.000 CIDEr=1.000
#   paraphrase : BLEU=0.000 ROUGE=0.462 CIDEr=0.077  ← BLEU は言い換えに厳しい
#   wrong      : BLEU=0.000 ROUGE=0.000 CIDEr=0.000
```

この表が示すのは「**指標ごとに性格が違う**」こと。BLEU は言い換え（"vehicle" vs "car"）を 0 と切り捨てますが、ROUGE/CIDEr は中間点を付けます。だから単一指標を盲信せず、複数を併記するのが定石です。

⚠️ **CIDEr の重要な注意**: IDF は本来「データセット全体の参照コーパス」から求める量です。評価対象が1枚だけ、あるいは参照がほぼ同一だと、すべての n-gram が「全文書に出現」して **IDF=0 に潰れ、CIDEr が常に 0** になります。本章の `compute_cider(..., idf_corpus=...)` は、実運用に倣って IDF を別コーパスから計算できるようにしています。

### 6.2 参照なし指標 — 画像と本当に合っているか

- **CLIPScore**: CLIP で画像とキャプションを埋め込み、**コサイン類似度 × 100** を取る（負は 0 にクリップ）。**人手の参照キャプションが要らない**のが最大の利点で、アノテーションのない画像でも品質の当たりが付けられます。

<figure class="lec-fig"><svg viewBox="0 0 560 280" role="img" aria-label="画像とキャプションをCLIPで同一空間の単位ベクトルに埋め込み、両者の角度θのコサインを100倍してCLIPScoreとする" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><text x="290" y="26" text-anchor="middle" font-size="12.5" font-weight="700" fill="#18181b">画像と文を CLIP で同じ空間に埋め込む</text><circle cx="160" cy="160" r="104" fill="#fafafa" stroke="#d4d4d8" stroke-width="1.5"/><line x1="160" y1="160" x2="209" y2="68" stroke="#2563eb" stroke-width="2"/><circle cx="209" cy="68" r="5" fill="#2563eb"/><text x="216" y="62" font-size="12" font-weight="700" fill="#1d4ed8">画像</text><line x1="160" y1="160" x2="240" y2="93" stroke="#ea580c" stroke-width="2"/><circle cx="240" cy="93" r="5" fill="#ea580c"/><text x="247" y="96" font-size="12" font-weight="700" fill="#c2410c">良い文</text><line x1="160" y1="160" x2="75" y2="100" stroke="#dc2626" stroke-width="1.8" stroke-dasharray="5 3"/><circle cx="75" cy="100" r="5" fill="#dc2626"/><text x="70" y="92" text-anchor="middle" font-size="11" font-weight="700" fill="#dc2626">ずれた文</text><path d="M 176 130 A 34 34 0 0 1 187 138" fill="none" stroke="#71717a" stroke-width="1.4"/><text x="182" y="122" text-anchor="middle" font-size="12" font-weight="700" fill="#3f3f46">θ</text><circle cx="160" cy="160" r="4" fill="#18181b"/><text x="312" y="92" font-size="12" fill="#3f3f46">cos θ ＝ 近さ（類似度）</text><text x="312" y="126" font-size="14" font-weight="700" fill="#c2410c">CLIPScore = 100 · cos θ</text><text x="312" y="148" font-size="10.5" fill="#52525b">（負値は 0 にクリップ）</text><text x="312" y="186" font-size="11.5" font-weight="700" fill="#15803d">θ 小 → 高スコア（一致）</text><text x="312" y="212" font-size="11.5" font-weight="700" fill="#dc2626">θ 大 → 低スコア（hallucination）</text></svg><figcaption><b>CLIPScore</b> は、画像とキャプションをそれぞれ <b>CLIP</b> で<b>同じ埋め込み空間</b>の単位ベクトルに変換し、その<b>コサイン類似度</b>を測ります。<code>CLIPScore = 100 · cos θ</code>（負は 0 にクリップ）。角度 θ が<b>小さいほど一致</b>して高スコア、画像とずれた文は θ が大きく低スコアになります。<b>人手の参照キャプションが不要</b>なのが利点です。</figcaption></figure>

```python
# CLIPScore = max(0, 100 · cos(image_embed, text_embed))
img_emb = F.normalize(clip.get_image_features(**px).pooler_output, dim=-1)
txt_emb = F.normalize(clip.get_text_features(**tx).pooler_output, dim=-1)
score = ((img_emb * txt_emb).sum(-1) * 100).clamp(min=0)
```

⚠️ **落とし穴（本章で実際に踏む）**: `torchmetrics.multimodal.CLIPScore` は内部で `get_image_features()` の戻り値を**テンソル前提**で `.norm()` を呼びますが、transformers v5 では `BaseModelOutputWithPooling`（本体は `.pooler_output`）が返るため、**現状そのままでは `AttributeError` で落ちます**。原理は上のように単純なので、本章では `caption_metrics.clip_score_pairs` として**手で実装**しています（`get_*_features(...).pooler_output` を取り、`F.normalize` してからコサイン）。これは第16回 CLIP で学んだ「`get_*_features` は v5 では output オブジェクト・かつ未正規化」という知識の直接の応用です。

### 6.3 パラメータとスコアの関係

`num_beams` を振るとスコアが動きます。合成3シーンでの実測例:

```
num_beams |   BLEU  ROUGE_L   CIDEr | CLIPScore(mean)
        1 |  0.000    0.667   0.239 | 30.75
        3 |  0.458    0.729   0.279 | 30.39
        5 |  0.252    0.610   0.268 | 32.08
```

beam=3 が参照あり指標で最良、beam=5 は CLIPScore が最良、という具合に**指標によって最適設定が違う**ことが読み取れます。「どの指標で最適化するか」を先に決めるのが実験設計の第一歩です。

> METEOR / SPICE は Java 実行環境（と外部 jar）が必要なため本講座では扱いません（概念のみ）。SPICE は「シーングラフ（物体・属性・関係）」の一致を見る指標で、CIDEr と併用されることが多い、という事実だけ知っておけば十分です。

---

## 7. 実務での使い分け

CPU で現実的に動く小型モデルの指針です。まず **BLIP-base** を基準に置き、必要に応じて差し替えるのが安全です。

| 状況 | 推奨 |
|---|---|
| とりあえず良いキャプションが欲しい | **BLIP-base**（素直に強い・条件付け可） |
| decoder-only 型の挙動を見たい | **GIT-base**（短く端的な傾向） |
| ViT + GPT-2 の接着構成を学びたい | **ViT-GPT2**（反復・揺れの観察に好適） |
| 高品質・指示追従・対話が必要 | BLIP-2 / 各種 VLM（**GPU 推奨**。第25回） |
| 参照キャプションが無い | **CLIPScore** で評価・スクリーニング |

CPU 運用の勘所: 推論は必ず `model.eval()` ＋ `torch.inference_mode()`。CPU では `float16` は遅い/未対応 op が多いので **`float32` のまま**使う（`torch_dtype` を指定しない）。生成は **`max_new_tokens` を小さく**（20 前後）すれば数秒で終わります。`BLIP-2 (2.7b)` は約15GB級でメモリ・時間ともに CPU 非現実的なので、本章の既定からは外しています。

---

## 🛠 章末ミニプロジェクト — キャプション・ベンチマーク（`mini_project.py`）

本章の全要素を1本に統合します。

1. 合成ギャラリー（実画像があれば `data/24_image_captioning/` を優先）に対し、
2. **3モデル（BLIP/GIT/ViT-GPT2）× 2設定（greedy/beam4）= 6通り**でキャプションを生成し、
3. **参照あり（BLEU・ROUGE-L・CIDEr）＋参照なし（CLIPScore）**で採点し、
4. **CIDEr 降順のリーダーボード**（図・JSON・テキスト）を出力する。

<figure class="lec-fig"><svg viewBox="0 0 660 210" role="img" aria-label="ミニプロジェクトは、画像ギャラリーを入力し、3モデル×2設定の6通りでキャプション生成し、参照ありと参照なしの指標で採点し、CIDEr降順のリーダーボードを出力する4ステップ" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><text x="330" y="30" text-anchor="middle" font-size="14" font-weight="700" fill="#18181b">ミニプロジェクト — 入力から リーダーボードまで4ステップ</text><rect x="14" y="68" width="146" height="88" rx="8" fill="#eff6ff" stroke="#2563eb" stroke-width="2"/><rect x="176" y="68" width="146" height="88" rx="8" fill="#fff7ed" stroke="#ea580c" stroke-width="2"/><rect x="338" y="68" width="146" height="88" rx="8" fill="#fff7ed" stroke="#ea580c" stroke-width="2"/><rect x="500" y="68" width="146" height="88" rx="8" fill="#dbeafe" stroke="#1d4ed8" stroke-width="2"/><text x="87" y="98" text-anchor="middle" font-size="13.5" font-weight="700" fill="#1d4ed8">① ギャラリー</text><text x="87" y="120" text-anchor="middle" font-size="10.5" fill="#52525b">合成 or 実画像</text><text x="87" y="138" text-anchor="middle" font-size="10.5" fill="#52525b">data/ を優先</text><text x="249" y="98" text-anchor="middle" font-size="13.5" font-weight="700" fill="#c2410c">② 6通り生成</text><text x="249" y="120" text-anchor="middle" font-size="10.5" fill="#52525b">3モデル × 2設定</text><text x="249" y="138" text-anchor="middle" font-size="10.5" fill="#52525b">greedy / beam4</text><text x="411" y="98" text-anchor="middle" font-size="13.5" font-weight="700" fill="#c2410c">③ 採点</text><text x="411" y="120" text-anchor="middle" font-size="10.5" fill="#52525b">BLEU・ROUGE-L</text><text x="411" y="138" text-anchor="middle" font-size="10.5" fill="#52525b">CIDEr ＋ CLIPScore</text><text x="573" y="98" text-anchor="middle" font-size="13.5" font-weight="700" fill="#1d4ed8">④ リーダーボード</text><text x="573" y="120" text-anchor="middle" font-size="10.5" fill="#52525b">CIDEr 降順</text><text x="573" y="138" text-anchor="middle" font-size="10.5" fill="#52525b">図 / JSON / txt</text><line x1="160" y1="112" x2="170" y2="112" stroke="#71717a" stroke-width="2"/><polygon points="176,112 166,107 166,117" fill="#71717a"/><line x1="322" y1="112" x2="332" y2="112" stroke="#71717a" stroke-width="2"/><polygon points="338,112 328,107 328,117" fill="#71717a"/><line x1="484" y1="112" x2="494" y2="112" stroke="#71717a" stroke-width="2"/><polygon points="500,112 490,107 490,117" fill="#71717a"/><text x="330" y="190" text-anchor="middle" font-size="11.5" fill="#3f3f46">参照キャプションが無ければ主指標は自動で CLIPScore に切替（has_references 判定）</text></svg><figcaption><b>章末ミニプロジェクト</b>は、<b>① 画像ギャラリー</b>（合成、または <code>data/24_image_captioning/</code> の実画像を優先）→ <b>② 6通り生成</b>（<b>3モデル × 2設定</b>＝BLIP/GIT/ViT-GPT2 を greedy と beam4 で）→ <b>③ 採点</b>（参照あり <b>BLEU・ROUGE-L・CIDEr</b> ＋ 参照なし <b>CLIPScore</b>）→ <b>④ リーダーボード出力</b>（CIDEr 降順・図/JSON/テキスト）の順に流れます。実データに参照が無ければ、主指標は自動で <b>CLIPScore</b> に切り替わります。</figcaption></figure>

```
=== リーダーボード（CIDEr 降順） ===
  1. blip/beam4    BLEU=0.252 CIDEr=0.268 CLIP=32.08 <= BEST
  2. git/greedy    BLEU=0.420 CIDEr=0.243 CLIP=30.20
  3. blip/greedy   BLEU=0.000 CIDEr=0.239 CLIP=30.75
  ...
  6. vitgpt2/greedy ...                    CLIP=25.10
```

ねらいは「**どのモデル × どの探索戦略が、この題材で一番マシか**を一望できる比較表を作る」こと。実データ投入時に参照が無ければ、主指標は自動で CLIPScore に切り替わります（`has_references` 判定）。出力は `outputs/24_image_captioning/mini_*.{png,json,txt}` に保存されます。

---

## ✅ 到達チェックリスト

自分の言葉で説明でき、コードを書ける状態を目指します。

- [ ] キャプションが **Vision エンコーダ + 自己回帰テキストデコーダ** であることを説明できる。
- [ ] BLIP（cross-attention）/ ViT-GPT2（VisionEncoderDecoder）/ GIT（画像トークン接頭辞・decoder-only）の違いを言える。
- [ ] `AutoModelForImageTextToText` ＋ `processor → generate → decode` を手書きできる。
- [ ] `pipeline('image-text-to-text')` を `text=""` で無条件キャプションに使える。
- [ ] 旧 `AutoModelForVision2Seq` / `AutoFeatureExtractor` が **v5 で廃止**されたことを知っている。
- [ ] 無条件と条件付き（接頭辞）の違いを実装できる。
- [ ] greedy / beam / sampling / `repetition_penalty` / `max_new_tokens` の効果を説明・制御できる。
- [ ] `batch_decode` で複数画像を一括処理できる。
- [ ] BLEU（modified precision ＋ BP）を自分で実装できる。
- [ ] CIDEr が **TF-IDF n-gram コサイン**であり、IDF が**コーパス依存**だと説明できる。
- [ ] CLIPScore を `get_*_features().pooler_output` ＋正規化で**手実装**できる（torchmetrics 版が v5 で壊れる理由も）。
- [ ] 指標ごとに最適な生成設定が変わりうることを、ベンチマークで示せる。

---

## ❓ よくある落とし穴・FAQ・デバッグ

**Q. `ImportError: cannot import name 'AutoModelForVision2Seq'`**
A. v5 で削除されました。`AutoModelForImageTextToText` を使ってください（本章の3モデルすべてこれで読めます）。

**Q. `pipeline('image-text-to-text')` が `ValueError: You must provide text` で落ちる**
A. このパイプラインは `text` 必須です。無条件キャプションは `pipe(images=img, text="")`。なお BLIP はチャットテンプレートを持たないので、`messages`（role/content）形式を渡すと `Cannot use apply_chat_template ...` になります（messages 形式は第25回の VLM 用）。

**Q. ViT-GPT2 で `processor(images=...)` が `You need to specify either text ...` で落ちる**
A. ViT-GPT2 の `AutoProcessor` は中身が GPT2Tokenizer だけで画像を扱えません。`AutoImageProcessor`（画像）と `AutoTokenizer`（文字）を**別々に**使ってください。本章は `caption_helpers.load_captioner("vitgpt2")` がこれを吸収します。

**Q. `torchmetrics` の `CLIPScore` が `AttributeError: 'BaseModelOutputWithPooling' object has no attribute 'norm'`**
A. transformers v5 で `get_image_features` の戻り値が output オブジェクトに変わったのが原因。本章のように `caption_metrics.clip_score_pairs`（`.pooler_output` ＋ `F.normalize` ＋コサイン×100）で**自前実装**するのが確実です。

**Q. CIDEr がいつも 0 になる**
A. IDF が潰れています。1枚だけ／参照がほぼ同一だと全 n-gram の IDF=0 になります。`idf_corpus` に多様な参照を渡す（＝データセット全体で IDF を計算する）ことで解決します。

**Q. キャプションが同じ単語を繰り返す（"a a a ..." や "blue blue blue"）**
A. greedy の典型症状。`num_beams>1`、`repetition_penalty=1.2〜1.5`、`no_repeat_ngram_size=2〜3` を試してください。

**Q. サンプリングの結果が毎回変わって比較できない**
A. 仕様です（多様性）。比較・再現したいときは直前に `torch.manual_seed(0)` を呼びます。

**Q. CPU で遅い／落ちる**
A. `max_new_tokens` を小さく、`num_beams` を控えめに。`float16` を指定しない（CPU は `float32`）。BLIP-2 など大型は GPU 前提。`model.eval()` と `torch.inference_mode()` を忘れない。

**Q. 初回だけ時間がかかる**
A. モデル重みの初回ダウンロードです。`HF_HOME` をキャッシュにして2回目以降を高速化、再現実行は `HF_HUB_OFFLINE=1` を活用します。

---

## 🚀 発展トピック・参考

- **BLIP-2 / InstructBLIP**: 凍結した画像エンコーダ・LLM の間に軽量な **Q-Former** を挟む方式。指示追従が強い反面、CPU には重い（GPU 推奨）。decoder-only VLM 全般は第25回で扱います。
- **CIDEr-D**: 本章の簡易版に「×10」「文長のガウス罰」「stemming」を加えた標準版。長さや繰り返しのごまかしに強い。実運用では `pycocoevalcap` の実装が定番です。
- **CLIPScore の発展**: 参照込みの **RefCLIPScore**、より新しい **PAC-S** など。参照なし指標は hallucination 検出にも使われます。
- **指標の限界**: n-gram 系は語順・言い換え・含意に弱い。最近は **LLM-as-a-judge** や学習型指標（BERTScore など）も併用されます。指標は「人間評価の安価な代理」であって真の正解ではない、という姿勢が大切です。
- 公式ドキュメント:
  - transformers Image-Text-to-Text タスク: https://huggingface.co/docs/transformers/en/tasks/image_text_to_text
  - BLIP: https://huggingface.co/docs/transformers/en/model_doc/blip
  - GIT: https://huggingface.co/docs/transformers/en/model_doc/git
  - VisionEncoderDecoder: https://huggingface.co/docs/transformers/en/model_doc/vision-encoder-decoder
  - torchmetrics（text / multimodal）: https://lightning.ai/docs/torchmetrics/stable/

---

## 💡 実践ユースケース集

この章のキャプション生成は「画像を見て言葉で説明する」基礎力です。現場では次のような
「説明文を量産・再利用する」用途で効いてきます。2〜3個の応用と、すぐ動かせる出発点
（`use_case.py`）を載せます。

### 1.（同梱・動く）自動 alt-text / 画像説明ジェネレータ（`use_case.py`）

- **何に使うか**: ブログ・EC・社内ドキュメントに大量の画像を載せるとき、アクセシビリティ用の
  `<img alt="...">`（スクリーンリーダーが読み上げる代替テキスト）を、1枚ずつ手書きする代わりに
  **下書きを自動生成 → 人が確認・修正**する。SEO の画像 alt 自動付与にも転用できます。
- **作り方の要点**: BLIP で各画像にキャプション → alt-text 作法（冗長な "a photo of" を削る／
  文頭大文字化＋末尾ピリオド／長さ目安 125 文字）で整形 → **CLIPScore（§6.2）で画像と説明の
  整合度を測り、低いものを「要レビュー」に flag** → JSON・自己完結 HTML ギャラリー・サイドカー
  `.txt` として書き出す。`mini_project.py`（評価ベンチ）とは別物で、こちらは**そのまま貼れる成果物**
  を作る実ツールです。
- **注意**: 自動生成 alt は **hallucination（画像にない記述）** を起こしうるので、`alt=""`
  （装飾画像扱い）にせず必ず内容を入れ、CLIPScore が低い／長すぎる画像は人手確認に回すこと。
  誤った alt は「無し」より有害です。

```bash
# 既定（BLIP・ビームサーチ）で全画像に alt-text を付ける
uv run python lectures/24_image_captioning/use_case.py
# モデルを切り替えて挙動を比べる（blip / git / vitgpt2）
uv run python lectures/24_image_captioning/use_case.py git
```

- **実データの置き方**: `data/24_image_captioning/` に自分の `.png/.jpg` を置くだけで、その
  フォルダが対象になります（参照キャプション不要。CLIPScore は参照なしで動くため品質スクリーニングも
  そのまま機能）。画像が無ければ合成シーン（夕焼け/赤い車/木）に自動フォールバックして必ず完走します。
- **出力**: `outputs/24_image_captioning/` に `use_case_alt_text.json`（台帳）・`use_case_gallery.html`
  （画像を base64 埋め込みした実 `<img alt>` 付き・ブラウザで読み上げ確認可）・`use_case_preview.png`・
  `alt_text/<name>.alt.txt`（CMS が読むサイドカー形式）。
- **拡張アイデア**: 多言語 alt（英語生成＋翻訳で日本語併記）／詳細版 longdesc（`max_new_tokens` 増）の
  出し分け／CLIPScore がしきい値未満の画像だけ「要レビュー」キューに出して人手に回す品質ゲート／
  WordPress・microCMS の API へ alt を一括 PUT する薄いアダプタ。

### 2. 商品画像の説明文・検索タグ自動生成（EC カタログ）

- **何に使うか**: EC サイトの大量の商品画像に、一覧用の短い説明やサムネイル代替テキスト、検索用タグの
  下書きを付ける。新規入荷のたびに人手で書く負担を下げ、表記ゆれも減らせます。
- **作り方の要点**: 条件付きキャプション（§4、`text="a photo of a"`）で**語り口・観点を固定**して
  フォーマットを揃え、`02` で学んだ `num_beams`/`repetition_penalty` で安定した1文に寄せる。
  生成文を名詞・色などへ簡易分割すれば検索タグの種になります。
- **注意**: ブランド名・型番・素材など**画像から読めない属性**はキャプションに出ません。商品 DB の
  メタデータと必ず突き合わせ、生成文は「見た目の説明」に限定して使うのが安全です。

### 3. 写真ライブラリの一括キャプション付け＆self-retrieval 整理

- **何に使うか**: 撮りためた写真・素材フォルダに説明文を一括付与して、後から「言葉で探せる」状態に
  整える。第16/17回の CLIP 検索や FAISS と組み合わせれば、キャプションを介した素材管理になります。
- **作り方の要点**: バッチ推論（§5、`processor(images=[...])`＋`batch_decode`）で全画像を一括処理し、
  各画像にキャプションを JSON で保存。CLIPScore で「説明が画像と合っているか」を採点して、低スコアの
  ものだけ再生成（beam を増やす等）すると品質を底上げできます。
- **注意**: 似た構図の写真には似たキャプションが付きがちで識別子になりにくい。撮影日・場所などの
  メタデータを併用し、キャプションは「中身の手がかり」と割り切ること。CPU では枚数に比例して時間が
  かかるので、`max_new_tokens` を小さく保ち夜間バッチに回すのが現実的です。

---

## ▶ 動かし方（コマンド）

```bash
# 依存グループをインストール（初回のみ）
uv sync --group dl --group hf --group metrics

# 各スクリプト（結果は outputs/24_image_captioning/ に保存。初回はモデル DL が走る）
uv run python lectures/24_image_captioning/caption_helpers.py     # 合成シーンの確認（DL 無し）
uv run python lectures/24_image_captioning/caption_metrics.py     # 指標のスモーク（DL 無し）
uv run python lectures/24_image_captioning/01_blip_caption.py     # BLIP 無条件/条件付き/pipeline
uv run python lectures/24_image_captioning/02_vitgpt2_git.py      # 3モデル比較・生成パラメータ
uv run python lectures/24_image_captioning/03_caption_metrics.py  # BLEU/ROUGE/CIDEr/CLIPScore
uv run python lectures/24_image_captioning/mini_project.py        # 章末ベンチマーク
uv run python lectures/24_image_captioning/use_case.py            # 実践: 自動 alt-text 生成ツール
uv run python lectures/24_image_captioning/use_case.py git        #   （任意）モデルを git/vitgpt2 に切替

# 演習: まず TODO を自分で埋める（未実装でも exit 0、FAIL 表示されるだけ）
uv run python lectures/24_image_captioning/exercises.py
# 模範解答の挙動（全 PASS）
uv run python lectures/24_image_captioning/exercises_solutions.py
#   または: SHOW_SOLUTION=1 uv run python lectures/24_image_captioning/exercises.py

# （任意）実画像で試す: data/24_image_captioning/ に .png/.jpg を置くと自動で使われる
#   （参照キャプションが無いので mini_project は CLIPScore 主指標に自動で切り替わる）
```

---

> 参照ライブラリ（2026-06 時点）: **torch 2.12+cpu / torchvision 0.27+cpu / transformers 5.11 / torchmetrics 1.9 / huggingface_hub / Pillow / OpenCV(headless) / matplotlib**
> 使用モデル: `Salesforce/blip-image-captioning-base` ／ `nlpconnect/vit-gpt2-image-captioning` ／ `microsoft/git-base` ／ CLIPScore 用 `openai/clip-vit-base-patch32`
> すべて CPU・`model.eval()` ＋ `torch.inference_mode()` 前提。生成は `max_new_tokens` を小さく保つ。ネットへ出るのはモデル重みの初回 DL のみ（入力画像は合成生成）。