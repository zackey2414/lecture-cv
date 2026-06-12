# 25_vqa_vlm: VQAと軽量VLMによる画像理解・グラウンディング

> トラック: **マルチモーダル** ／ レベル: **中級** ／ 必要な依存グループ: `dl` `hf`
> （チャットVLM `02_vlm_chat.py` の完全版だけ追加で `uv add num2words` が必要。無くてもスクリプトは exit 0 で動く）

第24回では「画像 → 説明文」を一方向に生成した（キャプション）。本章はそこに**質問**という軸を足す。
すなわち「この画像に四角はある？」「円は何色？」「図形は何個？」のように、画像へ問いを投げて答えを引き出す **VQA (Visual Question Answering)** と、
LLM と同じチャット形式で画像も扱える **VLM (Vision-Language Model)** を扱う。いずれも CPU で現実的に動く小型モデルだけを使い、最後まで手を動かして動かしていく。

---

## 🎯 この章のゴール

この章を終えると、次のことが「自分の手で再現できる」状態になる。

- VQA の2大アプローチ——**分類型**（ViLT: 固定の回答語彙から選ぶ）と**生成型**（BLIP-VQA: 回答テキストを生成する）——を実装し、長所と短所を説明できる。
- チャット VLM の**正準フロー** `apply_chat_template → generate → batch_decode` を、画像を `content` に埋め込む形で書ける（SmolVLM2-256M を CPU で実行）。
- `transformers v5` で `pipeline("visual-question-answering")` が**廃止**された事実と、その移行先（`AutoModelForImageTextToText` + chat / 専用クラスの `processor(image, question)`）を理解している。
- VQA の評価指標 **VQAv2 accuracy = `min(一致した人間の数 / 3, 1)`** を、**正規化**込みで実装し、簡易版と公式 leave-one-out 版の違いを説明できる。
- **グラウンディング**（「赤い円はどこ？」に点や箱で答える）を **IoU / point-in-box** で採点できる。moondream2 の `detect/point`、大型 Qwen2.5-VL の位置づけを概念として整理できる。

成果物は、(1) VQA を2方式で解くスクリプト、(2) チャット VLM スクリプト、(3) VQAv2 accuracy 評価スクリプト、(4) それらを統合した「レポートカード」ミニプロジェクトの4本（＋演習9問）。

---

## 1. 直感 — VQA とは「画像に対する読解問題」

キャプションが「画像を見て作文する」タスクなら、VQA は「画像を見て**設問に答える**」タスクだ。
入力は **画像 + 質問文**、出力は**回答**であり、たとえば `"What color is the circle?" → "red"`、`"How many shapes?" → "2"`、`"Is there a square?" → "yes"` のようになる。
人間の読解問題と同じで、答えは短い（多くは1〜3語）。だからこそ「色」「数」「有無」「位置」といった、画像理解のどの能力が欠けているかをピンポイントで測れるため、評価に向いたタスクになっている。

ここで最初に押さえたいのは、**答えの出し方に流派がある**という点だ。大きく分けて、次の3つがある。

| 流派 | 代表モデル | 答えの作り方 | 速度/性質 |
|---|---|---|---|
| 分類型 (discriminative) | ViLT, LXMERT | 3129語などの**固定候補**から1つ選ぶ（logits→argmax） | 速い・決定的。候補外は答えられない |
| 生成型 (generative) | BLIP-VQA, GIT | 回答テキストを**生成**する（`model.generate`） | 柔軟。少し遅く表記ゆれが出る |
| チャット型 (instruction VLM) | SmolVLM2, moondream2, Qwen2.5-VL, LLaVA | 会話の中で画像を見て**何でも**答える | 最も汎用。指示に従い推論もできる |

表を下に行くほど「賢く・汎用」になるが、その分「重く・確率的」になる。CPU で軽い順に並べると、ViLT（〜0.5GB・生成なし）< BLIP-VQA（〜1.5GB・短い生成）< SmolVLM2-256M（〜0.5GB だが生成あり）となる。本章ではまずこの3つを実際に動かし、最後に大型モデル（Qwen2.5-VL-7B 等）を「同じ流儀のまま重くなるだけ」と概念的に整理する。

実際に動かしてみると、流派ごとの性格がはっきり出る。合成シーン（左に赤い円・右に青い四角）で「四角は何色？」と訊くと、本章の実行では分類型 ViLT は `yellow`（外した）、生成型 BLIP は `blue`（当てた）と分かれ、チャット型 SmolVLM2 は別シーンで図形の数・色・形をすべて正しく言い当てた。VQA モデルは実写真で学習されているため、抽象図形のような**学習分布外**の入力では平気で間違える。この「賢いモデルでも分布外では崩れる」という感覚を持っておくと、後に出てくる評価指標の必要性が腹落ちする。

---

## 2. 理論/仕組み — 画像をどう言語モデルに食わせるか

### 2.1 分類型と生成型の中身

**分類型 (ViLT)** は、画像パッチ列とテキストトークン列を1本の Transformer に流し込み、`[CLS]` 相当の出力に**回答分類ヘッド**（3129クラス）を載せたものだ。出力は `logits` ∈ ℝ^3129 となり、`argmax` で1クラスを選んで `id2label` で文字列に変換する。生成を伴わないため**速く・決定的**で、同じ入力なら毎回同じ答えを返す。その代償が「候補語彙の外は絶対に出ない」ことであり、たとえば `"vermilion"` は候補に無ければ出てこない。

**生成型 (BLIP-VQA)** は Encoder-Decoder の構造をとる。画像エンコーダと質問を条件に、テキストデコーダが回答を**1トークンずつ生成**していく仕組みで、`model.generate(max_new_tokens=...)` を使う点はキャプションと同じだ。語彙に縛られず柔軟な反面、デコードのぶん遅く、`"2"` と `"two"` のような**表記ゆれ**も混ざる。そのため、後段の評価では**正規化**が必要になる（後述）。

<figure class="lec-fig"><svg viewBox="0 0 640 256" role="img" aria-label="分類型VQAは固定語彙からargmaxで1つ選び決定的、生成型VQAはEncoder-Decoderで1トークンずつ生成し自由だが表記ゆれが出る" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><text x="24" y="48" font-size="13" font-weight="700" fill="#1d4ed8">分類型 ViLT</text><rect x="22" y="64" width="92" height="44" rx="5" fill="#eff6ff" stroke="#2563eb" stroke-width="1.6"/><text x="68" y="91" text-anchor="middle" font-size="13" fill="#1d4ed8">画像 + 質問</text><line x1="116" y1="86" x2="130" y2="86" stroke="#71717a" stroke-width="2"/><polygon points="138,86 129,81 129,91" fill="#71717a"/><rect x="138" y="64" width="114" height="44" rx="5" fill="#dbeafe" stroke="#2563eb" stroke-width="1.6"/><text x="195" y="91" text-anchor="middle" font-size="14" font-weight="700" fill="#1d4ed8">Transformer</text><line x1="254" y1="86" x2="268" y2="86" stroke="#71717a" stroke-width="2"/><polygon points="276,86 267,81 267,91" fill="#71717a"/><rect x="276" y="64" width="104" height="44" rx="5" fill="#f4f4f5" stroke="#71717a" stroke-width="1.6"/><text x="328" y="91" text-anchor="middle" font-size="12.5" fill="#3f3f46">logits(3129)</text><line x1="382" y1="86" x2="396" y2="86" stroke="#71717a" stroke-width="2"/><polygon points="404,86 395,81 395,91" fill="#71717a"/><text x="393" y="58" text-anchor="middle" font-size="11.5" fill="#52525b">argmax</text><rect x="406" y="64" width="84" height="44" rx="5" fill="#dc2626"/><text x="448" y="92" text-anchor="middle" font-size="16" font-weight="700" fill="#ffffff">red</text><rect x="520" y="72" width="86" height="28" rx="14" fill="#15803d"/><text x="563" y="91" text-anchor="middle" font-size="13" font-weight="700" fill="#ffffff">決定的</text><text x="24" y="176" font-size="13" font-weight="700" fill="#c2410c">生成型 BLIP-VQA</text><rect x="22" y="192" width="92" height="44" rx="5" fill="#fff7ed" stroke="#ea580c" stroke-width="1.6"/><text x="68" y="219" text-anchor="middle" font-size="13" fill="#c2410c">画像 + 質問</text><line x1="116" y1="214" x2="130" y2="214" stroke="#71717a" stroke-width="2"/><polygon points="138,214 129,209 129,219" fill="#71717a"/><rect x="138" y="192" width="114" height="44" rx="5" fill="#ffedd5" stroke="#ea580c" stroke-width="1.6"/><text x="195" y="219" text-anchor="middle" font-size="13.5" font-weight="700" fill="#c2410c">Enc → Dec</text><line x1="254" y1="214" x2="268" y2="214" stroke="#71717a" stroke-width="2"/><polygon points="276,214 267,209 267,219" fill="#71717a"/><rect x="276" y="192" width="104" height="44" rx="5" fill="#f4f4f5" stroke="#71717a" stroke-width="1.6"/><text x="328" y="219" text-anchor="middle" font-size="12.5" fill="#3f3f46">1語ずつ生成</text><line x1="382" y1="214" x2="396" y2="214" stroke="#71717a" stroke-width="2"/><polygon points="404,214 395,209 395,219" fill="#71717a"/><text x="393" y="186" text-anchor="middle" font-size="11.5" fill="#52525b">generate</text><rect x="406" y="192" width="84" height="44" rx="5" fill="#2563eb"/><text x="448" y="220" text-anchor="middle" font-size="16" font-weight="700" fill="#ffffff">blue</text><rect x="520" y="200" width="86" height="28" rx="14" fill="#ea580c"/><text x="563" y="219" text-anchor="middle" font-size="13" font-weight="700" fill="#ffffff">確率的</text></svg><figcaption><b>分類型</b>（ViLT）は画像と質問を1つの Transformer に通し、<b>3129語の固定語彙</b>から <code>argmax</code> で1つ選ぶ。生成がなく<b>速く決定的</b>だが候補外は答えられない。<b>生成型</b>（BLIP-VQA）は <b>Encoder→Decoder</b> で回答を<b>1トークンずつ生成</b>するため自由だが、少し遅く <code>2</code>/<code>two</code> のような<b>表記ゆれ</b>が出る。</figcaption></figure>

### 2.2 チャット型 VLM の仕組みと「画像はどこに入れるか」

チャット VLM は、LLM の前段に**画像エンコーダ + 射影層**を足した構造をとる。画像をパッチ特徴に変換し、それを「画像トークン」として通常のテキストトークン列に**差し込む**仕組みだ。したがって入力プロンプトの中には `<image>` のような特殊トークンの“穴”があり、そこへ画像パッチ埋め込みが流し込まれる。本章のスクリプトで展開後の文字列を実際に覗いてみると、次のようになっている。

```
<|im_start|>User:<image>What color is the circle?<end_of_utterance>
Assistant:
```

`<image>` の位置に画像が入り、`Assistant:` の続きをモデルが生成する。ここで初学者が最もハマるのが、**画像をどこで渡すか**という点だ。`transformers v5` では「メッセージの `content` の中」に入れるのが正準であり、`generate` への別引数や `pipeline` の別フィールドに渡す旧来のやり方は**動かない**。

```python
messages = [{
    "role": "user",
    "content": [
        {"type": "image", "image": pil_image},   # ★画像はここ（content の中）
        {"type": "text",  "text": "What color is the circle?"},
    ],
}]
```

このメッセージを `apply_chat_template` に通すと、モデル固有のテンプレート（上の `<|im_start|>...` 形式）へ自動展開され、画像も同時に前処理される。チャット型は「色」も「数」も「説明」も「比較」も、**1つのモデル**のまま指示文を変えるだけでこなせる。これこそが、分類型・生成型に対する決定的な強みである。

<figure class="lec-fig"><svg viewBox="0 0 660 250" role="img" aria-label="画像はエンコーダと射影層で画像トークンに変換され、プロンプト列の画像枠の位置に差し込まれてからLLMが答えを生成する" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><rect x="24" y="36" width="44" height="44" rx="3" fill="#ffffff" stroke="#71717a" stroke-width="1.6"/><circle cx="37" cy="58" r="8" fill="#dc2626"/><rect x="48" y="50" width="14" height="14" fill="#2563eb"/><line x1="70" y1="58" x2="84" y2="58" stroke="#71717a" stroke-width="2"/><polygon points="92,58 83,53 83,63" fill="#71717a"/><rect x="96" y="42" width="110" height="34" rx="5" fill="#dbeafe" stroke="#2563eb" stroke-width="1.6"/><text x="151" y="64" text-anchor="middle" font-size="12.5" font-weight="700" fill="#1d4ed8">エンコーダ+射影</text><line x1="208" y1="58" x2="222" y2="58" stroke="#71717a" stroke-width="2"/><polygon points="230,58 221,53 221,63" fill="#71717a"/><rect x="236" y="49" width="18" height="18" rx="2" fill="#f97316" stroke="#c2410c" stroke-width="1.4"/><rect x="258" y="49" width="18" height="18" rx="2" fill="#f97316" stroke="#c2410c" stroke-width="1.4"/><rect x="280" y="49" width="18" height="18" rx="2" fill="#f97316" stroke="#c2410c" stroke-width="1.4"/><text x="267" y="40" text-anchor="middle" font-size="11.5" font-weight="700" fill="#c2410c">画像トークン</text><line x1="267" y1="70" x2="153" y2="143" stroke="#c2410c" stroke-width="1.6" stroke-dasharray="5 3"/><polygon points="145,148 151,138 156,147" fill="#c2410c"/><rect x="24" y="160" width="60" height="40" rx="5" fill="#dbeafe" stroke="#2563eb" stroke-width="1.6"/><text x="54" y="185" text-anchor="middle" font-size="12.5" fill="#1d4ed8">User:</text><rect x="92" y="160" width="96" height="40" rx="5" fill="#fff7ed" stroke="#c2410c" stroke-width="1.8" stroke-dasharray="4 3"/><rect x="104" y="170" width="20" height="20" rx="2" fill="#f97316" stroke="#c2410c" stroke-width="1.2"/><rect x="128" y="170" width="20" height="20" rx="2" fill="#f97316" stroke="#c2410c" stroke-width="1.2"/><rect x="152" y="170" width="20" height="20" rx="2" fill="#f97316" stroke="#c2410c" stroke-width="1.2"/><text x="140" y="216" text-anchor="middle" font-size="11" fill="#c2410c">画像枠</text><rect x="196" y="160" width="150" height="40" rx="5" fill="#f4f4f5" stroke="#71717a" stroke-width="1.6"/><text x="271" y="185" text-anchor="middle" font-size="13" fill="#3f3f46">質問テキスト</text><rect x="354" y="160" width="96" height="40" rx="5" fill="#dbeafe" stroke="#2563eb" stroke-width="1.6"/><text x="402" y="185" text-anchor="middle" font-size="12.5" fill="#1d4ed8">Assistant:</text><line x1="452" y1="180" x2="466" y2="180" stroke="#71717a" stroke-width="2"/><polygon points="474,180 465,175 465,185" fill="#71717a"/><rect x="476" y="160" width="66" height="40" rx="5" fill="#1d4ed8"/><text x="509" y="185" text-anchor="middle" font-size="14" font-weight="700" fill="#ffffff">LLM</text><line x1="544" y1="180" x2="558" y2="180" stroke="#71717a" stroke-width="2"/><polygon points="566,180 557,175 557,185" fill="#71717a"/><rect x="568" y="160" width="72" height="40" rx="5" fill="#dc2626"/><text x="604" y="186" text-anchor="middle" font-size="15" font-weight="700" fill="#ffffff">red</text></svg><figcaption>チャットVLMは、画像を<b>エンコーダ＋射影層</b>で「<b>画像トークン</b>」に変換し、プロンプト列の <code>&lt;image&gt;</code> の穴（図の<b>画像枠</b>）へ差し込む。だから messages の <b>content の中</b>に画像を入れる必要がある。最後に <code>Assistant:</code> の続きを LLM が生成する。</figcaption></figure>

---

## 3. 正準API — transformers v5 での書き方

### 3.1 まず最重要の破壊的変更

`transformers v5` では **`pipeline("visual-question-answering")` と `pipeline("image-to-text")`（prompt 付きの旧用法）が削除された**。そのため、ネット上の古いチュートリアルはそのままでは動かない。移行先は次のとおりである。

| やりたいこと | v4 までの書き方（廃止/非推奨） | v5 の正準 |
|---|---|---|
| 分類型 VQA | `pipeline("visual-question-answering")` | `ViltForQuestionAnswering` + `ViltProcessor`（直接） |
| 生成型 VQA | `pipeline("visual-question-answering")` | `BlipForQuestionAnswering` + `processor(image, question)` |
| チャット VQA / 対話 | （該当なし） | `AutoModelForImageTextToText` + `apply_chat_template` |
| 画像プロセッサ取得 | `AutoFeatureExtractor` | `AutoImageProcessor` / `AutoProcessor`（`AutoFeatureExtractor` は廃止） |

迷ったら「**分類/生成の単発 VQA は専用クラスを直接、対話は `AutoModelForImageTextToText` + chat**」と覚えておけばよい。

### 3.2 3つの呼び出しパターン

```python
import torch

# (1) 分類型 ViLT: 生成しない。logits → argmax → id2label
from transformers import ViltProcessor, ViltForQuestionAnswering
proc = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa").eval()
enc = proc(image, "What color is the circle?", return_tensors="pt")
with torch.inference_mode():
    logits = model(**enc).logits
answer = model.config.id2label[int(logits.argmax(-1))]   # 例: "red"

# (2) 生成型 BLIP-VQA: 短い回答を生成
from transformers import BlipProcessor, BlipForQuestionAnswering
proc = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base").eval()
enc = proc(image, "How many shapes?", return_tensors="pt")
with torch.inference_mode():
    out = model.generate(**enc, max_new_tokens=10)
answer = proc.decode(out[0], skip_special_tokens=True)

# (3) チャット VLM SmolVLM2: apply_chat_template → generate → batch_decode
from transformers import AutoProcessor, AutoModelForImageTextToText
proc = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM2-256M-Instruct")
model = AutoModelForImageTextToText.from_pretrained(
    "HuggingFaceTB/SmolVLM2-256M-Instruct", dtype=torch.float32).eval()
messages = [{"role": "user", "content": [
    {"type": "image", "image": image},
    {"type": "text",  "text": "What color is the circle?"}]}]
inputs = proc.apply_chat_template(messages, add_generation_prompt=True,
                                  tokenize=True, return_dict=True, return_tensors="pt")
prompt_len = inputs["input_ids"].shape[1]               # ★プロンプト長を覚える
with torch.inference_mode():
    ids = model.generate(**inputs, max_new_tokens=24, do_sample=False)
answer = proc.batch_decode(ids[:, prompt_len:], skip_special_tokens=True)[0]  # 新規生成だけ
```

<figure class="lec-fig"><svg viewBox="0 0 640 220" role="img" aria-label="チャットVLMの正準フローはmessagesをapply_chat_templateでinputsにし、generateで出力idsを得て、prompt_len以降をbatch_decodeして答えにする" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><text x="320" y="30" text-anchor="middle" font-size="14" font-weight="700" fill="#18181b">チャットVLMの正準フロー（messages → 答え）</text><rect x="14" y="92" width="120" height="58" rx="6" fill="#eff6ff" stroke="#2563eb" stroke-width="1.8"/><text x="74" y="116" text-anchor="middle" font-size="14" font-weight="700" fill="#1d4ed8">messages</text><text x="74" y="136" text-anchor="middle" font-size="11" fill="#52525b">画像 + 質問</text><rect x="180" y="92" width="120" height="58" rx="6" fill="#dbeafe" stroke="#2563eb" stroke-width="1.8"/><text x="240" y="116" text-anchor="middle" font-size="14" font-weight="700" fill="#1d4ed8">inputs</text><text x="240" y="136" text-anchor="middle" font-size="11" fill="#52525b">input_ids + 画像</text><rect x="346" y="92" width="120" height="58" rx="6" fill="#f4f4f5" stroke="#71717a" stroke-width="1.8"/><text x="406" y="116" text-anchor="middle" font-size="14" font-weight="700" fill="#3f3f46">ids（出力）</text><text x="406" y="136" text-anchor="middle" font-size="11" fill="#52525b">プロンプト + 生成</text><rect x="512" y="92" width="114" height="58" rx="6" fill="#15803d"/><text x="569" y="116" text-anchor="middle" font-size="14" font-weight="700" fill="#ffffff">answer</text><text x="569" y="136" text-anchor="middle" font-size="11" fill="#ffffff">Green.</text><line x1="134" y1="121" x2="171" y2="121" stroke="#71717a" stroke-width="2"/><polygon points="180,121 171,116 171,126" fill="#71717a"/><line x1="300" y1="121" x2="337" y2="121" stroke="#71717a" stroke-width="2"/><polygon points="346,121 337,116 337,126" fill="#71717a"/><line x1="466" y1="121" x2="503" y2="121" stroke="#71717a" stroke-width="2"/><polygon points="512,121 503,116 503,126" fill="#71717a"/><text x="157" y="84" text-anchor="middle" font-size="11" fill="#3f3f46">apply_chat_template</text><text x="323" y="84" text-anchor="middle" font-size="11" fill="#3f3f46">generate</text><text x="489" y="84" text-anchor="middle" font-size="11" fill="#3f3f46">batch_decode</text><text x="489" y="172" text-anchor="middle" font-size="10.5" font-weight="700" fill="#15803d">ids[:, prompt_len:]</text></svg><figcaption><b>チャットVLM</b>の正準フローは <b>messages</b> を <code>apply_chat_template</code> で <b>inputs</b> にし、<code>generate</code> で<b>プロンプト＋生成</b>を連結した <b>ids</b> を得て、最後に <code>ids[:, prompt_len:]</code> で<b>新規トークンだけ</b>を <code>batch_decode</code> して<b>答え</b>にする3段の流れです。各箱がデータ、矢印上のラベルが呼び出す関数を表します。</figcaption></figure>

3つに共通する作法を、3点だけ強調しておく。第一に、**`model.eval()` と `torch.inference_mode()`** は推論で必須である（勾配を切り、メモリと速度を確保する）。第二に、CPU では **`dtype=torch.float32`** を使う（`float16` は CPU で遅く、未対応の op もある）。第三に、チャット型では生成後に **`generated[:, prompt_len:]`** でプロンプト部分を切り落とすこと。これを忘れると質問文ごと decode され、答えが質問のオウム返しに見えてしまう。

<figure class="lec-fig"><svg viewBox="0 0 640 230" role="img" aria-label="generateの出力はプロンプトと新規生成トークンの連結で、prompt_lenで切って新規トークンだけをdecodeする" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><text x="320" y="32" text-anchor="middle" font-size="14" font-weight="700" fill="#18181b">generate 出力 ＝ プロンプト ＋ 新規生成</text><rect x="46" y="58" width="300" height="46" rx="5" fill="#dbeafe" stroke="#2563eb" stroke-width="1.8"/><text x="196" y="86" text-anchor="middle" font-size="13" fill="#1d4ed8">プロンプト（画像＋質問）</text><rect x="346" y="58" width="248" height="46" rx="5" fill="#ffedd5" stroke="#ea580c" stroke-width="1.8"/><text x="470" y="86" text-anchor="middle" font-size="13" font-weight="700" fill="#c2410c">新規生成（答え）</text><line x1="346" y1="50" x2="346" y2="112" stroke="#3f3f46" stroke-width="2"/><text x="346" y="44" text-anchor="middle" font-size="12" font-weight="700" fill="#18181b">prompt_len</text><line x1="470" y1="108" x2="470" y2="138" stroke="#16a34a" stroke-width="2.5"/><polygon points="470,146 463,136 477,136" fill="#16a34a"/><text x="486" y="130" font-size="12.5" font-weight="700" fill="#15803d">ids[:, prompt_len:]</text><rect x="46" y="152" width="300" height="44" rx="5" fill="#fafafa" stroke="#d4d4d8" stroke-width="1.5" stroke-dasharray="5 3"/><line x1="52" y1="156" x2="340" y2="192" stroke="#dc2626" stroke-width="1.4"/><line x1="52" y1="192" x2="340" y2="156" stroke="#dc2626" stroke-width="1.4"/><text x="196" y="180" text-anchor="middle" font-size="13" font-weight="600" fill="#71717a">捨てる</text><rect x="346" y="152" width="248" height="44" rx="5" fill="#ffedd5" stroke="#ea580c" stroke-width="2"/><text x="470" y="180" text-anchor="middle" font-size="13" font-weight="700" fill="#c2410c">decode する</text></svg><figcaption><code>generate</code> の出力は<b>プロンプト＋新規生成</b>の連結なので、<b><code>ids[:, prompt_len:]</code></b> で<b>新規トークンだけ</b>を <code>decode</code> する。これを忘れて全体を decode すると、答えが<b>質問のオウム返し</b>に見える。</figcaption></figure>

---

## 4. 実装を1つずつ

### 4.1 `01_vqa_basics.py` — 分類型 vs 生成型

同じ合成シーン（左に赤い円・右に青い四角）へ同じ4問を投げ、ViLT と BLIP-VQA の答えを並べる。狙いは「どちらが正しいか」を競わせることではなく、**2つの流派の挙動差**を体感することだ。実行すると、ViLT は softmax の信頼度つきで即答し（生成がないので速い）、BLIP は短文を生成する。本章の実行では「四角は何色？」で ViLT=`yellow`／BLIP=`blue` と**答えが割れた**。分類型は固定語彙に押し込むぶん分布外で滑りやすく、一方の生成型は当てることもあるが、「図形は何個？」では両方とも `4`（実際は2）と数え間違えた。

ここから得られる教訓は2つある。**(a) VQA モデルは実写真分布で訓練されており、合成図形は苦手**である（だから本章の評価は満点にならないが、それが正常だ）。そして **(b) 速度と柔軟性はトレードオフ**の関係にあり、用途に応じて選ぶ必要がある。なお出力としては、図の右パネルに両モデルの答えを並べ、`01_vqa_basics.json` に信頼度込みで保存する。

### 4.2 `02_vlm_chat.py` — チャット VLM の正準フロー

SmolVLM2-256M で `apply_chat_template → generate → batch_decode` を最後まで通す。まず `tokenize=False` でテンプレート展開後の**文字列を表示**し、`<image>` トークンの位置を目で確認する（仕組みの可視化）。そのうえで同じシーンBへ「三角は何色？」「図形は何個？」「黄色いのはどの形？」「一文で説明して」と順に訊いていく。

本章の実行では、SmolVLM2 は `Green.` / `There are three shapes` / `the yellow shape is a circle` / `Three shapes ... on a white background` と、**すべて正しく**答えた。256M という極小モデルでありながら、チャット型は「数える」「色と形を結びつける」といった推論を分類型・生成型よりも上手にこなす。これこそがチャット VLM を使う動機だ。なお SmolVLM のプロセッサは `num2words` に依存するため、未導入の環境ではスクリプトは「メッセージ構造だけ表示して generate はスキップ」し、`uv add num2words` を案内して **exit 0** で終わる（処理を落とさない設計になっている）。

> 大型化しても**書き方は同じ**だ。`Qwen/Qwen2.5-VL-7B-Instruct` は `process_vision_info` で画像/動画を集めてから、同じ `apply_chat_template → generate` に渡す。ただし CPU ではメモリ十数GB・1問数分かかるため「**GPU推奨**」となる。本章では概念のみを扱い、CPU では SmolVLM2-256M を主役に据える。

### 4.3 `03_vqa_accuracy.py` — VQAv2 accuracy を実装する

VQA の採点は、分類の accuracy ほど単純ではない。なぜなら、**同じ問いでも人間によって答えがばらつく**（red / crimson / a red one）からだ。そこで VQAv2 ベンチマークは1問につき**人間10人の回答**を持ち、次式で採点する。

```
accuracy(予測) = min( 予測と一致した人間の数 / 3,  1.0 )
```

3人以上が同じ答えなら満点、1人なら 0.33、誰とも合わなければ 0 となる。ここで重要なのは、**比較の前に必ず正規化する**ことだ。具体的には、小文字化・句読点除去・数詞→数字（two→2）・冠詞（a/an/the）除去を行う。スクリプトは、正規化の有無で `"Red."` と `"red"` の一致判定が変わる様子を対比表示し、「正規化を忘れると正答を誤判定する」という典型バグを体感させる。さらに**簡易版** `min(#agree/3,1)` と**公式 leave-one-out 版**（各人を1人ずつ抜いた残り9人で採点し、10通りを平均する。自分の回答を二重に数えない）を両方実装して比べる。両者の結果は近いが、後者の方がわずかに辛めに出る。

<figure class="lec-fig"><svg viewBox="0 0 640 256" role="img" aria-label="VQAv2 accuracyは人間10人のうち予測と一致した人数aでmin(a/3,1)を計算し、3人以上一致で満点になる" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><text x="320" y="34" text-anchor="middle" font-size="14" font-weight="700" fill="#18181b">VQAv2 accuracy ＝ min( 一致人数 ÷ 3, 1 )</text><rect x="40" y="56" width="120" height="38" rx="6" fill="#dc2626"/><text x="100" y="81" text-anchor="middle" font-size="14" font-weight="700" fill="#ffffff">予測: red</text><text x="40" y="124" font-size="12.5" fill="#3f3f46">人間10人の回答:</text><circle cx="60" cy="150" r="12" fill="#16a34a"/><circle cx="96" cy="150" r="12" fill="#16a34a"/><circle cx="132" cy="150" r="12" fill="#16a34a"/><circle cx="168" cy="150" r="12" fill="#16a34a"/><circle cx="204" cy="150" r="12" fill="#d4d4d8"/><circle cx="240" cy="150" r="12" fill="#d4d4d8"/><circle cx="276" cy="150" r="12" fill="#d4d4d8"/><circle cx="312" cy="150" r="12" fill="#d4d4d8"/><circle cx="348" cy="150" r="12" fill="#d4d4d8"/><circle cx="384" cy="150" r="12" fill="#d4d4d8"/><line x1="48" y1="170" x2="180" y2="170" stroke="#15803d" stroke-width="1.5"/><line x1="192" y1="170" x2="396" y2="170" stroke="#71717a" stroke-width="1.5"/><text x="114" y="188" text-anchor="middle" font-size="12" font-weight="700" fill="#15803d">一致 4人</text><text x="294" y="188" text-anchor="middle" font-size="12" fill="#71717a">不一致 6人</text><text x="40" y="216" font-size="13" font-weight="700" fill="#18181b">例: a=4人 → min(4÷3, 1) = <tspan fill="#15803d">1.0</tspan> → 満点</text><text x="40" y="242" font-size="12" fill="#52525b">一致 0→0.0 ／ 1→0.33 ／ 2→0.67 ／ 3人以上→1.0</text></svg><figcaption>VQAv2 は1問に<b>人間10人</b>の回答を持ち、予測と<b>一致した人数 a</b> で <b>min(a ÷ 3, 1)</b> を採点する。<b>3人以上</b>一致で満点、1人なら約 0.33。比較の前に<b>正規化</b>（小文字化・冠詞除去・<code>two→2</code>）を忘れると <code>Red.</code> と <code>red</code> を取り違える。</figcaption></figure>

実モデル ViLT の予測でベンチを採点すると、色・有無は満点となる一方、数え問題で 0 になり、**mean VQAv2 ≈ 0.67** に落ち着く。設問別のスコアは `03_vqa_accuracy.png` に棒グラフで残す。なお、モデルが無い環境でも固定予測を用いれば、採点ロジック自体は最後まで動く。

---

## 5. 落とし穴（このタスク特有のハマり所）

VQA/VLM は「動いたのに答えが変」というハマり方をしやすい。そこで、頻度の高いものを先に潰しておく。

- **`pipeline("visual-question-answering")` を呼んで `KeyError`/`ValueError`**：v5 で廃止。専用クラスか `AutoModelForImageTextToText` + chat に移行する（§3.1）。
- **画像を `generate` の別引数や `content` の外で渡して `ValueError: ... no images were passed`**：チャット VLM では画像は必ず `content` の中（`{"type":"image","image":pil}`）。`apply_chat_template` がそこから前処理する。
- **答えが質問のオウム返しに見える**：`generate` 後にプロンプト部分を切っていない。`generated[:, prompt_len:]` を decode する。
- **`SmolVLM` ロードで `ImportError: num2words`**：SmolVLM のプロセッサ依存。`uv add num2words`。本章のスクリプトは未導入でも案内して exit 0。
- **CPU で `float16` を指定して激遅/エラー**：CPU は `dtype=torch.float32`。`device_map="auto"` は accelerate 前提なので CPU では使わず `.to("cpu")`。
- **正規化忘れで accuracy が不当に低い**：`"Red."` ≠ `"red"`、`"two"` ≠ `"2"`。比較前に必ず `normalize_answer`。
- **分類型 ViLT に語彙外の答えを期待する**：3129語に無い答えは構造上出ない。自由な答えが要るなら生成型/チャット型へ。
- **数え問題・否定・空間関係が弱い**：VQA モデルは色・物体有無に強いが、計数や「左/右」「〜でない」は崩れやすい（本章でも数えを外す）。評価設計でこの偏りを意識する。

---

## 6. 実務での使い分け

「どのモデルを選ぶか」は精度だけで決まるわけではなく、**回答語彙の自由度・速度・メモリ・運用の堅さ**まで含めて判断する。以下に、実務での目安をまとめる。

- **固定された質問・選択肢で大量バッチ／低レイテンシ**なら **分類型 (ViLT)**。生成しないので速く決定的、出力が語彙に閉じているので後処理も楽。製造ラインの良否判定的な「決まった問い」に向く。
- **答えの幅は欲しいが軽さも要る**なら **生成型 (BLIP-VQA)**。短い自由回答が出せて、CPU でも実用速度。ただし表記ゆれ前提で**正規化を必ず挟む**。
- **指示理解・推論・複数質問を1モデルで**なら **チャット型 VLM**。CPU 制約なら SmolVLM2-256M/500M や moondream2、GPU があれば Qwen2.5-VL / InternVL / LLaVA。プロンプト設計（「一語で答えて」等）で出力を整える。
- **位置を答えさせたい（グラウンディング）**なら、`detect/point` を持つ moondream2 や、オープン語彙検出（OWLv2 / Grounding DINO、第20回）と組み合わせる。VQA モデル単体は座標を返さない。

評価の指標選びも、同じく実務上の判断になる。**人手回答が複数あるなら VQAv2 accuracy** が向く（表記ゆれに頑健だからだ）。正解が一意に決まる用途なら exact-match でもよいが、その場合は人間の多様性を捨てている点を自覚しておく。さらに、生成が長文化する用途では BLEU/ROUGE/CLIPScore（第24回）や LLM-as-judge も併用するとよい。

---

## 🛠 章末ミニプロジェクト: VLM「レポートカード」

`mini_project.py` は、本章の要素を1本に統合したものだ。小さな VQA ベンチマーク（合成シーン＋既知の正解＋人間10人の回答＋グラウンディング箱）に対し、次の2つを行う。

1. **(A) VQA 採点**：ViLT で質問応答させ、**VQAv2 accuracy と exact-match** を算出。
2. **(B) グラウンディング採点**：「赤い円はどこ？」に対して位置（箱・点）を予測し、**IoU と point-in-box** で採点。

<figure class="lec-fig"><svg viewBox="0 0 660 270" role="img" aria-label="ミニプロジェクトは1つのVQAベンチを(A)VQA採点と(B)グラウンディング採点の2系統に分岐させ、結果を1枚絵とJSONのレポートカードへ合流させる" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><text x="330" y="32" text-anchor="middle" font-size="14" font-weight="700" fill="#18181b">ミニプロジェクト — 1つのベンチを2能力で採点しレポート化</text><rect x="18" y="124" width="160" height="72" rx="7" fill="#f4f4f5" stroke="#71717a" stroke-width="2"/><text x="98" y="158" text-anchor="middle" font-size="15" font-weight="700" fill="#18181b">VQAベンチ</text><text x="98" y="180" text-anchor="middle" font-size="11" fill="#52525b">合成+正解+10人+GT箱</text><line x1="178" y1="160" x2="208" y2="160" stroke="#71717a" stroke-width="2"/><line x1="208" y1="104" x2="208" y2="216" stroke="#71717a" stroke-width="2"/><line x1="208" y1="104" x2="231" y2="104" stroke="#71717a" stroke-width="2"/><polygon points="240,104 231,99 231,109" fill="#71717a"/><line x1="208" y1="216" x2="231" y2="216" stroke="#71717a" stroke-width="2"/><polygon points="240,216 231,211 231,221" fill="#71717a"/><rect x="240" y="80" width="200" height="48" rx="7" fill="#fff7ed" stroke="#ea580c" stroke-width="2"/><text x="340" y="100" text-anchor="middle" font-size="14" font-weight="700" fill="#c2410c">(A) VQA採点</text><text x="340" y="118" text-anchor="middle" font-size="11" fill="#52525b">ViLT → VQAv2 acc・exact-match</text><rect x="240" y="192" width="200" height="48" rx="7" fill="#eff6ff" stroke="#2563eb" stroke-width="2"/><text x="340" y="212" text-anchor="middle" font-size="13" font-weight="700" fill="#1d4ed8">(B) グラウンディング採点</text><text x="340" y="230" text-anchor="middle" font-size="11" fill="#52525b">位置予測 → IoU・point-in-box</text><line x1="440" y1="104" x2="474" y2="104" stroke="#71717a" stroke-width="2"/><line x1="440" y1="216" x2="474" y2="216" stroke="#71717a" stroke-width="2"/><line x1="474" y1="104" x2="474" y2="216" stroke="#71717a" stroke-width="2"/><line x1="474" y1="160" x2="495" y2="160" stroke="#71717a" stroke-width="2"/><polygon points="504,160 495,155 495,165" fill="#71717a"/><rect x="504" y="124" width="140" height="72" rx="7" fill="#15803d"/><text x="574" y="158" text-anchor="middle" font-size="14" font-weight="700" fill="#ffffff">レポートカード</text><text x="574" y="180" text-anchor="middle" font-size="11" fill="#ffffff">1枚絵PNG + JSON</text></svg><figcaption><b>章末ミニプロジェクト</b>の全体像です。1つの <b>VQAベンチ</b>（合成シーン＋既知の正解＋人間10人の回答＋GT箱）を、<b>(A) VQA採点</b>（ViLT で <code>VQAv2 accuracy</code>・exact-match）と <b>(B) グラウンディング採点</b>（<code>IoU</code>・point-in-box）の<b>2系統に分岐</b>させ、両者の結果を<b>1枚絵（PNG）と JSON のレポートカード</b>へ<b>合流</b>させます。能力ごとの強弱が1枚で見渡せます。</figcaption></figure>

そして、左に VQA スコア棒グラフ・右にグラウンディング箱の重なり（GT緑／予測赤破線＋中心×）を並べた1枚絵と JSON を出力する。

<figure class="lec-fig"><svg viewBox="0 0 640 280" role="img" aria-label="グラウンディングの採点。IoUは予測箱とGT箱の重なり面積を和集合面積で割った値、point-in-boxは予測点がGT箱の中にあるか" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><text x="158" y="38" text-anchor="middle" font-size="13" font-weight="700" fill="#18181b">IoU（箱の重なり）</text><rect x="112" y="104" width="74" height="66" fill="#ffedd5"/><rect x="56" y="66" width="130" height="104" fill="none" stroke="#16a34a" stroke-width="2.5"/><rect x="112" y="104" width="130" height="104" fill="none" stroke="#dc2626" stroke-width="2" stroke-dasharray="6 4"/><text x="74" y="60" font-size="12" font-weight="700" fill="#15803d">GT（緑）</text><text x="240" y="226" text-anchor="end" font-size="12" font-weight="700" fill="#dc2626">予測（赤破線）</text><text x="149" y="141" text-anchor="middle" font-size="12" font-weight="700" fill="#c2410c">重なり</text><text x="149" y="256" text-anchor="middle" font-size="12.5" font-weight="700" fill="#18181b">IoU ＝ 重なり ÷ 和集合</text><line x1="328" y1="48" x2="328" y2="248" stroke="#e4e4e7" stroke-width="1.5"/><text x="478" y="38" text-anchor="middle" font-size="13" font-weight="700" fill="#18181b">point-in-box（点が箱内か）</text><rect x="400" y="78" width="150" height="110" fill="none" stroke="#16a34a" stroke-width="2.5"/><text x="418" y="72" font-size="12" font-weight="700" fill="#15803d">GT 箱</text><line x1="467" y1="125" x2="483" y2="141" stroke="#dc2626" stroke-width="3"/><line x1="467" y1="141" x2="483" y2="125" stroke="#dc2626" stroke-width="3"/><text x="475" y="162" text-anchor="middle" font-size="11" fill="#dc2626">予測点</text><text x="475" y="222" text-anchor="middle" font-size="13" font-weight="700" fill="#15803d">箱の中 → 正解</text></svg><figcaption>グラウンディングの採点指標。<b>IoU</b> は<b>予測箱とGT箱の重なり面積 ÷ 和集合面積</b>で、1.0 が完全一致（図は <b>GT 緑</b>・<b>予測 赤破線</b>）。<b>point-in-box</b> は<b>予測点がGT箱の内側</b>にあるかを ○/× で見る単純版で、点（×）が箱内なら正解。</figcaption></figure>

グラウンディングの予測部分には、本来 moondream2 の `model.detect("red circle")` / `model.point("red circle")` が返す箱・点を使う。ただし ~2B でCPUだと重く、`trust_remote_code` も必要になるため、ここでは**色マスクの簡易ローカライザ**で予測箱・点を作って評価フローを学ぶ（合成シーンは色が既知なので、確実に localize できる）。**実務ではこの `localize_by_color` を VLM の出力に差し替えるだけ**で、同じ採点コードがそのまま使える。このように「評価の器」を先に作っておくことが、このミニプロジェクトの主眼だ。

```
$ uv run python lectures/25_vqa_vlm/mini_project.py
=== (A) VQA 採点 ===   -> mean VQAv2=0.667  mean exact-match=0.667  (source=ViLT)
=== (B) グラウンディング採点 === -> mean IoU=1.000  point accuracy=1.000
saved: outputs/25_vqa_vlm/mini_report_card.png, .../mini_report_card.json
```

VQA は満点にならず（数え問題で落ちる）、一方でグラウンディングは満点になる。このように**能力ごとの強弱がレポートで一目で分かる**のが特徴だ。これこそが「モデルを多面的に測る」ということである。

---

## ✅ 到達チェックリスト

- [ ] 分類型 VQA（ViLT）を `logits → argmax → id2label` で実装できる。
- [ ] 生成型 VQA（BLIP-VQA）を `processor(image, question) → generate → decode` で実装できる。
- [ ] 分類型と生成型の長所・短所（語彙の自由度・速度・決定性）を説明できる。
- [ ] チャット VLM の正準フロー `apply_chat_template → generate → batch_decode` を書け、**画像を `content` に埋め込む**理由を説明できる。
- [ ] `generate` 後に `prompt_len` でスライスして新規トークンだけ decode する理由を説明できる。
- [ ] `transformers v5` で `pipeline("visual-question-answering")` が廃止されたこと・移行先を説明できる。
- [ ] VQAv2 accuracy `min(#agree/3,1)` を**正規化込み**で実装でき、簡易版と leave-one-out 版の違いを説明できる。
- [ ] なぜ VQA に正規化が必須か（`"Red." ≠ "red"`）を例で説明できる。
- [ ] グラウンディングを IoU / point-in-box で採点でき、moondream2 の `detect/point` の位置づけを説明できる。
- [ ] CPU 制約下のモデル選択（ViLT / BLIP-VQA / SmolVLM2 / 大型は GPU 推奨）を根拠つきで判断できる。

---

## ❓ よくある落とし穴・FAQ・デバッグ

**Q. `02_vlm_chat.py` が「num2words が必要」と出て答えが出ない。**
A. SmolVLM のプロセッサの依存です。`uv add num2words` で完全版が動きます。未導入でもスクリプトはメッセージ構造を表示して exit 0 で終わるよう設計してあります。

**Q. 初回実行が遅い／止まって見える。**
A. 初回は Hugging Face からモデル重みをDLしています（ViLT 〜0.5GB、BLIP-VQA 〜1.5GB、SmolVLM2 〜0.5GB）。2回目以降は `~/.cache/huggingface` のキャッシュから即ロードされます。オフライン再現は `HF_HUB_OFFLINE=1`。

**Q. チャット VLM の答えが質問のコピーになる。**
A. `generate` の出力にはプロンプトも含まれます。`generated[:, inputs["input_ids"].shape[1]:]` を decode してください。

**Q. accuracy が思ったより低い。**
A. まず `normalize_answer` を通しているか確認（`"2"` と `"two"` 等）。それでも低いなら、合成図形は VQA モデルの**学習分布外**で素の精度が出にくいのが原因。実写真や `data/25_vqa_vlm/` に置いた画像で試すと挙動が分かりやすいです。

**Q. CPU で重すぎる／メモリが足りない。**
A. `max_new_tokens` を小さく（20前後）、モデルは 256M 級を選ぶ。`torch.set_num_threads(物理コア数)` で安定。Qwen2.5-VL-7B などは CPU 非現実的なので GPU を使ってください。

**Q. moondream2 を使いたい。**
A. `trust_remote_code=True` と **`revision` 固定**（頻繁に更新され挙動が変わる）が必須で、`einops`/`timm`（本リポの `hf` 群に同梱）に依存。~2B なので CPU では1問数十秒。`model.caption/query/detect/point` の4 API を持ち、特に `detect/point` はグラウンディングに使えます（本章はミニプロジェクトで評価の器だけ用意）。

**デバッグの常道**：(1) まず ViLT（生成なし・決定的）で配線を確認 → (2) 正規化と accuracy を `exercises.py` の純計算で固める → (3) 最後にチャット VLM の生成を足す。生成系は最後に回すと切り分けが楽です。

---

## 🚀 発展トピック・参考

- **グラウンディング専用**：オープン語彙検出（OWLv2 / Grounding DINO、第20回）＋ SAM（第22回）で Grounded-SAM 的に「文で指す→箱→マスク」。moondream2 の `point/detect` は単体で軽量グラウンディング。
- **大型 VLM**：Qwen2.5-VL / InternVL / LLaVA-OneVision。`process_vision_info` で画像・動画・複数枚を扱う。動画 VQA、文書 VQA（第26回 Donut/DocVQA）へ接続。
- **評価の深掘り**：VQAv2 公式評価（句読点・数値の細則）、生成 VQA の LLM-as-judge、幻覚（hallucination）評価（POPE 等）。
- **プロンプト設計**：「一語で答えて」「選択肢から選んで」で出力を構造化、Chain-of-Thought で計数を改善。
- 公式ドキュメント：
  - Transformers VQA タスク: https://huggingface.co/docs/transformers/tasks/visual_question_answering
  - Image-Text-to-Text / chat templates: https://huggingface.co/docs/transformers/tasks/image_text_to_text
  - SmolVLM: https://huggingface.co/HuggingFaceTB/SmolVLM2-256M-Instruct
  - moondream2: https://huggingface.co/vikhyatk/moondream2
  - ViLT VQA: https://huggingface.co/dandelin/vilt-b32-finetuned-vqa ／ BLIP-VQA: https://huggingface.co/Salesforce/blip-vqa-base
  - VQA 評価仕様: https://visualqa.org/evaluation.html

---

## 💡 実践ユースケース集

VQA/VLM は、研究のためのベンチだけでなく、**「画像に質問して答えを得る」現実のツール**の核にもなります。
ここでは、評価（accuracy / IoU）を追う `mini_project.py`（ベンチ寄りの統合課題）とは別に、**そのまま製品に
なりうる小ツール**をいくつか挙げます。いずれも共通する作り方は「**画像 + 質問文 → 回答エンジン（VQAモデル）
→ 短い回答**」であり、回答エンジンを ViLT / BLIP-VQA / チャットVLM のどれにするかだけが変わります。

### ① 画像Q&Aアシスタント（`use_case.py`・動く出発点）

- **何に使うか**: 商品写真への問い合わせ窓口（「色は？」「個数は？」を自動回答）、書類・標識・
  メーターの内容確認、視覚障害者向けの画像説明、チャットボットの画像理解バックエンド。**1 枚の
  画像に複数の質問**をまとめて投げ、それぞれに短い回答を返します。
- **作り方の要点**: 回答エンジンに **BLIP-VQA（生成型）** を借り、`processor(image, question)` →
  `generate` → `decode` で自由回答を得ます。アプリ側は「**画像ごとに質問リストをループし、
  Q&A を 1 枚絵 + JSON にまとめる**」薄いラッパだけ。正解は不要なので、`mini_project.py` の
  採点コードと違って**任意の画像に任意の質問**を投げられます。
- **注意**: VQA モデルは**実写真分布**で学習されているため、計数・否定・空間関係は崩れやすいです
  （本ツールでも合成図形の個数を `4`/`6` と誤りますが、それが正常です）。一語の回答が欲しいなら「答えを一語で」等の
  プロンプト整形を、信頼度が欲しいなら ViLT（softmax 確率）への差し替えを検討します。

```bash
uv run python lectures/25_vqa_vlm/use_case.py
# → 合成シーン or data/25_vqa_vlm/ の実画像に複数質問を投げ、
#   画像ごとの Q&A パネルと回答 JSON を outputs/25_vqa_vlm/ に保存
```

- **data 配置**: `data/25_vqa_vlm/` に**画像**（`*.png` / `*.jpg` / `*.jpeg` / `*.bmp` / `*.webp`）を
  置くと実入力で動きます（例: `data/25_vqa_vlm/product.jpg`）。さらに `data/25_vqa_vlm/questions.txt`
  を置けば**1 行 1 問**でカスタム質問を使えます（`#` 始まりはコメント）。画像が無ければ合成シーンで、
  BLIP-VQA が使えなければ**オフラインのルールベース回答**で必ず完走します（exit 0）。
- **拡張アイデア**: 回答エンジンを **チャット VLM**（SmolVLM2-256M。`uv add num2words` 後に
  `vqa_helpers.load_smolvlm` + `apply_chat_template`）に替えて**説明・推論**も返す、回答に**信頼度**を
  付ける（ViLT へ）、質問をテンプレ化して **CSV バッチ問い合わせ**で台帳に自動タグ付け、など。

### ② 商品・在庫の自動タグ付け（バッチ VQA）

- **何に使うか**: EC の商品画像から「色」「カテゴリ」「個数」「ロゴの有無」などの**属性を自動抽出**して
  検索用メタデータを作る。人手のタグ付けを VQA で下書きします。
- **作り方の要点**: ①の Q&A アシスタントを**固定の質問テンプレ**（"What color?" / "What category?" /
  "Is there a logo?"）に限定し、画像フォルダを一括処理して CSV/JSON に落とすだけ。語彙を固定したいなら
  **分類型 ViLT** が決定的で後処理が楽です。
- **注意**: 自由回答には表記ゆれが出る（`"two"` と `"2"`）ため、`normalize_answer`（§3）で正規化してから
  台帳に書き込みます。重要な属性については VQA の出力を**人手レビューの下書き**として扱い、誤りやすい計数は
  別ロジック（検出器での個数カウント）に回すのが安全です。

### ③ アクセシビリティ／対話ボットの画像理解バックエンド

- **何に使うか**: 視覚障害者向けに「写真に何が写っているか」を読み上げる、チャットボットがユーザの
  アップロード画像に**自然言語で答える**バックエンド。
- **作り方の要点**: **チャット VLM**（SmolVLM2 / moondream2）を `apply_chat_template → generate →
  batch_decode`（§3.2）で動かし、ユーザの**任意の問い**にそのまま応答します。会話履歴を `messages` に
  積めば文脈つきの追加質問にも対応できます。
- **注意**: VLM は**幻覚（hallucination）**を起こす（写っていない物を「ある」と言う）ため、重要用途では
  「画像に無ければ無いと答えて」と指示し、必要に応じて検出器・OCR（第20/26回）で**事実を裏取り**します。
  CPU では SmolVLM2-256M/500M を主役にし、大型（Qwen2.5-VL）は GPU 前提と割り切ります。

---

## ▶ 動かし方（コマンド）

```bash
# 依存（未導入なら）。本講座のルートで実行。
uv sync --group dl --group hf
# チャットVLMの完全版を動かす場合のみ（無くても 02 は exit 0 で動く）
uv add num2words

# 1) 分類型 vs 生成型 VQA（ViLT / BLIP-VQA）
uv run python lectures/25_vqa_vlm/01_vqa_basics.py
# 2) チャット VLM の正準フロー（SmolVLM2-256M）
uv run python lectures/25_vqa_vlm/02_vlm_chat.py
# 3) VQAv2 accuracy の実装と採点
uv run python lectures/25_vqa_vlm/03_vqa_accuracy.py
# 4) 章末ミニプロジェクト（VQA + グラウンディングのレポートカード）
uv run python lectures/25_vqa_vlm/mini_project.py
# 5) 実践ユースケース: 画像Q&Aアシスタント（画像 × 複数質問 → 回答）
#    data/25_vqa_vlm/ に画像を置けば実画像で、無ければ合成シーンで動く
uv run python lectures/25_vqa_vlm/use_case.py

# 演習（自己採点。最初は FAIL でも exit 0）
uv run python lectures/25_vqa_vlm/exercises.py
# 模範解答（全 PASS）
uv run python lectures/25_vqa_vlm/exercises_solutions.py
```

出力はすべて `outputs/25_vqa_vlm/`（PNG 図 / JSON）に保存される。画像は既定ではネット不要の合成シーンを使い、`data/25_vqa_vlm/` に画像を置けばそちらを優先する。ネットに出るのは**モデル重みのDLのみ**だ。

---

> 参照ライブラリ（版）: **torch 2.12+cpu / torchvision 0.27+cpu / transformers 5.11**（+ huggingface_hub, timm, einops, sentencepiece; チャットVLM完全版は num2words）。2026-06 時点。
> モデル: `dandelin/vilt-b32-finetuned-vqa`, `Salesforce/blip-vqa-base`, `HuggingFaceTB/SmolVLM2-256M-Instruct`（moondream2 / Qwen2.5-VL は概念）。CPU 前提・`model.eval()` + `torch.inference_mode()` ・`dtype=torch.float32`。