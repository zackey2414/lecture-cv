# 25_vqa_vlm: VQAと軽量VLMによる画像理解・グラウンディング

> トラック: **マルチモーダル** ／ レベル: **中級** ／ 必要な依存グループ: `dl` `hf`
> （チャットVLM `02_vlm_chat.py` の完全版だけ追加で `uv add num2words` が必要。無くてもスクリプトは exit 0 で動く）

第24回では「画像 → 説明文」を一方向に生成した（キャプション）。本章はそこに**質問**という軸を足す。
「この画像に四角はある？」「円は何色？」「図形は何個？」——画像へ問いを投げて答えを引き出す **VQA (Visual Question Answering)** と、
LLM と同じチャット形式で画像も扱える **VLM (Vision-Language Model)** を、CPU で現実的な小型モデルだけで最後まで動かす。

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
入力は **画像 + 質問文**、出力は**回答**。`"What color is the circle?" → "red"`、`"How many shapes?" → "2"`、`"Is there a square?" → "yes"`。
人間の読解問題と同じで、答えは短い（多くは1〜3語）。だからこそ「色」「数」「有無」「位置」など、画像理解のどの能力が欠けているかをピンポイントで測れる、評価に向いたタスクになっている。

ここで最初に押さえたいのは、**答えの出し方に流派がある**こと。大きく3つだ。

| 流派 | 代表モデル | 答えの作り方 | 速度/性質 |
|---|---|---|---|
| 分類型 (discriminative) | ViLT, LXMERT | 3129語などの**固定候補**から1つ選ぶ（logits→argmax） | 速い・決定的。候補外は答えられない |
| 生成型 (generative) | BLIP-VQA, GIT | 回答テキストを**生成**する（`model.generate`） | 柔軟。少し遅く表記ゆれが出る |
| チャット型 (instruction VLM) | SmolVLM2, moondream2, Qwen2.5-VL, LLaVA | 会話の中で画像を見て**何でも**答える | 最も汎用。指示に従い推論もできる |

下に行くほど「賢く・汎用」だが「重く・確率的」になる。CPU で軽い順に、ViLT（〜0.5GB・生成なし）< BLIP-VQA（〜1.5GB・短い生成）< SmolVLM2-256M（〜0.5GB だが生成あり）。本章ではこの3つを実際に動かし、最後に「大型は同じ流儀で重いだけ」（Qwen2.5-VL-7B 等）と概念整理する。

実際に動かすと流派の性格がはっきり出る。合成シーン（左に赤い円・右に青い四角）で「四角は何色？」と訊くと、本章の実行では分類型 ViLT は `yellow`（外した）、生成型 BLIP は `blue`（当てた）、チャット型 SmolVLM2 は別シーンで図形の数・色・形をすべて正しく言い当てた。VQA モデルは実写真で学習されているため、抽象図形のような**学習分布外**の入力では平気で間違える——この「賢いモデルでも分布外で崩れる」感覚を持っておくと、評価指標の必要性が腹落ちする。

---

## 2. 理論/仕組み — 画像をどう言語モデルに食わせるか

### 2.1 分類型と生成型の中身

**分類型 (ViLT)** は、画像パッチ列とテキストトークン列を1本の Transformer に流し込み、`[CLS]` 相当の出力に**回答分類ヘッド**（3129クラス）を載せたものだ。出力は `logits` ∈ ℝ^3129。`argmax` で1クラスを選び、`id2label` で文字列にする。生成が無いので**速く・決定的**——同じ入力なら毎回同じ答え。代償は「候補語彙の外は絶対に出ない」こと。`"vermilion"` は候補に無ければ出てこない。

**生成型 (BLIP-VQA)** は Encoder-Decoder。画像エンコーダ + 質問を条件に、テキストデコーダが回答を**1トークンずつ生成**する。`model.generate(max_new_tokens=...)` を使う点はキャプションと同じだ。語彙に縛られず柔軟だが、デコードのぶん遅く、`"2"` と `"two"` のような**表記ゆれ**が混ざる。だから後段の評価では**正規化**が要る（後述）。

### 2.2 チャット型 VLM の仕組みと「画像はどこに入れるか」

チャット VLM は LLM の前段に**画像エンコーダ + 射影層**を足した構造だ。画像をパッチ特徴に変換し、それを「画像トークン」として通常のテキストトークン列に**差し込む**。だから入力プロンプトの中には `<image>` のような特殊トークンの“穴”があり、そこに画像パッチ埋め込みが流し込まれる。本章のスクリプトで実際に展開後の文字列を覗くと、こうなっている。

```
<|im_start|>User:<image>What color is the circle?<end_of_utterance>
Assistant:
```

`<image>` の位置に画像が入り、`Assistant:` の続きをモデルが生成する。ここで初学者が最もハマるのが、**画像をどこで渡すか**だ。`transformers v5` では「メッセージの `content` の中」に入れるのが正準で、`generate` への別引数や `pipeline` の別フィールドに渡す旧来のやり方は**動かない**。

```python
messages = [{
    "role": "user",
    "content": [
        {"type": "image", "image": pil_image},   # ★画像はここ（content の中）
        {"type": "text",  "text": "What color is the circle?"},
    ],
}]
```

このメッセージを `apply_chat_template` に通すと、モデル固有のテンプレート（上の `<|im_start|>...` 形式）へ自動展開され、画像も同時に前処理される。チャット型は「色」も「数」も「説明」も「比較」も**1つのモデル**で、指示文を変えるだけでこなせる——これが分類型/生成型に対する決定的な強みだ。

---

## 3. 正準API — transformers v5 での書き方

### 3.1 まず最重要の破壊的変更

`transformers v5` で **`pipeline("visual-question-answering")` と `pipeline("image-to-text")`（prompt付きの旧用法）は削除された**。ネット上の古いチュートリアルはそのままでは動かない。移行先は次のとおり。

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

3つに共通する作法を3点だけ強調する。第一に、**`model.eval()` と `torch.inference_mode()`** は推論で必須（勾配を切りメモリと速度を確保する）。第二に、CPU では **`dtype=torch.float32`**（`float16` は CPU で遅い/未対応 op がある）。第三に、チャット型では生成後に **`generated[:, prompt_len:]`** でプロンプト部分を切り落とすこと——これを忘れると質問文ごと decode され、答えが質問のオウム返しに見えてしまう。

---

## 4. 実装を1つずつ

### 4.1 `01_vqa_basics.py` — 分類型 vs 生成型

同じ合成シーン（左に赤い円・右に青い四角）へ同じ4問を投げ、ViLT と BLIP-VQA の答えを並べる。狙いは「どちらが正しいか」ではなく、**2つの流派の挙動差**を体感することだ。実行すると、ViLT は softmax の信頼度つきで即答（生成なしなので速い）、BLIP は短文を生成する。本章の実行では「四角は何色？」で ViLT=`yellow`／BLIP=`blue` と**割れた**——分類型は固定語彙に押し込むぶん分布外で滑りやすく、生成型は当てることもあるが「図形は何個？」では両方 `4`（実際は2）と数え間違えた。

ここから得る教訓は2つ。**(a) VQA モデルは実写真分布で訓練されており、合成図形は苦手**（だから本章の評価は満点にならない＝それが正常）。**(b) 速度と柔軟性はトレードオフ**で、用途に応じて選ぶ。図の右パネルに両モデルの答えを並べ、`01_vqa_basics.json` に信頼度込みで保存する。

### 4.2 `02_vlm_chat.py` — チャット VLM の正準フロー

SmolVLM2-256M で `apply_chat_template → generate → batch_decode` を最後まで通す。まず `tokenize=False` でテンプレート展開後の**文字列を表示**し、`<image>` トークンの位置を目で確認する（仕組みの可視化）。次に同じシーンBへ「三角は何色？」「図形は何個？」「黄色いのはどの形？」「一文で説明して」と訊く。

本章の実行では SmolVLM2 は `Green.` / `There are three shapes` / `the yellow shape is a circle` / `Three shapes ... on a white background` と、**すべて正しく**答えた。256M という極小モデルでも、チャット型は「数える」「色と形を結びつける」といった推論を分類型/生成型より上手にこなす。これがチャット VLM を使う動機だ。なお SmolVLM のプロセッサは `num2words` に依存するため、未導入の環境ではスクリプトは「メッセージ構造だけ表示して generate はスキップ」し、`uv add num2words` を案内して **exit 0** で終わる（落とさない設計）。

> 大型化しても**書き方は同じ**。`Qwen/Qwen2.5-VL-7B-Instruct` は `process_vision_info` で画像/動画を集めてから同じ `apply_chat_template → generate` に渡すが、CPU ではメモリ十数GB・1問数分なので「**GPU推奨**」。本章では概念のみとし、CPU は SmolVLM2-256M を主役にする。

### 4.3 `03_vqa_accuracy.py` — VQAv2 accuracy を実装する

VQA の採点は分類の accuracy ほど単純ではない。**同じ問いに人間でも答えがばらつく**（red / crimson / a red one）からだ。VQAv2 ベンチマークは1問につき**人間10人の回答**を持ち、次式で採点する。

```
accuracy(予測) = min( 予測と一致した人間の数 / 3,  1.0 )
```

3人以上が同じ答えなら満点、1人なら 0.33、誰とも合わなければ 0。**比較の前に必ず正規化**する：小文字化・句読点除去・数詞→数字（two→2）・冠詞（a/an/the）除去。スクリプトは正規化の有無で `"Red."` と `"red"` の一致判定が変わる様子を対比表示し、「正規化を忘れると正答を誤判定する」典型バグを体感させる。さらに**簡易版** `min(#agree/3,1)` と**公式 leave-one-out 版**（各人を1人ずつ抜いた残り9人で採点し10通りを平均；自分の回答を二重に数えない）を両方実装して比べる。両者は近いが、後者の方がわずかに辛い。

実モデル ViLT の予測でベンチを採点すると、色・有無は満点、数え問題で 0 になり、**mean VQAv2 ≈ 0.67**。`03_vqa_accuracy.png` に設問別スコアを棒グラフで残す。モデルが無い環境でも固定予測で採点ロジックは最後まで動く。

---

## 5. 落とし穴（このタスク特有のハマり所）

VQA/VLM は「動いたのに答えが変」というハマり方をしやすい。最頻のものを先に潰しておく。

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

「どのモデルを選ぶか」は精度だけでなく、**回答語彙の自由度・速度・メモリ・運用の堅さ**で決まる。実務での目安をまとめる。

- **固定された質問・選択肢で大量バッチ／低レイテンシ**なら **分類型 (ViLT)**。生成しないので速く決定的、出力が語彙に閉じているので後処理も楽。製造ラインの良否判定的な「決まった問い」に向く。
- **答えの幅は欲しいが軽さも要る**なら **生成型 (BLIP-VQA)**。短い自由回答が出せて、CPU でも実用速度。ただし表記ゆれ前提で**正規化を必ず挟む**。
- **指示理解・推論・複数質問を1モデルで**なら **チャット型 VLM**。CPU 制約なら SmolVLM2-256M/500M や moondream2、GPU があれば Qwen2.5-VL / InternVL / LLaVA。プロンプト設計（「一語で答えて」等）で出力を整える。
- **位置を答えさせたい（グラウンディング）**なら、`detect/point` を持つ moondream2 や、オープン語彙検出（OWLv2 / Grounding DINO、第20回）と組み合わせる。VQA モデル単体は座標を返さない。

評価の指標選びも実務判断だ。**人手回答が複数あるなら VQAv2 accuracy**（表記ゆれに頑健）。正解が一意に決まる用途なら exact-match でもよいが、人間の多様性を捨てている点は自覚する。生成が長文化する用途では BLEU/ROUGE/CLIPScore（第24回）や LLM-as-judge も併用する。

---

## 🛠 章末ミニプロジェクト: VLM「レポートカード」

`mini_project.py` は本章の要素を1本に統合する。小さな VQA ベンチマーク（合成シーン＋既知の正解＋人間10人の回答＋グラウンディング箱）に対し、

1. **(A) VQA 採点**：ViLT で質問応答させ、**VQAv2 accuracy と exact-match** を算出。
2. **(B) グラウンディング採点**：「赤い円はどこ？」に対して位置（箱・点）を予測し、**IoU と point-in-box** で採点。

を行い、左に VQA スコア棒グラフ・右にグラウンディング箱の重なり（GT緑／予測赤破線＋中心×）を並べた1枚絵と JSON を出力する。

グラウンディングの予測部分は、本来 moondream2 の `model.detect("red circle")` / `model.point("red circle")` が返す箱・点を使う。ただし ~2B でCPUだと重く `trust_remote_code` も要るため、ここでは**色マスクの簡易ローカライザ**で予測箱・点を作って評価フローを学ぶ（合成シーンは色が既知なので確実に localize できる）。**実務ではこの `localize_by_color` を VLM の出力に差し替えるだけ**で、同じ採点コードがそのまま使える——「評価の器」を先に作っておくのがこのミニプロジェクトの主眼だ。

```
$ uv run python lectures/25_vqa_vlm/mini_project.py
=== (A) VQA 採点 ===   -> mean VQAv2=0.667  mean exact-match=0.667  (source=ViLT)
=== (B) グラウンディング採点 === -> mean IoU=1.000  point accuracy=1.000
saved: outputs/25_vqa_vlm/mini_report_card.png, .../mini_report_card.json
```

VQA が満点にならない（数え問題で落ちる）一方、グラウンディングは満点——という**能力ごとの強弱がレポートで一目で分かる**。これが「モデルを多面的に測る」ということだ。

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

VQA/VLM は研究のためのベンチだけでなく、**「画像に質問して答えを得る」現実のツール**の核になります。
評価（accuracy / IoU）を追う `mini_project.py`（ベンチ寄りの統合課題）とは別に、**そのまま製品に
なりうる小ツール**をいくつか挙げます。共通する作り方は「**画像 + 質問文 → 回答エンジン（VQAモデル）
→ 短い回答**」で、回答エンジンを ViLT / BLIP-VQA / チャットVLM のどれにするかだけが変わります。

### ① 画像Q&Aアシスタント（`use_case.py`・動く出発点）

- **何に使うか**: 商品写真への問い合わせ窓口（「色は？」「個数は？」を自動回答）、書類・標識・
  メーターの内容確認、視覚障害者向けの画像説明、チャットボットの画像理解バックエンド。**1 枚の
  画像に複数の質問**をまとめて投げ、それぞれに短い回答を返します。
- **作り方の要点**: 回答エンジンに **BLIP-VQA（生成型）** を借り、`processor(image, question)` →
  `generate` → `decode` で自由回答を得ます。アプリ側は「**画像ごとに質問リストをループし、
  Q&A を 1 枚絵 + JSON にまとめる**」薄いラッパだけ。正解は不要なので、`mini_project.py` の
  採点コードと違って**任意の画像に任意の質問**を投げられます。
- **注意**: VQA モデルは**実写真分布**で学習されているため、計数・否定・空間関係は崩れやすい
  （本ツールでも合成図形の個数を `4`/`6` と誤ります＝正常）。一語で欲しいなら「答えを一語で」等の
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
- **注意**: 自由回答は表記ゆれが出る（`"two"` と `"2"`）ので、`normalize_answer`（§3）で正規化してから
  台帳に書き込みます。重要属性は VQA の出力を**人手レビューの下書き**として扱い、誤りやすい計数は
  別ロジック（検出器での個数カウント）に回すのが安全です。

### ③ アクセシビリティ／対話ボットの画像理解バックエンド

- **何に使うか**: 視覚障害者向けに「写真に何が写っているか」を読み上げる、チャットボットがユーザの
  アップロード画像に**自然言語で答える**バックエンド。
- **作り方の要点**: **チャット VLM**（SmolVLM2 / moondream2）を `apply_chat_template → generate →
  batch_decode`（§3.2）で動かし、ユーザの**任意の問い**にそのまま応答します。会話履歴を `messages` に
  積めば文脈つきの追加質問にも対応できます。
- **注意**: VLM は**幻覚（hallucination）**を起こす（写っていない物を「ある」と言う）ので、重要用途では
  「画像に無ければ無いと答えて」と指示し、必要なら検出器・OCR（第20/26回）で**事実を裏取り**します。
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

出力はすべて `outputs/25_vqa_vlm/`（PNG 図 / JSON）。画像はネット不要の合成シーン、`data/25_vqa_vlm/` に画像を置けばそちらを優先。ネットに出るのは**モデル重みのDLのみ**。

---

> 参照ライブラリ（版）: **torch 2.12+cpu / torchvision 0.27+cpu / transformers 5.11**（+ huggingface_hub, timm, einops, sentencepiece; チャットVLM完全版は num2words）。2026-06 時点。
> モデル: `dandelin/vilt-b32-finetuned-vqa`, `Salesforce/blip-vqa-base`, `HuggingFaceTB/SmolVLM2-256M-Instruct`（moondream2 / Qwen2.5-VL は概念）。CPU 前提・`model.eval()` + `torch.inference_mode()` ・`dtype=torch.float32`。