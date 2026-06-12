# 第26回 OCRと文書理解 — Tesseract / EasyOCR / TrOCR・Donut / LayoutLM・CER / WER

> トラック: **マルチモーダル** ／ レベル: **中級** ／ 依存グループ: `dl`（torch/torchvision）・`hf`（transformers/sentencepiece ほか）・`metrics`（torchmetrics を CER/WER の検算に使用）。CPU だけで完走します（初回のみ TrOCR / Donut の重みを HuggingFace からダウンロード）。古典 OCR の `pytesseract` と深層 OCR の `easyocr` は **OS パッケージや巨大モデルを伴うため既定依存に含めず**、`try/except` でガードして「導入済みなら動かす／未導入なら案内のみ」に統一しています（＝必ず exit 0）。

## 🎯 この章のゴール

第24回（画像キャプション）・第25回（VQA/VLM）では、「画像を見て自由文を生成・応答する」というマルチモーダルの王道を学びました。本章で扱うのは、そのなかでも**産業利用が桁違いに多い**領域――すなわち**「文書画像から文字を読み取り（OCR）、さらに帳票の意味を理解する（文書理解）」**です。実際、請求書・領収書・帳票・スキャン PDF など、「紙やスクショの中の文字を、構造化されたデータに変える」ニーズは、どの業界にも存在します。この章を終えるころには、その全工程を **3系統の OCR ＋ OCR フリー文書理解 ＋ 定量評価**として自力で組めるようになっているはずです。

到達点は5つです。第一に、**OCR には2段構造（テキスト検出＋テキスト認識）がある**ことを理解し、`pytesseract` / `easyocr` の**座標付き出力**（単語ごとの box と confidence）と、`TrOCR` の **`generate` による系列生成**という、出力フォーマットの根本的な違いを説明できること。第二に、**TrOCR（VisionEncoderDecoder）**を `processor → generate → batch_decode` で動かし、印字テキスト行を読めること。第三に、**OCR を一切介さない Donut**を `task_prompt → generate → token2json` で動かし、請求書から項目を抽出・質問応答できること。第四に、**CER（文字誤り率）/ WER（単語誤り率）を編集距離の定義から自前実装**し、`torchmetrics` と一致を検算できること。第五に、**前処理（NFKC 正規化・小文字化）を揃えることで CER が劇的に変わる**こと、そして DocVQA の標準指標 **ANLS** の考え方を、数値で体感できることです。

本章のスクリプトはすべて、ネットもデータセット DL も無しで完走するよう、入力を**合成データ**（PIL で描いた印字テキスト行と合成請求書）として生成し、**正解文字列・正解項目を画素単位で厳密に保持**します。正解が手元にあるからこそ、CER/WER も DocVQA の採点も再現可能です。しかも TrOCR は印字テキストに、Donut は帳票 QA に強いため、合成データであっても教材として意味のある数字（実測で TrOCR コーパス CER 0.0115、Donut DocVQA exact-match 4/4）が得られます。実画像で試したい場合は、`data/26_ocr_document/` に文書画像を置けば `03` と `mini_project` が自動で拾います。

---

## 1. OCRと文書理解の地図 — 4つのアプローチの位置づけ

「文書画像を読む」と一口に言っても、手法は大きく4系統に分かれます。**(1) 古典 OCR（Tesseract）**は、二値化・連結成分・パターンマッチといった画像処理＋統計の積み重ねで、**軽量・高速・CPU で即動く**のが強み。**(2) 深層 OCR（EasyOCR）**は、CNN ベースのテキスト検出器（CRAFT 等）＋ CNN-RNN の認識器を組み合わせ、**傾き・多言語・自然画像中の文字**に強い。**(3) Transformer 系生成 OCR（TrOCR）**は、画像を ViT で符号化し、テキストデコーダで**1文字ずつ生成**する seq2seq で、**手書き・難読な印字**で高精度を出します。そして **(4) OCR フリー文書理解（Donut / LayoutLM 系）**は、「文字を読む」工程すら内部に畳み込み、**画像から直接、構造化された答え（JSON や回答文字列）を生成**します。

この4系統は「**何を入力に取り、何を出力するか**」で整理すると腑に落ちます。(1)(2) は「画像 → **単語列＋座標＋確信度**」を返す**検出＋認識器**で、レイアウト復元や塗りつぶし（マスキング）に使える座標情報が得られます。(3) は「行画像 → **文字列**」を返す**認識専用**で、座標は出しません（行の切り出しは別途必要）。(4) は「文書画像＋タスク指示 → **答えそのもの**」を返す**エンドツーエンド**で、「請求書番号は？」に直接 `12345` と答える――途中の OCR テキストを介しません。本章はこの4系統を**実際に動かして対比**し、最後に CER/WER/ANLS で**数字として比較**します。

実務での使い分けの初期方針も、ここで掴んでおきましょう。**大量・定型・速度優先**なら古典 Tesseract、**多言語・自然画像・傾き**なら EasyOCR、**手書き・高精度な1行認識**なら TrOCR、**「項目を抜きたい／質問に答えたい」**なら Donut/LayoutLM、というのが大づかみの指針です。どれが「正解」ということはなく、**コスト・速度・精度・出力フォーマット**のトレードオフで選びます。まずはこの地図を頭に入れ、次節で「OCR の2段構造」という共通の骨格を押さえます。

## 2. OCRの2段構造 — テキスト検出 ＋ テキスト認識、そして「座標 vs generate」

OCR は本来、**(A) テキスト検出（detection）= 画像のどこに文字があるか（box/領域）を見つける**工程と、**(B) テキスト認識（recognition）= その領域の画素を文字列に変換する**工程の、**2段**で成り立ちます。Tesseract も EasyOCR もこの2段を内部に持ち、結果として**単語ごとの座標（box）と確信度（conf）**を返せます。`pytesseract.image_to_data` は `level/left/top/width/height/conf/text` の列を、`easyocr.readtext` は `(4点の bbox, text, conf)` のリストを返す――どちらも「**文字列だけでなく、どこにあったか**」が分かるのがポイントです。この座標は、帳票のレイアウト復元、特定欄だけの抽出、個人情報のマスキングなどに不可欠です。

一方 **TrOCR は認識（B）専用**で、**検出（A）は担いません**。入力は「すでに1行に切り出された画像」を想定し、`model.generate` で**文字列を系列生成**します。つまり TrOCR の出力は**文字列のみで座標は無い**――ここが Tesseract/EasyOCR との決定的な違いです。複数行・複雑なレイアウトの文書を TrOCR で読むには、**別途テキスト検出（行の切り出し）**が必要になります。逆に言えば、行が既に切れている（あるいは1行だけの）ケースでは、TrOCR は検出の誤りに影響されず、認識精度だけで勝負できます。

この「**座標付き出力か、`generate` による文字列生成か**」という違いは、後段の使い方を大きく左右します。座標が要る（レイアウト・マスキング）なら Tesseract/EasyOCR、純粋な認識精度が要る（手書き・難読）なら TrOCR、という選択になります。そして Donut に至っては、検出も認識も**タスクに溶かし込み**、「請求書番号は？」に直接答える――OCR の2段すら見えなくなります。次節からは、この4系統を**1つずつ実際に動かし**ます。まず軽量な古典・深層 OCR（座標付き）から。

## 3. Tesseract — 軽量・高速な古典 OCR（座標付き、要 OS パッケージ）

**Tesseract** は Google が保守する歴史ある OSS の OCR エンジンで、Python からは `pytesseract` という薄いラッパー経由で使います。重要なのは、**`pytesseract` は本体ではない**こと――実体の `tesseract` コマンドは**OS パッケージ**（`apt-get install -y tesseract-ocr`、日本語は `tesseract-ocr-jpn` を追加）として別途インストールが必要です。これを忘れると `TesseractNotFoundError` で落ちます。本講座では巨大化を避けるため既定依存に含めず、`01_tesseract_easyocr.py` は `import pytesseract` を `try/except` で囲み、**未導入なら「導入方法と出力フォーマット」を案内するだけ**にしています（だから本環境でも exit 0）。

API は2つを押さえれば十分です。**全文がほしいなら `image_to_string(img)`**、**単語ごとの座標と確信度がほしいなら `image_to_data(img, output_type=Output.DICT)`** です。後者は `level/page/block/par/line/word/left/top/width/height/conf/text` の**列指向の dict** を返し、各単語の `left,top,width,height`（box）と `conf`（0〜100、-1 は無効）が取れます。`01` は導入済みならこの box を画像に重ね描きして保存します。Tesseract の強みは**圧倒的な軽さと速さ**（モデル DL 不要、CPU で一瞬）で、**印字・整ったレイアウト**に向きます。弱みは**傾き・低解像度・装飾フォント・自然画像中の文字**に弱いこと、そして**前処理（二値化・傾き補正）の良し悪しで精度が大きく振れる**ことです。

```python
import pytesseract
text = pytesseract.image_to_string(img)                       # 全文（文字列）
data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
for i, w in enumerate(data["text"]):                          # 単語ごとに box + conf
    if w.strip() and float(data["conf"][i]) >= 0:
        box = (data["left"][i], data["top"][i], data["width"][i], data["height"][i])
```

実務では「**まず Tesseract で試し、精度が足りなければ深層系へ**」という順序がコスト効率的です。とくに大量の定型帳票（同じフォーマットのスキャン）では、適切な前処理（コントラスト調整・二値化・傾き補正）と組み合わせれば、Tesseract だけで十分なことが多々あります。日本語は `lang="jpn"`（縦書きは `jpn_vert`）を指定し、`tesseract-ocr-jpn` を入れるのを忘れないこと。次は、深層学習ベースでより頑健な EasyOCR を見ます。

## 4. EasyOCR — 深層の検出＋認識（多言語・傾きに強い）

**EasyOCR** は、CRAFT 系のテキスト検出器と CNN-RNN（CRNN）の認識器を組み合わせた、深層学習ベースの OCR です。`easyocr.Reader(["en"], gpu=False)` のように**言語リスト**と `gpu=False`（CPU 前提）を指定して `Reader` を作り、`reader.readtext(np_image)` で **`[(4点の bbox, text, conf), ...]`** を得ます。初回は検出/認識モデルを自動ダウンロードします。Tesseract と同じく**座標付き出力**ですが、box が「軸並行の矩形」ではなく**4点の四角形（quad）**で返るため、**傾いた文字**にも追従できるのが特徴です。`01` は導入済みなら、この quad を画像に重ね描きします。

EasyOCR の強みは、**多言語（80+ 言語、日本語含む）を1つの API で**扱え、**自然画像中の文字・傾き・湾曲**に Tesseract より頑健なことです。看板・商品パッケージ・スクリーンショットなど「レイアウトが整っていない」入力で威力を発揮します。一方の弱みは、**Tesseract より重い**（モデル DL とメモリが必要、CPU だと遅め）こと、そして**密な文書全体**より「ところどころに文字がある画像」に向くチューニングであることです。Tesseract が「整った文書を速く」、EasyOCR が「乱れた文字を頑健に」と、得意分野が補完的だと捉えると選びやすくなります。

```python
import easyocr
reader = easyocr.Reader(["en"], gpu=False)        # CPU 前提なら gpu=False を明示
for quad, text, conf in reader.readtext(np_image):  # quad は4点（傾きに追従）
    print(text, round(conf, 2))
```

ここまでの2系統（Tesseract / EasyOCR）に共通するのは、**「検出＋認識」で座標つきの結果を返す**点でした。本講座の実行環境ではどちらも未導入なので `01` は案内のみで止まりますが、**出力フォーマットの違い（Tesseract は `[l,t,w,h]` の列、EasyOCR は4点 quad）**は必ず頭に入れてください。次は、座標を出さない代わりに**認識精度に振り切った** TrOCR を、実際に動かします。

## 5. TrOCR — Transformer 系の生成 OCR（VisionEncoderDecoder + generate）

**TrOCR**（`microsoft/trocr-base-printed`）は、**ViT（画像）エンコーダ + テキストデコーダ**を結合した `VisionEncoderDecoderModel` です。画像をパッチ列として ViT に通し、その表現を条件に、デコーダが**1トークンずつ文字列を生成**します。つまり TrOCR は**分類でも検出でもなく、画像から文字列への翻訳（seq2seq）**――機械翻訳と同じ `generate` の枠組みで OCR を解きます。使い方は3手順で、`processor(images=img, return_tensors="pt").pixel_values` で前処理 → `model.generate(pixel_values, max_new_tokens=..., num_beams=...)` で生成 → `processor.batch_decode(ids, skip_special_tokens=True)` で文字列化、です。`-printed` は印字用、`-handwritten` は手書き用と用途別に重みが分かれています。

`02_trocr.py` は、合成印字テキスト行（`Invoice 2026` など6行）を TrOCR で読み、GT と比べて CER を測ります。その実測結果（`02_trocr.json`）を下表にまとめました。まず注目してほしいのは、**`-printed` モデルは出力が大文字寄りになりがち**で、`Invoice 2026` が `INVOICE 2026` と返る点です。**大文字小文字を区別したまま（raw）だと CER が高く出る**のに、**NFKC＋小文字化で正規化すると CER 0**になる――この差こそが、「評価の前処理がいかに重要か」を物語っています（第6節で深掘り）。一方、難しめの小さい文字（16px の `Contract No: Z9-8471`）では `Z` を `2` と読み違え、正規化後でも CER 0.05 が残りました。これが TrOCR の「認識の限界」を示す、現実的な数字です。

| GT（テキスト行） | TrOCR 予測 | CER(raw) | CER(norm) |
| --- | --- | --- | --- |
| `Invoice 2026` | `INVOICE 2026` | 0.500 | **0.000** |
| `Total: 980 USD` | `TOTAL: 980 USD` | 0.286 | **0.000** |
| `Serial AB-12345` | `SERIAL AB-12345` | 0.333 | **0.000** |
| `Hello World` | `HELLO WORLD` | 0.727 | **0.000** |
| `Contract No: Z9-8471`（16px） | `CONTRACT NO: 29-8471` | 0.450 | 0.050 |
| **コーパス CER（マイクロ平均）** | — | **0.4023** | **0.0115** |

```python
from transformers import TrOCRProcessor, VisionEncoderDecoderModel  # ＝ AutoProcessor / AutoModelForVision2Seq でも可
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed").to(device).eval()
with torch.inference_mode():
    pixel_values = processor(images=line_img, return_tensors="pt").pixel_values.to(device)
    ids = model.generate(pixel_values, max_new_tokens=32, num_beams=4)  # beam で安定化
text = processor.batch_decode(ids, skip_special_tokens=True)[0]
```

`generate` のパラメータ（`num_beams`, `max_new_tokens`）はキャプション（第24回）と同じノブです。`num_beams=1`（greedy）は速いが、難しい入力では beam search（`num_beams=4`）の方が安定します（本章の合成行ではどちらも同結果でした）。transformers v5 では `TrOCRProcessor`/`VisionEncoderDecoderModel` のほか、汎用の **`AutoProcessor` / `AutoModelForVision2Seq`** でも同じモデルをロードできます（`AutoImageProcessor` が画像前処理、`AutoFeatureExtractor` は廃止）。TrOCR は座標を出さないので、複数行文書では「検出で行を切る → 各行を TrOCR」という組み合わせになる点を、もう一度確認しておきましょう。次は、ここまで何度も出てきた **CER/WER の正体**に踏み込みます。

## 6. 評価 — CER / WER を「編集距離」から理解する

OCR の精度は **CER（Character Error Rate）= 編集距離(文字) / 参照文字数 = (置換 S + 削除 D + 挿入 I) / N** で測ります。ここで**編集距離（レーベンシュタイン距離）**とは、予測文字列を正解文字列に変えるのに必要な「1文字の置換・削除・挿入」の**最小回数**で、動的計画法（DP）で求まります。`ocr_helpers.edit_distance` は1行ぶんの DP（メモリ O(n)）で実装しており、**文字列にも単語リストにも同じ関数が使える**のがミソです（要素の等価判定しか使わないため）。これを文字単位で割れば CER、空白で区切った**単語単位**で割れば **WER（Word Error Rate）**になります。CER も WER も 0 が完璧で、**挿入が多いと 1 を超え得る**点に注意（誤りは参照長で正規化するため）。

CER/WER を測る前に、**必ず行うべきなのが正規化**です。というのも、OCR は「全角/半角」「大文字/小文字」「連続空白」を取り違えやすく、それを誤りとして数えてしまうと、**本来の認識力を過小評価**してしまうからです。`normalize_text` は **NFKC 正規化**（全角英数・全角空白・互換文字を半角の正準形へ：`Ｉｎｖｏｉｃｅ`→`Invoice`, `１２３`→`123`）→ **小文字化** → **連続空白の圧縮**を行います。`04_cer_wer_eval.py` の実測がその効果を雄弁に語ります（下表）。**まったく同じ内容**でも、生のままだと CER 1.0（全部違う扱い）、NFKC だけで 0.43、NFKC＋小文字化で **0.0**――評価の土俵を揃えないと数字は無意味、という教訓です。**日本語は分かち書きが曖昧で WER が安定しない**ため、**CER を主指標**にします。

| 比較（`Invoice No.123` vs 全角大文字 `ＩＮＶＯＩＣＥ　Ｎｏ．１２３`） | CER |
| --- | --- |
| 正規化なし（生の編集距離） | 1.0000 |
| NFKC のみ（大小文字は区別） | 0.4286 |
| **NFKC ＋ 小文字化** | **0.0000** |

自前実装は**必ず標準実装と検算**します。`04` は `the quick brown fox` vs `the quikc brown box`（`quick`→`quikc` の2置換、`fox`→`box` の1置換）で、手作り関数と **`torchmetrics.text.CharErrorRate / WordErrorRate`** がともに **CER=0.1579 / WER=0.5000** で一致することを確認します。`jiwer.cer` も同じ定義（あれば自動で照合、無ければ案内のみ）。CER は**マイクロ平均**（全サンプルの総編集距離 ÷ 総参照文字数）で集計するのが標準で、サンプルごとの CER を単純平均（マクロ）すると短い行の誤りが過大評価される、という集計の作法も `corpus_cer` で押さえます。

```python
def edit_distance(a, b):                  # 文字列でも単語リストでも動く（DP, メモリO(n)）
    m, n = len(a), len(b)
    if m == 0 or n == 0: return n or m
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        cur = [i] + [0] * n
        for j in range(1, n + 1):
            cost = 0 if a[i-1] == b[j-1] else 1
            cur[j] = min(cur[j-1] + 1, prev[j] + 1, prev[j-1] + cost)  # 挿入/削除/置換
        prev = cur
    return prev[n]

cer = edit_distance(norm(ref), norm(hyp)) / len(norm(ref))   # 文字単位
wer = edit_distance(norm(ref).split(), norm(hyp).split()) / len(norm(ref).split())  # 単語単位
```

この「編集距離 → CER/WER」と「正規化を揃える」の2点が、OCR 評価の核心です。ここを自分の手で実装できれば、どんな OCR エンジンの良し悪しも数字で語れます。次は、OCR を**飛び越えて**文書の意味に直接アクセスする Donut へ進みます。

## 7. Donut — OCR フリー文書理解（task_prompt → generate → token2json）

**Donut**（`naver-clova-ix/donut-base-finetuned-docvqa`）は、**Swin Transformer（画像）エンコーダ + BART（テキスト）デコーダ**からなる、**OCR を一切使わない**文書理解モデルです。発想が革命的で、「画像から文字を読む（OCR）→ 読んだテキストを理解する」という2段を、**1つの seq2seq に畳み込み**ます。デコーダは**タスクを示す特殊トークン**（DocVQA なら `<s_docvqa>`）から始まる構造化トークン列を生成し、`processor.token2json` がそれを JSON（`{question, answer}` など）に整形します。OCR を介さないので、**OCR の誤りが伝播しない**・**レイアウトを暗黙に学習している**という利点があります。

DocVQA（文書への質問応答）の使い方は、デコーダの**初期入力（task_prompt）**を組み立てるのが肝です。`<s_docvqa><s_question>{質問}</s_question><s_answer>` までを `decoder_input_ids` として与え、**続き（答え）をモデルに生成させ**ます。`03_donut_docvqa.py` は合成請求書（会社名・請求番号・日付・合計）に対し4つの質問を投げ、実測（`03_docvqa.json`）で**全問正解（exact-match 4/4、平均 ANLS 1.000）**でした。`What is the invoice number?` → `12345`、`What is the total?` → `980 usd`、`What is the company name?` → `acme corp`、`What is the date?` → `2026-06-11`。**OCR テキストを一度も経由していない**のに、画像から直接これらを当てている点に注目してください。

```python
from transformers import DonutProcessor, VisionEncoderDecoderModel
processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa")
model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa").to(device).eval()

task_prompt = f"<s_docvqa><s_question>{question}</s_question><s_answer>"   # ← 答えの直前まで与える
decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False,
                                        return_tensors="pt").input_ids.to(device)
pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)
out = model.generate(pixel_values, decoder_input_ids=decoder_input_ids, max_length=64,
                     pad_token_id=processor.tokenizer.pad_token_id,
                     eos_token_id=processor.tokenizer.eos_token_id, use_cache=True)
seq = processor.batch_decode(out)[0]                  # 特殊トークン込みの生成列
answer = processor.token2json(seq).get("answer", "")  # → {"question":..., "answer":"12345"}
```

ここで**素の `donut-base`（finetune 前）は使えない**ことに注意してください。`donut-base` は事前学習だけのバックボーンで、DocVQA の task_prompt に意味ある答えを返しません。**必ずタスクに合った finetune 済み重み**（`-finetuned-docvqa`、レシート構造化なら `-finetuned-cord-v2`、文書分類なら `-finetuned-rvlcdip`）を選びます。Donut の強みは**OCR フリーで誤り伝播が無い**・**構造化出力（JSON）が直接得られる**こと、弱みは**未知レイアウト・長文書への一般化**と、**新タスクには finetune が要る**ことです。次に、Donut とは対照的な「OCR の座標を陽に使う」LayoutLM 系と、高レベル pipeline を見ます。

## 8. LayoutLMv3 と document-question-answering pipeline、そして ANLS

Donut が「OCR フリー」なのに対し、**LayoutLMv3** は逆方向のアプローチで、**テキスト（OCR で得た単語）＋座標（box）＋画像パッチ**の3つを融合して文書を理解します。つまり**OCR が前段に必要**で、`pytesseract` 等で得た `words` と `boxes` をモデルに渡します。レイアウト情報を陽に使うため、表・フォームのように**位置関係が意味を持つ文書**に強いのが特徴です。transformers の高レベル API **`pipeline("document-question-answering")`** はこの系統を手軽に試せる窓口で、LayoutLM 系モデルを使う場合は内部で OCR（要 tesseract）を呼び、**Donut 系モデルを渡せば OCR フリー**で動きます。`03` は `pipeline("document-question-answering", model="naver-clova-ix/donut-base-finetuned-docvqa")` を使い、直接 `generate` した答えと**同じ結果（`12345`）**が出ることを確認します。

```python
from transformers import pipeline
pipe = pipeline("document-question-answering", model="naver-clova-ix/donut-base-finetuned-docvqa")
pipe(image=invoice, question="What is the total?")   # → [{"answer": "980 usd"}]（OCRフリー）
# 注: LayoutLM 系（impira/layoutlm-document-qa 等）を使う場合は内部で OCR(tesseract) が必要
```

文書 QA の評価には、OCR と同じ exact-match だけでなく **ANLS（Average Normalized Levenshtein Similarity）**という DocVQA 標準指標を使います。ANLS は「正規化レーベンシュタイン類似度 `1 - 編集距離/max(len)`」を計算し、それが**しきい値 0.5 以上ならその値を、未満なら 0**をスコアにします。狙いは「**綴りの軽い揺れ（`Acme Corp` vs `acme corp.`）には寛容、まったく別の答えには 0 点**」という、人間の採点感覚に近い評価です。exact-match が「1文字でも違えば 0」と厳しすぎるのに対し、ANLS は**部分点**を与えます。`03` は exact-match と ANLS の両方を出し、`ocr_helpers.anls` で実装しています（本章の合成請求書では両者とも満点でしたが、実文書では ANLS の方が実態に即した数字になります）。

実務での Donut vs LayoutLM の選択は、「**OCR を信頼できるか**」「**レイアウトが効くか**」「**finetune できるか**」で決まります。OCR が高品質で座標が効く定型フォームなら LayoutLM、OCR の誤りを避けたい・構造化 JSON が直接ほしいなら Donut、という大づかみです。どちらも「項目抽出・帳票理解」という同じゴールに、**陽に座標を使う/使わない**という逆の道で到達します。ここまでで4系統＋評価が出そろったので、次節で**使い分けの総まとめ**をします。

## 9. 使い分けの指針 — 4系統＋評価の総まとめ

最後に、本章で扱った手法を実務目線で一覧します。**「座標が要るか」「精度 vs 速度」「OCR を介すか」「finetune できるか」**の4軸で見ると、現場での選択がほぼ即決できます。

| 手法 | 種別 | 出力 | 座標 | CPU 速度 | 得意 | 弱み |
| --- | --- | --- | --- | --- | --- | --- |
| **Tesseract** | 古典 OCR | 文字列＋box＋conf | あり（矩形） | 最速 | 整った印字・定型帳票 | 傾き・低解像度・装飾に弱い／要 OS パッケージ |
| **EasyOCR** | 深層 OCR | 文字列＋quad＋conf | あり（4点） | 中 | 多言語・傾き・自然画像 | 重い・密文書よりまばらな文字向き |
| **TrOCR** | 生成 OCR | 文字列のみ | **なし** | 中〜遅 | 手書き・難読な1行認識 | 行検出は別途・座標なし |
| **Donut** | OCRフリー文書理解 | 構造化JSON/回答 | なし | 遅 | 帳票の項目抽出・DocVQA | 未知レイアウト・要 finetune |
| **LayoutLMv3** | レイアウト融合 | 回答/トークン分類 | 使う（入力） | 中 | 表・フォーム（位置が効く） | **前段に OCR が必要** |

評価指標も役割で選びます。**逐語の認識精度**には **CER（日本語は主指標）/ WER（英語など分かち書きが明確な言語）**、**文書 QA**には **exact-match（厳格）と ANLS（部分点）**。どの指標でも、**評価前に正規化（NFKC・小文字化）を予測と正解で揃える**のが鉄則です。「**まず軽い古典で試し、精度が足りなければ深層・生成へ、項目抽出が目的なら文書理解モデルへ**」――この段階的な発想を持てば、新しい文書 OCR の課題に出会っても、最小コストで適切な手法に当たりを付けられます。

## 🛠 章末ミニプロジェクト — 帳票の「読み取り → 構造化 → 評価」一気通貫

`mini_project.py` は、本章の要素（TrOCR の生成 OCR ／ Donut の文書理解 ／ CER・WER・ANLS の評価）を**1本のパイプライン**に統合します。合成請求書を題材に、**ステージA【テキスト認識】**で印字テキスト行を TrOCR で読み GT と比べて **CER/WER** を算出、**ステージB【文書理解】**で同じ請求書を Donut/DocVQA に渡し項目に関する質問へ回答 → **exact-match と ANLS** で採点します。最後に、両ステージのスコアを「**高いほど良い**」に揃えた要約棒グラフ（`mini_summary.png`：認識は `1-CER`、理解は exact-match と ANLS）と、総合レポート（`mini_report.json`）を出力します。

実測では、ステージA がコーパス CER 0.0115 / 平均 WER 0.0556（小さい文字の `Z→2` 誤読を除けばほぼ完璧）、ステージB が exact-match 4/4・平均 ANLS 1.000 でした。**どちらのモデルが落ちても（ネット不通など）パイプラインは止めず、できた範囲でレポートを出す**よう設計してあるので、必ず exit 0 になります。これは「**部分的に失敗しても全体が止まらない**」という、実運用パイプラインの堅牢性の最小形でもあります。`data/26_ocr_document/` に実画像を置けば、その文書で（GT 無しなら回答のみ）動きます。

```bash
uv run python lectures/26_ocr_document/mini_project.py
# → outputs/26_ocr_document/mini_summary.png（総合スコア）, mini_recognition.png（行ごとの GT/予測/CER）, mini_report.json
```

この統合課題を自分の手で動かし、**「認識精度（CER）」と「理解精度（ANLS）」を別々に評価できる**こと、そして**両者をどう組み合わせて1つの文書処理パイプラインにするか**を体得することが、本章のゴールです。余力があれば、ステージA の TrOCR を Tesseract/EasyOCR（導入時）に差し替えて CER を比べる、ステージB の質問を増やす、といった拡張に挑戦してください。

## ✅ 到達チェックリスト

次の項目を「人に説明でき／コードで再現できる」かで、定着を自己確認してください。

- [ ] OCR の**2段構造（検出＋認識）**を説明でき、Tesseract/EasyOCR の**座標付き出力**と TrOCR の **`generate`（座標なし）**の違いを言える
- [ ] `pytesseract.image_to_string` と `image_to_data`（`level/left/top/width/height/conf/text`）の役割を説明できる
- [ ] `easyocr.Reader(["en"], gpu=False).readtext` が **`(4点bbox, text, conf)`** を返すことと、Tesseract との得意分野の違いを言える
- [ ] TrOCR を `TrOCRProcessor` + `VisionEncoderDecoderModel`（or `AutoProcessor`/`AutoModelForVision2Seq`）で `processor → generate → batch_decode` の3手順で動かせる
- [ ] **CER =(S+D+I)/N** と **WER** を編集距離の DP から**自前実装**でき、`torchmetrics` と一致を検算できる
- [ ] **NFKC＋小文字化**で予測と正解を揃えると CER が変わることを数値で示せ、**日本語は CER が主指標**である理由を言える
- [ ] Donut を `task_prompt（<s_docvqa><s_question>…<s_answer>）→ generate → token2json` で動かし、請求書から項目を抽出・QA できる
- [ ] `donut-base`（finetune 前）が使えず、**タスク別 finetune 重み**が必要な理由を説明できる
- [ ] `pipeline("document-question-answering")` を Donut（OCRフリー）/ LayoutLM（要OCR）で使い分けられる
- [ ] **ANLS** の定義（`1-編集距離/max(len)`、閾値 0.5 で 0 に丸め）と、exact-match より文書 QA に向く理由を言える

## ❓ よくある落とし穴・FAQ・デバッグ

本章で詰まりやすい点を「症状 → 原因 → 対処」でまとめます。OCR 特有・transformers v5 特有の罠が多いので、エラーが出たらまずここを確認してください。

| 症状 | ほぼ確実な原因 | 対処 |
| --- | --- | --- |
| `TesseractNotFoundError` | `pytesseract`（ラッパー）だけ入れ、**OS の tesseract 本体**が無い | `apt-get install -y tesseract-ocr`（日本語は `tesseract-ocr-jpn` を追加） |
| 日本語が読めない/文字化け | `lang` 未指定 or 言語データ未導入 | `image_to_string(img, lang="jpn")`＋`tesseract-ocr-jpn`、縦書きは `jpn_vert` |
| EasyOCR が極端に遅い/落ちる | `gpu=True` のまま CPU 環境 | `Reader(langs, gpu=False)` を明示。初回はモデル DL が走る |
| TrOCR の出力が**全部大文字** | `-printed` モデルの癖。**評価前に正規化していない** | CER 計算前に NFKC＋`lower()` で正規化（生 CER 0.40→正規化後 0.01） |
| TrOCR で複数行がうまく読めない | TrOCR は**認識専用（検出なし）**。1行入力前提 | 先に行検出（EasyOCR/レイアウト解析）で**1行ずつ切ってから**渡す |
| Donut の答えが意味不明 | **`donut-base`（finetune 前）**を使っている | `naver-clova-ix/donut-base-finetuned-docvqa` 等**タスク別 finetune 重み**を使う |
| Donut が空/途中で切れる | `decoder_input_ids`（task_prompt）未指定、`max_length` 不足 | `<s_docvqa>…<s_answer>` を与え、`max_length`/`eos_token_id` を設定 |
| `token2json` が `{}` になる | 生成列に開始/終了タグが揃っていない | `batch_decode(out)`（特殊トークン込み）を渡す。`skip_special_tokens=False` |
| `pipeline("document-question-answering")` が OCR を要求 | LayoutLM 系モデルは**前段 OCR（tesseract）必須** | OCR フリーにしたいなら **Donut 系モデル**を `model=` に渡す |
| `AutoFeatureExtractor` で ImportError | transformers v5 で**廃止** | `AutoImageProcessor`（画像）/ `AutoProcessor`（画像＋テキスト）を使う |
| CER が 1.0 を超える/異常に高い | 正規化していない、または**参照と予測を逆**に渡した | `cer(reference, hypothesis)` の引数順を確認し、両方を同じ正規化に通す |
| CPU で推論が遅すぎる | `float16`/`half` を CPU 指定、`num_beams` 過大 | CPU は `float32`、`num_beams` を 1〜4、`max_new_tokens` を小さく |
| 毎回モデルを再 DL（Docker） | キャッシュ未マウント | `~/.cache/huggingface`（`HF_HOME`）をボリューム化、`HF_HUB_OFFLINE=1` で再現 |

とくに上位3つ――**tesseract 本体の未導入**、**TrOCR 出力の大文字化（＝正規化の必要性）**、**Donut の finetune 前重みの誤用**――は本章の「あるある」なので、症状を見たら即座に原因を言い当てられるようにしておきましょう。

## 🚀 発展トピック・参考

- **手書き OCR**: `microsoft/trocr-base-handwritten` / `-large-handwritten` で手書き文字に挑戦。印字版との CER 比較で「印字 vs 手書きの難しさ」を数値化できる。
- **行検出 → 認識のパイプライン**: TrOCR は認識専用なので、実文書では DBNet/CRAFT 等の**テキスト検出**で行を切り出してから TrOCR に渡す。EasyOCR の検出だけを使い、認識を TrOCR に差し替える構成も有効。
- **構造化抽出（CORD）**: `naver-clova-ix/donut-base-finetuned-cord-v2` でレシートを `{menu, total, ...}` の JSON へ構造化。`token2json` がネストした項目を返す様子を観察する。
- **ANLS の本来の定義**: DocVQA 公式では複数の正解候補に対する最大 ANLS を取る。本章は1正解での近似なので、複数正解への拡張を試すとより実態に近い評価になる。
- **日本語 OCR の実務**: 日本語は文字種が多くフォント依存も強い。Tesseract `jpn`、EasyOCR `ja`、`manga-ocr`（漫画/縦書き特化）などを CER で比較すると、言語・ドメイン適合の重要性が分かる。
- 公式ドキュメント: [TrOCR](https://huggingface.co/docs/transformers/model_doc/trocr) / [Donut](https://huggingface.co/docs/transformers/model_doc/donut) / [LayoutLMv3](https://huggingface.co/docs/transformers/model_doc/layoutlmv3) / [document-question-answering pipeline](https://huggingface.co/docs/transformers/main_classes/pipelines) / [torchmetrics CER/WER](https://lightning.ai/docs/torchmetrics/stable/text/char_error_rate.html) / [pytesseract](https://github.com/madmaze/pytesseract) / [EasyOCR](https://www.jaided.ai/easyocr/documentation/)。

## ▶ 動かし方

このモジュールは `dl`（torch/torchvision）・`hf`（transformers/sentencepiece ほか）・`metrics`（torchmetrics で CER/WER を検算）に依存します。TrOCR・Donut はいずれも HuggingFace transformers 同梱で、CPU だけで完走します（初回のみ重みをダウンロード、以降はキャッシュから即起動）。`pytesseract`/`easyocr`/`jiwer` は**任意**（未導入でも全スクリプトが exit 0）です。プロジェクトルートで以下を順に実行してください。

```bash
# 依存グループをインストール（初回のみ）
uv sync --group dl --group hf --group metrics

# 道具箱の自己点検（モデル不要・純計算）＋合成データの確認図
uv run python lectures/26_ocr_document/ocr_helpers.py

# 各スクリプト（結果は outputs/26_ocr_document/ に保存）
uv run python lectures/26_ocr_document/01_tesseract_easyocr.py   # 古典/深層 OCR（座標付き。未導入なら案内のみ）
uv run python lectures/26_ocr_document/02_trocr.py               # TrOCR で印字行を読み CER 算出
uv run python lectures/26_ocr_document/03_donut_docvqa.py        # Donut/DocVQA で帳票 QA（exact-match/ANLS）
uv run python lectures/26_ocr_document/04_cer_wer_eval.py        # CER/WER の定義・正規化・エンジン比較
uv run python lectures/26_ocr_document/mini_project.py           # 章末: 読み取り→構造化→評価の統合
uv run python lectures/26_ocr_document/use_case.py               # 実践: レシート読取（行検出→TrOCR→構造化 JSON）

# 演習: まず TODO を自分で埋める（最初は全 FAIL だが exit 0）
uv run python lectures/26_ocr_document/exercises.py
# 模範解答（実行すると全 PASS）
uv run python lectures/26_ocr_document/exercises_solutions.py

# （任意）実画像で試す: data/26_ocr_document/ に文書画像を置くと 03 / mini_project が拾う
# （任意）古典/深層 OCR も動かす: uv add --group ocr pytesseract easyocr + OS に tesseract-ocr
# （任意）別モデルに差し替え: TROCR_MODEL=... / DONUT_MODEL=... を環境変数で指定
```

実行後は `outputs/26_ocr_document/` の図と json を解説と照合してください。とくに `02_trocr_lines.png`（行ごとの GT/予測/CER）、`04_engine_cer.png`（エンジン別コーパス CER）、`03_docvqa_annotated.png`（請求書への回答）、`mini_summary.png`（認識×理解の総合スコア）の4枚を見ると、本章の要点が視覚的に腑に落ちます。図中の文字は CJK フォントの豆腐（□）を避けるため ASCII にしてあります。

## このモジュールの構成（スクリプト一覧）

各スクリプトは単一責務で、上から読むと「古典/深層 OCR を知る → 生成 OCR を動かす → 文書理解へ → 評価で締める → 統合する」と理解が積み上がります。device 判定・合成データ生成・正規化・CER/WER/ANLS・モデルロードといった共通処理は `ocr_helpers.py` に集約し、各スクリプトはそれを import します。

| ファイル | 役割（単一責務） |
| --- | --- |
| `ocr_helpers.py` | device 判定・合成データ（行/請求書＋GT）・正規化・編集距離/CER/WER/ANLS・TrOCR/Donut ロード＋推論・可視化。**道具箱** |
| `01_tesseract_easyocr.py` | 古典 Tesseract / 深層 EasyOCR の**座標付き OCR**。未導入なら API と導入法の案内のみ |
| `02_trocr.py` | TrOCR（VisionEncoderDecoder）で印字行を読み、CER(raw/norm)・greedy/beam を比較 |
| `03_donut_docvqa.py` | Donut/DocVQA（task_prompt→token2json）で帳票 QA。exact-match/ANLS、pipeline との整合 |
| `04_cer_wer_eval.py` | CER/WER を定義から検算（手作り vs torchmetrics）・正規化の効果・エンジン横並び比較 |
| `mini_project.py` | 章末統合: 認識(TrOCR/CER)×理解(Donut/ANLS)を1パイプラインに。総合レポート出力 |
| `use_case.py` | 実践ユースケース: レシート/書類リーダー。行検出→TrOCR→正規表現で構造化→`receipt.json`、Donut で合計をクロスチェック |
| `exercises.py` | TODO 形式の演習8問（自己採点ランナー付き・純計算・モデル DL 不要） |
| `exercises_solutions.py` | 演習の模範解答（実行すると全 PASS） |

## 💡 実践ユースケース集

本章の OCR・文書理解は「紙やスクショの中の文字を、下流システムが使える構造化データに変える」ための道具立てです。ここでは、教材の合成データを離れて**現実の小ツール**としてどう組むかを 3 つ挙げます。1 つ目は実際に動く `use_case.py` として同梱しました。

### ① レシート/書類リーダー（同梱: `use_case.py`）

**何に使うか**: レシート写真や帳票スキャンを 1 枚渡すと、**店名・日付・明細（品名と金額）・合計**を抽出して `receipt.json` に構造化し、経費精算アプリや帳簿入力の前段に流し込む小ツールです。章末ミニプロジェクトが「1 行ずつ切り出した合成行で **CER/WER/ANLS を測る（評価）**」のに対し、こちらは採点ではなく**成果物（構造化 JSON）を作る**のが目的という違いがあります。

**作り方の要点**: TrOCR は**認識専用で検出を持たない**（第2・5節）ため、複数行が並んだ 1 枚を直接読めません。そこで `use_case.py` はまず**横方向の射影プロファイル**（各行に黒画素＝インクがあるかを調べ、連続区間を 1 行とみなす）で**自前の行検出**を行い、各行を TrOCR で読み、**正規表現で店名/日付/明細/合計に構造化**します。さらに Donut/DocVQA に「What is the total?」を投げ、合計金額を**OCR フリーにクロスチェック**します（TrOCR の行読み取りと Donut の文書理解、両方を実地で組み合わせる構成）。

**注意**: 射影プロファイルは整ったスキャン/合成には十分ですが、**斜め・湾曲・影**には弱いので、実運用では二値化（大津法）や傾き補正の前処理を足します。正規表現パーサも「末尾に金額がある行＝明細」という素朴な仮定なので、数量×単価・税・小計・通貨記号が混ざる実レシートに合わせて育てる必要があります。

```bash
uv run python lectures/26_ocr_document/use_case.py
# → outputs/26_ocr_document/use_case_receipt.png（行検出の可視化）, use_case_receipt.json（構造化レコード）

# 実データで動かす: data/26_ocr_document/ にレシート/帳票画像（.png/.jpg/.jpeg/.bmp/.tif）を置くと
# その最初の1枚を自動で読み取る（置かなければ合成レシートで必ず完走＝exit 0）
```

**拡張アイデア**: 行検出に連結成分/傾き補正を足して斜めレシートへ対応／明細パーサに数量・税・軽減税率(*)を追加／Donut を `donut-base-finetuned-cord-v2`（レシート構造化）に差し替えて `token2json` のネスト構造を直接使い正規表現を置き換える／出力 JSON を会計ソフト取り込み形式（CSV 等）に整形／TrOCR の繰り返し・空行から「要確認」フラグを立てる。

### ② スキャン PDF の全文テキスト化＋個人情報マスキング

**何に使うか**: 契約書・申込書のスキャン画像を**検索可能なテキスト**にし、同時に氏名・口座番号などを**黒塗り（マスキング）**して共有用に出す前処理ツール。社内ナレッジ検索や RAG の取り込み前段で需要が大きい用途です。

**作り方の要点**: ここでは**座標が要る**ので TrOCR ではなく **Tesseract の `image_to_data`（`left/top/width/height/conf/text`）か EasyOCR の 4 点 quad** を使い、単語ごとの box を取得します。マスキング対象は「正規表現（口座番号・電話番号）」や「キーワード近傍」で当たりを付け、該当 box を `PIL.ImageDraw.rectangle` で塗りつぶします。全文は別途プレーンテキストとして保存します。

**注意**: マスキングは**座標付き OCR が前提**（TrOCR・Donut は座標を出さないので不向き）。OCR の取りこぼし＝マスク漏れ＝情報漏洩に直結するため、conf 閾値を低めにして**過検出側に倒す**、最後に人手レビューを挟むなど、安全側の設計が必須です。日本語は `tesseract-ocr-jpn` か EasyOCR `['ja']` を使います。

### ③ 帳票の項目抽出 API（Donut で OCR フリーに JSON 化）

**何に使うか**: 請求書・注文書をアップロードすると `{invoice_no, date, total, ...}` の **JSON を返す社内 API**。OCR の誤り伝播を避けたい・レイアウトが効く定型フォームで威力を発揮します。

**作り方の要点**: `pipeline("document-question-answering", model="naver-clova-ix/donut-base-finetuned-docvqa")` に必要項目ぶんの質問（「What is the invoice number?」等）を投げ、答えを 1 つの dict にまとめて返すだけ。構造化レシートなら `donut-base-finetuned-cord-v2` の `token2json` で**ネストした明細 JSON を一発取得**できます。OCR・行検出・正規表現が不要になるのが Donut 系の最大の利点です。

**注意**: Donut は**タスク別 finetune 重みが必須**（`donut-base` 素体は意味ある答えを返さない）で、**未知レイアウトへの一般化が弱い**。新フォームには finetune が要る点と、CPU では 1 枚あたり数秒かかる点を運用前提に織り込みます。回答の妥当性は exact-match だけでなく **ANLS**（綴り揺れに寛容）で監視すると実態に即します。

---

> 本教材で参照・検証したライブラリとバージョン（2026-06 時点の安定版で動作確認）:
> Python 3.12 ／ torch 2.12.0+cpu ／ torchvision 0.27.0+cpu ／ transformers 5.11.0 ／ huggingface-hub 1.18.0 ／ torchmetrics 1.9.0 ／ sentencepiece 0.2.1 ／ numpy 2.4.6 ／ Pillow 12.2.0 ／ matplotlib 3.10.9 ／ opencv-python-headless 4.13.0
> 使用モデル: `microsoft/trocr-base-printed`（TrOCR・VisionEncoderDecoder）／ `naver-clova-ix/donut-base-finetuned-docvqa`（Donut・DocVQA）。いずれも HuggingFace transformers 同梱で、初回のみ重みを取得しキャッシュします。`pytesseract`/`easyocr`/`jiwer` は任意（未導入でも全スクリプト exit 0）。transformers v5 準拠（`AutoProcessor`/`AutoModelForVision2Seq`/`AutoImageProcessor`、`AutoFeatureExtractor` は廃止）。