# 第20回 オープン語彙物体検出 — OWL-ViT / OWLv2・Grounding DINO

> トラック: **検出** ／ レベル: **中級** ／ 依存グループ: `dl`（torch/torchvision）・`hf`（transformers/timm ほか）。CPU だけで完走します（初回のみモデル重みを HuggingFace からダウンロード）。

## 🎯 この章のゴール

第18回では物体検出を「学習時に決めた固定クラス（COCO の 80 種など）の枠で、どこに何があるかを当てる」タスクとして学びました。しかし現実の要望は、「**棚にある"赤い消火器"を探したい**」「**この画像から"信号機"だけ抜きたい**」のように、その場で決めた任意の語で物を見つけたい、というものです。固定クラスの検出器は、学習し直さない限りこうした要望には答えられません。そこで本章で扱う**オープン語彙物体検出（open-vocabulary detection, OVD）**は、第16回の CLIP ゼロショット「分類」を「**検出**（=位置＋任意ラベル）」へ拡張し、推論時に渡したテキストだけで未学習カテゴリを検出できるようにします。

この章を終えると、3つのことが自分の手でできるようになります。第一に、**OWL-ViT / OWLv2** に「候補ラベルのリスト」を渡して検出し、`post_process_grounded_object_detection` で box/score/label を取り出せること（`target_sizes=(H,W)` の座標変換も自分で扱う）。第二に、**Grounding DINO** に「小文字＋ピリオド区切りのキャプション」を渡し、`box_threshold` と `text_threshold` の2つの閾値で過検出/未検出を制御できること。第三に、GT 付きの画像で **precision/recall/F1 を閾値スイープ**して PR 曲線と F1 最大点を求め、「どの閾値で切るべきか」を定量的に選べることです。

本章のスクリプトは、ネット接続もデータセットの DL も無しで完走できるよう、入力画像を**その場で合成**します（明るい背景に「赤い円」「青い四角」「緑の三角」「黄色い円」を描いた、GT ボックス付きのシーン）。OVD モデルは「色」「形」という概念を強く捉えるため、合成画像でも `"a red circle"` がきちんと当たり、教材として意味のある検出・評価が得られます。なお、合成画像では検出が乏しいこともありますが、その場合でもスクリプトは必ず `exit 0` で終わります。**実写で実用的に試したい人は、`data/20_open_vocabulary_detection/` に画像を置けば自動で使われます**（ただし GT が無いため、03 の評価だけは合成シーンに切り替わります）。ダウンロードが走るのは、初回のモデル重み取得（OWL-ViT / OWLv2 / Grounding DINO）のときだけです。

---

## 1. 閉語彙 vs 開語彙 — なぜテキストで検出できるのか

ふつうの検出器（第18回の Faster R-CNN や DETR）は、出力ヘッドに「COCO の 80 クラス」のような固定された分類層を持ちます。そのため、学習時に消火器クラスが無ければ対応するノードも存在せず、推論時に「これは消火器か」と問うことはできません。これを**閉語彙（closed-vocabulary）**検出と呼びます。一方、オープン語彙検出は、検出した各領域を**固定ノードで分類する代わりに、テキスト埋め込みとの類似度で名付ける**という発想へ切り替えます。つまり、領域の画像埋め込みと「a fire extinguisher」という文の埋め込みが近ければ、その領域を消火器と呼ぶわけです。これは第16回の CLIP の原理を、画像全体ではなく**各候補領域**に適用したものにほかなりません。

<figure class="lec-fig"><svg viewBox="0 0 660 330" role="img" aria-label="閉語彙は固定分類ヘッドで学習済みクラスしか出せず、開語彙は領域埋め込みとテキスト埋め込みのコサイン類似度で任意ラベルを名付ける" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><line x1="330" y1="24" x2="330" y2="300" stroke="#e4e4e7" stroke-width="1.5"/><text x="176" y="42" text-anchor="middle" font-size="17" font-weight="700" fill="#2563eb">閉語彙 — 固定クラス</text><rect x="42" y="96" width="72" height="72" fill="#fff7ed" stroke="#c2410c" stroke-width="2"/><text x="78" y="137" text-anchor="middle" font-size="14" font-weight="700" fill="#c2410c">領域</text><line x1="116" y1="132" x2="158" y2="132" stroke="#71717a" stroke-width="2"/><polygon points="166,132 156,127 156,137" fill="#71717a"/><rect x="170" y="84" width="140" height="130" fill="#eff6ff" stroke="#2563eb" stroke-width="1.8"/><text x="240" y="106" text-anchor="middle" font-size="12.5" font-weight="700" fill="#2563eb">固定ヘッド 80クラス</text><rect x="186" y="116" width="108" height="16" fill="#dbeafe" stroke="#2563eb" stroke-width="1"/><rect x="186" y="138" width="108" height="16" fill="#2563eb"/><rect x="186" y="160" width="108" height="16" fill="#dbeafe" stroke="#2563eb" stroke-width="1"/><rect x="186" y="182" width="108" height="16" fill="#dbeafe" stroke="#2563eb" stroke-width="1"/><text x="176" y="252" text-anchor="middle" font-size="13" fill="#52525b">学習した語しか出せない</text><text x="494" y="42" text-anchor="middle" font-size="17" font-weight="700" fill="#c2410c">開語彙 — 任意ラベル</text><rect x="352" y="96" width="72" height="72" fill="#fff7ed" stroke="#c2410c" stroke-width="2"/><text x="388" y="137" text-anchor="middle" font-size="12" font-weight="700" fill="#c2410c">領域埋込</text><line x1="426" y1="132" x2="470" y2="132" stroke="#16a34a" stroke-width="2"/><polygon points="478,132 468,127 468,137" fill="#16a34a"/><text x="448" y="122" text-anchor="middle" font-size="16" font-weight="700" fill="#16a34a">≈</text><rect x="482" y="86" width="156" height="128" fill="#fff7ed" stroke="#c2410c" stroke-width="1.8"/><rect x="496" y="100" width="128" height="22" fill="#ffedd5" stroke="#ea580c" stroke-width="1"/><text x="560" y="115" text-anchor="middle" font-size="11.5" fill="#c2410c">a red circle</text><rect x="496" y="128" width="128" height="22" fill="#ffedd5" stroke="#ea580c" stroke-width="1"/><text x="560" y="143" text-anchor="middle" font-size="11" fill="#c2410c">a fire extinguisher</text><rect x="496" y="156" width="128" height="22" fill="#ffedd5" stroke="#ea580c" stroke-width="1"/><text x="560" y="171" text-anchor="middle" font-size="11" fill="#71717a">＋ 任意の語OK</text><text x="494" y="252" text-anchor="middle" font-size="13" fill="#52525b">推論時に語を差し替え自由</text></svg><figcaption>閉語彙の検出器は領域を<b>固定の分類ヘッド</b>（学習済みクラス）に押し込むため、学習した語しか出せません。オープン語彙検出は、領域の埋め込みと<b>候補ラベルのテキスト埋め込み</b>との<b>コサイン類似度</b>で名付けるので、<code>candidate_labels</code> を推論時に差し替えるだけで任意の語を検出できます（第16回 CLIP の領域版）。</figcaption></figure>

この「領域 × テキスト」の照合をどう実装するかによって、手法は大きく2系統に分かれます。一方の**OWL-ViT / OWLv2** は、ViT で画像をパッチに分け、各パッチ（=候補領域）の埋め込みと、候補ラベルをエンコードしたテキスト埋め込みとのコサイン類似度を取って分類します。検出ヘッドは「box の回帰」だけを担い、クラス分類はテキスト埋め込みとの内積に置き換わっている、と捉えると分かりやすいでしょう。したがって、候補ラベルを推論時に差し替えるだけで、何クラスでも自由に検出できます。もう一方の**Grounding DINO** はさらに一歩進めて、テキストと画像特徴を検出器の途中で**クロスアテンションにより早期融合（early fusion）**します。これにより「the man on the left（左の男）」のような**参照表現**にも反応しやすくなりますが、その代わり入力テキストの書式が厳密になります（第6節）。

OVD のうれしさは、**幻覚しにくさ**にも表れます。閉語彙の分類器には「与えられた候補のどれかに必ず割り当てる」傾向（第16回の softmax の副作用）がありますが、OVD は各領域とテキストの類似度が閾値を超えたものだけを検出として残します。そのため「シーンに無いラベル」はスコアが上がらず、ほとんど検出されません。本章では候補ラベルにわざと「a dog」「a traffic light」という**シーンに無い語**を混ぜ、それらが検出されない（=幻覚しない）ことを実測で確認します。それでは原理を頭に入れたうえで、次節で実際に動かしてみましょう。

## 2. 最短で動かす — `pipeline('zero-shot-object-detection')`

理屈はいったん脇に置き、まず成功体験を得るのが近道です。transformers の高レベル API `pipeline("zero-shot-object-detection")` は、前処理（画像のリサイズ・正規化、テキストのトークン化）から推論、後処理（box の座標変換・閾値フィルタ）までを一手に引き受けてくれます。`01_owlvit_owlv2.py` はまずこれを使い、合成シーンに対して `candidate_labels`（その場で渡す任意のラベル候補）を検出します。この `candidate_labels` を自由に決められることこそが、「オープン語彙」を体感するポイントです。

下が pipeline 呼び出しの核です。`task` と `model` を指定し、あとは画像・候補ラベル・閾値を渡すだけです。`device` は CPU なら `-1`、CUDA があれば `0` を渡します。返り値は、検出ごとに `{'score':…, 'label':…, 'box':{'xmin':…,'ymin':…,'xmax':…,'ymax':…}}` を並べたリストです。

```python
from transformers import pipeline

detector = pipeline("zero-shot-object-detection",
                    model="google/owlvit-base-patch32", device=-1)  # CPU は -1
labels = ["a red circle", "a blue square", "a green triangle",
          "a yellow circle", "a dog", "a traffic light"]   # 任意に決められる
outs = detector(image, candidate_labels=labels, threshold=0.1)  # score でフィルタ
# outs: [{'score':0.41,'label':'a green triangle','box':{'xmin':110,...}}, ...]
```

合成シーンでの実行結果は、`a green triangle (0.412)` `a red circle (0.385)` `a yellow circle (0.303)` `a blue square (0.294)` の4件です。**4物体すべてが正しく当たり**、`a dog` と `a traffic light` は1件も出ません。スコアが 0.3 前後と控えめなのは OWL-ViT-base の素の特性で、`threshold` をどこに置くかで検出数が変わります。このように pipeline は手軽な一方、内部の前処理・後処理はブラックボックスのままです。学習目的では `target_sizes=(H,W)` の座標変換を自分で扱える必要があるため、次節では同じ処理を手書きに分解していきます。

## 3. 手書きに分解する — `post_process_grounded_object_detection` と `target_sizes=(H,W)`

`01_owlvit_owlv2.py` は続いて、pipeline を `processor` と `model` の2部品に分解します。OVD モデルはどれも **`AutoProcessor` + `AutoModelForZeroShotObjectDetection`** でロードでき、後処理も **`processor.post_process_grounded_object_detection(...)`** に統一されています（これは transformers v5 の重要な作法です。第10節）。OWL の `processor` は「画像」と「候補ラベルの集合」を同時に前処理しますが、候補ラベルとして「**画像ごとのラベル集合**」を期待するため、`text=[labels]` のように**一段ネスト**して渡すのが鉄則です。

<figure class="lec-fig"><svg viewBox="0 0 560 280" role="img" aria-label="OWL検出はpipelineを入力・AutoProcessorの前処理・modelの推論・post_processの後処理に分解し、xyxy絶対座標のboxを得る" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><text x="280" y="32" text-anchor="middle" font-size="14" font-weight="700" fill="#18181b">OWL 検出のパイプライン — pipeline を部品に分解</text><rect x="40" y="64" width="200" height="80" rx="8" fill="#f4f4f5" stroke="#71717a" stroke-width="2"/><rect x="320" y="64" width="200" height="80" rx="8" fill="#fff7ed" stroke="#ea580c" stroke-width="2"/><rect x="320" y="196" width="200" height="80" rx="8" fill="#fff7ed" stroke="#ea580c" stroke-width="2"/><rect x="40" y="196" width="200" height="80" rx="8" fill="#eff6ff" stroke="#2563eb" stroke-width="2"/><text x="140" y="100" text-anchor="middle" font-size="15" font-weight="700" fill="#3f3f46">① 入力</text><text x="140" y="124" text-anchor="middle" font-size="12" fill="#71717a">画像 ＋ 候補ラベル</text><text x="420" y="100" text-anchor="middle" font-size="15" font-weight="700" fill="#c2410c">② 前処理</text><text x="420" y="124" text-anchor="middle" font-size="12" fill="#71717a">AutoProcessor</text><text x="420" y="232" text-anchor="middle" font-size="15" font-weight="700" fill="#c2410c">③ 推論</text><text x="420" y="256" text-anchor="middle" font-size="12" fill="#71717a">model(...)</text><text x="140" y="232" text-anchor="middle" font-size="15" font-weight="700" fill="#1d4ed8">④ 後処理・出力</text><text x="140" y="256" text-anchor="middle" font-size="12" fill="#71717a">post_process / box xyxy</text><line x1="246" y1="104" x2="314" y2="104" stroke="#71717a" stroke-width="2"/><polygon points="320,104 310,99 310,109" fill="#71717a"/><line x1="420" y1="144" x2="420" y2="190" stroke="#71717a" stroke-width="2"/><polygon points="420,196 415,186 425,186" fill="#71717a"/><line x1="314" y1="236" x2="246" y2="236" stroke="#71717a" stroke-width="2"/><polygon points="240,236 250,231 250,241" fill="#71717a"/></svg><figcaption><b>OWL 検出</b>は、高レベルの <code>pipeline</code> を <b>① 入力（画像＋候補ラベル）</b> → <b>② 前処理（AutoProcessor）</b> → <b>③ 推論（model）</b> → <b>④ 後処理（post_process_grounded_object_detection）</b> に分解できます。後処理を通すと、生の <code>pred_boxes</code> が <b>xyxy の絶対座標</b>の box へ変換されます。このとき <code>target_sizes</code> を <b>(H, W)</b> 順で渡す点が肝心です（次の図）。</figcaption></figure>

```python
import torch
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

proc = AutoProcessor.from_pretrained("google/owlvit-base-patch32")
model = AutoModelForZeroShotObjectDetection.from_pretrained(
    "google/owlvit-base-patch32").to(device).eval()       # eval() を忘れない

inputs = proc(text=[labels], images=image, return_tensors="pt").to(device)  # [labels] と一段ネスト
with torch.inference_mode():                                # 推論は勾配を切る
    outputs = model(**inputs)

target_sizes = torch.tensor([image.size[::-1]])             # ★ (H, W) 順！ image.size は (W,H)
result = proc.post_process_grounded_object_detection(
    outputs=outputs, threshold=0.1,
    target_sizes=target_sizes, text_labels=[labels])[0]
boxes  = result["boxes"]        # (N,4) xyxy 絶対座標
scores = result["scores"]       # (N,)
texts  = result["text_labels"]  # 候補ラベルの文字列リスト
```

ここで本章最大の落とし穴となるのが、**`target_sizes` の順序**です。モデルの生出力 `pred_boxes` は **cxcywh の正規化座標（0〜1）**であり、これを `post_process_*` が「`target_sizes` で指定した画像サイズ」に合わせて **xyxy の絶対座標**へ変換してくれます。このとき `target_sizes` は、必ず **`(height, width)` 順**で渡さなければなりません。ところが PIL の `image.size` は **`(width, height)`** を返すため、`image.size[::-1]` のように**反転**させて渡す必要があります。ここを取り違えると、box が横長/縦長に歪む典型的なバグになります（演習 ex2 ではこの座標変換を手で実装します）。なお、可視化の前には必ずこの後処理を通してください――生の `pred_boxes` をそのまま描いても、まともな box にはなりません。

<figure class="lec-fig"><svg viewBox="0 0 540 340" role="img" aria-label="モデルのcxcywh正規化座標をtarget_sizes=(H,W)でxyxy絶対座標へ変換する。(W,H)のまま渡すとboxが歪む" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><text x="120" y="38" text-anchor="middle" font-size="13.5" font-weight="700" fill="#18181b">モデル生出力</text><rect x="44" y="64" width="152" height="152" fill="#fafafa" stroke="#d4d4d8" stroke-width="1.5"/><text x="38" y="60" text-anchor="end" font-size="11" fill="#71717a">0</text><rect x="92" y="104" width="64" height="80" fill="#dbeafe" stroke="#2563eb" stroke-width="2"/><circle cx="124" cy="144" r="3.5" fill="#2563eb"/><text x="120" y="242" text-anchor="middle" font-size="12.5" fill="#2563eb">cxcywh 正規化</text><text x="120" y="262" text-anchor="middle" font-size="11.5" fill="#71717a">全要素 0〜1</text><line x1="206" y1="150" x2="262" y2="150" stroke="#71717a" stroke-width="2"/><polygon points="270,150 260,145 260,155" fill="#71717a"/><text x="236" y="138" text-anchor="middle" font-size="11.5" font-weight="700" fill="#c2410c">post_process</text><text x="236" y="172" text-anchor="middle" font-size="13" font-weight="700" fill="#c2410c">× (H, W)</text><text x="402" y="52" text-anchor="middle" font-size="12.5" font-weight="700" fill="#15803d">(H, W) を渡す = 正しい</text><rect x="312" y="60" width="180" height="104" fill="#fafafa" stroke="#d4d4d8" stroke-width="1.5"/><rect x="370" y="86" width="76" height="56" fill="none" stroke="#16a34a" stroke-width="2.5"/><text x="402" y="196" text-anchor="middle" font-size="12.5" font-weight="700" fill="#dc2626">(W, H) を渡す = 歪む</text><rect x="312" y="204" width="180" height="104" fill="#fafafa" stroke="#d4d4d8" stroke-width="1.5"/><rect x="356" y="230" width="116" height="38" fill="none" stroke="#dc2626" stroke-width="2.5"/><text x="402" y="326" text-anchor="middle" font-size="12" fill="#dc2626">横長に歪む</text></svg><figcaption>モデルの生出力 <code>pred_boxes</code> は <b>cxcywh の正規化座標（0〜1）</b>です。<code>post_process_grounded_object_detection</code> がこれを <b>xyxy の絶対座標</b>へ変換しますが、<code>target_sizes</code> は必ず <b>(H, W) 順</b>で渡します。PIL の <code>image.size</code> は <b>(W, H)</b> を返すので <code>image.size[::-1]</code> で反転し、(W, H) のまま渡すと box が横長/縦長に<b>歪みます</b>。</figcaption></figure>

## 4. OWL-ViT vs OWLv2 — スコアの出方と幻覚のしにくさ

`01` は、同じ候補ラベルを使って **OWL-ViT**（`owlvit-base-patch32`）と **OWLv2**（`owlv2-base-patch16-ensemble`）を比較します。OWLv2 は OWL-ViT の後継で、**自己学習（self-training）**――OWL-ViT 自身に大量の画像へ擬似ラベルを付けさせ、それで再学習する――によって精度と確信度を底上げしたモデルです。実測してみると、同じ4物体に対するスコアは下表のように大きく異なります。OWL-ViT が 0.3 前後なのに対し、OWLv2 は 0.7〜0.8 と、**一段高い確信度**で当てます。

| モデル | red circle | blue square | green triangle | yellow circle | 無いラベルの検出 |
| --- | --- | --- | --- | --- | --- |
| OWL-ViT (base-patch32) | 0.385 | 0.294 | 0.412 | 0.303 | **0 件** |
| OWLv2 (base-patch16-ensemble) | **0.804** | **0.723** | **0.755** | **0.761** | **0 件** |

この差は、実務の閾値設定に直結します。OWL-ViT はスコアが低めなので `threshold` を 0.1 程度まで下げないと取りこぼしますが、OWLv2 は 0.2〜0.3 でも十分に拾えます（スコアのスケールがモデルごとに異なるため、**閾値はモデルとセットで決める**のが鉄則です。第9節で定量化します）。一方で、両モデルとも**「a dog」「a traffic light」を1件も検出しません**。これこそが OVD の「幻覚しにくさ」であり、固定クラス分類器が「候補のどれかに必ず分類してしまう」のとは対照的です。`01_owlvit_detections.png` / `01_owlv2_detections.png` は、検出 box（色つき実線）と GT box（灰色点線）を重ねた図で、両者がほぼ一致していることを目視で確認できます。精度の高さだけを見れば OWLv2 が有利ですが、`base-patch16-ensemble` は OWL-ViT-base-patch32 より**重く、ロードも遅い**ため、CPU で素早く試すなら OWL-ViT、精度が要るなら OWLv2、という使い分けになります。

## 5. Grounding DINO — キャプション形式と box/text 閾値（過検出の体感）

`02_grounding_dino.py` は、OWL とは入力の渡し方が異なるもう1系統――**Grounding DINO**（`grounding-dino-tiny`）を扱います。OWL が「候補ラベルの**リスト**」を取るのに対し、Grounding DINO は1本の「**キャプション**」を取り、そのキャプション中の語句に対応する物体を検出します。ここで書式は厳密で、**小文字に統一し、各物体をピリオド `.` で区切る**必要があります（例: `"a cat. a remote control."`）。そこで本章では、候補ラベルを `labels_to_caption()` でこの形式に整形してから渡します。書式を崩すと検出が安定しないため、これが最も頻度の高い落とし穴です。

```python
# 候補ラベル → Grounding DINO 用キャプション（小文字＋ピリオド区切り）
caption = "a red circle. a blue square. a green triangle. a yellow circle. a dog. a traffic light."
inputs = proc(images=image, text=caption, return_tensors="pt").to(device)
with torch.inference_mode():
    outputs = model(**inputs)
result = proc.post_process_grounded_object_detection(
    outputs, input_ids=inputs["input_ids"],   # ★ どの語に紐づくか判定するため input_ids が要る
    threshold=0.35,            # box_threshold: 検出ボックスの確信度
    text_threshold=0.25,       # text_threshold: ボックスを語に対応づける確信度
    target_sizes=torch.tensor([image.size[::-1]]))[0]   # ここも (H, W)
```

Grounding DINO には**閾値が2つ**あります。`box_threshold`（後処理引数では `threshold`）は「検出ボックスそのものの確信度」、`text_threshold` は「そのボックスをキャプション中のどの語に結びつけるかの確信度」を表します。`02` は、これを**緩い設定**（box≥0.15, text≥0.15）と**厳しい設定**（box≥0.35, text≥0.25）で比較します。結果は劇的です。緩い設定では **19 件**も検出され、その大半が `a traffic light`（シーンに無い）への過検出や、`a` `a light` のような**語の断片への誤対応**でした。一方、厳しい設定では**ちょうど4件**――4物体に正しく1つずつ――に収まります。

| 設定 | box_threshold | text_threshold | 検出数 | 内訳 |
| --- | --- | --- | --- | --- |
| loose | 0.15 | 0.15 | **19** | 正解4 ＋ "a traffic light" 多数 ＋ "a" / "a light" など断片 |
| strict | 0.35 | 0.25 | **4** | 4物体に1つずつ（過検出なし） |

この実験の教訓は、「**Grounding DINO はデフォルト閾値のままだと過検出/未検出になりやすく、2つの閾値の調整が必須**」ということです。`02_gdino_loose.png` と `02_gdino_strict.png` を見比べると、緩い設定では box が乱立し、厳しい設定ではスッキリ4つに収まる様子が一目で分かります。なお Grounding DINO は OWL より重く（`timm` 依存で、初回 DL も大きめ）、環境次第で失敗しうるため、`02` はロード/推論を `try/except` で包み、失敗した場合は**概念紹介だけを出して必ず `exit 0`** になるようにしてあります。

## 6. OWL と Grounding DINO の使い分け

ここまでで2系統を触ってきました。ここで、実務での使い分けの指針を整理しておきましょう。まず **OWL-ViT / OWLv2** は、「検出したいカテゴリのリストが決まっている」場面に向きます。候補ラベルを配列で渡すだけで、各ラベルが独立に評価され、結果にも「どのラベルか（`labels` のインデックス）」がきれいに付きます。したがって、在庫検品や特定物体の有無チェックのように「**語彙が列挙できる**」用途に素直です。OWLv2 は精度重視、OWL-ViT-base-patch32 は CPU で軽く試す用、と覚えておいてください。

一方の **Grounding DINO** は、「**自然言語のフレーズ・参照表現で指したい**」場面に向きます。早期融合のおかげで `"a person wearing a red hat"` のような修飾付き表現や関係表現に強く、後段で **SAM** にボックスを渡せば、任意領域のセグメンテーション（**Grounded-SAM**、第23回で扱う）へ発展させられます。反面、キャプション書式（小文字＋ピリオド）と2つの閾値という「お作法」が増え、断片語への過検出にも気を配る必要があります。まとめると、**列挙できる固定ラベルなら OWL、自由な言語表現や下流の SAM 連携なら Grounding DINO**、というのが第一感の使い分けです。

実装面の共通点も押さえておきましょう。どちらも `AutoModelForZeroShotObjectDetection` でロードでき、後処理は `post_process_grounded_object_detection`、`target_sizes` は `(H,W)`、出力は xyxy 絶対座標の box です。違いは「クエリの渡し方（リスト or キャプション）」と「閾値の数（OWL は1つ、GDINO は box/text の2つ）」だけです。この共通骨格を `ovd_helpers.detect_owl` / `detect_gdino` に薄くまとめてあるので、両者の差分が読み取りやすくなっています。

## 7. 検出をどう評価するか — IoU・貪欲マッチング・P/R/F1

「それっぽく検出できた」を「どれだけ正しいか」へ変えるのが評価です。本章は第19回（mAP 自作）と同じ筋で、**IoU → 貪欲マッチング → TP/FP/FN → precision/recall/F1** を numpy で組み立てます。まず **IoU（Intersection over Union）** は、2つの box の「重なり面積 ÷ 和集合面積」であり、1.0 が完全一致、0.0 が無重なりを表します。これを、検出が GT に「当たった」とみなす基準（本章は **IoU≥0.5**）として使います。

<figure class="lec-fig"><svg viewBox="0 0 600 300" role="img" aria-label="IoUは予測boxとGT boxの重なり面積を和集合面積で割った値。1.0が完全一致でIoU0.5以上を当たりとする" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><rect x="150" y="120" width="70" height="100" fill="#ffedd5"/><rect x="70" y="70" width="150" height="150" fill="none" stroke="#71717a" stroke-width="2" stroke-dasharray="5 4"/><rect x="150" y="120" width="150" height="130" fill="none" stroke="#16a34a" stroke-width="2.5"/><text x="185" y="174" text-anchor="middle" font-size="12.5" font-weight="700" fill="#c2410c">交差</text><text x="74" y="62" text-anchor="start" font-size="12.5" font-weight="700" fill="#71717a">GT (正解)</text><text x="298" y="266" text-anchor="end" font-size="12.5" font-weight="700" fill="#16a34a">予測</text><text x="360" y="144" font-size="17" font-weight="700" fill="#18181b">IoU =</text><text x="478" y="120" text-anchor="middle" font-size="14" fill="#ea580c">交差(重なり)</text><line x1="420" y1="134" x2="540" y2="134" stroke="#18181b" stroke-width="1.5"/><text x="478" y="154" text-anchor="middle" font-size="14" fill="#3f3f46">和集合(全体)</text><text x="430" y="200" text-anchor="middle" font-size="12.5" fill="#52525b">1.0 = 完全一致 / 0 = 無重なり</text><text x="430" y="230" text-anchor="middle" font-size="13" font-weight="700" fill="#16a34a">当たり判定: IoU ≥ 0.5</text></svg><figcaption><b>IoU（Intersection over Union）</b>は、予測 box と GT box の<b>重なり面積 ÷ 和集合面積</b>です。<b>1.0 が完全一致</b>・0 が無重なりで、本章は <b>IoU ≥ 0.5</b> を「当たった」とみなす基準に使います。OVD ではラベルの一致も要求する（クラス込み）ため、ラベルが違えば box が重なっても当たりにはなりません。</figcaption></figure>

次に**貪欲マッチング**です。まず予測を**スコア降順**に並べ、各予測について「**同じラベルかつ未マッチ**で IoU≥0.5 の GT」のうち IoU 最大のものに対応づけます。対応がつけば **TP（真陽性）**、対応する GT が無ければ **FP（偽陽性）** です。そして、最後まで誰にも対応されなかった GT が **FN（偽陰性）** となります。1つの GT に複数の予測が当たっても、**TP は最初の1つだけ**で残りは FP です――この「1 GT につき 1 検出」という規則こそが、過検出を正しくペナルティする鍵になります。なお OVD ではラベルの一致も要求する（**クラス込み**マッチング）ため、`a green triangle` の box が `a red circle` の GT に重なっても TP にはなりません。

<figure class="lec-fig"><svg viewBox="0 0 600 300" role="img" aria-label="予測をスコア降順に並べGTへ貪欲マッチング。緑の線が対応するTP、余った予測がFP、余ったGTがFN" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><text x="130" y="38" text-anchor="middle" font-size="13" font-weight="700" fill="#2563eb">予測 (スコア降順)</text><text x="470" y="38" text-anchor="middle" font-size="13" font-weight="700" fill="#71717a">GT (正解)</text><line x1="200" y1="78" x2="400" y2="78" stroke="#16a34a" stroke-width="2.5"/><line x1="200" y1="130" x2="400" y2="130" stroke="#16a34a" stroke-width="2.5"/><rect x="60" y="58" width="140" height="40" fill="#dbeafe" stroke="#2563eb" stroke-width="1.8"/><text x="130" y="83" text-anchor="middle" font-size="13" font-weight="700" fill="#1d4ed8">P1   0.80</text><rect x="60" y="110" width="140" height="40" fill="#dbeafe" stroke="#2563eb" stroke-width="1.8"/><text x="130" y="135" text-anchor="middle" font-size="13" font-weight="700" fill="#1d4ed8">P2   0.72</text><rect x="60" y="162" width="140" height="40" fill="#fff7ed" stroke="#dc2626" stroke-width="1.8"/><text x="130" y="187" text-anchor="middle" font-size="13" font-weight="700" fill="#dc2626">P3   0.40</text><rect x="400" y="58" width="140" height="40" fill="#f4f4f5" stroke="#71717a" stroke-width="1.8"/><text x="470" y="83" text-anchor="middle" font-size="13" font-weight="700" fill="#3f3f46">GT-A</text><rect x="400" y="110" width="140" height="40" fill="#f4f4f5" stroke="#71717a" stroke-width="1.8"/><text x="470" y="135" text-anchor="middle" font-size="13" font-weight="700" fill="#3f3f46">GT-B</text><rect x="400" y="162" width="140" height="40" fill="#fff7ed" stroke="#dc2626" stroke-width="1.8"/><text x="470" y="187" text-anchor="middle" font-size="13" font-weight="700" fill="#dc2626">GT-C</text><text x="208" y="186" text-anchor="start" font-size="12" font-weight="700" fill="#dc2626">FP</text><text x="392" y="186" text-anchor="end" font-size="12" font-weight="700" fill="#dc2626">FN</text><text x="300" y="250" text-anchor="middle" font-size="12.5" font-weight="700" fill="#c2410c">1 GT に 1 検出だけ TP / 余りは FP・FN</text></svg><figcaption>予測を<b>スコア降順</b>に並べ、各予測を「同じラベルかつ未マッチで IoU≥0.5」の GT のうち IoU 最大に対応づけます。対応がつけば <b>TP</b>（緑の線）、余った予測は <b>FP</b>、誰にも対応されなかった GT は <b>FN</b> です。<b>1 つの GT に対応する TP は最初の 1 件だけ</b>で、同じ GT に当たった残りの予測（P3）は FP になります。</figcaption></figure>

```python
# precision/recall/F1 の定義（TP/FP/FN から）
precision = TP / (TP + FP)        # 検出のうち正しかった割合
recall    = TP / (TP + FN)        # GT のうち拾えた割合
F1        = 2 * P * R / (P + R)   # 両者の調和平均（バランス指標）
```

`precision` は「検出した中で正しい割合（過検出すると下がる）」、`recall` は「あるべき物体を拾えた割合（取りこぼすと下がる）」、`F1` はその調和平均です。検出を厳しく絞れば precision は上がるが recall は下がり、緩めれば逆になる――この**トレードオフ**を、1本の閾値が決めます。だからこそ「どの閾値が良いか」を測る必要があり、それを担うのが次節の閾値スイープです。これらの計算は `ovd_helpers.iou_xyxy` / `greedy_match` / `prf` にまとめてあり、演習 ex1/ex3/ex4 で手を動かして再現します。

## 8. 閾値スイープで P/R/F1 — 閾値選択を定量化する

`03_threshold_sweep_eval.py` は、GT 付き合成シーンを使って「**どの閾値で切るのが妥当か**」を数字で求めます。やり方はシンプルで、3ステップです。(1) モデルを**十分低い閾値**で1回だけ走らせ、「全候補（スコア付き）」を集める。(2) スコア閾値 `t` を 0.05〜0.95 で掃引し、`t` 以上の検出だけを残して第7節のマッチングで TP/FP/FN を数え、P/R/F1 を出す。(3) **F1 が最大になる `t`** を「推奨閾値」とする。1回の推論結果を後処理で何度も閾値フィルタするだけなので、モデルを何度も走らせる必要はありません。

<figure class="lec-fig"><svg viewBox="0 0 420 320" role="img" aria-label="閾値スイープは低い閾値で1回だけ推論して全候補を集め、閾値tを掃引してP/R/F1を出し、F1最大のtを推奨閾値とする3ステップ" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><text x="210" y="32" text-anchor="middle" font-size="14" font-weight="700" fill="#18181b">閾値スイープの 3 ステップ</text><rect x="60" y="52" width="300" height="66" rx="8" fill="#fff7ed" stroke="#ea580c" stroke-width="2"/><rect x="60" y="146" width="300" height="66" rx="8" fill="#fff7ed" stroke="#ea580c" stroke-width="2"/><rect x="60" y="240" width="300" height="66" rx="8" fill="#fafafa" stroke="#16a34a" stroke-width="2.5"/><text x="210" y="82" text-anchor="middle" font-size="15" font-weight="700" fill="#c2410c">① 低い閾値で 1 回だけ推論</text><text x="210" y="104" text-anchor="middle" font-size="11.5" fill="#71717a">全候補（スコア付き）を集める</text><text x="210" y="176" text-anchor="middle" font-size="15" font-weight="700" fill="#c2410c">② 閾値 t を 0.05〜0.95 で掃引</text><text x="210" y="198" text-anchor="middle" font-size="11.5" fill="#71717a">各 t で TP/FP/FN → P/R/F1</text><text x="210" y="270" text-anchor="middle" font-size="15" font-weight="700" fill="#15803d">③ F1 最大の t を推奨閾値に</text><text x="210" y="292" text-anchor="middle" font-size="11.5" fill="#15803d">低すぎ＝過検出 / 高すぎ＝取りこぼし</text><line x1="210" y1="118" x2="210" y2="140" stroke="#71717a" stroke-width="2"/><polygon points="210,146 205,136 215,136" fill="#71717a"/><line x1="210" y1="212" x2="210" y2="234" stroke="#71717a" stroke-width="2"/><polygon points="210,240 205,230 215,230" fill="#71717a"/></svg><figcaption><b>閾値スイープ</b>は <b>① モデルを十分低い閾値で 1 回だけ推論</b>して全候補（スコア付き）を集め、<b>② スコア閾値 <code>t</code> を 0.05〜0.95 で掃引</b>して各 <code>t</code> で TP/FP/FN → P/R/F1 を計算し、<b>③ F1 が最大になる <code>t</code> を推奨閾値</b>とする 3 ステップです。<b>推論は最初の 1 回だけ</b>で、あとは後処理の閾値フィルタを繰り返すだけなので速く回せます。</figcaption></figure>

OWLv2 のスイープ結果が下表です。閾値が低い 0.05 では、低スコアの誤検出（FP=7）が混じって precision が 0.36 まで落ちますが、0.10 で 0.80 まで回復し、0.20〜0.70 では **P=R=F1=1.0** の plateau（4物体を過不足なく検出）に達します。さらに 0.75 を超えると、本物の検出まで切り捨ててしまい、recall が 0.75→0.25 と崩れます。結果として **F1 最大は t=0.70 で P=R=F1=1.0** です。「低すぎると過検出で precision が落ち、高すぎると取りこぼしで recall が落ちる」という山なりの構造が、はっきりと読み取れます。

<figure class="lec-fig"><svg viewBox="0 0 620 310" role="img" aria-label="しきい値を上げるとprecisionは上がりrecallは下がる。F1は山なりで最大点を推奨閾値とする" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><line x1="110" y1="30" x2="138" y2="30" stroke="#2563eb" stroke-width="3"/><text x="143" y="34" text-anchor="start" font-size="11.5" fill="#2563eb">P 適合率</text><line x1="248" y1="30" x2="276" y2="30" stroke="#16a34a" stroke-width="3"/><text x="281" y="34" text-anchor="start" font-size="11.5" fill="#16a34a">R 再現率</text><line x1="386" y1="30" x2="414" y2="30" stroke="#ea580c" stroke-width="3"/><text x="419" y="34" text-anchor="start" font-size="11.5" fill="#ea580c">F1</text><line x1="70" y1="46" x2="70" y2="250" stroke="#71717a" stroke-width="1.8"/><line x1="70" y1="250" x2="566" y2="250" stroke="#71717a" stroke-width="1.8"/><text x="60" y="56" text-anchor="end" font-size="11" fill="#71717a">1.0</text><text x="60" y="265" text-anchor="end" font-size="11" fill="#71717a">0</text><polyline points="70,178 112,92 160,50 544,50" fill="none" stroke="#2563eb" stroke-width="2.5"/><polyline points="70,50 446,50 494,100 524,200" fill="none" stroke="#16a34a" stroke-width="2.5"/><polyline points="70,144 112,74 160,50 460,50 500,78 526,170" fill="none" stroke="#ea580c" stroke-width="3"/><line x1="424" y1="50" x2="424" y2="250" stroke="#c2410c" stroke-width="1.5" stroke-dasharray="5 4"/><circle cx="424" cy="50" r="4" fill="#c2410c"/><text x="424" y="270" text-anchor="middle" font-size="11.5" font-weight="700" fill="#c2410c">↑ F1 最大 = 推奨閾値</text><text x="120" y="212" text-anchor="middle" font-size="11.5" fill="#dc2626">過検出 (FP多)</text><text x="506" y="150" text-anchor="middle" font-size="11.5" fill="#dc2626">取りこぼし (FN多)</text><text x="318" y="292" text-anchor="middle" font-size="13" fill="#3f3f46">しきい値 t →</text><text x="40" y="150" text-anchor="middle" font-size="12" fill="#3f3f46" transform="rotate(-90 40 150)">P / R / F1</text></svg><figcaption>スコア閾値 <code>t</code> を上げると、誤検出が減って <b>precision（適合率）は上がる</b>一方、本物の検出も切り捨てて <b>recall（再現率）は下がります</b>。両者の調和平均 <b>F1</b> は<b>山なり</b>になり、その<b>最大点を推奨閾値</b>とします。<b>低すぎる</b>と過検出で precision が落ち、<b>高すぎる</b>と取りこぼしで recall が落ちます。</figcaption></figure>

| しきい値 t | TP | FP | FN | precision | recall | F1 |
| --- | --- | --- | --- | --- | --- | --- |
| 0.05 | 4 | 7 | 0 | 0.36 | 1.00 | 0.53 |
| 0.10 | 4 | 1 | 0 | 0.80 | 1.00 | 0.89 |
| 0.20〜0.70 | 4 | 0 | 0 | **1.00** | **1.00** | **1.00** |
| 0.75 | 3 | 0 | 1 | 1.00 | 0.75 | 0.86 |
| 0.80 | 1 | 0 | 3 | 1.00 | 0.25 | 0.40 |

Grounding DINO でも同じスイープをかけると、**過検出の激しさ**がさらにくっきり現れます。t=0.05 では FP が **60 件**にも達して precision はわずか 0.06、t=0.20 で 0.50 まで上がり、0.30〜0.75 で plateau 1.0、そして F1 最大は t=0.75 です。第5節で見た「緩いと断片語まで拾う」性質が、低閾値域の大量 FP として定量化されているわけです。`03_sweep_owlv2.png` / `03_sweep_gdino.png` は P/R/F1 の3曲線と F1 最大点の縦線を、`03_pr_curve.png` は両モデルの **PR 曲線**（recall を横軸、precision を縦軸）を重ねた図です。合成シーンは検出が容易なので曲線はきれいに角張りますが、**実写を `data/` に置くと、precision と recall がなだらかにトレードオフする現実的な曲線**になります。この「F1 最大点を推奨閾値とする」手続きこそが、本章の完成物そのものです。

## 9. Cluster-CLIP baselines との対応（最終章への布石）

本講座の最終章（第40・41回）で扱う Cluster-CLIP パイプラインでは、**OWLv2 を「テキスト指定で領域を切り出すベースライン検出器」**として使います。本章の `detect_owl`（候補ラベル→box/score/label）が、まさにそのベースライン実装の最小形です。OVD は「CLIP の共有埋め込み空間（第16回）を、画像全体ではなく**領域単位**で使う」技術なので、第16回の「埋め込み→正規化→コサイン」という骨格が、そのまま「領域埋め込み→テキスト埋め込みとの照合」へ地続きに繋がっています。

実務でも OVD は、「**ラベル付きデータが無い／少ないカテゴリを、まずテキストだけで検出してみる**」起点として強力です。OWLv2 や Grounding DINO で粗く検出 → 良さそうな検出を擬似ラベルとして集める → 軽量な閉語彙検出器（第18回）へ蒸留する、という流れは現場でよく使われます。本章で「候補ラベルを変えるだけで検出対象を差し替えられる」「閾値スイープで品質を定量化できる」を身につけておけば、この応用にもそのまま入っていけます。

## 10. transformers v5 の注意点（古いコードが動かない理由）

ネット上の OWL-ViT / Grounding DINO チュートリアルは 4.x 時代のものが多く、そのまま写すと動かないことがあります。そこで、本章で踏んだ **transformers 5.x の作法**を3つまとめておきます。第一に、後処理メソッドが **`post_process_grounded_object_detection` に統一**されました。OWL 系の古い記事にある `post_process_object_detection` は、v5 の OWL では使いません（OWL-ViT/OWLv2 のプロセッサが公開するのは `post_process_grounded_object_detection` だけです）。引数も `threshold` / `target_sizes` / `text_labels`（OWL）、`input_ids` / `threshold` / `text_threshold`（GDINO）と整理されています。

```python
# 4.x（古い記事に多い・v5 では動かない/非推奨）
# results = processor.post_process_object_detection(outputs, target_sizes=...)

# 5.x（本章の正準）
results = processor.post_process_grounded_object_detection(
    outputs=outputs, threshold=0.1,
    target_sizes=torch.tensor([image.size[::-1]]),  # (H, W)！
    text_labels=[labels])                            # OWL は候補ラベルを渡せる
```

第二に、画像の前処理は **`AutoImageProcessor`（fast 実装のみ）** に一本化され、torchvision が事実上必須になりました（`dl` グループで入っているので問題ありません）。また Grounding DINO のバックボーンは **`timm`** に依存するため、`hf` グループに `timm` を含めています（未導入だと `from_pretrained` でバックボーンの読み込みに失敗します）。第三に、**モデルキャッシュ**は初回の `from_pretrained` で `~/.cache/huggingface`（環境変数 `HF_HOME`）に保存され、次回以降は即座に起動します。ただし Docker では、このディレクトリをボリュームマウントしないと、コンテナを作り直すたびに再 DL が走ってしまいます。完全オフラインで回すなら `HF_HUB_OFFLINE=1` を設定してください。

## 11. このモジュールの構成（スクリプト一覧）

各スクリプトは単一責務で、上から順に読むと「動かす → 比較する → 評価する」と理解が積み上がる構成です。いずれも結果を `outputs/20_open_vocabulary_detection/` に図と json で保存し、画面表示には依存しません（matplotlib は Agg）。device 判定・合成シーン生成・モデルロード・検出ラッパ・IoU/マッチング/PRF・描画といった共通処理は `ovd_helpers.py` にまとめてあり、各スクリプトはそれを import します。

| ファイル | 役割（単一責務） |
| --- | --- |
| `ovd_helpers.py` | device 判定・GT 付き合成シーン生成・OWL/GDINO ロードと検出ラッパ・IoU/貪欲マッチング/PRF・図保存。道具箱 |
| `01_owlvit_owlv2.py` | `pipeline` で最短検出 → `AutoModel` 手書き（`post_process_grounded_object_detection`・`target_sizes=(H,W)`）→ OWL-ViT vs OWLv2 比較・幻覚しにくさの確認 |
| `02_grounding_dino.py` | Grounding DINO のキャプション形式・`box_threshold`/`text_threshold` の効き方・過検出/厳しめの比較（重い場合は概念紹介にフォールバック） |
| `03_threshold_sweep_eval.py` | GT 付きシーンで閾値スイープ → P/R/F1 → F1 最大点と PR 曲線（OWLv2 vs Grounding DINO） |
| `mini_project.py` | 章末ミニプロジェクト。OWL-ViT vs OWLv2 を自作 AP@0.5・mAP@[.5:.95]・F1 運用点で厳密比較し、評価レポート（ダッシュボード＋JSON）を生成する総合課題 |
| `use_case.py` | 実践ユースケース（練習用の出発点）。自由文（`a yellow umbrella` 等）を CLI から渡して任意物体を検出・枠表示する「任意物体ファインダー」。実写を `data/` に置けば実用ツールになる |
| `exercises.py` | TODO 形式の演習8問（IoU/座標変換/マッチング/PRF/キャプション整形/IoU行列/NMS/AP補間）。自己採点ランナー `grade()` 付き |
| `exercises_solutions.py` | 演習の模範解答（全 PASS）。採点ロジックは `exercises.grade()` を再利用（重複なし） |

`ovd_helpers.py` だけは「読み物」ではなく「再利用する道具」です。とくに `detect_owl` / `detect_gdino`（検出して xyxy 絶対座標を返す）、`build_scene`（GT 付き4物体シーン）、`greedy_match`（クラス込み TP/FP/FN）の3つが、3スクリプトすべての土台になっています。`ovd_helpers.py` を単体実行すると、道具箱のスモークテスト（合成シーン描画＋IoU/PRF の動作確認、**モデル DL 不要**）になります。まずこれを動かしてから 01 へ進むと、各スクリプトが何を import しているかが腑に落ちるはずです。

## 🛠 章末ミニプロジェクト — 2検出器の厳密比較レポート

`mini_project.py` は、この章の学び（OWL でのテキスト検出・`target_sizes=(H,W)` の後処理・IoU/貪欲マッチング・P/R/F1 の閾値スイープ）を1本に統合した総合課題です。01 が OWL-ViT と OWLv2 を「スコアの高さ」で**目視**比較しただけだったのに対し、ここでは**自作の検出メトリクスで定量的に比較**し、評価レポート（4枚パネルのダッシュボード＋JSON）を `outputs/20_open_vocabulary_detection/` へ出力します。マスターすべき要素は3つです。

- **(1) 2検出器の厳密比較**: OWL-ViT と OWLv2 を AP@0.5・mAP@[.5:.95]・F1 最大運用点で並べる。AP@0.5 は**両者そろって 1.0 に飽和**して見分けがつかない一方、mAP@[.5:.95] は **OWL-ViT 0.93 < OWLv2 1.00** と差が出ます。
- **(2) COCO 流 mAP の自作**: 第19回の自作 mAP を実モデルへ適用。スコア降順マッチング→**全点補間**で AP@0.5 を出し、IoU を 0.50:0.05:0.95 で動かして mAP@[.5:.95] を計算。OWL-ViT は **IoU=0.95 で AP が 0.33 まで落ちる**（＝箱が緩く局在が甘い）ことが可視化され、「**AP@0.5 では隠れる局在精度の差**」を体感できます。
- **(3) 運用点はモデルで違う**: F1 最大のしきい値が **OWL-ViT t≈0.25 / OWLv2 t≈0.70** と大きく異なります。スコアのスケールがモデルで別物なので、片方の閾値を流用すると取りこぼし／過検出になる――「**閾値はモデルとセットで F1 最大点で決める**」を数値で示し、併せて『シーンに無いラベル』の誤検出（幻覚）を監査します。

<figure class="lec-fig"><svg viewBox="0 0 660 330" role="img" aria-label="ミニプロジェクトは2検出器で検出し貪欲マッチング、AP/mAP算出、F1最大運用点の決定を経てダッシュボードとJSONを出力する流れ" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><text x="330" y="32" text-anchor="middle" font-size="14" font-weight="700" fill="#18181b">ミニプロジェクト — 2 検出器を定量比較してレポート化</text><rect x="24" y="64" width="180" height="64" rx="8" fill="#fff7ed" stroke="#ea580c" stroke-width="2"/><rect x="240" y="64" width="180" height="64" rx="8" fill="#fff7ed" stroke="#ea580c" stroke-width="2"/><rect x="456" y="64" width="180" height="64" rx="8" fill="#fff7ed" stroke="#ea580c" stroke-width="2"/><rect x="456" y="214" width="180" height="64" rx="8" fill="#fff7ed" stroke="#ea580c" stroke-width="2"/><rect x="240" y="214" width="180" height="64" rx="8" fill="#eff6ff" stroke="#2563eb" stroke-width="2"/><text x="114" y="92" text-anchor="middle" font-size="14.5" font-weight="700" fill="#c2410c">① 2 検出器で検出</text><text x="114" y="112" text-anchor="middle" font-size="11" fill="#71717a">OWL-ViT / OWLv2</text><text x="330" y="92" text-anchor="middle" font-size="14.5" font-weight="700" fill="#c2410c">② 貪欲マッチング</text><text x="330" y="112" text-anchor="middle" font-size="11" fill="#71717a">IoU 0.5・クラス込み</text><text x="546" y="92" text-anchor="middle" font-size="14.5" font-weight="700" fill="#c2410c">③ AP / mAP 算出</text><text x="546" y="112" text-anchor="middle" font-size="11" fill="#71717a">AP@0.5・mAP@[.5:.95]</text><text x="546" y="242" text-anchor="middle" font-size="14.5" font-weight="700" fill="#c2410c">④ F1 最大運用点</text><text x="546" y="262" text-anchor="middle" font-size="11" fill="#71717a">閾値スイープで決定</text><text x="330" y="242" text-anchor="middle" font-size="14.5" font-weight="700" fill="#1d4ed8">⑤ レポート出力</text><text x="330" y="262" text-anchor="middle" font-size="11" fill="#71717a">4 パネル＋JSON</text><line x1="204" y1="96" x2="234" y2="96" stroke="#71717a" stroke-width="2"/><polygon points="240,96 230,91 230,101" fill="#71717a"/><line x1="420" y1="96" x2="450" y2="96" stroke="#71717a" stroke-width="2"/><polygon points="456,96 446,91 446,101" fill="#71717a"/><line x1="546" y1="128" x2="546" y2="208" stroke="#71717a" stroke-width="2"/><polygon points="546,214 541,204 551,204" fill="#71717a"/><line x1="454" y1="246" x2="426" y2="246" stroke="#71717a" stroke-width="2"/><polygon points="420,246 430,241 430,251" fill="#71717a"/></svg><figcaption><b>章末ミニプロジェクト</b>は、<b>① 2 検出器（OWL-ViT / OWLv2）で検出</b> → <b>② 貪欲マッチング</b>（IoU≥0.5・クラス込み）→ <b>③ AP@0.5 / mAP@[.5:.95] を算出</b> → <b>④ F1 最大運用点</b>を閾値スイープで決定 → <b>⑤ ダッシュボード（4 パネル）＋JSON を出力</b>、の順に流れます。01 がスコアの高さの<b>目視</b>比較だったのに対し、ここでは<b>自作メトリクスで定量比較</b>します。</figcaption></figure>

`mini_project_dashboard.png` の4パネル（最良モデルの検出・AP/mAP 棒・AP vs IoU 曲線・F1 vs 閾値曲線）と `mini_project_report.json` を、上の3点と照らし合わせてください。なお、モデルのロード/推論が失敗する環境では、箱の締まりとスコアを変えた**強/弱の合成検出**にフォールバックし、必ず `exit 0` になります。

## ✅ 到達チェックリスト

この章を「身につけた」と言える基準です。手を動かして全部 ✅ にできるか確認してください。

- [ ] オープン語彙検出が「領域をテキスト埋め込みとの類似度で名付ける（第16回 CLIP の領域版）」技術だと自分の言葉で説明できる。
- [ ] `AutoProcessor` + `AutoModelForZeroShotObjectDetection` で OWL をロードし、`post_process_grounded_object_detection` で box/score/label を取り出せる。
- [ ] `target_sizes` を **`(H,W)`** 順（`image.size[::-1]`）で渡す理由を説明でき、間違えると box がどう歪むか言える。
- [ ] OWL（候補ラベルの**リスト**）と Grounding DINO（**小文字＋ピリオド区切りキャプション**）の入力差と、GDINO の **box/text 2閾値**を使い分けられる。
- [ ] IoU → 貪欲マッチング（クラス込み・1GT に1検出）→ TP/FP/FN → P/R/F1 を **numpy で自作**できる（演習 ex1/ex3/ex4）。
- [ ] スコア閾値スイープで **F1 最大点を推奨閾値**として選べる（03）。
- [ ] **AP（全点補間）と mAP@[.5:.95]** を自作でき、AP@0.5 と mAP@[.5:.95] が示す情報の違い（順位品質 vs 局在精度）を説明できる（演習 ex8・mini_project）。
- [ ] NMS が「過検出した重複ボックスを1つに畳む」ことを実装できる（演習 ex7）。
- [ ] 演習 8問すべてが `ALL PASS`、`mini_project.py`・01〜03 がいずれも `exit 0`。

## ✍️ 演習問題

演習は `exercises.py` に TODO 形式で入っています。各 TODO を実装し `uv run python lectures/20_open_vocabulary_detection/exercises.py` を実行すると自己採点できます（`exercises_solutions.py` が解答）。

1. 2つの box `[x1,y1,x2,y2]` の IoU（交差面積 ÷ 和集合面積）を返す（`ex1_iou` の TODO）。完全一致なら 1.0、無重なりなら 0.0。
2. 正規化 cxcywh `[cx,cy,w,h]`（0〜1）を絶対 xyxy へ変換する（`ex2_cxcywh_norm_to_xyxy` の TODO）。x 方向は width、y 方向は height を掛ける。
3. クラス込みの貪欲マッチング（スコア降順・同ラベルかつ未マッチで IoU≥閾値の GT に対応づけ）で TP/FP/FN を数える（`ex3_greedy_match` の TODO）。
4. TP/FP/FN から precision・recall・F1 を計算する（`ex4_prf` の TODO）。分母 0 のときは 0.0。
5. 候補ラベルのリストを Grounding DINO 用キャプション（小文字化＋トリム＋ピリオド区切り）へ整形する（`ex5_labels_to_caption` の TODO）。
6. N 個と M 個の box 群のペアワイズ IoU 行列 `(N, M)` を返す（`ex6_iou_matrix` の TODO）。`M[i,j]` は `boxes_a[i]` と `boxes_b[j]` の IoU。
7. 単一クラスの NMS（スコア降順に採用し IoU≥閾値の重複を抑制）で『残す予測のインデックス』を返す（`ex7_nms` の TODO）。
8. マッチ済み検出（スコア・TP フラグ）から AP（PR 曲線の全点補間）を計算する（`ex8_average_precision` の TODO）。第19回 mAP 自作の核。

## ❓ よくある落とし穴・FAQ・デバッグ

§13 の「症状→原因→対処」表と併せて、考え方・設計の疑問に答えます。

**Q. OWL と Grounding DINO、結局どっちを使えばいい？**
A. **列挙できる固定ラベル**なら OWL（候補配列を渡すだけ・ラベルが綺麗に付く）。**自由な言語表現・参照表現や下流の SAM 連携**なら Grounding DINO。CPU で軽く試すなら OWL-ViT、精度重視なら OWLv2。迷ったら OWLv2 から。

**Q. `mAP` と書いてあるけど、AP@0.5 と mAP@[.5:.95] のどっち？**
A. 文脈次第なので**必ず明記する**。PASCAL 流は mAP@0.5、COCO の主指標は mAP@[.5:.95]（IoU 0.50:0.05:0.95 の平均）。本章 mini_project は両方出すので、**AP@0.5 が同じでも mAP@[.5:.95] が違う**＝箱の締まり（局在精度）の差、という読み方を確認できます。

**Q. AP@0.5 が 1.0 なのに mAP@[.5:.95] が 1.0 未満。なぜ？**
A. AP@0.5 は「IoU≥0.5 で当たっていれば満点」。箱が少しズレていても 0.5 は超えるので満点になりがちです。mAP@[.5:.95] は IoU=0.90/0.95 のような**厳しい基準**も平均に含むため、箱が緩い検出器は高 IoU 側で AP が落ち、平均が下がります。これが「局在精度」を測るということです（mini_project の OWL-ViT がまさにこの挙動）。

**Q. しきい値はいくつにすればいい？**
A. **モデル・候補ラベル・画像ごとに変わる**ので固定値の正解はありません。03 と mini_project のように、低閾値で1回推論→後処理でスコア閾値を掃引→F1 最大点を選ぶ、が定石。OWL-ViT は ~0.1–0.25、OWLv2 は ~0.3–0.7 が目安ですが、**必ず自分のデータで F1 を測って決める**こと。

**Q. 検出が0件 / box が出ない。どう切り分ける？**
A. ①まず `ovd_helpers.py` を単体実行（モデル DL 不要のスモークテスト）→ IoU/PRF と合成シーンが正常か。②`01` の pipeline 出力で「4物体が当たり・無いラベルが0件」を確認。③box が歪むなら `target_sizes=(H,W)` を疑う。④検出が0件なら閾値を 0.01 まで下げて「そもそも候補が出ているか」を見る（出ていれば閾値問題、出ていなければ入力／ラベル／書式問題）。

## 🚀 発展トピック・参考

- **Grounded-SAM（第23回）**: Grounding DINO の box を SAM に `input_boxes` として渡し、テキスト指定領域をセグメンテーション。本章の `detect_gdino` がその前段そのもの。
- **mAP@[.5:.95] と pycocotools**: 自作 mAP の検算は `COCOeval(iouType='bbox')` が正準（第19回）。AP_S/M/L（面積別）や AR@{1,10,100} まで出せる。自作値と突き合わせる癖をつける。
- **NMS と過検出**: Grounding DINO の緩い設定で出る重複／断片は、演習 ex7 の NMS や `torchvision.ops.batched_nms`（クラス別 NMS）で後処理できる。OWL/GDINO は内部で NMS 済みなので**二重抑制に注意**。
- **プロンプト設計**: 候補ラベルの言い回し（`"red circle"` / `"a red circle"` / `"a photo of a red circle"`）で低確信ボックスの数や検出が変わることがある。CLIP のプロンプト設計（第16回）と同じ発想で、検出対象ごとに最適な表現を探る。
- **OWLv2 の self-training**: OWLv2 は OWL-ViT 自身の擬似ラベルで再学習して確信度・精度を底上げした後継。実務では「軽い OWL-ViT で試作 → OWLv2 で本番」。
- **蒸留・最終章への接続**: OVD で粗く検出 → 良い検出を擬似ラベル化 → 軽量な閉語彙検出器（第18回）へ蒸留、は現場頻出。最終章（第40・41回）の Cluster-CLIP は OWLv2 をベースライン検出器に使う。
- 参考: HuggingFace `transformers` の zero-shot-object-detection ドキュメント、OWL-ViT／OWLv2／Grounding DINO の各モデルカード、COCO 評価（pycocotools）。

## 💡 実践ユースケース集

ここまでの「テキストで検出 → 閾値で品質を制御 → P/R/F1 で評価」という流れを、現実の小さなアプリへ落とすと何が作れるかを、3つ挙げます。1つ目は同梱の `use_case.py`（動く出発点）で、残りの2つは作り方の要点だけを示します。いずれも、本章の `detect_owl` / `detect_gdino`（候補ラベル/キャプション → xyxy 絶対座標 box）が中核です。

### ① 任意物体ファインダー（同梱 `use_case.py` — まずこれを動かす）

**何に使うか**: 「この画像に "黄色い傘" は写っている？」のように、その場で決めた自由文で物を探し、枠表示する最小ツールです。固定クラス検出器では学習し直さないと答えられない問いにも、`candidate_labels` を差し替えるだけで答えられます。防犯カメラの不審物探しや、写真整理での「写っている物探し」の出発点になります。評価レポート寄りの `mini_project.py`（ベンチ寄り）とは別物で、こちらは CLI から自由文を渡して使う**現実の小ツール**として完結しています。

```bash
# 既定のデモクエリ（合成シーンに有る語＋無い語）で動かす
uv run python lectures/20_open_vocabulary_detection/use_case.py
# 自分の探したい物を自由文で渡す（複数可・スペースを含むフレーズは引用符で）
uv run python lectures/20_open_vocabulary_detection/use_case.py "a red circle" "a yellow umbrella"
# 検出器を選ぶ: 速さ重視=owlvit / 精度重視=owlv2(既定) / 自由文フレーズ向き=gdino
OVD_MODEL=owlvit uv run python lectures/20_open_vocabulary_detection/use_case.py
# しきい値を上書き（小さいほど拾う・過検出も増える）
OVD_THRESHOLD=0.15 uv run python lectures/20_open_vocabulary_detection/use_case.py "a person"
```

**`data/` の置き方（実用化のキモ）**: `data/20_open_vocabulary_detection/` に `.png/.jpg` を1枚以上置くと、その先頭画像が検出対象になります（合成シーンは使われなくなります）。合成シーンは色・形の語にはよく反応しますが、`umbrella`/`person` のような実世界の語には当然ヒットしません（=「見つからない」を体験するためのものです）。逆に、**実写を置けば、自由文で本物の物体を探す実用ツールとして動きます**。結果は `outputs/20_open_vocabulary_detection/use_case_finder.png`（注釈画像）と `use_case_finder.json`（クエリ別の発見数・最高スコア・box）に保存されます。

**拡張アイデア（練習）**: 画像フォルダを総当りして「指定物が写っている画像」を探す画像検索へ／検出 box を SAM に渡して領域マスク化（Grounded-SAM・第23回）／同ラベルの box 数を数えて在庫・人数カウント／動画フレームに回し、指定物が現れたフレームだけ通知するアラート化。**注意**: しきい値には固定の正解がありません（§8）。実運用では自分のデータで F1 最大点を測って決め、合成シーンの結果をそのまま実写に当てはめないようにしてください。

### ② 在庫検品・棚卸しチェッカー（OWL で「列挙ラベルの有無・点数」）

**何に使うか**: 棚やコンテナの写真に対して、「消火器・ヘルメット・三角コーン…」のように**列挙できる固定ラベル**の有無や点数を自動チェックする検品ツールです。語彙が列挙できる用途では OWL-ViT/OWLv2 が素直で、結果にも「どのラベルか」が綺麗に付きます。**作り方の要点**: `detect_owl` に検品リストを candidate_labels として渡し、ラベルごとに box 数を数える → 期待数と突き合わせて不足/過剰を判定する、という流れです。スコア閾値は §8 のスイープで品目ごとに較正します。**注意**: 同一物体が複数の box に割れる過検出は計数を狂わせるため、必要なら NMS（演習 ex7・クラス別 `batched_nms`）で重複を畳んでから数えてください（OWL は内部で NMS 済みなので、二重抑制には注意します）。

### ③ 参照表現セグメンテーションの前段（Grounding DINO → SAM）

**何に使うか**: 「左の人が持っているカバン」のような**自然言語の参照表現**で領域を切り出したい場面です。早期融合の Grounding DINO は修飾付き・関係表現に強く、その box を SAM に `input_boxes` として渡せば、任意領域のマスクが得られます（**Grounded-SAM**、第23回で本格的に扱います）。**作り方の要点**: `detect_gdino` に「小文字＋ピリオド区切り」のキャプションを渡して box を得（§5）、その box を SAM のプロンプトにする、という2段構成です。**注意**: GDINO はキャプション書式を崩すと不安定になり、`box_threshold`/`text_threshold` の2つを調整しないと断片語への過検出が起きます。`use_case.py` を `OVD_MODEL=gdino` で動かすと、この前段（テキスト → box）だけを単体で体感できます。

## 12. 動かし方

このモジュールは `dl`（torch/torchvision）・`hf`（transformers/timm ほか）グループに依存します。CPU だけで完走し、初回のみ OWL-ViT・OWLv2・Grounding DINO の重みを HuggingFace からダウンロードします（以降はキャッシュから起動）。プロジェクトルートで以下を順に実行してください。

```bash
# 依存グループをインストール（初回のみ）
uv sync --group dl --group hf

# まず道具箱のスモークテスト（モデル DL 不要・合成シーン描画＋IoU/PRF 確認）
uv run python lectures/20_open_vocabulary_detection/ovd_helpers.py

# 各スクリプトを実行（結果は outputs/20_open_vocabulary_detection/ に保存される）
uv run python lectures/20_open_vocabulary_detection/01_owlvit_owlv2.py
uv run python lectures/20_open_vocabulary_detection/02_grounding_dino.py
uv run python lectures/20_open_vocabulary_detection/03_threshold_sweep_eval.py

# 章末ミニプロジェクト（OWL-ViT vs OWLv2 の評価レポート。ダッシュボード＋JSON を保存）
uv run python lectures/20_open_vocabulary_detection/mini_project.py

# 実践ユースケース: 任意物体ファインダー（自由文で物を探して枠表示する小ツール）
uv run python lectures/20_open_vocabulary_detection/use_case.py
uv run python lectures/20_open_vocabulary_detection/use_case.py "a red circle" "a yellow umbrella"
OVD_MODEL=owlvit uv run python lectures/20_open_vocabulary_detection/use_case.py  # 速さ重視

# 演習: まずは TODO を自分で埋める（最初は全部 FAIL だが exit 0）
uv run python lectures/20_open_vocabulary_detection/exercises.py
# どうしても分からない時だけ、模範解答（全 PASS）を見る
uv run python lectures/20_open_vocabulary_detection/exercises_solutions.py
# もしくは同じ採点ロジックで模範解答を採点（採点を共有して確認）
SHOW_SOLUTION=1 uv run python lectures/20_open_vocabulary_detection/exercises.py

# （任意）実写で試す: data/20_open_vocabulary_detection/ に .png/.jpg を置くと自動で使われる
#  → 01/02 は実写を検出。03 の評価は GT が無いため合成シーンに自動で切り替わる。
```

実行後は、`outputs/20_open_vocabulary_detection/` の図を解説と照らし合わせてください。とくに `02_gdino_loose.png`（過検出だらけ）と `02_gdino_strict.png`（4つに収束）、`03_sweep_owlv2.png`（P/R/F1 の山）と `03_pr_curve.png`（PR 曲線）を見ると、本章の2大テーマ（**閾値で過検出/取りこぼしが決まる・F1 最大点で閾値を選ぶ**）が視覚的に腑に落ちます。なお、図中の文字は CJK フォントの豆腐（□）を避けるため ASCII にしてあります。また、合成画像なのに色がおかしい場合は、cv2 の BGR を RGB へ変換し忘れていないかを確認してください（本章は `build_scene` で変換済みです）。

## 13. よくあるエラーと対処（チェックリスト）

最後に、本章でつまずきやすい点を「症状 → 原因 → 対処」でまとめます。OVD・transformers v5 特有の罠が多いので、詰まったらまずここを見てください。

| 症状 | ほぼ確実な原因 | 対処 |
| --- | --- | --- |
| box が横長/縦長に歪む | `target_sizes` を `(W,H)` で渡した | `image.size[::-1]` で **`(H,W)`** にして渡す |
| 可視化で box がバラバラ/画面外 | 生の `pred_boxes`(cxcywh 正規化)を直接描いた | `post_process_grounded_object_detection` を必ず通す |
| `post_process_object_detection` が無い | v5 の OWL は `grounded` 版に統一 | `post_process_grounded_object_detection` を使う |
| OWL で候補ラベルが通らない/形がおかしい | `text=labels`（ネスト忘れ） | `text=[labels]` と一段ネストして渡す |
| Grounding DINO が何も検出しない/断片を拾う | キャプション書式が不正（大文字・区切り無し） | 小文字＋ピリオド区切り `"a cat. a dog."` にする |
| Grounding DINO の後処理でエラー | `input_ids` を渡していない | `post_process_...(outputs, input_ids=inputs["input_ids"], ...)` |
| Grounding DINO が過検出/未検出 | `box_threshold`/`text_threshold` が不適切 | 2つの閾値を上げて過検出を抑える（第5節） |
| `timm` 関連でロード失敗 | Grounding DINO は timm 依存 | `hf` グループ（timm を含む）を入れる |
| CPU で推論が極端に遅い | `float16`/`half` を CPU で使っている | CPU は `float32`。`inference_mode()` を付ける |
| 毎回モデルを再DLする（Docker） | キャッシュをマウントしていない | `~/.cache/huggingface`（`HF_HOME`）をボリューム化 |

この表の上3つ（`(H,W)` 順・`post_process` を通す・`grounded` 版）が OWL+v5 の「あるある」で、真ん中3つが Grounding DINO の書式まわりです。症状を見たら原因を即座に言い当てられるよう、頭に入れておきましょう。

## 14. まとめ

本章では、**オープン語彙物体検出**が「検出領域を固定ノードで分類する代わりに、テキスト埋め込みとの類似度で名付ける」技術であること（第16回 CLIP の領域版）から出発し、`pipeline` での最短検出、`AutoModel` 手書きでの `post_process_grounded_object_detection` と **`target_sizes=(H,W)`** の座標変換、**OWL-ViT vs OWLv2**（スコアの出方と幻覚しにくさ）、**Grounding DINO** のキャプション形式と `box_threshold`/`text_threshold` による過検出制御、そして **IoU→貪欲マッチング→P/R/F1 の閾値スイープ**で F1 最大点を推奨閾値とする定量評価までを、すべて GT 付きの合成シーンで「自分で再現し、数字で確認できる」レベルで扱いました。通底する勘所は「**候補ラベル/キャプションは推論時に自由に決められる**」「**閾値が過検出と取りこぼしのトレードオフを決め、F1 最大点で選ぶ**」の2つです。

ここで身につけた「テキストで検出 → 閾値で品質を制御 → P/R/F1 で評価」という骨格は、次の第23回（テキストプロンプトセグメンテーション・Grounded-SAM）で「Grounding DINO の box を SAM に渡す」流れへ、また最終章（第40・41回）の Cluster-CLIP パイプラインで「OWLv2 ベースライン検出器」としてそのまま繋がります。まずは演習を全問 PASS させ、`02` の「緩い閾値で 19 件・厳しい閾値で 4 件」と `03` の「F1 最大が OWLv2 で t=0.70・GDINO で t=0.75」を自分の言葉で説明できるようにしてから、次へ進んでください。

---

> 本教材で参照・検証したライブラリとバージョン（2026-06 時点の安定版で動作確認）:
> Python 3.12 ／ torch 2.12.0+cpu ／ torchvision 0.27.0+cpu ／ transformers 5.11.0 ／ huggingface-hub 1.18.0 ／ timm 1.0.27 ／ safetensors 0.8.0 ／ numpy 2.4.6 ／ Pillow 12.2.0 ／ matplotlib 3.10.9 ／ opencv-python-headless 4.13.0（合成シーンの描画）／ pycocotools 2.0.11（第19回の自作 mAP・COCOeval 突き合わせで使用）
> 使用モデル: `google/owlvit-base-patch32`（OWL-ViT）／ `google/owlv2-base-patch16-ensemble`（OWLv2）／ `IDEA-Research/grounding-dino-tiny`（Grounding DINO）。いずれも初回のみ HuggingFace から重みを取得しキャッシュします。