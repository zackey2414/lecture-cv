# 第23回 テキストプロンプト/参照セグメンテーション — CLIPSeg と Grounded-SAM

> トラック: **セグメンテーション** ／ レベル: **中級** ／ 依存グループ: `dl`（torch/torchvision）・`hf`（transformers/timm ほか）。CPU だけで完走します（初回のみモデル重みを HuggingFace からダウンロード）。評価指標（IoU/Dice）は numpy で自前実装するので `metrics` グループは不要です。

## 🎯 この章のゴール

第21回ではセマンティックセグメンテーション（画像の全画素を**固定クラス**へ振り分ける）を、第22回ではインスタンス分割と SAM（点やボックスの**プロンプト**で対象を切り出す）を学びました。本章のテーマはその先――**「文（テキスト）で対象を指定して、その領域だけをマスク化する」参照セグメンテーション（referring segmentation）**です。固定クラスにも、点・箱という幾何プロンプトにも縛られず、`"a red circle"` のような自然言語で「どこを塗るか」を指示できるのが核心です。第16回（CLIP）で身につけた「画像とテキストを同じ空間で結びつける」発想が、ついに**画素単位のマスク**へと降りてきます。

この章では2つのアプローチを手で動かして対比します。ひとつは **CLIPSeg**（`CIDAS/clipseg-rd64-refined`）――CLIP の上に軽量なデコーダを載せ、**1モデルで「文 → 確率マップ」を直接出す**手法です。出力は `outputs.logits` で、`torch.sigmoid` で 0〜1 の確率にし、**しきい値で2値マスク化**します。もうひとつは **Grounded-SAM**――**Grounding DINO**（オープン語彙の box 検出, 第20回の延長）で文が指す物体の箱を出し、その箱を **SAM**（第22回）の `input_boxes` に渡して画素マスクへ切り出す、**役割の違う2モデルを連結した2段パイプライン**です。

到達点は4つです。第一に、CLIPSeg で `logits → sigmoid → 閾値` の後処理を自分で書き、**参照テキストごとに確率ヒートマップとマスク**を得られること。第二に、予測マスクと正解マスク(GT)の **IoU・Dice** を numpy で計算し、**sigmoid しきい値をスイープして IoU 最大点を探す**こと（既定の 0.5 が最適とは限らないと体感する）。第三に、Grounding DINO → SAM の**2段構成を組み**、検出 box を SAM でマスク化できること。第四に、**Grounding DINO の box 閾値**が最終マスクの取りこぼし（recall）を支配することを、閾値スイープで数字とともに確かめることです。本章のスクリプトはすべて、ネットもデータセットDLも無しで完走するよう、入力を**合成シーン**（赤い円・青い四角・緑の三角）として描き、各図形の **GT マスクを画素単位で厳密に保持**します。CLIPSeg も Grounding DINO も色＋形のような単純概念には合成画像でも反応するので、教材として意味のある IoU が出ます（実写で試したい人は `data/23_text_prompt_segmentation/` に画像を置けば自動で使われます）。

---

## 1. 参照セグメとは何か — 検出・セマンティック・SAM との位置づけ

ここまでに学んだセグメ／検出を「**何で対象を指定するか**」で並べると、参照セグメの居場所がはっきりします。セマンティックセグメ（第21回）は**学習時に固定したクラス集合**（道路・人・車…）でしか塗れません。インスタンス分割や SAM（第22回）は、SAM なら**点・ボックスという幾何プロンプト**で「ここ」を指しますが、**「赤い方の椅子」「左から2番目の人」のような言語的な指定はできません**。一方、オープン語彙検出（第20回, OWL-ViT/Grounding DINO）は**任意のテキスト**で物体を指せますが、出力は**箱**であって画素マスクではありません。参照セグメは、この最後のピース――**任意のテキストで指定し、出力は画素マスク**――を埋めるタスクです。

なぜこれが嬉しいのか。実務では「画像の中の**特定の対象だけ**を、**学習し直さずに**、**自然言語で**切り出したい」場面が頻出します。たとえば商品画像から「the price tag（値札）」だけを抜く、医用画像から「the tumor region」を粗く当たりを付ける、自動運転ログから「the pedestrian crossing」を塗る、など。固定クラス分類器を毎回作り直すのは非現実的で、点や箱を人手で打つのも面倒です。**文で指せて画素が返る**参照セグメは、この「柔軟さ」と「画素精度」を同時に満たします。

本章で扱う2手法は、この目標への**対照的な2つの道**です。CLIPSeg は「文 → マスク」を**1つのモデルで一気に**解く軽量・直接型。Grounded-SAM は「文 → 箱（検出器）→ マスク（SAM）」と**2段に分けて**、それぞれ得意なモデルに分業させる合成型です。前者は速くて手軽、後者は重いが境界がシャープで差し替えが効く――この**設計上のトレードオフ**を、実際に動かしながら体得するのがこの章の主眼です。まずは評価の土台（合成シーンと IoU/Dice）を固め、次に CLIPSeg から触っていきます。

## 2. 評価の土台 — 合成シーンと IoU/Dice の定義

参照セグメの良し悪しは「**文が指した領域**を、どれだけ**正確に画素で当てたか**」で測ります。そのためには「正解マスク（GT）」が画素単位で分かっている必要があります。本章では `seg_helpers.build_scene()` が、明るい灰色の背景に**赤い円・青い四角・緑の三角**を重ならない位置に描いた1枚のシーンを作り、同時に**各図形だけを別キャンバスに描いて画素>0 を取る**ことで、厳密な GT マスク（bool 配列）を得ています。重ならせないのは、参照セグメが「どの領域か」を一意に指したいタスクだからです（`outputs/23_text_prompt_segmentation/00_scene_and_gt.png` にシーンと GT が並びます）。

マスクどうしの一致は **IoU（Intersection over Union）** と **Dice** で測ります。予測マスク P と正解マスク G について、IoU は**重なり面積を和集合面積で割った値** `|P∩G| / |P∪G|`、Dice は `2|P∩G| / (|P|+|G|)`（F1 スコアと同値）です。どちらも 1.0 が完全一致で、同じ重なりなら **Dice ≥ IoU**（Dice の方が甘く評価する）という関係があります。実装は次のように numpy だけで完結します。両方とも空マスクのときに 0 で割らないよう、`1.0`（完全一致扱い）を返すのが実務上のお約束です。

```python
def mask_iou(pred, gt):                       # |∩| / |∪|
    p, g = pred.astype(bool), gt.astype(bool)
    union = np.logical_or(p, g).sum()
    return 1.0 if union == 0 else float(np.logical_and(p, g).sum()) / float(union)

def mask_dice(pred, gt):                       # 2|∩| / (|P|+|G|)  = F1
    p, g = pred.astype(bool), gt.astype(bool)
    denom = p.sum() + g.sum()
    return 1.0 if denom == 0 else 2.0 * float(np.logical_and(p, g).sum()) / float(denom)
```

この2つの関数が、本章の全評価の土台です。検出の mAP（第19回）が「箱の IoU でマッチングしてから順位を見る」複雑な指標だったのに対し、参照セグメは「**前景マスクどうしの単純な重なり**」を見るだけなので、定義がそのまま実装になります。GT を厳密に持っているからこそ、次節以降で「CLIPSeg のしきい値を変えると IoU がどう動くか」を曲線で追えるのです。

## 3. CLIPSeg — logits → sigmoid → 閾値でマスクにする

`01_clipseg.py` の主役 **CLIPSeg** は、CLIP の画像エンコーダの上に軽量な FiLM 条件付きデコーダを載せたモデルです。テキスト（プロンプト）でデコーダを条件付けることで、**「その文が指す領域」の画素ごとのスコア**を出します。使い方は CLIP とよく似ていて、`CLIPSegProcessor` で**画像と複数プロンプトを同時に前処理**し（長さを揃えるため `padding=True`）、`model(**inputs).logits` で **(プロンプト数, 352, 352) のロジット**を得ます。CLIPSeg の出力は常に内部解像度 352×352 なので、原寸へ補間してから確率化するのが定石です。

ここで CLIP（第16回）との**決定的な違い**を押さえてください。CLIP は画像とテキストを**1本のベクトル**に潰して類似度を測りました。CLIPSeg はそれを**画素のグリッド**に展開し、各画素について「この文に合うか」をロジットで出します。だから後処理は分類と同じ `sigmoid`――各画素を独立に 0〜1 の確率にし、**しきい値で前景/背景に二値化**します（softmax ではない点に注意。画素ごとの独立判定なので sigmoid です）。`seg_helpers.clipseg_probs` がこの流れをまとめており、`F.interpolate` で原寸へ上げてから `torch.sigmoid` を掛けています。

```python
inputs = processor(text=prompts, images=[image]*len(prompts), padding=True, return_tensors="pt")
logits = model(**inputs).logits                       # (P, 352, 352) ロジット
up = F.interpolate(logits.unsqueeze(1), size=(h, w),  # 原寸 (H,W) へ補間
                   mode="bilinear", align_corners=False).squeeze(1)
probs = torch.sigmoid(up)                              # (P, H, W) 0〜1 の確率
masks = probs >= 0.5                                   # しきい値で2値マスク化
```

合成シーンに3つのプロンプト（`a red circle` / `a blue square` / `a green triangle`）を投げた実測が下表です（`01_clipseg_metrics.json`）。しきい値 0.5 でも IoU は 0.88〜0.95 と高く、`01_clipseg_panel.png` のヒートマップを見ると、各プロンプトが対応する図形だけを正しく赤く（高確率に）灯しています。CLIPSeg が「色＋形」という概念を**ゼロショットで画素レベルに**落とせていることが分かります。

| プロンプト | prob 最大 | IoU@0.5 | Dice@0.5 |
| --- | --- | --- | --- |
| a red circle | 0.91 | 0.945 | 0.972 |
| a blue square | 0.98 | 0.942 | 0.970 |
| a green triangle | 0.98 | 0.884 | 0.939 |

さらに `01` は「**シーンに無い概念**」も投げます。`"a yellow star"`（黄色い星はシーンに存在しない）のプロンプトでは **prob 最大が 0.03、0.5 を超える画素は 0** でした。CLIPSeg の sigmoid は画素を独立に評価するので、**該当物が無ければ全画素が低いまま**――しきい値で「無い」と正しく判定できます。これは「候補のどれか1つに無理やり確率を寄せる」softmax 型の分類（第16回）とは対照的で、**参照セグメに sigmoid が向く理由**そのものです。マスクが「それっぽく」出たら、次は「どこで切るか（しきい値）」を真面目に考えます。

## 4. しきい値の選び方 — 既定の 0.5 が最適とは限らない

CLIPSeg の出力は**連続値の確率マップ**なので、最終的な2値マスクは「どのしきい値で切るか」で変わります。前節は便宜上 0.5 を使いましたが、これは本当に最適でしょうか。`03_referring_iou_eval.py` の前半（A パート）は、各プロンプトについて**しきい値を 0.05〜0.95 でスイープし、毎回 IoU を測って曲線を描き**ます（`best_threshold` 関数が IoU 最大点を返します）。下が実測の要約で、図 `03_clipseg_threshold_sweep.png` には3本の山なりカーブと「最適点」「既定 0.5」の縦線が描かれています。

| プロンプト | IoU@0.5 | 最良しきい値 | 最良 IoU |
| --- | --- | --- | --- |
| a red circle | 0.945 | 0.65 | 0.965 |
| a blue square | 0.942 | 0.60 | 0.947 |
| a green triangle | 0.884 | 0.70 | 0.936 |

読み取れることは明快です。**最良しきい値は 0.60〜0.70 にあり、どのプロンプトでも 0.5 ではありません**。曲線は山なりで、しきい値が低すぎると確率の裾野まで拾って**過剰に塗り（マスクが膨らんで IoU 低下）**、高すぎると確信の高い中心しか残らず**塗り残し（マスクが痩せて IoU 低下）**ます。つまり 0.5 は「無難な初期値」ではあっても「最適」ではなく、**対象やプロンプトごとにピークがずれる**のです。green triangle が最も高い 0.70 を要するのは、三角形の鋭い頂点付近で確率がなだらかに落ちるため、低めで切ると背景まで拾ってしまうから――と図から推測できます。

実務的な含意はこうです。GT が手元にある（検証セットがある）なら、本節のように**しきい値をスイープして最適点を選ぶ**のが正攻法です。GT が無い運用時は、0.5 を起点にしつつ「塗りすぎなら上げる／塗り残すなら下げる」と**対象に応じて手で調整**します。CLIPSeg の「確率マップを返す」性質は、この**後段のしきい値で挙動を後から調整できる**柔軟さの裏返しでもあります。次は、まったく別の設計――2つのモデルを連結する Grounded-SAM――に進みます。

## 5. Grounded-SAM — 検出（Grounding DINO）→ セグメ（SAM）の2段構成

`02_grounded_sam.py` が組むのは **Grounded-SAM**、すなわち「**文で箱を出す検出器**」と「**箱を画素マスクに変える SAM**」を直列につないだパイプラインです。段1の **Grounding DINO**（`IDEA-Research/grounding-dino-tiny`）は第20回で触れたオープン語彙検出器で、`"a red circle. a blue square. a green triangle."` のように**小文字＋各物体をピリオド区切り**にしたテキストを受け、文が指す物体の bounding box を返します（この「小文字＋ピリオド区切り」は Grounding DINO の作法で、守らないと検出が安定しません）。段2の **SAM**（`facebook/sam-vit-base`, 第22回）は、その箱を `input_boxes` プロンプトとして受け、箱の中の物体を**シャープな画素マスク**に切り出します。

なぜ1段（CLIPSeg）で済むのにわざわざ2段にするのか。理由は**役割分担と品質**です。検出器は「**どこに何があるか**」を見つけるのが得意、SAM は「**与えられた領域を高精度に切り出す**」のが得意で、両者を分業させると CLIPSeg より**境界の鋭いマスク**が得られます。しかも段1を別の検出器（より語彙の広いモデルや、自前学習の検出器）に**差し替えるだけ**で語彙や精度を伸ばせる――この**モジュール性**が Grounded-SAM の強みです。コードの骨格は、検出結果の各 box をループして SAM に渡すだけです。SAM は box ごとに候補マスクを3枚返すので、`iou_scores`（自己推定品質）の argmax を採用します。

```python
# 段1: Grounding DINO で box 検出（target_sizes は (height, width) 順）
results = gdino_proc.post_process_grounded_object_detection(
    outputs, inputs["input_ids"], threshold=0.25, text_threshold=0.25, target_sizes=[(h, w)])[0]

# 段2: 各 box を SAM でマスク化（候補3枚から iou_scores 最大を選ぶ）
inputs = sam_proc(image, input_boxes=[[box]], return_tensors="pt")
masks = sam_proc.post_process_masks(out.pred_masks.cpu(),
            inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu())[0]
best = int(torch.argmax(out.iou_scores[0, 0]))        # (3,) の中で最良
mask = masks[0, best].numpy().astype(bool)
```

合成シーンでの実測（`02_grounded_sam.json`）では、Grounding DINO が3物体すべてを検出し（スコア: red circle 0.90 / green triangle 0.84 / blue square 0.83）、それを SAM でマスク化した**最終マスク IoU は 0.998〜1.000**――ほぼ完璧でした（`02_grounded_sam_panel.png` の左に検出箱、右に SAM マスクが出ます）。CLIPSeg の 0.88〜0.95 と比べると、**SAM の境界の鋭さ**が IoU の差として効いていることが分かります。なお合成画像はテクスチャが乏しく、実写と違って Grounding DINO の検出が振るわないこともあります。本スクリプトは**検出ゼロを検知したら GT の外接箱を SAM に与えるフォールバック**に切り替え、どんな入力でも SAM 単体の品質は確認でき、かつ必ず exit 0 になるよう作ってあります。

## 6. box 閾値の影響 — 取りこぼし(recall) とのトレードオフ

Grounded-SAM の最終マスク品質は、**段1の検出が成否を握ります**。検出されなかった物体は、SAM に渡す箱が無いので**そもそもマスク化できません**。そして検出されるか否かを決めるのが Grounding DINO の **box 閾値（threshold）**です。`03` の後半（B パート）は、同じシーンに対して box 閾値を変え、**検出数**と**平均マスク IoU**がどう動くかを測ります（GDINO の forward は重いので1回だけ実行し、`post_process` を閾値違いで呼び直す＝安価、というのが実装の勘所です）。図 `03_gsam_box_threshold.png` がそのトレードオフ曲線です。

| box 閾値 | 検出数 | 平均マスク IoU |
| --- | --- | --- |
| 0.25 | 3 | 0.999 |
| 0.50 | 3 | 0.999 |
| 0.80 | 3 | 0.999 |
| 0.85 | 1 | 0.998 |
| 0.90 | 1 | 0.998 |
| 0.95 | 0 | 0.000 |

挙動がはっきり出ています。閾値 0.80 までは3物体すべてを検出して IoU ≈ 0.999 ですが、**0.85 に上げると検出が1個に激減**（スコア 0.90 の red circle しか残らない）、**0.95 では検出ゼロ＝マスクも IoU も 0** に崩壊します。これは検出の普遍的なトレードオフそのものです。**閾値を上げる**と誤検出（false positive）は減りますが、確信度の低い真の物体を**取りこぼし（recall 低下）**ます。**閾値を下げる**と取りこぼしは減りますが、背景や紛らわしい領域を**過検出**します。Grounded-SAM では、この「段1の recall」が**そのまま最終マスクの網羅性の上限**になる――検出できなかった物体は永遠にマスク化されない――という点が CLIPSeg との大きな違いです。

実務での使い分けに直結します。「**取りこぼしたくない**（医用・安全系）」なら box 閾値を低めにして過検出を後段でフィルタ、「**誤検出を避けたい**（自動処理の信頼性重視）」なら高めにして確実なものだけ通す、というのが基本方針です。CLIPSeg にも sigmoid しきい値という似たノブがありましたが（第4節）、あちらは「**1つの確率マップの切り方**」を変えるだけで対象が消えることはないのに対し、Grounded-SAM の box 閾値は「**そもそも対象を見つけるか否か**」を左右する――同じ「閾値」でも効き方の階層が違う、と理解すると見通しが良くなります。

## 7. pipeline('mask-generation') との比較 — プロンプト有り/無しの違い

参照セグメ（CLIPSeg・Grounded-SAM）は**「文で指した特定の対象」**を出す、いわば**プロンプト駆動**の切り出しでした。これと対になるのが SAM の**自動マスク生成**――`pipeline("mask-generation", model="facebook/sam-vit-base")` です。これは**プロンプトを与えず**、画像全体に点のグリッドを敷き、各点から SAM を走らせて**画像中のあらゆる部位を網羅的にマスク化**します。「この画像にある“もの”を全部、ラベル無しで切り分ける」プロンプトフリーのセグメで、用途は「とりあえず全部分割して後で選ぶ」「アノテーション支援」などです。

両者は**目的が逆**です。参照セグメは「**何を切るかが先に決まっていて**、それを文で指す」のに対し、mask-generation は「**何があるか分からないので**、全部出してから人/後段が選ぶ」。前者は出力が**少数の意味付きマスク**、後者は**大量の意味なしマスク**になります。実務では、対象が言葉で言えるなら参照セグメ、探索的に全部見たいなら mask-generation、と選びます。なお mask-generation は**点グリッドの数だけ SAM を走らせる**ので CPU では重く、本講座の `02` では既定で**概念紹介に留め**、`RUN_MASKGEN=1` を付けたときだけ `points_per_side=8` に絞って軽量実行する作りにしています（`02_grounded_sam.json` に結果が載ります）。

```bash
# 既定は概念のみ（CPU 負荷を避ける）。試したいときだけ環境変数で有効化:
RUN_MASKGEN=1 uv run python lectures/23_text_prompt_segmentation/02_grounded_sam.py
```

3つを一望すると、**CLIPSeg = 文→マスク（1段・軽量・ソフト境界）**、**Grounded-SAM = 文→箱→マスク（2段・高精度・差し替え可）**、**mask-generation = プロンプト無しで全部（網羅的・重い）**、という棲み分けになります。同じ「SAM を使う」でも、`input_boxes` を与える Grounded-SAM と、点グリッドで自動生成する mask-generation では**得られるものがまったく違う**――この対比が腑に落ちれば、現場で「どれを使うべきか」を即座に判断できます。

## 8. 使い分けの指針 — CLIPSeg と Grounded-SAM

最後に、本章の2手法を実務目線で並べます。**CLIPSeg** は1モデル・軽量で、CPU でも数秒、しきい値で挙動を後から調整でき、`"sky"` `"road"` のような**領域的・非物体的な概念**（境界が曖昧な“もの”）も塗れるのが強みです。弱みは**境界がソフト**（確率の裾が滲む）で、複数インスタンスの**個体分離が苦手**なこと。一方 **Grounded-SAM** は2段で重い代わりに、**境界がシャープ**で**個体ごとに箱→マスク**が出せ、検出器を差し替えれば**語彙も精度も拡張**できます。弱みは**段1の検出に失敗すると即マスク無し**（recall が上限を決める）で、依存も2モデルぶん重いこと。

| 観点 | CLIPSeg（1段・直接） | Grounded-SAM（2段・合成） |
| --- | --- | --- |
| 構成 | 文 → 確率マップ（1モデル） | 文 → 箱(GDINO) → マスク(SAM) |
| 出力 | ソフトな確率ヒートマップ | シャープな2値マスク |
| 個体分離 | 苦手（領域をまとめて塗る） | 得意（box ごとに分かれる） |
| 非物体概念(sky 等) | 塗れる | 検出器が箱を出せず苦手 |
| 速度/依存 | 速い・軽い | 重い・2モデル |
| 主なノブ | sigmoid しきい値（切り方） | box 閾値（見つけるか否か）＋ SAM |
| 失敗モード | 境界が滲む/塗り過ぎ | 検出漏れ＝マスク無し |

この表の一行で迷ったら、**「対象は“もの”か“領域”か」「個体を分けたいか」「速度と精度のどちらを優先するか」**で選ぶのが実用的です。発展として、Grounding DINO の box をそのまま CLIPSeg の条件に混ぜる、SAM のマスクで CLIPSeg の確率を後段リファインする、といったハイブリッドも考えられます（本章のコードはどちらの部品も `seg_helpers` に分離してあるので、組み替えの土台になります）。まずは2つの素直なパイプラインを確実に動かし、IoU で違いを数字にできることを目標にしてください。

## 9. このモジュールの構成（スクリプト一覧）

各スクリプトは単一責務で、上から読むと「直接型を動かす → 合成型を組む → 数字で評価する」と理解が積み上がります。すべて `outputs/23_text_prompt_segmentation/` に図と json を保存し、画面表示には依存しません。device 判定・合成シーン生成・モデルロード・CLIPSeg 後処理・IoU/Dice・可視化といった共通処理は `seg_helpers.py` にまとめ、各スクリプトはそれを import します。

| ファイル | 役割（単一責務） |
| --- | --- |
| `seg_helpers.py` | device 判定・合成シーン＋GT 生成・CLIPSeg/SAM/GDINO ロード・`clipseg_probs`・IoU/Dice・可視化。道具箱 |
| `01_clipseg.py` | CLIPSeg で `logits→sigmoid→閾値`。3プロンプトの確率マップ・IoU/Dice・「不在の概念」検証 |
| `02_grounded_sam.py` | Grounding DINO → SAM の2段構成。検出箱→マスク・最終 IoU・mask-generation の対比（概念） |
| `03_referring_iou_eval.py` | CLIPSeg しきい値スイープ（IoU 最大点）＋ Grounded-SAM の box 閾値スイープ（recall トレードオフ） |
| `exercises.py` | TODO 形式の演習（自己採点ランナー付き。`SHOW_SOLUTION=1` で模範解答） |

`seg_helpers.py` だけは「読み物」ではなく「再利用する道具」です。とくに `clipseg_probs`（`logits→補間→sigmoid` を1つにまとめた中核）、`mask_iou`/`mask_dice`/`best_threshold`（評価の土台）、`build_scene`（GT 付き合成シーン）が3スクリプト全部の基盤になっています。まず helper を一読してから 01 へ進むと、各スクリプトが何を import しているかが腑に落ちます。

## 10. 動かし方

このモジュールは `dl`（torch/torchvision）・`hf`（transformers/timm/safetensors ほか）グループに依存します。CLIPSeg・SAM・Grounding DINO は**いずれも HuggingFace transformers に同梱**されているため、Ultralytics 系の `detect` グループは不要です（評価の IoU/Dice も numpy 自前なので `metrics` も不要）。CPU だけで完走し、初回のみ3モデルの重みを HuggingFace からダウンロードします（以降はキャッシュから即起動）。プロジェクトルートで以下を順に実行してください。

```bash
# 依存グループをインストール（初回のみ）
uv sync --group dl --group hf

# 各スクリプトを実行（結果は outputs/23_text_prompt_segmentation/ に保存される）
uv run python lectures/23_text_prompt_segmentation/seg_helpers.py          # 道具箱のスモークテスト＋シーン図
uv run python lectures/23_text_prompt_segmentation/01_clipseg.py
uv run python lectures/23_text_prompt_segmentation/02_grounded_sam.py
uv run python lectures/23_text_prompt_segmentation/03_referring_iou_eval.py

# 演習: まずは TODO を自分で埋める（最初は全部 FAIL だが exit 0）
uv run python lectures/23_text_prompt_segmentation/exercises.py
# どうしても分からない時だけ、模範解答の挙動を見る
SHOW_SOLUTION=1 uv run python lectures/23_text_prompt_segmentation/exercises.py

# （任意）実画像で試す: data/23_text_prompt_segmentation/image.png を置き、
#         prompts.txt に 1 行 1 プロンプトを書くと自動で使われる（GT 無しなら可視化のみ）。
# （任意）SAM 自動マスク生成も見たい: RUN_MASKGEN=1 を付けて 02 を実行
```

実行後は `outputs/23_text_prompt_segmentation/` の図を解説と照らし合わせてください。とくに `01_clipseg_panel.png`（プロンプトごとの確率ヒートマップとマスク）、`03_clipseg_threshold_sweep.png`（IoU の山なりカーブ、ピークが 0.5 でない）、`02_grounded_sam_panel.png`（検出箱と SAM マスク）の3枚を見ると、本章の要点が視覚的に腑に落ちます。図中の文字は CJK フォントの豆腐（□）を避けるため ASCII にしてあります。色が反転して見える場合は、合成画像を RGB のまま扱っているか（cv2 経由で BGR が混ざっていないか）を確認してください。

## 11. よくあるエラーと対処（チェックリスト）

最後に、本章でつまずきやすい点を「症状 → 原因 → 対処」でまとめます。CLIPSeg/SAM/Grounding DINO 特有の罠が多いので、詰まったらまずここを見てください。

| 症状 | ほぼ確実な原因 | 対処 |
| --- | --- | --- |
| CLIPSeg のマスクが画像とずれる | 出力 352×352 を原寸へ補間していない | `F.interpolate(logits, size=(h,w))` で原寸化してから sigmoid |
| 複数プロンプトで長さ不一致エラー | `padding` を指定していない | `processor(text=prompts, ..., padding=True)` |
| CLIPSeg の確率を softmax で読んだ | 画素は独立判定なので sigmoid が正しい | `torch.sigmoid(logits)`。softmax は使わない |
| Grounding DINO が何も検出しない | テキストが大文字/区切り無し、合成画像で確信度が低い | 小文字＋ピリオド区切り（`"a cat. a dog."`）、`threshold` を下げる |
| `post_process_*` で box が歪む | `target_sizes` を (W,H) で渡した | (height, width) 順で渡す（`image.size[::-1]`） |
| `box_threshold=` で TypeError | transformers v5 で引数名が `threshold` に変更 | `post_process_grounded_object_detection(..., threshold=)` |
| SAM のマスクが低解像度/ずれる | `post_process_masks` を通していない | `original_sizes`・`reshaped_input_sizes` を渡して原寸化 |
| SAM のマスクが意図とずれる | 候補3枚から選んでいない | `iou_scores` の argmax を採用（multimask） |
| CPU で推論が極端に遅い | `float16`/`half` を CPU で使用、mask-generation を全点で実行 | CPU は `float32`、mask-generation は `points_per_side` を絞る |
| 毎回モデルを再DLする（Docker） | キャッシュをマウントしていない | `~/.cache/huggingface`（`HF_HOME`）をボリューム化 |

この表の項目が、本章で遭遇しがちな不具合のほぼ全てです。とくに上3つ（原寸補間・`padding`・sigmoid）は CLIPSeg の、`threshold` 引数名と `target_sizes` の順序は Grounding DINO（transformers v5）の「あるある」なので、症状を見たら原因を即座に言い当てられるようにしておきましょう。

## 12. まとめ

本章では、**「文で対象を指定して画素マスクを得る」参照セグメンテーション**を2つの設計で実装しました。**CLIPSeg** は `logits → sigmoid → 閾値` の1段・直接型で、合成シーンの3物体を IoU 0.88〜0.95 で当て、`"a yellow star"`（不在概念）を確率 0.03 で正しく「無い」と判断しました。しきい値スイープでは**最良点が 0.60〜0.70 にあり、既定の 0.5 が最適でない**ことを曲線で確かめました。**Grounded-SAM** は Grounding DINO →SAM の2段・合成型で、検出箱を SAM が IoU ≈ 0.999 のシャープなマスクに変換し、**box 閾値を上げると取りこぼし（recall）で検出が 3→1→0 に崩れる**様子を数字で見ました。通底するのは「**確率マップは後段のしきい値で、2段パイプラインは前段の検出 recall で、品質が決まる**」という勘所です。

ここで身につけた「テキスト→マスク」の発想と、IoU/Dice・しきい値スイープという評価の型は、第24回以降のキャプション／VQA／VLM（言語と画像をより深く結ぶタスク群）へと自然につながります。まずは演習を全問 PASS させ、`03` の「CLIPSeg のピークが 0.5 でない」「box 閾値 0.95 でマスクが消える」という2つの結果を自分の言葉で説明できるようにしてから、次へ進んでください。

---

> 本教材で参照・検証したライブラリとバージョン（2026-06-11 時点の安定版で動作確認）:
> Python 3.12 ／ torch 2.12.0+cpu ／ torchvision 0.27.0+cpu ／ transformers 5.11.0 ／ huggingface-hub 1.18.0 ／ timm 1.0.27 ／ safetensors 0.8.0 ／ numpy 2.4.6 ／ Pillow 12.2.0 ／ matplotlib 3.10.9 ／ opencv-python-headless 4.13.0（合成シーンの描画）
> 使用モデル: `CIDAS/clipseg-rd64-refined`（CLIPSeg）／ `facebook/sam-vit-base`（SAM）／ `IDEA-Research/grounding-dino-tiny`（Grounding DINO）。いずれも HuggingFace transformers 同梱で、初回のみ重みを取得しキャッシュします。Grounded-SAM の発展（より広い語彙の検出器への差し替え）や SAM2/MobileSAM への置換は、`seg_helpers.load_*` を差し替えるだけで試せます。