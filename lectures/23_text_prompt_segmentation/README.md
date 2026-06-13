# 第23回 テキストプロンプト/参照セグメンテーション — CLIPSeg と Grounded-SAM

> トラック: **セグメンテーション** ／ レベル: **中級** ／ 依存グループ: `dl`（torch/torchvision）・`hf`（transformers/timm ほか）。CPU だけで完走します（初回のみモデル重みを HuggingFace からダウンロード）。評価指標（IoU/Dice）は numpy で自前実装するので `metrics` グループは不要です。

## 🎯 この章のゴール

第21回ではセマンティックセグメンテーション（画像の全画素を**固定クラス**へ振り分ける）を、第22回ではインスタンス分割と SAM（点やボックスの**プロンプト**で対象を切り出す）を学びました。本章で扱うのはその先――**「文（テキスト）で対象を指定し、その領域だけをマスク化する」参照セグメンテーション（referring segmentation）**です。固定クラスにも、点・箱という幾何プロンプトにも縛られず、`"a red circle"` のような自然言語で「どこを塗るか」を指示できる点が核心になります。第16回（CLIP）で身につけた「画像とテキストを同じ空間で結びつける」発想が、ついに**画素単位のマスク**へと降りてくるわけです。

この章では、2つのアプローチを実際に手で動かしながら対比します。ひとつは **CLIPSeg**（`CIDAS/clipseg-rd64-refined`）――CLIP の上に軽量なデコーダを載せ、**1モデルで「文 → 確率マップ」を直接出す**手法です。その出力 `outputs.logits` を `torch.sigmoid` で 0〜1 の確率に変換し、**しきい値で2値マスク化**します。もうひとつは **Grounded-SAM**――まず **Grounding DINO**（オープン語彙の box 検出, 第20回の延長）で文が指す物体の箱を出し、その箱を **SAM**（第22回）の `input_boxes` に渡して画素マスクへ切り出す、**役割の違う2モデルを連結した2段パイプライン**です。

到達点は4つです。第一に、CLIPSeg で `logits → sigmoid → 閾値` という後処理を自分で書き、**参照テキストごとに確率ヒートマップとマスク**を得ること。第二に、予測マスクと正解マスク(GT)の **IoU・Dice** を numpy で計算したうえで、**sigmoid しきい値をスイープして IoU 最大点を探す**こと（その過程で、既定の 0.5 が最適とは限らないと体感します）。第三に、Grounding DINO → SAM の**2段構成を組み**、検出 box を SAM でマスク化できること。第四に、**Grounding DINO の box 閾値**が最終マスクの取りこぼし（recall）を支配することを、閾値スイープで数字とともに確かめることです。なお本章のスクリプトはすべて、ネットもデータセット DL も無しで完走するよう、入力を**合成シーン**（赤い円・青い四角・緑の三角）として描き、各図形の **GT マスクを画素単位で厳密に保持**します。CLIPSeg も Grounding DINO も、色＋形のような単純な概念であれば合成画像にも反応するため、教材として意味のある IoU が得られます（実写で試したい人は `data/23_text_prompt_segmentation/` に画像を置けば自動で使われます）。

---

## 1. 参照セグメとは何か — 検出・セマンティック・SAM との位置づけ

ここまでに学んだセグメ／検出を「**何で対象を指定するか**」という観点で並べると、参照セグメの居場所がはっきりします。まずセマンティックセグメ（第21回）は、**学習時に固定したクラス集合**（道路・人・車…）でしか塗れません。次にインスタンス分割や SAM（第22回）は、SAM であれば**点・ボックスという幾何プロンプト**で「ここ」を指せますが、**「赤い方の椅子」「左から2番目の人」のような言語的な指定はできません**。一方、オープン語彙検出（第20回, OWL-ViT/Grounding DINO）は**任意のテキスト**で物体を指せるものの、出力は**箱**であって画素マスクではありません。参照セグメは、この最後のピース――**任意のテキストで指定し、出力は画素マスク**――を埋めるタスクなのです。

<figure class="lec-fig"><svg viewBox="0 0 620 300" role="img" aria-label="対象の指定方法と出力の2軸で参照セグメの位置づけを示す象限図" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><line x1="66" y1="256" x2="66" y2="44" stroke="#71717a" stroke-width="2"/><polygon points="66,38 61,48 71,48" fill="#71717a"/><line x1="66" y1="256" x2="600" y2="256" stroke="#71717a" stroke-width="2"/><polygon points="606,256 596,251 596,261" fill="#71717a"/><rect x="86" y="56" width="244" height="94" fill="#f4f4f5" stroke="#d4d4d8" stroke-width="1.5"/><rect x="344" y="56" width="244" height="94" rx="4" fill="#fff7ed" stroke="#ea580c" stroke-width="2.5"/><rect x="86" y="158" width="244" height="94" fill="#f4f4f5" stroke="#d4d4d8" stroke-width="1.5"/><rect x="344" y="158" width="244" height="94" fill="#f4f4f5" stroke="#d4d4d8" stroke-width="1.5"/><text x="208" y="100" text-anchor="middle" font-size="16" font-weight="700" fill="#3f3f46">セマンティック・SAM</text><text x="208" y="124" text-anchor="middle" font-size="12.5" fill="#71717a">固定クラス / 点・箱で指定</text><text x="466" y="98" text-anchor="middle" font-size="18" font-weight="700" fill="#c2410c">★ 参照セグメ</text><text x="466" y="124" text-anchor="middle" font-size="12.5" fill="#c2410c">任意テキスト → 画素マスク</text><text x="208" y="202" text-anchor="middle" font-size="16" font-weight="700" fill="#3f3f46">物体検出</text><text x="208" y="226" text-anchor="middle" font-size="12.5" fill="#71717a">固定クラス → 箱</text><text x="466" y="202" text-anchor="middle" font-size="16" font-weight="700" fill="#3f3f46">オープン語彙検出</text><text x="466" y="226" text-anchor="middle" font-size="12.5" fill="#71717a">任意テキスト → 箱</text><text x="44" y="150" text-anchor="middle" font-size="12.5" fill="#52525b" style="writing-mode:vertical-rl;text-orientation:upright">出力</text><text x="208" y="278" text-anchor="middle" font-size="12.5" fill="#52525b">固定クラス・幾何で指定</text><text x="466" y="278" text-anchor="middle" font-size="12.5" fill="#52525b">任意テキストで指定</text></svg><figcaption>ここまでのタスクを <b>「何で指定するか」×「何を出力するか」</b> の2軸で並べた図です。<b>セマンティックセグメや SAM</b> は画素マスクを出せても固定クラスや点・箱でしか指せず、<b>オープン語彙検出</b> は任意テキストで指せても出力は箱どまりです。<b>参照セグメ</b> はこの空いた象限――<b>任意テキストで指定し、出力は画素マスク</b>――を埋めるタスクです。</figcaption></figure>

では、なぜこれが嬉しいのでしょうか。実務では「画像の中の**特定の対象だけ**を、**学習し直さずに**、**自然言語で**切り出したい」という場面が頻出します。たとえば、商品画像から「the price tag（値札）」だけを抜く、医用画像で「the tumor region」に粗く当たりを付ける、自動運転ログの「the pedestrian crossing」を塗る、といった具合です。こうした場面で固定クラス分類器を毎回作り直すのは非現実的ですし、点や箱を人手で打つのも面倒です。その点、**文で指せて画素が返る**参照セグメは、この「柔軟さ」と「画素精度」を同時に満たしてくれます。

本章で扱う2手法は、この目標へ至る**対照的な2つの道**です。CLIPSeg は「文 → マスク」を**1つのモデルで一気に**解く軽量・直接型、Grounded-SAM は「文 → 箱（検出器）→ マスク（SAM）」と**2段に分け**、それぞれ得意なモデルに分業させる合成型です。前者は速くて手軽、後者は重いものの境界がシャープで差し替えが効く――この**設計上のトレードオフ**を実際に動かしながら体得するのが、この章の主眼になります。そこでまずは評価の土台（合成シーンと IoU/Dice）を固め、その後 CLIPSeg から触っていきましょう。

## 2. 評価の土台 — 合成シーンと IoU/Dice の定義

参照セグメの良し悪しは「**文が指した領域**を、どれだけ**正確に画素で当てたか**」で測ります。そのためには、まず「正解マスク（GT）」が画素単位で分かっている必要があります。本章では `seg_helpers.build_scene()` が、明るい灰色の背景に**赤い円・青い四角・緑の三角**を重ならない位置で描いた1枚のシーンを作り、同時に**各図形だけを別キャンバスに描いて画素>0 を取る**ことで、厳密な GT マスク（bool 配列）を得ています。図形どうしを重ねないのは、参照セグメが「どの領域か」を一意に指したいタスクだからです（`outputs/23_text_prompt_segmentation/00_scene_and_gt.png` にシーンと GT が並びます）。

マスクどうしの一致は、**IoU（Intersection over Union）** と **Dice** で測ります。予測マスク P と正解マスク G について、IoU は**重なり面積を和集合面積で割った値** `|P∩G| / |P∪G|`、Dice は `2|P∩G| / (|P|+|G|)`（F1 スコアと同値）です。どちらも 1.0 が完全一致で、同じ重なりであれば常に **Dice ≥ IoU**（Dice の方が甘く評価する）という関係が成り立ちます。実装は、次のように numpy だけで完結します。なお両者とも、空マスクのときに 0 で割らないよう `1.0`（完全一致扱い）を返すのが実務上のお約束です。

<figure class="lec-fig"><svg viewBox="0 0 640 300" role="img" aria-label="予測マスクPと正解マスクGの重なりからIoUとDiceを求める図" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><rect x="90" y="66" width="180" height="140" fill="#ffedd5" stroke="#ea580c" stroke-width="2.5"/><rect x="210" y="110" width="180" height="140" fill="#dbeafe" stroke="#2563eb" stroke-width="2.5"/><rect x="210" y="110" width="60" height="96" fill="#15803d"/><text x="104" y="90" font-size="15" font-weight="700" fill="#c2410c">P 予測</text><text x="380" y="238" text-anchor="end" font-size="15" font-weight="700" fill="#1d4ed8">G 正解(GT)</text><text x="240" y="164" text-anchor="middle" font-size="13" font-weight="700" fill="#ffffff">P∩G</text><rect x="430" y="70" width="194" height="152" rx="8" fill="#fafafa" stroke="#e4e4e7" stroke-width="1.5"/><text x="446" y="104" font-size="16" font-weight="700" fill="#c2410c">IoU</text><text x="446" y="128" font-size="14" fill="#18181b">= |P∩G| / |P∪G|</text><text x="446" y="170" font-size="16" font-weight="700" fill="#1d4ed8">Dice</text><text x="446" y="194" font-size="13.5" fill="#18181b">= 2|P∩G| / (|P|+|G|)</text><text x="527" y="216" text-anchor="middle" font-size="12" fill="#52525b">同じ重なりなら Dice ≥ IoU</text></svg><figcaption><b>IoU</b> と <b>Dice</b> は、予測マスク <b>P</b>(オレンジ枠)と正解マスク <b>G</b>(青枠)の重なりで一致度を測ります。緑が共通部分 <b>P∩G</b>、両方の矩形が覆う全体が和集合 <b>P∪G</b> です。<code>IoU = |P∩G| / |P∪G|</code>、<code>Dice = 2|P∩G| / (|P|+|G|)</code> で、どちらも 1.0 が完全一致。同じ重なりなら必ず <b>Dice ≥ IoU</b> になります。</figcaption></figure>

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

この2つの関数が、本章のすべての評価の土台になります。検出の mAP（第19回）が「箱の IoU でマッチングしてから順位を見る」複雑な指標だったのに対し、参照セグメは「**前景マスクどうしの単純な重なり**」を見るだけなので、定義がそのまま実装になります。そして GT を厳密に持っているからこそ、次節以降で「CLIPSeg のしきい値を変えると IoU がどう動くか」を曲線で追えるのです。

## 3. CLIPSeg — logits → sigmoid → 閾値でマスクにする

`01_clipseg.py` の主役 **CLIPSeg** は、CLIP の画像エンコーダの上に軽量な FiLM 条件付きデコーダを載せたモデルです。テキスト（プロンプト）でこのデコーダを条件付けることで、**「その文が指す領域」の画素ごとのスコア**を出力します。使い方は CLIP とよく似ており、`CLIPSegProcessor` で**画像と複数プロンプトを同時に前処理**し（長さを揃えるため `padding=True`）、`model(**inputs).logits` で **(プロンプト数, 352, 352) のロジット**を得ます。この出力は常に内部解像度 352×352 なので、原寸へ補間してから確率化するのが定石です。

ここで、CLIP（第16回）との**決定的な違い**を押さえておきましょう。CLIP は画像とテキストを**1本のベクトル**に潰して類似度を測りました。これに対し CLIPSeg は、それを**画素のグリッド**に展開し、各画素について「この文に合うか」をロジットで出します。したがって後処理も分類と同じ `sigmoid`――各画素を独立に 0〜1 の確率へ変換し、**しきい値で前景/背景に二値化**します（softmax ではない点に注意。画素ごとの独立判定だからこそ sigmoid なのです）。この流れは `seg_helpers.clipseg_probs` にまとめてあり、`F.interpolate` で原寸へ上げてから `torch.sigmoid` を掛けています。

<figure class="lec-fig"><svg viewBox="0 0 660 280" role="img" aria-label="CLIPSegの後処理パイプライン logitsからsigmoidを経てしきい値で2値マスク化" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><rect x="16" y="92" width="104" height="82" rx="6" fill="#2563eb" stroke="#1d4ed8" stroke-width="2"/><text x="68" y="130" text-anchor="middle" font-size="16" font-weight="700" fill="#ffffff">CLIPSeg</text><text x="68" y="150" text-anchor="middle" font-size="11" fill="#dbeafe">文で条件付け</text><rect x="147" y="92" width="104" height="82" rx="6" fill="#fff7ed" stroke="#f97316" stroke-width="1.8"/><text x="199" y="128" text-anchor="middle" font-size="14.5" font-weight="700" fill="#c2410c">logits</text><text x="199" y="150" text-anchor="middle" font-size="12" fill="#52525b">(P,352,352)</text><rect x="278" y="92" width="104" height="82" rx="6" fill="#fff7ed" stroke="#f97316" stroke-width="1.8"/><text x="330" y="138" text-anchor="middle" font-size="14.5" font-weight="700" fill="#c2410c">(P, H, W)</text><rect x="409" y="92" width="104" height="82" rx="6" fill="#ffedd5" stroke="#ea580c" stroke-width="2"/><text x="461" y="128" text-anchor="middle" font-size="14.5" font-weight="700" fill="#c2410c">確率 0〜1</text><text x="461" y="150" text-anchor="middle" font-size="11" fill="#71717a">ヒートマップ</text><rect x="540" y="92" width="104" height="82" rx="6" fill="#ffffff" stroke="#18181b" stroke-width="2"/><text x="592" y="138" text-anchor="middle" font-size="14.5" font-weight="700" fill="#18181b">2値マスク</text><line x1="120" y1="133" x2="143" y2="133" stroke="#71717a" stroke-width="2"/><polygon points="149,133 141,128 141,138" fill="#71717a"/><line x1="251" y1="133" x2="274" y2="133" stroke="#71717a" stroke-width="2"/><polygon points="280,133 272,128 272,138" fill="#71717a"/><line x1="382" y1="133" x2="405" y2="133" stroke="#71717a" stroke-width="2"/><polygon points="411,133 403,128 403,138" fill="#71717a"/><line x1="513" y1="133" x2="536" y2="133" stroke="#71717a" stroke-width="2"/><polygon points="542,133 534,128 534,138" fill="#71717a"/><text x="134" y="84" text-anchor="middle" font-size="11.5" fill="#3f3f46">出力</text><text x="265" y="84" text-anchor="middle" font-size="11.5" fill="#3f3f46">補間(原寸)</text><text x="396" y="84" text-anchor="middle" font-size="11.5" fill="#3f3f46">sigmoid</text><text x="527" y="84" text-anchor="middle" font-size="11.5" fill="#3f3f46">≥ 0.5</text><text x="330" y="212" text-anchor="middle" font-size="12" fill="#52525b">各画素を独立に判定 → softmax ではなく sigmoid</text></svg><figcaption>CLIPSeg の後処理は <b>logits → sigmoid → しきい値</b> の3手です。モデルは常に内部解像度の <code>logits (P, 352, 352)</code> を出すので、まず <code>F.interpolate</code> で <b>原寸 (H, W) へ補間</b> し、<code>torch.sigmoid</code> で各画素を <b>0〜1 の確率</b> に変換、最後に <code>≥ 0.5</code> で <b>2値マスク</b> にします。画素ごとに独立評価するため <b>softmax ではなく sigmoid</b> を使うのが要点です。</figcaption></figure>

```python
inputs = processor(text=prompts, images=[image]*len(prompts), padding=True, return_tensors="pt")
logits = model(**inputs).logits                       # (P, 352, 352) ロジット
up = F.interpolate(logits.unsqueeze(1), size=(h, w),  # 原寸 (H,W) へ補間
                   mode="bilinear", align_corners=False).squeeze(1)
probs = torch.sigmoid(up)                              # (P, H, W) 0〜1 の確率
masks = probs >= 0.5                                   # しきい値で2値マスク化
```

合成シーンに3つのプロンプト（`a red circle` / `a blue square` / `a green triangle`）を投げた実測が、下表です（`01_clipseg_metrics.json`）。しきい値 0.5 でも IoU は 0.88〜0.95 と高く、`01_clipseg_panel.png` のヒートマップを見ると、各プロンプトが対応する図形だけを正しく赤く（高確率に）灯しています。ここから、CLIPSeg が「色＋形」という概念を**ゼロショットで画素レベルに**落とし込めていることが分かります。

| プロンプト | prob 最大 | IoU@0.5 | Dice@0.5 |
| --- | --- | --- | --- |
| a red circle | 0.91 | 0.945 | 0.972 |
| a blue square | 0.98 | 0.942 | 0.970 |
| a green triangle | 0.98 | 0.884 | 0.939 |

さらに `01` は、「**シーンに無い概念**」も投げます。`"a yellow star"`（黄色い星はシーンに存在しない）というプロンプトでは、**prob 最大が 0.03、0.5 を超える画素は 0** でした。CLIPSeg の sigmoid は画素を独立に評価するため、**該当物が無ければ全画素が低いまま**となり、しきい値で「無い」と正しく判定できます。これは「候補のどれか1つに無理やり確率を寄せる」softmax 型の分類（第16回）とは対照的で、まさに**参照セグメに sigmoid が向く理由**そのものです。こうしてマスクが「それっぽく」出たら、次は「どこで切るか（しきい値）」を真面目に考えていきます。

## 4. しきい値の選び方 — 既定の 0.5 が最適とは限らない

CLIPSeg の出力は**連続値の確率マップ**なので、最終的な2値マスクは「どのしきい値で切るか」によって変わります。前節では便宜上 0.5 を使いましたが、これは本当に最適なのでしょうか。`03_referring_iou_eval.py` の前半（A パート）は、各プロンプトについて**しきい値を 0.05〜0.95 でスイープし、毎回 IoU を測って曲線を描き**ます（`best_threshold` 関数が IoU 最大点を返します）。下が実測の要約で、図 `03_clipseg_threshold_sweep.png` には、3本の山なりカーブと「最適点」「既定 0.5」の縦線が描かれています。

| プロンプト | IoU@0.5 | 最良しきい値 | 最良 IoU |
| --- | --- | --- | --- |
| a red circle | 0.945 | 0.65 | 0.965 |
| a blue square | 0.942 | 0.60 | 0.947 |
| a green triangle | 0.884 | 0.70 | 0.936 |

読み取れることは明快です。**最良しきい値は 0.60〜0.70 にあり、どのプロンプトでも 0.5 ではありません**。曲線が山なりになるのは、しきい値が低すぎると確率の裾野まで拾って**過剰に塗り（マスクが膨らんで IoU 低下）**、逆に高すぎると確信の高い中心しか残らず**塗り残す（マスクが痩せて IoU 低下）**からです。つまり 0.5 は「無難な初期値」ではあっても「最適」ではなく、**対象やプロンプトごとにピークがずれる**わけです。なかでも green triangle が最も高い 0.70 を要するのは、三角形の鋭い頂点付近で確率がなだらかに落ちるため、低めで切ると背景まで拾ってしまうから――と図から推測できます。

<figure class="lec-fig"><svg viewBox="0 0 640 300" role="img" aria-label="sigmoidしきい値に対するIoUの山なりカーブ 最良点は0.5でなく約0.65" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><line x1="70" y1="250" x2="70" y2="50" stroke="#71717a" stroke-width="2"/><polygon points="70,44 65,54 75,54" fill="#71717a"/><line x1="70" y1="250" x2="600" y2="250" stroke="#71717a" stroke-width="2"/><polygon points="606,250 596,245 596,255" fill="#71717a"/><text x="40" y="150" text-anchor="middle" font-size="14" font-weight="700" fill="#3f3f46" transform="rotate(-90 40 150)">IoU</text><text x="335" y="290" text-anchor="middle" font-size="13" fill="#3f3f46">sigmoid しきい値 →</text><text x="70" y="268" text-anchor="middle" font-size="11" fill="#71717a">0</text><text x="330" y="268" text-anchor="middle" font-size="11" fill="#71717a">0.5</text><text x="590" y="268" text-anchor="middle" font-size="11" fill="#71717a">1.0</text><polyline points="96,145 174,117 252,94 330,79 382,70 408,67 460,75 512,98 564,145" fill="none" stroke="#ea580c" stroke-width="3"/><line x1="330" y1="250" x2="330" y2="79" stroke="#71717a" stroke-width="1.5" stroke-dasharray="5 3"/><text x="322" y="72" text-anchor="end" font-size="12" fill="#52525b">既定 0.5</text><line x1="408" y1="250" x2="408" y2="67" stroke="#ea580c" stroke-width="1.5" stroke-dasharray="5 3"/><circle cx="408" cy="67" r="5" fill="#ea580c"/><text x="420" y="60" font-size="12.5" font-weight="700" fill="#c2410c">最良 ≈ 0.65</text><text x="150" y="200" text-anchor="middle" font-size="12.5" font-weight="700" fill="#dc2626">低い → 塗りすぎ</text><text x="510" y="200" text-anchor="middle" font-size="12.5" font-weight="700" fill="#dc2626">高い → 塗り残し</text></svg><figcaption>CLIPSeg の確率マップをどの <b>しきい値</b> で切るかで IoU は変わり、曲線は <b>山なり</b> になります。低すぎると確率の裾まで拾って <b>塗りすぎ</b>(マスクが膨張)、高すぎると確信の高い中心しか残らず <b>塗り残し</b>(マスクが痩せる)になり、どちらも IoU が下がります。実測では <b>最良点は約 0.60〜0.70</b> にあり、<b>既定の 0.5 は最適ではありません</b>。GT があるなら本図のようにスイープして最大点を選ぶのが正攻法です。</figcaption></figure>

実務的な含意は、こうです。GT が手元にある（検証セットがある）なら、本節のように**しきい値をスイープして最適点を選ぶ**のが正攻法です。一方、GT が無い運用時は、0.5 を起点にしつつ「塗りすぎなら上げる／塗り残すなら下げる」というように**対象に応じて手で調整**します。CLIPSeg の「確率マップを返す」という性質は、この**後段のしきい値で挙動を後から調整できる**柔軟さの裏返しでもあるのです。さて次は、まったく別の設計――2つのモデルを連結する Grounded-SAM――へ進みましょう。

## 5. Grounded-SAM — 検出（Grounding DINO）→ セグメ（SAM）の2段構成

`02_grounded_sam.py` が組むのは **Grounded-SAM**、すなわち「**文で箱を出す検出器**」と「**箱を画素マスクに変える SAM**」を直列につないだパイプラインです。段1の **Grounding DINO**（`IDEA-Research/grounding-dino-tiny`）は第20回で触れたオープン語彙検出器で、`"a red circle. a blue square. a green triangle."` のように**小文字＋各物体をピリオド区切り**にしたテキストを受け取り、文が指す物体の bounding box を返します（この「小文字＋ピリオド区切り」は Grounding DINO の作法であり、守らないと検出が安定しません）。続く段2の **SAM**（`facebook/sam-vit-base`, 第22回）は、その箱を `input_boxes` プロンプトとして受け、箱の中の物体を**シャープな画素マスク**へ切り出します。

では、なぜ1段（CLIPSeg）で済むのに、わざわざ2段にするのでしょうか。理由は**役割分担と品質**にあります。検出器は「**どこに何があるか**」を見つけるのが得意、SAM は「**与えられた領域を高精度に切り出す**」のが得意で、両者を分業させると CLIPSeg より**境界の鋭いマスク**が得られます。しかも段1を別の検出器（より語彙の広いモデルや、自前学習の検出器）に**差し替えるだけ**で語彙や精度を伸ばせる――この**モジュール性**こそが Grounded-SAM の強みです。コードの骨格は、検出結果の各 box をループして SAM に渡すだけ。SAM は box ごとに候補マスクを3枚返すので、`iou_scores`（自己推定品質）の argmax を採用します。

<figure class="lec-fig"><svg viewBox="0 0 640 250" role="img" aria-label="Grounded-SAMの2段構成 段1のGrounding DINOが文から箱を出し段2のSAMが箱をマスクにする" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><text x="195" y="64" text-anchor="middle" font-size="13" font-weight="700" fill="#1d4ed8">段1: 検出</text><text x="437" y="64" text-anchor="middle" font-size="13" font-weight="700" fill="#c2410c">段2: セグメ</text><rect x="12" y="78" width="88" height="84" rx="6" fill="#f4f4f5" stroke="#d4d4d8" stroke-width="1.5"/><text x="56" y="116" text-anchor="middle" font-size="15" font-weight="700" fill="#3f3f46">文</text><text x="56" y="138" text-anchor="middle" font-size="10.5" fill="#71717a">小文字+.</text><rect x="120" y="78" width="150" height="84" rx="6" fill="#dbeafe" stroke="#2563eb" stroke-width="2"/><text x="195" y="116" text-anchor="middle" font-size="13.5" font-weight="700" fill="#1d4ed8">Grounding DINO</text><text x="195" y="138" text-anchor="middle" font-size="11" fill="#52525b">文 → 箱</text><rect x="286" y="78" width="70" height="84" rx="6" fill="#ffffff" stroke="#d4d4d8" stroke-width="1.5"/><rect x="300" y="106" width="42" height="30" fill="none" stroke="#ea580c" stroke-width="2" stroke-dasharray="4 3"/><text x="321" y="98" text-anchor="middle" font-size="13" font-weight="700" fill="#c2410c">box</text><rect x="372" y="78" width="130" height="84" rx="6" fill="#ffedd5" stroke="#ea580c" stroke-width="2"/><text x="437" y="116" text-anchor="middle" font-size="16" font-weight="700" fill="#c2410c">SAM</text><text x="437" y="138" text-anchor="middle" font-size="11" fill="#71717a">箱 → マスク</text><rect x="520" y="78" width="104" height="84" rx="6" fill="#18181b"/><text x="572" y="112" text-anchor="middle" font-size="13.5" font-weight="700" fill="#ffffff">画素マスク</text><ellipse cx="572" cy="138" rx="22" ry="13" fill="#ffffff"/><line x1="100" y1="120" x2="114" y2="120" stroke="#71717a" stroke-width="2"/><polygon points="120,120 112,115 112,125" fill="#71717a"/><line x1="270" y1="120" x2="280" y2="120" stroke="#71717a" stroke-width="2"/><polygon points="286,120 278,115 278,125" fill="#71717a"/><line x1="356" y1="120" x2="366" y2="120" stroke="#71717a" stroke-width="2"/><polygon points="372,120 364,115 364,125" fill="#71717a"/><line x1="502" y1="120" x2="514" y2="120" stroke="#71717a" stroke-width="2"/><polygon points="520,120 512,115 512,125" fill="#71717a"/><text x="320" y="200" text-anchor="middle" font-size="12" fill="#c2410c">段1で検出できなければマスクも無い（recall が上限）</text><text x="320" y="222" text-anchor="middle" font-size="11.5" fill="#52525b">段1は別の検出器に差し替え可能（モジュール性）</text></svg><figcaption><b>Grounded-SAM</b> は役割の違う2モデルを直列につなぎます。<b>段1の Grounding DINO</b>(オープン語彙検出)が <code>"a red circle. a blue square."</code> のような <b>小文字+ピリオド区切り</b> の文から <b>箱(box)</b> を出し、<b>段2の SAM</b> がその箱を <code>input_boxes</code> として受け <b>シャープな画素マスク</b> へ切り出します。段1を別の検出器に差し替えれば語彙を拡張でき(モジュール性)、逆に <b>段1が検出しそこねた物体はマスク化されない</b>(検出 recall が網羅性の上限)点が CLIPSeg との違いです。</figcaption></figure>

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

合成シーンでの実測（`02_grounded_sam.json`）では、Grounding DINO が3物体すべてを検出し（スコア: red circle 0.90 / green triangle 0.84 / blue square 0.83）、それを SAM でマスク化した**最終マスク IoU は 0.998〜1.000**――ほぼ完璧でした（`02_grounded_sam_panel.png` の左に検出箱、右に SAM マスクが出ます）。CLIPSeg の 0.88〜0.95 と比べれば、**SAM の境界の鋭さ**が IoU の差として効いていることが分かります。ただし合成画像はテクスチャが乏しく、実写と違って Grounding DINO の検出が振るわないこともあります。そこで本スクリプトは、**検出ゼロを検知したら GT の外接箱を SAM に与えるフォールバック**に切り替えることで、どんな入力でも SAM 単体の品質を確認でき、かつ必ず exit 0 になるよう作ってあります。

## 6. box 閾値の影響 — 取りこぼし(recall) とのトレードオフ

Grounded-SAM の最終マスク品質は、**段1の検出が握っています**。検出されなかった物体は、SAM に渡す箱が無いため**そもそもマスク化できません**。そして、検出されるか否かを決めるのが Grounding DINO の **box 閾値（threshold）**です。`03` の後半（B パート）は、同じシーンに対して box 閾値を変えながら、**検出数**と**平均マスク IoU**がどう動くかを測ります（GDINO の forward は重いので1回だけ実行し、`post_process` を閾値違いで呼び直す＝安価、というのが実装の勘所です）。その結果をトレードオフ曲線として描いたのが、図 `03_gsam_box_threshold.png` です。

| box 閾値 | 検出数 | 平均マスク IoU |
| --- | --- | --- |
| 0.25 | 3 | 0.999 |
| 0.50 | 3 | 0.999 |
| 0.80 | 3 | 0.999 |
| 0.85 | 1 | 0.998 |
| 0.90 | 1 | 0.998 |
| 0.95 | 0 | 0.000 |

挙動がはっきり出ています。閾値 0.80 までは3物体すべてを検出して IoU ≈ 0.999 ですが、**0.85 に上げると検出が1個へ激減**し（スコア 0.90 の red circle しか残らない）、**0.95 では検出ゼロ＝マスクも IoU も 0** へ崩壊します。これは、検出の普遍的なトレードオフそのものです。すなわち**閾値を上げる**と誤検出（false positive）は減るものの、確信度の低い真の物体を**取りこぼし（recall 低下）**ます。逆に**閾値を下げる**と取りこぼしは減りますが、背景や紛らわしい領域を**過検出**します。Grounded-SAM では、この「段1の recall」が**そのまま最終マスクの網羅性の上限**になる――検出できなかった物体は永遠にマスク化されない――という点が、CLIPSeg との大きな違いです。

この性質は、実務での使い分けに直結します。「**取りこぼしたくない**（医用・安全系）」なら box 閾値を低めにして過検出を後段でフィルタし、「**誤検出を避けたい**（自動処理の信頼性重視）」なら高めにして確実なものだけ通す、というのが基本方針です。CLIPSeg にも sigmoid しきい値という似たノブがありましたが（第4節）、あちらは「**1つの確率マップの切り方**」を変えるだけで対象が消えることはありません。これに対し Grounded-SAM の box 閾値は「**そもそも対象を見つけるか否か**」を左右する――同じ「閾値」でも効き方の階層が違う、と理解すると見通しが良くなります。

## 7. pipeline('mask-generation') との比較 — プロンプト有り/無しの違い

参照セグメ（CLIPSeg・Grounded-SAM）は、**「文で指した特定の対象」**を出す、いわば**プロンプト駆動**の切り出しでした。これと対になるのが、SAM の**自動マスク生成**――`pipeline("mask-generation", model="facebook/sam-vit-base")` です。これは**プロンプトを与えず**、画像全体に点のグリッドを敷き、各点から SAM を走らせて**画像中のあらゆる部位を網羅的にマスク化**します。いわば「この画像にある“もの”を全部、ラベル無しで切り分ける」プロンプトフリーのセグメであり、用途は「とりあえず全部分割して後で選ぶ」「アノテーション支援」などです。

両者は、**目的が逆**です。参照セグメが「**何を切るかが先に決まっていて**、それを文で指す」のに対し、mask-generation は「**何があるか分からないので**、全部出してから人/後段が選ぶ」。したがって前者は出力が**少数の意味付きマスク**、後者は**大量の意味なしマスク**になります。実務では、対象が言葉で言えるなら参照セグメ、探索的に全部見たいなら mask-generation、と選び分けます。なお mask-generation は**点グリッドの数だけ SAM を走らせる**ため CPU では重く、本講座の `02` では既定で**概念紹介に留め**、`RUN_MASKGEN=1` を付けたときだけ `points_per_side=8` に絞って軽量実行する作りにしています（`02_grounded_sam.json` に結果が載ります）。

```bash
# 既定は概念のみ（CPU 負荷を避ける）。試したいときだけ環境変数で有効化:
RUN_MASKGEN=1 uv run python lectures/23_text_prompt_segmentation/02_grounded_sam.py
```

3つを一望すると、**CLIPSeg = 文→マスク（1段・軽量・ソフト境界）**、**Grounded-SAM = 文→箱→マスク（2段・高精度・差し替え可）**、**mask-generation = プロンプト無しで全部（網羅的・重い）**、という棲み分けが見えてきます。同じ「SAM を使う」でも、`input_boxes` を与える Grounded-SAM と、点グリッドで自動生成する mask-generation とでは**得られるものがまったく違う**――この対比が腑に落ちれば、現場で「どれを使うべきか」を即座に判断できるはずです。

## 8. 使い分けの指針 — CLIPSeg と Grounded-SAM

最後に、本章の2手法を実務目線で並べてみます。**CLIPSeg** は1モデル・軽量で、CPU でも数秒で動き、しきい値で挙動を後から調整でき、`"sky"` `"road"` のような**領域的・非物体的な概念**（境界が曖昧な“もの”）も塗れるのが強みです。弱みは、**境界がソフト**（確率の裾が滲む）で、複数インスタンスの**個体分離が苦手**なこと。一方 **Grounded-SAM** は、2段で重い代わりに、**境界がシャープ**で**個体ごとに箱→マスク**を出せ、検出器を差し替えれば**語彙も精度も拡張**できます。弱みは、**段1の検出に失敗すると即マスク無し**（recall が上限を決める）になることと、依存も2モデルぶん重いことです。

| 観点 | CLIPSeg（1段・直接） | Grounded-SAM（2段・合成） |
| --- | --- | --- |
| 構成 | 文 → 確率マップ（1モデル） | 文 → 箱(GDINO) → マスク(SAM) |
| 出力 | ソフトな確率ヒートマップ | シャープな2値マスク |
| 個体分離 | 苦手（領域をまとめて塗る） | 得意（box ごとに分かれる） |
| 非物体概念(sky 等) | 塗れる | 検出器が箱を出せず苦手 |
| 速度/依存 | 速い・軽い | 重い・2モデル |
| 主なノブ | sigmoid しきい値（切り方） | box 閾値（見つけるか否か）＋ SAM |
| 失敗モード | 境界が滲む/塗り過ぎ | 検出漏れ＝マスク無し |

この表を前に迷ったら、**「対象は“もの”か“領域”か」「個体を分けたいか」「速度と精度のどちらを優先するか」**という観点で選ぶのが実用的です。さらに発展として、Grounding DINO の box をそのまま CLIPSeg の条件に混ぜる、SAM のマスクで CLIPSeg の確率を後段リファインする、といったハイブリッドも考えられます（本章のコードはどちらの部品も `seg_helpers` に分離してあるので、組み替えの土台になります）。まずは2つの素直なパイプラインを確実に動かし、IoU で違いを数字にできることを目標にしてください。

## 9. このモジュールの構成（スクリプト一覧）

各スクリプトは単一責務で構成されており、上から順に読むと「直接型を動かす → 合成型を組む → 数字で評価する」と理解が積み上がります。すべて `outputs/23_text_prompt_segmentation/` に図と json を保存し、画面表示には依存しません。device 判定・合成シーン生成・モデルロード・CLIPSeg 後処理・IoU/Dice・可視化といった共通処理は `seg_helpers.py` にまとめてあり、各スクリプトはそれを import して使います。

| ファイル | 役割（単一責務） |
| --- | --- |
| `seg_helpers.py` | device 判定・合成シーン＋GT 生成・CLIPSeg/SAM/GDINO ロード・`clipseg_probs`・IoU/Dice・可視化。道具箱 |
| `01_clipseg.py` | CLIPSeg で `logits→sigmoid→閾値`。3プロンプトの確率マップ・IoU/Dice・「不在の概念」検証 |
| `02_grounded_sam.py` | Grounding DINO → SAM の2段構成。検出箱→マスク・最終 IoU・mask-generation の対比（概念） |
| `03_referring_iou_eval.py` | CLIPSeg しきい値スイープ（IoU 最大点）＋ Grounded-SAM の box 閾値スイープ（recall トレードオフ） |
| `mini_project.py` | 章末ミニプロジェクト。3設定（CLIPSeg@0.5 / CLIPSeg@best / Grounded-SAM）を同じシーンで競わせ、IoU 棒グラフ・パネル・JSON で勝敗を出す |
| `use_case.py` | 実践ユースケース。文で対象を指して **ぼかす/消す(inpaint)/透過切り出し** をする現実の編集ツール（評価ベンチの mini_project とは別物。実画像を `data/` に置けばそのまま実用） |
| `exercises.py` | TODO 形式の演習9問（易→難。自己採点ランナー付き。`SHOW_SOLUTION=1` で模範解答） |
| `exercises_solutions.py` | 演習の全問模範解答ランナー（採点ロジック・解答とも `exercises.py` を再利用。全 PASS を確認する用） |

`seg_helpers.py` だけは「読み物」ではなく「再利用する道具」です。とくに `clipseg_probs`（`logits→補間→sigmoid` を1つにまとめた中核）、`mask_iou`/`mask_dice`/`best_threshold`（評価の土台）、`build_scene`（GT 付き合成シーン）が、3スクリプトすべての基盤になっています。したがって、まず helper を一読してから 01 へ進むと、各スクリプトが何を import しているかが腑に落ちるはずです。

## 10. 動かし方

このモジュールは、`dl`（torch/torchvision）・`hf`（transformers/timm/safetensors ほか）グループに依存します。CLIPSeg・SAM・Grounding DINO は**いずれも HuggingFace transformers に同梱**されているため、Ultralytics 系の `detect` グループは不要です（評価の IoU/Dice も numpy 自前なので `metrics` も要りません）。CPU だけで完走し、初回のみ3モデルの重みを HuggingFace からダウンロードします（以降はキャッシュから即起動）。プロジェクトルートで、以下を順に実行してください。

```bash
# 依存グループをインストール（初回のみ）
uv sync --group dl --group hf

# 各スクリプトを実行（結果は outputs/23_text_prompt_segmentation/ に保存される）
uv run python lectures/23_text_prompt_segmentation/seg_helpers.py          # 道具箱のスモークテスト＋シーン図
uv run python lectures/23_text_prompt_segmentation/01_clipseg.py
uv run python lectures/23_text_prompt_segmentation/02_grounded_sam.py
uv run python lectures/23_text_prompt_segmentation/03_referring_iou_eval.py

# 章末ミニプロジェクト: 3手法を1枚のベンチで比較（IoU 棒グラフ・パネル・JSON を保存）
uv run python lectures/23_text_prompt_segmentation/mini_project.py

# 実践ユースケース: 文で対象を指して「ぼかす/消す/透過切り出し」する編集ツール
uv run python lectures/23_text_prompt_segmentation/use_case.py
# 対象プロンプトを変える: USECASE_PROMPT="a blue square" を付けて実行

# 演習: まずは TODO を自分で埋める（最初は全部 FAIL だが exit 0）
uv run python lectures/23_text_prompt_segmentation/exercises.py
# どうしても分からない時だけ、模範解答の挙動を見る
SHOW_SOLUTION=1 uv run python lectures/23_text_prompt_segmentation/exercises.py
# 全問の模範解答を一括実行して「正解なら ALL PASS」を確認する
uv run python lectures/23_text_prompt_segmentation/exercises_solutions.py

# （任意）実画像で試す: data/23_text_prompt_segmentation/image.png を置き、
#         prompts.txt に 1 行 1 プロンプトを書くと自動で使われる（GT 無しなら可視化のみ）。
# （任意）SAM 自動マスク生成も見たい: RUN_MASKGEN=1 を付けて 02 を実行
```

実行後は、`outputs/23_text_prompt_segmentation/` の図を解説と照らし合わせてください。とくに `01_clipseg_panel.png`（プロンプトごとの確率ヒートマップとマスク）、`03_clipseg_threshold_sweep.png`（IoU の山なりカーブ、ピークが 0.5 でない）、`02_grounded_sam_panel.png`（検出箱と SAM マスク）の3枚を見れば、本章の要点が視覚的に腑に落ちます。なお図中の文字は、CJK フォントの豆腐（□）を避けるため ASCII にしてあります。また色が反転して見える場合は、合成画像を RGB のまま扱っているか（cv2 経由で BGR が混ざっていないか）を確認してください。

## 11. よくあるエラーと対処（チェックリスト）

最後に、本章でつまずきやすい点を「症状 → 原因 → 対処」の形でまとめます。CLIPSeg/SAM/Grounding DINO 特有の罠が多いので、詰まったらまずここを見てください。

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

この表の項目が、本章で遭遇しがちな不具合のほぼすべてです。とくに上3つ（原寸補間・`padding`・sigmoid）は CLIPSeg の、`threshold` 引数名と `target_sizes` の順序は Grounding DINO（transformers v5）の「あるある」なので、症状を見たら原因を即座に言い当てられるようにしておきましょう。

## 12. まとめ

本章では、**「文で対象を指定して画素マスクを得る」参照セグメンテーション**を、2つの設計で実装しました。**CLIPSeg** は `logits → sigmoid → 閾値` の1段・直接型で、合成シーンの3物体を IoU 0.88〜0.95 で当て、`"a yellow star"`（不在概念）を確率 0.03 で正しく「無い」と判断しました。さらにしきい値スイープでは、**最良点が 0.60〜0.70 にあり、既定の 0.5 が最適でない**ことを曲線で確かめました。一方 **Grounded-SAM** は Grounding DINO → SAM の2段・合成型で、検出箱を SAM が IoU ≈ 0.999 のシャープなマスクへ変換し、**box 閾値を上げると取りこぼし（recall）で検出が 3→1→0 に崩れる**様子を数字で見ました。両者に通底するのは、「**確率マップは後段のしきい値で、2段パイプラインは前段の検出 recall で、品質が決まる**」という勘所です。

ここで身につけた「テキスト→マスク」の発想と、IoU/Dice・しきい値スイープという評価の型は、第24回以降のキャプション／VQA／VLM（言語と画像をより深く結ぶタスク群）へと自然につながっていきます。まずは演習を全問 PASS させ、`03` の「CLIPSeg のピークが 0.5 でない」「box 閾値 0.95 でマスクが消える」という2つの結果を自分の言葉で説明できるようにしてから、次へ進んでください。

---

## 13. 🛠 章末ミニプロジェクト — 3手法を1枚のベンチで競わせる

ここまでに分解して学んだ部品（CLIPSeg の `logits→sigmoid→閾値`、IoU/Dice、しきい値スイープ、Grounding DINO→SAM の2段構成）を**1本のスクリプトに統合**し、「同じシーンを3つの設定で解き、表・図・JSON で勝敗を出す」のが `mini_project.py` です。これは本章の総合課題の完成形であり、まず動かして結果を眺め、次に「なぜこの順位になるか」を本文の知識で説明できれば、この章の内容は身についたと言えます。

**比較する3設定**は、本章の「素朴 → 改善 → 別アプローチ」という学びの流れそのものです。

1. **CLIPSeg @0.5** … 既定しきい値で2値化した「素朴な使い方」（第3節）。
2. **CLIPSeg @best** … 各プロンプトでしきい値を 0.05〜0.95 でスイープし IoU 最大点を採る（第4節）。
3. **Grounded-SAM** … Grounding DINO の box を SAM でマスク化する2段・高精度型（第5節）。

<figure class="lec-fig"><svg viewBox="0 0 640 330" role="img" aria-label="章末ミニプロジェクトの全体フロー 同じ合成シーンを3つの設定で解きIoUで勝敗を出す" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><text x="320" y="26" text-anchor="middle" font-size="14" font-weight="700" fill="#18181b">ミニプロジェクト — 同じシーンを3設定で競わせ IoU で勝敗</text><rect x="235" y="42" width="170" height="52" rx="7" fill="#f4f4f5" stroke="#71717a" stroke-width="2"/><text x="320" y="68" text-anchor="middle" font-size="14.5" font-weight="700" fill="#3f3f46">同じ合成シーン</text><text x="320" y="86" text-anchor="middle" font-size="11.5" fill="#71717a">GT を画素で保持</text><rect x="18" y="158" width="196" height="66" rx="7" fill="#fff7ed" stroke="#f97316" stroke-width="2"/><rect x="222" y="158" width="196" height="66" rx="7" fill="#ffedd5" stroke="#ea580c" stroke-width="2"/><rect x="426" y="158" width="196" height="66" rx="7" fill="#dbeafe" stroke="#2563eb" stroke-width="2"/><text x="116" y="188" text-anchor="middle" font-size="15" font-weight="700" fill="#c2410c">① CLIPSeg @0.5</text><text x="116" y="208" text-anchor="middle" font-size="11.5" fill="#71717a">素朴 / 既定しきい値 0.5</text><text x="320" y="188" text-anchor="middle" font-size="15" font-weight="700" fill="#c2410c">② CLIPSeg @best</text><text x="320" y="208" text-anchor="middle" font-size="11.5" fill="#71717a">改善 / しきい値スイープ</text><text x="524" y="188" text-anchor="middle" font-size="15" font-weight="700" fill="#1d4ed8">③ Grounded-SAM</text><text x="524" y="208" text-anchor="middle" font-size="11.5" fill="#52525b">別アプローチ / 2段</text><rect x="210" y="268" width="220" height="46" rx="7" fill="#fafafa" stroke="#18181b" stroke-width="2"/><text x="320" y="290" text-anchor="middle" font-size="14.5" font-weight="700" fill="#18181b">IoU で比較 → 勝敗</text><text x="320" y="307" text-anchor="middle" font-size="11.5" fill="#52525b">棒グラフ / パネル / JSON</text><line x1="320" y1="94" x2="320" y2="148" stroke="#71717a" stroke-width="2"/><polygon points="320,158 315,148 325,148" fill="#71717a"/><line x1="320" y1="94" x2="126" y2="155" stroke="#71717a" stroke-width="2"/><polygon points="116,158 124,150 127,166" fill="#71717a"/><line x1="320" y1="94" x2="514" y2="155" stroke="#71717a" stroke-width="2"/><polygon points="524,158 513,160 516,150" fill="#71717a"/><line x1="320" y1="224" x2="320" y2="258" stroke="#71717a" stroke-width="2"/><polygon points="320,268 315,258 325,258" fill="#71717a"/><line x1="116" y1="224" x2="278" y2="266" stroke="#71717a" stroke-width="2"/><polygon points="288,268 277,270 280,261" fill="#71717a"/><line x1="524" y1="224" x2="362" y2="266" stroke="#71717a" stroke-width="2"/><polygon points="352,268 360,261 363,270" fill="#71717a"/></svg><figcaption><b>章末ミニプロジェクトの全体像</b>です。入口の <b>1枚の合成シーン</b>(GT を画素で厳密に保持)を、<b>① CLIPSeg @0.5</b>(素朴・既定しきい値)、<b>② CLIPSeg @best</b>(しきい値スイープで改善)、<b>③ Grounded-SAM</b>(2段の別アプローチ)の <b>3設定で同時に解き</b>、最後に <b>IoU で比較して勝敗</b>を <code>mini_project_compare.png</code>・パネル・JSON にまとめます。<b>素朴 → 改善 → 別アプローチ</b> という本章の学びの流れが、そのまま3つの枝になっています。</figcaption></figure>

**成果物**（`outputs/23_text_prompt_segmentation/` に保存）は3つです。`mini_project_compare.png`（オブジェクト×手法の IoU 棒グラフ）、`mini_project_panel.png`（GT／CLIPSeg@best／Grounded-SAM のマスク重ね合わせ）、`mini_project_report.json`（全数値＋**オブジェクト別の勝者**＋**手法別の平均 IoU・勝者数**）。合成シーンでの実測は概ね次のようになります（数値は環境で多少前後します）。

| オブジェクト | CLIPSeg@0.5 | CLIPSeg@best | Grounded-SAM | 勝者 |
| --- | --- | --- | --- | --- |
| red circle | 0.945 | 0.965 | 0.998 | Grounded-SAM |
| blue square | 0.942 | 0.947 | 1.000 | Grounded-SAM |
| green triangle | 0.884 | 0.936 | 1.000 | Grounded-SAM |
| **平均 IoU** | **0.924** | **0.950** | **0.999** | — |

読み取るべきは2点です。第一に **CLIPSeg@best > CLIPSeg@0.5**――しきい値を選び直すだけで平均 IoU が上がり、「0.5 は最適でない」（第4節）ことが数字で裏付けられます。第二に **Grounded-SAM が全オブジェクトで勝つ**――SAM の境界の鋭さが IoU 差として効く（第5節）ことが、勝者数 3-0 という形で現れます。**課題**: (a) `mini_project.py` を読み、`eval_clipseg` がどこで `best_threshold` を呼んでいるか、`best_sam_mask_for_gt` がなぜ「GT ごとに IoU 最大の SAM マスク」を選ぶのかを説明する。(b) `_SCENE` に4つ目の図形（例: 黄色い星）を `seg_helpers.py` に追加し、ベンチがそのまま4オブジェクトに拡張されることを確かめる。(c) 勝敗が変わるよう、CLIPSeg に有利な「滲んだ境界でも良い」設定（例: Dice を主指標にする）を考え、`winner` の判定軸を IoU から Dice に変えるとどうなるかを試す。

## 14. ✅ 到達チェックリスト

次の問いに「コードのどの行がそれを担うか」まで指させれば、この章は合格です。手を動かして確認してください。

- [ ] 参照セグメが、セマンティックセグメ（固定クラス）・SAM（点/箱）・オープン語彙検出（箱）と**何が違うか**を1文で言える（任意テキストで指定し、出力は画素マスク）。
- [ ] CLIPSeg の出力を **`sigmoid`** で確率化する理由（画素ごとの独立判定）を、`softmax` を使わない理由とセットで説明できる。
- [ ] CLIPSeg のロジット（352×352）を **原寸へ補間してから** sigmoid する必要性を述べ、`F.interpolate(..., size=(h,w))` を書ける。
- [ ] **IoU と Dice** の式を諳んじ、同じ重なりで **Dice ≥ IoU** になる理由を説明できる。
- [ ] しきい値スイープで **最良点が 0.5 でない**ことを図で示し、「低すぎ＝塗り過ぎ／高すぎ＝塗り残し」を語れる。
- [ ] Grounding DINO のテキストが **小文字＋ピリオド区切り**であるべき理由と、`post_process_*` の **`target_sizes` が (H, W) 順**であることを知っている。
- [ ] SAM が box ごとに**候補マスクを3枚返す**こと、`iou_scores` の argmax を採る理由、`post_process_masks` で原寸化する必要を説明できる。
- [ ] **box 閾値を上げると検出が 3→1→0 に崩れる**（recall が最終マスクの上限を決める）ことを、CLIPSeg のしきい値（切り方を変えるだけ）との違いとして対比できる。
- [ ] `pipeline("mask-generation")`（プロンプト無し・網羅）と参照セグメ（プロンプト有り・特定対象）の**目的が逆**であることを言える。
- [ ] 演習 ex8（貪欲マッチング）と ex9（AP 補間）で、**1つの GT に TP は1つだけ**・**precision を右から単調化して面積を取る**という検出評価の核を実装できる。

## ✍️ 演習問題

演習は `exercises.py` に TODO 形式で入っています。各 TODO を実装し `uv run python lectures/23_text_prompt_segmentation/exercises.py` を実行すると自己採点できます（`exercises_solutions.py` が解答）。

1. CLIPSeg のロジットを sigmoid で確率化し、`threshold` 以上を前景にした bool マスクを返す（`ex1_logits_to_mask` の TODO）。
2. 2値マスクの IoU = |∩| / |∪| を返す（和集合が空なら 1.0）（`ex2_mask_iou` の TODO）。
3. 2値マスクの Dice = 2|∩| / (|P|+|G|)（= F1）を返す（分母が 0 なら 1.0）（`ex3_mask_dice` の TODO）。
4. 確率マップをしきい値でスイープし、IoU 最大のしきい値とその IoU を `(best_threshold, best_iou)` で返す（`ex4_best_threshold` の TODO）。
5. 2つの box `[x0, y0, x1, y1]` の IoU を返す（交差は 0 でクランプ）（`ex5_box_iou` の TODO）。
6. 複数オブジェクトの IoU 平均（mIoU）を返す（空リストなら 0.0）（`ex6_mean_iou` の TODO）。
7. 予測マスクの pixel 単位 precision / recall を `(precision, recall)` で返す（`ex7_pixel_pr` の TODO）。
8. 検出 box を score 降順に GT へ貪欲マッチングし、`(TP, FP, FN)` を返す（1 つの GT に TP は 1 つだけ）（`ex8_greedy_match` の TODO）。
9. PR 曲線を全点補間して Average Precision (AP) を返す（`n_gt` が 0 なら 0.0）（`ex9_average_precision` の TODO）。

## 15. ❓ よくある落とし穴・FAQ・デバッグ

第11節の「症状→原因→対処」表が**実行時エラー**の早見表なら、ここは**概念のつまずき**の Q&A です。両方を行き来すると理解が固まります。

**Q1. CLIPSeg の確率を `softmax` で正規化したら、どのプロンプトも合計1になって変です。**
A. それこそが `softmax` の罠です。`softmax` は「候補のどれか1つ」に確率を寄せる**排他的**な正規化で、参照セグメには不向きです。CLIPSeg は**各画素を独立に**「この文に合うか」を判定するので、**`sigmoid`** が正しいのです。だからこそ「シーンに無い概念（`a yellow star`）」を投げても全画素が低いまま＝しきい値で「無い」と判定できます（第3節）。これが `softmax` だと、無理やりどこかが高くなり、不在を表現できません。

**Q2. 全プロンプトで IoU がほぼ 0 です。**
A. まずは原寸補間を疑ってください。CLIPSeg の生出力は 352×352 固定なので、`F.interpolate` で原寸 `(h, w)` に上げないまま GT と重ねると、画像とマスクの**格子がずれて**重なりが消えます。次に GT 整合（`seg_helpers.build_scene` の GT は RGB のまま作るので、cv2 の `imread/imwrite` を挟んで BGR が混ざっていないか）を確認します。いずれにせよ、`outputs/.../00_scene_and_gt.png` で GT が想定どおりの位置かを目視するのが最短です。

**Q3. Grounding DINO が合成画像で何も検出しません（検出数 0）。**
A. 合成画像はテクスチャが乏しく、実写を前提とする Grounding DINO では確信度が出にくい**ドメインギャップ**が原因で、これは異常ではありません。本章のスクリプト（`02`・`mini_project`）は**検出ゼロを検知したら GT の外接 box を SAM に渡すフォールバック**に切り替えるので、SAM 単体の品質は必ず確認でき、かつ exit 0 で完走します。実写で試したい場合は、`data/23_text_prompt_segmentation/image.png` を置いてください。あわせて、検出の作法（小文字＋ピリオド区切り、`threshold` を下げる）も見直すとよいでしょう。

**Q4. `box_threshold=` を渡したら `TypeError` になりました。**
A. transformers v5 で `post_process_grounded_object_detection` の引数名が **`threshold`** に変わりました（旧 `box_threshold`）。`text_threshold` は別引数として残ります。本章のコードは `threshold=...` で統一しています。

**Q5. `mini_project` で Grounded-SAM が常に勝ちます。CLIPSeg が劣るのはモデルが悪いから？**
A. いいえ、これは**設計の違い**です。CLIPSeg は1モデルで「文→確率マップ」を出すため境界がソフトに滲み、IoU では SAM のシャープな境界に一歩譲ります。しかし CLIPSeg には、**軽量・高速**で、`sky` のような**非物体の領域**も塗れ、しきい値で後から挙動を調整できるという強みがあります（第8節の表）。したがって「速度・手軽さ・領域概念」を重視する場面なら、むしろ CLIPSeg が正解です。勝敗は**指標と要件しだい**であり、IoU 単独で優劣を決めないようにしましょう。

**Q6. box 閾値を上げると平均マスク IoU が `0.000` に落ちました。バグですか？**
A. 仕様どおりです。閾値 0.95 では Grounding DINO が**1つも検出しない**ため、SAM に渡す箱が無く、マスクを作れません（IoU=0）。これは Grounded-SAM の本質――**段1の検出 recall が最終マスクの網羅性の上限**――を示す現象で、第6節で意図的に観察させています。CLIPSeg のしきい値が「1枚の確率マップの切り方」を変えるだけなのとは対照的に、box 閾値は「そもそも対象を見つけるか否か」を左右するのです。

**Q7. 演習 ex8（貪欲マッチング）で TP が GT 数を超えてしまいます。**
A. 「**1つの GT に対応できる検出は1つだけ**」を破っています。score 降順に走査し、TP にした GT を「使用済み」フラグで除外してください。同じ GT に2つ目の検出が当たっても、それは TP ではなく **FP**（二重カウントは検出評価の典型バグ）です。ex9 の AP も、この「未使用 GT への一意マッチ」を前提に PR 曲線を作ります。

**Q8. CPU で推論が極端に遅い／固まります。**
A. CPU では `float16`/`half` が遅い・未対応のことが多いので **`float32` のまま**使います（本章は明示的に dtype 指定せず float32）。`pipeline("mask-generation")` は**点グリッドの数だけ SAM を走らせる**ので最も重く、本章では既定で概念紹介に留め、`RUN_MASKGEN=1` のときだけ `points_per_side=8` に絞ります。初回はモデル重みの DL も入るので、Docker ではキャッシュ（`HF_HOME` / `~/.cache/huggingface`）をボリュームマウントして再 DL を避けます。

## 16. 🚀 発展トピック・参考

本章の2手法は「テキスト→マスク」の入口です。現場や研究では、次のような方向へ広がっていきます。

- **より新しい SAM 系**: `SAM2`（動画対応・高速）・`MobileSAM`/`SlimSAM`（CPU で数秒の軽量版）への置換は、`seg_helpers.load_sam` の model_id を差し替えるだけで試せます。Grounded-SAM の段2を SAM2 にすると境界と速度が改善します。
- **Grounded-SAM 2 / 検出器の差し替え**: 段1を `Grounding DINO 1.5`・`OWLv2`・自前学習の検出器に替えると語彙・精度が伸びます。**モジュール性**が2段構成の最大の利点で、「検出は得意なモデル、切り出しは SAM」と分業させる設計思想が核です。
- **推論セグメンテーション（reasoning segmentation）**: `LISA`・`SEEM` は「**右側の一番大きい果物**」のような**推論を要する参照表現**を LLM/VLM と結合して解きます。CLIPSeg が苦手な「言語的な含意」を扱う次世代の参照セグメです。
- **参照セグメ用データセットと指標**: `RefCOCO/+/g`・`PhraseCut` が定番ベンチで、評価には本章の IoU に加え **cIoU（cumulative IoU）**・**gIoU（generalized IoU）**・**Precision@X**（IoU≥X を成功とする割合）・**boundary IoU**（境界帯での一致）が使われます。境界の質を測りたいときは boundary IoU を足すと CLIPSeg と Grounded-SAM の差がより鮮明になります。
- **ハイブリッド**: Grounding DINO の box を CLIPSeg の条件に混ぜる、SAM のマスクで CLIPSeg の確率を後段リファインする、といった組み合わせも有効です。本章のコードは部品を `seg_helpers` に分離してあるので、組み替えの土台にできます。

参考リンク（モデルカード/ドキュメント）: CLIPSeg `CIDAS/clipseg-rd64-refined`、SAM `facebook/sam-vit-base`、Grounding DINO `IDEA-Research/grounding-dino-tiny`（いずれも HuggingFace transformers 同梱）。transformers の `image-segmentation` / `mask-generation` / `zero-shot-object-detection` パイプラインのドキュメントも併読すると、本章のスクリプトが「正準フローのどこを手書きしているか」が見通せます。

## 17. 💡 実践ユースケース集

ここまでは「文 → マスク」とその評価を学んできました。最後に、**作ったマスクを“どう使うか”**――参照セグメが現場で価値を生む応用を、3つ挙げます。いずれも本章の `clipseg_probs`（`logits→補間→sigmoid`）＋しきい値という同じ部品の上に乗っています。共通の勘所は、**(1) マスクは硬い 0/1 ではなく境界を羽根付き(feather)にしてから合成する**こと、**(2) CLIPSeg の確信度が低い合成画像では閾値で何も選べないことがあるので、上位確率で救済して“必ず何かを選ぶ”フォールバックを持つ**こと、**(3) 加工は元配列を壊さず新配列を返す**こと、の3点です。

- **ことばで匿名化（プライバシー・ぼかし）**: 「文で対象を指してその領域だけぼかす」用途。何に使うか=顔・ナンバープレート・名札・住所看板などを、座標を手で指定せず `"a face"` のような文で選んで一括ぼかし。作り方の要点=`clipseg_probs` の確率マップを羽根付きαにし、`cv2.GaussianBlur` した画像と元画像を `α合成`（マスク内=ぼかし／外=元）。注意=CLIPSeg のマスクは輪郭が粗いので、漏れを避けたい匿名化では**マスクを少し膨張(dilate)**してから使う／確信度が低い対象は閾値を下げる。厳密さが要る本番は Grounded-SAM で境界を締めると安全。
- **不要物の除去（簡易オブジェクト除去 / inpaint）**: 「文で指した物を消して背景で埋める」用途。何に使うか=写り込んだ通行人・電線・ゴミなどの除去。作り方の要点=2値マスクを `cv2.inpaint(..., INPAINT_TELEA)` の inpaint マスクに渡し、周囲画素から塗り直す（塗り残し防止に dilate）。注意=`cv2.inpaint` は**小さな領域向け**で、大きな穴や複雑背景は不自然になりがち。本格的な除去は拡散モデル(diffusers)のインペイントに差し替える。マスクが対象より小さいと“フチが残る”ので、除去用途では少し広めに取る。
- **対象の切り出し（背景透過・素材化）**: 「文で指した対象を背景透過 PNG で抜き出す」用途。何に使うか=商品写真の背景除去、ステッカー/素材作り、合成用カットアウト。作り方の要点=羽根付きαを RGBA の**アルファチャンネル**に入れて `Image.fromarray(rgba, "RGBA").save(...)`。注意=境界をきれいに出したいなら CLIPSeg より **SAM の鋭いマスク**が向く（Grounded-SAM の段2を流用）。透過の確認は市松模様に重ねて目視するとよい。

### 動く出発点: `use_case.py`（文で選んで「ぼかす/消す/抜き出す」）

上の3応用を**1本にまとめた最小ツール**が `use_case.py` です。`mini_project.py`（手法を IoU で競わせる評価ベンチ）とは別物で、こちらは**GT も IoU も使わず**、1枚の入力から `blur`（ぼかし）/ `remove`（inpaint 除去）/ `cutout`（透過 PNG）の3成果物を一度に出力します。

<figure class="lec-fig"><svg viewBox="0 0 660 300" role="img" aria-label="use_case.pyの処理フロー 入力をCLIPSegでマスク化しblur remove cutoutの3編集に枝分かれして成果物を出す" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><text x="330" y="26" text-anchor="middle" font-size="14" font-weight="700" fill="#18181b">use_case.py — 文で選んで「ぼかす / 消す / 抜き出す」</text><rect x="20" y="120" width="110" height="60" rx="7" fill="#f4f4f5" stroke="#71717a" stroke-width="2"/><text x="75" y="146" text-anchor="middle" font-size="14" font-weight="700" fill="#3f3f46">入力 1枚</text><text x="75" y="166" text-anchor="middle" font-size="11" fill="#71717a">合成 or data/</text><rect x="180" y="118" width="140" height="64" rx="7" fill="#dbeafe" stroke="#2563eb" stroke-width="2"/><text x="250" y="144" text-anchor="middle" font-size="14" font-weight="700" fill="#1d4ed8">CLIPSeg でマスク</text><text x="250" y="164" text-anchor="middle" font-size="11" fill="#52525b">閾値 + 羽根付きα</text><rect x="400" y="44" width="240" height="58" rx="7" fill="#fff7ed" stroke="#f97316" stroke-width="2"/><rect x="400" y="121" width="240" height="58" rx="7" fill="#ffedd5" stroke="#ea580c" stroke-width="2"/><rect x="400" y="198" width="240" height="58" rx="7" fill="#dbeafe" stroke="#2563eb" stroke-width="2"/><text x="416" y="70" font-size="14" font-weight="700" fill="#c2410c">blur — ぼかし</text><text x="416" y="90" font-size="11.5" fill="#52525b">use_case_blur.png</text><text x="416" y="147" font-size="14" font-weight="700" fill="#c2410c">remove — 除去(inpaint)</text><text x="416" y="167" font-size="11.5" fill="#52525b">use_case_removed.png</text><text x="416" y="224" font-size="14" font-weight="700" fill="#1d4ed8">cutout — 透過PNG</text><text x="416" y="244" font-size="11.5" fill="#52525b">use_case_cutout.png</text><text x="155" y="140" text-anchor="middle" font-size="10.5" fill="#3f3f46">文で指定</text><line x1="130" y1="150" x2="174" y2="150" stroke="#71717a" stroke-width="2"/><polygon points="180,150 170,155 170,145" fill="#71717a"/><line x1="320" y1="150" x2="393" y2="80" stroke="#71717a" stroke-width="2"/><polygon points="400,73 396,84 389,76" fill="#71717a"/><line x1="320" y1="150" x2="390" y2="150" stroke="#71717a" stroke-width="2"/><polygon points="400,150 390,155 390,145" fill="#71717a"/><line x1="320" y1="150" x2="393" y2="220" stroke="#71717a" stroke-width="2"/><polygon points="400,227 389,224 396,216" fill="#71717a"/></svg><figcaption><b><code>use_case.py</code> の処理フロー</b>です。<b>入力1枚</b>(合成シーン、または <code>data/</code> の実画像)を <b>CLIPSeg でマスク化</b>(確率を閾値で切り、境界を羽根付きαに)し、そのマスクから <b>3つの編集に枝分かれ</b>します ―― <b>blur</b>(<code>GaussianBlur</code> をα合成してぼかす)、<b>remove</b>(<code>cv2.inpaint</code> で消して背景を埋める)、<b>cutout</b>(αチャンネルへ入れ <b>背景透過 RGBA</b> で抜き出す)。<b>GT も IoU も使わず</b>、文で指した対象を加工した3成果物を一度に出力します。</figcaption></figure>

```bash
# 実行（合成シーンなら即動く。結果は outputs/23_text_prompt_segmentation/ に保存）
uv run python lectures/23_text_prompt_segmentation/use_case.py

# 対象プロンプトを変える（既定は prompts.txt の1行目 or 合成シーンの先頭 "a red circle"）
USECASE_PROMPT="a blue square" uv run python lectures/23_text_prompt_segmentation/use_case.py
```

- **実データの置き方**: `data/23_text_prompt_segmentation/image.png` を置くとそれを入力に使い、`data/23_text_prompt_segmentation/prompts.txt` の1行目を対象プロンプトにします（`USECASE_PROMPT` で上書き可）。画像が無ければ合成シーンに自動フォールバックし、**ネットもデータも無しで必ず exit 0**。合成画像は確信度が低く閾値に届かないことがあるため、その場合は**上位確率で救済**して必ず何かを選びます（検出/セグメは合成だと反応が弱くても問題ありません。実写を `data/` に置けば、そのまま匿名化/除去/切り出しの実用ツールになります）。
- **出力**: `use_case_blur.png` / `use_case_removed.png` / `use_case_cutout.png`（背景透過 RGBA）/ `use_case_panel.png`（input・mask・3編集を横並び）/ `use_case_report.json`（プロンプト・閾値・選択画素数・各パラメータ・出力先）。
- **拡張アイデア**: 閾値・ぼかし強度・inpaint 半径を `argparse` で引数化して CLI 化／マスクを SAM(`seg_helpers.load_sam`) で境界リファインしてから加工（Grounded-SAM 流用で輪郭が鋭くなる）／`remove` を拡散モデルのインペイントに差し替え／`prompts.txt` を複数行にしてバッチ匿名化／動画の各フレームへ適用して「文で指した対象だけモザイク」の動画を作る。

---

> 本教材で参照・検証したライブラリとバージョン（2026-06-12 時点の安定版で動作確認）:
> Python 3.12 ／ torch 2.12.0+cpu ／ torchvision 0.27.0+cpu ／ transformers 5.11.0 ／ huggingface-hub 1.18.0 ／ timm 1.0.27 ／ safetensors 0.8.0 ／ numpy 2.4.6 ／ Pillow 12.2.0 ／ matplotlib 3.10.9 ／ opencv-python-headless 4.13.0（合成シーンの描画）
> 使用モデル: `CIDAS/clipseg-rd64-refined`（CLIPSeg）／ `facebook/sam-vit-base`（SAM）／ `IDEA-Research/grounding-dino-tiny`（Grounding DINO）。いずれも HuggingFace transformers 同梱で、初回のみ重みを取得しキャッシュします。Grounded-SAM の発展（より広い語彙の検出器への差し替え）や SAM2/MobileSAM への置換は、`seg_helpers.load_*` を差し替えるだけで試せます。
