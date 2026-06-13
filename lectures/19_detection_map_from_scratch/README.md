# 第19回 ★物体検出 mAP の自力実装 — IoU → マッチング → PR 曲線 → AP 補間 → mAP

> トラック: **評価指標** ／ レベル: **中級** ／ 依存グループ: `dl`（torch / torchvision）・`metrics`（pycocotools）。画像モデルもネット接続も不要で、合成した「GT ボックスと予測ボックス（スコア付き）」だけで完結します。

## 🎯 この章のゴール

この章を終えたとき、あなたは「物体検出の mAP は、ブラックボックスのライブラリ関数ではなく、IoU と並べ替えと累積和の組み合わせにすぎない」という確信を持てるようになります。というのも、第14回で身につけた「混同行列 → PR 曲線 → 面積」という分類評価の骨格は、実は検出にもそのまま効くからです。違うのはたった一点、「予測が "場所"（ボックス）を持つため、まず IoU でどの正解（GT）を当てたのかを決める対応付けが要る」ことだけです。そして、その対応付けの手続き —— confidence 降順で GT へ貪欲にマッチングし、1つの GT を二度数えない —— を自分の手で書けるようになることが、この章の第一の到達点です。

対応付けができたら、次は TP/FP の列を confidence 降順に累積して precision/recall の軌跡（PR 曲線）を作り、その "面積" を AP に要約します。ここで重要なのは「AP の値は補間方式に依存する」という点です。同じ PR 曲線でも、PASCAL VOC 2007 の 11 点、VOC2010+ の全点、COCO の 101 点では、少しずつ違う数字が出ます。さらに、IoU 閾値を 0.50:0.05:0.95 で動かして平均すれば COCO の主指標 mAP@[.5:.95] になり、IoU=0.5 だけなら mAP@0.5（旧 PASCAL 風）になります。これらを **numpy だけで一から組み上げ、pycocotools の COCOeval と小数点以下まで一致することを `assert` で確かめる** —— ここまでが本章の合格ラインです。

到達点を一言でいえば、**GT と予測（スコア付き）さえあれば、IoU・マッチング・PR・AP・mAP を AI 補助なしで書け、その値が COCO 公式実装と一致することを自分で検証できる**こと。こうして mAP を「ライブラリの戻り値」ではなく「式と手順」で語れるようになると、検出モデルを比較するときに数字のズレ（補間方式違い・ソート漏れ・bbox 形式違い）を一目で見抜けるようになります。

---

## 1. IoU — 「どの予測がどの GT を当てたか」を測る物差し

分類の評価は「正解ラベル」と「予測ラベル」を突き合わせれば済みました。しかし検出では、予測が画像上の **ボックス**（位置と大きさ）を持ちます。そのため「この予測ボックスは、本当にあの GT を当てているのか？」を判定する物差しが必要になり、それが **IoU（Intersection over Union）** です。IoU は `交差面積 / 和集合面積` で定義され、2 つのボックスがどれだけ重なっているかを 0〜1 で表します。完全一致なら 1、まったく重ならなければ 0 です。検出評価では「IoU が閾値（例えば 0.5）以上なら同じ物体を指している」とみなします。

<figure class="lec-fig"><svg viewBox="0 0 600 300" role="img" aria-label="IoUは2つのボックスの交差面積を和集合面積で割った0から1の値" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><text x="300" y="36" text-anchor="middle" font-size="17" font-weight="700" fill="#18181b">IoU = 交差面積 ÷ 和集合面積</text><rect x="70" y="80" width="160" height="130" fill="#ffffff" stroke="#16a34a" stroke-width="2.5"/><rect x="150" y="120" width="160" height="130" fill="none" stroke="#2563eb" stroke-width="2.5"/><rect x="150" y="120" width="80" height="90" fill="#ffedd5" stroke="#c2410c" stroke-width="2"/><text x="74" y="74" font-size="14" font-weight="700" fill="#15803d">GT（正解）</text><text x="246" y="270" font-size="14" font-weight="700" fill="#2563eb">予測</text><text x="190" y="170" text-anchor="middle" font-size="13" font-weight="700" fill="#c2410c">交差</text><rect x="370" y="120" width="200" height="20" fill="#f4f4f5" stroke="#d4d4d8" stroke-width="1.5"/><rect x="370" y="120" width="42" height="20" fill="#f97316"/><text x="370" y="160" text-anchor="middle" font-size="12.5" fill="#52525b">0</text><text x="570" y="160" text-anchor="middle" font-size="12.5" fill="#52525b">1</text><text x="470" y="188" text-anchor="middle" font-size="15" font-weight="700" fill="#c2410c">この例 ≈ 0.21</text></svg><figcaption>IoU（Intersection over Union）は、2 つのボックスの <b>交差面積 ÷ 和集合面積</b> で重なり具合を <b>0〜1</b> で表します。図の例（<b>GT</b> 緑 × <b>予測</b> 青）は交差が小さく <b>IoU ≈ 0.21</b>。完全一致なら 1、まったく重ならなければ 0 で、検出評価では <code>IoU ≥ 0.5</code> など閾値以上を「同じ物体を当てた」とみなします。</figcaption></figure>

実装は驚くほど単純です。交差矩形の左上は「2 つの左上の大きい方」、右下は「2 つの右下の小さい方」を取り、幅・高さが負（重なりなし）なら 0 にクリップして交差面積を求めます。和集合は `面積A + 面積B − 交差` です。`det_helpers.py` の `box_iou_numpy()` はこれをブロードキャストで全ペア一括計算し、`01_iou_matching.py` で `torchvision.ops.box_iou` と完全一致（最大差 0.0）することを確認します。下のスニペットがその核心です。

```python
lt = np.maximum(a[:, None, :2], b[None, :, :2])  # 交差の左上 = それぞれの左上の大きい方
rb = np.minimum(a[:, None, 2:], b[None, :, 2:])  # 交差の右下 = それぞれの右下の小さい方
wh = (rb - lt).clip(0)                            # 負（重なりなし）は 0 に
inter = wh[..., 0] * wh[..., 1]
union = area_a[:, None] + area_b[None, :] - inter
iou = inter / union
```

ここで必ず押さえてほしいのが **bbox の座標形式** です。本講座が内部で使うのは `xyxy`（左上 x, 左上 y, 右下 x, 右下 y）ですが、COCO の JSON では `xywh`（左上 x, 左上 y, 幅, 高さ）が使われ、さらに DETR など一部のモデルは `cxcywh`（中心 x, 中心 y, 幅, 高さ）を内部で持ちます。この 3 つを取り違えると IoU が壊れ、評価が丸ごと無意味になってしまいます。そこで本章では「IoU・マッチングは xyxy で行い、pycocotools へ渡すときだけ xywh へ変換する」という流儀を徹底し、その変換は `to_coco_gt()` / `to_coco_dt()` が担います。

## 2. confidence 降順の貪欲マッチング — TP / FP / FN と二重カウント防止

IoU で「重なり度」が測れたら、次は予測と GT を 1 対 1 に **対応付け** ます。手順はこうです。まず予測を **confidence（スコア）の高い順** に並べ、上から 1 件ずつ「まだ使われていない GT のうち、IoU が閾値以上で最大のもの」へ割り当てていきます。割り当てられれば **TP（真陽性）**、見つからなければ **FP（偽陽性）** で、最後まで誰にも当てられなかった GT が **FN（偽陰性＝検出漏れ）** です。スコアの高い予測ほど優先的に GT を取れる、という点が「confidence 依存」の核心で、ここを外すと後段の PR 曲線が別物になってしまいます。

<figure class="lec-fig"><svg viewBox="0 0 600 300" role="img" aria-label="confidence降順の貪欲マッチング。高スコア予測がGTを取りTP、同じGTを狙う低スコア予測はFP、当てられないGTはFN" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><text x="170" y="40" text-anchor="middle" font-size="14" font-weight="700" fill="#3f3f46">予測（スコア降順 ↓）</text><text x="475" y="40" text-anchor="middle" font-size="14" font-weight="700" fill="#3f3f46">正解 GT</text><rect x="55" y="70" width="230" height="50" rx="7" fill="#2563eb"/><text x="170" y="101" text-anchor="middle" font-size="15" font-weight="700" fill="#ffffff">P0  conf 0.9  → TP</text><rect x="55" y="160" width="230" height="50" rx="7" fill="#dc2626"/><text x="170" y="191" text-anchor="middle" font-size="15" font-weight="700" fill="#ffffff">P1  conf 0.8  → FP</text><rect x="405" y="70" width="140" height="50" rx="7" fill="#ffffff" stroke="#16a34a" stroke-width="2.5"/><text x="475" y="101" text-anchor="middle" font-size="15" font-weight="700" fill="#15803d">GA</text><rect x="405" y="180" width="140" height="50" rx="7" fill="#fafafa" stroke="#dc2626" stroke-width="2.5"/><text x="475" y="211" text-anchor="middle" font-size="14" font-weight="700" fill="#dc2626">GB → FN</text><line x1="289" y1="95" x2="399" y2="95" stroke="#2563eb" stroke-width="2.5"/><polygon points="405,95 395,90 395,100" fill="#2563eb"/><line x1="289" y1="178" x2="394" y2="116" stroke="#dc2626" stroke-width="2" stroke-dasharray="5 3"/><text x="392" y="132" text-anchor="middle" font-size="17" font-weight="700" fill="#dc2626">×</text></svg><figcaption>予測を <b>confidence（スコア）の高い順</b> に処理し、各予測を「まだ使われていない GT のうち IoU 最大」へ貪欲に割り当てます。<b>P0（0.9）</b>は GA を取って <b>TP</b>。同じ GA を狙う <b>P1（0.8）</b>は GA が使用済みのため <b>FP</b> になり、これが <b>二重カウント防止</b>です。どの予測にも当てられなかった <b>GB</b> は <b>FN</b>（検出漏れ）になります。</figcaption></figure>

このアルゴリズムで一番大事なのが **二重カウントの防止** です。1 つの GT に複数の予測が重なったとき、TP として数えてよいのは **1 つだけ**（最もスコアの高い予測）で、残りはすべて FP になります。もしこれを許すと、同じ物体に箱を何個も重ねるだけで TP が水増しされ、precision が不正に上がってしまうからです。`01_iou_matching.py` の `demo_double_counting()` は、同じ GT に重なる 2 つの予測（スコア 0.9 と 0.8）を入れ、高スコアの方だけが TP・もう一方は FP になることを最小例で示します。実行ログは次のように出ます。

```text
[二重カウント防止] 同じ GT に2予測 → pred0(score0.9)=TP, pred1(score0.8)=FP  （TP は1つだけが正しい）
```

`01_iou_matching.py` は、12 枚の合成画像のうち 1 枚を取り、カテゴリごとにこのマッチングを実行して GT（緑）・TP（青）・FP（赤）を 1 枚の図 `01_matching.png` に描きます。このとき、カテゴリを跨いだマッチングは決して起きない（円の予測が四角の GT を当てることはない）点にも注意してください。実行すると「GT=8 TP=7 FP=5 FN=1 → precision=0.583 recall=0.875」のように、**1 つのスコア閾値・1 枚の画像** での precision/recall が出ます。ただし、これはまだ「点」にすぎません。スコア閾値を上下に動かせば precision と recall は連動して動く —— その軌跡を曲線にするのが、次の節の仕事です。

## 3. PR 曲線の構築 — 累積和で precision/recall の列を作る

第14回で見たとおり、precision/recall は「あるスコア閾値で予測を採用/棄却に固めた後」の値です。検出でも考え方は同じで、スコア閾値を高い方から下げていくと、採用される予測が増え、TP も FP も積み上がっていきます。この「積み上がり」を表現するのが **累積和（cumsum）** です。予測をスコア降順に並べ、`tp_cum = cumsum(tp)`、`fp_cum = cumsum(1-tp)` を取れば、各時点での `recall = tp_cum / (全GT数)`、`precision = tp_cum / (tp_cum + fp_cum)` が一気に求まります。これが PR 曲線の生データになります。

ここで、検出ならではの注意が 2 つあります。1 つ目は、**マッチング（TP/FP 判定）は画像ごとに行うが、PR 曲線は画像をまたいで積む** という二段構えです。GT の「使用済み」状態は 1 枚の画像内で閉じる一方、precision/recall を積むときには全画像の予測をまとめ、改めてスコア降順に並べ替えます。2 つ目は、**recall の分母がカテゴリ全体の GT 数** だという点です。検出漏れ（FN）の GT には対応する予測が存在しないため TP 列には現れませんが、分母に入ることで recall を正しく下げます。この流れは `02_pr_ap_interpolation.py` の `build_pr()` が実装しています。

```python
order = np.argsort(-scores_all, kind="stable")  # 画像横断で改めてスコア降順に（安定ソート）
tp_cum = np.cumsum(tp_all[order])
fp_cum = np.cumsum(1.0 - tp_all[order])
recall = tp_cum / n_gt_total                     # 分母はカテゴリ全体の GT 数（FN も含む）
precision = tp_cum / (tp_cum + fp_cum)
```

ここで、ソートに `kind="stable"`（安定ソート）を指定している点に注目してください。スコアが同点の予測があると、並べ替えの順序次第で TP/FP の積まれ方が変わり、pycocotools と値がズレてしまいます。COCO 公式も `mergesort`（安定ソート）を使っているので、これに合わせておくのが鉄則です。本章の合成データはスコアが連続値で同点がほぼ無いため影響は小さいものの、実データでは同点が頻発するので、安定ソートは "おまじない" ではなく必須です。

## 4. AP 補間の 3 方式 — 11 点 / 全点 / COCO 101 点

PR 曲線を 1 つの数に要約したものが **AP（Average Precision）** です。ところが「曲線の下の面積」と一口に言っても、その求め方は歴史的に複数あり、**同じ PR 曲線でも方式によって少し違う値** になります。これを知らないと、論文や他チームの mAP と突き合わせたときに「なぜか 0.01 ずれる」と悩むことになります。そこで本章では、代表的な 3 方式を `02_pr_ap_interpolation.py` で実装し、実測して比べます。どの方式にも共通する前処理が **単調包絡**（`np.maximum.accumulate(precision[::-1])[::-1]`）で、これは recall を上げても precision は上がらないように、曲線のギザギザを右から均す操作です。

<figure class="lec-fig"><svg viewBox="0 0 520 340" role="img" aria-label="PR曲線の生データ、単調包絡、AP標本点。同じ曲線でも11点や101点で要約すると値が少し変わる" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><polygon points="70,50 130,50 130,100.6 190,100.6 190,123.6 250,123.6 250,137.4 310,137.4 310,155.8 370,155.8 370,172 430,172 430,188 470,188 470,280 70,280" fill="#ffedd5" opacity="0.6"/><line x1="70" y1="288" x2="70" y2="44" stroke="#71717a" stroke-width="1.8"/><polygon points="70,38 65,48 75,48" fill="#71717a"/><line x1="62" y1="280" x2="486" y2="280" stroke="#71717a" stroke-width="1.8"/><polygon points="492,280 482,275 482,285" fill="#71717a"/><polyline points="70,50 130,50 130,119 190,100.6 190,142 250,123.6 250,153.5 310,137.4 310,169.6 370,155.8 370,183.4 430,172 430,192.6 470,188" fill="none" stroke="#71717a" stroke-width="1.6"/><polyline points="70,50 130,50 130,100.6 190,100.6 190,123.6 250,123.6 250,137.4 310,137.4 310,155.8 370,155.8 370,172 430,172 430,188 470,188" fill="none" stroke="#ea580c" stroke-width="3"/><circle cx="70" cy="50" r="4" fill="#c2410c"/><circle cx="110" cy="50" r="4" fill="#c2410c"/><circle cx="150" cy="100.6" r="4" fill="#c2410c"/><circle cx="190" cy="100.6" r="4" fill="#c2410c"/><circle cx="230" cy="123.6" r="4" fill="#c2410c"/><circle cx="270" cy="137.4" r="4" fill="#c2410c"/><circle cx="310" cy="137.4" r="4" fill="#c2410c"/><circle cx="350" cy="155.8" r="4" fill="#c2410c"/><circle cx="390" cy="172" r="4" fill="#c2410c"/><circle cx="430" cy="172" r="4" fill="#c2410c"/><circle cx="470" cy="188" r="4" fill="#c2410c"/><line x1="322" y1="66" x2="352" y2="66" stroke="#71717a" stroke-width="1.6"/><text x="358" y="70" font-size="12.5" fill="#52525b">生 PR（raw）</text><line x1="322" y1="90" x2="352" y2="90" stroke="#ea580c" stroke-width="3"/><text x="358" y="94" font-size="12.5" fill="#c2410c">単調包絡</text><circle cx="337" cy="112" r="4" fill="#c2410c"/><text x="358" y="116" font-size="12.5" fill="#c2410c">11/101 点標本</text><text x="34" y="165" text-anchor="middle" font-size="13" fill="#3f3f46" transform="rotate(-90 34 165)">precision</text><text x="270" y="308" text-anchor="middle" font-size="13" fill="#3f3f46">recall →</text><text x="58" y="296" font-size="12" fill="#52525b">0</text><text x="470" y="298" text-anchor="middle" font-size="12" fill="#52525b">1.0</text></svg><figcaption>同じ PR 曲線でも AP の出し方は複数あります。<b>生 PR（灰）</b>は採用する予測を増やすほど上下するギザギザ、<b>単調包絡（橙）</b>は <code>np.maximum.accumulate</code> で右から均した非増加の上限線です。AP はこの包絡の下の <b>面積</b>（薄橙）で、<b>標本点</b>を PASCAL VOC は 11 点・COCO は 101 点で取って平均します。方式が違えば同じ曲線でも数値が少しずれます。</figcaption></figure>

| 補間方式 | 標本化 | 計算 | 使われる場面 |
| --- | --- | --- | --- |
| PASCAL VOC 2007（11 点） | recall = 0, 0.1, …, 1.0 の 11 点 | 各点で「recall≥r の最大 precision」を取り平均 | 古い検出論文 |
| VOC2010+（全点） | PR の全変化点 | 単調化した PR 曲線の真下の面積（Σ Δrecall × precision） | PASCAL 後期・一部の比較 |
| COCO（101 点） | recall = 0, 0.01, …, 1.0 の 101 点 | 各点の単調 precision を平均 | **現在の標準（COCO/mAP）** |

実行すると、同じカテゴリ・同じ IoU=0.5 でも `PASCAL 11点=0.7273 / 全点=0.7605 / COCO 101点=0.7581` のように数字が割れます。これはどれかが「正しい」というより、**方式が違えば物差しも違う** というだけのことです。だからこそ実務では「mAP いくつ」と単独で言わず、必ず「どの補間方式・どの IoU 閾値か」を添えます。`02_pr_curve.png` には raw PR（ギザギザ）・単調包絡・11 点標本が重ねて描かれており、3 方式が同じ曲線の異なる要約であることを目で確認できます。

この節には、もう 1 つ、実務で最も多い事故の再現を盛り込んでいます。それは **confidence 降順ソートを忘れる** と AP が壊れる、というものです。AP は「スコアの高い予測から順に採用する」ことが大前提なので、並べ替えずに cumsum すると PR 列が無意味な軌跡になってしまいます。そこで `02` は、わざとソートを抜いた場合の値も出力します。

```text
=== よくある事故: confidence 降順ソートを忘れる ===
  正しい(ソートあり) COCO-AP = 0.7581
  間違い(ソート無し) COCO-AP = 0.4087  ← 別物の数字になる
```

正しい 0.7581 に対し、ソート無しは 0.4087 と、半分近くまで落ちています。したがって、自作評価器を書いてデバッグするとき、値が妙に低かったら真っ先に「スコア降順に並べたか」を疑ってください。

## 5. mAP@0.5 と mAP@[.5:.95] — IoU 閾値ループとカテゴリ平均

AP はあくまで「1 カテゴリ・1 つの IoU 閾値」の値です。これを 2 段階で平均すると、検出の代表指標 **mAP** になります。まず **カテゴリで平均** すれば、その IoU 閾値での mAP（mean AP）が得られます。さらに **IoU 閾値でも平均** すれば、COCO の主指標 **mAP@[.5:.95]** —— IoU を 0.50, 0.55, …, 0.95 の 10 段階で動かして平均したもの —— になります。一方、IoU=0.5 だけなら **mAP@0.5**（旧 PASCAL 風で、位置がそこそこ合っていれば OK の緩い指標）、IoU=0.75 だけなら **mAP@0.75**（位置精度に厳しい指標）です。この二重平均は、`03_map_vs_pycocotools.py` の `evaluate_scratch()` がカテゴリ × IoU 閾値の二重ループで AP 表を埋めることで実現します。

<figure class="lec-fig"><svg viewBox="0 0 620 310" role="img" aria-label="AP表はカテゴリかけるIoU閾値。列をカテゴリ平均するとmAP@0.5やmAP@0.75、全体平均がmAP[.5:.95]" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><text x="310" y="34" text-anchor="middle" font-size="14" font-weight="700" fill="#18181b">AP 表（行＝カテゴリ, 列＝IoU 閾値 0.50 → 0.95）</text><rect x="130" y="78" width="38" height="102" fill="#ea580c" opacity="0.88"/><rect x="168" y="78" width="38" height="102" fill="#ea580c" opacity="0.80"/><rect x="206" y="78" width="38" height="102" fill="#ea580c" opacity="0.72"/><rect x="244" y="78" width="38" height="102" fill="#ea580c" opacity="0.64"/><rect x="282" y="78" width="38" height="102" fill="#ea580c" opacity="0.56"/><rect x="320" y="78" width="38" height="102" fill="#ea580c" opacity="0.48"/><rect x="358" y="78" width="38" height="102" fill="#ea580c" opacity="0.40"/><rect x="396" y="78" width="38" height="102" fill="#ea580c" opacity="0.32"/><rect x="434" y="78" width="38" height="102" fill="#ea580c" opacity="0.24"/><rect x="472" y="78" width="38" height="102" fill="#ea580c" opacity="0.18"/><line x1="130" y1="112" x2="510" y2="112" stroke="#ffffff" stroke-width="1.5"/><line x1="130" y1="146" x2="510" y2="146" stroke="#ffffff" stroke-width="1.5"/><line x1="168" y1="78" x2="168" y2="180" stroke="#ffffff" stroke-width="1.2"/><line x1="206" y1="78" x2="206" y2="180" stroke="#ffffff" stroke-width="1.2"/><line x1="244" y1="78" x2="244" y2="180" stroke="#ffffff" stroke-width="1.2"/><line x1="282" y1="78" x2="282" y2="180" stroke="#ffffff" stroke-width="1.2"/><line x1="320" y1="78" x2="320" y2="180" stroke="#ffffff" stroke-width="1.2"/><line x1="358" y1="78" x2="358" y2="180" stroke="#ffffff" stroke-width="1.2"/><line x1="396" y1="78" x2="396" y2="180" stroke="#ffffff" stroke-width="1.2"/><line x1="434" y1="78" x2="434" y2="180" stroke="#ffffff" stroke-width="1.2"/><line x1="472" y1="78" x2="472" y2="180" stroke="#ffffff" stroke-width="1.2"/><rect x="130" y="78" width="380" height="102" fill="none" stroke="#c2410c" stroke-width="2"/><text x="120" y="100" text-anchor="end" font-size="13" font-weight="700" fill="#18181b">A</text><text x="120" y="134" text-anchor="end" font-size="13" font-weight="700" fill="#18181b">B</text><text x="120" y="168" text-anchor="end" font-size="13" font-weight="700" fill="#18181b">C</text><line x1="149" y1="182" x2="149" y2="198" stroke="#2563eb" stroke-width="2"/><polygon points="149,204 144,196 154,196" fill="#2563eb"/><text x="149" y="222" text-anchor="middle" font-size="13" font-weight="700" fill="#2563eb">mAP@0.5</text><line x1="339" y1="182" x2="339" y2="198" stroke="#2563eb" stroke-width="2"/><polygon points="339,204 334,196 344,196" fill="#2563eb"/><text x="339" y="222" text-anchor="middle" font-size="13" font-weight="700" fill="#2563eb">mAP@0.75</text><rect x="120" y="250" width="390" height="44" rx="8" fill="#fff7ed" stroke="#c2410c" stroke-width="2"/><text x="315" y="278" text-anchor="middle" font-size="14" font-weight="700" fill="#c2410c">全カテゴリ × 全 IoU を平均 ＝ mAP@[.5:.95]</text></svg><figcaption>AP は本来 <b>1 カテゴリ・1 つの IoU 閾値</b> の値です（表の 1 マス）。各列を <b>カテゴリで平均</b> すると、その IoU での mAP に。<code>IoU=0.50</code> だけなら <b>mAP@0.5</b>、<code>0.75</code> だけなら <b>mAP@0.75</b>。さらに <b>全カテゴリ × 全 IoU（0.50:0.05:0.95 の 10 段階）を平均</b> したものが COCO の主指標 <b>mAP@[.5:.95]</b> です。左（IoU 緩い）ほど色が濃く AP が高く、右（厳しい）ほど薄く AP が下がります。</figcaption></figure>

```python
for ci, c in enumerate(cat_ids):              # カテゴリごと
    for ti, thr in enumerate(IOU_THRS):       # IoU 0.50,0.55,...,0.95 ごと
        # 画像ごとにマッチング → 画像横断でスコア降順に累積 → COCO 101点 AP
        ap[ci, ti] = coco101_ap(tp_all[order], n_gt_total)
map_5095 = np.nanmean(ap)        # 全カテゴリ・全閾値の平均 = COCO 主指標
map_50   = np.nanmean(ap[:, 0])  # IoU=0.50 のみ
map_75   = np.nanmean(ap[:, 5])  # IoU=0.75 のみ
```

実行すると `mAP@0.5=0.8215 / mAP@0.75=0.6962 / mAP@[.5:.95]=0.5275` のように出ます。ここでのポイントは **IoU を厳しくするほど mAP が下がる** ことで、`03_map_compare.png` の左パネルは、これを横軸 IoU 閾値・縦軸 mAP の右肩下がりの曲線として描きます。mAP@0.5 が高くても mAP@[.5:.95] が低いモデルは「物体の存在は当てるが位置がやや甘い」と読め、逆に両者が近ければ「ボックスがタイトに合っている」と読めます。この読み分けができると、検出モデルの強み・弱みを数字から具体的に語れるようになります。

なお、AP 表を `np.nan` で初期化し `np.nanmean` で平均しているのは、**GT が存在しないカテゴリを平均から除外** するためです。COCO 公式も、同じ状況を「AP = −1」で表して無視します。本章の合成データは全カテゴリに GT があるので影響はありませんが、実データの評価では「そのカテゴリが画像に 1 つも無い」ことが普通に起きるので、この除外処理は欠かせません。

## 6. pycocotools COCOeval との検算 — bbox 形式・maxDets・areaRng

自作評価器が正しいかどうかは、**COCO 公式実装 pycocotools と突き合わせる** ことで確かめます。`03_map_vs_pycocotools.py` は、同じ合成データを `COCO`（GT）と `loadRes`（予測）に読ませ、`COCOeval(iouType='bbox')` で `evaluate → accumulate → summarize` を回して、12 個の標準統計 `stats` を得ます。そして、その `stats[0]=AP@[.5:.95]`、`stats[1]=AP50`、`stats[2]=AP75` が自作値と一致するかを `assert np.isclose(..., atol=1e-6)` で検証します。実行結果は次の通りで、差は浮動小数の丸め誤差レベル（1e-16）に収まります。

```text
[検証OK] 自作 mAP@[.5:.95]/mAP@0.5/mAP@0.75 が pycocotools と一致 (最大差 < 1e-6)。
         差: 1.11e-16 / 1.11e-16 / 1.11e-16
```

この完全一致を出すために、自作の `coco101_ap()` は pycocotools の `accumulate` と同じ手順を厳密に踏んでいます。すなわち、precision に極小の `np.spacing(1)` を足して 0 除算を避け、`np.maximum.accumulate` で右から単調化し、recall 閾値の位置を `np.searchsorted(recall, rec_thrs, side="left")` で探し、届かない recall 閾値の precision は 0 とする、という流れです。マッチングのほうも COCO と同じ「未使用 GT の中で IoU 最大へ割り当て」「安定ソート」を採用しています。これらが 1 つでもズレると小数が合わなくなるので、**一致はそのまま "中身を正しく理解した証拠"** になります。

ここで、COCO 形式の落とし穴を 3 つ押さえておきます。下表のとおり、bbox 形式・maxDets・areaRng を既定から変えると、数字が比較不能になります。また、実行ログに出る `AP_S/M/L = -1.0000 / 0.5295 / 0.5190` のうち `AP_S` が −1 なのは、合成データに「小物体（面積 < 32² px）」が存在せず、pycocotools が −1（=該当なし）を返しているからです。実写を `data/` に置けば小物体も現れ、`AP_S` が意味を持つようになります。

| 項目 | 既定値 | 取り違えると | 対処 |
| --- | --- | --- | --- |
| bbox 形式 | COCO JSON は `xywh` | xyxy のまま渡すと箱が歪み AP が崩壊 | `to_coco_*` で xyxy→xywh 変換 |
| maxDets | AP は画像あたり上位 100 検出 | 多すぎる予測を切ると recall 上限が下がる | 既定 100 を変えない（本章は <100/画像） |
| areaRng | 'all' は [0, 1e10] | 面積帯を変えると AP_S/M/L が比較不能 | 小<32², 中 32²〜96², 大>96² の既定を踏襲 |

`AR`（Average Recall）も `summarize` から読めます。AR@1 / AR@10 / AR@100 は「各画像で上位 1 / 10 / 100 個の検出だけ使ったときの再現率」で、検出器が "拾える上限" を測る指標です。本章はこの AR を概念紹介に留め、自作実装は mAP@0.5・mAP@[.5:.95] に集中します（AP_S/M/L・AR は pycocotools の値をそのまま表示します）。

## 7. なぜ自作と公式が一致するのか・どこでズレるのか

この章の核心は「自作 numpy が COCO 公式と一致する」体験です。一致するのは偶然ではなく、両者が **同じアルゴリズム** を実行しているからにほかなりません。具体的には、(1) IoU の定義、(2) confidence 降順・安定ソート、(3) 未使用 GT の中で IoU 最大へ貪欲割り当て（二重カウント防止）、(4) 画像横断の cumsum による PR 列、(5) 単調包絡＋101 点標本での AP、という 5 ステップが完全に一致しています。逆に言えば、この 5 つのどこか 1 つでも実装を間違えれば小数が合わなくなるわけで、**一致しないときの差分の出方からバグの場所を逆算できる** ことこそ、自作して照合することの最大の効用です。

<figure class="lec-fig"><svg viewBox="0 0 660 220" role="img" aria-label="評価パイプライン5段。IoU、スコア降順ソート、貪欲マッチング、PR列の累積、AP補間。前半は画像ごとの対応付け、後半は画像横断の集計" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><text x="330" y="30" text-anchor="middle" font-size="14" font-weight="700" fill="#18181b">評価パイプライン — この5段が COCO 公式と一致する</text><text x="200" y="50" text-anchor="middle" font-size="12" font-weight="700" fill="#c2410c">① 画像ごとに対応付け</text><text x="530" y="50" text-anchor="middle" font-size="12" font-weight="700" fill="#2563eb">② 画像横断で集計</text><rect x="14" y="62" width="108" height="110" rx="7" fill="#ffedd5" stroke="#c2410c" stroke-width="2"/><circle cx="68" cy="84" r="12" fill="#c2410c"/><text x="68" y="89" text-anchor="middle" font-size="13" font-weight="700" fill="#ffffff">1</text><text x="68" y="120" text-anchor="middle" font-size="15" font-weight="700" fill="#18181b">IoU</text><text x="68" y="142" text-anchor="middle" font-size="11" fill="#52525b">交差 ÷ 和集合</text><rect x="146" y="62" width="108" height="110" rx="7" fill="#ffedd5" stroke="#c2410c" stroke-width="2"/><circle cx="200" cy="84" r="12" fill="#c2410c"/><text x="200" y="89" text-anchor="middle" font-size="13" font-weight="700" fill="#ffffff">2</text><text x="200" y="118" text-anchor="middle" font-size="13" font-weight="700" fill="#18181b">スコア降順</text><text x="200" y="140" text-anchor="middle" font-size="11" fill="#52525b">安定ソート</text><rect x="278" y="62" width="108" height="110" rx="7" fill="#ffedd5" stroke="#c2410c" stroke-width="2"/><circle cx="332" cy="84" r="12" fill="#c2410c"/><text x="332" y="89" text-anchor="middle" font-size="13" font-weight="700" fill="#ffffff">3</text><text x="332" y="116" text-anchor="middle" font-size="12" font-weight="700" fill="#18181b">貪欲マッチング</text><text x="332" y="138" text-anchor="middle" font-size="10" fill="#52525b">二重カウント防止</text><rect x="410" y="62" width="108" height="110" rx="7" fill="#dbeafe" stroke="#2563eb" stroke-width="2"/><circle cx="464" cy="84" r="12" fill="#2563eb"/><text x="464" y="89" text-anchor="middle" font-size="13" font-weight="700" fill="#ffffff">4</text><text x="464" y="120" text-anchor="middle" font-size="14" font-weight="700" fill="#18181b">PR 列</text><text x="464" y="142" text-anchor="middle" font-size="11" fill="#52525b">cumsum 累積</text><rect x="542" y="62" width="108" height="110" rx="7" fill="#dbeafe" stroke="#2563eb" stroke-width="2"/><circle cx="596" cy="84" r="12" fill="#2563eb"/><text x="596" y="89" text-anchor="middle" font-size="13" font-weight="700" fill="#ffffff">5</text><text x="596" y="116" text-anchor="middle" font-size="15" font-weight="700" fill="#18181b">AP</text><text x="596" y="138" text-anchor="middle" font-size="11" fill="#52525b">単調包絡</text><text x="596" y="156" text-anchor="middle" font-size="11" fill="#52525b">101点標本</text><line x1="122" y1="117" x2="138" y2="117" stroke="#52525b" stroke-width="2"/><polygon points="146,117 136,112 136,122" fill="#52525b"/><line x1="254" y1="117" x2="270" y2="117" stroke="#52525b" stroke-width="2"/><polygon points="278,117 268,112 268,122" fill="#52525b"/><line x1="386" y1="117" x2="402" y2="117" stroke="#52525b" stroke-width="2"/><polygon points="410,117 400,112 400,122" fill="#52525b"/><line x1="518" y1="117" x2="534" y2="117" stroke="#52525b" stroke-width="2"/><polygon points="542,117 532,112 532,122" fill="#52525b"/></svg><figcaption>第14回の <b>PR 曲線 → 面積</b> の骨格に、検出では <b>IoU マッチング</b> を 1 枚かませるだけです。①<b>IoU</b>（交差 ÷ 和集合）で重なりを測り、②<b>スコア降順</b>に安定ソートし、③未使用 GT へ <b>貪欲マッチング</b>（<b>二重カウント防止</b>）——ここまでは<b>画像ごと</b>。続けて全画像をまとめ ④<code>cumsum</code> で <b>PR 列</b>、⑤<b>単調包絡 ＋ 101 点標本</b>で <b>AP</b> を出す後半は<b>画像横断</b>です。この 5 段が完全一致するから、自作 mAP が pycocotools と小数点以下まで揃います。</figcaption></figure>

そこで、意図的にズレを作って原因を説明できるようにしておくと、デバッグ力が一段上がります。代表的なズレ要因を下にまとめました。これらは本章のスクリプトで（一部は意図的に）再現しているので、「症状 → 原因」の対応を体で覚えておくと、現場で他人の評価コードを読むときにも効きます。

| ズレの症状 | 原因 | 直し方 |
| --- | --- | --- |
| AP が妙に低い | confidence 降順ソートを忘れた | スコア降順（安定ソート）に並べてから cumsum |
| 数字が微妙に違う | 補間方式が違う（11 点 / 全点 / 101 点） | COCO に合わせるなら 101 点 |
| precision が不当に高い | 1 GT への二重 TP を許した | マッチした GT を「使用済み」にして再利用禁止 |
| box が歪んで AP 崩壊 | xyxy のまま COCO へ渡した | xywh へ変換してから loadRes |
| 同点で結果が揺れる | 不安定ソートを使った | `kind="stable"`（mergesort）を指定 |

## 8. このモジュールの構成（スクリプト一覧）

各スクリプトは単一責務で、上から順に読めば「IoU とマッチング → PR と AP 補間 → mAP と公式照合」と理解が積み上がる構成です。いずれも結果を `outputs/19_detection_map_from_scratch/` に図と JSON として保存し、画面表示には依存しません。なお、合成データの生成・COCO 形式変換・IoU・正準マッチングエンジンは `det_helpers.py` にまとめてあり、各スクリプトはそれを import して使います。

| ファイル | 役割（単一責務） |
| --- | --- |
| `det_helpers.py` | 合成検出データ生成・COCO(xywh) 変換・`box_iou_numpy` ・正準マッチング `match_image` ・`output_dir()` |
| `01_iou_matching.py` | IoU の自作（torchvision と一致確認）、confidence 降順の貪欲マッチング、二重カウント防止、TP/FP/FN の可視化 |
| `02_pr_ap_interpolation.py` | 累積和で PR 列を構築、AP の 3 補間方式（11 点 / 全点 / COCO 101 点）、ソート漏れバグの再現 |
| `03_map_vs_pycocotools.py` | カテゴリ × IoU 閾値で mAP@0.5 / mAP@0.75 / mAP@[.5:.95] を算出、pycocotools COCOeval と一致を `assert` 検証 |
| `mini_project.py` | 章末ミニプロジェクト。同一 GT に strong/weak の2検出器を合成し、自作 mAP で比較→pycocotools で検算→F1 最大の運用しきい値を決定（4枚パネル図 + JSON） |
| `exercises.py` | TODO 形式の演習9問（IoU / マッチング / PR / 3補間方式 / xywh変換 / NMS / mAP集約）。自己採点ランナー付きで、未実装でも `exit 0`。`SHOW_SOLUTION=1` で模範解答に差し替え |
| `exercises_solutions.py` | 演習の模範解答ランナー（全問 PASS）。採点ロジックと解答は `exercises.py` を再利用（重複なし） |

この中で `det_helpers.py` だけは「読み物」ではなく「再利用する道具」です。中身も厚くコメントしてあるので、最初に一読してから 01 へ進むと、各スクリプトが何のデータで実験しているかが腑に落ちます。実画像で試したい人は、自分の GT/予測を `gt_boxes / pred_boxes / pred_scores / pred_labels` の形に整え、`make_detection_dataset` の戻り値と同じ dict にすれば、評価器へそのまま流せます。

## 9. 動かし方

このモジュールは `numpy` / `torch` / `torchvision` / `pycocotools` / `matplotlib` に依存します。とはいえ画像モデルもネット接続も不要で、データは合成で自動生成されるため、依存さえ入っていればすぐ実行できます。プロジェクトルートで以下を順に実行してください（初回は依存の解決に少し時間がかかります）。

```bash
# 依存グループを用意（初回のみ）。dl=torch/torchvision, metrics=pycocotools
uv sync --group dl --group metrics

# 各スクリプトを実行（結果は outputs/19_detection_map_from_scratch/ に保存される）
uv run python lectures/19_detection_map_from_scratch/01_iou_matching.py
uv run python lectures/19_detection_map_from_scratch/02_pr_ap_interpolation.py
uv run python lectures/19_detection_map_from_scratch/03_map_vs_pycocotools.py

# 章末ミニプロジェクト（2検出器を自作 mAP で比較→pycocotools 検算→運用しきい値決定）
uv run python lectures/19_detection_map_from_scratch/mini_project.py

# 演習: まずは TODO を自分で埋める（最初は全部 FAIL。それでも exit 0 で落ちない）
uv run python lectures/19_detection_map_from_scratch/exercises.py
# どうしても分からない時だけ、模範解答の挙動を見る（どちらも全問 PASS）
SHOW_SOLUTION=1 uv run python lectures/19_detection_map_from_scratch/exercises.py
uv run python lectures/19_detection_map_from_scratch/exercises_solutions.py
```

実行後は `outputs/19_detection_map_from_scratch/` に生成された画像と JSON を確認してください。`01_matching.png`（GT=緑 / TP=青 / FP=赤）、`02_pr_curve.png`（raw PR・単調包絡・11 点標本）、`03_map_compare.png`（左: IoU 閾値ごとの mAP 低下曲線、右: 自作と pycocotools の棒が重なる＝一致）を本文の解説と照らし合わせると、理解が定着します。各 JSON には自作と公式双方の数値が記録されているので、一致を自分の目でも確かめられます。なお、`03` は初回に pycocotools のインデックス作成ログを標準出力へ出しますが、これはネットアクセスではなくローカル処理のログであり、モデル DL も発生しません。

> **合成データの限界と実写への拡張**: 本章の合成データは「GT を少しずらした予測 + 検出漏れ + 無関係な誤検出」で PR 曲線に起伏を作っていますが、小物体（面積 < 32² px）を含まないため `AP_S` は −1（該当なし）になります。`data/` に COCO 形式（`xywh`）の GT/予測を置き、`make_detection_dataset` の代わりにそれを読み込めば、面積別 AP や AR を含めた実用的な評価ができます。評価ロジック自体は、合成でも実写でも完全に同じです。

## 10. よくある落とし穴（チェックリスト）

最後に、この章でつまずきやすい点を「症状 → 原因 → 対処」でまとめます。実装中に詰まったら、まずここを見てください。

| 症状 | ほぼ確実な原因 | 対処 |
| --- | --- | --- |
| IoU の値が torchvision と合わない | bbox 形式の取り違え（xyxy / xywh / cxcywh） | IoU は xyxy で計算。COCO 受け渡し時だけ xywh へ |
| precision が不当に高い | 1 つの GT に複数の TP を許した | マッチした GT を使用済みにして再利用禁止 |
| 自作 AP が妙に低い | confidence 降順ソートを忘れた | スコア降順（`kind="stable"`）に並べてから cumsum |
| 他者の mAP と 0.0x ずれる | 補間方式が違う（11 点 / 全点 / 101 点） | COCO に合わせるなら 101 点・101 recall 閾値 |
| 「mAP」だけ言われて噛み合わない | mAP@0.5 と mAP@[.5:.95] の取り違え | IoU 閾値（と補間方式）を必ず明記 |
| pycocotools に渡すと box が歪む | xyxy のまま loadRes へ渡した | `to_coco_dt` で xyxy→xywh 変換 |
| AP_S が −1 になる | その面積帯に GT が無い（合成は小物体なし） | 実写を入れる／その帯を評価対象から外す |
| GT 0 件のカテゴリで NaN/エラー | 平均に空カテゴリを含めた | `np.nan` 初期化 + `np.nanmean` で除外（COCO は −1） |

この 8 項目が、検出評価でつまずく原因のほぼ全てです。逆に言えば、この 8 つを自分の言葉で説明でき、回避コードまで書けるようになれば、この章のゴールに到達しています。

## 11. まとめ

この章では、物体検出の mAP を「IoU → confidence 降順マッチング（二重カウント防止）→ cumsum による PR 列 → AP 補間 → IoU 閾値・カテゴリ平均」という一本道としてとらえ、すべて numpy で一から組み上げました。そのうえで、自作値が pycocotools COCOeval と小数点以下まで一致することを `assert` で検証し、さらに補間方式違い・ソート漏れ・bbox 形式違いという「数字がズレる典型原因」を意図的に再現して、差分から原因を逆算できる状態を作りました。第14回の「混同行列 → PR 曲線 → 面積」の骨格が、検出では「IoU マッチング」を一枚かませるだけで成立する —— この連続性こそ、評価指標トラックの背骨です。

次の第20回以降は、ここで作った mAP の目を持って、オープン語彙検出やセグメンテーションの評価（mask AP / mIoU / Dice / PQ）へ進みます。それらも本章と同じく「対応付け → 累積 → 面積/比」の応用にすぎません。ですから、まずは演習を自力で全問 PASS させ、`assert` で自作と公式の一致を体感してから次へ進んでください。

---

## 🛠 章末ミニプロジェクト — 2台の検出器の「評価レポート」を一気通貫で書く

ここまでで、部品（IoU → マッチング → PR → AP → mAP）は全部そろいました。最後は、それらを統合して **「現場で評価レポートを書く」** ところまでやり切ります。`mini_project.py` は、次の 4 ステップを 1 本のスクリプトで通します。これは、実務で検出器を「採用するか／どのしきい値で運用するか」を決めるときの最小フローそのものです。

1. **同一 GT に 2 台の検出器を合成**: まず GT（正解）を 1 セットだけ作り、その**同じ GT** に対して品質の異なる予測を 2 通り合成します（`strong` = 取りこぼし少・位置精度高・誤検出少／`weak` = その逆）。検出器を比べるときは「同じ正解・同じ画像」で測るのが鉄則で、GT が違えば mAP は比較不能になってしまうからです。
2. **自作 mAP で評価**: 03 で組んだロジック（`coco101_ap` / カテゴリ × IoU 閾値ループ）を自己完結で再実装し、両検出器の `mAP@0.5 / mAP@0.75 / mAP@[.5:.95]` とカテゴリ別 AP50 を算出します。
3. **pycocotools で検算**: 各検出器について自作値が COCOeval と `< 1e-6` で一致することを `assert` します。**検算が通って初めてレポートの数字が信用できる** —— これが「自作できる」ことの実利です。
4. **運用しきい値を F1 で決める**: PR 機構は評価だけの道具ではありません。`strong` 検出器について confidence しきい値を 0.05〜0.95 で掃引し、`precision / recall / F1` を描いて **F1 が最大になる 1 点** を推奨しきい値として選びます。mAP は「しきい値非依存の総合力」、運用は「1 点を選ぶ」—— この役割分担を体で覚えます。

<figure class="lec-fig"><svg viewBox="0 0 660 240" role="img" aria-label="ミニプロジェクトの4ステップ。同一GTに2検出器を合成し、自作mAPで評価、pycocotoolsで検算、F1で運用しきい値を決める" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><text x="330" y="34" text-anchor="middle" font-size="15" font-weight="700" fill="#18181b">2台の検出器を比べ、運用しきい値を決めるまで</text><rect x="16" y="64" width="138" height="120" rx="8" fill="#ffedd5" stroke="#c2410c" stroke-width="2"/><circle cx="85" cy="92" r="14" fill="#c2410c"/><text x="85" y="97" text-anchor="middle" font-size="15" font-weight="700" fill="#ffffff">1</text><text x="85" y="126" text-anchor="middle" font-size="13" font-weight="700" fill="#18181b">同一GTに</text><text x="85" y="146" text-anchor="middle" font-size="13" font-weight="700" fill="#18181b">2検出器を合成</text><text x="85" y="168" text-anchor="middle" font-size="11" fill="#52525b">strong / weak</text><rect x="176" y="64" width="138" height="120" rx="8" fill="#dbeafe" stroke="#2563eb" stroke-width="2"/><circle cx="245" cy="92" r="14" fill="#2563eb"/><text x="245" y="97" text-anchor="middle" font-size="15" font-weight="700" fill="#ffffff">2</text><text x="245" y="126" text-anchor="middle" font-size="13" font-weight="700" fill="#18181b">自作mAPで</text><text x="245" y="146" text-anchor="middle" font-size="13" font-weight="700" fill="#18181b">評価</text><text x="245" y="168" text-anchor="middle" font-size="11" fill="#52525b">@.5 / .75 / .5:.95</text><rect x="336" y="64" width="138" height="120" rx="8" fill="#ffffff" stroke="#16a34a" stroke-width="2"/><circle cx="405" cy="92" r="14" fill="#16a34a"/><text x="405" y="97" text-anchor="middle" font-size="15" font-weight="700" fill="#ffffff">3</text><text x="405" y="126" text-anchor="middle" font-size="13" font-weight="700" fill="#18181b">pycocotoolsで</text><text x="405" y="146" text-anchor="middle" font-size="13" font-weight="700" fill="#18181b">一致を検算</text><text x="405" y="168" text-anchor="middle" font-size="11" fill="#52525b">差 ≈ 0</text><rect x="496" y="64" width="138" height="120" rx="8" fill="#fff7ed" stroke="#ea580c" stroke-width="2"/><circle cx="565" cy="92" r="14" fill="#ea580c"/><text x="565" y="97" text-anchor="middle" font-size="15" font-weight="700" fill="#ffffff">4</text><text x="565" y="126" text-anchor="middle" font-size="13" font-weight="700" fill="#18181b">F1最大の点を</text><text x="565" y="146" text-anchor="middle" font-size="13" font-weight="700" fill="#18181b">運用しきい値に</text><text x="565" y="168" text-anchor="middle" font-size="11" fill="#52525b">推奨しきい値</text><line x1="154" y1="124" x2="166" y2="124" stroke="#52525b" stroke-width="2"/><polygon points="176,124 166,119 166,129" fill="#52525b"/><line x1="314" y1="124" x2="326" y2="124" stroke="#52525b" stroke-width="2"/><polygon points="336,124 326,119 326,129" fill="#52525b"/><line x1="474" y1="124" x2="486" y2="124" stroke="#52525b" stroke-width="2"/><polygon points="496,124 486,119 486,129" fill="#52525b"/></svg><figcaption><b>ミニプロジェクト</b>は 4 ステップの一気通貫です。①同一の <b>GT</b> に品質の違う 2 検出器（<b>strong / weak</b>）を合成し、②自作の <b>mAP</b>（<code>@0.5 / 0.75 / [.5:.95]</code>）で評価、③<b>pycocotools</b> の COCOeval と一致するか検算（差はほぼ 0）、④<b>strong</b> の PR を掃引して <b>F1 が最大</b>になる 1 点を運用しきい値に選びます。mAP は<b>しきい値非依存の総合力</b>、運用は<b>1 点を選ぶ</b>——この役割分担を 1 本のスクリプトで体験します。</figcaption></figure>

成果物は `outputs/19_detection_map_from_scratch/` に出ます。`mini_detector_report.png` は 4 枚パネル構成で、(左上) 2 検出器の PR 曲線（strong が右上に張り出す）、(右上) IoU 閾値に対する mAP の右肩下がり（strong が常に上）、(左下) カテゴリ別 AP50 の棒比較、(右下) strong の F1 掃引と推奨しきい値の縦線、を一望できます。また、`mini_project_report.json` には両検出器の全指標・pycocotools との最大差・推奨しきい値が記録されます。実行すると `strong: mAP@[.5:.95]≈0.74 / weak≈0.16`、`推奨しきい値≈0.55 (F1≈0.99)` のような数字が出ます。

> **発展課題（自分で手を動かす）**: (a) `make_predictions` の `shift_frac` だけを 0.06→0.18 に上げ、`mAP@0.5` はあまり下がらないのに `mAP@[.5:.95]` が大きく下がること（＝位置が甘いと厳しい IoU で効く）を確認する。(b) しきい値選びの基準を F1 から「recall を 0.9 以上に保ちつつ precision 最大」へ変え、運用要件で最適点が動くことを見る。(c) `data/` に実画像の COCO 形式（GT/予測）を置き、`make_ground_truth` の代わりにそれを読んで `AP_S`（小物体）が意味を持つようにする。

## ✅ 到達チェックリスト

この章を「マスターした」と言えるかどうかのセルフチェックです。すべて “理由つきで” 説明でき、コードで再現できれば合格です。

- [ ] IoU を定義式（交差 / 和集合）から numpy で書け、`torchvision.ops.box_iou` と一致させられる。
- [ ] bbox の 3 形式 `xyxy / xywh / cxcywh` を区別でき、COCO へ渡すときに `xyxy→xywh` を変換できる。
- [ ] confidence 降順の貪欲マッチングを書け、**1 つの GT に複数 TP を許さない**（二重カウント防止）理由を説明できる。
- [ ] マッチングは画像ごと・PR の累積は画像横断、という **二段構え** を説明でき、recall の分母がカテゴリ全体の GT 数（FN 込み）だと言える。
- [ ] cumsum で precision/recall 列を作り、**単調包絡**（`np.maximum.accumulate`）が何をしているか説明できる。
- [ ] AP の 3 補間方式（PASCAL 11 点 / VOC2010+ 全点 / COCO 101 点）を実装でき、同じ PR でも値が違う理由を言える。
- [ ] カテゴリ平均と IoU 閾値平均の二段で `mAP@0.5` と `mAP@[.5:.95]` を作れ、両者の差から「位置精度の甘さ」を読める。
- [ ] GT 0 件カテゴリを `np.nan + np.nanmean` で平均から除外する理由（COCO は −1）を説明できる。
- [ ] 自作 mAP が pycocotools COCOeval と `< 1e-6` で一致することを `assert` で示せる。
- [ ] 「AP が妙に低い／微妙にズレる／precision が不当に高い」の各症状から、原因（ソート漏れ・補間方式違い・二重カウント）を逆算できる。
- [ ] ミニプロジェクトで 2 検出器を比較し、PR から F1 最大の運用しきい値を選べる。

## ✍️ 演習問題

演習は `exercises.py` に TODO 形式で入っています。各 TODO を実装し `uv run python lectures/19_detection_map_from_scratch/exercises.py` を実行すると自己採点できます（`exercises_solutions.py` が解答）。

1. 2 つの箱（`xyxy`）の IoU を 1 つ計算して返す（`ex1_iou`）。交差矩形は左上＝max 同士・右下＝min 同士で求め、負は 0 にクリップし、`交差 / (面積A + 面積B − 交差)` を返す（和集合 0 なら 0.0）。
2. confidence 降順の貪欲マッチングで、各予測の TP=1.0 / FP=0.0 フラグを「スコア降順の順序で」返す（`ex2_match`）。未使用かつ IoU≥閾値の GT のうち IoU 最大へ割り当て、1 つの GT は一度だけ使う（二重カウント防止）。
3. スコア降順の TP フラグ列から precision 列・recall 列を組み立てて返す（`ex3_pr`）。`cumsum` で TP/FP を累積し、`recall = tp_cum / n_gt`、`precision = tp_cum / (tp_cum + fp_cum)` を `(precision, recall)` の順で返す。
4. COCO 101 点補間で AP を返す（`ex4_ap_coco101`）。precision を右から単調化し、recall 閾値 0, 0.01, …, 1.0 の各点の precision を `np.searchsorted(..., side="left")` で拾って 101 個を平均する（pycocotools と一致）。
5. PASCAL VOC 2007 方式の 11 点補間 AP を返す（`ex5_ap_11point`）。recall = 0, 0.1, …, 1.0 の各点で「recall≥r を満たす点の precision の最大値」（無ければ 0）を取り、11 個を平均する。
6. VOC2010+ 方式の「全点」AP を返す（`ex6_ap_all_point`）。端点を足して precision を右から単調化し、recall が変化した位置だけ `Σ(Δrecall × precision)` を積む、単調化 PR 曲線の真下の面積。
7. `xyxy` 形式の `(N,4)` ボックス群を COCO の `xywh` 形式へ変換して返す（`ex7_xyxy_to_xywh`）。`[x1, y1, x2, y2] → [x1, y1, x2−x1, y2−y1]`。ここを誤ると箱が歪んで AP が崩壊する。
8. Non-Maximum Suppression を実装し、残す予測の index 配列（スコア降順）を返す（`ex8_nms`）。スコア降順に先頭を採用し、それと IoU>閾値 の箱を捨てる操作を繰り返す（`torchvision.ops.nms` と同じ残存集合）。
9. AP[カテゴリ, IoU 閾値] の表から mAP@0.5 / mAP@0.75 / mAP@[.5:.95] を返す（`ex9_map_aggregate`）。列 0 の平均・列 5 の平均・全要素の平均を `np.nanmean` で（空カテゴリ＝NaN を除外して）求める。

## ❓ よくある落とし穴・FAQ・デバッグ

第 10 節のチェックリスト（症状 → 原因 → 対処）に加え、つまずきやすい点を Q&A 形式で補足します。

- **Q. 自作 AP が pycocotools と少しだけ（0.0x）ズレる。** まず補間方式を疑ってください。COCO は **101 点・101 recall 閾値**、`np.searchsorted(..., side="left")`、`precision + np.spacing(1)`、右からの単調化までを完全一致させる必要があります。次に疑うべきはソートの安定性（`kind="stable"`＝mergesort）です。本章の合成データはスコアが連続値で同点がほぼ無いので影響は小さいですが、実データでは同点が頻発し、不安定ソートだと値が揺れます。
- **Q. AP が極端に低い（半分くらい）。** ほぼ確実に **confidence 降順ソートの忘れ**です。`02` がわざと再現しています（正しい 0.7581 → ソート無し 0.4087）。cumsum の前に必ずスコア降順へ並べ替えてください。
- **Q. precision が 1.0 近くに張り付いて不自然に高い。** 1 つの GT に複数の予測を TP として数えていませんか。マッチした GT を「使用済み」にして再利用を禁止してください（`gt_used` フラグ）。
- **Q. pycocotools に渡すと箱が画面外に飛ぶ／AP が 0 になる。** `xyxy` のまま `loadRes` へ渡しています。COCO の `bbox` は **`xywh`**。`to_coco_dt`（演習なら `ex7_xyxy_to_xywh`）で変換してください。`target_sizes` を使う検出モデル（DETR 等）では `(H, W)` 順の取り違えも同種のバグです。
- **Q. `AP_S` が `-1.0000` と出る。** バグではありません。その面積帯（小物体 < 32² px）に GT が 1 つも無いと pycocotools は `-1`（該当なし）を返します。本章の合成データは小物体を含まないので `AP_S=-1` が正常です。実画像を入れれば意味を持ちます。
- **Q. GT が 1 つも無いカテゴリでクラッシュ／NaN が伝播する。** AP 表を `np.nan` で初期化し `np.nanmean` で平均すれば、空カテゴリは自動的に除外されます（COCO は内部で −1 として無視）。`np.mean` を使うと NaN が全体に伝播します。
- **Q. NMS（演習 8）の結果が `torchvision.ops.nms` と一致しない。** NMS とマッチングは別物です。NMS は「スコア最高の箱を残し、それと IoU > 閾値の箱を捨てる」を繰り返す重複除去で、評価のマッチングとは独立に**推論の後処理**として行います。YOLO11 のように NMS 内蔵のモデルへさらに NMS をかけて二重抑制しないよう注意。比較は順序ではなく**残存 index の集合**で行います。
- **Q. `mAP` と言われて話が噛み合わない。** `mAP@0.5`（旧 PASCAL 風・位置に緩い）と `mAP@[.5:.95]`（COCO 主指標・位置に厳しい）は別物です。必ず IoU 閾値（と補間方式）を添えて話してください。
- **デバッグの定石**: 値が合わないときは「①スコア降順か → ②二重カウントしてないか → ③補間方式（点数）は合ってるか → ④bbox は xywh か → ⑤空カテゴリを除外したか」の順に潰すと、ほぼ全ての mAP バグは特定できます。本章の `assert` 群はこの 5 点を 1 つずつ踏み外すと必ず落ちるように作ってあります。

## 🚀 発展トピック・参考

- **IoU の拡張**: 評価は素の IoU ですが、**学習の損失**には `GIoU / DIoU / CIoU`（重なりゼロでも勾配が出る・中心距離やアスペクト比を加味）が使われます。`torchvision.ops.generalized_box_iou` で挙動を比べると、素の IoU が「重ならないと 0 で勾配消失」する弱点が分かります。
- **後処理の改良**: 素の NMS は重なる正解物体を消しすぎることがあります。**Soft-NMS**（IoU に応じてスコアを減衰）、**class-agnostic NMS / batched_nms**（クラス跨ぎ抑制の有無）、**NMS-free 検出器**（DETR 系・YOLO26）の違いは、recall 上限と二重検出のトレードオフとして mAP に効きます。
- **誤差の分解**: mAP は 1 つの数に潰れて「なぜ低いか」が見えません。**TIDE**（Bolya et al.）は誤差を分類誤り・位置ズレ・重複・背景・見逃しに分解し、改善の打ち所を教えてくれます。本章の TP/FP/FN の枠組みがその土台です。
- **データセットごとの作法**: COCO は `maxDets=100`・101 点・`areaRng` 既定。**LVIS** は希少クラスを含む長尾評価（AP_r/AP_c/AP_f）、**Open Images** は階層ラベルと group-of box で評価規則が異なります。指標名が同じ `mAP` でも前提が違う点に注意。
- **タスクの横展開**: `iouType` を変えるだけで、`segm`（マスク IoU の mask AP）、`keypoints`（OKS による keypoint mAP）へ同じ COCOeval が使えます。次回以降の `mIoU / Dice / PQ = SQ × RQ` も「対応付け → 累積 → 面積/比」という本章の骨格の応用です。
- **公式実装を読む**: `pycocotools/cocoeval.py` の `evaluate / accumulate / summarize` は本章の自作とほぼ 1:1 対応します（`ious` 計算 → `dtMatches` → `precision` テンソル → `summarize`）。一度ソースを通読すると、自作との一致が「なぜ起きるか」が腑に落ちます。
- 参考: COCO 評価 [cocodataset.org/#detection-eval](https://cocodataset.org/#detection-eval) ／ pycocotools [github.com/ppwwyyxx/cocoapi](https://github.com/ppwwyyxx/cocoapi) ／ torchvision ops [docs.pytorch.org/vision/stable/ops.html](https://docs.pytorch.org/vision/stable/ops.html) ／ torchmetrics detection [lightning.ai/docs/torchmetrics](https://lightning.ai/docs/torchmetrics/stable/)。

---

> 本教材で参照・検証したライブラリとバージョン（2026-06 時点・CPU で動作確認）:
> Python 3.12 ／ numpy 2.4.6 ／ torch 2.12.0+cpu ／ torchvision 0.27.0+cpu ／ pycocotools 2.0.11 ／ matplotlib 3.10.9。
> 本講座の評価トラックの想定スタック（2026-06 時点）は torch 2.12+cpu / torchvision 0.27+cpu / pycocotools 2.0.11 / scikit-learn 1.9 / torchmetrics 1.9 で、後続回では transformers 5.11・faiss-cpu も併用します（本回では未使用）。pycocotools は C 拡張のため numpy 2.x との ABI 不一致に注意（2.0.11 は cp310–cp314 の wheel 配布で無ビルド導入可）。