# 第22回 インスタンス/パノプティックセグメンテーションと SAM — mask AP・PQ

> トラック: **セグメンテーション** ／ レベル: **中級** ／ 依存グループ: `dl`（torch/torchvision）・`hf`（transformers）・`metrics`（pycocotools/torchmetrics）。モデル重みの初回 DL 以外はネット不要で、入力画像は合成生成します。

## 🎯 この章のゴール

この章を終えたとき、あなたは「セグメンテーション」と一括りに呼ばれてきたものが、実は**3つの別タスク**——セマンティック（画素ごとのクラス）、インスタンス（個体を区別したマスク）、パノプティック（things と stuff を全画素にもれなく統合）——に分かれること、そして**プロンプト型の SAM** がそのどれとも違う第4の軸（クラスを当てず、指した“もの”を切り出す）にあることを、出力フォーマットのレベルで説明できるようになります。あわせて、Mask R-CNN の `masks` がなぜ `(N, 1, H, W)` の**確率**で返るのか、SAM がなぜ1プロンプトに**3枚**のマスクを返すのか、といった「最初の関門」も、自分の手で一つずつ通り抜けます。

さらに、これらのモデルを**どう評価するか**も、式と実装の両面から身につけます。インスタンスは、物体検出の mAP（第19回）で使う IoU を box から mask へ置き換えた **mask AP**（`COCOeval(iouType="segm")` と RLE）で測り、パノプティックは検出とは別系統の指標 **PQ = SQ × RQ** で測ります。とくに PQ では、`numpy` で一から組んだ自作値が `torchmetrics.detection.PanopticQuality` と一致することを `assert` で確認し、「ライブラリのブラックボックス」を「中で何をしているかが分かる道具」へと変えていきます。

到達点を一言でいえば、**Mask R-CNN / Mask2Former / SAM の出力を正しく後処理して可視化でき、mask AP と PQ を“式が分かる状態”で計算できる**こと。合成図形では検出が乏しい（0件のこともある）という現実も含めて、「動かす・読む・測る」を一通り回せるようになるのが合格ラインです。

---

## 1. 3つのセグメンテーションと SAM の位置づけ

まず地図を持ちましょう。**セマンティックセグメンテーション**（第21回）は各画素にクラスだけを振るので、隣り合う2匹の犬は同じ「犬」に塗られ、個体を区別しません。**インスタンスセグメンテーション**は逆に「犬1 / 犬2」を別物として、それぞれにマスクを付けます（ただし背景=stuff は扱わない）。**パノプティックセグメンテーション**はこの2つを統合し、things（数えられる前景: 人・車）のインスタンスと stuff（数えられない背景: 空・道路・芝）の領域を、**全画素にもれなく・重なりなく**1枚へまとめます。こうして各画素は、ちょうど1つの `(category, instance)` を持つことになります。

この3つはいずれも「学習済みのクラスを当てる」枠組みですが、**SAM（Segment Anything）**だけは設計思想が違います。SAM はクラスを一切当てず、点や箱の**プロンプトで指した領域の輪郭マスク**を返す、クラス非依存（class-agnostic）のセグメンタです。「これは犬だ」とは言わないが、「ここにある“もの”の形はこれだ」とは答える。そのため未知の対象でも、合成図形でも、指しさえすれば切り出せます。本章の4スクリプトは、この地図の各点を1つずつ歩いていきます。

<figure class="lec-fig"><svg viewBox="0 0 540 300" role="img" aria-label="同じシーンに対するセマンティック・インスタンス・パノプティック・SAMの出力の違い" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><text x="136" y="34" text-anchor="middle" font-size="16" font-weight="700" fill="#18181b">セマンティック</text><rect x="24" y="46" width="224" height="42" fill="#dbeafe"/><rect x="24" y="88" width="224" height="54" fill="#e4e4e7"/><rect x="24" y="46" width="224" height="96" fill="none" stroke="#d4d4d8" stroke-width="1.5"/><circle cx="94" cy="108" r="22" fill="#f97316"/><circle cx="174" cy="96" r="18" fill="#f97316"/><text x="404" y="34" text-anchor="middle" font-size="16" font-weight="700" fill="#18181b">インスタンス</text><rect x="292" y="46" width="224" height="96" fill="#ffffff" stroke="#d4d4d8" stroke-width="1.5"/><circle cx="362" cy="108" r="22" fill="#ea580c"/><circle cx="442" cy="96" r="18" fill="#2563eb"/><text x="136" y="180" text-anchor="middle" font-size="16" font-weight="700" fill="#18181b">パノプティック</text><rect x="24" y="192" width="224" height="42" fill="#dbeafe"/><rect x="24" y="234" width="224" height="54" fill="#d4d4d8"/><rect x="24" y="192" width="224" height="96" fill="none" stroke="#d4d4d8" stroke-width="1.5"/><circle cx="94" cy="254" r="22" fill="#ea580c"/><circle cx="174" cy="242" r="18" fill="#2563eb"/><text x="404" y="180" text-anchor="middle" font-size="16" font-weight="700" fill="#18181b">SAM</text><rect x="292" y="192" width="224" height="96" fill="#ffffff" stroke="#d4d4d8" stroke-width="1.5"/><circle cx="442" cy="242" r="18" fill="#d4d4d8"/><circle cx="362" cy="254" r="22" fill="#ffedd5" stroke="#c2410c" stroke-width="2.5" stroke-dasharray="5 3"/><circle cx="362" cy="254" r="5" fill="#dc2626"/></svg><figcaption>同じシーンでも、タスクで“出力の意味”が変わります。<b>セマンティック</b>は画素にクラスだけを振り、2匹を同じ「犬」に塗って<b>個体を区別しません</b>（2円が同色）。<b>インスタンス</b>は個体を分けますが背景(stuff)は扱いません（背景は白）。<b>パノプティック</b>は things と stuff を<b>全画素にもれなく</b>統合します。<b>SAM</b>はクラスを当てず、<b>点で指した“もの”の形だけ</b>を切り出します（class-agnostic）。</figcaption></figure>

| タスク | 個体の区別 | 背景(stuff) | 代表モデル | 主な評価指標 |
| --- | --- | --- | --- | --- |
| セマンティック | しない | 扱う | DeepLab/SegFormer | mIoU / Dice（第21回） |
| インスタンス | する | 扱わない | **Mask R-CNN** | **mask AP** |
| パノプティック | する | 扱う | **Mask2Former** | **PQ = SQ×RQ** |
| プロンプト型 | プロンプト次第 | プロンプト次第 | **SAM** | マスクと GT の IoU |

表の通り、評価指標もタスクごとに別物です。だからこそ「セグメンテーションの精度」と言われたら、まず「どのタスクか」を確認するのが第一歩になります。以降は Mask R-CNN（§2）→ Mask2Former（§3）→ SAM（§4）→ 評価（§5・§6）の順に進みます。

## 2. Mask R-CNN でインスタンスセグメンテーション（`01_maskrcnn_instance.py`）

torchvision の `maskrcnn_resnet50_fpn_v2` は、物体検出（Faster R-CNN）にマスク予測の枝を足した2段階モデルです。ロードは、Weights enum から重み・前処理・クラス名を一括で取得する torchvision の正準パターンに従います。検出モデルの入力は「`[0,1]` の float CHW テンソルのリスト」であり、ImageNet 正規化をモデル内部が行う点が画像分類とは異なります（`weights.transforms()` に任せ、二重に正規化しないこと）。そして推論時は、`model.eval()` と `torch.inference_mode()` をセットで使うのが必須の作法です。

出力は、1枚あたり `{boxes, labels, scores, masks}` の dict です。ただし、ここに**この回最大の関門**があります。`masks` は `(N, 1, H, W)` の**確率マップ**（0〜1）であって、bool のマスクではありません。そのため可視化や評価に使うには、チャンネル次元を潰し、閾値で二値化する必要があります。この処理を忘れたまま `(N,1,H,W)` を `draw_segmentation_masks`（uint8 画像 + **bool** マスクを要求）へ渡すのが、典型的なバグです。

<figure class="lec-fig"><svg viewBox="0 0 600 210" role="img" aria-label="Mask R-CNNのmasksは確率マップでsqueezeと0.5の二値化でboolにしてから描画する" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><text x="300" y="26" text-anchor="middle" font-size="14" font-weight="700" fill="#18181b">masks は確率 → squeeze(1) → 0.5 で二値化 → bool</text><rect x="48" y="66" width="24" height="24" fill="#fafafa" stroke="#ffffff" stroke-width="1"/><rect x="72" y="66" width="24" height="24" fill="#e4e4e7" stroke="#ffffff" stroke-width="1"/><rect x="96" y="66" width="24" height="24" fill="#fafafa" stroke="#ffffff" stroke-width="1"/><rect x="120" y="66" width="24" height="24" fill="#fafafa" stroke="#ffffff" stroke-width="1"/><rect x="48" y="90" width="24" height="24" fill="#d4d4d8" stroke="#ffffff" stroke-width="1"/><rect x="72" y="90" width="24" height="24" fill="#71717a" stroke="#ffffff" stroke-width="1"/><rect x="96" y="90" width="24" height="24" fill="#3f3f46" stroke="#ffffff" stroke-width="1"/><rect x="120" y="90" width="24" height="24" fill="#e4e4e7" stroke="#ffffff" stroke-width="1"/><rect x="48" y="114" width="24" height="24" fill="#d4d4d8" stroke="#ffffff" stroke-width="1"/><rect x="72" y="114" width="24" height="24" fill="#3f3f46" stroke="#ffffff" stroke-width="1"/><rect x="96" y="114" width="24" height="24" fill="#3f3f46" stroke="#ffffff" stroke-width="1"/><rect x="120" y="114" width="24" height="24" fill="#71717a" stroke="#ffffff" stroke-width="1"/><rect x="48" y="138" width="24" height="24" fill="#fafafa" stroke="#ffffff" stroke-width="1"/><rect x="72" y="138" width="24" height="24" fill="#71717a" stroke="#ffffff" stroke-width="1"/><rect x="96" y="138" width="24" height="24" fill="#71717a" stroke="#ffffff" stroke-width="1"/><rect x="120" y="138" width="24" height="24" fill="#e4e4e7" stroke="#ffffff" stroke-width="1"/><rect x="48" y="66" width="96" height="96" fill="none" stroke="#52525b" stroke-width="1.5"/><text x="96" y="184" text-anchor="middle" font-size="12.5" fill="#52525b">(N, 1, H, W)・float 確率</text><line x1="150" y1="114" x2="294" y2="114" stroke="#52525b" stroke-width="2"/><polygon points="302,114 292,109 292,119" fill="#52525b"/><text x="223" y="102" text-anchor="middle" font-size="13" font-weight="700" fill="#c2410c">squeeze(1) → 二値化</text><text x="223" y="132" text-anchor="middle" font-size="11.5" fill="#52525b">ch軸削除 → bool 化</text><rect x="312" y="66" width="24" height="24" fill="#ffffff" stroke="#e4e4e7" stroke-width="1"/><rect x="336" y="66" width="24" height="24" fill="#ffffff" stroke="#e4e4e7" stroke-width="1"/><rect x="360" y="66" width="24" height="24" fill="#ffffff" stroke="#e4e4e7" stroke-width="1"/><rect x="384" y="66" width="24" height="24" fill="#ffffff" stroke="#e4e4e7" stroke-width="1"/><rect x="312" y="90" width="24" height="24" fill="#ffffff" stroke="#e4e4e7" stroke-width="1"/><rect x="336" y="90" width="24" height="24" fill="#f97316" stroke="#e4e4e7" stroke-width="1"/><rect x="360" y="90" width="24" height="24" fill="#f97316" stroke="#e4e4e7" stroke-width="1"/><rect x="384" y="90" width="24" height="24" fill="#ffffff" stroke="#e4e4e7" stroke-width="1"/><rect x="312" y="114" width="24" height="24" fill="#ffffff" stroke="#e4e4e7" stroke-width="1"/><rect x="336" y="114" width="24" height="24" fill="#f97316" stroke="#e4e4e7" stroke-width="1"/><rect x="360" y="114" width="24" height="24" fill="#f97316" stroke="#e4e4e7" stroke-width="1"/><rect x="384" y="114" width="24" height="24" fill="#f97316" stroke="#e4e4e7" stroke-width="1"/><rect x="312" y="138" width="24" height="24" fill="#ffffff" stroke="#e4e4e7" stroke-width="1"/><rect x="336" y="138" width="24" height="24" fill="#f97316" stroke="#e4e4e7" stroke-width="1"/><rect x="360" y="138" width="24" height="24" fill="#f97316" stroke="#e4e4e7" stroke-width="1"/><rect x="384" y="138" width="24" height="24" fill="#ffffff" stroke="#e4e4e7" stroke-width="1"/><rect x="312" y="66" width="96" height="96" fill="none" stroke="#52525b" stroke-width="1.5"/><text x="360" y="184" text-anchor="middle" font-size="12.5" fill="#15803d">(N, H, W)・bool</text><line x1="414" y1="114" x2="468" y2="114" stroke="#52525b" stroke-width="2"/><polygon points="476,114 466,109 466,119" fill="#52525b"/><rect x="486" y="74" width="96" height="80" fill="#dbeafe" stroke="#2563eb" stroke-width="1.5"/><rect x="504" y="92" width="60" height="46" rx="6" fill="#71717a"/><rect x="504" y="92" width="60" height="46" rx="6" fill="#f97316" opacity="0.55"/><text x="534" y="174" text-anchor="middle" font-size="12" fill="#18181b">可視化(overlay)</text></svg><figcaption>Mask R-CNN の <code>masks</code> は <b>(N, 1, H, W) の確率マップ（0〜1）</b>で、bool ではありません。<code>squeeze(1)</code> でチャンネル軸を落として <b>(N, H, W)</b> にし、<b>0.5 で二値化</b>して bool にしてから <code>draw_segmentation_masks</code>（uint8 画像＋<b>bool</b> マスクを要求）へ渡します。確率のまま渡すのが典型的なバグです（左＝濃いほど高確率、右＝0.5 超だけが True）。</figcaption></figure>

```python
keep = pred["scores"] >= 0.5                  # スコア閾値でフィルタ
masks_prob = pred["masks"][keep]              # (M, 1, H, W) ★確率
masks_bool = (masks_prob.squeeze(1) > 0.5)    # (M, H, W) bool に二値化
canvas = draw_segmentation_masks(img_uint8, masks_bool, alpha=0.6)  # uint8 画像 + bool
```

このスニペットの `squeeze(1)` と `> 0.5` こそが後処理の核心です。なお、本講座の合成図形は COCO のクラス（人・車・犬…）に一致しないため、Mask R-CNN は自信を持って検出できず、**検出0件**になることがあります。とはいえこれはモデルの故障ではなく想定内の挙動であり、スクリプトは0件でも `draw` に空テンソルを渡さないよう早期分岐し、必ず `exit 0` で完了します。実写を `data/22_instance_panoptic_sam/` に置けば、人や車がきちんと検出され、実用的なオーバーレイになります。

## 3. Mask2Former でパノプティックセグメンテーション（`02_mask2former_panoptic.py`）

パノプティックの代表が **Mask2Former** です。これは「N 個の固定クエリが、それぞれ1枚のマスク＋1個のクラスを予測する」Transformer 型の統一アーキテクチャで、**同じ重みのまま後処理を変えるだけ**で instance / semantic / panoptic を出し分けられるのが最大の特徴です。HuggingFace では `post_process_panoptic_segmentation` がパノプティック専用の後処理で、`segmentation`（各画素=segment id の `(H,W)` マップ）と `segments_info`（各セグメントの `id / label_id / score`）を返します。CPU 前提では軽量な `swin-tiny` 版を使い、`swin-large` や ViT-Huge は避けます。

<figure class="lec-fig"><svg viewBox="0 0 540 250" role="img" aria-label="Mask2Formerは共通の重みで後処理を差し替えるだけでインスタンス・セマンティック・パノプティックの3タスクを出し分ける" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><text x="270" y="26" text-anchor="middle" font-size="14" font-weight="700" fill="#18181b">Mask2Former：1つの重み → 後処理で3タスクを出し分け</text><rect x="70" y="46" width="400" height="52" rx="8" fill="#fff7ed" stroke="#ea580c" stroke-width="2"/><text x="270" y="72" text-anchor="middle" font-size="14" font-weight="700" fill="#c2410c">Mask2Former（共通の重み）</text><text x="270" y="91" text-anchor="middle" font-size="12" fill="#52525b">N 個のクエリ → 各マスク + 1 クラス</text><line x1="270" y1="98" x2="270" y2="124" stroke="#71717a" stroke-width="2"/><line x1="99" y1="124" x2="441" y2="124" stroke="#71717a" stroke-width="2"/><line x1="99" y1="124" x2="99" y2="144" stroke="#71717a" stroke-width="2"/><polygon points="99,150 94,140 104,140" fill="#71717a"/><line x1="270" y1="124" x2="270" y2="144" stroke="#71717a" stroke-width="2"/><polygon points="270,150 265,140 275,140" fill="#71717a"/><line x1="441" y1="124" x2="441" y2="144" stroke="#71717a" stroke-width="2"/><polygon points="441,150 436,140 446,140" fill="#71717a"/><rect x="24" y="150" width="150" height="74" rx="8" fill="#fafafa" stroke="#71717a" stroke-width="2"/><rect x="195" y="150" width="150" height="74" rx="8" fill="#fafafa" stroke="#71717a" stroke-width="2"/><rect x="366" y="150" width="150" height="74" rx="8" fill="#eff6ff" stroke="#2563eb" stroke-width="2.5"/><text x="99" y="184" text-anchor="middle" font-size="14" font-weight="700" fill="#18181b">インスタンス</text><text x="99" y="206" text-anchor="middle" font-size="10.5" fill="#71717a">post_process_instance…</text><text x="270" y="184" text-anchor="middle" font-size="14" font-weight="700" fill="#18181b">セマンティック</text><text x="270" y="206" text-anchor="middle" font-size="10.5" fill="#71717a">post_process_semantic…</text><text x="441" y="180" text-anchor="middle" font-size="14" font-weight="700" fill="#1d4ed8">パノプティック</text><text x="441" y="199" text-anchor="middle" font-size="10.5" fill="#52525b">post_process_panoptic…</text><text x="441" y="217" text-anchor="middle" font-size="11" font-weight="700" fill="#c2410c">（本章）</text></svg><figcaption><b>Mask2Former</b> は<b>統一アーキテクチャ</b>が強みです。<b>1 つの重み</b>が「N 個のクエリ → 各クエリ 1 枚のマスク + 1 クラス」を予測し、<b>後処理を差し替えるだけ</b>で <code>post_process_instance / semantic / panoptic_segmentation</code> の 3 タスクを出し分けられます。本章で使うのは<b>パノプティック</b>で、<code>segmentation</code>（各画素＝segment id の (H,W) マップ）と <code>segments_info</code> を返します。</figcaption></figure>

ここでも座標系の落とし穴があります。`target_sizes` は `(height, width)` 順で渡しますが、PIL の `image.size` は `(width, height)` なので、`image.size[::-1]` と反転させる必要があります（検出の `post_process_object_detection` と同じ約束）。

```python
inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)
result = processor.post_process_panoptic_segmentation(
    outputs, target_sizes=[image.size[::-1]], threshold=0.5)[0]  # (W,H) を反転して (H,W) に
segmentation = result["segmentation"]      # (H, W) 各画素=segment id（-1=未割当）
segments_info = result["segments_info"]    # [{id, label_id, score, was_fused}]
```

合成図形では確信のある領域が見つからず `segments_info` が空（あるいはごく少数）になることがあります。その場合でも本スクリプトは、**各クエリが提案している生のマスク**（上位クエリを `masks_queries_logits` から sigmoid して可視化）を描き、「Mask2Former＝クエリ→マスク＋クラス」という仕組みそのものを見せます。PQ の数値計算はここでは行わず、GT を完全に制御できる §6（`04_*.py`）に回します——合成シーンの曖昧な検出結果で PQ を出しても学びが薄いからです。

## 4. SAM でプロンプト型セグメンテーション（`03_sam_prompt_seg.py`）

SAM は本章で唯一、**合成画像でも“絵になる”**デモです。クラスを当てないので、点や箱で指した対象をそのまま高精度に切り出せます。HF では `pip install segment-anything` ではなく `transformers` の `SamModel` / `SamProcessor`（`facebook/sam-vit-base`）を使います。CPU でさらに軽くしたいときは `MODEL_ID` を `Zigeng/SlimSAM-uniform-77` に変えるだけです。

正準フローには、注意点が2つあります。第一に、**点ラベルは 1=前景 / 0=背景**であり、取り違えるとマスクが反転します。第二に、SAM の `pred_masks` は **256×256 の低解像**で返るので、`processor.post_process_masks(...)` を通して原寸へ戻さないと位置が合いません。さらに SAM は曖昧性を考慮して**1プロンプトにつき3枚**のマスクを返すので、最終的には `iou_scores`（予測した品質）が最大の1枚を採ります。

<figure class="lec-fig"><svg viewBox="0 0 660 250" role="img" aria-label="SAMは1プロンプトに3枚のマスクを返しiou_scores最大の1枚を採用する" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><text x="330" y="28" text-anchor="middle" font-size="14" font-weight="700" fill="#18181b">SAM：1プロンプト → 3マスク → iou_scores 最大を採用</text><rect x="24" y="64" width="110" height="110" fill="#fafafa" stroke="#d4d4d8" stroke-width="1.5"/><rect x="44" y="84" width="70" height="70" rx="12" fill="#e4e4e7" stroke="#71717a" stroke-width="1.5"/><rect x="56" y="104" width="40" height="34" rx="6" fill="#d4d4d8" stroke="#71717a" stroke-width="1"/><circle cx="76" cy="120" r="11" fill="none" stroke="#dc2626" stroke-width="2"/><circle cx="76" cy="120" r="5" fill="#dc2626"/><line x1="140" y1="118" x2="190" y2="118" stroke="#52525b" stroke-width="2"/><polygon points="198,118 188,113 188,123" fill="#52525b"/><text x="168" y="106" text-anchor="middle" font-size="13" font-weight="700" fill="#c2410c">SAM</text><rect x="212" y="58" width="110" height="46" fill="#ffffff" stroke="#d4d4d8" stroke-width="1.5"/><rect x="232" y="68" width="22" height="26" rx="4" fill="#f97316"/><text x="332" y="86" font-size="12" fill="#52525b">IoU 0.88</text><rect x="212" y="110" width="110" height="46" fill="#fff7ed" stroke="#16a34a" stroke-width="3"/><rect x="224" y="118" width="74" height="30" rx="6" fill="#f97316"/><text x="332" y="138" font-size="13" font-weight="700" fill="#15803d">IoU 0.97</text><rect x="212" y="162" width="110" height="46" fill="#ffffff" stroke="#d4d4d8" stroke-width="1.5"/><rect x="230" y="172" width="40" height="26" rx="5" fill="#f97316"/><text x="332" y="190" font-size="12" fill="#52525b">IoU 0.85</text><line x1="400" y1="133" x2="452" y2="133" stroke="#52525b" stroke-width="2"/><polygon points="460,133 450,128 450,138" fill="#52525b"/><text x="426" y="122" text-anchor="middle" font-size="12" font-weight="700" fill="#c2410c">argmax</text><rect x="466" y="80" width="120" height="106" fill="#fff7ed" stroke="#16a34a" stroke-width="2.5"/><rect x="486" y="100" width="80" height="60" rx="8" fill="#f97316"/><text x="526" y="204" text-anchor="middle" font-size="13" font-weight="700" fill="#15803d">採用マスク</text></svg><figcaption>SAM はクラスを当てず、<b>点プロンプト（1=前景／0=背景）</b>で指した“もの”を切り出します。曖昧さに備えて <b>1 プロンプトにつき 3 枚</b>のマスク（部分・全体など）を返すので、自己申告の品質 <code>iou_scores</code> が<b>最大の 1 枚</b>を <code>argmax</code> で採ります。<code>pred_masks</code> は 256×256 と低解像なので <code>post_process_masks</code> で<b>原寸へ戻す</b>のを忘れないこと。<code>iou_scores</code> は GT との真の IoU ではありません（1.0 を超えることもある）。</figcaption></figure>

```python
inputs = processor(image, input_points=[[[x, y]]], input_labels=[[1]], return_tensors="pt")
outputs = model(**inputs)                              # pred_masks:(B,1,3,256,256)
masks = processor.post_process_masks(                   # 低解像→原寸（必須）
    outputs.pred_masks, inputs["original_sizes"], inputs["reshaped_input_sizes"])[0][0]  # (3,H,W)
best = masks[outputs.iou_scores[0, 0].argmax()]         # 3枚から品質最大を選ぶ
```

ここで強調したいのは、`iou_scores` は**モデルが自己申告した品質スコア**であって、GT との真の IoU ではない（1.0 を超えることすらある）点です。スクリプトは合成 GT がある場合に**真の IoU**も併記し、両者を区別します。実際に合成シーンで動かすと、点プロンプト・箱プロンプトともに **GT との IoU が 0.98 前後**という鮮やかな結果になり、「クラスを知らなくても形は正確に切れる」という SAM の本質が体感できます。

## 5. mask AP と RLE（`04_maskap_pq_eval.py` 前半）

インスタンスの評価は、第19回の物体検出 mAP を**そのまま**マスクへ拡張します。やることは1つだけ——マッチングに使う IoU を、**box IoU から mask IoU（交差画素 / 和集合画素）に置き換える**。あとは予測を confidence 降順に並べ、未マッチの GT に IoU≥閾値で貪欲対応し、TP/FP を累積して PR 曲線→AP を出す、という流れは検出と同一です。これが **mask AP** で、`COCOeval(iouType="segm")` が公式実装です。

<figure class="lec-fig"><svg viewBox="0 0 660 175" role="img" aria-label="mask APの計算手順。confidence降順ソート、IoU閾値で貪欲マッチ、TP FP累積、PR曲線、APの順。検出mAPと同じでIoUをboxからmaskに替えるだけ" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><text x="330" y="24" text-anchor="middle" font-size="14" font-weight="700" fill="#18181b">mask AP の計算手順 — 検出の mAP と同じ流れ</text><rect x="108" y="42" width="180" height="28" rx="6" fill="#ffedd5" stroke="#c2410c" stroke-width="1.5"/><text x="198" y="61" text-anchor="middle" font-size="12" font-weight="700" fill="#c2410c">IoU を box→mask に置換</text><line x1="198" y1="70" x2="198" y2="86" stroke="#c2410c" stroke-width="2"/><polygon points="198,92 193,82 203,82" fill="#c2410c"/><rect x="16" y="92" width="100" height="58" rx="7" fill="#fff7ed" stroke="#ea580c" stroke-width="2"/><rect x="148" y="92" width="100" height="58" rx="7" fill="#ffedd5" stroke="#c2410c" stroke-width="2.5"/><rect x="280" y="92" width="100" height="58" rx="7" fill="#fff7ed" stroke="#ea580c" stroke-width="2"/><rect x="412" y="92" width="100" height="58" rx="7" fill="#fff7ed" stroke="#ea580c" stroke-width="2"/><rect x="544" y="92" width="100" height="58" rx="7" fill="#eff6ff" stroke="#2563eb" stroke-width="2"/><text x="66" y="117" text-anchor="middle" font-size="12.5" font-weight="700" fill="#c2410c">① confidence</text><text x="66" y="137" text-anchor="middle" font-size="11" fill="#52525b">降順ソート</text><text x="198" y="117" text-anchor="middle" font-size="12.5" font-weight="700" fill="#c2410c">② 貪欲マッチ</text><text x="198" y="137" text-anchor="middle" font-size="11" fill="#52525b">IoU≥閾値で対応</text><text x="330" y="117" text-anchor="middle" font-size="12.5" font-weight="700" fill="#c2410c">③ TP/FP を</text><text x="330" y="137" text-anchor="middle" font-size="11" fill="#52525b">累積</text><text x="462" y="126" text-anchor="middle" font-size="13" font-weight="700" fill="#c2410c">④ PR 曲線</text><text x="594" y="126" text-anchor="middle" font-size="13" font-weight="700" fill="#1d4ed8">⑤ AP</text><line x1="116" y1="121" x2="142" y2="121" stroke="#71717a" stroke-width="2"/><polygon points="148,121 138,116 138,126" fill="#71717a"/><line x1="248" y1="121" x2="274" y2="121" stroke="#71717a" stroke-width="2"/><polygon points="280,121 270,116 270,126" fill="#71717a"/><line x1="380" y1="121" x2="406" y2="121" stroke="#71717a" stroke-width="2"/><polygon points="412,121 402,116 402,126" fill="#71717a"/><line x1="512" y1="121" x2="538" y2="121" stroke="#71717a" stroke-width="2"/><polygon points="544,121 534,116 534,126" fill="#71717a"/></svg><figcaption><b>mask AP</b> の計算は、第19回の検出 mAP と<b>同じ流れ</b>です。違いはただ 1 つ、マッチングの <b>IoU を box から mask（交差画素/和集合画素）へ置き換える</b>こと。あとは <b>① 予測を confidence 降順にソート → ② 未マッチ GT へ IoU≥閾値で貪欲マッチ → ③ TP/FP を累積 → ④ PR 曲線 → ⑤ その下の面積＝AP</b> と進みます。公式実装は <code>COCOeval(iouType="segm")</code> で、マスクは <b>RLE</b> で渡します。</figcaption></figure>

公式実装にマスクを渡すには **RLE（Run-Length Encoding）** という圧縮形式を使います。`pycocotools.mask.encode` が bool マスク（Fortran 連続=列優先で渡すのが約束）を RLE 辞書 `{"size", "counts"}` に変換し、`area` や `bbox` もこの RLE から計算できます。GT は COCO 形式の dict（`images` には `height/width` が必須）、予測は `[{image_id, category_id, segmentation(RLE), score}]` のリストで渡します。

```python
rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))  # bool → RLE（counts は bytes）
# GT: images に height/width、annotations に segmentation=RLE, area, bbox, iscrowd
ev = COCOeval(coco_gt, coco_dt, iouType="segm")  # box ではなく segm
ev.evaluate(); ev.accumulate(); ev.summarize()
mAP, AP50, AP75 = ev.stats[0], ev.stats[1], ev.stats[2]
```

スクリプトは、`numpy` で組んだ素朴な **AP@0.5**（PR 全点積分）と、`COCOeval` の公式 segm AP を並べて表示します。これにより「自分の手で書いた AP の考え方」と「論文比較で正準とされる実装」が地続きであることを確認できます。`COCOeval` の `summarize()` は `AP_S/M/L`（面積別）や `AR`（maxDets 別）も出すので、`areaRng` と `maxDets` の既定値が結果を左右することも目で追えます。

## 6. パノプティック PQ = SQ × RQ（`04_maskap_pq_eval.py` 後半）

パノプティックは things と stuff を全画素に割り当てるため、AP とは別の指標 **PQ（Panoptic Quality）** で測ります。PQ はカテゴリごとに次の3つを計算し、全カテゴリで平均します。まず IoU > 0.5 を満たす GT-予測セグメントを**一意マッチ**させます（0.5 超なので相手は高々1つに定まる、というのが PQ の設計の妙）。マッチを TP、余った予測を FP、取りこぼした GT を FN とすると、

- **SQ（Segmentation Quality）= マッチした組の平均 IoU** … 当てた領域の「形の良さ」
- **RQ（Recognition Quality）= TP / (TP + 0.5 FP + 0.5 FN)** … 検出の F1（過不足の少なさ）
- **PQ = SQ × RQ** … 形の良さと検出の確かさの積

この分解が PQ の読み方を明快にします。SQ が高く RQ が低ければ「当てた所は綺麗だが取りこぼし/誤検出が多い」、逆なら「数は合うが輪郭が雑」。スクリプトでは、わざと**取りこぼし（FN）と誤検出（FP）を1つずつ仕込んだ**合成パノプティックを使い、RQ が 1 未満（この設定で約 0.83）になる様子を見せます。

<figure class="lec-fig"><svg viewBox="0 0 660 250" role="img" aria-label="PQ評価はIoU0.5超で一意マッチしTP FP FNからSQとRQを出しPQはSQかけるRQ" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><text x="88" y="48" text-anchor="middle" font-size="15" font-weight="700" fill="#1d4ed8">GT</text><text x="250" y="48" text-anchor="middle" font-size="15" font-weight="700" fill="#c2410c">予測</text><line x1="114" y1="78" x2="224" y2="78" stroke="#16a34a" stroke-width="3"/><line x1="114" y1="128" x2="224" y2="128" stroke="#16a34a" stroke-width="3"/><ellipse cx="88" cy="78" rx="26" ry="17" fill="#dbeafe" stroke="#2563eb" stroke-width="2"/><ellipse cx="88" cy="128" rx="26" ry="17" fill="#dbeafe" stroke="#2563eb" stroke-width="2"/><ellipse cx="250" cy="78" rx="26" ry="17" fill="#ffedd5" stroke="#ea580c" stroke-width="2"/><ellipse cx="250" cy="128" rx="26" ry="17" fill="#ffedd5" stroke="#ea580c" stroke-width="2"/><ellipse cx="88" cy="182" rx="26" ry="17" fill="#fff7ed" stroke="#dc2626" stroke-width="2" stroke-dasharray="4 3"/><ellipse cx="250" cy="182" rx="26" ry="17" fill="#fff7ed" stroke="#dc2626" stroke-width="2" stroke-dasharray="4 3"/><text x="128" y="187" font-size="13" font-weight="700" fill="#dc2626">FN</text><text x="210" y="187" text-anchor="end" font-size="13" font-weight="700" fill="#dc2626">FP</text><rect x="372" y="60" width="266" height="150" rx="8" fill="#fafafa" stroke="#d4d4d8" stroke-width="1.5"/><text x="505" y="98" text-anchor="middle" font-size="18" font-weight="700" fill="#c2410c">PQ = SQ × RQ</text><text x="390" y="142" font-size="13" fill="#18181b">SQ = マッチ組の平均 IoU</text><text x="390" y="178" font-size="13" fill="#18181b">RQ = TP / (TP + 0.5FP + 0.5FN)</text></svg><figcaption>パノプティックは検出系の AP ではなく <b>PQ</b> で測ります。GT と予測のセグメントを <b>IoU &gt; 0.5</b> で一意マッチさせ（緑線＝マッチ＝<b>TP</b>）、余った予測を <b>FP</b>、取りこぼした GT を <b>FN</b>（赤の破線）とします。<b>SQ</b>＝マッチ組の平均 IoU（形の良さ）、<b>RQ</b>＝TP/(TP+0.5FP+0.5FN)（検出の F1）、そして <b>PQ＝SQ×RQ</b>。図は TP×2・FP×1・FN×1 の模式図で、カテゴリ別に出して最後に平均します。</figcaption></figure>

```python
# preds/target は (H, W, 2): 最後の次元が (category_id, instance_id)
metric = PanopticQuality(things={1, 2}, stuffs={10, 11}, return_sq_and_rq=True)
pq, sq, rq = metric(torch.tensor(pred)[None], torch.tensor(gt)[None])
# 自作(numpy)のカテゴリ別集計→平均が、上の torchmetrics と一致する
assert np.isclose(pq_manual, float(pq), atol=1e-3)
```

`torchmetrics.detection.PanopticQuality` の入力は `(B, H, W, 2)` の整数テンソルで、最後の次元が `(category_id, instance_id)` の組です。`things` は数えられるカテゴリ、`stuffs` は背景カテゴリの ID 集合で、どちらにも属さない ID は void として無視されます。スクリプトは `numpy` でカテゴリ別に SQ/RQ/PQ を集計して平均する自作実装を書き、それが torchmetrics と**小数第3位まで一致**することを `assert` で保証します。この「自作＝公式」の一致体験こそ、PQ を式で理解できた証拠です。

## 7. なぜ合成画像か／実写で試すには

本モジュールが入力を合成生成する理由は2つあります。第一に、本講座は**ネットへ出るのをモデル重みの DL だけ**に限定したいので、入力画像はローカルで決定的に作ります。第二に、評価指標（mask AP / PQ）は「予測と GT の形さえ分かっていれば」計算でき、**GT を完全に制御できる合成データの方がむしろ学びやすい**からです。実際 §6 の PQ は、FN/FP を意図的に1つずつ仕込んで RQ の挙動を狙い通り再現しています。

一方で、Mask R-CNN や Mask2Former は COCO の実物体を覚えているので、抽象的な合成図形には自信を持てず**検出が乏しく（0件のことも）**なります。これは正常な挙動で、各スクリプトは0件でも例外を出さず `exit 0` します。実写で本来の力を見たいときは、`data/22_instance_panoptic_sam/` に `.png` / `.jpg` を1枚置いてください。各スクリプトの `load_user_or_synthetic()` が自動でそれを優先し、人・車・空・地面などに対する本物のインスタンス/パノプティック結果が得られます（その場合 GT マスクは無いので、SAM の真 IoU 等は表示されません）。

## 8. このモジュールの構成（スクリプト一覧）

各スクリプトは単一責務で、上から順に読めば「インスタンス → パノプティック → プロンプト型 → 評価」と理解が積み上がります。すべて `lectures/22_instance_panoptic_sam/outputs/` に図と JSON を保存し、画面表示（`cv2.imshow`）には依存しません。合成シーンと GT マスクの生成、device 判定、可視化の小道具は `seg_helpers.py` にまとめ、各スクリプトはそれを import します。

| ファイル | 役割（単一責務） |
| --- | --- |
| `seg_helpers.py` | 合成シーン＋GT インスタンスの生成、device、ラベルマップのカラー化、図の保存。各スクリプトが import する道具箱 |
| `01_maskrcnn_instance.py` | Mask R-CNN でインスタンス。`masks (N,1,H,W)` 確率→`>0.5` で bool 化、`draw_segmentation_masks` で可視化 |
| `02_mask2former_panoptic.py` | Mask2Former でパノプティック。`post_process_panoptic_segmentation` の `segmentation`/`segments_info`、クエリ可視化 |
| `03_sam_prompt_seg.py` | SAM の点/箱プロンプト。`post_process_masks` で原寸化、3マスクから `iou_scores` 最大を選択、GT との真 IoU |
| `04_maskap_pq_eval.py` | mask AP（`COCOeval` segm + RLE）と PQ=SQ×RQ（自作 numpy ＝ torchmetrics を assert で照合） |
| `mini_project.py` | **章末ミニプロジェクト**。SAM の点プロンプトで各物体を切り出し、`mask AP@0.5`（自作）と `PQ=SQ×RQ`（自作＝torchmetrics）で採点。同じ予測でも AP と PQ で FP の効き方が違うことを数値で見せる総合課題 |
| `use_case.py` | **実践ユースケース**。SAM の点プロンプトで指した物体を切り出し、背景を透明にした **RGBA PNG** を書き出す「クリック切り抜きツール」。採点はせず“納品物（透過素材）”を作る現実の小ツール（`mini_project.py` の評価系とは別物） |
| `exercises.py` | TODO 形式の演習（**全10問**・易→難・自己採点ランナー付き。`SHOW_SOLUTION=1` で模範解答に差し替え） |
| `exercises_solutions.py` | 演習の完成形（全10問 PASS）。採点ロジックは `exercises.py` 側を再利用（重複なし） |

表の通り `seg_helpers.py` だけは「読み物」ではなく「再利用する道具」です。中身も厚くコメントしてあるので、最初に一読してから 01 へ進むと、各スクリプトが何を題材に実験しているかが腑に落ちます。`mini_project.py` は 01〜04 を一通り終えてから取り組むと、各部品が1本の評価パイプラインに統合される様子が掴めます。

## 9. 動かし方

このモジュールは `torch` / `torchvision`（`dl`）、`transformers`（`hf`）、`pycocotools` / `torchmetrics`（`metrics`）に依存します。初回実行時のみ Mask R-CNN・Mask2Former(swin-tiny)・SAM(vit-base) の重みが自動 DL されます（数百 MB 規模。以後はキャッシュされ高速）。プロジェクトルートで以下を順に実行してください。

```bash
# 依存グループを用意（初回のみ）
uv sync --group dl --group hf --group metrics

# 各スクリプトを実行（結果は lectures/22_instance_panoptic_sam/outputs/ に保存される）
uv run python lectures/22_instance_panoptic_sam/seg_helpers.py             # 道具箱のスモークテスト
uv run python lectures/22_instance_panoptic_sam/01_maskrcnn_instance.py    # インスタンス
uv run python lectures/22_instance_panoptic_sam/02_mask2former_panoptic.py # パノプティック
uv run python lectures/22_instance_panoptic_sam/03_sam_prompt_seg.py       # SAM（点/箱）
uv run python lectures/22_instance_panoptic_sam/04_maskap_pq_eval.py       # mask AP・PQ

# 章末ミニプロジェクト（SAM で対話セグメンテーション → mask AP / PQ で採点）
uv run python lectures/22_instance_panoptic_sam/mini_project.py

# 実践ユースケース（SAM の点プロンプトで物体を切り出し透明背景 PNG を作る小ツール）
uv run python lectures/22_instance_panoptic_sam/use_case.py

# 演習: まずは TODO を自分で埋める（最初は全部 FAIL。それでも exit 0 で落ちない）
uv run python lectures/22_instance_panoptic_sam/exercises.py
# どうしても分からない時だけ、模範解答の挙動を見る
SHOW_SOLUTION=1 uv run python lectures/22_instance_panoptic_sam/exercises.py
# 全問の完成形（ALL PASS を確認）
uv run python lectures/22_instance_panoptic_sam/exercises_solutions.py
```

実行後は `lectures/22_instance_panoptic_sam/outputs/` の画像と JSON を確認してください。`03_sam_prompt_seg.png`（点/箱で指した領域が綺麗に切れている）と `04_eval_metrics.json`（自作 PQ と torchmetrics が一致）を、本文の解説と照らし合わせると理解が定着します。実写で試したい場合は §7 の通り `data/22_instance_panoptic_sam/` に画像を置いてから再実行します。

> 補足: `needs_groups` には概念紹介として **Ultralytics SAM**（`SAM('mobile_sam.pt')` / `SAM('sam2.1_t.pt')`）も挙げられますが、`ultralytics` は `opencv-python`（full 版）を引き込み、本講座既定の `opencv-python-headless` と**衝突**します。本スクリプトは衝突を避けるため HF SAM のみを実行経路に使い、Ultralytics 版は「軽量・1行で動く別実装」として概念に留めます（試すなら別環境で `uv add ultralytics`）。

## 10. よくある落とし穴（チェックリスト）

最後に、この章でつまずきやすい点を「症状 → 原因 → 対処」でまとめます。実装中に詰まったら、まずここを見てください。

| 症状 | ほぼ確実な原因 | 対処 |
| --- | --- | --- |
| `draw_segmentation_masks` が例外/真っ黒 | masks が `(N,1,H,W)` の float 確率のまま | `squeeze(1) > 0.5` で `(N,H,W)` の **bool** にする |
| マスクや bbox が画像とズレる | `target_sizes` に `(W,H)` を渡した | `(height, width)` 順＝`image.size[::-1]` を渡す |
| `draw_*` で「uint8 を要求」と怒られる | float の画像テンソルを渡した | 画像を **uint8** の `(3,H,W)` にしてから描画 |
| SAM のマスクが小さい/ずれる | 256×256 の低解像のまま使った | `processor.post_process_masks(...)` で原寸へ戻す |
| SAM のマスクが反転する | 点ラベルの前景/背景を取り違え | `input_labels` は **1=前景 / 0=背景** |
| `iou_scores` を真の IoU と誤解 | SAM の自己申告品質を IoU と混同 | 評価は GT との実 IoU を別に計算（1超もある） |
| `COCOeval` が `KeyError: height` | `images` に `height/width` が無い | 画像メタに必ず `height` と `width` を入れる |
| RLE 化で値がおかしい | C 連続のまま `mask.encode` した | `np.asfortranarray(...)`（列優先）で渡す |
| 合成画像で検出 0 件 | COCO クラスに無い抽象図形 | 想定内。実写を `data/` に置くか、SAM/評価で学ぶ |
| 自作 PQ が torchmetrics と合わない | カテゴリ別→平均でなく全体で集計した | カテゴリごとに SQ/RQ を出し**最後に平均** |

この10項目が、インスタンス/パノプティック/SAM でつまずく原因のほぼ全てです。逆にいえば、これらを自分の言葉で説明でき、回避コードを書けるようになれば、この章のゴールに到達しています。

## 11. まとめ

この章では、セグメンテーションが**セマンティック/インスタンス/パノプティック**の3タスクに分かれること、そして**SAM** がクラス非依存のプロンプト型として独立した軸にあることを、出力フォーマットと後処理のレベルで押さえました。Mask R-CNN の `(N,1,H,W)` 確率マスク、Mask2Former のクエリ→`segments_info`、SAM の3マスク＋`post_process_masks` という「最初の関門」を一つずつ通り抜け、評価では mask AP（`COCOeval` segm + RLE）と PQ=SQ×RQ（自作＝torchmetrics）を式で理解しました。

次回（第23回）は、ここで学んだ SAM を **テキストプロンプト**で動かす方向へ進みます。CLIPSeg で「文で指定した領域」を直接マスク化し、さらに Grounding DINO の検出 box を SAM の `input_boxes` に渡す **Grounded-SAM** の2段構成へ。本章の「プロンプト型セグメンテーション」と「IoU/Dice 評価」が、そのまま下地になります。まずは演習を自力で全問 PASS させ、`assert` で自作と公式実装の一致を体感してから次へ進んでください。

---

## 🛠 章末ミニプロジェクト（`mini_project.py`）

ここまでの「動かす（01〜03）」と「測る（04）」を、1本の評価パイプラインに統合します。テーマは **「クラスを知らない SAM を“点で教えて”インスタンス分割し、mask AP と PQ で採点する」**。具体的には、

1. 合成シーンの **各 GT 物体の重心へ点プロンプト**を打ち、SAM で物体を切り出す（＝対話的インスタンスセグメンテーション）。SAM はクラス非依存なので、プロンプト元の GT のクラスをそのまま予測ラベルに使います。
2. さらに **背景にも1点**打ち、わざと **低スコアの誤検出（FP）** を1つ仕込みます。
3. 得た予測マスク群を、本章の2指標 **mask AP@0.5（自作・全点補間）** と **PQ＝SQ×RQ（自作 ＝ `torchmetrics`）** で採点します。

<figure class="lec-fig"><svg viewBox="0 0 660 250" role="img" aria-label="章末ミニプロジェクトの3ステップ。各GT重心へ点プロンプトでSAM切り出し、背景に1点で低スコアの誤検出を仕込み、mask APとPQで採点する" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><text x="330" y="28" text-anchor="middle" font-size="14" font-weight="700" fill="#18181b">ミニプロジェクト：点で教える SAM を 2 指標で採点</text><rect x="22" y="56" width="186" height="70" rx="8" fill="#fff7ed" stroke="#ea580c" stroke-width="2"/><rect x="238" y="56" width="186" height="70" rx="8" fill="#fff7ed" stroke="#ea580c" stroke-width="2"/><rect x="454" y="56" width="186" height="70" rx="8" fill="#eff6ff" stroke="#2563eb" stroke-width="2"/><text x="115" y="86" text-anchor="middle" font-size="14" font-weight="700" fill="#c2410c">① GT重心へ点プロンプト</text><text x="115" y="108" text-anchor="middle" font-size="12" fill="#52525b">SAM で各物体を切り出す</text><text x="331" y="86" text-anchor="middle" font-size="14" font-weight="700" fill="#c2410c">② 背景にも1点</text><text x="331" y="108" text-anchor="middle" font-size="12" fill="#52525b">低スコアの誤検出(FP)を1個</text><text x="547" y="86" text-anchor="middle" font-size="14" font-weight="700" fill="#1d4ed8">③ 2指標で採点</text><text x="547" y="108" text-anchor="middle" font-size="12" fill="#52525b">mask AP / PQ=SQ×RQ</text><line x1="208" y1="91" x2="232" y2="91" stroke="#71717a" stroke-width="2"/><polygon points="238,91 228,86 228,96" fill="#71717a"/><line x1="424" y1="91" x2="448" y2="91" stroke="#71717a" stroke-width="2"/><polygon points="454,91 444,86 444,96" fill="#71717a"/><line x1="547" y1="126" x2="547" y2="152" stroke="#71717a" stroke-width="2"/><polygon points="547,158 542,148 552,148" fill="#71717a"/><rect x="388" y="158" width="252" height="78" rx="8" fill="#fafafa" stroke="#d4d4d8" stroke-width="1.5"/><text x="514" y="180" text-anchor="middle" font-size="12.5" font-weight="700" fill="#18181b">同じ予測でも FP の効き方が違う</text><rect x="402" y="190" width="110" height="30" rx="5" fill="#dbeafe" stroke="#2563eb" stroke-width="1.5"/><rect x="520" y="190" width="110" height="30" rx="5" fill="#ffedd5" stroke="#ea580c" stroke-width="1.5"/><text x="457" y="210" text-anchor="middle" font-size="12.5" font-weight="700" fill="#1d4ed8">mask AP ≈ 1.000</text><text x="575" y="210" text-anchor="middle" font-size="12.5" font-weight="700" fill="#c2410c">PQ ≈ 0.69</text><text x="514" y="231" text-anchor="middle" font-size="11" fill="#71717a">AP=末尾FPに鈍感／PQ=FPでRQ低下</text></svg><figcaption><b>章末ミニプロジェクト</b>の流れです。<b>① 各 GT 物体の重心へ点プロンプト</b>を打って SAM で物体を切り出し、<b>② 背景にも 1 点</b>打って<b>低スコアの誤検出(FP)を 1 個</b>わざと仕込み、<b>③ mask AP@0.5 と PQ＝SQ×RQ</b>で採点します。観察の勘どころは、低スコア FP は <b>confidence 降順の末尾</b>に来るので <code>mask AP</code> をほぼ下げない一方、<b>PQ は未マッチの FP で必ず RQ が下がる</b>（この設定で PQ≈0.69）点です。</figcaption></figure>

このとき観察してほしいのが、**同じ予測でも AP と PQ で FP の効き方が違う**ことです。低スコアの FP は confidence 降順の末尾に来るので **mAP@0.5 は 1.000 のまま**（スコアが低い誤検出は AP をほぼ下げない）。一方 PQ は集合ベースなので、未マッチの予測＝FP として **RQ を必ず下げ**（この設定で PQ≈0.69）ます。「AP が高いのに PQ が低い」モデルは“当てた所は綺麗だが余計な検出が多い”——という読み方が、数値の対比から腑に落ちます。

実装面では、SAM の **interactive な使い方の肝**も入れてあります。重い画像エンコーダ（`get_image_embeddings`）は**1回だけ**走らせ、各プロンプトの `forward` には `image_embeddings=` でその埋め込みを**使い回す**ので、CPU でも複数点を高速に処理できます。SAM が読み込めない環境では、GT を少し膨張させた疑似予測へ自動フォールバックし、採点パイプライン自体は必ず完走します（合成データなので「モデルが無くても exit 0」）。

```bash
uv run python lectures/22_instance_panoptic_sam/mini_project.py
# → lectures/22_instance_panoptic_sam/outputs/mini_project.png（input / GT / SAM予測 / overlay）
#   lectures/22_instance_panoptic_sam/outputs/mini_project_report.json（per予測IoU・mask AP・PQ）
```

**発展課題**（自分で改造してみる）: ①点プロンプトを**箱プロンプト**に替えて IoU/PQ がどう変わるか比べる。②背景 FP の `score` を高くして、AP が下がり始める閾値を探す（AP の score 依存性の体感）。③`mask AP@0.5` を **AP@[.5:.95]**（IoU 閾値を 0.50:0.05:0.95 で平均）に拡張し、`04` の `COCOeval` segm と突き合わせる。

## ✅ 到達チェックリスト

この章を「理解した」と言える状態かを、次の項目で自己点検してください（言葉で説明でき、かつコードで再現できれば合格）。

- [ ] セグメンテーションの **3タスク（セマンティック/インスタンス/パノプティック）＋SAM** を、出力フォーマットの違いで説明できる。
- [ ] Mask R-CNN の `masks` が **`(N,1,H,W)` の確率**であることを知り、`squeeze(1) > 0.5` で **bool** 化して `draw_segmentation_masks` に渡せる。
- [ ] `post_process_*` の `target_sizes` が **`(H,W)` 順**（`image.size[::-1]`）だと分かり、座標ズレを自力で直せる。
- [ ] SAM の **3マスク**から `iou_scores` 最大を選び、`post_process_masks` で**原寸化**できる。`iou_scores`（自己申告品質）と **GT との真の IoU** を区別できる。
- [ ] **mask IoU**（交差画素/和集合画素）を書け、score 降順の**貪欲マッチ**で TP/FP/FN を数えられる。
- [ ] PR 曲線から **全点補間 AP** と **11点補間 AP** を numpy で計算でき、両者の違いを説明できる。
- [ ] クラス横断の **mask AP@0.5（mAP）** を、クラス別 AP の平均として組める。
- [ ] **PQ＝SQ×RQ** の式（SQ＝マッチ組の平均 IoU、RQ＝TP/(TP+0.5FP+0.5FN)）を書け、**カテゴリ別に出して最後に平均**する手順を守れる。
- [ ] 自作 PQ が `torchmetrics.detection.PanopticQuality` と**小数第3位まで一致**することを `assert` で確認できる。
- [ ] COCO の **RLE（列優先・背景始まり）** の意味が分かり、counts からマスクを復元できる。
- [ ] `exercises.py` の**全10問**を自力で PASS でき、`mini_project.py` の AP と PQ の対比を説明できる。

## ✍️ 演習問題

演習は `exercises.py` に TODO 形式で入っています。各 TODO を実装し `uv run python lectures/22_instance_panoptic_sam/exercises.py` を実行すると自己採点できます（`exercises_solutions.py` が解答）。

1. 2つの bool マスクの **mask IoU**（交差画素 / 和集合画素、和集合が 0 のときは 0.0）を返す（`ex1_mask_iou`）。
2. Mask R-CNN の `masks (N,1,H,W)` 確率マップを、チャンネル次元を潰して閾値超を `True` にした `(N,H,W)` の **bool** へ二値化する（`ex2_binarize_masks`）。
3. 予測を score 降順に並べ、未マッチの GT へ IoU≥閾値で貪欲マッチして `(TP, FP, FN)` を数える（`ex3_match_counts`）。
4. 1カテゴリぶんの `(SQ, RQ, PQ)`（SQ＝マッチ組の平均 IoU、RQ＝TP/(TP+0.5FP+0.5FN)、PQ＝SQ×RQ）を返す（`ex4_pq_single_category`）。
5. SAM が返す3枚のマスクから `iou_scores` 最大の1枚 `(H,W)` を選ぶ（`ex5_select_best_sam_mask`）。
6. PR 曲線から **全点補間（all-point）** で AP を求める。両端を補い、precision の包絡線を取り、recall 変化点の長方形面積を積む（`ex6_ap_all_point`）。
7. PR 曲線から VOC2007 流の **11点補間** で AP を求める（recall 閾値 0.0〜1.0 の11点で precision 最大値を平均）（`ex7_ap_11point`）。
8. クラスを跨いだ **mask AP@0.5（mAP）** を、カテゴリ別の AP を平均して求める（`ex8_mask_ap50`）。
9. COCO の **非圧縮 RLE**`counts`（列優先・背景始まり）を `(H,W)` の bool マスクへ復元する（`ex9_rle_decode`）。
10. カテゴリ別の `(gt_masks, pred_masks)` から全体 `(SQ, RQ, PQ)` を、カテゴリごとに出して最後に平均して求める（`ex10_pq_overall`）。

## ❓ よくある落とし穴・FAQ・デバッグ

§10 の「症状→原因→対処」表に加え、つまずきやすい疑問を Q&A 形式でまとめます。

**Q1. SAM の `iou_scores` が 1.0 を超えるのはバグ？** いいえ。`iou_scores` は SAM が**自己申告する品質の回帰値**で、真の IoU の推定であって [0,1] に厳密にクリップされません。1.0 を超えることもあります。本当の精度が知りたければ **GT との IoU を別途計算**します（`03` と `mini_project.py` はこの2つを併記）。

**Q2. mAP は 1.0 なのに PQ が低い。どちらが正しい？** どちらも正しく、**測っているものが違う**だけです。mAP は confidence でランク付けした検出の質（低スコア FP は末尾なので効きにくい）。PQ は集合の質（FP は必ず RQ を下げる）。`mini_project.py` はこの差をわざと作って見せています。「AP が高い＝良いモデル」と早合点しないこと。

**Q3. 自作 PQ が `torchmetrics` と合いません。** ほぼ次のどれか。①**全体で集計**してしまった（正しくは**カテゴリ別→最後に平均**）。②`things`/`stuffs` に入れていない ID を評価に混ぜた（どちらでもない ID は **void** で無視される）。③入力テンソルの最後の次元が `(category_id, instance_id)` の順になっていない。④`stuff` の `instance_id` を物体ごとに変えてしまった（stuff は1カテゴリ1セグメントとして `instance_id` を固定）。

**Q4. `COCOeval` が `KeyError: 'height'` で落ちる。** GT dict の `images` に **`height`/`width` が無い**のが原因。COCO の画像メタには必ず両方を入れます。あわせて RLE は `np.asfortranarray(...)`（**列優先**）で `mask.encode` に渡します（C 連続のままだと値が壊れます）。

**Q5. 合成画像で Mask R-CNN / Mask2Former の検出が 0 件です。** **想定内**です。これらは COCO の実物体を覚えているので、抽象図形には確信を持てません（モデルの故障ではない）。評価指標は GT を完全制御できる合成データ（`04`・`mini_project.py`）で学び、実写を見たいときは `data/22_instance_panoptic_sam/` に画像を置きます。

**Q6. CPU で遅い／メモリが厳しい。** `swin-large`・ViT-Huge・`blip2` 級は CPU では非現実的。本章既定の `mask2former-swin-tiny`・`sam-vit-base`（or `SlimSAM-uniform-77`）を使い、SAM は **画像埋め込みを1回だけ**計算してプロンプトで使い回す（`mini_project.py` 参照）。必要に応じて `torch.set_num_threads(物理コア数)`。

**デバッグの定石**: マスク系で結果が変なときは、まず **`mask.dtype`（bool か）・`mask.shape`（`(H,W)` か `(N,H,W)` か）・画素値の範囲（確率 0〜1 か bool か）** の3点を `print` で確認する。`draw_*` 系の例外の8割はここで原因が割れます。座標ズレを疑ったら `target_sizes` に渡した値が `(H,W)` か `(W,H)` かを確認します。

## 🚀 発展トピック・参考

- **SAM 2 / MobileSAM / SlimSAM**: SAM2 は動画にも対応したメモリ機構を持ち、MobileSAM/SlimSAM は CPU 向けの軽量蒸留版。本講座は依存衝突を避けて HF SAM を使いますが、Ultralytics 版（`SAM('mobile_sam.pt')` / `SAM('sam2.1_t.pt')`）は別環境なら1行で試せます（`ultralytics` は `opencv-python` full を引き込み、本講座既定の headless と排他なので注意）。
- **Mask2Former の出し分け**: 同じ重みで `post_process_instance_segmentation` / `post_process_semantic_segmentation` / `post_process_panoptic_segmentation` を切り替えるだけで3タスクを出せます。`02` を改造して instance/semantic 版の出力フォーマットを見比べると統一アーキテクチャの旨味が分かります。
- **mask AP の深掘り**: `COCOeval` の `AP_S/M/L`（面積別）・`AR@{1,10,100}`（maxDets 別）や、`areaRng`/`maxDets` を変えたときの数値変化。COCO は **101 点補間**（recall を 0:0.01:1 でサンプル）で、本章の全点補間とは別流儀です。
- **PQ の分解読み**: `PQ = SQ × RQ` を things/stuff 別（`PQ^Th` / `PQ^St`）に分けて報告すると、前景と背景のどちらが弱いかが見えます。panopticapi が公式実装、`torchmetrics.detection.PanopticQuality` が手軽な再現実装です。
- **Grounded-SAM への接続（次章）**: 第23回では Grounding DINO の検出 box を SAM の `input_boxes` に渡し、**テキスト→検出→分割**の2段パイプラインを組みます。本章の「箱プロンプト SAM」と「IoU 評価」がそのまま土台になります。
- **公式ドキュメント**: torchvision models（<https://docs.pytorch.org/vision/stable/models.html>）／ HF Mask2Former・SAM（<https://huggingface.co/docs/transformers>）／ torchmetrics PanopticQuality（<https://lightning.ai/docs/torchmetrics/stable/>）／ pycocotools（<https://github.com/ppwwyyxx/cocoapi>）。

## 💡 実践ユースケース集

この章の SAM（クラス非依存のプロンプト型セグメンタ）は、そのまま現場の小ツールに化けます。`mini_project.py` が「切り出した結果を **指標で採点する**」のに対し、ここでは「切り出した結果を **そのまま納品物として使う**」応用を3つ挙げます。1つ目は実際に動く `use_case.py` として同梱しました。

### ① クリック切り抜き → 透明背景 PNG（`use_case.py`：動く出発点）

**何に使うか**: 商品写真の背景抜き、資料・スライドに貼る切り抜き素材、アイコン作成。「この物体だけ欲しい」を、SAM の**点プロンプト1点**で解決します。

**作り方の要点**: 画像の一点を“クリック”＝`input_points=[[[x, y]]]` / `input_labels=[[1]]`（1=前景）として SAM に渡し、`post_process_masks` で原寸に戻したベストマスク（`iou_scores` 最大の1枚）を取得。そのマスクを**アルファチャンネル**に流し込み（マスク内 255／外 0）、物体の bbox＋余白でトリミングして `RGBA` の PNG として保存します。透過の確認用に、結果をチェッカー柄へ合成したプレビューも出します。

**注意**: 透過 PNG は `Image.fromarray(rgba, "RGBA")` で保存する（`draw_*` 系は uint8 を要求する点も同じ）。マスクの縁に背景色が薄く残る（“緑のにじみ”）ときはアルファを1〜2px **収縮**するか軽くぼかす。点は必ず物体の**内側**に置く（外すと空マスク→スキップ）。

```bash
uv run python lectures/22_instance_panoptic_sam/use_case.py
# → lectures/22_instance_panoptic_sam/outputs/use_case_cutout_01.png ...（透明背景の切り抜き・物体ごとに1枚）
#   lectures/22_instance_panoptic_sam/outputs/use_case_preview.png   （入力＋クリック点／マスク／透明合成）
#   lectures/22_instance_panoptic_sam/outputs/use_case_cutouts.json  （座標・面積・bbox 等のメタ）
```

**`data/22_instance_panoptic_sam/` への配置**: `.png` / `.jpg` を1枚置くと、その先頭画像を入力に使います（無ければ合成シーンで必ず完走）。狙った物体を切りたいときは `use_case.py` の `CLICK_POINTS` に `(x, y)` を列挙すれば、その座標を“クリック”として複数枚まとめて切り出せます。なお SAM はクラス非依存なので、合成図形でも実写でも指した領域をきれいに切れます（実写で本領を発揮させるには SAM 重みの初回 DL が必要）。

**拡張アイデア**: 背景点（`input_labels=0`）を足して切り抜きを精緻化／箱プロンプト（`input_boxes`）でドラッグ枠切り抜き／アルファのフェザー（フチぼかし）／`data/` 内を一括バッチ処理して素材を量産／切り抜きを別背景に合成して合成写真を作る／`argparse` で座標を受け取る簡易 CLI 化。

### ② インスタンス自動マスキング（個人情報・不要物の自動ぼかし）

**何に使うか**: 公開用画像から「人だけ」「ナンバープレートだけ」を検出して塗りつぶす／ぼかす自動処理。

**作り方の要点**: Mask R-CNN（`maskrcnn_resnet50_fpn_v2`、`01_*` 参照）で人や車のインスタンスマスクを取り、`masks (N,1,H,W)` を `squeeze(1) > 0.5` で bool 化。その領域だけ `cv2.GaussianBlur` やモザイクで置換します。SAM と違い**クラス名で対象を選べる**のが利点。

**注意**: 合成図形では COCO クラスに当たらず検出0件になりがち（想定内）。実写を `data/` に置くこと。閾値（`score`）を上げ過ぎると取りこぼし、下げ過ぎると誤マスクが増えます。

### ③ パノプティック前景/背景分離（背景差し替え・空の置換）

**何に使うか**: パノプティックの things（前景）と stuff（背景＝空・道路）を分け、背景だけ別画像へ差し替える（空の置換など）。

**作り方の要点**: Mask2Former（`02_*` 参照）の `post_process_panoptic_segmentation` から `segmentation` と `segments_info` を取り、`isthing` で things/stuff を仕分け。stuff 領域をマスクにして背景合成へ回します。各画素が**ちょうど1つ**の `(category, instance)` を持つパノプティックの性質が、漏れ・重なりのない分離に効きます。

**注意**: things と stuff のカテゴリ系統（COCO panoptic）の id 対応を取り違えない。CPU では `mask2former-swin-tiny` を既定にし、`swin-large` 級は避ける。

---

> 本教材で参照・検証したライブラリとバージョン（2026-06 時点・CPU で動作確認）:
> Python 3.12 ／ numpy 2.4.6 ／ opencv-python-headless 4.13 ／ torch 2.12.0+cpu ／ torchvision 0.27.0+cpu ／ transformers 5.11.0 ／ pycocotools 2.0.11 ／ torchmetrics 1.9.0 ／ matplotlib 3.10.9。
> 使用モデル: `maskrcnn_resnet50_fpn_v2`（torchvision Weights API）／ `facebook/mask2former-swin-tiny-coco-panoptic` ／ `facebook/sam-vit-base`（軽量化は `Zigeng/SlimSAM-uniform-77`）。本講座セグメ/評価トラックの想定スタック（2026-06 時点）は torch 2.12+cpu / torchvision 0.27+cpu / transformers 5.11 / pycocotools 2.0.11 / torchmetrics 1.9 です。