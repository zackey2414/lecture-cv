# 第18回 物体検出 入門 — torchvision weights API・HF DETR（＋YOLO/RT-DETRの位置づけ）

> トラック: **検出** ／ レベル: **入門** ／ 必要な依存グループ: `dl` `hf` `metrics`
> （`uv sync --group dl --group hf --group metrics`。`detect`（Ultralytics）は概念紹介のみで実行には不要）

## 🎯 この章のゴール

これまでの章では「画像1枚に1つのラベルを付ける」分類や、「画像全体を1本のベクトルにする」埋め込みを扱ってきました。本章からは一段難しい **物体検出（object detection）** に入ります。物体検出とは、画像の中の「何が」「どこに」あるかを、**クラス名＋確信度＋矩形（バウンディングボックス）** の組として複数同時に答えるタスクです。出力が「1個」ではなく「可変個の箱の集合」になる——この点こそが分類との決定的な違いであり、後処理（閾値・NMS）や評価（mAP）が分類より複雑になる根本理由でもあります。

この章を終えたとき、あなたは検出の主要3系統 — **torchvision の R-CNN/SSD/RetinaNet**、**HuggingFace の DETR/RT-DETR**、そして概念としての **Ultralytics YOLO** — を、それぞれの「正準的な書き方」で読み書きできるようになります。とくに torchvision の **weights API（4点セット）**、検出モデル特有の **[0,1] float CHW テンソルのリスト入力**、**score 閾値 → NMS → 可視化** の定番パイプライン、DETR の **`post_process_object_detection` と `target_sizes=(H,W)` の罠**、そして COCO ラベルの **index0 = `__background__`** という落とし穴を、すべて自分の手で再現して確認します。

到達点を一言でいえば、**「同じ画像を別フレームワークの検出器に通しても、`{boxes(xyxy), scores, labels, names}` という同じ形に正規化して、閾値・NMS・可視化・評価を1経路で回せる」** ことです。最後は `torchmetrics` で mAP@0.5 / mAP@[.5:.95] を算出するところまで通します（mAP を numpy で一から実装するのは第19回）。本章のコードは **CPU だけ・合成画像だけ** で完走します（ネットへ出るのはモデル重みのDLのみ）。

---

## 1. 物体検出とは — 分類との違いと「3つの出力」

物体検出器の出力は、画像1枚につき次の3つが**同じ長さ N（検出数）で並んだもの**です。N は画像ごとに変わります（何も無ければ 0、混雑シーンなら数十）。

<figure class="lec-fig"><svg viewBox="0 0 640 300" role="img" aria-label="検出器の出力はboxes(N,4)・scores(N,)・labels(N,)という同じ長さNの3配列で、同じ列が1つの検出に対応する" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><text x="320" y="30" text-anchor="middle" font-size="14" font-weight="700" fill="#18181b">検出 N 個 ＝ 同じ長さ・同じ並びの 3 配列</text><rect x="20" y="92" width="84" height="92" fill="#eff6ff" stroke="#2563eb" stroke-width="1.8"/><rect x="40" y="112" width="22" height="50" fill="none" stroke="#dc2626" stroke-width="2"/><rect x="66" y="140" width="30" height="24" fill="none" stroke="#1d4ed8" stroke-width="2"/><text x="62" y="202" text-anchor="middle" font-size="12.5" fill="#52525b">入力画像</text><line x1="108" y1="138" x2="146" y2="138" stroke="#71717a" stroke-width="2"/><polygon points="152,138 144,134 144,142" fill="#71717a"/><rect x="270" y="58" width="318" height="52" fill="#fff7ed" stroke="#d4d4d8" stroke-width="1.4"/><rect x="270" y="120" width="318" height="44" fill="#fafafa" stroke="#d4d4d8" stroke-width="1.4"/><rect x="270" y="174" width="318" height="44" fill="#fafafa" stroke="#d4d4d8" stroke-width="1.4"/><line x1="376" y1="58" x2="376" y2="218" stroke="#e4e4e7"/><line x1="482" y1="58" x2="482" y2="218" stroke="#e4e4e7"/><rect x="376" y="58" width="106" height="160" fill="#ffedd5" opacity="0.55"/><rect x="376" y="58" width="106" height="160" fill="none" stroke="#ea580c" stroke-width="2.5"/><text x="154" y="89" font-size="14" font-weight="700" fill="#c2410c">boxes (N,4)</text><text x="154" y="147" font-size="14" font-weight="700" fill="#2563eb">scores (N,)</text><text x="154" y="201" font-size="14" font-weight="700" fill="#16a34a">labels (N,)</text><text x="323" y="89" text-anchor="middle" font-size="13" fill="#3f3f46">xyxy</text><text x="429" y="89" text-anchor="middle" font-size="13" font-weight="700" fill="#c2410c">xyxy</text><text x="535" y="89" text-anchor="middle" font-size="13" fill="#3f3f46">xyxy</text><text x="323" y="147" text-anchor="middle" font-size="13" fill="#3f3f46">0.97</text><text x="429" y="147" text-anchor="middle" font-size="13" font-weight="700" fill="#c2410c">0.88</text><text x="535" y="147" text-anchor="middle" font-size="13" fill="#3f3f46">0.41</text><text x="323" y="201" text-anchor="middle" font-size="13" fill="#3f3f46">1</text><text x="429" y="201" text-anchor="middle" font-size="13" font-weight="700" fill="#c2410c">3</text><text x="535" y="201" text-anchor="middle" font-size="13" fill="#3f3f46">17</text><text x="429" y="244" text-anchor="middle" font-size="12.5" font-weight="700" fill="#ea580c">列 1 ＝ 1 つの検出</text><text x="429" y="263" text-anchor="middle" font-size="11.5" fill="#71717a">box・score・label が対応</text></svg><figcaption>検出器は1枚の画像から<b>可変個 N</b> の検出を返し、その実体は <b>boxes <code>(N,4)</code></b>・<b>scores <code>(N,)</code></b>・<b>labels <code>(N,)</code></b> という<b>同じ長さ・同じ並び</b>の3配列です。<b>同じ列</b>（図の列1）が1つの検出に対応し、箱 <code>[x1,y1,x2,y2]</code>（xyxy）・確信度・クラスIDが束になります。N は画像ごとに変わり、何も無ければ 0 です。</figcaption></figure>

| 出力 | 形 | 意味 |
| --- | --- | --- |
| `boxes` | `(N, 4)` | 矩形。本章は一貫して **xyxy**＝`[x1, y1, x2, y2]`（左上・右下の絶対座標, float） |
| `scores` | `(N,)` | 各検出の確信度（0〜1）。降順に並ぶとは限らない |
| `labels` | `(N,)` | クラスID（整数）。COCO なら 0〜90 |

ここで最初の関門が **クラスID → クラス名** の対応です。torchvision の COCO 検出器は `weights.meta["categories"]` という**長さ91のリスト**を持ち、その **index 0 は `__background__`**（背景）です。つまり `labels` の整数 1 が `person`、3 が `car` を指します。生の整数 `0` を「最初のクラス」と勘違いして名前を引くと、すべて1つズレた名前が付いてしまう——これは検出の超頻出バグなので、必ず `categories[int(label)]` で引く癖をつけます。

<figure class="lec-fig"><svg viewBox="0 0 560 280" role="img" aria-label="COCOのcategoriesは長さ91でindex0が__background__。labelの整数1はcategories[1]=personを指す" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><text x="280" y="30" text-anchor="middle" font-size="14" font-weight="700" fill="#18181b">COCO の categories は長さ 91・index 0 は背景</text><text x="92" y="58" text-anchor="middle" font-size="12" fill="#71717a">0</text><text x="199" y="58" text-anchor="middle" font-size="12" fill="#71717a">1</text><text x="287" y="58" text-anchor="middle" font-size="12" fill="#71717a">2</text><text x="375" y="58" text-anchor="middle" font-size="12" fill="#71717a">3</text><text x="486" y="58" text-anchor="middle" font-size="12" fill="#71717a">90</text><rect x="30" y="66" width="124" height="50" fill="#e4e4e7" stroke="#71717a" stroke-width="1.8"/><rect x="156" y="66" width="86" height="50" fill="#fff7ed" stroke="#ea580c" stroke-width="2.5"/><rect x="244" y="66" width="86" height="50" fill="#ffffff" stroke="#d4d4d8" stroke-width="1.5"/><rect x="332" y="66" width="86" height="50" fill="#ffffff" stroke="#d4d4d8" stroke-width="1.5"/><rect x="440" y="66" width="92" height="50" fill="#ffffff" stroke="#d4d4d8" stroke-width="1.5"/><text x="92" y="96" text-anchor="middle" font-size="11" fill="#52525b">__background__</text><text x="199" y="96" text-anchor="middle" font-size="12.5" font-weight="700" fill="#c2410c">person</text><text x="287" y="96" text-anchor="middle" font-size="12" fill="#3f3f46">bicycle</text><text x="375" y="96" text-anchor="middle" font-size="12.5" font-weight="700" fill="#1d4ed8">car</text><text x="486" y="96" text-anchor="middle" font-size="11" fill="#3f3f46">toothbrush</text><text x="429" y="98" text-anchor="middle" font-size="18" fill="#71717a">…</text><line x1="152" y1="168" x2="196" y2="118" stroke="#ea580c" stroke-width="1.4" stroke-dasharray="4 3" opacity="0.6"/><rect x="66" y="170" width="158" height="44" rx="6" fill="#fff7ed" stroke="#ea580c" stroke-width="2"/><text x="145" y="198" text-anchor="middle" font-size="14" font-weight="700" fill="#c2410c">予測 label = 1</text><line x1="228" y1="192" x2="262" y2="192" stroke="#71717a" stroke-width="2"/><polygon points="270,192 260,187 260,197" fill="#71717a"/><rect x="276" y="170" width="224" height="44" rx="6" fill="#fafafa" stroke="#16a34a" stroke-width="2"/><text x="388" y="198" text-anchor="middle" font-size="14" font-weight="700" fill="#15803d">names[1] = 'person'</text><text x="280" y="252" text-anchor="middle" font-size="12.5" fill="#dc2626">× 0 を最初のクラスと誤解すると全ラベルが 1 つずれる</text></svg><figcaption>torchvision の COCO 検出器は <code>weights.meta["categories"]</code> という<b>長さ91のリスト</b>を持ち、その <b>index 0 は <code>__background__</code></b>（背景）です。だから <b>labels の整数 1 は <code>categories[1]</code> ＝ <code>person</code></b>、3 は <code>car</code> を指します。生の整数をそのままクラス名と解釈すると<b>全ラベルが1つずれる</b>ので、必ず <code>categories[int(label)]</code> で引きます（DETR は <code>id2label</code> 辞書で <code>id2label[0]='N/A'</code>）。</figcaption></figure>

```python
names = weights.meta["categories"]   # ['__background__', 'person', 'bicycle', 'car', ...] 長さ91
name = names[int(label)]             # label=1 -> 'person'。index0 は背景なので生整数を名前と誤解しない
```

なお HuggingFace の DETR は同じことを `model.config.id2label`（辞書）で持ちます。フレームワークによって「リスト」か「辞書」か、背景クラスの扱い（DETR は `id2label[0]` が `'N/A'`）が微妙に違うので、**「ラベル整数をどう名前に変換するか」はモデルごとに確認する**のが鉄則です。本章のスクリプトは各モデルの正しい変換を使い分けています。

## 2. torchvision の weights API — 「4点セット」は全モデル共通

torchvision の検出モデルで最初に体に入れてほしいのは、**どのモデルでも書き方が同じ**ということです。モデルごとに変わるのは weights enum の名前だけで、次の「4点セット」は常に同じ形をしています。これを理解すれば、`fasterrcnn` を `retinanet` や `ssdlite320` に差し替えるのは enum を変えるだけの作業になります。

```python
from torchvision.models.detection import (
    ssdlite320_mobilenet_v3_large, SSDLite320_MobileNet_V3_Large_Weights,
)
weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT   # ① 学習済み重み（COCO）
model   = ssdlite320_mobilenet_v3_large(weights=weights)  # ② ネットワーク本体
model.eval()                                              #    ★推論モードへ（必須）
preproc = weights.transforms()                            # ③ そのモデル専用の前処理
names   = weights.meta["categories"]                      # ④ クラス名（index0=__background__）
```

`01_torchvision_detection.py` は、この4点セットを `ssdlite320`（1-stage・最軽量）・`retinanet_resnet50_fpn`（1-stage・focal loss）・`fasterrcnn_resnet50_fpn`（2-stage・高精度）の3モデルで**同じループ**に流し込みます。出力の `print` を見ると、3モデルすべてが `{boxes, labels, scores}` という同一の辞書を返し、後処理（後述の閾値・NMS）も完全に共通化できることが分かります。「APIが同じなら、モデル選定は精度・速度・サイズのトレードオフだけの問題に還元できる」——これが weights API の最大の利点です。

`.eval()` を呼ぶのは飾りではありません。検出モデルは内部に BatchNorm を多数持ち、学習モードのままだと統計が推論用に固定されず、結果が不安定になります。さらに推論本体は必ず `torch.inference_mode()`（または `no_grad()`）の中で実行し、勾配計算とメモリ確保を止めます。`.eval()` と `inference_mode()` はワンセットだと覚えてください。

## 3. 検出モデル特有の入力ルール — `[0,1]` float CHW テンソルの「リスト」

分類モデルに慣れた人が最初に戸惑うのが、検出モデルの**入力の作法**です。torchvision の検出モデルは「**`[0,1]` に収まる float の CHW テンソルの *リスト*（可変サイズ画像をバッチにできる）**」を受け取り、**ImageNet 正規化（mean/std を引く処理）はモデルの内部で行います**。つまり `weights.transforms()`（`ObjectDetection` プリセット）がやるのは「PIL → `[0,1]` float テンソル」への変換だけで、分類のように自分で `Normalize(mean, std)` を重ねてはいけません。なぜなら、二重正規化は典型的な精度劣化バグだからです。

```python
x = preproc(image)              # PIL -> (3, H, W) float, 値域 [0,1]（正規化はまだしない）
with torch.inference_mode():
    out = model([x])[0]         # ★入力は「テンソルのリスト」。[x] のように包む。[0] で1枚目の結果
print(out.keys())               # dict_keys(['boxes', 'labels', 'scores'])
```

`model([x])` の `[...]` を忘れて `model(x)` と書くとエラーになります。検出モデルが「リスト」を取るのは、**サイズの異なる複数画像を1回の呼び出しで処理できる**ようにするためです（内部で各画像を `min_size`/`max_size` に合わせてリサイズします）。`01` のスクリプトでは resnet50 系に `min_size=320, max_size=512` を渡して入力解像度を下げ、CPU でも1枚あたり 0.1〜0.3 秒で回るようにしています。解像度を下げると小さい物体の検出精度は落ちますが、CPU での学習・実験には十分です。

ここまでで「読み込み（4点セット）→ 前処理（`[0,1]` リスト）→ 推論（`eval`＋`inference_mode`）→ 出力（3つの並び）」という検出の骨格が一通り出そろいました。次はその**生出力**を実用的な検出に整える後処理です。

<figure class="lec-fig"><svg viewBox="0 0 660 260" role="img" aria-label="検出の骨格は読み込み・前処理・推論・後処理・可視化の一直線で、出力をboxes scores labelsに正規化すれば共通経路に載る" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><text x="330" y="40" text-anchor="middle" font-size="15" font-weight="700" fill="#18181b">検出の処理パイプライン全体（システムの流れ）</text><rect x="22" y="78" width="104" height="92" rx="6" fill="#ffedd5" stroke="#ea580c" stroke-width="2"/><text x="74" y="118" text-anchor="middle" font-size="15" font-weight="700" fill="#c2410c">① 読み込み</text><text x="74" y="146" text-anchor="middle" font-size="11.5" fill="#52525b">4点セット</text><rect x="150" y="78" width="104" height="92" rx="6" fill="#ffedd5" stroke="#ea580c" stroke-width="2"/><text x="202" y="118" text-anchor="middle" font-size="15" font-weight="700" fill="#c2410c">② 前処理</text><text x="202" y="146" text-anchor="middle" font-size="11.5" fill="#52525b">[0,1] リスト</text><rect x="278" y="78" width="104" height="92" rx="6" fill="#ffedd5" stroke="#ea580c" stroke-width="2"/><text x="330" y="118" text-anchor="middle" font-size="15" font-weight="700" fill="#c2410c">③ 推論</text><text x="330" y="146" text-anchor="middle" font-size="11.5" fill="#52525b">eval+推論モード</text><rect x="406" y="78" width="104" height="92" rx="6" fill="#eff6ff" stroke="#2563eb" stroke-width="2"/><text x="458" y="118" text-anchor="middle" font-size="15" font-weight="700" fill="#1d4ed8">④ 後処理</text><text x="458" y="146" text-anchor="middle" font-size="11.5" fill="#52525b">閾値 → NMS</text><rect x="534" y="78" width="104" height="92" rx="6" fill="#eff6ff" stroke="#2563eb" stroke-width="2"/><text x="586" y="118" text-anchor="middle" font-size="15" font-weight="700" fill="#1d4ed8">⑤ 可視化</text><text x="586" y="146" text-anchor="middle" font-size="11.5" fill="#52525b">draw_boxes</text><line x1="130" y1="124" x2="141" y2="124" stroke="#71717a" stroke-width="2"/><polygon points="150,124 140,119 140,129" fill="#71717a"/><line x1="258" y1="124" x2="269" y2="124" stroke="#71717a" stroke-width="2"/><polygon points="278,124 268,119 268,129" fill="#71717a"/><line x1="386" y1="124" x2="397" y2="124" stroke="#71717a" stroke-width="2"/><polygon points="406,124 396,119 396,129" fill="#71717a"/><line x1="514" y1="124" x2="525" y2="124" stroke="#71717a" stroke-width="2"/><polygon points="534,124 524,119 524,129" fill="#71717a"/><text x="202" y="206" text-anchor="middle" font-size="11.5" fill="#71717a">§2・§3（本節まで）</text><text x="458" y="206" text-anchor="middle" font-size="11.5" fill="#71717a">§4 で詳述</text><text x="586" y="206" text-anchor="middle" font-size="11.5" fill="#71717a">§5 で詳述</text></svg><figcaption>本章で扱う検出の骨格は <b>① 読み込み（4点セット）→ ② 前処理（<code>[0,1]</code> float CHW リスト）→ ③ 推論（<code>eval()</code>＋<code>inference_mode</code>）→ ④ 後処理（score 閾値 → NMS）→ ⑤ 可視化（<code>draw_bounding_boxes</code>）</b> の一直線です。①〜③ は本節まで（§2・§3）、④ は §4、⑤ は §5 で詳しく扱います。torchvision でも DETR でも、出力を <code>{boxes, scores, labels}</code> に正規化すればこの 1 経路に載せられます。</figcaption></figure>

## 4. score 閾値と NMS — 生検出を「使える検出」に整える

検出器の生出力は、低スコアの箱や、同じ物体を指す重複した箱で溢れています（SSD 系は1枚で常時300箱を返します）。これをそのまま使うことはなく、必ず2段階で間引きます。**(1) score 閾値**で確信度の低い箱を捨て、**(2) NMS（Non-Maximum Suppression）** で「同じ物体に重なった箱」を最高スコアの1つに統合します。

<figure class="lec-fig"><svg viewBox="0 0 640 260" role="img" aria-label="生検出の多数で重複した箱を、score閾値で低確信度を除き、NMSでIoUの高い重複を抑制して1物体1箱にする" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><text x="106" y="52" text-anchor="middle" font-size="13" font-weight="700" fill="#3f3f46">① 生検出（重複・多数）</text><text x="320" y="52" text-anchor="middle" font-size="13" font-weight="700" fill="#3f3f46">② score 閾値後</text><text x="534" y="52" text-anchor="middle" font-size="13" font-weight="700" fill="#15803d">③ NMS 後</text><rect x="16" y="64" width="180" height="150" fill="#fafafa" stroke="#d4d4d8" stroke-width="1.5"/><rect x="230" y="64" width="180" height="150" fill="#fafafa" stroke="#d4d4d8" stroke-width="1.5"/><rect x="444" y="64" width="180" height="150" fill="#fafafa" stroke="#d4d4d8" stroke-width="1.5"/><rect x="84" y="104" width="44" height="88" fill="#e4e4e7" stroke="#d4d4d8" stroke-width="1.2"/><rect x="76" y="96" width="58" height="104" fill="none" stroke="#dc2626" stroke-width="2.2"/><rect x="88" y="106" width="46" height="84" fill="none" stroke="#f97316" stroke-width="1.6"/><rect x="70" y="90" width="64" height="112" fill="none" stroke="#ea580c" stroke-width="1.6"/><rect x="146" y="150" width="38" height="32" fill="none" stroke="#71717a" stroke-width="1.4" stroke-dasharray="4 3"/><text x="165" y="146" text-anchor="middle" font-size="11" fill="#71717a">0.18</text><rect x="298" y="104" width="44" height="88" fill="#e4e4e7" stroke="#d4d4d8" stroke-width="1.2"/><rect x="290" y="96" width="58" height="104" fill="none" stroke="#dc2626" stroke-width="2.2"/><rect x="302" y="106" width="46" height="84" fill="none" stroke="#f97316" stroke-width="1.6"/><rect x="512" y="104" width="44" height="88" fill="#e4e4e7" stroke="#d4d4d8" stroke-width="1.2"/><rect x="504" y="96" width="58" height="104" fill="none" stroke="#16a34a" stroke-width="2.6"/><line x1="200" y1="139" x2="224" y2="139" stroke="#71717a" stroke-width="2"/><polygon points="230,139 222,135 222,143" fill="#71717a"/><text x="213" y="128" text-anchor="middle" font-size="10.5" fill="#2563eb">score≥0.5</text><line x1="414" y1="139" x2="438" y2="139" stroke="#71717a" stroke-width="2"/><polygon points="444,139 436,135 436,143" fill="#71717a"/><text x="429" y="128" text-anchor="middle" font-size="11" fill="#2563eb">NMS</text></svg><figcaption>検出器の<b>生出力</b>は、低スコアの箱や同じ物体に重なった箱で溢れています（SSD系は常時300箱）。<b>(1) score 閾値</b>で確信度の低い箱（図の <code>0.18</code>）を捨て、<b>(2) NMS</b> でスコア最大の箱を残し<b>IoU が大きい重複</b>を抑制して、<b>1 物体 ＝ 1 箱</b>に整えます。実務ではクラスを跨いで潰さない <code>batched_nms</code> を使います。</figcaption></figure>

NMS の手順はシンプルです。スコア降順に箱を見て、先頭（最高スコア）を採用し、それと **IoU（重なり率）が閾値を超える**残りの箱を捨て、残った中で同じことを繰り返します。torchvision は `nms`（1クラス用）と `batched_nms`（クラスごとに独立に NMS）を用意しています。**実務では `batched_nms` を使う**のが安全です。全クラスまとめて `nms` を掛けると、たまたま重なった「人」と「車」の箱が誤って一方に潰されてしまうからです。

<figure class="lec-fig"><svg viewBox="0 0 560 250" role="img" aria-label="IoUは2つの箱の交差面積を和集合面積で割った値で、0から1の重なり率を表す" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><text x="92" y="58" text-anchor="middle" font-size="13" font-weight="700" fill="#dc2626">A 予測</text><text x="268" y="232" text-anchor="middle" font-size="13" font-weight="700" fill="#1d4ed8">B 正解</text><rect x="150" y="104" width="60" height="82" fill="#ffedd5" opacity="0.85" stroke="#ea580c" stroke-width="2"/><rect x="64" y="66" width="146" height="120" fill="none" stroke="#dc2626" stroke-width="2.6"/><rect x="150" y="104" width="146" height="110" fill="none" stroke="#1d4ed8" stroke-width="2.6"/><text x="180" y="150" text-anchor="middle" font-size="12" font-weight="700" fill="#c2410c">交差</text><text x="340" y="128" text-anchor="end" font-size="18" font-weight="700" fill="#18181b">IoU =</text><text x="458" y="108" text-anchor="middle" font-size="13" fill="#c2410c">交差（重なり）</text><line x1="356" y1="122" x2="548" y2="122" stroke="#3f3f46" stroke-width="2"/><text x="458" y="146" text-anchor="middle" font-size="13" fill="#3f3f46">和集合（A∪B）</text><text x="458" y="186" text-anchor="middle" font-size="12" fill="#52525b">0 〜 1（1 で完全一致）</text></svg><figcaption><b>IoU（Intersection over Union・重なり率）</b>は、2つの箱の<b>交差（重なり）面積</b>を<b>和集合面積</b>で割った値で、<b>0（接しない）〜1（完全一致）</b>を取ります。<b>NMS</b> は「IoU が閾値超の重複箱」を抑制するのに、<b>mAP</b> は「予測と正解が当たりか」を判定するのに、同じ IoU を使います（演習 <code>ex1</code> で自作）。</figcaption></figure>

```python
from torchvision.ops import batched_nms

keep_mask = scores >= 0.5                       # (1) score 閾値
boxes, scores, labels = boxes[keep_mask], scores[keep_mask], labels[keep_mask]
keep = batched_nms(boxes, scores, labels, 0.5)  # (2) クラス別 NMS（IoU>0.5 を抑制）
boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
```

本章ではこの2段処理を `detection_helpers.apply_threshold_nms()` に切り出し、`01`〜`03` の全スクリプトで共有しています。閾値は合成画像だと検出が弱いので `0.3` に下げていますが、実写では `0.5` 前後が定番です。閾値を上げれば誤検出（FP）が減る代わりに見逃し（FN）が増える——この**トレードオフを数字で管理する**のが次章以降の mAP の話につながります。なお IoU と NMS の中身は本章の演習 `ex1`（IoU）・`ex4`（NMS）で自分の手で実装します。

## 5. 可視化 — `draw_bounding_boxes` は uint8 画像を要求する

検出結果は数字で見るだけでなく、画像に重ねて目で確認するのが鉄則です。torchvision の `draw_bounding_boxes` が標準ですが、ここに2つの落とし穴があります。**(1) 画像テンソルは `uint8`（0〜255）でなければならない**（`[0,1]` の float を渡すと例外か真っ黒になる）。**(2) 入力は CHW テンソル**で、PIL とは軸の並びが違う。本章の `to_uint8_chw()` がこの変換を一手に引き受けます。

```python
from torchvision.utils import draw_bounding_boxes

canvas = to_uint8_chw(pil_image)                 # PIL(RGB,HWC) -> uint8 CHW テンソル
labels_text = [f"{name} {score:.2f}" for ...]    # 箱に添える文字
drawn = draw_bounding_boxes(canvas, boxes, labels=labels_text, colors="red", width=2)
result = Image.fromarray(drawn.permute(1, 2, 0).numpy())  # CHW -> HWC で PIL に戻す
```

`01` を実行すると `outputs/18_object_detection_intro/01_torchvision_compare.png` に3モデルの比較パネルが保存されます。合成画像なので検出は乏しく、しかも**ラベルが的外れ**（例: 人物図形が `stop sign`）になります。これは故障ではなく、「COCO で学習した検出器は写実的な写真向けに最適化されており、抽象的な合成図形では妥当なクラスを当てられない」という大事な学びです（§10で詳述）。それでもパイプライン（前処理→推論→閾値→NMS→可視化）は完全に動いており、箱・スコア・NMS の挙動を観察するには十分です。

なお matplotlib の図中テキストをあえて ASCII にしているのは、日本語を入れると headless 環境の DejaVu フォントで「豆腐（□）」になってしまうためです。また、図の色が反転して見える場合は、合成画像を RGB のまま扱えているか（`cv2.imread/imwrite` 経由で BGR が混ざっていないか）を確認してください。

## 6. HuggingFace DETR — set prediction と `post_process`、`target_sizes=(H,W)` の罠

`02_detr_huggingface.py` では、Transformer ベースの検出器 **DETR（DEtection TRansformer, `facebook/detr-resnet-50`）** を扱います。DETR は torchvision の R-CNN/SSD/RetinaNet とは設計思想が根本的に異なります。すなわち、アンカーや NMS を使わず、**N 個（既定100）の「物体クエリ」がそれぞれ「1物体 or 背景(no-object)」を予測**し、学習時の **二部マッチング（Hungarian matching）** で「1物体には1クエリ」を強制します。その結果、**原理上は後段の NMS が不要**になります。これが DETR の最大の特徴です。

transformers v5 の正準フローは次の通りです。生出力 `pred_boxes` は **cxcywh の正規化座標（0〜1）** なので、必ず `post_process_object_detection` を通して **xyxy の絶対座標** に変換してから使います。

```python
from transformers import AutoImageProcessor, AutoModelForObjectDetection

proc  = AutoImageProcessor.from_pretrained("facebook/detr-resnet-50")
model = AutoModelForObjectDetection.from_pretrained("facebook/detr-resnet-50").eval()
inputs  = proc(images=pil, return_tensors="pt")
with torch.inference_mode():
    outputs = model(**inputs)                               # 生出力（cxcywh 正規化）
result = proc.post_process_object_detection(
    outputs, threshold=0.5, target_sizes=[(H, W)],          # ★(height, width) 順！
)[0]                                                        # {'scores','labels','boxes'(xyxy絶対)}
names = [model.config.id2label[int(i)] for i in result["labels"]]
```

ここで本章一番の落とし穴が **`target_sizes` の順序** です。これは `(height, width)` 順で渡しますが、**PIL の `image.size` は `(width, height)`** を返します。つまり `target_sizes=[image.size[::-1]]` と**反転して渡す**のが正解で、`image.size` をそのまま渡すと box が縦横に歪みます。`02` はわざと正しい `(H,W)` と誤った `(W,H)` の両方で後処理し、`02_detr_target_sizes.png` に並べます。右パネル（バグ）の箱が、正しい中パネルに対して潰れて別位置にズレているのが一目で分かります——この絵を一度見ておけば、実務で box がおかしいとき真っ先に `target_sizes` を疑えるようになります。

<figure class="lec-fig"><svg viewBox="0 0 600 232" role="img" aria-label="DETRのpost_processでtarget_sizesを(H,W)順に渡せば正しい箱、(W,H)順だと箱が縦横に歪む" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><text x="300" y="28" text-anchor="middle" font-size="14" font-weight="700" fill="#18181b">target_sizes の順序で DETR の box が変わる</text><text x="161" y="56" text-anchor="middle" font-size="13" font-weight="700" fill="#15803d">✓ (H, W) 順</text><text x="439" y="56" text-anchor="middle" font-size="13" font-weight="700" fill="#dc2626">✗ (W, H) 順</text><rect x="36" y="66" width="250" height="132" fill="#fafafa" stroke="#16a34a" stroke-width="2.5"/><rect x="314" y="66" width="250" height="132" fill="#fafafa" stroke="#dc2626" stroke-width="2.5"/><rect x="96" y="118" width="120" height="52" fill="#e4e4e7" stroke="#d4d4d8" stroke-width="1.2"/><rect x="90" y="112" width="132" height="64" fill="none" stroke="#16a34a" stroke-width="2.6"/><rect x="374" y="118" width="120" height="52" fill="#e4e4e7" stroke="#d4d4d8" stroke-width="1.2"/><rect x="402" y="90" width="76" height="106" fill="none" stroke="#dc2626" stroke-width="2.6"/><text x="161" y="216" text-anchor="middle" font-size="11.5" fill="#15803d">target_sizes=[(H, W)]</text><text x="439" y="216" text-anchor="middle" font-size="11.5" fill="#dc2626">image.size をそのまま渡す</text></svg><figcaption>DETR の生 <code>pred_boxes</code> は <b>cxcywh の正規化座標(0〜1)</b> です。<code>post_process_object_detection</code> に <b><code>target_sizes</code></b> を渡して初めて <b>xyxy の絶対座標</b>になります。これは <b>(H, W) 順</b>で渡すのが正解（<code>image.size[::-1]</code>）。PIL の <code>image.size</code> は <b>(W, H)</b> なので、そのまま渡すと x と y のスケールが入れ替わり、<b>箱が縦横に歪んで別位置にずれます</b>。</figcaption></figure>

DETR の `id2label` は背景クラスを `'N/A'` として持つ（torchvision の `__background__` に相当）など、ラベル変換の細部が torchvision と違う点にも注意してください。より軽量な **RT-DETR v2（`PekingU/rtdetr_v2_r18vd`）** も、まったく同じ `AutoModelForObjectDetection` ＋ `post_process_object_detection` で動きます。CPU で DETR が重いと感じたら RT-DETR の軽量版に差し替えるのが定石です（どちらもバックボーン読み込みに `timm` が必要）。

## 7. 2-stage / 1-stage / DETR の使い分けと CPU 向け軽量モデル

ここまでで3つの設計思想が出そろいました。実務でのモデル選定は、精度・速度・実装の手間のトレードオフであり、「どれが正解」というものはありません。代表的な系統を下表に整理したので、要件（リアルタイム性・小物体精度・後処理の単純さ）に応じて選んでください。

| 系統 | 代表モデル | 特徴 | 後段NMS | CPU向きの軽量版 |
| --- | --- | --- | --- | --- |
| 2-stage | Faster R-CNN, Mask R-CNN | RPN→ROI の2段。高精度・低速 | 必要 | `fasterrcnn_mobilenet_v3_large_320_fpn` |
| 1-stage (anchor) | RetinaNet, SSD, FCOS | 1パスで密に予測。高速 | 必要 | `ssdlite320_mobilenet_v3_large` |
| DETR系 (set pred) | DETR, RT-DETR, RF-DETR | クエリ＋二部マッチング | 不要 | `PekingU/rtdetr_v2_r18vd` |
| YOLO系 | YOLO11, YOLO26 | 高速・実装が容易（§8） | 11は内蔵/26は不要 | `yolo11n`, `yolo26n` |

表の右端が本章の CPU 方針です。一般論として **2-stage は精度が高いが遅く、1-stage は速いが小物体にやや弱い**、**DETR系は後処理が単純（NMS不要）だが学習が難しく収束が遅い**という性格があります。CPU で素早く回したいなら `ssdlite320`（最軽量）や `fasterrcnn_mobilenet_v3_large_320_fpn`、RT-DETR の r18 版を既定にし、入力解像度（`min_size`/`max_size` や YOLO の `imgsz`）を下げるのが効きます。半精度（fp16）は CPU では遅い/未対応なので **float32 のまま**使ってください。

`03_detection_benchmark.py` は、この使い分けを体感するために torchvision（`ssdlite320`）と HF DETR を**統一インターフェース**（`detect()` が `{boxes, scores, labels, names}` を返す薄いラッパ）で束ね、同じ閾値・同じ可視化経路で並べて推論時間を比較します。フレームワークが違っても出力を同じ形に正規化しておけば、評価も可視化もモデル非依存の1経路で書ける——これが実務のベンチマーク基盤の最小形です。

## 8. Ultralytics YOLO の位置づけ — なぜ「実行経路に入れない」か

物体検出といえば **YOLO** を思い浮かべる人も多いでしょう。Ultralytics の YOLO は `YOLO("yolo11n.pt")` でロードし `model(img)` で即推論、`results[0].boxes.xyxy/.conf/.cls` で結果を取り、`results[0].plot()` で可視化画像（numpy BGR）まで一発、という**圧倒的な手軽さ**が魅力です。CPU でも `yolo11n`/`yolo26n` は実用的に動きます。

それにもかかわらず、**本講座では YOLO を実行経路に入れません**。その理由は依存衝突にあります。`ultralytics` は `opencv-python`（GUI 付きフル版）を引き込みますが、本プロジェクトは headless 環境（Docker/CI）で動くよう `opencv-python-headless` を採用しています。この2つは**同じ `cv2` を提供する排他パッケージ**で、同居させると一方が他方を壊します。そこで YOLO は「概念紹介＋（試したい人向けに）別環境での `uv add` 案内」に留めています。

```python
# ── 参考: YOLO の1行推論（本講座では実行しない。別環境で uv add ultralytics）──
from ultralytics import YOLO
model = YOLO("yolo11n.pt")          # 重みは自動DL
results = model(img)                # 1行で推論
boxes = results[0].boxes.xyxy       # 検出box、.conf=スコア、.cls=クラスID
vis = results[0].plot()            # 可視化（numpy BGR）
# 注意: YOLO11 は出力に NMS 内蔵、YOLO26 は NMS-free。後段で再度 NMS を掛けて二重抑制しないこと。
```

`03` の出力 `print` でも、この注意書きを `yolo_note` として残しています。「ライブラリが便利でも、プロジェクトの依存方針（headless 統一）と衝突するなら採用しない」という判断は、実務で頻繁に直面するトレードオフです。本講座はその一例として YOLO を扱っています。

## 9. mAP の簡易算出 — torchmetrics で「定量評価」を体験する

検出の良し悪しは、最終的には**数値**で測ります。その標準が **mAP（mean Average Precision）** です。考え方は「予測を confidence 降順に並べ、IoU が閾値以上で未マッチの正解（GT）に貪欲に対応付け、対応すれば TP・余れば FP・未検出 GT は FN とし、PR 曲線を描いてその下の面積（AP）をクラス平均する」というものです。IoU 閾値を 0.5 に固定したものが **mAP@0.5**、0.50〜0.95 を 0.05 刻みで平均したものが **mAP@[.5:.95]**（COCO の主指標）です。

<figure class="lec-fig"><svg viewBox="0 0 660 260" role="img" aria-label="mAPはconfidence降順ソート・IoU貪欲マッチング・TP FP FN集計・PR曲線・AP面積のクラス平均という一本道で算出する" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><text x="330" y="40" text-anchor="middle" font-size="15" font-weight="700" fill="#18181b">mAP の算出ステップ（confidence 降順 → AP → クラス平均）</text><rect x="22" y="78" width="104" height="92" rx="6" fill="#eff6ff" stroke="#2563eb" stroke-width="2"/><text x="74" y="116" text-anchor="middle" font-size="13.5" font-weight="700" fill="#1d4ed8">① 降順ソート</text><text x="74" y="144" text-anchor="middle" font-size="11" fill="#52525b">confidence 順</text><rect x="150" y="78" width="104" height="92" rx="6" fill="#eff6ff" stroke="#2563eb" stroke-width="2"/><text x="202" y="116" text-anchor="middle" font-size="13.5" font-weight="700" fill="#1d4ed8">② マッチング</text><text x="202" y="144" text-anchor="middle" font-size="11" fill="#52525b">IoU で GT に</text><rect x="278" y="78" width="104" height="92" rx="6" fill="#eff6ff" stroke="#2563eb" stroke-width="2"/><text x="330" y="116" text-anchor="middle" font-size="13.5" font-weight="700" fill="#1d4ed8">③ TP/FP/FN</text><text x="330" y="144" text-anchor="middle" font-size="11" fill="#52525b">1 GT=1 予測</text><rect x="406" y="78" width="104" height="92" rx="6" fill="#eff6ff" stroke="#2563eb" stroke-width="2"/><text x="458" y="116" text-anchor="middle" font-size="13.5" font-weight="700" fill="#1d4ed8">④ PR 曲線</text><text x="458" y="144" text-anchor="middle" font-size="11" fill="#52525b">→ AP（面積）</text><rect x="534" y="78" width="104" height="92" rx="6" fill="#ffedd5" stroke="#ea580c" stroke-width="2"/><text x="586" y="116" text-anchor="middle" font-size="13.5" font-weight="700" fill="#c2410c">⑤ クラス平均</text><text x="586" y="144" text-anchor="middle" font-size="11" fill="#52525b">= mAP</text><line x1="130" y1="124" x2="141" y2="124" stroke="#71717a" stroke-width="2"/><polygon points="150,124 140,119 140,129" fill="#71717a"/><line x1="258" y1="124" x2="269" y2="124" stroke="#71717a" stroke-width="2"/><polygon points="278,124 268,119 268,129" fill="#71717a"/><line x1="386" y1="124" x2="397" y2="124" stroke="#71717a" stroke-width="2"/><polygon points="406,124 396,119 396,129" fill="#71717a"/><line x1="514" y1="124" x2="525" y2="124" stroke="#71717a" stroke-width="2"/><polygon points="534,124 524,119 524,129" fill="#71717a"/><text x="330" y="206" text-anchor="middle" font-size="11.5" fill="#52525b">IoU 0.5 固定＝mAP@0.5 ／ 0.50〜0.95 平均＝mAP@[.5:.95]</text></svg><figcaption>mAP（mean Average Precision）は<b>一本道の手順</b>で出します。<b>① 予測を confidence 降順にソート → ② IoU で正解(GT)に貪欲マッチング → ③ TP/FP/FN を判定 → ④ PR 曲線 → ⑤ その下の面積=AP をクラス平均=mAP</b>。マッチングは <b>1 GT に 1 予測</b>（重複は FP）が鉄則です。IoU 閾値を 0.5 に固定したものが <code>mAP@0.5</code>、0.50〜0.95 を 0.05 刻みで平均したものが <code>mAP@[.5:.95]</code>（COCO 主指標）です。</figcaption></figure>

本章では `torchmetrics.detection.MeanAveragePrecision` を使い、**予測と正解を box 辞書のリストで渡すだけ**で mAP を出す体験をします。`03` の `demo_map()` は、合成シーンには正解アノテーションが無いため、**決め打ちの GT と予測**（2枚分のミニデータセット）で計算手順を確認します。実モデルの mAP を測るときは、各画像の予測とデータセットの GT を同じ形式で渡すだけです。

```python
from torchmetrics.detection import MeanAveragePrecision

preds  = [dict(boxes=..., scores=..., labels=...), ...]  # 予測（scores 必須）
targets = [dict(boxes=..., labels=...), ...]             # 正解（scores は無し）
metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")
metric.update(preds, targets)
out = metric.compute()   # out["map"]=mAP@[.5:.95], out["map_50"], out["map_75"]
```

`03` を実行すると、`mAP@0.5 = 1.0` に対し `mAP@[.5:.95] = 0.875` のように**厳しい IoU 閾値ほど値が下がる**様子が観察できます。これは「ゆるい IoU では当たり扱いの箱も、ピクセル単位で見ると少しズレている」ことの定量化です。mAP の**中身を numpy で一から実装し、pycocotools の `COCOeval` と一致を確認する**のは第19回で行います。本章は「ライブラリでまず出してみる」ところまでです。

## 10. 合成画像の限界と「実画像の置き方」

本章のスクリプトは、ルール上ネットへ出てよいのが**モデル重みのDLのみ**なので、入力画像は `detection_helpers.make_scene_image()` が**合成生成**します（人物・車らしき単純図形）。すでに見たとおり、COCO 検出器はこの抽象図形に対して `stop sign` や `umbrella` といった**的外れなラベル**を付けがちです。これは検出器の故障ではなく、「写実的な写真で学習したモデルは、分布外の合成図形では信頼できない」という**ドメインギャップ**の生きた実例です。

重要なのは、**それでもスクリプトは必ず `exit 0` になる**よう設計してある点です。検出がゼロでもパイプラインは最後まで走り、図と JSON を保存します。「検出が乏しい＝コードが壊れている」ではないことを、最初に体で覚えておいてください。

実用的な検出結果を見たいときは、`data/18_object_detection_intro/` に実写の `.jpg`/`.png` を置いてください。`load_input_image()` がそれを自動で優先して読み込み、合成画像にフォールバックする導線になっています。人や車が写った普通の写真を1枚置くだけで、`person 0.99` のような妥当な検出に変わり、本章のパイプラインがそのまま実用に耐えることが確認できます。

```bash
# 実写で試す: このフォルダに写真を置くと自動で使われる（無ければ合成にフォールバック）
mkdir -p data/18_object_detection_intro
cp ~/Pictures/street.jpg data/18_object_detection_intro/
uv run python lectures/18_object_detection_intro/01_torchvision_detection.py
```

## 11. 動かし方

まず依存グループを入れ、スクリプトを順に実行します。初回はモデル重みのDL（ssdlite≈13MB / retinanet≈130MB / fasterrcnn≈160MB / DETR≈160MB）が走るためネット接続が要りますが、2回目以降はキャッシュから即起動します。すべて CPU・合成画像で完走し、結果は `outputs/18_object_detection_intro/` に図と JSON で保存されます。

### スクリプト一覧

| ファイル | 役割 | 主な出力 | 重みDL |
| --- | --- | --- | --- |
| `detection_helpers.py` | 共通の道具箱（device 判定 / 合成画像 / 後処理 / 可視化 / 保存） | `00_scene.png`（単体実行時） | なし |
| `01_torchvision_detection.py` | weights API で 3 モデル横断検出（4点セット・閾値・NMS・可視化） | `01_torchvision_compare.png` / `01_torchvision_results.json` | ssd/retina/frcnn |
| `02_detr_huggingface.py` | HF DETR の `post_process` と `target_sizes=(H,W)` の罠を可視化 | `02_detr_target_sizes.png` / `02_detr_results.json` | DETR |
| `03_detection_benchmark.py` | torchvision vs DETR を統一APIで比較し torchmetrics で mAP | `03_benchmark_compare.png` / `03_benchmark.json` | ssd/DETR |
| `mini_project.py` | **章末課題**: 検出パイプライン × 自作 mAP@0.5 を1本に統合 | `mini_detection_overlay.png` / `mini_pr_curve.png` / `mini_project.json` | ssdlite |
| `use_case.py` | **実践ユースケース**: 物体・人数カウンター（指定クラスを数えて注記を焼き込む小ツール） | `use_case_count_*.png` / `use_case_overview.png` / `use_case_counts.json` | ssdlite |
| `exercises.py` | 演習 10 問（IoU〜AP補間）。TODO を埋めて自己採点 | （標準出力の採点表） | なし |
| `exercises_solutions.py` | 演習の模範解答ランナー（全10問 PASS を確認） | （標準出力の採点表） | なし |

```bash
# 依存（深層学習の土台 + HuggingFace + 評価指標）
uv sync --group dl --group hf --group metrics

# 1) torchvision の weights API で3モデル横断検出（4点セット・閾値・NMS・可視化）
uv run python lectures/18_object_detection_intro/01_torchvision_detection.py

# 2) HF DETR と post_process / target_sizes=(H,W) の罠の可視化
uv run python lectures/18_object_detection_intro/02_detr_huggingface.py

# 3) torchvision vs DETR を統一APIで比較し、torchmetrics で mAP を出す
uv run python lectures/18_object_detection_intro/03_detection_benchmark.py

# 章末ミニプロジェクト: 検出 → 後処理 → 可視化 → 自作mAP の検算まで一気通貫
uv run python lectures/18_object_detection_intro/mini_project.py

# 実践ユースケース: 物体・人数カウンター（指定クラスを数えて画像に注記を焼き込む小ツール）
uv run python lectures/18_object_detection_intro/use_case.py
# 数えるクラス・閾値は環境変数で変えられる（例: 人だけを 0.5 以上で数える）
USE_CASE_CLASSES="person" USE_CASE_SCORE=0.5 \
  uv run python lectures/18_object_detection_intro/use_case.py

# 演習: まずは TODO を自分で埋める（最初は全部 FAIL だが exit 0）
uv run python lectures/18_object_detection_intro/exercises.py
# どうしても分からない時だけ、模範解答の挙動を見る
SHOW_SOLUTION=1 uv run python lectures/18_object_detection_intro/exercises.py
# 模範解答だけ走らせて答え合わせ（全10問 PASS）
uv run python lectures/18_object_detection_intro/exercises_solutions.py

# （任意）実画像で試す: data/18_object_detection_intro/ に .jpg/.png を置くと自動で使われる
```

実行後は、`01_torchvision_compare.png`（モデル横断の検出比較）、`02_detr_target_sizes.png`（`(H,W)` 正・`(W,H)` 誤の対比）、`03_benchmark_compare.png`（フレームワーク横断比較）、`mini_pr_curve.png`（PR 曲線と AP 面積）の各図を、解説と照らし合わせて眺めてください。とくに `02` の右パネルの「歪んだ箱」を一度見ておくと、`target_sizes` のバグを一生忘れません。

## 12. よくあるエラーと対処（チェックリスト）

最後に、本章でつまずきやすい点を「症状 → 原因 → 対処」でまとめます。検出特有・transformers v5 特有の罠が多いので、詰まったらまずここを見てください。

| 症状 | ほぼ確実な原因 | 対処 |
| --- | --- | --- |
| `model(x)` で TypeError | 検出モデルは「テンソルのリスト」を取る | `model([x])[0]` のように包む |
| 検出が全部ズレた名前 | `labels` の生整数を名前と誤解（index0=背景） | `weights.meta["categories"][int(label)]` で引く |
| DETR の box が縦横に歪む | `target_sizes` が `(W,H)` になっている | `(H,W)` 順で渡す（`image.size[::-1]`） |
| 可視化が真っ黒/例外 | `draw_bounding_boxes` に float を渡した | `uint8` の CHW テンソルに変換してから渡す |
| 隣接する別クラスの箱が消える | 全クラスまとめて `nms` を掛けた | `batched_nms(boxes, scores, labels, iou)` を使う |
| 精度がやけに低い | 検出モデルに ImageNet 正規化を二重がけ | 正規化はモデル内部。`weights.transforms()` に任せる |
| DETR ロードで backbone エラー | `timm` 未導入 | `uv sync --group hf`（`timm` 同梱）で入れる |
| 結果やメモリが不安定 | `eval()` / `inference_mode()` を忘れた | 推論前に `.eval()`、本体を `inference_mode()` で囲む |
| CPU で極端に遅い | `half`/fp16 を CPU で使用 or 解像度が高い | float32 のまま。`min_size`/`max_size` を下げる |
| `cv2` の import が壊れる | `ultralytics` が `opencv-python`(full) を導入 | YOLO は別環境で。本講座は headless 統一（§8） |

この表の上3つ（リスト入力・index0=背景・`target_sizes`=(H,W)）が、検出入門で最も多い不具合です。症状を見たら原因を即座に言い当てられるようにしておきましょう。

## 13. まとめ

本章では、物体検出を**「クラス＋確信度＋矩形の可変個集合」**として捉え直すところから出発し、torchvision の **weights API（4点セット）**、検出モデル特有の **`[0,1]` float CHW リスト入力・内部正規化**、**score 閾値 → `batched_nms` → `draw_bounding_boxes`** の定番パイプライン、そして HF DETR の **set prediction・`post_process_object_detection`・`target_sizes=(H,W)`** までを、すべて合成画像の上で「自分で再現し、図と数字で確認できる」レベルで扱いました。通底する勘所は3つ——**ラベルの index0 は背景**、**`target_sizes` は (H,W)**、**正規化はモデルに任せる**です。

さらに、torchvision と DETR を**統一インターフェース**で束ね、`torchmetrics` で **mAP@0.5 / mAP@[.5:.95]** を出すところまで通し、YOLO を「便利だが依存方針と衝突するため概念に留める」実務判断の例として位置づけました。ここで身につけた「検出器 → 閾値・NMS → 可視化・評価」という骨格は、次の第19回（mAP を numpy で一から実装し COCOeval と突き合わせる）、第20回（オープン語彙検出）、第21〜23回（セグメンテーション）へとそのまま繋がります。まずは演習を全問 PASS させ、`02` の「歪んだ箱」を自分の言葉で説明できるようにしてから次へ進んでください。

---

## 🛠 章末ミニプロジェクト — 「検出パイプライン × 自作 mAP」を1本に統合する

ここまでの部品（4点セット・`[0,1]` リスト入力・閾値・`batched_nms`・可視化・mAP）を、`mini_project.py` で**ひと続きの評価系**に組み上げます。これは「検出器を動かす」だけでなく、**その良し悪しを自分で数値化し、ライブラリで検算する**という実務の最小ループを体験する総合課題です。

**やること（2部構成）**

<figure class="lec-fig"><svg viewBox="0 0 660 320" role="img" aria-label="ミニプロジェクトは検出パイプラインで予測を作りpseudo-GTでmAPを測り、自作mAPとtorchmetricsを突き合わせてabs_diffを見る" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><text x="330" y="26" text-anchor="middle" font-size="14" font-weight="700" fill="#18181b">ミニプロジェクトの流れ：検出 → mAP 測定 → 自作 mAP で検算</text><rect x="130" y="44" width="400" height="46" rx="6" fill="#ffedd5" stroke="#ea580c" stroke-width="2"/><text x="330" y="64" text-anchor="middle" font-size="13.5" font-weight="700" fill="#c2410c">検出パイプライン（Part A）</text><text x="330" y="81" text-anchor="middle" font-size="11.5" fill="#52525b">ssdlite → 閾値 → batched_nms → 可視化</text><line x1="330" y1="90" x2="330" y2="103" stroke="#71717a" stroke-width="2"/><polygon points="330,112 325,102 335,102" fill="#71717a"/><rect x="160" y="112" width="340" height="46" rx="6" fill="#eff6ff" stroke="#2563eb" stroke-width="2"/><text x="330" y="132" text-anchor="middle" font-size="13.5" font-weight="700" fill="#1d4ed8">予測 と 正解(GT) を用意</text><text x="330" y="149" text-anchor="middle" font-size="11" fill="#52525b">pseudo-GT ／ controlled な GT</text><line x1="330" y1="158" x2="330" y2="176" stroke="#71717a" stroke-width="2"/><line x1="330" y1="176" x2="196" y2="191" stroke="#71717a" stroke-width="2"/><polygon points="186,192 195.4,185.9 196.5,195.9" fill="#71717a"/><line x1="330" y1="176" x2="464" y2="191" stroke="#71717a" stroke-width="2"/><polygon points="474,192 463.5,195.9 464.6,185.9" fill="#71717a"/><rect x="70" y="192" width="232" height="56" rx="6" fill="#eff6ff" stroke="#2563eb" stroke-width="2"/><text x="186" y="218" text-anchor="middle" font-size="13" font-weight="700" fill="#1d4ed8">自作 mAP@0.5</text><text x="186" y="236" text-anchor="middle" font-size="11" fill="#52525b">全点補間（PASCAL VOC）</text><rect x="358" y="192" width="232" height="56" rx="6" fill="#eff6ff" stroke="#2563eb" stroke-width="2"/><text x="474" y="218" text-anchor="middle" font-size="12.5" font-weight="700" fill="#1d4ed8">torchmetrics map_50</text><text x="474" y="236" text-anchor="middle" font-size="11" fill="#52525b">COCO 101 点補間</text><line x1="186" y1="248" x2="278" y2="268" stroke="#71717a" stroke-width="2"/><polygon points="288,270 277.2,272.8 279.3,262.9" fill="#71717a"/><line x1="474" y1="248" x2="382" y2="268" stroke="#71717a" stroke-width="2"/><polygon points="372,270 380.7,262.9 382.8,272.8" fill="#71717a"/><rect x="178" y="270" width="304" height="42" rx="6" fill="#fafafa" stroke="#16a34a" stroke-width="2.5"/><text x="330" y="290" text-anchor="middle" font-size="13" font-weight="700" fill="#15803d">突き合わせ：abs_diff ≈ 0.0016</text><text x="330" y="306" text-anchor="middle" font-size="11" fill="#52525b">補間方式の違い（誤りではない）</text></svg><figcaption>章末ミニプロジェクトは <b>2 部構成</b>です。<b>Part A</b> で <code>ssdlite320</code> の検出パイプライン（score 閾値 → <code>batched_nms</code> → 可視化）を回し、<b>pseudo-GT で mAP@0.5</b> を測ります。<b>Part B</b> では同じ予測と GT に対し<b>自作 mAP（全点補間）</b>と <b>torchmetrics（COCO 101 点補間）</b>を当て、両者を突き合わせます。<b>値がわずかに違う（<code>abs_diff ≈ 0.0016</code>）のは補間方式の差</b>であって誤りではない、というのが核心の学びです。</figcaption></figure>

- **Part A — 検出パイプライン**: `ssdlite320` を 4点セットでロードし、`[0,1]` float CHW リスト入力 → `eval()`＋`inference_mode()` 推論 → score 閾値＋クラス別 NMS → `draw_bounding_boxes` 可視化、を一気通貫で回します。合成画像のときは `make_scene_image()` が描く図形の**既知座標を pseudo-GT** とし、検出器の **mAP@0.5 を測定**します。COCO 学習器は抽象図形に弱いため、出力ラベルは `stop sign` ばかりになり、**mAP はほぼ 0**。これは故障ではなく、§10 で述べた**ドメインギャップを数値で見た**もので、`mini_detection_overlay.png` の「pseudo-GT（緑）vs 予測（赤）」を見比べると、箱の位置は近いのにラベルが総崩れ、という様子がよく分かります。

- **Part B — 自作 mAP の検算（第19回の予告）**: 演習 ex7〜ex9（IoU マッチング → PR 曲線 → 全点補間 AP）を**複数画像・複数クラスへ拡張**した `mean_average_precision_50()` を自作し、決め打ちの controlled シナリオ（person は TP2・FP2・FN1、car は TP1）で **mAP@0.5 を計算**、`torchmetrics` の `map_50` と突き合わせます。実行すると次のように出ます。

  ```text
  per-class AP@0.5 = {1: 0.6667, 3: 1.0}   (1=person, 3=car)
  self  mAP@0.5      = 0.8333   (全点補間)
  torchmetrics map_50 = 0.8317  (COCO 101点補間)
  ```

  **値がぴったり一致しない**のが学びの核です。自作は PASCAL VOC2010+ の**全点補間**、torchmetrics は **COCO 101点補間**なので、同じ PR 曲線でも AP がわずかに違います（`abs_diff ≈ 0.0016`）。「mAP という単一の数字にも“流儀”がある」ことを、`mini_pr_curve.png`（PR 曲線と AP 面積の陰影）と合わせて体に入れてください。person クラスは recall が 0.667 で頭打ち（FN が 1 個残る）になり、AP 面積がその先で 0 に落ちる様子が一目で分かります。

**発展課題（任意）**: ① `data/18_object_detection_intro/` に実写を置き、自分でアノテーションした GT で実測 mAP を出す。② 検出器を `fasterrcnn_resnet50_fpn` に替えて mAP と速度のトレードオフを比較。③ `ex10`（11点補間）でも AP を出し、3 方式（全点 / 11点 / COCO101点）の差を表にする。④ IoU 閾値を 0.5→0.75 に上げ、`map_50` と `map_75` の落差を観察する。

## ✅ 到達チェックリスト

次が「自分の言葉で説明でき、コードで再現できる」なら本章は卒業です。

- [ ] 物体検出の出力が **`{boxes(xyxy), scores, labels}` の可変個集合**であり、分類との違いが「N が画像ごとに変わる」点だと説明できる。
- [ ] torchvision の **weights API 4点セット**（`weights / build / transforms / categories`）を諳んじ、モデル差し替えが enum 1 行で済むと示せる。
- [ ] 検出モデルが **`[0,1]` float CHW テンソルの“リスト”** を取り、正規化は内部で行う（二重正規化禁止）ことを理由付きで言える。
- [ ] **COCO ラベルの index0 = `__background__`**、DETR の `id2label[0]='N/A'` を踏まえ、`categories[int(label)]` で名前を引ける。
- [ ] **score 閾値 → `batched_nms`** の2段後処理を書け、なぜ `nms` でなく `batched_nms` なのか（別クラスの誤抑制回避）を説明できる。
- [ ] `draw_bounding_boxes` が **uint8 CHW** を要求する理由と、float を渡したときの症状を知っている。
- [ ] DETR の生 `pred_boxes` が **cxcywh 正規化**で、`post_process_object_detection` の **`target_sizes=(H,W)`** を通して初めて xyxy 絶対座標になると言える（`image.size[::-1]`）。
- [ ] **IoU の定義**（交差/和集合）を実装でき、NMS と mAP マッチングの両方で使われると理解している。
- [ ] mAP の手順 **「confidence 降順 → IoU 貪欲マッチング → TP/FP → PR 曲線 → AP 補間 → クラス平均」** を口頭で再現でき、**1 GT に 1 予測**（重複は FP）を守れる。
- [ ] **AP 補間の3流儀**（PASCAL 11点 / 全点 / COCO 101点）で値が変わることを知り、`mAP@0.5` と `mAP@[.5:.95]` を取り違えない。
- [ ] 合成画像で検出が乏しくても**パイプラインは exit 0**で、それが「ドメインギャップ＝正常」だと判断できる。
- [ ] 演習 `exercises.py` を **全10問 PASS** できる（`exercises_solutions.py` で答え合わせ）。

## ✍️ 演習問題

演習は `exercises.py` に TODO 形式で入っています。各 TODO を実装し `uv run python lectures/18_object_detection_intro/exercises.py` を実行すると自己採点できます（`exercises_solutions.py` が解答）。

1. 2つの box（xyxy）の IoU ＝ 交差面積 / 和集合面積 を返す（重なりなしは 0、ゼロ割は 0.0）（`ex1_iou` の TODO）。
2. xyxy 形式の box を cxcywh（中心x・中心y・幅・高さ）に変換する（`ex2_xyxy_to_cxcywh` の TODO）。
3. score が閾値以上の検出だけを残し、生検出を間引く（`ex3_score_filter` の TODO）。
4. 1クラス NMS。confidence 降順に貪欲に箱を選び、IoU が閾値超の重複を抑制して keep した index を返す（`ex4_nms` の TODO）。
5. 正規化 cxcywh（0〜1）を、画像サイズ (H,W) に合わせた xyxy 絶対座標へ変換する（height と width の取り違えに注意）（`ex5_cxcywh_norm_to_xyxy_abs` の TODO）。
6. クラス別 NMS。labels ごとに独立に NMS を掛け、残った index を全クラス分まとめる（`ex6_batched_nms` の TODO）。
7. 予測を score 降順に GT へ貪欲対応付けし、TP/FP（0/1）を判定する（1 GT に 1 予測、重複は FP）（`ex7_match_detections` の TODO）。
8. score 降順の TP/FP 列から、累積 precision / recall を作る（`ex8_pr_curve` の TODO）。
9. 全点補間（PASCAL VOC2010+）で、PR 曲線の下面積として AP を求める（`ex9_ap_all_points` の TODO）。
10. 11点補間（PASCAL VOC2007）で AP を求め、補間方式で値が変わることを確認する（`ex10_ap_11_point` の TODO）。

## ❓ よくある落とし穴・FAQ・デバッグ

§12 の「症状→原因→対処」表と合わせて、**評価（mAP）まわり**の疑問を Q&A で補強します。

- **Q. 自作 mAP が torchmetrics と少しズレる。バグ？** A. たいていバグではなく**補間方式の違い**です。本章の自作は全点補間、torchmetrics は COCO 101点補間。`mini_project.py` の `abs_diff` が 0.01 未満ならまず一致とみなしてよい。完全一致を狙うなら 101 個の recall 点で補間を揃えます（第19回）。
- **Q. mAP が 0.0 になる。** A. ① 予測と GT で**ラベル整数の規約がズレている**（person を 1 と 0 で食い違わせている等）。mAP はクラス別評価なので、ラベルが噛み合わないと全 FP 扱いになる。② **box 形式の不一致**（GT が xywh、予測が xyxy）。`MeanAveragePrecision(box_format=...)` と実データを合わせる。③ 合成画像＋pseudo-GT なら**ドメインギャップで本当に 0**（正常）。
- **Q. `MeanAveragePrecision.compute()` が遅い／結果が混ざる。** A. `update()` を**バッチごとに呼んで最後に1回 `compute()`**。毎バッチ `compute()` したり、次の評価前に `reset()` し忘れると累積が壊れます。`preds` と `targets` は同じ device に揃える。
- **Q. NMS 後も同じ物体に箱が2つ残る。** A. ① IoU 閾値が高すぎ（`0.5` 付近へ）。② 別クラスとして検出され `batched_nms` が別物扱い（ラベルを見直す）。③ DETR は元々 NMS 不要なので、**後段で重ねて NMS をかけない**（二重抑制で逆に消えることも）。
- **Q. 予測ゼロ。閾値を下げるべき？** A. まず `n_raw`（後処理前の生検出数）を見る。生がゼロなら閾値ではなくモデル/入力の問題（合成画像のドメインギャップ）。生はあるのに kept がゼロなら閾値を下げる。本章は合成で `0.3` に下げています。
- **Q. `target_sizes` を直したのに box がまだ変。** A. `(H,W)` 順は直っても、**バッチで複数画像を渡すと `target_sizes` はリスト**（画像ごとに `(H,W)`）。1枚なら `[(H,W)]` と**リストで包む**のを忘れがち。
- **Q. CPU で DETR が重い。** A. `PekingU/rtdetr_v2_r18vd` に差し替え（同じ `AutoModelForObjectDetection`＋`post_process`）。torchvision なら `ssdlite320` か `fasterrcnn_mobilenet_v3_large_320_fpn`。`min_size/max_size` や `imgsz` を下げ、**fp16 は使わない**（CPU では遅い/未対応）。
- **Q. 1つの GT に複数の正しそうな予測。全部 TP でいい？** A. ダメ。**1 GT には最高スコアの1予測だけ TP**、残りは IoU が高くても FP。これを守らないと mAP が水増しされます（`ex7` の核心）。
- **デバッグの定石**: 箱が変なら **`(1)` 形式（xyxy/xywh/cxcywh）→ `(2)` 座標系（正規化/絶対）→ `(3)` `(H,W)` 順** の順で疑う。ラベルが変なら **index0=背景** と `id2label/categories` の引き方を確認。値（mAP）が変なら **ラベル規約・box_format・補間方式** を確認。

## 🚀 発展トピック・参考

- **IoU の発展**: `generalized_box_iou`（GIoU）/ DIoU / CIoU は、重なりゼロでも勾配が出るよう距離やアスペクト比を加味した IoU 系指標で、学習の loss や高度な NMS に使われます。`torchvision.ops.generalized_box_iou` で試せます。
- **NMS の発展**: Soft-NMS（重なる箱をスコア減衰させる）、class-agnostic NMS、weighted box fusion（複数モデルの箱を統合）など。DETR/YOLO26 のように **NMS-free** に向かう流れも押さえると、後処理設計の引き出しが増えます。
- **COCO 公式評価の細部（第19回）**: `pycocotools` の `COCOeval` は `AP / AP50 / AP75`、面積別 `AP_S(<32²)/M/L`、`AR@{1,10,100}` を出します。`areaRng` と `maxDets` の既定を変えると数値が比較不能になる点に注意。自作 mAP との一致確認が第19回の主題です。
- **モデルの広がり**: RT-DETR v2 / RF-DETR（DETR 系の高速・高精度化）、Ultralytics YOLO11/YOLO26（本講座は依存衝突で実行せず概念のみ）、Faster/Mask R-CNN（2-stage の定番）。検出の先は **オープン語彙検出（第20回: OWL-ViT/OWLv2/Grounding DINO）** と **セグメンテーション（第21〜23回: SegFormer/Mask R-CNN/SAM/CLIPSeg）** へ繋がります。
- **公式ドキュメント**: torchvision detection models（<https://docs.pytorch.org/vision/stable/models.html>）、torchvision ops（<https://docs.pytorch.org/vision/stable/ops.html>）、transformers object detection（<https://huggingface.co/docs/transformers>）、torchmetrics detection（<https://lightning.ai/docs/torchmetrics/stable/>）、COCO 評価（<https://cocodataset.org/#detection-eval>）。

## 💡 実践ユースケース集

検出は「箱を出す」こと自体より、**箱を“数える・絞る・知らせる”** ところで価値が出ます。本章で組んだ「検出器 → 閾値・NMS → 可視化」という骨格は、ほとんどそのまま現実の小ツールになります。以下に身近な応用を3つ挙げ、最後の1つは実際に動くスクリプト `use_case.py` として同梱しました。

### ① 物体・人数カウンター（同梱 `use_case.py`）

- **何に使うか**: 写真に写った人や車などを「クラスごとに何個あるか」数えて画像に焼き込み、フォルダ全体の合計を集計します。来店人数のざっくり計測、駐車場の在車台数チェック、棚の物品個数の目視補助、イベントの混雑度モニタなどの出発点です。
- **作り方の要点**: `ssdlite320` で検出 → 共通後処理（score 閾値 ＋ `batched_nms`）→ **指定クラスだけに絞って `count()`** → `draw_bounding_boxes` で箱を描き、その上に PIL で「`person: 3` / `car: 1` / `TOTAL: 4`」の半透明バナーを重ねるだけです。`mini_project.py` が「mAP で良し悪しを**測る**評価ベンチ」なのに対し、こちらは mAP を測らず「数えて注記する**現実の単機能ツール**」に振ってある点が違いです。
- **注意**: 合成画像では COCO 検出器が反応しにくく、カウントが 0（や `stop sign` の誤検出）になりがちです。これは故障ではなく**ドメインギャップ＝正常**。実写を `data/18_object_detection_intro/` に置けば、そのまま実用カウンタになります。クラス名の規約（`person` であって `people` ではない）と閾値（実写は 0.5 前後）に注意してください。

```bash
# 既定（person, car を数える）。data/ に画像があれば全部処理、無ければ合成1枚で完走。
uv run python lectures/18_object_detection_intro/use_case.py

# 実写で実用に: data/<id>/ に写真を置くと自動で全部処理される
mkdir -p data/18_object_detection_intro
cp ~/Pictures/*.jpg data/18_object_detection_intro/
# 数えるクラスと閾値は環境変数で調整（コードを書き換えない）
USE_CASE_CLASSES="person,car,bus" USE_CASE_SCORE=0.5 \
  uv run python lectures/18_object_detection_intro/use_case.py
```

**拡張アイデア**: ① `cv2.VideoCapture` で動画を N フレームおきに切り出し、時系列のカウント推移を CSV/折れ線にする。② ROI（駐車枠・レジ前など）の矩形を決め、その中に箱の中心が入るものだけ数える。③ `TOTAL` がしきい値を超えたら `"CROWDED"` を注記/ログする簡易の混雑アラート。④ 検出器を `fasterrcnn_resnet50_fpn` や RT-DETR(`PekingU/rtdetr_v2_r18vd`) に差し替えて遠景・小物体の人に強くする（CPU では遅くなるトレードオフ）。

### ② 「禁止エリア」侵入・存在チェック（ROI ゲート）

- **何に使うか**: 「立入禁止ゾーンに人がいないか」「搬入口に車が停まっていないか」を、検出ボックスと事前に決めた監視矩形（ROI）の重なりだけで判定する簡易アラート。
- **作り方の要点**: ①のカウンタを土台に、検出 box と ROI の **IoU もしくは box 中心の内外判定**を足すだけ。`torchvision.ops.box_iou` で ROI との重なりを測り、閾値超えがあれば `ALERT`。可視化は ROI を黄色、侵入 box を赤で描き分けます。
- **注意**: カメラが固定でないと ROI 座標がズレます。誤報を減らすには「数フレーム連続で侵入」を条件にする（単発のチラつき検出を無視）。夜間・逆光は検出が落ちるので閾値とモデルを実環境で調整します。

### ③ クラス別の自動タグ付け・画像仕分け

- **何に使うか**: 大量の写真を「人が写っている／車が写っている／何も写っていない」へ自動振り分けし、ギャラリーの検索タグやデータセット下ごしらえに使う。
- **作り方の要点**: 各画像を検出し、**score 閾値を超えるクラスの集合**をそのままタグにする（個数まで要らなければ存在判定だけでよい）。タグを JSON/CSV に書き出し、`shutil.move` でクラス別フォルダへ振り分けます。重い mAP も NMS の厳密さも不要で、`score >= 0.5` の有無だけで実用になります。
- **注意**: 1枚に複数クラスが共存する（人＋車）ので「単一ラベル分類」ではなく**マルチラベル**として扱うこと。閾値が低いと誤タグ、高いと取りこぼし——運用データで閾値を1度キャリブレーションしてから回します。

---

> 本教材で参照・検証したライブラリとバージョン（torch 2.12+cpu / torchvision 0.27+cpu / transformers 5.11 / pycocotools, 2026-06 時点の安定版で動作確認）:
> Python 3.12 ／ torch 2.12.0+cpu ／ torchvision 0.27.0+cpu ／ transformers 5.11.0 ／ huggingface-hub 1.18.0 ／ timm 1.0.27 ／ torchmetrics 1.9.0 ／ pycocotools 2.0.11（第19回で使用）／ numpy 2.4.6 ／ Pillow 12.2.0 ／ matplotlib 3.10.9 ／ opencv-python-headless 4.13.0（合成画像の描画）
> 使用モデル: torchvision `ssdlite320_mobilenet_v3_large` / `retinanet_resnet50_fpn` / `fasterrcnn_resnet50_fpn`（COCO 重み）／ HF `facebook/detr-resnet-50`（DETR）。初回のみ重みを取得しキャッシュします。Ultralytics YOLO（`yolo11n`/`yolo26n`）は概念紹介のみで、依存衝突（opencv-python full ↔ headless）を避けるため実行経路には含みません。
> スクリプト構成: 本編 `01`〜`03` ＋ 共通 `detection_helpers.py` ＋ 章末 `mini_project.py`（検出パイプライン×自作mAP）＋ 演習 `exercises.py`（全10問）／`exercises_solutions.py`（模範解答）。すべて CPU・合成データで `exit 0`。