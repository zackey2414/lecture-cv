# 39_clip_distillation: VLM/CLIP の蒸留 — TinyCLIP/MobileCLIP・埋め込み模倣

> トラック: **最適化・デプロイ** ／ レベル: **上級** ／ 必要な依存グループ: `dl` `hf` `embed` `distill`
> 前提モジュール: `38_knowledge_distillation`（温度付き KD・特徴量蒸留）／ `16_clip_zeroshot_retrieval`（CLIP の正準フロー）

---

## 🎯 この章のゴール

大きな CLIP（teacher）の知識を、**桁違いに小さい画像エンコーダ（student）へ蒸留**する手法を、手を動かして理解する。具体的には次ができるようになることを目指す。

- teacher CLIP の**画像埋め込みを student に模倣させる蒸留**（埋め込み回帰: コサイン + MSE）を自前で書ける。
- teacher の**画像-テキスト類似度行列**を、埋め込みを **L2 正規化**し **`logit_scale`（温度）** を掛けて作り、それを student に **KL（親和性蒸留）** で模倣させられる。
- **正規化・温度を揃えないと類似度のスケールがずれて蒸留が壊れる**ことを、数値で説明できる。
- **TinyCLIP（重み継承 + 親和性蒸留）** と **MobileCLIP/MobileCLIP2（DataCompDR によるデータセット強化）** の考え方の違いを言える。
- `open_clip` で MobileCLIP/TinyCLIP を**CPU ロード**し、効率CLIP（MobileCLIP/TinyCLIP）の TorchScript 配布形まで接続できる。
- 蒸留結果を **ゼロショット精度・検索 Recall@k・teacher との埋め込みコサイン整合度（保持率）** で評価できる。

このモジュールのトイ実験では、**teacher（合計 1.51 億パラメータ、画像タワー 8785 万）→ student（約 16 万パラメータ、x551 圧縮）** という極端な圧縮でも、合成データ上で**ゼロショット精度を 100% 保持**し、teacher 埋め込みとのコサイン整合度 0.99 を達成する様子を観察する。

---


## 1. 直感 — 「答え」ではなく「埋め込み空間」を移す

38 章の知識蒸留では、teacher が出す**クラス確率（ソフトターゲット）**を student に真似させた。しかし、CLIP のような VLM（Vision-Language Model）には固定のクラスが無い。代わりに CLIP が持っているのは、**画像とテキストを同じ向きに並べた共有埋め込み空間**そのものである。したがって VLM 蒸留のゴールは「正解ラベルを当てる」ことではなく、**teacher が描いた埋め込み空間の幾何（どの画像がどのテキストの近くに来るか）を、小さな student へ丸ごと移植する**ことになる。

そして CLIP 蒸留の気持ちよさは、**ラベルが一切要らない**点にある。teacher にラベル無し画像をたくさん通して埋め込みを取り出せば、それがそのまま student の教師信号になるからだ。student は「この画像は埋め込み空間のこの座標に来るべきだ」とだけ教わり、teacher の世界観（赤い丸は赤い丸どうし近く、青い三角からは遠い…）を再現していく。その結果、student は **teacher のテキスト埋め込みを使ってゼロショット分類・検索ができる**ようになる——自分ではテキストを一度も見ていないのに、だ。

<figure class="lec-fig"><svg viewBox="0 0 660 300" role="img" aria-label="teacherの共有埋め込み空間にstudentが安い画像を同じ座標へ写像する図" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><rect x="170" y="48" width="468" height="218" rx="12" fill="#f4f4f5" stroke="#d4d4d8" stroke-width="1.5"/><text x="404" y="40" text-anchor="middle" font-size="15" font-weight="700" fill="#3f3f46">teacher の共有埋め込み空間（コサイン近傍）</text><circle cx="258" cy="108" r="9" fill="#dc2626"/><circle cx="282" cy="124" r="9" fill="#dc2626"/><circle cx="266" cy="135" r="9" fill="#dc2626"/><text x="270" y="92" text-anchor="middle" font-size="13" font-weight="700" fill="#dc2626">赤い丸</text><rect x="520" y="96" width="16" height="16" fill="#16a34a"/><rect x="542" y="108" width="16" height="16" fill="#16a34a"/><rect x="526" y="120" width="16" height="16" fill="#16a34a"/><text x="534" y="90" text-anchor="middle" font-size="13" font-weight="700" fill="#15803d">緑の四角</text><polygon points="430,196 421,212 439,212" fill="#2563eb"/><polygon points="452,208 443,224 461,224" fill="#2563eb"/><polygon points="414,220 405,236 423,236" fill="#2563eb"/><text x="436" y="252" text-anchor="middle" font-size="13" font-weight="700" fill="#2563eb">青い三角</text><circle cx="270" cy="118" r="15" fill="none" stroke="#ea580c" stroke-width="2.5"/><rect x="36" y="112" width="74" height="74" rx="6" fill="#ffffff" stroke="#d4d4d8" stroke-width="1.5"/><circle cx="73" cy="149" r="20" fill="#dc2626"/><text x="73" y="206" text-anchor="middle" font-size="12.5" fill="#52525b">student の安い画像</text><line x1="112" y1="146" x2="246" y2="122" stroke="#ea580c" stroke-width="2.4"/><polygon points="254,121 241,118 244,130" fill="#ea580c"/><text x="186" y="110" text-anchor="middle" font-size="12" font-weight="700" fill="#ea580c">同じ座標へ</text></svg><figcaption>teacher は画像とテキストを<b>同じ向きに並べた共有埋め込み空間</b>を持ち、意味が近いもの（赤い丸どうし）は近く、別物（青い三角）は遠くに配置されます。<b>student</b> は安い <b>64px</b> 画像から、teacher が定めた<b>正しい座標</b>（オレンジの輪）を予測するよう学びます。<b>ラベルは不要</b>で、teacher の埋め込みがそのまま教師信号になります。</figcaption></figure>

本章のトイ実験では、teacher として `openai/clip-vit-base-patch32`（224px 入力・ViT-B/32・埋め込み 512 次元）を使い、student として 64px 入力の小さな CNN（約 16 万パラメータ）を使う。student は「安い小さな画像」から「teacher の 512 次元埋め込み」を予測する関数を学ぶ。入力解像度もモデルサイズも段違いに小さいのに、埋め込み空間は再現できる——これが蒸留の威力だ。

<figure class="lec-fig"><svg viewBox="0 0 660 300" role="img" aria-label="teacher 224px CLIPとstudent 64px CNNがともに512次元埋め込みを出し蒸留する図" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><rect x="28" y="46" width="78" height="48" rx="6" fill="#eff6ff" stroke="#2563eb" stroke-width="1.8"/><text x="67" y="75" text-anchor="middle" font-size="14" font-weight="700" fill="#1d4ed8">224px</text><line x1="106" y1="70" x2="146" y2="70" stroke="#71717a" stroke-width="2"/><polygon points="152,70 142,65 142,75" fill="#71717a"/><rect x="152" y="40" width="158" height="60" rx="6" fill="#ffedd5" stroke="#c2410c" stroke-width="2"/><text x="231" y="66" text-anchor="middle" font-size="13.5" font-weight="700" fill="#c2410c">teacher CLIP<tspan x="231" dy="19" font-size="12" font-weight="400" fill="#52525b">8785万・凍結 ❄</tspan></text><line x1="310" y1="70" x2="350" y2="70" stroke="#71717a" stroke-width="2"/><polygon points="356,70 346,65 346,75" fill="#71717a"/><rect x="356" y="44" width="92" height="52" rx="6" fill="#dbeafe" stroke="#2563eb" stroke-width="1.8"/><text x="402" y="75" text-anchor="middle" font-size="14" font-weight="700" fill="#1d4ed8">512次元</text><rect x="28" y="206" width="78" height="48" rx="6" fill="#eff6ff" stroke="#2563eb" stroke-width="1.8"/><text x="67" y="235" text-anchor="middle" font-size="14" font-weight="700" fill="#1d4ed8">64px</text><line x1="106" y1="230" x2="146" y2="230" stroke="#71717a" stroke-width="2"/><polygon points="152,230 142,225 142,235" fill="#71717a"/><rect x="152" y="200" width="158" height="60" rx="6" fill="#fff7ed" stroke="#ea580c" stroke-width="2"/><text x="231" y="226" text-anchor="middle" font-size="13.5" font-weight="700" fill="#ea580c">student CNN<tspan x="231" dy="19" font-size="12" font-weight="400" fill="#52525b">16万・x551 小</tspan></text><line x1="310" y1="230" x2="350" y2="230" stroke="#71717a" stroke-width="2"/><polygon points="356,230 346,225 346,235" fill="#71717a"/><rect x="356" y="204" width="92" height="52" rx="6" fill="#dbeafe" stroke="#2563eb" stroke-width="1.8"/><text x="402" y="235" text-anchor="middle" font-size="14" font-weight="700" fill="#1d4ed8">512次元</text><line x1="448" y1="70" x2="496" y2="138" stroke="#16a34a" stroke-width="2"/><polygon points="500,143 489,137 492,148" fill="#16a34a"/><line x1="448" y1="230" x2="496" y2="162" stroke="#16a34a" stroke-width="2"/><polygon points="500,157 492,152 489,163" fill="#16a34a"/><rect x="500" y="118" width="140" height="64" rx="8" fill="#fafafa" stroke="#16a34a" stroke-width="2"/><text x="570" y="144" text-anchor="middle" font-size="13.5" font-weight="700" fill="#15803d">蒸留ロス<tspan x="570" dy="19" font-size="13" font-weight="400" fill="#3f3f46">(1−cos)+MSE</tspan></text></svg><figcaption>teacher（<b>224px</b> 入力・ViT-B/32・8785万パラメータ）を<b>凍結</b>し、その <b>512次元</b>埋め込みを前計算します。<b>student</b>（<b>64px</b> 入力・16万パラメータ、<b>x551 小</b>）が同じ512次元を出力するよう <code>(1−cos)+MSE</code> で学習し、<b>勾配は student だけ</b>に流れます。入力もモデルも段違いに小さいのに、埋め込み空間を再現できます。</figcaption></figure>

## 2. 理論 — L2 正規化・logit_scale・3 種類の蒸留信号

CLIP の類似度計算は必ず次の正準形を取る。画像埋め込み `i` とテキスト埋め込み `t` を**それぞれ L2 正規化**し（`F.normalize`）、内積を取り、学習で得た**温度パラメータ `logit_scale`**（の `exp`）を掛ける。

```
logits = logit_scale * normalize(i) @ normalize(t).T     # exp(logit_scale) は CLIP では約 100
probs  = softmax(logits, dim=class)                       # CLIP は softmax（相互排他クラス）
```

では、なぜ正規化が要るのか。`get_image_features` が返す射影後の生埋め込みは、ノルムが画像ごとにバラバラ（本章の teacher では約 10〜12）であり、正規化しないと内積が「ベクトルの長さ」に引きずられて意味的な近さを測れない。逆に正規化すれば、内積はそのまま**コサイン類似度**になり、全ペアを公平に比較できる。一方 `logit_scale`（≈100）は **softmax の鋭さを決める温度**であり、これが無いとコサインは [-1, 1] に収まったまま softmax がほぼ一様になり、「どれが正解か」という teacher の確信が消えてしまう。したがって、**蒸留では teacher と student で正規化と `logit_scale` を必ず揃える**——これがこの章で最も重要な約束だ。

<figure class="lec-fig"><svg viewBox="0 0 620 300" role="img" aria-label="CLIP類似度の正準フロー 生埋め込みからL2正規化コサインlogit_scale softmaxまで" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><rect x="28" y="46" width="150" height="66" rx="7" fill="#fff7ed" stroke="#ea580c" stroke-width="2"/><text x="103" y="74" text-anchor="middle" font-size="13.5" font-weight="700" fill="#c2410c">① 生埋め込み i,t<tspan x="103" dy="20" font-size="12" font-weight="400" fill="#52525b">ノルム 10〜12</tspan></text><line x1="178" y1="79" x2="210" y2="79" stroke="#71717a" stroke-width="2"/><polygon points="216,79 206,74 206,84" fill="#71717a"/><rect x="216" y="46" width="150" height="66" rx="7" fill="#dbeafe" stroke="#2563eb" stroke-width="2"/><text x="291" y="74" text-anchor="middle" font-size="13.5" font-weight="700" fill="#1d4ed8">② L2 正規化<tspan x="291" dy="20" font-size="12" font-weight="400" fill="#52525b">ノルム=1 単位球面</tspan></text><line x1="366" y1="79" x2="398" y2="79" stroke="#71717a" stroke-width="2"/><polygon points="404,79 394,74 394,84" fill="#71717a"/><rect x="404" y="46" width="158" height="66" rx="7" fill="#eff6ff" stroke="#2563eb" stroke-width="2"/><text x="483" y="74" text-anchor="middle" font-size="13.5" font-weight="700" fill="#1d4ed8">③ 内積=コサイン<tspan x="483" dy="20" font-size="12" font-weight="400" fill="#52525b">範囲 [-1, 1]</tspan></text><line x1="483" y1="112" x2="483" y2="150" stroke="#71717a" stroke-width="2"/><polygon points="483,156 478,146 488,146" fill="#71717a"/><rect x="404" y="156" width="158" height="66" rx="7" fill="#ffedd5" stroke="#c2410c" stroke-width="2"/><text x="483" y="184" text-anchor="middle" font-size="13.5" font-weight="700" fill="#c2410c">④ × logit_scale<tspan x="483" dy="20" font-size="12" font-weight="400" fill="#52525b">≈100（温度）</tspan></text><line x1="404" y1="189" x2="372" y2="189" stroke="#71717a" stroke-width="2"/><polygon points="366,189 376,184 376,194" fill="#71717a"/><rect x="216" y="156" width="150" height="66" rx="7" fill="#fafafa" stroke="#16a34a" stroke-width="2"/><text x="291" y="184" text-anchor="middle" font-size="13.5" font-weight="700" fill="#15803d">⑤ softmax<tspan x="291" dy="20" font-size="12" font-weight="400" fill="#52525b">鋭い確率</tspan></text><line x1="216" y1="189" x2="190" y2="189" stroke="#71717a" stroke-width="2"/><polygon points="184,189 194,184 194,194" fill="#71717a"/><rect x="44" y="200" width="22" height="22" fill="#c2410c"/><rect x="78" y="164" width="22" height="58" fill="#c2410c"/><rect x="112" y="206" width="22" height="16" fill="#c2410c"/><line x1="40" y1="222" x2="160" y2="222" stroke="#71717a" stroke-width="1.5"/><text x="100" y="150" text-anchor="middle" font-size="12.5" font-weight="700" fill="#c2410c">確信のある分布</text></svg><figcaption>CLIP の類似度の<b>正準フロー</b>です。①生埋め込みはノルムがバラバラ → ②<b>L2 正規化</b>で全ベクトルを単位長（単位球面）に揃えると内積がそのまま ③<b>コサイン類似度</b>になり → ④学習で得た <code>logit_scale</code>（≈100）を<b>温度</b>として掛け → ⑤<b>softmax</b> で鋭い確率になります。<b>蒸留では teacher と student でこの正規化と温度を必ず揃えます</b>。</figcaption></figure>

teacher から student へ移す信号には、大きく 3 つの粒度がある。**(a) 埋め込み回帰**: student の埋め込みを teacher の埋め込みそのものに `1 - cos`（＋補助の MSE）で寄せる。最も直接的で安定だ。**(b) 親和性（類似度行列）蒸留**: teacher の画像-テキスト類似度行列を softmax してソフトターゲットにし、student の行列を `KL` で合わせる。これは 38 章の Hinton 蒸留を「クラス確率」ではなく「画像×テキストの確率」に置き換えたもので、TinyCLIP が使う形である。**(c) 対照蒸留（contrastive distillation）**: バッチ内の画像-テキスト対応を InfoNCE で学びつつ teacher の類似度も模倣する、実データの大規模学習で使う形。本章は CPU トイ実験なので、(a) を主役に据え、(b) を `03` で実装し、(c) は概念として触れるにとどめる。

<figure class="lec-fig"><svg viewBox="0 0 640 300" role="img" aria-label="teacherからstudentへ移す3つの蒸留信号 埋め込み回帰 親和性蒸留 対照蒸留" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><text x="320" y="30" text-anchor="middle" font-size="15" font-weight="700" fill="#3f3f46">teacher → student に移す 3 つの信号</text><rect x="18" y="48" width="192" height="208" rx="10" fill="#fafafa" stroke="#d4d4d8" stroke-width="1.5"/><circle cx="148" cy="110" r="14" fill="#71717a"/><circle cx="78" cy="172" r="14" fill="#ea580c"/><line x1="89" y1="164" x2="134" y2="120" stroke="#ea580c" stroke-width="2.2" stroke-dasharray="5 3"/><polygon points="140,114 128,117 133,127" fill="#ea580c"/><text x="114" y="212" text-anchor="middle" font-size="13.5" font-weight="700" fill="#c2410c">(a) 埋め込み回帰<tspan x="114" dy="19" font-size="11.5" font-weight="400" fill="#52525b">(1−cos)+MSE・直接</tspan></text><rect x="224" y="48" width="192" height="208" rx="10" fill="#fafafa" stroke="#d4d4d8" stroke-width="1.5"/><rect x="246" y="92" width="17" height="17" fill="#1d4ed8" stroke="#ffffff"/><rect x="263" y="92" width="17" height="17" fill="#dbeafe" stroke="#ffffff"/><rect x="280" y="92" width="17" height="17" fill="#dbeafe" stroke="#ffffff"/><rect x="246" y="109" width="17" height="17" fill="#dbeafe" stroke="#ffffff"/><rect x="263" y="109" width="17" height="17" fill="#1d4ed8" stroke="#ffffff"/><rect x="280" y="109" width="17" height="17" fill="#dbeafe" stroke="#ffffff"/><rect x="246" y="126" width="17" height="17" fill="#dbeafe" stroke="#ffffff"/><rect x="263" y="126" width="17" height="17" fill="#dbeafe" stroke="#ffffff"/><rect x="280" y="126" width="17" height="17" fill="#1d4ed8" stroke="#ffffff"/><line x1="300" y1="117" x2="332" y2="117" stroke="#52525b" stroke-width="2"/><polygon points="338,117 328,112 328,122" fill="#52525b"/><rect x="338" y="92" width="17" height="17" fill="#ea580c" stroke="#ffffff"/><rect x="355" y="92" width="17" height="17" fill="#ffedd5" stroke="#ffffff"/><rect x="372" y="92" width="17" height="17" fill="#ffedd5" stroke="#ffffff"/><rect x="338" y="109" width="17" height="17" fill="#ffedd5" stroke="#ffffff"/><rect x="355" y="109" width="17" height="17" fill="#ea580c" stroke="#ffffff"/><rect x="372" y="109" width="17" height="17" fill="#ffedd5" stroke="#ffffff"/><rect x="338" y="126" width="17" height="17" fill="#ffedd5" stroke="#ffffff"/><rect x="355" y="126" width="17" height="17" fill="#ffedd5" stroke="#ffffff"/><rect x="372" y="126" width="17" height="17" fill="#ea580c" stroke="#ffffff"/><text x="320" y="212" text-anchor="middle" font-size="13.5" font-weight="700" fill="#1d4ed8">(b) 親和性蒸留 KL<tspan x="320" dy="19" font-size="11.5" font-weight="400" fill="#52525b">行列を模倣・TinyCLIP</tspan></text><rect x="430" y="48" width="192" height="208" rx="10" fill="#fafafa" stroke="#d4d4d8" stroke-width="1.5"/><rect x="494" y="86" width="16" height="16" fill="#ea580c" stroke="#d4d4d8"/><rect x="510" y="86" width="16" height="16" fill="#f4f4f5" stroke="#d4d4d8"/><rect x="526" y="86" width="16" height="16" fill="#f4f4f5" stroke="#d4d4d8"/><rect x="542" y="86" width="16" height="16" fill="#f4f4f5" stroke="#d4d4d8"/><rect x="494" y="102" width="16" height="16" fill="#f4f4f5" stroke="#d4d4d8"/><rect x="510" y="102" width="16" height="16" fill="#ea580c" stroke="#d4d4d8"/><rect x="526" y="102" width="16" height="16" fill="#f4f4f5" stroke="#d4d4d8"/><rect x="542" y="102" width="16" height="16" fill="#f4f4f5" stroke="#d4d4d8"/><rect x="494" y="118" width="16" height="16" fill="#f4f4f5" stroke="#d4d4d8"/><rect x="510" y="118" width="16" height="16" fill="#f4f4f5" stroke="#d4d4d8"/><rect x="526" y="118" width="16" height="16" fill="#ea580c" stroke="#d4d4d8"/><rect x="542" y="118" width="16" height="16" fill="#f4f4f5" stroke="#d4d4d8"/><rect x="494" y="134" width="16" height="16" fill="#f4f4f5" stroke="#d4d4d8"/><rect x="510" y="134" width="16" height="16" fill="#f4f4f5" stroke="#d4d4d8"/><rect x="526" y="134" width="16" height="16" fill="#f4f4f5" stroke="#d4d4d8"/><rect x="542" y="134" width="16" height="16" fill="#ea580c" stroke="#d4d4d8"/><text x="526" y="212" text-anchor="middle" font-size="13.5" font-weight="700" fill="#c2410c">(c) 対照蒸留<tspan x="526" dy="19" font-size="11.5" font-weight="400" fill="#52525b">InfoNCE・大規模</tspan></text></svg><figcaption>teacher の知識を student へ移す信号は、粒度で 3 種類あります。<b>(a) 埋め込み回帰</b>は student 埋め込みを teacher 埋め込みそのものに <code>(1−cos)+MSE</code> で寄せる、最も直接的で安定な方法。<b>(b) 親和性蒸留</b>は画像×テキストの<b>類似度行列</b>を softmax して <code>KL</code> で一致させます（TinyCLIP 型）。<b>(c) 対照蒸留</b>はバッチ内対応を InfoNCE で学びつつ teacher も模倣します（大規模実データ向け）。本章は <b>(a)</b> が主役です。</figcaption></figure>

## 3. 正準 API — transformers の teacher と open_clip の効率 CLIP

teacher 側（transformers v5）の正準フローは次の通り。`CLIPModel` をロードし、`get_image_features` / `get_text_features` で埋め込みを取り、**自分で `F.normalize` してから** `logit_scale.exp()` を掛ける。

```python
from transformers import CLIPModel, CLIPProcessor
import torch, torch.nn.functional as F

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval()
proc  = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

with torch.inference_mode():
    img = model.get_image_features(**proc(images=pil_list, return_tensors="pt")).pooler_output
    txt = model.get_text_features(**proc(text=prompts, return_tensors="pt", padding=True)).pooler_output
img = F.normalize(img, dim=-1)   # ← 未正規化(ノルム約10〜12)なので必須
txt = F.normalize(txt, dim=-1)
logits = model.logit_scale.exp() * img @ txt.T
```

> ⚠️ transformers v5 では `get_image_features` は**テンソルではなく出力オブジェクト**を返す。射影後の埋め込みは `.pooler_output`（[N, 512]・**未正規化**）にある。一方 `model(**inputs).image_embeds` は **L2 正規化済み**で返る。どちらを使うにせよ「正規化されているか」を必ず確認すること（本章のヘルパは `pooler_output` を取り出して明示的に `F.normalize` する）。

効率 CLIP 側（`open_clip` 3.3）の正準フローはこう。`create_model_and_transforms` で**アーキ + 前処理**を同時に得て、`encode_image` / `encode_text` で埋め込みを取り、`get_tokenizer` でテキストをトークン化する。事前学習キーは `list_pretrained()` で確認する。

```python
import open_clip, torch.nn.functional as F
print([(n, t) for n, t in open_clip.list_pretrained() if "MobileCLIP" in n or "TinyCLIP" in n])
model, _, preprocess = open_clip.create_model_and_transforms("MobileCLIP-S1", pretrained="datacompdr")
tokenizer = open_clip.get_tokenizer("MobileCLIP-S1")
model.eval()
with torch.inference_mode():
    img = F.normalize(model.encode_image(preprocess(pil).unsqueeze(0)), dim=-1)
    txt = F.normalize(model.encode_text(tokenizer(["a dog", "a cat"])), dim=-1)
```

## 4. 実装をひとつずつ

**`01_teacher_similarity_matrix.py` — 正準フローの確認。** まず teacher だけで「encode → L2 正規化 → `logit_scale` → 類似度行列」を完走する。生埋め込みのノルムがバラバラ（約 10〜12）で、正規化すると全部 1.0 になることを数値で見せ、画像-テキスト確率行列をヒートマップに保存する。合成 3 クラス（赤い丸・緑の四角・青い三角）で teacher のゼロショット精度は 1.000 になる。

**`02_embedding_distillation.py` — この章の中心。** teacher の画像埋め込みを**一度だけ前計算**してキャッシュし（teacher は `eval()` + `inference_mode()` で凍結）、student を `(1 - cos) + MSE` で学習する。学習前は student と teacher の整合度が約 0.03（ランダム）なのが、約 40 エポック（CPU で数秒）で **0.99** まで上がる。teacher 画像タワー 8785 万 → student 約 16 万（**x551 圧縮**）。学習後の重みは `outputs/39_clip_distillation/student_distilled.pt` に保存される。

**`03_similarity_distillation_and_pitfall.py` — 親和性蒸留と落とし穴。** teacher の画像-テキスト類似度行列を softmax したソフトターゲットを、student の行列に `KL(log_softmax(student) ‖ softmax(teacher))`（`reduction='batchmean'`）で合わせる（TinyCLIP 型）。同時に**スケールずれの落とし穴**を数値で再現する: 正規化を忘れると行列がノルムに引きずられて暴走し、`logit_scale` を忘れると softmax の最大確率が 0.996 → 0.35 へと平坦化して teacher の確信が消える。

**`04_open_clip_mobileclip.py` — 効率 CLIP と参考実装接続。** `list_pretrained()` で MobileCLIP/TinyCLIP のキーを列挙し、アーキを**オフラインで（`pretrained=None`）構築**してパラメータ数と CPU レイテンシを比較する。事前学習重みのロードは `try/except` でガードし、ネットが無ければ概念に切り替える。最後に効率CLIP の TorchScript 配布形（例: MobileCLIP の `.ts`、約 600MB）に触れ、実運用での `load_clip_model()` 相当（`force_quick_gelu=True` で効率 CLIP を読み、CenterCrop を排した独自前処理で dense 特徴を取り出す）と、本章の自前 student が**「L2 正規化 + `logit_scale` で温度付与」という同じ約束を共有**していることを確認する。

## 5. 落とし穴（このモジュールで必ず踏む）

最頻の事故は**スケールずれ**だ。teacher と student で「正規化したか」「`logit_scale` を掛けたか」が食い違うと、同じ画像でも類似度の数値が桁で変わり、KL も MSE も意味をなさない。`03` の出力（正しい行列は範囲 [20, 33]、正規化忘れは [24, 40]（約 [24.5, 40.1]）で正しい行列を 1.21 倍した範囲になり必ず大きくなる、`logit_scale` 忘れは [0.2, 0.3]）を必ず自分の目で見ておくこと。次に多いのが **teacher を凍結し忘れる**事故で、`eval()` を呼ばないと BatchNorm/Dropout が動いて教師信号が毎回揺れ、optimizer に teacher のパラメータを渡すと誤って teacher 側を更新してしまう。

<figure class="lec-fig"><svg viewBox="0 0 640 300" role="img" aria-label="類似度行列の値レンジ 正しい 正規化忘れ logit_scale忘れ の比較" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><text x="320" y="30" text-anchor="middle" font-size="15" font-weight="700" fill="#18181b">類似度行列の値レンジ（スケールずれの結末）</text><line x1="175" y1="46" x2="175" y2="190" stroke="#e4e4e7"/><line x1="380" y1="46" x2="380" y2="190" stroke="#e4e4e7"/><line x1="585" y1="46" x2="585" y2="190" stroke="#e4e4e7"/><rect x="380" y="58" width="133" height="26" rx="4" fill="#16a34a"/><text x="20" y="68" font-size="12" font-weight="700" fill="#15803d">正しい<tspan x="20" dy="16" font-weight="400" fill="#52525b">[20, 33]</tspan></text><rect x="421" y="106" width="164" height="26" rx="4" fill="#dc2626"/><text x="20" y="116" font-size="12" font-weight="700" fill="#dc2626">正規化 忘れ<tspan x="20" dy="16" font-weight="400" fill="#52525b">[24, 40]・1.21倍</tspan></text><rect x="175" y="154" width="10" height="26" rx="2" fill="#dc2626"/><text x="20" y="164" font-size="12" font-weight="700" fill="#dc2626">logit_scale 忘れ<tspan x="20" dy="16" font-weight="400" fill="#52525b">[0.2, 0.3] ≈ 0</tspan></text><line x1="175" y1="190" x2="585" y2="190" stroke="#71717a" stroke-width="1.5"/><text x="175" y="208" text-anchor="middle" font-size="12" fill="#52525b">0</text><text x="585" y="208" text-anchor="middle" font-size="12" fill="#52525b">40</text><text x="320" y="244" text-anchor="middle" font-size="13" font-weight="700" fill="#c2410c">softmax 最大確率 0.996 → 0.35 へ平坦化（teacher の確信が消える）</text></svg><figcaption>同じ teacher 類似度でも、<b>正規化</b>と <code>logit_scale</code> を揃えないと行列の値レンジが激変します。正しくは <b>[20, 33]</b>、<b>正規化を忘れる</b>とノルムに引きずられて <b>[24, 40]</b>（約 1.21 倍）に膨張し、<b>logit_scale を忘れる</b>と <b>[0.2, 0.3]</b> に潰れて softmax が平坦化（最大確率 <b>0.996 → 0.35</b>）し teacher の確信が消えます。蒸留では teacher と student で<b>両方を必ず揃えます</b>。</figcaption></figure>

もうひとつは **MobileCLIP は x86 CPU では必ずしも速くない**という現実だ。本章の計測では、ランダム初期化の MobileCLIP-S1（FastViT 系、8497 万パラメータ）の画像エンコードは約 47ms で、teacher の ViT-B/32（約 16ms）より**遅い**。MobileCLIP の真価はモバイル GPU/ANE での再パラメータ化（reparameterization）にあり、PyTorch eager の x86 CPU ではその利得が出ない。「軽量 ＝ どの環境でも速い」ではなく、**どのハードウェアで速いのか**を必ず確認する。

最後は**用語の混同**である。まず「**知識蒸留（model distillation）**」は teacher→student へ知識を移すことを指す。これに対し「**dataset distillation**」は、学習データを少数の合成画像へ凝縮する全く別の研究だ。さらに MobileCLIP の「**dataset reinforcement（DataCompDR）**」は、teacher の埋め込みや合成キャプションを**事前計算してデータセットに焼き込む**ことで小モデルを強くする手法であり、これも別物である。MobileCLIP の論文の主張が「アーキより**データ（強化）が効く**」である点を押さえておきたい。

## 6. 実務の使い分け

**まず既製の蒸留済みモデルを試す。** 自分で蒸留する前に、`open_clip` の MobileCLIP-S0/S1/S2 や TinyCLIP が要件を満たさないか確認する。これらは大規模データで蒸留済みで、ゼロから蒸留するより圧倒的に強い。`list_pretrained()` でキーを確認し、CPU/エッジのレイテンシ要件に合うものを選ぶ。

**自前蒸留が要るのは「ドメイン特化」と「極端な小型化」のとき。** 自社の画像分布（製造ライン・医用・特定の商品画像など）でしか使わないなら、その分布のラベル無し画像で teacher 埋め込みを取り、小さな student に蒸留すると、汎用効率 CLIP より小さく・速く・そのドメインで十分な精度のモデルが作れる。本章の `02`/`mini_project` がそのレシピの最小形だ。**埋め込み回帰（a）から始め**、必要なら親和性蒸留（b）を足す、という順序が安全。

**評価は必ず 3 軸で。** ①ゼロショット精度（teacher との**保持率** = student_acc / teacher_acc）、②検索 Recall@k、③teacher 埋め込みとの**コサイン整合度**。①が落ちているのに③が高い、あるいはその逆、という乖離が出たら、正規化/`logit_scale`/テキスト埋め込みの不一致をまず疑う。デプロイ段（36/37 章）の ONNX 化・量子化と組み合わせるのが定石だ。

---

## 🛠 章末ミニプロジェクト（`mini_project.py`）

**課題:** teacher CLIP を x551 小さい student CNN に埋め込み蒸留し、**ゼロショット精度・検索 Recall@k・teacher 整合度の「保持率」**を一気通貫で評価せよ。蒸留前（ランダム student）との比較表と、棒グラフ `outputs/39_clip_distillation/mini_project_summary.png` を出力する。

完成形を実行すると次のような表が出る（CPU・合成データ・約数十秒）。

```
================ 評価テーブル ================
metric                   teacher   before KD    after KD
zero-shot acc              1.000       0.333       1.000
retrieval R@1              1.000       1.000       1.000
retrieval R@5              1.000       1.000       1.000
cosine align               1.000       0.033       0.986
agreement w/ teacher           -       0.333       1.000
---------------------------------------------
zero-shot 保持率 (student/teacher) = 100.0%
パラメータ圧縮: teacher画像タワー 87.8M -> student 160K (x551)
```

> 読み解きの勘所: **蒸留前でも検索 R@1/R@5 が 1.0** なのは、ランダム CNN でも「色」が支配的なので同一クラス画像どうしは互いに近いから。一方**ゼロショット精度と整合度は蒸留前は壊滅**（0.33 / 0.03）している——これは「画像どうしの相対的な近さ」と「teacher のテキスト空間との絶対的な整合」は別物で、後者こそ蒸留で獲得する信号だ、という重要な教訓だ。

---

## ✅ 到達チェックリスト

- [ ] `get_image_features().pooler_output` は**未正規化**だと知っており、比較前に `F.normalize` を掛けられる。
- [ ] `logits = logit_scale * normalize(i) @ normalize(t).T` の正準形を空で書ける。
- [ ] 埋め込み回帰蒸留 `(1 - cos) + MSE` を実装し、teacher を `eval()`+`inference_mode()` で凍結して前計算できる。
- [ ] 親和性蒸留の KL を `KL(log_softmax(student) ‖ softmax(teacher), batchmean)` の**正しい向き**で書ける。
- [ ] 正規化／`logit_scale` を揃えないとスケールがずれ softmax が平坦化することを数値で説明できる。
- [ ] TinyCLIP（重み継承＋親和性蒸留）と MobileCLIP（DataCompDR データセット強化）の違いを言える。
- [ ] `open_clip.list_pretrained()` で効率 CLIP のキーを確認し、CPU でロード（または概念代替）できる。
- [ ] ゼロショット精度・Recall@k・コサイン整合度の 3 軸で**保持率**を評価できる。
- [ ] 「知識蒸留／dataset distillation／dataset reinforcement」を取り違えない。

---

## ❓ 落とし穴・FAQ・デバッグ

**Q. 蒸留しても student のゼロショット精度が上がらない。**
A. まず正規化と `logit_scale` を teacher と完全に揃えているか確認。student 埋め込みを正規化し忘れている、`logit_scale` を student 側で掛けていない、teacher テキスト埋め込みを正規化していない、のいずれかが定番。`03` の `demo_scale_mismatch` を自分のコードに当てて行列のレンジを見る。

**Q. `get_image_features(...)` に `.shape` が無いと言われる。**
A. transformers v5 では戻り値が `BaseModelOutputWithPooling`。埋め込みは `.pooler_output` にある（[N, 512]、**未正規化**）。`model(**inputs).image_embeds` を使えば正規化済みで返る。

**Q. teacher の損失に勾配が流れて teacher が更新される / メモリが膨らむ。**
A. teacher は `model.eval()` の上で `torch.inference_mode()`（または `no_grad`）の中で呼ぶ。optimizer には **student のパラメータだけ**渡す（`AdamW(student.parameters())`）。埋め込みは一度前計算してキャッシュする。

**Q. MobileCLIP をロードしたら CPU で teacher より遅い。**
A. 仕様。MobileCLIP/FastViT はモバイル GPU/ANE 向けの再パラメータ化で速くなる設計で、x86 CPU の PyTorch eager では利得が出ない。CPU 最速を狙うなら、本章の小型 student のように**入力解像度とパラメータを両方小さくする**か、36/37 章の ONNX/量子化を併用する。

**Q. `open_clip.create_model('MobileCLIP-S1', pretrained='datacompdr')` がネット無しで失敗する。**
A. 重み（数百 MB）の DL が要る。オフラインなら `pretrained=None` でアーキだけ構築してパラメータ数・API を確認する（`04` はこの分岐を `try/except` で持つ）。重み未取得時は概念（TinyCLIP=重み継承＋親和性蒸留／MobileCLIP=DataCompDR）で代替する。

**Q. KL が `nan` になる / 学習が発散する。**
A. `F.kl_div` の**入力に `log_softmax`、ターゲットに `softmax`** を渡しているか（逆は壊れる）。`reduction='batchmean'`（既定の `'mean'` は要素数で割り数式とずれる）。`logit_scale` が大き過ぎる行列に追加の温度を二重掛けしていないかも確認。

**Q. student の入力前処理は teacher と同じにすべき？**
A. 必須ではない。student はもっと安い入力（本章は 64px・[-1,1] 正規化）でよく、**出力（埋め込み）を teacher に合わせる**のが蒸留。入力を小さくできることが student を速くする源泉でもある。

---

## 🚀 発展トピック・参考

- **TinyCLIP（ICCV 2023）**: 親和性蒸留（affinity mimicking）＋**重み継承（weight inheritance）**＋多段縮約で、CLIP を大幅小型化しつつゼロショットを保つ。本章 `03` の KL 蒸留がその中核。
- **MobileCLIP / MobileCLIP2（CVPR 2024 ほか）**: **DataCompDR** による dataset reinforcement（teacher のアンサンブル埋め込みと合成キャプションを事前計算しデータに焼き込む）で、アーキ更新より**データで**効率と精度を稼ぐ。`open_clip` 3.x が `MobileCLIP-S0/S1/S2/B` と `MobileCLIP2-*` を同梱。
- **対照蒸留（contrastive distillation）**: 実データ大規模学習では InfoNCE と teacher 模倣を併用する。本章 (a)/(b) の延長。
- **次の一手**: 蒸留した student を `36_onnx_runtime`/`37_runtime_edge_optimization` で ONNX 化・動的量子化し、`17_faiss_image_search`/`42_multimodal_vector_search` の検索基盤に載せると、エッジで動く CLIP 検索が完成する。効率CLIP の TorchScript 配布形（例: MobileCLIP）はまさにその実運用形。
- ドキュメント: open_clip <https://github.com/mlfoundations/open_clip> ／ transformers CLIP <https://huggingface.co/docs/transformers/model_doc/clip> ／ Apple ml-mobileclip <https://github.com/apple/ml-mobileclip>

---

## ▶ 動かし方

```bash
# 依存（CPU 前提。Linux は torch CPU ホイールが pyproject の index 経由で入る）
uv sync --group dl --group hf --group embed
# ※ distill グループは本章では自前実装のため必須ではない（torchdistill 等は発展用）

# 1) teacher の正準フロー（encode -> normalize -> logit_scale -> 類似度行列）
uv run python lectures/39_clip_distillation/01_teacher_similarity_matrix.py
# 2) 埋め込み回帰蒸留（この章の中心）
uv run python lectures/39_clip_distillation/02_embedding_distillation.py
# 3) 親和性(KL)蒸留 + スケールずれの落とし穴
uv run python lectures/39_clip_distillation/03_similarity_distillation_and_pitfall.py
# 4) open_clip で MobileCLIP/TinyCLIP・参考実装接続
uv run python lectures/39_clip_distillation/04_open_clip_mobileclip.py
# 章末ミニプロジェクト（保持率の総合評価）
uv run python lectures/39_clip_distillation/mini_project.py
# 演習（自己採点）と模範解答
uv run python lectures/39_clip_distillation/exercises.py
uv run python lectures/39_clip_distillation/exercises_solutions.py
```

図と重みは `outputs/39_clip_distillation/` に保存される（matplotlib は Agg、`imshow` は使わない）。初回のみ teacher CLIP の重みを HuggingFace から取得する（以降キャッシュ）。

---

> 版: **torch 2.12+cpu / torchvision 0.27+cpu / transformers 5.11 / open_clip 3.3**（2026-06）
> CPU 前提・`model.eval()` + `torch.inference_mode()`・学習は合成データの数十秒トイ。
> 前提: 38_knowledge_distillation（温度付き KD）／ 16_clip_zeroshot_retrieval（CLIP 正準フロー）。
