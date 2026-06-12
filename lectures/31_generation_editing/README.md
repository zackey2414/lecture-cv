# 31 画像生成・編集 — 拡散モデル text-to-image・img2img・インペイント・超解像・背景除去・評価

> トラック: **生成・編集** ／ レベル: **上級** ／ 前提: `13_classification_transfer_learning`
> 必要な依存グループ: `dl` `hf` `diffusion` `metrics`（`classical` 相当の OpenCV は main 依存）

---

## 🎯 この章のゴール

これまでの章は「画像を**理解する**（分類・検出・セグメ・埋め込み）」側に立っていました。本章では、その
理解を **生成・編集する**側へと拡張します。具体的には、まず拡散モデル **SD-Turbo** で text-to-image / img2img を
**CPU で数秒**で動かし、続いて**インペイント**（傷消し・物体除去）と**超解像**（古典補間 → Swin2SR）、
**背景除去**（GrabCut）を実装します。そして最後に **PSNR / SSIM / CLIPScore / FID** で生成・復元の品質を
「正しい向きで」評価できるようになります。

到達点は次の 5 つです。

- 拡散モデルの直感（ノイズ→反復デノイズ）と、なぜ **SD-Turbo は 1〜2 ステップ・guidance 0.0** で
  生成できるのか（蒸留）を説明でき、`AutoPipelineForText2Image` を再現性つきで使える。
- `AutoPipelineForImage2Image` で **strength と step の関係**（`round(steps*strength) ≥ 1`）を理解し、
  入力画像を保ちながらプロンプトで編集できる。
- **インペイント**を古典 `cv2.inpaint`(Telea/NS) で実装し、参照あり指標で復元度を測れる。
  大型の拡散インペイント／LaMa は「概念＋任意（ガード）」として位置づけられる。
- **超解像**を「古典補間（bicubic/Lanczos）↔ 深層（Swin2SR）」で定量比較でき、**背景除去**を
  GrabCut で行って別背景に合成できる。
- **評価指標**を自前実装（PSNR/SSIM）と CLIPScore（CLIP）で計算し、FID/KID/IS/LPIPS の
  「良し悪しの向き」と「必要サンプル数」を取り違えない。

> すべて **CPU・合成データ完結**で動きます（ネットに出るのはモデル重みの初回 DL のみ）。重みが
> 取れない環境でも、合成画像＋古典手法へフォールバックして **必ず exit 0** で終わるよう設計しています。

---

## 1. 拡散モデルと text-to-image（直感 → 理論 → 正準 API → 実装）

### 直感
拡散モデル（diffusion model）は、画像に少しずつガウシアンノイズを足して完全な砂嵐にする「前向き過程」を
学習で**逆再生**する生成器です。生成時はランダムノイズから出発し、「この絵にはまだどれだけノイズが
乗っているか」を予測しては引く、という**デノイズを反復**して画像へ近づけます。そして、テキストによる条件づけは、
各デノイズステップに「赤いりんご」などの意味ベクトル（CLIP テキスト埋め込み）を注入することで効きます。

<figure class="lec-fig"><svg viewBox="0 0 640 300" role="img" aria-label="拡散モデルは前向き過程で画像にノイズを足し、逆向きのデノイズ反復で純ノイズから画像を生成する" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><text x="340" y="62" text-anchor="middle" font-size="14" fill="#3f3f46">前向き過程：少しずつノイズを足す（学習）</text><line x1="130" y1="80" x2="548" y2="80" stroke="#c2410c" stroke-width="2.5"/><polygon points="558,80 546,74 546,86" fill="#c2410c"/><rect x="120" y="104" width="100" height="100" fill="#ffffff" stroke="#d4d4d8" stroke-width="1.5"/><ellipse cx="170" cy="126" rx="9" ry="4" fill="#16a34a"/><circle cx="170" cy="158" r="30" fill="#ea580c"/><rect x="290" y="104" width="100" height="100" fill="#fafafa" stroke="#d4d4d8" stroke-width="1.5"/><circle cx="340" cy="158" r="30" fill="#f97316" opacity="0.45"/><circle cx="312" cy="124" r="3.2" fill="#71717a"/><circle cx="360" cy="118" r="3.2" fill="#71717a"/><circle cx="330" cy="140" r="3.2" fill="#71717a"/><circle cx="372" cy="150" r="3.2" fill="#71717a"/><circle cx="305" cy="166" r="3.2" fill="#71717a"/><circle cx="350" cy="178" r="3.2" fill="#71717a"/><circle cx="378" cy="186" r="3.2" fill="#71717a"/><circle cx="322" cy="190" r="3.2" fill="#71717a"/><rect x="460" y="104" width="100" height="100" fill="#f4f4f5" stroke="#d4d4d8" stroke-width="1.5"/><circle cx="476" cy="120" r="3.2" fill="#71717a"/><circle cx="500" cy="114" r="3.2" fill="#71717a"/><circle cx="524" cy="124" r="3.2" fill="#71717a"/><circle cx="548" cy="118" r="3.2" fill="#71717a"/><circle cx="468" cy="138" r="3.2" fill="#71717a"/><circle cx="492" cy="146" r="3.2" fill="#71717a"/><circle cx="516" cy="140" r="3.2" fill="#71717a"/><circle cx="540" cy="150" r="3.2" fill="#71717a"/><circle cx="478" cy="164" r="3.2" fill="#71717a"/><circle cx="508" cy="170" r="3.2" fill="#71717a"/><circle cx="532" cy="166" r="3.2" fill="#71717a"/><circle cx="552" cy="180" r="3.2" fill="#71717a"/><circle cx="488" cy="186" r="3.2" fill="#71717a"/><circle cx="522" cy="190" r="3.2" fill="#71717a"/><text x="170" y="224" text-anchor="middle" font-size="12.5" fill="#3f3f46">x0：きれいな画像</text><text x="510" y="224" text-anchor="middle" font-size="12.5" fill="#3f3f46">xT：純ノイズ</text><line x1="550" y1="244" x2="132" y2="244" stroke="#1d4ed8" stroke-width="2.5"/><polygon points="122,244 134,238 134,250" fill="#1d4ed8"/><text x="340" y="270" text-anchor="middle" font-size="14" fill="#1d4ed8">生成：デノイズを反復（ノイズ → 画像）</text><text x="340" y="290" text-anchor="middle" font-size="12.5" fill="#52525b">通常SD 20〜50 回 ／ SD-Turbo 1〜2 回（蒸留）</text></svg><figcaption>拡散モデルは、画像に<b>少しずつガウシアンノイズを足す前向き過程</b>を学習し、それを<b>逆再生</b>する生成器です。生成時は<b>純ノイズから出発</b>し、「まだ乗っているノイズ」を予測して引く<b>デノイズを反復</b>して画像へ近づけます。各ステップに<b>CLIP テキスト埋め込み</b>を注入するとプロンプトが効きます。<b>SD-Turbo</b> はこの 20〜50 回の反復を<b>蒸留</b>で <code>1〜2</code> ステップへ圧縮しています。</figcaption></figure>

通常の Stable Diffusion はこの反復を 20〜50 回まわすため、CPU では 1 枚に数分かかります。これに対して本章で使う
**SD-Turbo** は **Adversarial Diffusion Distillation** で「多ステップの結果を 1〜2 ステップで再現する」よう
蒸留されており、**`num_inference_steps=1〜2`・`guidance_scale=0.0`** という設定で CPU でも 1 枚 1 秒前後で
生成できます。ここを外すと（特に `guidance_scale>0`）絵が崩れるので、Turbo 系の作法として固定で覚えます。

### 理論（最小限）
学習されるネットワーク（UNet）は、ノイズ付き画像 x_t と時刻 t から「乗っているノイズ ε」を回帰します。
classifier-free guidance（CFG）は「条件あり予測と条件なし予測の差」を `guidance_scale` 倍に増幅して
プロンプトへの忠実度を上げる技法ですが、**蒸留モデル SD-Turbo は CFG を内部に織り込み済み**のため、
外から `guidance_scale` をかけると二重適用になって破綻します。したがって Turbo では `0.0` が正解です。
再現性は初期ノイズの乱数で決まるので、`torch.manual_seed(seed)` で `generator` を固定すれば**同じ seed →
ビット一致の画像**が得られます（実装で確認します）。

### 正準 API
```python
import torch
from diffusers import AutoPipelineForText2Image

pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sd-turbo", torch_dtype=torch.float32)
pipe.to("cpu")
pipe.enable_attention_slicing()                 # メモリ削減（CPU/小VRAM）
image = pipe(
    prompt="a photo of a red apple on a wooden table",
    num_inference_steps=1, guidance_scale=0.0,  # ★ SD-Turbo の必須設定
    height=256, width=256,
    generator=torch.manual_seed(0),             # 再現性
).images[0]
```

### 実装（このリポジトリ）
`01_sd_turbo_t2i.py` が上記を体系化します。2 つのプロンプトを 256px・1step で生成し（各 1 秒前後）、
`torch.manual_seed` の再現性（同 seed でビット一致／別 seed で平均画素差）と、steps=1 vs 2 の差を測ります。
`enable_model_cpu_offload()` は GPU↔CPU 退避用なので **CPU のみ環境では使いません**（`enable_attention_slicing()`
で十分）。また `float32` を明示するのも要点で、CPU では `float16` は遅く、未対応 op もあって不安定だからです。

### 落とし穴 / 実務の使い分け
`guidance_scale` を上げてはいけない、サイズを大きくしすぎない（512 以上は CPU で急に重くなる）、
プロンプトを増やしすぎない、の 3 点が CPU 教材での鉄則です。品質最優先なら SDXL / FLUX などの大型を
GPU で多ステップ、レイテンシ最優先（プレビュー・大量生成）なら SD-Turbo / LCM / SDXL-Turbo を少ステップ、
という住み分けになります。

---

## 2. img2img（入力画像 × プロンプトで編集）

### 直感と理論
img2img は「ノイズ 100% から」ではなく「**入力画像に途中までノイズを足した状態から**」デノイズを始める
編集モードです。どれだけノイズを足すかを決めるのが **`strength`** です（0=元画像のまま、1=ほぼ作り直し）。SD-Turbo の
img2img では、**実際に走るステップ数 = `round(num_inference_steps * strength)`** であり、これが **1 未満だと
エラー**になります。つまり `strength=0.5` なら `num_inference_steps` を 2 以上にしないと動きません。
ここが最頻のつまずきどころです。

<figure class="lec-fig"><svg viewBox="0 0 640 240" role="img" aria-label="img2imgのstrengthは注入ノイズ量で、実効ステップはround(steps×strength)。0は元画像、1は作り直し" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><text x="320" y="34" text-anchor="middle" font-size="15" font-weight="700" fill="#18181b">実効ステップ ＝ round(steps × strength)、1 未満はエラー</text><text x="320" y="56" text-anchor="middle" font-size="12.5" fill="#52525b">例：steps ＝ 4 に固定して strength を動かす</text><polygon points="92,118 548,118 548,82" fill="#ffedd5" stroke="#f97316" stroke-width="1.5"/><line x1="92" y1="122" x2="548" y2="122" stroke="#3f3f46" stroke-width="2"/><line x1="92" y1="116" x2="92" y2="128" stroke="#3f3f46" stroke-width="1.5"/><line x1="320" y1="116" x2="320" y2="128" stroke="#3f3f46" stroke-width="1.5"/><line x1="434" y1="116" x2="434" y2="128" stroke="#3f3f46" stroke-width="1.5"/><line x1="548" y1="116" x2="548" y2="128" stroke="#3f3f46" stroke-width="1.5"/><text x="74" y="146" text-anchor="start" font-size="12.5" fill="#3f3f46">0 ＝ 元画像のまま</text><text x="556" y="146" text-anchor="end" font-size="12.5" fill="#3f3f46">1 ＝ 作り直し</text><line x1="320" y1="128" x2="320" y2="166" stroke="#2563eb" stroke-width="1.2"/><line x1="434" y1="128" x2="434" y2="166" stroke="#2563eb" stroke-width="1.2"/><line x1="548" y1="128" x2="548" y2="166" stroke="#2563eb" stroke-width="1.2"/><rect x="274" y="166" width="92" height="30" rx="7" fill="#eff6ff" stroke="#2563eb" stroke-width="1.5"/><text x="320" y="186" text-anchor="middle" font-size="13" fill="#1d4ed8">0.5 → 2 step</text><rect x="388" y="166" width="92" height="30" rx="7" fill="#eff6ff" stroke="#2563eb" stroke-width="1.5"/><text x="434" y="186" text-anchor="middle" font-size="13" fill="#1d4ed8">0.75 → 3 step</text><rect x="502" y="166" width="92" height="30" rx="7" fill="#eff6ff" stroke="#2563eb" stroke-width="1.5"/><text x="548" y="186" text-anchor="middle" font-size="13" fill="#1d4ed8">1.0 → 4 step</text></svg><figcaption>img2img は入力画像に<b>途中までノイズを足した状態</b>からデノイズを始める編集です。<b>strength</b> が<b>注入するノイズ量</b>で、<code>0</code> は元画像のまま、<code>1</code> はほぼ作り直しになります。実際に走る回数は <b>round(steps × strength)</b> で、これが <code>1</code> 未満だとエラーです。図は <code>steps=4</code> 固定の例で、strength <code>0.5 / 0.75 / 1.0</code> が <b>2 / 3 / 4 step</b> に対応します。</figcaption></figure>

### 正準 API と実装
```python
from diffusers import AutoPipelineForImage2Image
pipe = AutoPipelineForImage2Image.from_pretrained("stabilityai/sd-turbo", torch_dtype=torch.float32).to("cpu")
out = pipe(prompt="an oil painting ...", image=init_pil,
           num_inference_steps=4, strength=0.5, guidance_scale=0.0).images[0]  # eff steps = 2
```
`02_img2img_inpaint.py` の前半が、`steps=4` 固定で `strength=0.5/0.75/1.0`（effective steps=2/3/4）を並べ、
「strength が大きいほど元画像から離れてプロンプト寄りになる」ことを 1 枚に可視化します。実務では
「写真の画風変換」「ラフ→清書」「軽微なリタッチ」に向き、構図を保ちたいなら strength を小さく、
大胆に変えたいなら大きく、と調整します。

---

## 3. インペイント（部分編集・物体除去）

### 直感
インペイントは「**マスクで指定した領域だけを描き直す**」編集で、傷消し・透かし除去・不要物の除去が
典型用途です。アプローチは大きく 2 つあります。1 つは古典の **`cv2.inpaint`** で、マスク境界の色・勾配を内側へ
**伝播**させて埋めます（Telea 法＝高速な距離重み伝播、Navier-Stokes 法＝流体方程式に倣った伝播）。
もう 1 つは深層の拡散インペイント／LaMa で、マスク内を「周囲＋プロンプトに整合する**新しい内容**」で生成します。

<figure class="lec-fig"><svg viewBox="0 0 640 290" role="img" aria-label="インペイントは傷あり画像とマスク(白=埋める)からマスク領域だけを描き直す。古典は伝播、深層は生成" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><rect x="44" y="70" width="92" height="92" fill="#ffffff" stroke="#d4d4d8" stroke-width="1.5"/><line x1="58" y1="86" x2="124" y2="150" stroke="#dc2626" stroke-width="4"/><circle cx="72" cy="132" r="14" fill="#16a34a"/><text x="152" y="124" text-anchor="middle" font-size="22" fill="#52525b">＋</text><rect x="170" y="70" width="92" height="92" fill="#18181b" stroke="#3f3f46" stroke-width="1.5"/><line x1="184" y1="86" x2="250" y2="150" stroke="#ffffff" stroke-width="5"/><line x1="272" y1="116" x2="306" y2="116" stroke="#52525b" stroke-width="2"/><polygon points="314,116 304,111 304,121" fill="#52525b"/><rect x="318" y="94" width="104" height="44" rx="9" fill="#fff7ed" stroke="#ea580c" stroke-width="2"/><text x="370" y="121" text-anchor="middle" font-size="15" font-weight="700" fill="#c2410c">inpaint</text><line x1="430" y1="116" x2="462" y2="116" stroke="#52525b" stroke-width="2"/><polygon points="470,116 460,111 460,121" fill="#52525b"/><rect x="474" y="70" width="92" height="92" fill="#ffffff" stroke="#16a34a" stroke-width="2"/><circle cx="502" cy="132" r="14" fill="#16a34a"/><text x="90" y="182" text-anchor="middle" font-size="13" fill="#3f3f46">傷あり画像</text><text x="216" y="182" text-anchor="middle" font-size="13" fill="#3f3f46">マスク（白＝埋める）</text><text x="520" y="182" text-anchor="middle" font-size="13" fill="#3f3f46">復元結果</text><rect x="44" y="212" width="270" height="46" rx="10" fill="#eff6ff" stroke="#2563eb" stroke-width="1.8"/><text x="179" y="240" text-anchor="middle" font-size="13" font-weight="700" fill="#1d4ed8">古典：周囲色を伝播（細い傷に強い）</text><rect x="330" y="212" width="270" height="46" rx="10" fill="#fff7ed" stroke="#ea580c" stroke-width="1.8"/><text x="465" y="240" text-anchor="middle" font-size="13" font-weight="700" fill="#c2410c">深層：内容を生成（大穴に強い）</text></svg><figcaption>インペイントは<b>マスクで指定した領域だけを描き直す</b>編集です（傷消し・不要物除去）。マスクは <code>uint8</code> で<b>白(255) ＝ 埋める領域</b>。古典の <code>cv2.inpaint</code>(Telea/NS) は<b>周囲の色・勾配を内側へ伝播</b>させるので細い傷・点に強く、大穴は苦手です。拡散インペイント/LaMa は<b>新しい内容を生成</b>するので大穴・自然な置換に向きます。</figcaption></figure>

### 正準 API と実装
```python
restored = cv2.inpaint(damaged_bgr_or_rgb, mask_uint8, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
```
`02_img2img_inpaint.py` の後半が主役で、(1) 合成した傷をマスク指定で復元し **PSNR/SSIM** で「元にどれだけ
戻ったか」を測り（Telea/NS とも PSNR≈50dB, SSIM≈0.996）、(2) 矩形領域を「消したい物体」とみなして除去します。
拡散インペイント（`AutoPipelineForInpainting`）は**モデルが数 GB と重い**ため既定ではスキップし、環境変数
`GEN_INPAINT_MODEL` を与えたときだけ走るガード付きにしています（概念＋任意）。

### 落とし穴 / 使い分け
古典法は「周囲の色で塗りつぶす」ので、**小さく細い欠損（線傷・点）に強く、大穴・構造（窓・文字）復元は苦手**で
跡が平坦になりがちです。大きな除去・自然な置換が要るなら深層インペイント（拡散 / LaMa）の出番。
マスクは uint8 で「埋める領域=255」。マスクを少し膨張（dilate）させて境界の残渣を巻き込むと綺麗になります。

---

## 4. 超解像（古典補間 → Swin2SR）

### 直感と評価の作り方
超解像（SR）は低解像（LR）から高解像（HR）を復元するタスクです。評価は **「きれいな HR を持っていて、
それをダウンスケールして LR を自作 → 各手法で復元 → 元の HR と比較」** という参照あり設計が基本です
（ground truth が手元にあるから PSNR/SSIM が測れる）。`03_superres_bg_removal.py` はこの設計で、最近傍／
バイキュービック／Lanczos（古典）と **Swin2SR**（深層・×2）を同条件で比較します。

<figure class="lec-fig"><svg viewBox="0 0 640 300" role="img" aria-label="超解像の参照あり評価。きれいなHRを縮小してLRを自作し、復元したSRを元のHRとPSNR/SSIMで比較する" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><polyline points="88,92 88,52 564,52 564,110" fill="none" stroke="#16a34a" stroke-width="1.5" stroke-dasharray="5 4"/><text x="320" y="44" text-anchor="middle" font-size="12.5" fill="#15803d">HR を正解として参照</text><rect x="40" y="92" width="96" height="96" fill="#ffffff" stroke="#16a34a" stroke-width="2"/><circle cx="80" cy="128" r="20" fill="#ea580c"/><polygon points="100,168 130,168 115,138" fill="#2563eb"/><line x1="146" y1="140" x2="198" y2="140" stroke="#52525b" stroke-width="2"/><polygon points="206,140 196,135 196,145" fill="#52525b"/><text x="176" y="124" text-anchor="middle" font-size="12" fill="#52525b">縮小 ×½</text><rect x="214" y="116" width="24" height="24" fill="#f97316"/><rect x="238" y="116" width="24" height="24" fill="#ffedd5"/><rect x="214" y="140" width="24" height="24" fill="#dbeafe"/><rect x="238" y="140" width="24" height="24" fill="#2563eb"/><rect x="214" y="116" width="48" height="48" fill="none" stroke="#71717a" stroke-width="1.5"/><line x1="272" y1="140" x2="324" y2="140" stroke="#52525b" stroke-width="2"/><polygon points="332,140 322,135 322,145" fill="#52525b"/><rect x="344" y="92" width="96" height="96" fill="#ffffff" stroke="#d4d4d8" stroke-width="1.5"/><circle cx="384" cy="128" r="20" fill="#ea580c"/><polygon points="404,168 434,168 419,138" fill="#2563eb"/><line x1="446" y1="140" x2="504" y2="140" stroke="#52525b" stroke-width="2"/><polygon points="512,140 502,135 502,145" fill="#52525b"/><rect x="512" y="110" width="104" height="60" rx="10" fill="#f4f4f5" stroke="#52525b" stroke-width="1.8"/><text x="564" y="146" text-anchor="middle" font-size="14" font-weight="700" fill="#18181b">PSNR ／ SSIM</text><text x="88" y="210" text-anchor="middle" font-size="13" font-weight="700" fill="#15803d">HR（正解・原寸）</text><text x="238" y="210" text-anchor="middle" font-size="13" font-weight="700" fill="#52525b">LR（低解像）</text><text x="392" y="210" text-anchor="middle" font-size="13" font-weight="700" fill="#52525b">SR（復元）</text><text x="320" y="236" text-anchor="middle" font-size="12" fill="#52525b">復元：bicubic / Lanczos / Swin2SR（深層）</text></svg><figcaption>超解像の評価は<b>正解(HR)を自分で作る</b>のがコツです。きれいな <b>HR</b> を <b>×½ に縮小して LR を自作</b>し、各手法（bicubic / Swin2SR）で <b>復元(SR)</b> したのち、<b>元の HR と比較</b>して <code>PSNR</code>/<code>SSIM</code> を測ります。ground truth が手元にあるので定量評価でき、点線は HR を<b>正解として再利用</b>する流れです。</figcaption></figure>

### 正準 API と実装
古典は `cv2.resize(lr, (W,H), interpolation=cv2.INTER_CUBIC)`。深層は transformers の Swin2SR を使います。
```python
from transformers import AutoImageProcessor, Swin2SRForImageSuperResolution
proc = AutoImageProcessor.from_pretrained("caidas/swin2SR-classical-sr-x2-64")
model = Swin2SRForImageSuperResolution.from_pretrained("caidas/swin2SR-classical-sr-x2-64").eval()
with torch.inference_mode():
    sr = model(**proc(lr_pil, return_tensors="pt")).reconstruction  # (1,3,2H',2W')
```
本章のシーンでは Swin2SR が bicubic を **PSNR で約 +1.5dB** 上回ります。ただし Swin2SR は**ウィンドウ幅の倍数に内部
パディング**するため、出力が厳密に 2×H,2×W にならないことがあり、その場合は目標サイズへクロップ/リサイズして揃えます
（`gen_lab.swin2sr_upscale`）。

### 落とし穴 / 使い分け
`cv2.dnn_superres`（ESPCN/FSRCNN/EDSR）は **opencv-contrib 限定**で、本講座の `opencv-python-headless` には
含まれません（スクリプトは `hasattr` で存在確認し、無ければ案内のみ）。また **Swin2SR classical-sr は
「ノイズなしのバイキュービック縮小」を前提**に学習されているため、入力に強いノイズが乗ると古典補間に
負けることがあります（劣化ミスマッチ）。古典 Real-ESRGAN(basicsr) は新しい torchvision で
`functional_tensor` import エラー等の依存破綻を起こすため、CPU 教材では Swin2SR / dnn_superres で代替します。

---

## 5. 背景除去（GrabCut）

### 直感と API
背景除去は前景マスクを推定して切り抜く処理です。古典 **GrabCut** は、矩形ヒントで「外＝確実な背景」を
初期化し、前景/背景の色分布（GMM）とグラフカットを**反復**して前景を絞り込みます。

<figure class="lec-fig"><svg viewBox="0 0 660 230" role="img" aria-label="GrabCutは矩形ヒントで背景を初期化しGMMとグラフカットの反復で前景マスクを絞り別背景にα合成する" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><text x="330" y="38" text-anchor="middle" font-size="14" font-weight="700" fill="#18181b">GrabCut の前景抽出パイプライン</text><rect x="22" y="80" width="134" height="72" rx="8" fill="#f4f4f5" stroke="#71717a" stroke-width="2"/><rect x="186" y="80" width="134" height="72" rx="8" fill="#fff7ed" stroke="#ea580c" stroke-width="2"/><rect x="350" y="80" width="134" height="72" rx="8" fill="#eff6ff" stroke="#2563eb" stroke-width="2"/><rect x="514" y="80" width="134" height="72" rx="8" fill="#ffffff" stroke="#16a34a" stroke-width="2"/><text x="89" y="112" text-anchor="middle" font-size="14" font-weight="700" fill="#18181b">矩形ヒント</text><text x="89" y="134" text-anchor="middle" font-size="11" fill="#52525b">外 ＝ 確実な背景</text><text x="253" y="112" text-anchor="middle" font-size="14" font-weight="700" fill="#c2410c">GrabCut 反復</text><text x="253" y="134" text-anchor="middle" font-size="11" fill="#52525b">GMM ＋ グラフカット</text><text x="417" y="112" text-anchor="middle" font-size="14" font-weight="700" fill="#1d4ed8">前景マスク</text><text x="417" y="134" text-anchor="middle" font-size="11" fill="#52525b">FGD ｜ PR_FGD</text><text x="581" y="112" text-anchor="middle" font-size="14" font-weight="700" fill="#15803d">α 合成</text><text x="581" y="134" text-anchor="middle" font-size="11" fill="#52525b">別背景に合成</text><line x1="158" y1="116" x2="180" y2="116" stroke="#71717a" stroke-width="2"/><polygon points="186,116 176,111 176,121" fill="#71717a"/><line x1="322" y1="116" x2="344" y2="116" stroke="#71717a" stroke-width="2"/><polygon points="350,116 340,111 340,121" fill="#71717a"/><line x1="486" y1="116" x2="508" y2="116" stroke="#71717a" stroke-width="2"/><polygon points="514,116 504,111 504,121" fill="#71717a"/></svg><figcaption><b>GrabCut</b> による背景除去の流れです。まず物体を囲む<b>矩形ヒント</b>で「外側 ＝ 確実な背景」を与え、<b>前景/背景の色分布(GMM)とグラフカットを反復</b>して<b>前景マスク</b>（<code>FGD</code>／<code>PR_FGD</code>）を絞り込み、そのマスクで切り抜いて<b>別背景へ α 合成</b>します。境界の毛羽立ち（髪・透明部）が苦手な点は深層マッティングで補います。</figcaption></figure>

```python
mask = np.zeros(rgb.shape[:2], np.uint8)
cv2.grabCut(rgb, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
fg = np.where((mask==cv2.GC_FGD)|(mask==cv2.GC_PR_FGD), 1, 0).astype(np.uint8)
```
`03_superres_bg_removal.py` の後半が、前景を抜いて切り抜き画像と「別背景への α 合成」を作ります。
GrabCut は**境界の毛羽立ち（髪・透明部）が苦手**で、そこは深層マッティング（rembg / MODNet）が得意ですが、
本講座では rembg を実行経路に入れず**概念・任意**に留めます（依存が重く衝突しやすいため）。

---

## 6. 評価指標（PSNR / SSIM / CLIPScore / FID）

### 参照あり: PSNR / SSIM
復元・超解像・インペイントのように「正解画像がある」場合に使います。
- **PSNR** = `10*log10(MAX^2 / MSE)`。大きいほど良い（dB）。ピクセル差の素朴な指標で、知覚と乖離することも。
- **SSIM** = 輝度・コントラスト・構造の局所一致を掛けた指標。1 に近いほど良い。人の見えに PSNR より近い。

本講座は **skimage を使わず numpy/cv2 で自前実装**します（`gen_lab.psnr` / `gen_lab.ssim` はガウシアン窓
11×11, σ=1.5）。`04_eval_psnr_ssim_clipscore.py` は自前実装と **torchmetrics** の `PeakSignalNoiseRatio` /
`StructuralSimilarityIndexMeasure` を突き合わせ、**値がほぼ一致**することを確認します（境界処理の差で
SSIM が小数第 3 位だけずれることがある、と明示）。

### 参照なし・生成整合: CLIPScore
生成画像には正解画像が無いので、PSNR/SSIM は使えません。代わりに **CLIPScore** =
`w * max(cos(画像埋め込み, テキスト埋め込み), 0)`（慣例 w=2.5）で「**絵が指示プロンプトに合っているか**」を
測ります。CLIP の画像/テキストを同じ空間に射影し、コサイン類似度を取るだけ。`gen_lab.load_clip_scorer` が
`openai/clip-vit-base-patch32` で実装します（transformers v5 では `forward()` の `image_embeds`/`text_embeds`
が射影後ベクトル。**L2 正規化してから内積**を取るのが要点）。

### 分布距離: FID / KID / IS
生成画像群が「実画像の分布にどれだけ近いか」を測るのが **FID**（InceptionV3 特徴の Fréchet 距離、
`||μr-μg||^2 + Tr(Σr+Σg-2(ΣrΣg)^{1/2})`）。**小さいほど良い**。`04_*` は torchmetrics があれば
24 枚規模の小デモで「崩れた分布ほど FID 大」を見せますが、**FID は本来 数百〜数万枚で安定**する点を強調します
（小サンプルでは極端に不安定）。torchmetrics の FID は `torch-fidelity` が必要なため、未導入なら概念のみ。

### 早見表（向きを間違えない）
| 指標 | 種別 | 良い方向 |
|---|---|---|
| PSNR | 参照あり・忠実度 | ↑ 大きいほど良い |
| SSIM | 参照あり・構造 | ↑ 1 に近いほど良い |
| LPIPS | 参照あり・知覚距離 | ↓ 小さいほど良い（要 net 重み・任意）|
| CLIPScore | 生成・プロンプト整合 | ↑ 大きいほど良い |
| FID / KID | 生成・分布距離 | ↓ 小さいほど良い |
| IS | 生成・多様性×明瞭さ | ↑ 大きいほど良い |

<figure class="lec-fig"><svg viewBox="0 0 640 280" role="img" aria-label="評価指標は正解画像の有無で分かれる。参照ありはPSNR/SSIM/LPIPS、生成はCLIPScoreとFID/IS" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><rect x="240" y="22" width="160" height="40" rx="9" fill="#f4f4f5" stroke="#52525b" stroke-width="1.8"/><text x="320" y="47" text-anchor="middle" font-size="14" font-weight="700" fill="#18181b">評価したい画像</text><line x1="300" y1="62" x2="172" y2="104" stroke="#52525b" stroke-width="2"/><polygon points="172,104 180,96 183,106" fill="#52525b"/><line x1="340" y1="62" x2="468" y2="104" stroke="#52525b" stroke-width="2"/><polygon points="468,104 457,106 460,96" fill="#52525b"/><rect x="44" y="110" width="242" height="130" rx="12" fill="#fff7ed" stroke="#ea580c" stroke-width="2"/><text x="165" y="146" text-anchor="middle" font-size="14" font-weight="700" fill="#c2410c">参照あり（正解と比較）</text><text x="165" y="184" text-anchor="middle" font-size="14" font-weight="700" fill="#18181b">PSNR ↑　SSIM ↑</text><text x="165" y="214" text-anchor="middle" font-size="13" fill="#3f3f46">LPIPS ↓（知覚・任意）</text><rect x="354" y="110" width="242" height="130" rx="12" fill="#eff6ff" stroke="#2563eb" stroke-width="2"/><text x="475" y="146" text-anchor="middle" font-size="14" font-weight="700" fill="#1d4ed8">参照なし（生成画像）</text><text x="475" y="184" text-anchor="middle" font-size="14" font-weight="700" fill="#18181b">CLIPScore ↑（整合）</text><text x="475" y="214" text-anchor="middle" font-size="13" fill="#3f3f46">FID ↓ ／ IS ↑（分布）</text></svg><figcaption>評価指標は<b>正解画像があるか</b>でまず分かれます。復元・超解像・インペイントのように<b>正解があるなら参照あり指標</b>：<code>PSNR</code> ↑・<code>SSIM</code> ↑（<code>LPIPS</code> は ↓ で小さいほど良い）。生成画像は正解が無いので、<b>プロンプト整合は CLIPScore ↑</b>、<b>実画像分布への近さは FID ↓・IS ↑</b> で測ります。<b>↑ ↓ の向きを取り違えない</b>のが要点です。</figcaption></figure>

---

## 🛠 章末ミニプロジェクト（統合課題）

`mini_project.py` は本章を 1 本に統合した完成形です（CPU 数分以内・全ガード付き）。

1. **生成**: SD-Turbo（1step/guidance0）でプロンプト画像を生成（取れなければ合成シーンへフォールバック）。
2. **整合評価**: 生成画像 × プロンプトの **CLIPScore**（参照なし）。
3. **劣化 → 復元**: ×2 ダウンスケール → bicubic / Swin2SR で復元 → 元画像との **PSNR/SSIM** 比較。
4. **編集**: 矩形マスクで `cv2.inpaint` の物体除去。
5. **集計**: モンタージュ画像と全指標の JSON（`mini_project_report.json`）を出力。

<figure class="lec-fig"><svg viewBox="0 0 660 300" role="img" aria-label="ミニプロジェクトは生成から整合評価、劣化と復元、編集、集計の順に一気通貫で流れる" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><text x="330" y="34" text-anchor="middle" font-size="14" font-weight="700" fill="#18181b">ミニプロジェクト — ① から ⑤ を一気通貫</text><rect x="24" y="64" width="180" height="64" rx="8" fill="#fff7ed" stroke="#ea580c" stroke-width="2"/><rect x="240" y="64" width="180" height="64" rx="8" fill="#fff7ed" stroke="#ea580c" stroke-width="2"/><rect x="456" y="64" width="180" height="64" rx="8" fill="#fff7ed" stroke="#ea580c" stroke-width="2"/><rect x="456" y="214" width="180" height="64" rx="8" fill="#fff7ed" stroke="#ea580c" stroke-width="2"/><rect x="240" y="214" width="180" height="64" rx="8" fill="#eff6ff" stroke="#2563eb" stroke-width="2"/><text x="114" y="92" text-anchor="middle" font-size="15" font-weight="700" fill="#c2410c">① 生成</text><text x="114" y="112" text-anchor="middle" font-size="11" fill="#71717a">SD-Turbo 1step</text><text x="330" y="92" text-anchor="middle" font-size="15" font-weight="700" fill="#c2410c">② 整合評価</text><text x="330" y="112" text-anchor="middle" font-size="11" fill="#71717a">CLIPScore</text><text x="546" y="92" text-anchor="middle" font-size="15" font-weight="700" fill="#c2410c">③ 劣化 → 復元</text><text x="546" y="112" text-anchor="middle" font-size="11" fill="#71717a">bicubic / Swin2SR</text><text x="546" y="242" text-anchor="middle" font-size="15" font-weight="700" fill="#c2410c">④ 編集</text><text x="546" y="262" text-anchor="middle" font-size="11" fill="#71717a">cv2.inpaint で除去</text><text x="330" y="242" text-anchor="middle" font-size="15" font-weight="700" fill="#1d4ed8">⑤ 集計</text><text x="330" y="262" text-anchor="middle" font-size="11" fill="#71717a">montage + JSON</text><line x1="206" y1="96" x2="234" y2="96" stroke="#71717a" stroke-width="2"/><polygon points="240,96 230,91 230,101" fill="#71717a"/><line x1="422" y1="96" x2="450" y2="96" stroke="#71717a" stroke-width="2"/><polygon points="456,96 446,91 446,101" fill="#71717a"/><line x1="546" y1="130" x2="546" y2="208" stroke="#71717a" stroke-width="2"/><polygon points="546,214 541,204 551,204" fill="#71717a"/><line x1="454" y1="246" x2="426" y2="246" stroke="#71717a" stroke-width="2"/><polygon points="420,246 430,241 430,251" fill="#71717a"/></svg><figcaption><b>章末ミニプロジェクト</b>は入口の生成/合成画像から <b>① 生成（SD-Turbo 1step）→ ② 整合評価（CLIPScore）→ ③ 劣化させて復元（bicubic / Swin2SR を PSNR/SSIM で比較）→ ④ 編集（<code>cv2.inpaint</code> で物体除去）→ ⑤ 集計</b> の順に一気通貫で流れます。最後の <b>⑤</b> が<b>モンタージュ画像</b>と全指標の <code>mini_project_report.json</code> へ束ねる出力ステップです。</figcaption></figure>

```bash
uv run python lectures/31_generation_editing/mini_project.py
# → outputs/31_generation_editing/mini_project_montage.png / mini_project_report.json
```

発展課題（任意）: (a) 生成プロンプトを変えて CLIPScore の変化を観察、(b) 劣化にノイズを足して Swin2SR が
古典に負ける「劣化ミスマッチ」を再現、(c) `GEN_INPAINT_MODEL` を設定して拡散インペイントと古典の品質を比較。

---

## ✅ 到達チェックリスト

- [ ] 拡散モデルの「ノイズ→反復デノイズ」と SD-Turbo の蒸留（1〜2 step / guidance 0.0）を説明できる。
- [ ] `AutoPipelineForText2Image` を float32・attention slicing・`manual_seed` で CPU 実行できる。
- [ ] img2img の **effective steps = round(steps×strength) ≥ 1** を理解し、strength を調整できる。
- [ ] `cv2.inpaint`(Telea/NS) で傷消し・物体除去を実装し、PSNR/SSIM で復元度を測れる。
- [ ] 超解像を bicubic/Lanczos と Swin2SR で比較でき、`dnn_superres` が contrib 限定だと知っている。
- [ ] GrabCut で前景を抜き、別背景に α 合成できる。
- [ ] PSNR/SSIM を自前実装でき、torchmetrics とほぼ一致することを確認した。
- [ ] CLIPScore を CLIP 埋め込み（L2 正規化→コサイン）で計算できる。
- [ ] FID/KID/IS/LPIPS の「良い方向」と「必要サンプル数」を取り違えない。
- [ ] `exercises.py` を 10/10 PASS にできる。

---

## ❓ 落とし穴・FAQ・デバッグ

- **SD-Turbo で絵が崩れる/真っ黒**: `guidance_scale` を 0.0 にする（>0 は破綻）。`num_inference_steps` は 1〜2。
  通常の SD は safety checker の NSFW 判定で黒画像を返すことがある（SD-Turbo の既定は `safety_checker=None`）。
- **img2img で `num_inference_steps is less than 1` エラー**: `round(steps*strength) ≥ 1` を満たすよう steps か
  strength を上げる（例: strength=0.5 なら steps≥2）。
- **`get_image_features` がテンソルでない（transformers v5）**: v5 では出力オブジェクトを返す。射影後埋め込みは
  `model(**inputs).image_embeds` / `.text_embeds` を使い、**L2 正規化してからコサイン**を取る。
- **`cv2.dnn_superres` が無い（AttributeError）**: `opencv-python-headless` には含まれない（contrib 限定）。
  本講座は Swin2SR と `cv2.resize` で代替。
- **Swin2SR の出力が 2×サイズちょうどでない**: ウィンドウ倍数にパディングされるため。目標サイズへクロップ/resize。
- **Swin2SR が古典に負ける**: classical-sr は「ノイズなし bicubic 縮小」前提。ノイズ劣化では破綻しうる（劣化ミスマッチ）。
- **CPU で float16 を使って落ちる/激遅**: CPU は **float32** を明示。`device_map="auto"` は使わず `.to("cpu")`。
- **FID が出ない（ModuleNotFoundError）**: torchmetrics の FID は `torch-fidelity` が必要（`uv add --group metrics torch-fidelity`）。
- **色が変（青っぽい）**: OpenCV は BGR、PIL/torch/matplotlib は RGB。保存時に `cv2.cvtColor(..., RGB2BGR)`。
- **初回が遅い/固まる**: 重みの DL（SD-Turbo ~2.5GB）。2 回目以降は `~/.cache/huggingface` から高速。オフラインは
  `HF_HUB_OFFLINE=1`。本章スクリプトは DL 失敗時もフォールバックして exit 0。

---

## 🚀 発展トピック・参考

- **少ステップ生成の系譜**: LCM / LCM-LoRA、SDXL-Turbo、FLUX-schnell。蒸留・整合（consistency）モデルで
  「1〜4 step 高速生成」が主流に。Turbo の `guidance 0.0` 作法は共通。
- **制御つき生成**: ControlNet（エッジ/深度/ポーズで構図を制御）、IP-Adapter（参照画像で画風）、
  T2I-Adapter。`27_depth_pose_flow` の深度/姿勢と接続すると強力。
- **深層インペイント/除去**: LaMa（`simple-lama-inpainting`、Fourier 畳み込みで大穴に強い）、SD-inpainting、
  Grounded-SAM で「文で指定 → 検出 → SAM マスク → 拡散で除去/置換」。`23_text_prompt_segmentation` と連携。
- **深層マッティング**: rembg / MODNet（境界・透明部に強い。依存が重いので隔離 venv 推奨）。
- **評価の発展**: LPIPS（学習済み知覚距離・↓）、DISTS、CLIP-IQA、`pyiqa` の無参照指標（BRISQUE/NIQE/MUSIQ）。
  生成評価は FID 単独でなく CLIPScore（整合）+ FID（リアルさ）+ 人手評価の併用が実務標準。
- 公式ドキュメント: diffusers <https://huggingface.co/docs/diffusers> ／ SD-Turbo
  <https://huggingface.co/stabilityai/sd-turbo> ／ Swin2SR <https://huggingface.co/caidas/swin2SR-classical-sr-x2-64>
  ／ torchmetrics image <https://lightning.ai/docs/torchmetrics/stable/>。

---

## 💡 実践ユースケース集

本章の生成・編集・評価は、そのまま **EC（ネットショップ）の商品画像づくり**に効きます。
「撮り直さずに素材を増やす／余計なものを消す／低解像の在庫写真を出品基準に上げる」という、
現場でコストに直結する作業を小ツール化できます。

### ① EC 出品バリエーション生成 + ヒーロー自動選定 — `use_case.py`（動く出発点）

<figure class="lec-fig"><svg viewBox="0 0 660 290" role="img" aria-label="EC出品ワークフロー。ベース商品からバリエーション生成しCLIPScoreでヒーロー選定、並行して不要物除去し出品キットへまとめる" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><text x="330" y="30" text-anchor="middle" font-size="14" font-weight="700" fill="#18181b">EC 出品キット — 生成・採点・除去のワークフロー</text><rect x="22" y="116" width="110" height="60" rx="8" fill="#f4f4f5" stroke="#71717a" stroke-width="2"/><rect x="170" y="54" width="152" height="56" rx="8" fill="#fff7ed" stroke="#ea580c" stroke-width="2"/><rect x="348" y="54" width="140" height="56" rx="8" fill="#eff6ff" stroke="#2563eb" stroke-width="2"/><rect x="170" y="200" width="152" height="56" rx="8" fill="#fff7ed" stroke="#ea580c" stroke-width="2"/><rect x="520" y="124" width="120" height="60" rx="8" fill="#ffffff" stroke="#16a34a" stroke-width="2"/><text x="77" y="142" text-anchor="middle" font-size="14" font-weight="700" fill="#18181b">ベース商品</text><text x="77" y="160" text-anchor="middle" font-size="11" fill="#52525b">data/ から</text><text x="246" y="80" text-anchor="middle" font-size="14" font-weight="700" fill="#c2410c">バリエーション生成</text><text x="246" y="98" text-anchor="middle" font-size="11" fill="#52525b">SD-Turbo t2i</text><text x="418" y="80" text-anchor="middle" font-size="14" font-weight="700" fill="#1d4ed8">ヒーロー選定</text><text x="418" y="98" text-anchor="middle" font-size="11" fill="#52525b">CLIPScore 最大</text><text x="246" y="226" text-anchor="middle" font-size="14" font-weight="700" fill="#c2410c">不要物除去</text><text x="246" y="244" text-anchor="middle" font-size="11" fill="#52525b">cv2.inpaint</text><text x="580" y="150" text-anchor="middle" font-size="14" font-weight="700" fill="#15803d">出品キット</text><text x="580" y="168" text-anchor="middle" font-size="11" fill="#52525b">画像 + JSON</text><line x1="132" y1="146" x2="150" y2="146" stroke="#71717a" stroke-width="2"/><line x1="150" y1="82" x2="150" y2="228" stroke="#71717a" stroke-width="2"/><line x1="150" y1="82" x2="164" y2="82" stroke="#71717a" stroke-width="2"/><polygon points="170,82 160,77 160,87" fill="#71717a"/><line x1="150" y1="228" x2="164" y2="228" stroke="#71717a" stroke-width="2"/><polygon points="170,228 160,223 160,233" fill="#71717a"/><line x1="322" y1="82" x2="342" y2="82" stroke="#71717a" stroke-width="2"/><polygon points="348,82 338,77 338,87" fill="#71717a"/><line x1="488" y1="82" x2="504" y2="82" stroke="#71717a" stroke-width="2"/><line x1="322" y1="228" x2="504" y2="228" stroke="#71717a" stroke-width="2"/><line x1="504" y1="82" x2="504" y2="228" stroke="#71717a" stroke-width="2"/><line x1="504" y1="154" x2="514" y2="154" stroke="#71717a" stroke-width="2"/><polygon points="520,154 510,149 510,159" fill="#71717a"/></svg><figcaption><b>EC 出品キット生成</b>のワークフローです。<b>ベース商品</b>から SD-Turbo の text-to-image で<b>バリエーション</b>を量産し、<b>CLIPScore（プロンプト整合）が最大の 1 枚をヒーロー</b>として自動選定します。並行して写真の<b>不要物（値札・透かし）を <code>cv2.inpaint</code> で除去</b>し、両者を<b>出品キット</b>（画像＋JSON 目録）へまとめます。重みが取れない環境では古典手法へフォールバックして必ず exit 0 で終わります。</figcaption></figure>

- **何に使うか**: 1 つの商品について、背景・ライティング違いのサムネ候補を SD-Turbo の
  text-to-image でまとめて生成し、**CLIPScore（プロンプト整合）で一番『商品らしい』1 枚（ヒーロー）**を
  自動で選ぶ。A/B テスト用の主画像候補出しを 1 コマンドで。同時に、商品写真の**不要物（値札・透かし）を
  cv2.inpaint で除去**するクリーンアップも行い、結果を「出品キット（画像＋JSON 目録）」として書き出す。
- **作り方の要点**: バリエーションは `gl.load_t2i_pipeline()`＋`num_inference_steps=1〜2 / guidance_scale=0.0 /
  256〜384px`（Turbo 厳守）。採用は `gl.load_clip_scorer()` の CLIPScore 最大値で決定。除去は
  「マスク＝白(255)」を `cv2.inpaint(..., INPAINT_TELEA)` に渡すだけ。すべて CPU・ガード付きで、
  重みが取れなければ古典の色変換バリエーション＋古典インペイントへフォールバックして **必ず exit 0**。
- **実行コマンド**:
  ```bash
  uv run python lectures/31_generation_editing/use_case.py                  # 生成＋除去（既定 both）
  uv run python lectures/31_generation_editing/use_case.py --mode variations --num 4 \
      --prompt "a product photo of a perfume bottle"                        # バリエーションのみ
  uv run python lectures/31_generation_editing/use_case.py --mode cleanup   # 不要物除去のみ
  # → outputs/31_generation_editing/use_case_variations.png / use_case_cleanup.png / use_case_listing.json
  ```
- **`data/31_generation_editing/` の置き方**: `<任意名>.png`（最初の 1 枚をベース商品として優先読込）、
  `*mask*.png`（白=255 が「消したい領域」。あれば合成シールの代わりに採用）。両方無ければ合成ボトル＋
  ダミー値札で完走する。
- **拡張アイデア**: (a) バリエーションを **img2img**（`gl.load_img2img_pipeline`）にして同じ商品の一貫性を上げる
  （本筋は IP-Adapter / ControlNet）、(b) 除去マスクを輝度しきい値や `23_text_prompt_segmentation` の
  文プロンプトで**自動検出**、(c) 採用基準を CLIPScore 単独でなく無参照 IQA（鮮鋭さ）と複合化、
  (d) スタイル一覧・採用枚数をカテゴリ別プリセットにして量産。
- **注意**: SD-Turbo は `guidance_scale>0` で崩れる／`512px` 以上は CPU で急に重い。t2i の各生成は別シードの
  “別物”なので商品の同一性は保証されない（一貫性が要るなら img2img/IP-Adapter）。`mini_project.py` が
  「生成→劣化→復元→評価」の総合学習なのに対し、本ツールは**出品ワークフローに絞った現実の小ツール**。

### ② 不要物・透かし・値札の除去（写真クリーンアップ）

- **何に使うか**: 商品・不動産・中古品の写真から、写り込んだロゴ／日付スタンプ／値札シール／小さなゴミを消す。
- **作り方の要点**: 消す領域を uint8 マスク（除去=255）にし、**小さく細い欠損は `cv2.inpaint`(Telea/NS)** で十分。
  マスクは少し `dilate` して境界の残渣を巻き込むと跡が目立たない。
- **注意**: 古典法は周囲色の伝播なので**大穴・構造（文字・窓）復元は苦手**で平坦な跡が残る。広い除去や自然な置換が
  要るなら **LaMa（`simple-lama-inpainting`）や拡散インペイント**へ（依存が重く CPU では遅いので本講座は概念/任意）。

### ③ 低解像の在庫写真を出品基準へ（超解像リマスター）

- **何に使うか**: 過去の小さいサムネしか残っていない在庫画像を、撮り直さずに出品サイズへ引き上げる。
- **作り方の要点**: まず軽量・確実な**古典補間（bicubic/Lanczos, `cv2.resize`）**をベースラインにし、品質が要る所だけ
  **Swin2SR（`gl.load_swin2sr` / `gl.swin2sr_upscale`）**。before/after は PSNR/SSIM ではなく実画では目視＋無参照 IQA で確認。
- **注意**: Swin2SR classical-sr は「**ノイズなしの bicubic 縮小**」前提で学習されているため、JPEG ノイズや圧縮劣化が強い
  実写では古典補間に負けることがある（**劣化ミスマッチ**）。出力は内部パディングで厳密 2× にならない点も忘れずクロップ。

---

## ▶ 動かし方

```bash
# 依存（未導入なら）: uv sync --group dl --group hf --group diffusion --group metrics
uv run python lectures/31_generation_editing/01_sd_turbo_t2i.py          # text-to-image
uv run python lectures/31_generation_editing/02_img2img_inpaint.py       # img2img + インペイント
uv run python lectures/31_generation_editing/03_superres_bg_removal.py   # 超解像 + 背景除去
uv run python lectures/31_generation_editing/04_eval_psnr_ssim_clipscore.py  # 評価
uv run python lectures/31_generation_editing/mini_project.py             # 統合
uv run python lectures/31_generation_editing/use_case.py                 # 実践ユースケース(EC商品写真)

# 演習（TODO を実装 → 自己採点。未実装でも exit 0）
uv run python lectures/31_generation_editing/exercises.py
uv run python lectures/31_generation_editing/exercises_solutions.py      # 模範解答（全 PASS）
```

- 出力は `outputs/31_generation_editing/` に保存（headless・`imshow` は呼ばない）。
- 入力は合成生成。実画像で試すなら `data/31_generation_editing/` に画像を置くと自動で優先読込。

---

> 参照ライブラリ（版）: **torch 2.12+cpu / torchvision 0.27+cpu / diffusers 0.38 / transformers 5.11 /
> torchmetrics 1.9 / opencv-python-headless 4.13 / numpy 2.x**（2026-06 時点）。
> CPU 前提・`model.eval()`＋`torch.inference_mode()`・拡散は `enable_attention_slicing()`。
