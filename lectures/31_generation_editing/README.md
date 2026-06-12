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

---

## 🛠 章末ミニプロジェクト（統合課題）

`mini_project.py` は本章を 1 本に統合した完成形です（CPU 数分以内・全ガード付き）。

1. **生成**: SD-Turbo（1step/guidance0）でプロンプト画像を生成（取れなければ合成シーンへフォールバック）。
2. **整合評価**: 生成画像 × プロンプトの **CLIPScore**（参照なし）。
3. **劣化 → 復元**: ×2 ダウンスケール → bicubic / Swin2SR で復元 → 元画像との **PSNR/SSIM** 比較。
4. **編集**: 矩形マスクで `cv2.inpaint` の物体除去。
5. **集計**: モンタージュ画像と全指標の JSON（`mini_project_report.json`）を出力。

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
