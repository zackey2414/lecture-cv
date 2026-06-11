# 29_generation_editing: 画像生成・編集 — 拡散モデル text-to-image・インペイント・超解像・背景除去

> トラック: **生成・編集** ／ レベル: **上級** ／ 必要な依存グループ: `dl` `hf` `diffusion` `classical` `metrics`

## 🎯 この章のゴール
diffusersで生成系を概観し、SD-Turbo(steps=1/guidance=0.0)でCPUでも数秒〜十数秒のtext-to-image、マスク指定インペイント(diffusers/LaMa)、超解像(OpenCV dnn_superres/Swin2SR)、背景除去/マッティングを実装でき、FID/PSNR/SSIM/LPIPSで生成・復元品質を評価できる。

## 扱うトピック
- 拡散モデルの直感とAutoPipelineForText2Image(SD-Turbo、num_inference_steps=1/guidance_scale=0.0/attention_slicing)
- インペイント(AutoPipelineForInpainting、古典cv2.inpaintとLaMaの比較)
- 超解像(cv2.dnn_superres ESPCN/FSRCNN→Swin2SR)
- 背景除去/マッティング
- 評価: 分布指標FID/KID/IS(十分なサンプル数)、参照あり忠実度PSNR/SSIM/LPIPS
- CPU向け設定とsafety_checkerのNSFW黒画像注意

## 主要API
`AutoPipelineForText2Image` / `stabilityai/sd-turbo` / `AutoPipelineForInpainting` / `pipe.enable_attention_slicing` / `cv2.dnn_superres.DnnSuperResImpl_create` / `Swin2SRForImageSuperResolution` / `torchmetrics.image.FrechetInceptionDistance` / `torchmetrics.image.StructuralSimilarityIndexMeasure`

## 評価方法
超解像/インペイントなど参照ありタスクはPSNR=10log10(MAX^2/MSE)・SSIM・LPIPS(知覚距離、-1..1正規化必須)で復元忠実度を評価。text-to-image生成は実画像群と生成群のFID(InceptionV3特徴のFréchet距離)・KID・ISを数百枚規模で算出する。指標ごとの良し悪し方向を明示。

## 完成物
SD-Turboでプロンプト生成、マスクインペイントで物体除去、dnn_superres/Swin2SRで超解像を行うスクリプトと、FID/PSNR/SSIM/LPIPS評価コード。

## CPU / GPU メモ
CPUはSD-Turbo(1step)・enable_attention_slicing()で数秒〜十数秒、enable_model_cpu_offload()は不要。dnn_superres(contrib)とSwin2SRはCPU可、Real-ESRGAN(basicsr)は依存破綻のため避ける。FIDは小サンプルで不安定。

## 予定スクリプト
- `01_sd_turbo_t2i.py`
- `02_inpaint_lama.py`
- `03_superres.py`
- `04_fid_psnr_ssim_lpips.py`

---
> ⚠️ この回はロードマップ上の**プレースホルダ**です。教材本体（解説＋実行コード＋演習）は順次作成します。

> 依存追加の例: `uv add --group dl <packages>`（必要グループ: `dl` `hf` `diffusion` `classical` `metrics`）
