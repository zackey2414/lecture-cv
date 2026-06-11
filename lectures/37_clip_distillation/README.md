# 37_clip_distillation: VLM/CLIPの蒸留(b) — TinyCLIP/MobileCLIP・埋め込み模倣

> トラック: **最適化・デプロイ** ／ レベル: **上級** ／ 必要な依存グループ: `dl` `hf` `embed` `distill`

## 🎯 この章のゴール
大CLIPを小型エンコーダへ蒸留する手法を理解し、埋め込みをL2正規化しlogit_scaleで温度付与したteacherの画像-テキスト類似度行列をstudentにKL/MSEで模倣させるcontrastive distillation、TinyCLIP(重み継承+親和性蒸留)とMobileCLIP(DataCompDRデータセット強化)の違いを学び、open_clipで小型CLIPをCPUロードしzero-shotを回して参照リポのmobileclip(mobileclip_blt.ts)に接続できる。

## 扱うトピック
- CLIP蒸留: 埋め込みL2正規化+logit_scale(温度)を揃えた類似度行列の模倣
- TinyCLIP(重み継承+親和性蒸留)とMobileCLIP/MobileCLIP2(DataCompDRデータセット強化、アーキよりデータ)
- open_clip.create_model_and_transforms('MobileCLIP-S0'/'TinyCLIP-...')でCPUロード
- encode_image/encode_textとF.normalize、list_pretrainedで事前学習キー確認
- 参照リポのmobileclip_blt.ts(TorchScript)とエッジ展開
- 知識蒸留/dataset distillation/dataset reinforcementの用語区別

## 主要API
`open_clip.create_model_and_transforms` / `open_clip.get_tokenizer` / `model.encode_image` / `model.encode_text` / `torch.nn.functional.normalize` / `model.logit_scale` / `open_clip.list_pretrained`

## 評価方法
蒸留した小CLIPの品質を、ラベル付き小データでのzero-shot分類accuracyとteacher CLIPとの差で評価し、画像テキスト検索Recall@kも比較する。teacherとstudentの埋め込みコサイン類似度(整合度)を測り、logit_scale/正規化を揃えないとスケールずれが起きることを示す。

## 完成物
teacher CLIPの埋め込み/類似度をstudentに模倣させる蒸留ループ(または小型MobileCLIP/TinyCLIPのロード)と、zero-shot accuracy・Recall@k・埋め込み類似度の評価コード。

## CPU / GPU メモ
open_clipのMobileCLIP/TinyCLIP推論はCPUで動くが重みDLが数百MB(mobileclip_blt.tsは599MB)でキャッシュ前提、小型モデルで実習。MobileCLIP2系は新しめのopen_clipとimage_mean/std上書きが要る場合あり。

## 予定スクリプト
- `01_clip_contrastive_distill.py`
- `02_mobileclip_tinyclip_load.py`
- `03_distilled_clip_eval.py`

---
> ⚠️ この回はロードマップ上の**プレースホルダ**です。教材本体（解説＋実行コード＋演習）は順次作成します。

> 依存追加の例: `uv add --group dl <packages>`（必要グループ: `dl` `hf` `embed` `distill`）
