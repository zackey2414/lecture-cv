# 33_quantization_pruning: 量子化と枝刈り — PTQ(動的/静的)・QAT・torchao・pruningの実効速度の罠

> トラック: **最適化・デプロイ** ／ レベル: **中級** ／ 必要な依存グループ: `dl` `quant`

## 🎯 この章のゴール
int8量子化のscale/zero-point・対称/非対称・per-tensor/per-channelを理解し、動的PTQ(CPU向き)/静的PTQ(fuse+QuantStub+キャリブレーション)/QATとtorchao quantize_を使い分け、非構造化pruningはマスクを掛けるだけで実速度もサイズも縮まないという最大の誤解を実測で確認し、構造化pruningで実圧縮できる。

## 扱うトピック
- 量子化理論(scale/zero-point、対称/非対称、per-channel、PTQ/QATの使い分け)
- 動的量子化(quantize_dynamic、Linear/LSTM、CNNはほぼ効かない)
- 静的PTQ(fuse_modules+QuantStub/DeQuantStub+キャリブレーション)とQAT
- torchao quantize_(Int8DynamicActivationInt8WeightConfig)とバックエンド(fbgemm/qnnpack)
- 非構造化/構造化pruning(l1_unstructured/ln_structured/global)とprune.remove
- 『prune.removeはマスクを焼くだけ』実効速度の罠の実測

## 主要API
`torch.ao.quantization.quantize_dynamic` / `prepare/convert` / `QuantStub` / `fuse_modules` / `torch.backends.quantized.engine` / `torchao.quantization.quantize_` / `torch.nn.utils.prune.l1_unstructured` / `prune.remove`

## 評価方法
圧縮効果を三角関係で評価する: 量子化/pruning前後でaccuracy(またはmAP)の劣化・latency(p50/p99)・モデルサイズ(MB)を測って比較表にする。特に非構造化pruningでスパース率を上げても実latency・サイズが変わらないことを実測で示し、構造化pruningとの差を定量化する。

## 完成物
小分類モデルに動的/静的量子化・torchao quantize_・非構造/構造pruningを適用し、accuracy/latency/サイズの三指標を比較するスクリプト。

## CPU / GPU メモ
量子化は本質的にCPU向け機能。バックエンドはx86=fbgemm/qnnpack(Apple Silicon=qnnpack)をengineで明示。torchaoのint4/float8はGPU/ARM限定でCPUはint8 dynamicが確実。CPUに疎カーネルが無くpruningは『概念とサイズ削減』として教える。

## 予定スクリプト
- `01_dynamic_static_qat.py`
- `02_torchao_quantize.py`
- `03_pruning_speed_trap.py`

---
> ⚠️ この回はロードマップ上の**プレースホルダ**です。教材本体（解説＋実行コード＋演習）は順次作成します。

> 依存追加の例: `uv add --group dl <packages>`（必要グループ: `dl` `quant`）
