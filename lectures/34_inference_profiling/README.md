# 32_inference_profiling: 推論高速化の地図 — 計測ファースト・プロファイリング・autocast・torch.compile

> トラック: **最適化・デプロイ** ／ レベル: **初級** ／ 必要な依存グループ: `dl`

## 🎯 この章のゴール
最適化前に必ず計測する原則を体得し、レイテンシ(p50/p99)とスループット(img/s)の違い、eval+inference_modeでの勾配/Dropout無効化、ウォームアップ→多数回反復→中央値評価の正しいベンチ手順、torch.profilerでのボトルネック特定、CPUはbf16のautocast・torch.compile(Inductor)での高速化を書ける。

## 扱うトピック
- 計測ファースト原則とレイテンシp50/p99 vs スループット
- model.eval()/torch.inference_mode()の効果とウォームアップ
- 正しいベンチ(perf_counter・中央値/分位点、GPUはcuda.synchronize)
- torch.profilerでCPU時間集計(key_averages().table)とchrome trace
- 数値精度(CPUはfp16が遅くbf16が現実的)とtorch.autocast(device_type='cpu')
- torch.compile(mode/初回コンパイルコスト/グラフブレイク)

## 主要API
`model.eval()` / `torch.inference_mode()` / `time.perf_counter` / `torch.utils.benchmark.Timer` / `torch.profiler.profile` / `key_averages().table` / `torch.autocast` / `torch.compile`

## 評価方法
本モジュール自体が評価(ベンチ)回。同一モデルでウォームアップ有無・eval/inference_mode有無・bf16 autocast・torch.compileの各条件のlatency(p50/p99)とスループットを測り、公平な比較表を作る。ウォームアップを省いた誤計測との差も示す。

## 完成物
任意モデルのlatency/スループットを正しい手順で計測しprofilerで律速演算子を特定、bf16/compileの効果を表にするベンチマークユーティリティ。

## CPU / GPU メモ
CPUではcuda.synchronizeは不要(呼ぶと誤り)、低精度はbf16/int8が正解でfp16は逆に遅い。torch.compileはCPUでもInductorが有効だが初回コンパイルと再コンパイルに注意。

## 予定スクリプト
- `01_benchmark_correct.py`
- `02_profiler.py`
- `03_autocast_compile.py`

---
> ⚠️ この回はロードマップ上の**プレースホルダ**です。教材本体（解説＋実行コード＋演習）は順次作成します。

> 依存追加の例: `uv add --group dl <packages>`（必要グループ: `dl`）
