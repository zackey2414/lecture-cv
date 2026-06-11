# 35_runtime_edge_optimization: ランタイム/エッジ最適化 — OpenVINO・CoreML・LiteRT・TensorRT概要

> トラック: **最適化・デプロイ** ／ レベル: **上級** ／ 必要な依存グループ: `dl` `onnx`

## 🎯 この章のゴール
プラットフォーム別ランタイムを理解し、CPU/IntelはOpenVINO(convert_model+compile_model('CPU')、NNCFで量子化)、MacはCoreML(coremltools.convert)、モバイルはLiteRT(litert-torch)で最適化でき、GPU専用TensorRTは概要/netron可視化に留め、速度/精度のトレードオフで手法選択の意思決定順序を整理する。

## 扱うトピック
- OpenVINO(openvino.convert_model/compile_model('CPU'))とNNCFのPTQ/QAT
- CoreML(coremltools.convert、変換はLinux可だが予測はmacOS必須)
- LiteRT(litert-torch、PyTorch→.tflite、旧ai-edge-torchから改名)
- TensorRT/torch-tensorrtはGPU専用で概要・図解・netron可視化のみ
- 手法選択の意思決定順序(eval+inference_mode→bf16+compile→ONNX動的量子化→静的PTQ→QAT/pruning)
- CPUのみで実習できるのはOpenVINOとMacのCoreML

## 主要API
`openvino.convert_model` / `openvino.compile_model` / `nncf.quantize` / `coremltools.convert` / `litert_torch` / `netron.start` / `torch_tensorrt.compile`

## 評価方法
各ランタイム(eager torch/ONNX/OpenVINO/CoreML)でlatency(p50/p99)とスループット、精度保持(accuracy/mAP)を同一指標で公平比較し、モデルサイズも併記して用途別の最適手法を意思決定する。NNCF量子化前後の精度-速度トレードオフも測る。

## 完成物
同一モデルをOpenVINO(とMacではCoreML)へ変換して推論ベンチを取り、torch/ONNX/OpenVINOのlatency/精度/サイズ比較表を出すスクリプト。

## CPU / GPU メモ
OpenVINOはIntel最適だがAMD x86でも動作しCPU実習の主力、Apple SiliconはCoreML推奨。CoreMLの予測実行はmacOS必須。TensorRTはCPU実行不可で概要のみ。litert-torch変換はCPU広く対応。

## 予定スクリプト
- `01_openvino_convert_bench.py`
- `02_coreml_convert.py`
- `03_runtime_decision.py`

---
> ⚠️ この回はロードマップ上の**プレースホルダ**です。教材本体（解説＋実行コード＋演習）は順次作成します。

> 依存追加の例: `uv add --group dl <packages>`（必要グループ: `dl` `onnx`）
