# 34_onnx_runtime: ONNXエクスポートとonnxruntime — グラフ最適化・動的量子化

> トラック: **最適化・デプロイ** ／ レベル: **中級** ／ 必要な依存グループ: `dl` `onnx`

## 🎯 この章のゴール
torch.onnx.export(dynamo)でPyTorchモデルをONNX化し、onnx.checkerとonnxruntimeで数値一致(atol/rtol)を検証、InferenceSession(CPUExecutionProvider)でグラフ最適化/スレッド設定して推論、quantize_dynamicでCPUに最も効くint8化を行い、optimumでTransformersもONNX化できる。

## 扱うトピック
- torch.onnx.export(dynamo=True, onnxscript必須)とopset/dynamic_shapes
- onnx.checker.check_modelとonnxruntimeでの数値一致検証(atol/rtol)
- InferenceSession(providers=['CPUExecutionProvider'])とSessionOptions(graph最適化/intra_op_num_threads)
- onnxruntime.quantization.quantize_dynamic(QUInt8、CPUで最も手軽に効くint8化)
- optimum ORTModelForXXX(export=True)でTransformers ONNX化
- netronでのグラフ可視化

## 主要API
`torch.onnx.export` / `onnx.checker.check_model` / `onnxruntime.InferenceSession` / `SessionOptions` / `GraphOptimizationLevel.ORT_ENABLE_ALL` / `onnxruntime.quantization.quantize_dynamic` / `QuantType.QUInt8` / `optimum.onnxruntime.ORTModel`

## 評価方法
ONNX化の正しさをtorch出力とonnxruntime出力の数値一致(atol/rtol、最大絶対誤差)で検証し、eager torch・ONNX・ONNX int8動的量子化のlatency/スループットとサイズ、int8の精度劣化(accuracy/mAP)を比較して費用対効果を定量化する。

## 完成物
分類/検出モデルをONNXエクスポート→数値一致検証→onnxruntime推論→動的量子化し、torch比のlatency/サイズ/精度を出す一連のスクリプト。

## CPU / GPU メモ
onnxruntimeのpip版はCPU(CPUExecutionProvider)。onnxruntime-gpuと同時インストールは競合のため不可。CPUではint8動的量子化が最も費用対効果が高い。量子化済み/MPSモデルはエクスポート失敗しやすくfp32からエクスポートする。

## 予定スクリプト
- `01_onnx_export_verify.py`
- `02_ort_inference_optimize.py`
- `03_onnx_dynamic_quant.py`

---
> ⚠️ この回はロードマップ上の**プレースホルダ**です。教材本体（解説＋実行コード＋演習）は順次作成します。

> 依存追加の例: `uv add --group dl <packages>`（必要グループ: `dl` `onnx`）
