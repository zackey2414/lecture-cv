"""04: エッジ/プラットフォーム別ランタイム — OpenVINO・CoreML・LiteRT・TensorRT・netron。

デプロイ先(CPU/Intel・Mac・モバイル・NVIDIA GPU)ごとに「正解の出口」が違う。
この回は各ランタイムの立ち位置・使い方・CPUのみ環境での実習可否を整理する。
重い/プラットフォーム依存の依存(openvino/nncf/coremltools/litert/torch-tensorrt/netron)は
*実行経路で必須にしない*。import をガードし、導入済みなら最小コードを実演、未導入なら
概念と導入コマンドを案内する(本講座の CPU-only 方針)。

対応表(どこで何を使うか):
  - CPU / Intel        : OpenVINO(+ NNCF で PTQ/QAT)。AMD x86 でも動く。CPU 実習の主力。
  - Mac(Apple Silicon): CoreML(coremltools)。変換は Linux でも可だが*予測は macOS 必須*。
  - モバイル(Android) : LiteRT(litert-torch, 旧 ai-edge-torch)。PyTorch→.tflite。
  - NVIDIA GPU         : TensorRT(torch-tensorrt)。*CPU 実行不可* → 概要/可視化のみ。
  - 可視化             : netron(ONNX/TorchScript/CoreML/OpenVINO/TFLite を図示)。

実行:
  uv run python lectures/37_runtime_edge_optimization/04_edge_runtimes_concept.py
利用可能な依存があればサンプル変換も試す(無くても exit 0)。
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys
import warnings

import torch

warnings.filterwarnings("ignore")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from bench_lab import (  # noqa: E402
    ensure_output_dir,
    make_resnet18,
    synthetic_batch,
)


def _available(module: str) -> bool:
    """import 可能かを副作用なく判定(重い import を避けつつ存在チェック)。"""
    return importlib.util.find_spec(module) is not None


def section_openvino(out: pathlib.Path, model: torch.nn.Module) -> None:
    """OpenVINO: CPU 実習の主力。convert_model → compile_model('CPU') → 推論。"""
    print("=" * 70)
    print("[OpenVINO]  CPU/Intel 向け(AMD x86 でも動作)。CPU 実習の主力。")
    print("  正準フロー:")
    print("    import openvino as ov")
    print("    ov_model = ov.convert_model(torch_model, example_input=example)  # or ONNX を渡す")
    print("    compiled = ov.compile_model(ov_model, 'CPU')")
    print("    result = compiled(example.numpy())[compiled.output(0)]")
    print("  量子化: NNCF の nncf.quantize(model, calibration_dataset) で PTQ(INT8)。")
    print("  導入: uv add --group edge openvino nncf")
    if not _available("openvino"):
        print("  [未導入] → 概念のみ。Intel CPU で最も費用対効果が高い経路。")
        return
    try:
        import openvino as ov

        example = torch.from_numpy(synthetic_batch(1, seed=1))
        ov_model = ov.convert_model(model, example_input=example)
        compiled = ov.compile_model(ov_model, "CPU")
        out_key = compiled.output(0)
        y = compiled(example.numpy())[out_key]
        print(f"  [実演] OpenVINO 推論 OK: 出力 shape={tuple(y.shape)}")
        ov.save_model(ov_model, str(out / "resnet18_openvino.xml"))
        print(f"  [保存] {(out / 'resnet18_openvino.xml').name} (+ .bin)")
    except Exception as e:  # noqa: BLE001
        print(f"  [スキップ] OpenVINO 実演失敗: {type(e).__name__}: {str(e)[:80]}")


def section_coreml() -> None:
    """CoreML: Mac 向け。変換は Linux でも可だが予測実行は macOS 必須。"""
    print("=" * 70)
    print("[CoreML]  Apple(Mac/iOS)向け。Neural Engine/GPU/CPU を自動配分。")
    print("  正準フロー(Mac 受講者向け):")
    print("    import coremltools as ct")
    print("    traced = torch.jit.trace(model.eval(), example)")
    print("    mlmodel = ct.convert(traced, inputs=[ct.TensorType(shape=example.shape)])")
    print("    mlmodel.save('resnet18.mlpackage')   # 予測は macOS で mlmodel.predict(...)")
    print("  量子化/最適化: ct.optimize.coreml(パレット化/線形量子化)。")
    print("  注意: *変換*は Linux でも通ることがあるが、*予測実行*は macOS 必須。")
    print("  導入: uv add --group edge coremltools  (主に Mac)")
    if not _available("coremltools"):
        print("  [未導入] → 概念のみ。Mac トラックで実習する。")


def section_litert() -> None:
    """LiteRT(旧 ai-edge-torch): PyTorch→.tflite。モバイル向け。"""
    print("=" * 70)
    print("[LiteRT]  モバイル(Android/組込)向け。PyTorch→.tflite に変換。")
    print("  正準フロー:")
    print("    import litert_torch  # 旧 ai-edge-torch は更新停止 → litert-torch へ改名")
    print("    edge_model = litert_torch.convert(model.eval(), (example,))")
    print("    edge_model.export('resnet18.tflite')")
    print("  量子化: PT2E 量子化レシピと併用して int8 .tflite を作るのが定番。")
    print("  導入: uv add --group edge litert-torch")
    if not _available("litert_torch"):
        print("  [未導入] → 概念のみ。.tflite は Android/組込ランタイムで実行する。")


def section_tensorrt() -> None:
    """TensorRT: NVIDIA GPU 専用。CPU 実行不可 → 概要のみ。"""
    print("=" * 70)
    print("[TensorRT / torch-tensorrt]  NVIDIA GPU 専用。*CPU では実行不可*。")
    print("  立ち位置: GPU で最速級。レイヤ融合/精度キャリブレーション(int8)/カーネル自動チューニング。")
    print("  正準フロー(GPU 環境):")
    print("    import torch_tensorrt")
    print("    trt = torch_tensorrt.compile(model.cuda(), inputs=[example.cuda()],")
    print("                                 enabled_precisions={torch.float16})")
    print("  本講座(CPU)では概念・図解・netron 可視化に留める。GPU があれば段違いに速い。")


def section_netron(out: pathlib.Path, model: torch.nn.Module) -> None:
    """netron: モデルグラフの可視化。ONNX を書き出しておくと netron で開ける。"""
    print("=" * 70)
    print("[netron]  ONNX/TorchScript/CoreML/OpenVINO/TFLite のグラフを図示するビューア。")
    print("  使い方: netron.start('model.onnx')  または  https://netron.app にドラッグ&ドロップ。")
    print("  導入: uv add --group viz netron")
    # 可視化対象として ONNX を書き出しておく(netron 未導入でもファイルは作れる)
    try:
        example = torch.from_numpy(synthetic_batch(1, seed=2))
        onnx_path = out / "resnet18_for_netron.onnx"
        with torch.inference_mode():
            torch.onnx.export(
                model, (example,), str(onnx_path),
                input_names=["input"], output_names=["logits"], opset_version=17, dynamo=False,
            )
        print(f"  [準備] 可視化用 ONNX を書き出し: {onnx_path.name}")
        if _available("netron"):
            print("  [導入済] netron.start(...) でブラウザ表示できる(本スクリプトは起動しない)。")
        else:
            print("  [未導入] 上の .onnx を https://netron.app に投げれば層構造を確認できる。")
    except Exception as e:  # noqa: BLE001
        print(f"  [スキップ] ONNX 書き出し失敗: {type(e).__name__}: {str(e)[:80]}")


def main() -> None:
    out = ensure_output_dir()
    model = make_resnet18()
    print("プラットフォーム別ランタイムの地図(CPUのみ環境では OpenVINO と Mac の CoreML が実習可)\n")
    section_openvino(out, model)
    section_coreml()
    section_litert()
    section_tensorrt()
    section_netron(out, model)
    print("=" * 70)
    print("\n[まとめ] デプロイ先で出口を選ぶ:")
    print("  CPU/Intel→OpenVINO / Mac→CoreML / モバイル→LiteRT / NVIDIA GPU→TensorRT")
    print("  どれも『ONNX/TorchScript を中間ハブにして変換』が基本線。変換後は必ず数値一致を検証。")


if __name__ == "__main__":
    main()
