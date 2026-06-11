"""05: torchao の quantize_ — 量子化の「現行の正準 API」（概念紹介・任意依存）。

背景:
  - 本講座で使ってきた torch.ao.quantization(eager/FX) は **非推奨**で、将来 torch から
    削除予定（実行すると DeprecationWarning が出る）。後継は **torchao** の
    `quantize_(model, Config)` API で、PyTorch ネイティブの量子化/スパース化を担う。
  - ただし torchao は **任意依存**（本講座の標準環境には未導入）で、int4/float8 などの
    高速パスは主に CUDA / Apple Silicon(ARM) 向け。x86 CPU で確実に効くのは
    Int8DynamicActivationInt8WeightConfig(int8 動的) 程度。CPU 実習の主力は引き続き
    torch.ao(02/03) と 次回の ONNX Runtime 動的量子化 が無難。

この回の方針:
  - torchao が import できれば、TinyMLP に int8 動的量子化(quantize_)を実際に適用して
    サイズ・精度を before/after で示す。
  - import できなければ「概念紹介 + 導入案内」を表示して **正常終了(exit 0)** する。
    重い/任意依存を実行経路に必須化しないため。

導入（CPU 環境では効果が限定的な点に注意）:
  uv add --group quant torchao

実行:
  uv run python lectures/35_quantization_pruning/05_torchao_quantize.py
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from quant_lab import (  # noqa: E402
    accuracy,
    configure_backend,
    get_data,
    get_trained_mlp,
    quiet_warnings,
    state_dict_size_mb,
)

CONCEPT = """\
[概念] torchao quantize_ の使い方（現行の正準 API）

  import torch
  from torchao.quantization import (
      quantize_,
      Int8DynamicActivationInt8WeightConfig,  # int8 動的（x86 CPU で動く）
      Int8WeightOnlyConfig,                    # 重みのみ int8（weight-only）
      Int4WeightOnlyConfig,                    # int4（主に CUDA / ARM 向け）
  )

  model = build_model().eval()
  # その場で重みを int8 などへ置換（in-place 変換）
  quantize_(model, Int8DynamicActivationInt8WeightConfig())
  # torch.compile と併用するとカーネル融合でさらに速くなる（特に GPU）
  model = torch.compile(model)

ポイント:
  - torch.ao.quantization.quantize_dynamic(eager) の置き換え。Config を渡すだけで
    weight-only / dynamic-activation / int4 / float8 を切り替えられる。
  - 静的PTQ/QAT は torchao の PT2E フロー(prepare_pt2e/convert_pt2e + X86InductorQuantizer)
    へ移行している。
  - CPU での実効速度は限定的。x86 CPU で確実に縮めたいなら int8 dynamic か、
    次回 36 の onnxruntime.quantization.quantize_dynamic(QUInt8) が費用対効果◎。
"""


def _try_torchao_demo() -> bool:
    """torchao が使えれば int8 動的量子化を実演する。使えなければ False。"""
    try:
        from torchao.quantization import (  # noqa: PLC0415
            Int8DynamicActivationInt8WeightConfig,
            quantize_,
        )
    except Exception as exc:  # noqa: BLE001  （未導入/環境差を広く捕捉）
        print(f"[skip] torchao を import できませんでした（{type(exc).__name__}）。")
        return False

    import torchao  # noqa: PLC0415

    print(f"[torchao] version = {getattr(torchao, '__version__', 'unknown')}")
    _, _, x_te, y_te = get_data()
    model = get_trained_mlp()
    before = (accuracy(model, x_te, y_te), state_dict_size_mb(model))

    try:
        quantize_(model, Int8DynamicActivationInt8WeightConfig())
        after = (accuracy(model, x_te, y_te), state_dict_size_mb(model))
        print(f"    before: acc={before[0]:.3f} size={before[1]:.3f}MB")
        print(f"    after : acc={after[0]:.3f} size={after[1]:.3f}MB  (int8 dynamic)")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"[note] quantize_ 実行は環境依存で失敗しました（{type(exc).__name__}: {exc}）。")
        print("       x86 CPU では int8 dynamic 以外の Config は未対応のことが多いです。")
        return False


def main() -> None:
    quiet_warnings()
    engine = configure_backend()
    print(f"[setup] backend={engine}\n")

    print("torch.ao.quantization は非推奨 → 後継は torchao の quantize_。\n")
    ran = _try_torchao_demo()
    if not ran:
        print(CONCEPT)
        print("[案内] torchao は任意依存です。導入: uv add --group quant torchao")
        print("       CPU 実習の主力は torch.ao(02/03) と 次回 36 の ONNX 動的量子化です。")

    print("\n[まとめ]")
    print("  - 現行の正準量子化 API は torchao の quantize_(model, Config)。")
    print("  - だが CPU での実効速度は限定的。x86 CPU は int8 系 / ONNX 動的量子化が確実。")
    print("  - この回は任意依存のため、未導入でも概念紹介で正常終了する設計。")


if __name__ == "__main__":
    main()
