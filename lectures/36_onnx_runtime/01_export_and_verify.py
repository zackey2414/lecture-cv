"""01: PyTorch → ONNX エクスポートと『数値一致』の検証。

学ぶこと:
  - torch.onnx.export でモデルを ONNX(*.onnx) に書き出す。torch 2.9+ の既定は
    dynamo=True（torch.export 基盤の新エクスポータ）だが onnxscript が必須。本講座の標準
    環境には未導入なので、ここでは旧来の TorchScript ベース exporter（dynamo=False）を使う。
  - エクスポート後に必ず2段階で検証する:
      (a) onnx.checker.check_model … グラフの型・shape・opset の整合性を静的にチェック。
      (b) torch と onnxruntime の出力が atol/rtol 内で一致するか … 実際に同じ入力を流して
          最大絶対誤差を測る。ここを省くと「壊れた ONNX を本番に出す」最悪の事故になる。
  - dynamic_axes でバッチ次元を動的にし、export 時と違うバッチサイズでも推論できることを確認する。

実行:
  uv run python lectures/36_onnx_runtime/01_export_and_verify.py
結果は lectures/36_onnx_runtime/outputs/ に保存される。
"""

from __future__ import annotations

import pathlib
import sys
import warnings

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from onnx_helpers import (  # noqa: E402
    check_onnx,
    ensure_output_dir,
    export_onnx,
    file_size_mb,
    make_dataset,
    make_session,
    numerical_agreement,
    pin_threads,
    run_session,
    set_seed,
    train_tiny_cnn,
)


def main() -> None:
    warnings.filterwarnings("ignore")  # 旧 exporter の DeprecationWarning 等を抑制（中身は README で解説）
    set_seed(0)
    pin_threads(1)
    out = ensure_output_dir()

    # === 1. 題材モデルを用意（CPU で一瞬で学習する極小 CNN） ==============
    print("[1] TinyCNN を合成図形データで軽く学習（数秒）")
    model = train_tiny_cnn(epochs=3, verbose=True)

    # === 2. ONNX エクスポート（dynamo=False / 動的バッチ） ================
    # example はバッチ 1。dynamic_axes でバッチ次元を 'batch' という動的軸に張る。
    x, y = make_dataset(n_per_class=8, seed=42)
    example = x[:1]  # (1,3,32,32)
    onnx_path = out / "tinycnn.onnx"
    export_onnx(model, example, onnx_path, dynamic_batch=True, opset=18)
    print(f"\n[2] ONNX を書き出し: {onnx_path.name}  ({file_size_mb(onnx_path):.3f} MB)")

    # === 3. 静的検証: onnx.checker =======================================
    check_onnx(onnx_path)
    print("[3] onnx.checker.check_model: OK（グラフの型・shape・opset 整合）")

    # === 4. 数値一致検証: torch vs onnxruntime ===========================
    # 同じ入力（バッチ 24）を両者に流し、出力ロジットの最大絶対/相対誤差を測る。
    xb = x  # (24,3,32,32)
    sess = make_session(onnx_path, optimize=True, intra_threads=1)
    with torch.inference_mode():
        ref = model(xb).numpy()
    got = run_session(sess, xb.numpy())

    agree = numerical_agreement(got, ref)
    atol, rtol = 1e-4, 1e-3
    ok = bool(np.allclose(got, ref, atol=atol, rtol=rtol))
    print("\n[4] torch と onnxruntime の出力一致を検証")
    print(f"    出力 shape: torch={ref.shape}  onnx={got.shape}")
    print(f"    最大絶対誤差 = {agree['max_abs']:.3e}   最大相対誤差 = {agree['max_rel']:.3e}")
    print(f"    np.allclose(atol={atol}, rtol={rtol}) = {ok}  → {'一致 ✅' if ok else '不一致 ❌'}")

    # 予測クラスが完全一致することも確認（実務ではここが最重要）。
    pred_match = float((ref.argmax(1) == got.argmax(1)).mean())
    print(f"    top-1 予測一致率 = {pred_match:.3f}")

    # === 5. 動的バッチの確認 =============================================
    # export は batch=1 だったが、dynamic_axes のおかげで別バッチでも通る。
    print("\n[5] 動的バッチの確認（export は batch=1 だが任意バッチで推論可）")
    for b in (1, 4, 16):
        xb2 = torch.randn(b, 3, 32, 32)
        o = run_session(sess, xb2.numpy())
        print(f"    batch={b:>2} → 出力 shape {o.shape}")

    # === 6. まとめ =======================================================
    print("\n[まとめ]")
    print("  - torch.onnx.export(dynamo=False) は onnxscript 不要の旧経路。新既定は dynamo=True。")
    print("  - エクスポート後は『checker』と『torch との数値一致(atol/rtol)』を必ず両方やる。")
    print("  - dynamic_axes でバッチを動的軸にすると、export 時と違うバッチでも推論できる。")


if __name__ == "__main__":
    main()
