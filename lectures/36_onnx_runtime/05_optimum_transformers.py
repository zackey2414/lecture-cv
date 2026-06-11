"""05: Transformers の ONNX 化 — optimum(ORTModel) の概念 + 自前エクスポートの実演。

学ぶこと:
  - HuggingFace Transformers のモデルを ONNX 化する正準ツールが optimum / optimum-onnx。
    一行で「ロード時に ONNX へ変換し、onnxruntime で推論する」ことができる:

        from optimum.onnxruntime import ORTModelForImageClassification
        ort_model = ORTModelForImageClassification.from_pretrained(
            "microsoft/resnet-18", export=True)          # ← その場で ONNX 化
        ort_model.save_pretrained("resnet18_onnx")        # ← 量子化・配布もここから

    forward の使い勝手は transformers と同じまま、裏側が onnxruntime になる。
  - optimum は任意依存（標準環境には未導入）。import を try/except で守り、無ければ
    『概念紹介＋導入案内』にフォールバックする。代わりに、同じ原理を自前の小型 Transformer
    エンコーダで実演する: torch.onnx.export → checker → onnxruntime で数値一致を確認。
    optimum が内部でやっていることの本質（export → verify → run）はこれと同じ。

  導入したい場合（重い依存のため任意）:
      uv add --group onnx "optimum[onnx]"

実行:
  uv run python lectures/36_onnx_runtime/05_optimum_transformers.py
"""

from __future__ import annotations

import pathlib
import sys
import warnings

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from onnx_helpers import (  # noqa: E402
    check_onnx,
    ensure_output_dir,
    file_size_mb,
    make_session,
    numerical_agreement,
    set_seed,
)


class TinyTextEncoder(nn.Module):
    """Transformers を模した極小エンコーダ（埋め込み + TransformerEncoder + 分類ヘッド）。

    本物の BERT/ViT を DL せずとも『系列 → ロジット』の ONNX 化が学べるよう最小構成にした。
    """

    def __init__(self, vocab: int = 2000, dim: int = 96, seq: int = 8, classes: int = 3) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab, dim)
        self.pos = nn.Parameter(torch.randn(1, seq, dim) * 0.02)
        layer = nn.TransformerEncoderLayer(dim, nhead=4, dim_feedforward=384, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.head = nn.Linear(dim, classes)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        h = self.emb(input_ids) + self.pos
        h = self.encoder(h)
        return self.head(h.mean(dim=1))  # 平均プールして分類


def _try_optimum() -> bool:
    """optimum.onnxruntime が import できるか確認する（あれば True）。"""
    try:
        import optimum.onnxruntime  # noqa: F401

        return True
    except Exception:
        return False


def main() -> None:
    warnings.filterwarnings("ignore")
    set_seed(0)
    out = ensure_output_dir()

    # === 1. optimum の有無を確認 =========================================
    has_optimum = _try_optimum()
    print("[1] optimum(ORTModel) の利用可否:", "利用可" if has_optimum else "未導入（概念紹介に切替）")
    if has_optimum:
        print("    → 本番では ORTModelForXXX.from_pretrained(name, export=True) が最短。")
    else:
        print("    導入: uv add --group onnx \"optimum[onnx]\"")
        print("    使い方（概念）:")
        print("      from optimum.onnxruntime import ORTModelForImageClassification")
        print("      m = ORTModelForImageClassification.from_pretrained('microsoft/resnet-18',")
        print("                                                         export=True)")
        print("      m.save_pretrained('resnet18_onnx')  # 量子化/配布もここから")

    # === 2. 自前の小型 Transformer を ONNX 化（optimum の中身を実演） ====
    print("\n[2] 小型 Transformer エンコーダを自前で ONNX 化（export→verify→run）")
    model = TinyTextEncoder().eval()
    input_ids = torch.randint(0, 2000, (4, 8))  # (batch, seq) のトークン列を模す

    onnx_path = out / "tiny_text_encoder.onnx"
    # 注意: nn.TransformerEncoder は eval+inference_mode 下で融合カーネル
    # (_transformer_encoder_layer_fwd) に切り替わり、旧 exporter が対応しない。
    # トレースは grad 有効（inference_mode で包まない）で行うと分解された経路になり export できる。
    torch.onnx.export(
        model,
        (input_ids,),
        str(onnx_path),
        dynamo=False,
        opset_version=18,
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes={"input_ids": {0: "batch"}, "logits": {0: "batch"}},
    )
    print(f"    書き出し: {onnx_path.name}  ({file_size_mb(onnx_path):.3f} MB)")

    # === 3. 検証（checker + 数値一致） ===================================
    check_onnx(onnx_path)
    sess = make_session(onnx_path, optimize=True, intra_threads=1)
    got = sess.run(None, {"input_ids": input_ids.numpy()})[0]
    with torch.inference_mode():
        ref = model(input_ids).numpy()
    agree = numerical_agreement(got, ref)
    ok = bool(np.allclose(got, ref, atol=1e-4, rtol=1e-3))
    print(f"\n[3] 数値一致: 最大絶対誤差 {agree['max_abs']:.3e}  allclose={ok} "
          f"→ {'一致 ✅' if ok else '不一致 ❌'}")

    # === 4. 動的量子化（テキスト系は Linear/MatMul が主役で int8 が効きやすい）
    from onnxruntime.quantization import QuantType, quantize_dynamic

    int8_path = out / "tiny_text_encoder_int8.onnx"
    quantize_dynamic(str(onnx_path), str(int8_path), weight_type=QuantType.QUInt8)
    ratio = file_size_mb(onnx_path) / max(file_size_mb(int8_path), 1e-9)
    print(f"\n[4] 動的量子化: {file_size_mb(onnx_path):.3f}MB → "
          f"{file_size_mb(int8_path):.3f}MB（{ratio:.2f}x 縮小）")
    print("    Transformer は Linear/MatMul 比率が高く、CPU では int8 動的量子化の費用対効果が高い。")
    print("    補足: Embedding テーブルは量子化対象外なので、語彙が大きいと比は理論上限(4x)に届かない。")
    print("    また極小モデルでは量子化ノードの固定オーバーヘッドが勝ち、逆に増えることもある（要実測）。")

    print("\n[まとめ]")
    print("  - optimum の ORTModelForXXX(export=True) は『DL→ONNX化→ORT 推論』を一行に畳む。")
    print("  - 中身は本スクリプトと同じ export→checker→数値一致→（必要なら）量子化の流れ。")
    print("  - optimum 未導入でも、torch.onnx.export で同じことが手で書ける（原理理解が大事）。")


if __name__ == "__main__":
    main()
