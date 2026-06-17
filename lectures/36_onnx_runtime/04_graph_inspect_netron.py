"""04: ONNX グラフの可視化と中身の点検（netron 紹介 + Python API で構造把握）。

学ぶこと:
  - ONNX は「計算グラフ」をそのまま保存したファイル。中身は onnx の Python API で読める:
      * graph.input / graph.output … 入出力テンソルの名前と shape（動的軸は記号名）。
      * graph.node … 演算子（op_type）のリスト。Conv/Relu/Gemm などの並び。
      * graph.initializer … 重み定数（パラメータ）。総数とバイト数からモデル規模が分かる。
  - onnxruntime に SessionOptions.optimized_model_filepath を渡すと、グラフ最適化『後』の
    モデルを書き出せる。最適化前後でノード数・op 種別を比べると、Conv+ReLU 融合や
    Gemm+Activation 融合（FusedGemm）が実際に起きていることを目で確認できる。
  - netron はブラウザ/アプリでグラフを綺麗に可視化する定番ツール（pip install netron →
    netron.start("model.onnx")）。本講座の標準環境には入れていないので、ここでは Python API で
    『テキスト版 netron』を作り、netron は概念として案内する。

実行:
  uv run python lectures/36_onnx_runtime/04_graph_inspect_netron.py
結果（図と要約 JSON）は lectures/36_onnx_runtime/outputs/ に保存される。
"""

from __future__ import annotations

import collections
import json
import pathlib
import sys
import warnings

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from onnx_helpers import (  # noqa: E402
    ensure_output_dir,
    export_onnx,
    make_dataset,
    set_seed,
    train_tiny_cnn,
)


def _shape_of(value_info) -> list:
    """ValueInfoProto から shape（次元）を取り出す。動的軸は記号名（文字列）で返す。"""
    dims = []
    for d in value_info.type.tensor_type.shape.dim:
        dims.append(d.dim_param if d.dim_param else d.dim_value)
    return dims


def _summarize(path: pathlib.Path) -> dict:
    """ONNX ファイルを読み、入出力・ノード op 種別・initializer をテキスト要約する。"""
    import onnx

    model = onnx.load(str(path))
    g = model.graph
    op_hist = dict(collections.Counter(n.op_type for n in g.node))
    n_init = len(g.initializer)
    init_bytes = sum(len(init.raw_data) for init in g.initializer)
    return {
        "opset": int(model.opset_import[0].version),
        "ir_version": int(model.ir_version),
        "inputs": {i.name: _shape_of(i) for i in g.input},
        "outputs": {o.name: _shape_of(o) for o in g.output},
        "num_nodes": len(g.node),
        "op_histogram": op_hist,
        "num_initializers": n_init,
        "initializer_MB": round(init_bytes / (1024 * 1024), 4),
    }


def _dump_optimized(src: pathlib.Path, dst: pathlib.Path) -> None:
    """onnxruntime のグラフ最適化『後』のモデルをファイルに書き出す。"""
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.optimized_model_filepath = str(dst)
    so.log_severity_level = 3  # ハード固有最適化の注意ログを抑制
    ort.InferenceSession(str(src), sess_options=so, providers=["CPUExecutionProvider"])


def main() -> None:
    warnings.filterwarnings("ignore")
    set_seed(0)
    out = ensure_output_dir()

    # === 1. モデルを ONNX 化 =============================================
    print("[1] TinyCNN を ONNX 化（動的バッチ）")
    model = train_tiny_cnn(epochs=2, verbose=False)
    x, _ = make_dataset(n_per_class=2, seed=1)
    raw = export_onnx(model, x[:1], out / "tinycnn.onnx", dynamic_batch=True)

    # === 2. 最適化前の構造を点検 =========================================
    before = _summarize(raw)
    print("\n[2] 最適化前のグラフ構造（テキスト版 netron）")
    _print_summary(before)

    # === 3. 最適化後の構造と比較 =========================================
    opt = out / "tinycnn_optimized.onnx"
    _dump_optimized(raw, opt)
    after = _summarize(opt)
    print("\n[3] グラフ最適化『後』の構造（ORT_ENABLE_ALL で書き出し）")
    _print_summary(after)

    print("\n[4] 最適化で何が起きたか")
    print(f"    ノード数: {before['num_nodes']} → {after['num_nodes']}")
    fused = set(after["op_histogram"]) - set(before["op_histogram"])
    gone = set(before["op_histogram"]) - set(after["op_histogram"])
    print(f"    消えた op: {sorted(gone)}   現れた融合 op: {sorted(fused)}")
    print("    → ReLU が Conv/Gemm に融合され（FusedGemm 等）、メモリ往復と呼び出しが減る。")

    # === 4. 要約 JSON と図を保存 =========================================
    (out / "04_graph_summary.json").write_text(
        json.dumps({"before": before, "after": after}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _save_chart(out, before, after)
    print("\n[5] 保存:", (out / "04_graph_summary.json").name, "/", (out / "04_graph_opt.png").name)

    print("\n[まとめ]")
    print("  - ONNX は計算グラフそのもの。onnx API で入出力・op 並び・重みを点検できる。")
    print("  - optimized_model_filepath で最適化後モデルを書き出し、融合の効果を目視できる。")
    print("  - 綺麗な可視化は netron（pip install netron → netron.start('model.onnx')）。")


def _print_summary(s: dict) -> None:
    """要約 dict を読みやすく表示する。"""
    print(f"    opset={s['opset']}  ir={s['ir_version']}  "
          f"nodes={s['num_nodes']}  init={s['num_initializers']} ({s['initializer_MB']}MB)")
    print(f"    inputs : {s['inputs']}")
    print(f"    outputs: {s['outputs']}")
    print(f"    op_histogram: {s['op_histogram']}")


def _save_chart(out: pathlib.Path, before: dict, after: dict) -> None:
    """最適化前後の op 種別ヒストグラムを並べた棒グラフを保存する。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    ops = sorted(set(before["op_histogram"]) | set(after["op_histogram"]))
    bvals = [before["op_histogram"].get(o, 0) for o in ops]
    avals = [after["op_histogram"].get(o, 0) for o in ops]
    xpos = np.arange(len(ops))
    w = 0.38

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(xpos - w / 2, bvals, w, label="before opt", color="tab:orange")
    ax.bar(xpos + w / 2, avals, w, label="after opt (ENABLE_ALL)", color="tab:blue")
    ax.set_xticks(xpos)
    ax.set_xticklabels(ops, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("count")
    ax.set_title("ONNX op histogram: graph optimization (fusion)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "04_graph_opt.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
