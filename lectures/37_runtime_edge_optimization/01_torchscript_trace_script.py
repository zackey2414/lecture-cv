"""01: TorchScript — trace と script の違い・保存/再ロード・『trace の罠』。

TorchScript は PyTorch モデルを「Python 非依存で配布・実行できる中間表現(.pt)」に
固める仕組み。エッジ/サーバ配布の最初の一歩としてよく使う。捕まえ方は2通り:

  - torch.jit.trace : 実際に1回 forward を *なぞって* グラフを記録する。
        速くて手軽だが「通った道」しか残らない。データ依存の if/for は固定化され、
        別の入力では*間違った分岐のまま*動く(= trace の罠)。
  - torch.jit.script: ソースコードを*解析*してグラフ化する。
        制御フロー(分岐・ループ)を保持できるが、未対応の Python 構文だと失敗する。

使い分けの原則: 分岐の無い純粋な計算(多くの CNN)は trace で十分。入力で経路が
変わるモデル(可変長・早期 return・動的ループ)は script を使う。

実行:
  uv run python lectures/37_runtime_edge_optimization/01_torchscript_trace_script.py
結果は lectures/37_runtime_edge_optimization/outputs/ に保存される。
"""

from __future__ import annotations

import pathlib
import sys
import warnings

import numpy as np
import torch
from torch import nn

warnings.filterwarnings("ignore")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from bench_lab import (  # noqa: E402
    Latency,
    apply_bench_threads,
    ensure_output_dir,
    make_resnet18,
    measure_latency,
    synthetic_batch,
)


class BranchyNet(nn.Module):
    """データ依存の分岐を持つトイモデル(trace の罠を見せるため)。

    入力の総和が正なら2倍、負なら符号反転する。trace は1度通った分岐しか残せない。
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.sum() > 0:
            return x * 2.0
        return -x


def demo_numerical_match(model: nn.Module, x: torch.Tensor) -> None:
    """trace / script の出力が eager と一致することを確認する(変換の正しさ検証)。"""
    with torch.inference_mode():
        ref = model(x)
    traced = torch.jit.trace(model, x)
    scripted = torch.jit.script(model)
    with torch.inference_mode():
        d_trace = (traced(x) - ref).abs().max().item()
        d_script = (scripted(x) - ref).abs().max().item()
    print("[1] eager との数値一致(最大絶対誤差):")
    print(f"    trace : {d_trace:.2e}  (0 に近ければ変換成功)")
    print(f"    script: {d_script:.2e}")


def demo_save_load(model: nn.Module, x: torch.Tensor, out: pathlib.Path) -> None:
    """.pt に保存 → torch.jit.load で再ロードし、ロード後も一致するか確認。"""
    traced = torch.jit.trace(model, x)
    # freeze: eval 済みモデルの定数畳み込み等を適用し、推論用に軽くする。
    traced = torch.jit.freeze(traced)
    path = out / "01_resnet18_traced.pt"
    traced.save(str(path))

    reloaded = torch.jit.load(str(path))  # Python の元クラス定義が無くてもロードできる
    with torch.inference_mode():
        ref = model(x)
        d = (reloaded(x) - ref).abs().max().item()
    size_mb = path.stat().st_size / 1e6
    print("\n[2] 保存 → 再ロード(配布形):")
    print(f"    保存先: {path.name}  ({size_mb:.2f} MB)")
    print(f"    再ロード後の eager との誤差: {d:.2e}")
    print("    → .pt は Python の元コード無しで torch.jit.load できる(配布に向く)")


def demo_trace_trap(out: pathlib.Path) -> None:
    """trace の罠: 別分岐の入力で trace は間違え、script は正しい。"""
    net = BranchyNet().eval()
    pos = torch.ones(4)  # sum>0 → 期待は x*2
    neg = -torch.ones(4)  # sum<0 → 期待は -x = +1

    traced = torch.jit.trace(net, pos)  # 「正の道」だけを記録してしまう
    scripted = torch.jit.script(net)  # 分岐ごと保持する

    with torch.inference_mode():
        t_pos = traced(pos)[0].item()
        t_neg = traced(neg)[0].item()  # ← 罠: まだ x*2 のまま(-2 になる)
        s_neg = scripted(neg)[0].item()  # ← 正しく -x = +1

    print("\n[3] trace の罠(データ依存分岐):")
    print(f"    入力 +1 (sum>0, 期待 x*2= 2): trace={t_pos:+.1f}")
    print(f"    入力 -1 (sum<0, 期待 -x = 1): trace={t_neg:+.1f}  ← 間違い!(+の分岐に固定された)")
    print(f"    入力 -1 (sum<0, 期待 -x = 1): script={s_neg:+.1f}  ← 正しい(分岐を保持)")
    print("    教訓: 分岐が入力で変わるモデルは trace せず script を使う(or 分岐を消す)。")


def demo_latency(model: nn.Module, out: pathlib.Path) -> None:
    """eager vs trace vs script のレイテンシ(p50/初回)を測って表示・保存。"""
    x1 = synthetic_batch(1, seed=3)
    example = torch.from_numpy(x1)

    traced = torch.jit.freeze(torch.jit.trace(model, example))
    scripted = torch.jit.freeze(torch.jit.script(model).eval())

    def eager_infer(x: np.ndarray) -> np.ndarray:
        with torch.inference_mode():
            return model(torch.from_numpy(x)).numpy()

    def traced_infer(x: np.ndarray) -> np.ndarray:
        with torch.inference_mode():
            return traced(torch.from_numpy(x)).numpy()

    def scripted_infer(x: np.ndarray) -> np.ndarray:
        with torch.inference_mode():
            return scripted(torch.from_numpy(x)).numpy()

    results: dict[str, Latency] = {
        "eager": measure_latency(eager_infer, x1),
        "trace": measure_latency(traced_infer, x1),
        "script": measure_latency(scripted_infer, x1),
    }

    print("\n[4] レイテンシ(resnet18, batch=1):")
    print(f"    {'runtime':8s} {'first(ms)':>10s} {'p50(ms)':>9s} {'p99(ms)':>9s}")
    base = results["eager"].p50_ms
    for name, lat in results.items():
        speed = base / lat.p50_ms
        print(f"    {name:8s} {lat.first_ms:10.1f} {lat.p50_ms:9.2f} {lat.p99_ms:9.2f}  ({speed:.2f}x)")
    print("    注: first(初回)には JIT 最適化の代金が乗る。比較は必ずウォームアップ後の p50 で。")

    _save_plot(out, results)
    print(f"\n[5] 図を保存: {(out / '01_torchscript_latency.png').name}")


def _save_plot(out: pathlib.Path, results: dict[str, Latency]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(results.keys())
    p50 = [results[n].p50_ms for n in names]
    first = [results[n].first_ms for n in names]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - 0.2, first, width=0.4, label="first call (incl. JIT)", color="tab:orange")
    ax.bar(x + 0.2, p50, width=0.4, label="steady p50", color="tab:blue")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("latency (ms)")
    ax.set_title("TorchScript: first call vs steady-state (resnet18, batch=1)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "01_torchscript_latency.png", dpi=110)
    plt.close(fig)


def main() -> None:
    apply_bench_threads()
    out = ensure_output_dir()
    model = make_resnet18()
    x = torch.from_numpy(synthetic_batch(2, seed=5))

    demo_numerical_match(model, x)
    demo_save_load(model, x, out)
    demo_trace_trap(out)
    demo_latency(model, out)

    print("\n[まとめ]")
    print("  - trace=実行をなぞる(速い・分岐は固定)/ script=ソース解析(分岐保持・対応構文に制約)")
    print("  - 変換後は必ず eager との数値一致を確認してから配布する")
    print("  - .pt は torch.jit.load で Python 非依存ロード可能(エッジ/サーバ配布の定番)")


if __name__ == "__main__":
    main()
