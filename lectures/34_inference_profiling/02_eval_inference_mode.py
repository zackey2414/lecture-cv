"""02: model.eval() と torch.inference_mode() の効果 — 正しさと速さの両面。

学ぶこと:
  - model.eval() は何を変えるか:
      * Dropout … 学習時は確率的に間引く → eval では“素通し”で決定的になる。
      * BatchNorm … 学習時はバッチ統計、eval では蓄積した移動統計を使う（バッチに依らず安定）。
    つまり eval() を忘れると“推論結果そのもの”が変わる（速度以前の正しさの問題）。
  - torch.inference_mode() / no_grad() は何を変えるか:
      * 自動微分の計算グラフを作らない → 速くて省メモリ。
      * inference_mode は no_grad より強く、バージョンカウンタ等も省くのでさらに軽い。
    推論で勾配を切り忘れると、無駄にグラフを構築して遅く・メモリ食いになる。

実行:
  uv run python lectures/34_inference_profiling/02_eval_inference_mode.py
"""

from __future__ import annotations

import pathlib
import sys

import torch
import torch.nn as nn

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from profiling_lab import (  # noqa: E402
    benchmark,
    build_model,
    ensure_out_dir,
    format_table,
    get_device,
    make_input,
    save_latency_bars,
)


class TinyNet(nn.Module):
    """eval() の効果（Dropout/BN）を可視化するための極小ネット。"""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(16, 16)
        self.bn = nn.BatchNorm1d(16)
        self.drop = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(16, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.drop(torch.relu(self.bn(self.fc1(x)))))


def show_eval_changes_outputs() -> None:
    """同じ入力でも train() と eval() で出力が変わることを実測で示す。"""
    torch.manual_seed(0)
    net = TinyNet()
    x = torch.randn(8, 16)

    # train モード: Dropout が確率的なので、2回呼ぶと出力が一致しない
    net.train()
    with torch.no_grad():
        a = net(x)
        b = net(x)
    train_deterministic = torch.allclose(a, b)

    # eval モード: Dropout 素通し・BN 移動統計 → 何度呼んでも同じ
    net.eval()
    with torch.inference_mode():
        c = net(x)
        d = net(x)
    eval_deterministic = torch.allclose(c, d)

    print("[1] model.eval() は“推論結果そのもの”を変える（速度以前の正しさ）")
    print(f"    train(): 2回の出力が一致 = {train_deterministic}  ← Dropout が毎回違う（非決定的）")
    print(f"    eval() : 2回の出力が一致 = {eval_deterministic}  ← Dropout素通し/BN移動統計（決定的）")
    print("    → eval() を忘れると推論が確率的に揺れ、BN もバッチに依存して結果が変わる。\n")


def show_grad_graph_effect(device: torch.device) -> None:
    """inference_mode の有無で“勾配グラフ”が作られるかを確認する。"""
    model = build_model("resnet18").to(device)
    x = make_input(batch_size=1).to(device)

    out_grad = model(x)  # 文脈なし: 勾配追跡が有効（グラフを作る）
    with torch.no_grad():
        out_nograd = model(x)
    with torch.inference_mode():
        out_infer = model(x)

    print("[2] inference_mode/no_grad は“勾配グラフ”を作らない（速さと省メモリの源）")
    print(f"    文脈なし        : out.requires_grad = {out_grad.requires_grad}  ← グラフ構築（推論には無駄）")
    print(f"    no_grad()       : out.requires_grad = {out_nograd.requires_grad}")
    print(f"    inference_mode(): out.requires_grad = {out_infer.requires_grad}  ← さらに軽量\n")


def bench_conditions(device: torch.device, out: pathlib.Path) -> None:
    """grad あり / no_grad / inference_mode のレイテンシを公平比較する。"""
    model = build_model("resnet18").to(device)
    x = make_input(batch_size=4).to(device)

    def run_grad() -> object:
        return model(x)  # 勾配グラフを作るので最も重い

    def run_nograd() -> object:
        with torch.no_grad():
            return model(x)

    def run_infer() -> object:
        with torch.inference_mode():
            return model(x)

    results = [
        benchmark(run_grad, label="forward (grad on)", batch_size=4, n_warmup=5, n_iter=25, device=device),
        benchmark(run_nograd, label="no_grad()", batch_size=4, n_warmup=5, n_iter=25, device=device),
        benchmark(run_infer, label="inference_mode()", batch_size=4, n_warmup=5, n_iter=25, device=device),
    ]
    print("[3] レイテンシ比較（baseline = grad on）")
    print(format_table(results, baseline_label="forward (grad on)"))
    save_latency_bars(results, out / "02_eval_inference_mode.png", title="grad vs no_grad vs inference_mode")
    print("\n    図を保存:", (out / "02_eval_inference_mode.png").name)
    print("    → 推論は必ず eval() + inference_mode()。これだけで“無料の高速化”が得られる。")


def main() -> None:
    out = ensure_out_dir()
    device = get_device()
    show_eval_changes_outputs()
    show_grad_graph_effect(device)
    bench_conditions(device, out)


if __name__ == "__main__":
    main()
