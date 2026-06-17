"""第37回 章末ミニプロジェクト — マルチランタイム・ベンチ＆意思決定レポート。

統合課題:
  同一モデル(resnet18)を複数ランタイムへ変換し、*同一指標*で公平比較して
  「用途別の最適ランタイム」を自動で意思決定し、表(Markdown)と図を成果物として残す。

含むランタイム(この環境で使えるものだけ自動収集):
  - eager fp32 / eager bf16 autocast
  - TorchScript(trace / script)
  - torch.compile(C++ ツールチェインがあれば)
  - ONNX Runtime(fp32 / int8 動的量子化)
  - OpenVINO(導入済みなら。CPU 実習の主力。未導入なら自動スキップ)

成果物(lectures/37_runtime_edge_optimization/outputs/ に保存):
  - mini_project_report.md : latency(p50/p99)・throughput・size・精度保持の比較表＋推奨
  - mini_project_bench.png : レイテンシ/スループットの可視化

意思決定:
  1) レイテンシ最小   2) サイズ最小(精度保持>=99%)  3) 総合おすすめ
  を fp32 eager を基準に選び、根拠つきで出力する。

実行:
  uv run python lectures/37_runtime_edge_optimization/mini_project.py
"""

from __future__ import annotations

import pathlib
import sys
import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from bench_lab import (  # noqa: E402
    BenchRow,
    Runtime,
    apply_bench_threads,
    benchmark_runtimes,
    collect_runtimes,
    ensure_output_dir,
    file_size_mb,
    make_resnet18,
    print_table,
    save_bench_plot,
    synthetic_batch,
)


def build_openvino(model: torch.nn.Module, out: pathlib.Path) -> Runtime | None:
    """OpenVINO ランタイム(導入済みなら)。CPU/Intel 実習の主力。未導入なら None。"""
    import importlib.util

    if importlib.util.find_spec("openvino") is None:
        print("  [skip] OpenVINO 未導入(uv add --group edge openvino で実習可)。概念は 04 参照。")
        return None
    try:
        import openvino as ov

        example = torch.from_numpy(synthetic_batch(1, seed=7))
        ov_model = ov.convert_model(model, example_input=example)
        xml = out / "resnet18_openvino.xml"
        ov.save_model(ov_model, str(xml))
        compiled = ov.compile_model(ov_model, "CPU")
        out_key = compiled.output(0)

        def infer(x: np.ndarray) -> np.ndarray:
            return compiled(x)[out_key]

        # .xml + .bin の合計サイズ
        size = file_size_mb(xml) + file_size_mb(xml.with_suffix(".bin"))
        return Runtime("openvino_cpu", infer, size, "OpenVINO CPU")
    except Exception as e:  # noqa: BLE001
        print(f"  [skip] OpenVINO 変換失敗: {type(e).__name__}: {str(e)[:80]}")
        return None


def decide(rows: list[BenchRow]) -> list[str]:
    """ベンチ表から用途別の最適ランタイムを根拠つきで選ぶ。"""
    msgs: list[str] = []

    # 1) レイテンシ最小
    fastest = min(rows, key=lambda r: r.latency.p50_ms)
    base = next(r for r in rows if r.name == "eager_fp32")
    speed = base.latency.p50_ms / fastest.latency.p50_ms
    msgs.append(
        f"レイテンシ最小: **{fastest.name}** "
        f"(p50={fastest.latency.p50_ms:.2f}ms, eager 比 {speed:.2f}x)"
    )

    # 2) サイズ最小(精度保持>=99% を満たす中で)
    keep = [r for r in rows if r.top1_agree >= 0.99 and np.isfinite(r.size_mb)]
    if keep:
        smallest = min(keep, key=lambda r: r.size_mb)
        msgs.append(
            f"サイズ最小(精度保持>=99%): **{smallest.name}** "
            f"({smallest.size_mb:.1f}MB, top1一致 {smallest.top1_agree:.1%})"
        )

    # 3) 総合おすすめ: 精度保持>=99% かつ p50 最小(壊さず一番速い)
    safe = [r for r in rows if r.top1_agree >= 0.99]
    if safe:
        best = min(safe, key=lambda r: r.latency.p50_ms)
        msgs.append(
            f"総合おすすめ(精度を保ちつつ最速): **{best.name}** "
            f"(p50={best.latency.p50_ms:.2f}ms, {best.size_mb:.1f}MB, {best.note})"
        )
    return msgs


def write_report(rows: list[BenchRow], decisions: list[str], path: pathlib.Path, threads: int) -> None:
    """比較表と意思決定を Markdown レポートに書き出す(成果物)。"""
    base_p50 = next(r for r in rows if r.name == "eager_fp32").latency.p50_ms
    lines: list[str] = []
    lines.append("# 第37回 ミニプロジェクト: マルチランタイム・ベンチ結果\n")
    lines.append(f"- モデル: resnet18 / スレッド数: {threads}(固定)/ 基準: eager_fp32\n")
    lines.append("## 比較表\n")
    lines.append(
        "| runtime | p50(ms) | p99(ms) | x速 | img/s | size(MB) | top1一致 | maxdiff | note |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---|")
    for r in rows:
        spd = base_p50 / r.latency.p50_ms if r.latency.p50_ms > 0 else float("nan")
        size = f"{r.size_mb:.2f}" if np.isfinite(r.size_mb) else "—"
        lines.append(
            f"| {r.name} | {r.latency.p50_ms:.2f} | {r.latency.p99_ms:.2f} | {spd:.2f}x | "
            f"{r.throughput_ips:.1f} | {size} | {r.top1_agree:.1%} | {r.max_abs_diff:.4f} | {r.note} |"
        )
    lines.append("\n## 意思決定(用途別の最適ランタイム)\n")
    for m in decisions:
        lines.append(f"- {m}")
    lines.append("\n## 注意\n")
    lines.append("- int8 動的量子化はサイズ削減は確実だが、Conv 主体の CNN では速度が伸びない/")
    lines.append("  逆に遅くなることがある(実効速度の罠)。必ず実測で確認する。")
    lines.append("- 数値はマシン/スレッド数/負荷で変わる。相対比較として読む。")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    threads = apply_bench_threads()
    out = ensure_output_dir()
    print(f"[setup] resnet18 / threads={threads} / out={out}\n")

    model = make_resnet18()

    print("[1] ランタイム収集(torch系・ONNX系・OpenVINO)...")
    runtimes = collect_runtimes(model, out, include_compile=True)
    ov = build_openvino(model, out)
    if ov is not None:
        runtimes.append(ov)
    print(f"    収集: {[r.name for r in runtimes]}\n")

    print("[2] 公平ベンチ実行中...\n")
    eval_batch = synthetic_batch(24, seed=99)
    rows = benchmark_runtimes(runtimes, eval_batch, latency_iters=40, throughput_batch=16)

    print("[3] 結果:")
    print_table(rows)

    decisions = decide(rows)
    print("\n[4] 意思決定:")
    for m in decisions:
        print("   - " + m.replace("**", ""))

    report = out / "mini_project_report.md"
    plot = out / "mini_project_bench.png"
    write_report(rows, decisions, report, threads)
    save_bench_plot(rows, plot, "Mini-project: runtime comparison (resnet18, CPU)")
    print("\n[5] 成果物を保存:")
    print(f"   - {report.name}")
    print(f"   - {plot.name}")

    print("\n[完了] 計測→比較→意思決定の一連を 1 本で再現できた。")
    print("       本番では『目標(latency/size/精度)』を決め、05 の順序で詰めるのが定石。")


if __name__ == "__main__":
    main()
