"""04: 数値精度と autocast — CPU は fp16 が遅く bf16 が現実的。

学ぶこと:
  - 浮動小数点の精度とメモリ/速度のトレードオフ。fp32（既定）/ fp16 / bf16 の違い。
      * fp16  … 仮数10bit・指数5bit。表現範囲が狭くオーバーフローしやすい。GPU(Tensor Core)向け。
      * bf16  … 仮数 7bit・指数 8bit（fp32 と同じ指数幅）。範囲が広く推論で扱いやすい。
    ★CPU では fp16 演算はソフトウェア実装で“遅い/未対応”が多い。CPU の低精度は bf16 か int8 が正解。
  - torch.autocast(device_type="cpu", dtype=torch.bfloat16) で“混合精度推論”を体験する。
    autocast は対象演算（matmul/conv 等）だけを bf16 で実行し、敏感な箇所は fp32 に保つ。
  - 速度だけでなく“どれだけ結果がズレたか”（最大絶対誤差・argmax 一致）をセットで評価する。

注意: CPU での bf16 高速化は CPU の対応命令（AVX512-BF16 / AMX 等）に依存し、環境により
      効いたり効かなかったりする。本スクリプトは“実測して正直に表示する”方針を取る。

実行:
  uv run python lectures/34_inference_profiling/04_autocast_bf16.py
"""

from __future__ import annotations

import pathlib
import sys

import torch

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


def compare_outputs(model: torch.nn.Module, x: torch.Tensor) -> None:
    """fp32 と bf16 autocast の出力差（最大絶対誤差・argmax一致）を測る。"""
    with torch.inference_mode():
        y_fp32 = model(x)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            y_bf16 = model(x).float()  # 比較のため fp32 に戻す

    max_abs = (y_fp32 - y_bf16).abs().max().item()
    argmax_match = (y_fp32.argmax(-1) == y_bf16.argmax(-1)).float().mean().item()
    print("[1] 精度: fp32 と bf16 autocast の出力差")
    print(f"    最大絶対誤差 = {max_abs:.5f}  /  top1（argmax）一致率 = {argmax_match * 100:.1f}%")
    print("    → bf16 は小さくズレるが分類の答え（argmax）はほぼ保たれる、が定番の挙動。\n")


def explain_fp16_on_cpu(model: torch.nn.Module, x: torch.Tensor) -> None:
    """CPU の fp16 autocast は遅い/不安定になりがち、という事実を安全に確認する。"""
    print("[2] CPU で fp16 autocast を試す（遅い/未対応が多いという“事実”の確認）")
    try:
        with torch.inference_mode(), torch.autocast(device_type="cpu", dtype=torch.float16):
            _ = model(x)
        print("    実行は通ったが、CPU の fp16 はソフトウェア寄りで高速化は期待しにくい。")
    except Exception as e:  # noqa: BLE001  未対応 op で落ちても講座は止めない
        print(f"    fp16 autocast で例外: {type(e).__name__}（CPU では fp16 が未対応な op がある証拠）")
    print("    → CPU の低精度は bf16（範囲が広く安定）か int8（第35回）を使うのが定石。\n")


def bench_precision(device: torch.device, out: pathlib.Path) -> None:
    """fp32 と bf16 autocast のレイテンシを公平比較する。"""
    model = build_model("resnet18").to(device)
    x = make_input(batch_size=4).to(device)

    def run_fp32() -> object:
        with torch.inference_mode():
            return model(x)

    def run_bf16() -> object:
        with torch.inference_mode(), torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            return model(x)

    results = [
        benchmark(run_fp32, label="fp32 (baseline)", batch_size=4, n_warmup=8, n_iter=30, device=device),
        benchmark(run_bf16, label="bf16 autocast", batch_size=4, n_warmup=8, n_iter=30, device=device),
    ]
    print("[3] レイテンシ比較（baseline = fp32）")
    print(format_table(results, baseline_label="fp32 (baseline)"))
    save_latency_bars(results, out / "04_autocast_bf16.png", title="fp32 vs bf16 autocast (CPU)")
    speed = results[0].p50_ms / results[1].p50_ms
    print(f"\n    bf16 / fp32 の速度比 = {speed:.2f}x", end="  ")
    if speed > 1.05:
        print("（この CPU は bf16 命令が効いて高速化）")
    elif speed < 0.95:
        print("（この CPU では bf16 がむしろ遅い＝命令未対応。autocast を当てれば速い、ではない）")
    else:
        print("（ほぼ互角。bf16 の効果は CPU 命令に強く依存する）")
    print("    図を保存:", (out / "04_autocast_bf16.png").name)


def main() -> None:
    out = ensure_out_dir()
    device = get_device()
    model = build_model("resnet18").to(device)
    x = make_input(batch_size=4).to(device)

    compare_outputs(model, x)
    explain_fp16_on_cpu(model, x)
    bench_precision(device, out)

    print("\n[まとめ] 精度を落とす最適化は『速度・メモリ』と『結果のズレ』を必ずセットで測る。")
    print("         CPU は bf16 が現実的（fp16 は GPU 向け）。効くかは CPU 命令次第なので実測する。")


if __name__ == "__main__":
    main()
