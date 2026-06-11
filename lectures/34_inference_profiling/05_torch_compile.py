"""05: torch.compile による高速化 — Inductor の仕組みと“現場の落とし穴”。

学ぶこと:
  - torch.compile(model) は TorchDynamo でグラフを捕捉し、Inductor が C++/OpenMP カーネルを
    生成・コンパイルする。CPU でもカーネル融合などで効くことがある。
  - mode の選択肢（"default" / "reduce-overhead" / "max-autotune"）と初回コンパイルコスト。
  - ★初回呼び出しは“コンパイル時間”を含むので非常に遅い。必ずウォームアップしてから計測する。
  - ★落とし穴: Inductor の CPU バックエンドは C++ コンパイラと Python 開発ヘッダ（Python.h）を
    必要とする。これらが無い環境では“最初の呼び出し時”に CppCompileError で失敗する。
    本スクリプトは make_compiled_runner() で安全に try/except し、失敗しても eager に退避して
    exit 0 を保つ（＝現実のCI/最小コンテナで起きる事象をそのまま教材化する）。

実行:
  uv run python lectures/34_inference_profiling/05_torch_compile.py
"""

from __future__ import annotations

import pathlib
import sys
import time

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from profiling_lab import (  # noqa: E402
    benchmark,
    build_model,
    ensure_out_dir,
    format_table,
    get_device,
    make_compiled_runner,
    make_input,
    save_latency_bars,
)


def main() -> None:
    out = ensure_out_dir()
    device = get_device()
    model = build_model("resnet18").to(device)
    x = make_input(batch_size=4).to(device)

    def run_eager() -> object:
        with torch.inference_mode():
            return model(x)

    eager = benchmark(run_eager, label="eager (baseline)", batch_size=4, n_warmup=8, n_iter=30, device=device)
    print(f"[1] eager（compile なし）: p50={eager.p50_ms:.3f}ms  throughput={eager.throughput_img_s:.1f} img/s\n")

    # === 2. torch.compile を安全に試す ===================================
    # make_compiled_runner はトリガ呼び出し（＝初回コンパイル）まで含めて try/except する。
    print("[2] torch.compile(mode='default') を試す（初回はコンパイルで遅い）")
    t0 = time.perf_counter()
    runner, ok, msg = make_compiled_runner(model, x)  # 内部で初回コンパイルが走る
    compile_s = time.perf_counter() - t0

    results = [eager]
    if ok and runner is not None:
        print(f"    コンパイル成功。初回（コンパイル込み）= {compile_s:.2f}s")
        compiled = benchmark(runner, label="torch.compile", batch_size=4, n_warmup=8, n_iter=30, device=device)
        results.append(compiled)
        print(f"    定常状態: p50={compiled.p50_ms:.3f}ms  throughput={compiled.throughput_img_s:.1f} img/s")
        speed = eager.p50_ms / compiled.p50_ms
        print(f"    eager 比 = {speed:.2f}x（CPU は劇的には速くならないこともある。要実測）\n")
    else:
        print(f"    コンパイル不可のため eager に退避（exit 0 は維持）。理由: {msg}")
        print("    → 原因の典型: Inductor の CPU バックエンドが必要とする C++ コンパイラや")
        print("       Python 開発ヘッダ（Python.h, 例えば python3-dev / python3.12-dev）が無い。")
        print("       最小コンテナ/CI で頻発する。Dockerfile に build-essential と python3-dev を入れると解決。\n")

    # === 3. 比較表と図 ===================================================
    print("[3] 比較表（baseline = eager）")
    print(format_table(results, baseline_label="eager (baseline)"))
    save_latency_bars(results, out / "05_torch_compile.png", title="eager vs torch.compile (CPU)")
    print("\n    図を保存:", (out / "05_torch_compile.png").name)

    # === 4. 知っておくべき性質（グラフブレイク・dynamic shape） ===========
    print("\n[4] torch.compile の性質（効かない/遅くなる原因の定番）")
    print("    - 初回コンパイルが重い → 1回しか推論しないなら割に合わない。サーバ常駐向き。")
    print("    - グラフブレイク: Python の動的分岐や未対応 op でグラフが分断され、最適化効果が薄れる。")
    print("      原因調査は fullgraph=True（ブレイクで例外化）や TORCH_LOGS='graph_breaks' を使う。")
    print("    - 動的 shape: 入力サイズが変わるたび再コンパイルが多発し得る。固定なら dynamic=False。")
    print("    - mode: 'reduce-overhead'（起動オーバヘッド削減）, 'max-autotune'（探索に時間をかけ最速狙い）。")

    print("\n[まとめ] torch.compile は『初回コスト・グラフブレイク・dynamic shape』に注意。")
    print("         CPU 効果は環境依存。必ず eager と公平にベンチして“本当に速いか”を確認する。")


if __name__ == "__main__":
    main()
