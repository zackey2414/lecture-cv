"""04: 枝刈り(pruning)と『実効速度の罠』— マスクを掛けても密行列のまま。

この回の主役は、初学者が最もハマる誤解の実測検証:
  「torch.nn.utils.prune は重みに 0 マスクを掛けるだけ。prune.remove してもテンソルは
   密(dense)のままで、形・dtype は変わらない。だから CPU の実レイテンシも保存サイズも縮まない。」

学ぶこと:
  - 非構造化(unstructured) L1 枝刈り: 小さい重みから個別に 0 にする。スパース率を上げると
    精度は落ちるが、density はそのまま → レイテンシ不変・サイズ不変（むしろマスク付きは増える）。
  - 疎フォーマット(CSR)に変換すれば「保存サイズ」は縮められる。だが CPU に疎行列の高速
    カーネルがほぼ無いため「実速度」は縮まない（density のままの密 GEMM が一番速い）。
  - 構造化(structured) 枝刈り → 小さい『密』モデルに作り直す と、行列そのものが小さくなり
    実速度もサイズも本当に縮む。実圧縮したいなら構造化 + リビルド。

実行:
  uv run python lectures/35_quantization_pruning/04_pruning_speed_trap.py
結果は lectures/35_quantization_pruning/outputs/ に保存される。
"""

from __future__ import annotations

import copy
import io
import json
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from quant_lab import (  # noqa: E402
    accuracy,
    benchmark,
    configure_backend,
    ensure_output_dir,
    get_data,
    get_trained_mlp,
    global_unstructured_prune,
    quiet_warnings,
    remove_prune_reparam,
    state_dict_size_mb,
    structured_compress_mlp,
    weight_sparsity,
)


def _sparse_csr_size_mb(model: torch.nn.Module) -> float:
    """各 Linear/Conv 重みを CSR(疎)で保存したときの合計サイズ(MB)。

    密で 0 を並べる代わりに「非ゼロの値と位置」だけ持つので、高スパースなら小さくなる。
    ただしこれは『保存サイズ』の話で、実行速度は別問題（CPU に速い疎カーネルが無い）。
    """
    total = 0
    buf = io.BytesIO()
    for m in model.modules():
        if isinstance(m, (torch.nn.Linear, torch.nn.Conv2d)):
            w = m.weight.detach()
            csr = w.reshape(w.shape[0], -1).to_sparse_csr()
            buf.seek(0)
            buf.truncate(0)
            torch.save((csr.crow_indices(), csr.col_indices(), csr.values()), buf)
            total += len(buf.getvalue())
    return total / 1e6


def main() -> None:
    quiet_warnings()
    engine = configure_backend()
    out = ensure_output_dir()
    _, _, x_te, y_te = get_data()
    sample = torch.randn(256, 1, 16, 16)
    print(f"[setup] backend={engine}, threads={torch.get_num_threads()}\n")

    mlp = get_trained_mlp()
    base_acc = accuracy(mlp, x_te, y_te)
    base_size = state_dict_size_mb(mlp)
    base_bench = benchmark(mlp, sample)
    print(f"[0] ベースライン TinyMLP: acc={base_acc:.3f} size={base_size:.3f}MB "
          f"p50={base_bench['p50']:.3f}ms\n")

    # === 1. 非構造化枝刈り: スパース率↑ で精度↓・でも size/latency は不変 ===
    print("[1] 非構造化 グローバル L1 枝刈り — スパース率を上げても密のまま")
    print(f"    {'sparsity':>9}{'acc':>8}{'dense_MB':>11}{'csr_MB':>9}{'p50_ms':>9}")
    sweep: list[dict] = []
    for amount in (0.0, 0.3, 0.5, 0.7, 0.9):
        pruned = copy.deepcopy(mlp)
        if amount > 0:
            global_unstructured_prune(pruned, amount)
            remove_prune_reparam(pruned)  # マスクを焼く（それでも密テンソル）
        sp = weight_sparsity(pruned)
        acc = accuracy(pruned, x_te, y_te)
        dense_mb = state_dict_size_mb(pruned)
        csr_mb = _sparse_csr_size_mb(pruned)
        p50 = benchmark(pruned, sample)["p50"]
        sweep.append({"sparsity": round(sp, 3), "accuracy": round(acc, 4),
                      "dense_mb": round(dense_mb, 4), "csr_mb": round(csr_mb, 4),
                      "p50_ms": round(p50, 4)})
        print(f"    {sp:>9.2f}{acc:>8.3f}{dense_mb:>11.4f}{csr_mb:>9.4f}{p50:>9.3f}")
    print("    → dense_MB と p50 はスパース率に依らずほぼ一定（密 GEMM のまま）。")
    print("       csr_MB だけは縮むが、それは『保存』の話。実行は密のまま速くならない。\n")

    # === 2. マスク付き state_dict は逆に「増える」 =====================
    masked = copy.deepcopy(mlp)
    global_unstructured_prune(masked, 0.9)  # prune.remove せず weight_orig+weight_mask を保持
    masked_size = state_dict_size_mb(masked)
    print("[2] prune.remove する前の state_dict は weight_orig + weight_mask を持つ")
    print(f"    base={base_size:.3f}MB → マスク付き(90%スパース)={masked_size:.3f}MB "
          f"（{masked_size / base_size:.2f}x に増加！）")
    print("    → 「枝刈りしたのにファイルが大きくなった」のはマスクを別途保存しているから。\n")

    # === 3. 構造化枝刈り → 小さい密モデルへリビルド（本当に縮む） =======
    print("[3] 構造化枝刈り + リビルド — 隠れニューロンを間引いて小さい密 MLP に作り直す")
    print(f"    {'keep_frac':>10}{'acc':>8}{'size_MB':>10}{'p50_ms':>9}{'params':>10}")
    structured: list[dict] = []
    for keep in (1.0, 0.5, 0.3):
        small = structured_compress_mlp(mlp, keep_frac=keep)
        acc = accuracy(small, x_te, y_te)
        size_mb = state_dict_size_mb(small)
        p50 = benchmark(small, sample)["p50"]
        n_params = sum(p.numel() for p in small.parameters())
        structured.append({"keep_frac": keep, "accuracy": round(acc, 4),
                           "size_mb": round(size_mb, 4), "p50_ms": round(p50, 4),
                           "params": n_params})
        print(f"    {keep:>10.2f}{acc:>8.3f}{size_mb:>10.4f}{p50:>9.3f}{n_params:>10d}")
    print("    → keep_frac を下げると size も p50 も実際に縮む（行列が小さくなったから）。\n")

    # === 4. 保存 =======================================================
    _save_figure(out, base_size, sweep, structured)
    payload = {
        "backend": engine,
        "baseline": {"accuracy": round(base_acc, 4), "size_mb": round(base_size, 4),
                     "p50_ms": round(base_bench["p50"], 4)},
        "unstructured_sweep": sweep,
        "masked_state_dict_mb": round(masked_size, 4),
        "structured_rebuild": structured,
    }
    (out / "04_pruning.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("[4] 保存:", (out / "04_pruning_speed_trap.png").name, "/", (out / "04_pruning.json").name)

    print("\n[まとめ]")
    print("  - 非構造化枝刈りは『マスクを掛けるだけ』。density・形・dtype は不変 → 実速度もサイズも縮まない。")
    print("  - prune.remove はマスクを重みに焼くだけで、やはり密。CSR にすれば保存は縮むが実行は速くならない。")
    print("  - 実圧縮したいなら 構造化枝刈り + リビルド か 量子化(02/03)。CPU では量子化が確実。")


def _save_figure(out: pathlib.Path, base_size: float, sweep: list[dict],
                 structured: list[dict]) -> None:
    """非構造化(罠) と 構造化(本当に縮む) を対比した図を保存する。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.0))

    sp = [r["sparsity"] for r in sweep]
    # 左: 非構造化 — スパース率 vs (accuracy, p50)。p50 が動かないのが要点。
    ax = axes[0]
    ax.plot(sp, [r["accuracy"] for r in sweep], "o-", color="tab:blue", label="accuracy")
    ax.set_xlabel("sparsity (unstructured)")
    ax.set_ylabel("accuracy", color="tab:blue")
    ax.set_ylim(0, 1.05)
    ax2 = ax.twinx()
    ax2.plot(sp, [r["p50_ms"] for r in sweep], "s--", color="tab:red", label="p50 latency")
    ax2.set_ylabel("p50 latency (ms)", color="tab:red")
    ax.set_title("unstructured: acc drops, latency flat")

    # 中: 非構造化 — dense vs CSR の保存サイズ
    ax = axes[1]
    ax.plot(sp, [r["dense_mb"] for r in sweep], "o-", color="tab:gray", label="dense save")
    ax.plot(sp, [r["csr_mb"] for r in sweep], "^-", color="tab:green", label="CSR save")
    ax.set_xlabel("sparsity")
    ax.set_ylabel("state_dict size (MB)")
    ax.set_title("dense size flat; only CSR shrinks (storage only)")
    ax.legend(fontsize=8)

    # 右: 構造化リビルド — keep_frac vs (size, p50)
    ax = axes[2]
    keep = [r["keep_frac"] for r in structured]
    ax.plot(keep, [r["size_mb"] for r in structured], "o-", color="tab:purple", label="size (MB)")
    ax.set_xlabel("keep fraction (structured rebuild)")
    ax.set_ylabel("size (MB)", color="tab:purple")
    ax.invert_xaxis()
    ax3 = ax.twinx()
    ax3.plot(keep, [r["p50_ms"] for r in structured], "s--", color="tab:orange", label="p50 (ms)")
    ax3.set_ylabel("p50 latency (ms)", color="tab:orange")
    ax.set_title("structured rebuild: really shrinks")

    fig.tight_layout()
    fig.savefig(out / "04_pruning_speed_trap.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
