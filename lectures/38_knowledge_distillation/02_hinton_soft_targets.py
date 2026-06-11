"""02: ソフトターゲットと温度付き KL ダイバージェンス（Hinton 蒸留の心臓部）。

学ぶこと:
  - ハードラベル(one-hot)は「正解は1つ」しか教えないが、teacher の softmax 出力(soft target)は
    「2 番手以降の確率」=クラス間の類似構造(暗黙知, dark knowledge)も教えてくれる。
  - 温度 T でロジットを軟化する: softmax(logits / T)。T を上げるほど分布がなだらかになり、
    暗黙知が露わになる（T=1 は素の softmax、T→∞ で一様分布へ）。
  - 蒸留のソフト損失は KL ダイバージェンス。正準形を厳守する:
        soft = F.kl_div(F.log_softmax(student/T, 1), F.softmax(teacher/T, 1),
                        reduction='batchmean') * T**2
      * 入力は log_softmax(student)、ターゲットは softmax(teacher)（順序・log を逆にすると崩れる）
      * reduction='batchmean'（数式の 1/N。既定の 'mean' は要素数で割れてしまう）
      * 勾配が 1/T^2 に縮むのを補正するため T**2 を掛ける
  - 合成損失 = alpha*ソフト + (1-alpha)*ハード(CE)。

実行:
  uv run python lectures/38_knowledge_distillation/02_hinton_soft_targets.py
結果は outputs/38_knowledge_distillation/ に保存される。
"""

from __future__ import annotations

import pathlib
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from kd_lab import (  # noqa: E402
    ensure_output_dir,
    kd_kl_loss,
    quiet_warnings,
    response_kd_loss,
    set_seed,
    soften_logits,
)


def _entropy(p: torch.Tensor) -> float:
    """確率分布の平均エントロピー(nats)。大きいほど「なだらか」。"""
    return float(-(p * (p + 1e-12).log()).sum(dim=1).mean())


def _demo_soft_targets() -> None:
    """ハードラベル vs ソフトターゲットの情報量の違いを見る。"""
    print("[1] ハードラベル vs ソフトターゲット（暗黙知）")
    # teacher のロジット例: 正解は「猫(0)」だが「犬(1)」にもそこそこ似ている、という構造を持つ。
    logits = torch.tensor([[8.0, 5.0, 0.5, -2.0]])  # 猫 / 犬 / 車 / 飛行機 を想定
    labels = ["cat", "dog", "car", "plane"]
    hard = torch.zeros(1, 4)
    hard[0, 0] = 1.0
    print("    ハードラベル        :", {labels[i]: round(float(hard[0, i]), 3) for i in range(4)})
    for T in (1.0, 4.0):
        p = soften_logits(logits, T)
        print(f"    ソフト(T={T:>3.0f})        :",
              {labels[i]: round(float(p[0, i]), 3) for i in range(4)})
    print("    → ハードは『猫=1』だけ。ソフトは『猫>犬≫車>飛行機』という似ている度合いまで伝える。")
    print("      T を上げるほど犬や車の確率が持ち上がり、暗黙知が読み取りやすくなる。\n")


def _demo_temperature_entropy() -> None:
    """温度 T を上げると分布のエントロピーが単調に増える（軟化する）ことを確認。"""
    print("[2] 温度 T とエントロピー（T が大きいほどなだらか）")
    g = torch.Generator().manual_seed(0)
    logits = torch.randn(256, 6, generator=g) * 3.0  # それなりに尖ったロジット
    for T in (0.5, 1.0, 2.0, 4.0, 8.0):
        ent = _entropy(soften_logits(logits, T))
        print(f"    T={T:>4.1f}  平均エントロピー={ent:.3f} nats")
    print("    → T<1 は分布を尖らせ、T>1 はならす。蒸留は T=2〜5 あたりがよく使われる。\n")


def _demo_kl_direction() -> None:
    """KL の引数順序・log の有無を間違えると値が変わる（学習が崩れる）ことを示す。"""
    print("[3] KL ダイバージェンスの正準形（順序・log・reduction を厳守）")
    g = torch.Generator().manual_seed(1)
    s_logits = torch.randn(32, 6, generator=g)
    t_logits = torch.randn(32, 6, generator=g) * 2.0
    T = 4.0

    correct = kd_kl_loss(s_logits, t_logits, T)  # 正: log_softmax(student) / softmax(teacher)
    # 誤1: 入力とターゲットを入れ替える（teacher を log 側に）→ 別物の値
    wrong_swap = F.kl_div(F.log_softmax(t_logits / T, 1),
                          F.softmax(s_logits / T, 1), reduction="batchmean") * T ** 2
    # 誤2: reduction='mean'（要素数で割れて数式とずれる）
    wrong_mean = F.kl_div(F.log_softmax(s_logits / T, 1),
                          F.softmax(t_logits / T, 1), reduction="mean") * T ** 2
    print(f"    正準(batchmean)        : {float(correct):.4f}")
    print(f"    引数を入れ替えた KL    : {float(wrong_swap):.4f}  ← 別物（順序が大事）")
    print(f"    reduction='mean'       : {float(wrong_mean):.4f}  ← {6}倍小さい（クラス数で割れる）")
    print("    → 'batchmean' を使い、入力=log_softmax(student)・ターゲット=softmax(teacher) を守る。\n")


def _demo_t2_scaling() -> None:
    """T**2 補正の意味: T を変えてもソフト損失の勾配スケールが揃うことを確認。"""
    print("[4] なぜ T**2 を掛けるのか（勾配スケールの補正）")
    g = torch.Generator().manual_seed(2)
    t_logits = torch.randn(64, 6, generator=g) * 2.0
    print("    student ロジットに対するソフト損失の勾配ノルム:")
    for T in (1.0, 2.0, 4.0):
        s_logits = torch.zeros(64, 6, requires_grad=True)
        # T**2 補正あり
        loss_scaled = kd_kl_loss(s_logits, t_logits, T)
        g_scaled = torch.autograd.grad(loss_scaled, s_logits, retain_graph=True)[0].norm()
        # T**2 補正なし
        s2 = torch.zeros(64, 6, requires_grad=True)
        soft_raw = F.kl_div(F.log_softmax(s2 / T, 1), F.softmax(t_logits / T, 1),
                            reduction="batchmean")
        g_raw = torch.autograd.grad(soft_raw, s2)[0].norm()
        print(f"    T={T:>3.0f}:  T^2補正あり={float(g_scaled):.4f}   補正なし={float(g_raw):.4f}")
    print("    → 補正なしだと T を上げるほど勾配が 1/T^2 に縮み、実効学習率が変わって不安定。")
    print("      T**2 を掛けると T によらず勾配スケールがほぼ揃う。\n")


def _demo_combined_loss() -> None:
    """合成損失 alpha*soft + (1-alpha)*hard の振る舞いを確認。"""
    print("[5] 合成損失 = alpha*ソフト + (1-alpha)*ハード(CE)")
    g = torch.Generator().manual_seed(3)
    s_logits = torch.randn(16, 6, generator=g, requires_grad=True)
    t_logits = torch.randn(16, 6, generator=g) * 2.0
    y = torch.randint(0, 6, (16,), generator=g)
    for alpha in (0.0, 0.5, 1.0):
        loss = response_kd_loss(s_logits, t_logits, y, temperature=4.0, alpha=alpha)
        tag = {0.0: "素のCEのみ", 0.5: "半々", 1.0: "ソフトのみ"}[alpha]
        print(f"    alpha={alpha:.1f} ({tag}):  loss={float(loss):.4f}")
    print("    → alpha=0 は素の教師あり学習、alpha=1 はソフトのみ。実務は 0.5〜0.9 を起点に調整。\n")


def _save_figure(out: pathlib.Path) -> None:
    """温度ごとのソフトターゲットと、T とエントロピーの関係を可視化して保存。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    # 左: 1 サンプルのソフトターゲットを温度別に棒グラフ
    logits = torch.tensor([[8.0, 5.0, 0.5, -2.0]])
    labels = ["cat", "dog", "car", "plane"]
    ax = axes[0]
    width = 0.25
    xs = torch.arange(4)
    for k, T in enumerate((1.0, 4.0, 8.0)):
        p = soften_logits(logits, T)[0]
        ax.bar(xs + (k - 1) * width, p.numpy(), width, label=f"T={T:.0f}")
    ax.set_xticks(xs.numpy())
    ax.set_xticklabels(labels)
    ax.set_title("softened targets: higher T reveals dark knowledge")
    ax.set_ylabel("probability")
    ax.legend(fontsize=8)

    # 右: T vs エントロピー
    ax = axes[1]
    g = torch.Generator().manual_seed(0)
    big = torch.randn(256, 6, generator=g) * 3.0
    ts = [0.5, 1, 1.5, 2, 3, 4, 6, 8]
    ents = [_entropy(soften_logits(big, t)) for t in ts]
    ax.plot(ts, ents, "o-", color="tab:purple")
    ax.set_title("temperature vs entropy (softness)")
    ax.set_xlabel("temperature T")
    ax.set_ylabel("mean entropy (nats)")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out / "02_soft_targets.png", dpi=110)
    plt.close(fig)


def main() -> None:
    quiet_warnings()
    set_seed(0)
    out = ensure_output_dir()

    _demo_soft_targets()
    _demo_temperature_entropy()
    _demo_kl_direction()
    _demo_t2_scaling()
    _demo_combined_loss()

    _save_figure(out)
    print("[6] 図を保存:", (out / "02_soft_targets.png").name)
    print("\n[まとめ]")
    print("  - soft target は暗黙知(クラス間の類似)を伝える。T で軟化の強さを調整。")
    print("  - KL は log_softmax(student)/softmax(teacher)・batchmean・T**2 を厳守。")
    print("  - 次は 03 でこの損失を使い、素の student と蒸留 student の精度を比べる。")


if __name__ == "__main__":
    main()
