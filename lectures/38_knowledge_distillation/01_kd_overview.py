"""01: 知識蒸留の全体像 — 圧縮の4本柱 / teacher 凍結 / 圧縮率。

学ぶこと:
  - モデルを「小さく・速く」する4本柱: 蒸留 / 量子化 / プルーニング / 低ランク分解。
    本モジュールは1本目の「蒸留(distillation)」: 大きな teacher の知識を小さな student へ移す。
  - 蒸留の3系統:
      * レスポンスベース … 出力ロジット(soft target)を真似る（Hinton 蒸留, 02〜03）
      * 特徴量ベース     … 中間特徴を真似る（FitNets, 04）
      * 関係ベース       … サンプル間/層間の関係を真似る（RKD, 04 で概観）
  - teacher は必ず凍結する: eval() + requires_grad=False。
    これを忘れると擬似ラベルが揺れ、optimizer が teacher を誤って更新する（最大の落とし穴）。
  - 圧縮率 = teacher のパラメータ数 / student のパラメータ数。まず「どれだけ小さいか」を測る。

実行:
  uv run python lectures/38_knowledge_distillation/01_kd_overview.py
結果は outputs/38_knowledge_distillation/ に保存される。
"""

from __future__ import annotations

import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from kd_lab import (  # noqa: E402
    StudentCNN,
    StudentMLP,
    TeacherCNN,
    compression_ratio,
    count_params,
    ensure_output_dir,
    freeze_module,
    quiet_warnings,
    set_seed,
)


def _show_four_pillars() -> None:
    """圧縮4本柱を一覧表示する（本モジュールの立ち位置を明示）。"""
    print("[1] モデル圧縮の4本柱（独立した刃。組み合わせ可）")
    pillars = [
        ("蒸留 distillation", "大モデルの知識を小モデルへ移す       ← 本モジュール"),
        ("量子化 quantization", "fp32→int8 等で精度を落として軽く   (35回)"),
        ("プルーニング pruning", "不要な重みを刈って疎にする          (35回)"),
        ("低ランク分解 low-rank", "重み行列を小さな積へ近似する"),
    ]
    for name, desc in pillars:
        print(f"    - {name:24s} {desc}")
    print()


def _demo_teacher_freeze() -> None:
    """teacher 凍結の効果を before/after で確認する。"""
    print("[2] teacher 凍結 — eval() + requires_grad=False")
    teacher = TeacherCNN()
    trainable_before = sum(p.requires_grad for p in teacher.parameters())
    training_mode_before = teacher.training

    freeze_module(teacher)  # ← これが凍結の中身
    trainable_after = sum(p.requires_grad for p in teacher.parameters())

    print(f"    凍結前: training={training_mode_before}  "
          f"requires_grad=True のテンソル数={trainable_before}")
    print(f"    凍結後: training={teacher.training}  "
          f"requires_grad=True のテンソル数={trainable_after}")
    print("    → 凍結後は全パラメータが学習対象外。BN/dropout も eval で固定され、")
    print("      teacher の出力(擬似ラベル)が毎回ブレなくなる。\n")


def _demo_optimizer_safety() -> None:
    """student だけを optimizer に渡す（teacher を混ぜない）正しい配線を示す。"""
    print("[3] optimizer には student のパラメータだけを渡す")
    teacher = freeze_module(TeacherCNN())
    student = StudentCNN()
    # 正: student のみ。teacher を混ぜると擬似ラベルが学習で動いてしまう。
    opt = torch.optim.AdamW(student.parameters(), lr=2e-3)
    num_groups = sum(len(g["params"]) for g in opt.param_groups)
    teacher_tensors = {id(p) for p in teacher.parameters()}
    leaked = any(id(p) in teacher_tensors for g in opt.param_groups for p in g["params"])
    print(f"    optimizer が抱えるテンソル数={num_groups}（student のみ）")
    print(f"    teacher のパラメータが混入しているか: {leaked}（False が正しい）\n")


def _compare_sizes(out: pathlib.Path) -> dict[str, int]:
    """teacher / student の規模と圧縮率をまとめ、棒グラフに保存する。"""
    teacher = TeacherCNN()
    student_cnn = StudentCNN()
    student_mlp = StudentMLP()
    sizes = {
        "TeacherCNN": count_params(teacher),
        "StudentCNN": count_params(student_cnn),
        "StudentMLP": count_params(student_mlp),
    }
    print("[4] 規模と圧縮率")
    for name, n in sizes.items():
        print(f"    {name:12s} params={n:>8,d}")
    print(f"    圧縮率 Teacher/StudentCNN = {compression_ratio(teacher, student_cnn):>5.1f}x")
    print(f"    圧縮率 Teacher/StudentMLP = {compression_ratio(teacher, student_mlp):>5.1f}x")
    print("    → StudentCNN は teacher の十数分の一。これを精度を保ったまま学ばせるのが蒸留。")
    print("      MLP は最初の Linear(1024,h) が重く、conv の StudentCNN より遥かにパラメータが多い")
    print("      → 蒸留の student には conv 系の小モデルが効率的。\n")

    _save_figure(out, sizes)
    print("[5] 図を保存:", (out / "01_model_sizes.png").name)
    return sizes


def _save_figure(out: pathlib.Path, sizes: dict[str, int]) -> None:
    """パラメータ数の棒グラフを保存する（ASCII タイトルでフォント警告回避）。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.2))
    names = list(sizes.keys())
    vals = [sizes[n] for n in names]
    bars = ax.bar(names, vals, color=["tab:red", "tab:blue", "tab:green"])
    ax.set_title("teacher vs student: parameter count (smaller = cheaper)")
    ax.set_ylabel("num parameters")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{v:,}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "01_model_sizes.png", dpi=110)
    plt.close(fig)


def main() -> None:
    quiet_warnings()
    set_seed(0)
    out = ensure_output_dir()

    _show_four_pillars()
    _demo_teacher_freeze()
    _demo_optimizer_safety()
    _compare_sizes(out)

    print("[まとめ]")
    print("  - 蒸留は圧縮4本柱の1つ。teacher(大)の知識を student(小)へ移す。")
    print("  - teacher は eval()+requires_grad=False で凍結し、optimizer には student だけ渡す。")
    print("  - 次は 02 でソフトターゲットと温度付き KL（蒸留の心臓部）を手で確かめる。")


if __name__ == "__main__":
    main()
