"""03: レスポンス蒸留(Hinton KD)で学習 — 素の student vs 蒸留 student。

学ぶこと:
  - 同じ小 student アーキを、(A) 素の教師あり学習 と (B) レスポンス蒸留 で学習し、
    テスト accuracy を比較して「暗黙知の移転」を定量化する。
  - セットアップ（KD が効きやすい王道）:
      * teacher … クリーンなフルデータ(660枚)で学習 → テスト acc 高い
      * student … 少量サブセット(132枚)で学習。しかもラベルの 40% が汚れている。
      * teacher の soft target はノイズに惑わされないので、KD はノイズ過学習を抑える正則化になる。
  - 温度 T と alpha を振って accuracy がどう変わるかを観察する（最適値はスイープで探す）。

実行:
  uv run python lectures/38_knowledge_distillation/03_response_kd_train.py
結果は outputs/38_knowledge_distillation/ に保存される。
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from kd_lab import (  # noqa: E402
    ALPHA,
    STUDENT_EPOCHS,
    STUDENT_LR,
    TEMPERATURE,
    StudentCNN,
    accuracy,
    ensure_output_dir,
    freeze_module,
    get_data,
    get_trained_teacher,
    quiet_warnings,
    set_seed,
    train_response_kd,
    train_supervised,
)


def _train_baseline(d: dict) -> StudentCNN:
    """(A) 素の教師あり学習: ノイジーなハードラベルだけで学ぶ。"""
    set_seed(0)
    student = StudentCNN()
    train_supervised(student, d["x_sub"], d["y_sub_noisy"],
                     epochs=STUDENT_EPOCHS, lr=STUDENT_LR)
    return student


def _train_kd(d: dict, teacher, temperature: float, alpha: float) -> StudentCNN:
    """(B) レスポンス蒸留: teacher の soft target + ノイジーハードを混ぜて学ぶ。"""
    set_seed(0)
    student = StudentCNN()
    train_response_kd(student, teacher, d["x_sub"], d["y_sub_noisy"],
                      temperature=temperature, alpha=alpha,
                      epochs=STUDENT_EPOCHS, lr=STUDENT_LR)
    return student


def _headline_comparison(d: dict, teacher) -> dict[str, float]:
    """素の student vs 蒸留 student の代表比較。"""
    print("[1] 素の student vs 蒸留 student（同じアーキ・同じデータ）")
    teacher_acc = accuracy(teacher, d["x_te"], d["y_te"])
    base = _train_baseline(d)
    kd = _train_kd(d, teacher, TEMPERATURE, ALPHA)
    acc_base = accuracy(base, d["x_te"], d["y_te"])
    acc_kd = accuracy(kd, d["x_te"], d["y_te"])
    print(f"    teacher (フルデータ)        acc = {teacher_acc:.3f}")
    print(f"    (A) baseline student        acc = {acc_base:.3f}  ← ノイジーラベルで学習")
    print(f"    (B) 蒸留 student (T={TEMPERATURE:.0f},a={ALPHA})  acc = {acc_kd:.3f}  ← soft target で正則化")
    gain = acc_kd - acc_base
    print(f"    → 蒸留による改善: {gain:+.3f}（teacher の暗黙知がノイズ過学習を抑えた）\n")
    return {"teacher": teacher_acc, "baseline": acc_base, "kd": acc_kd}


def _temperature_sweep(d: dict, teacher) -> dict[float, float]:
    """温度 T を振って accuracy の変化を見る（alpha は固定）。"""
    print(f"[2] 温度スイープ（alpha={ALPHA} 固定）")
    result = {}
    for T in (1.0, 2.0, 4.0, 8.0, 16.0):
        student = _train_kd(d, teacher, T, ALPHA)
        result[T] = accuracy(student, d["x_te"], d["y_te"])
        print(f"    T={T:>4.0f}:  acc={result[T]:.3f}")
    print("    → T が小さすぎると soft が尖って情報が乏しく、大きすぎると一様に近づく。中庸が良い。\n")
    return result


def _alpha_sweep(d: dict, teacher) -> dict[float, float]:
    """alpha を振って accuracy の変化を見る（T は固定）。"""
    print(f"[3] alpha スイープ（T={TEMPERATURE:.0f} 固定）")
    result = {}
    for alpha in (0.0, 0.3, 0.5, 0.7, 0.9, 1.0):
        student = _train_kd(d, teacher, TEMPERATURE, alpha)
        result[alpha] = accuracy(student, d["x_te"], d["y_te"])
        tag = " (素のCE)" if alpha == 0.0 else (" (ソフトのみ)" if alpha == 1.0 else "")
        print(f"    alpha={alpha:.1f}:  acc={result[alpha]:.3f}{tag}")
    print("    → alpha=0(素のCE)が最低。少しでもソフトを混ぜると改善する（暗黙知の効果）。\n")
    return result


def _save_figure(out: pathlib.Path, head: dict, t_sweep: dict, a_sweep: dict) -> None:
    """代表比較とスイープ結果を図に保存する。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    ax = axes[0]
    names = ["teacher", "baseline", "kd"]
    vals = [head[n] for n in names]
    bars = ax.bar(names, vals, color=["tab:red", "tab:gray", "tab:blue"])
    ax.set_title("teacher vs baseline vs KD student")
    ax.set_ylabel("test accuracy")
    ax.set_ylim(0, 1)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{v:.2f}", ha="center", va="bottom", fontsize=9)

    ax = axes[1]
    ts = list(t_sweep.keys())
    ax.plot(ts, [t_sweep[t] for t in ts], "o-", color="tab:green")
    ax.set_xscale("log", base=2)
    ax.set_title(f"temperature sweep (alpha={ALPHA})")
    ax.set_xlabel("temperature T")
    ax.set_ylabel("test accuracy")
    ax.grid(alpha=0.3)

    ax = axes[2]
    al = list(a_sweep.keys())
    ax.plot(al, [a_sweep[a] for a in al], "o-", color="tab:orange")
    ax.axhline(a_sweep[0.0], ls="--", color="gray", lw=1, label="pure CE (alpha=0)")
    ax.set_title(f"alpha sweep (T={TEMPERATURE:.0f})")
    ax.set_xlabel("alpha (soft weight)")
    ax.set_ylabel("test accuracy")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out / "03_response_kd.png", dpi=110)
    plt.close(fig)


def main() -> None:
    quiet_warnings()
    out = ensure_output_dir()

    d = get_data()
    teacher = freeze_module(get_trained_teacher())  # 凍結した teacher を1度だけ用意

    head = _headline_comparison(d, teacher)
    t_sweep = _temperature_sweep(d, teacher)
    a_sweep = _alpha_sweep(d, teacher)

    _save_figure(out, head, t_sweep, a_sweep)
    print("[4] 図を保存:", (out / "03_response_kd.png").name)
    print("\n[まとめ]")
    print("  - 同じ小 student でも、teacher の soft target を混ぜると accuracy が上がる。")
    print("  - T と alpha は感度が高いハイパラ。スイープして最適点を探す。")
    print("  - 次は 04 で『出力だけでなく中間特徴も合わせる』特徴量蒸留を足す。")


if __name__ == "__main__":
    main()
