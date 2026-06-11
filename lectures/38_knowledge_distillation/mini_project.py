"""章末ミニプロジェクト: 蒸留ベンチ — 素の student vs 各種蒸留を「精度 × 圧縮率 × 速度」で比較。

統合課題:
  本モジュールの部品（温度付きレスポンス蒸留・特徴量蒸留・DeiT 流 distillation token）を
  1つにまとめ、「同じ小 student アーキを、素の学習 / 各種蒸留 で学習し、
  テスト accuracy・圧縮率・推論レイテンシ(p50/p99) を同一指標で比較する」ベンチを完成させる。
  これがこのモジュールの deliverable。

完成形がやること:
  1. クリーンなフルデータで teacher を学習（やや大きめ CNN）。
  2. 生徒用サブセット(少量・ラベル 40% ノイズ)で student を 4 通り学習:
       (A) baseline           … 蒸留なし（ノイジーハードラベルのみ）
       (B) response KD        … 温度付きソフトターゲット + ハード
       (C) response+feature   … さらに中間特徴を MSE で合わせる(FitNets)
       (D) DeiT distill token … 2ヘッドで hard distillation、推論時に平均
  3. accuracy / 圧縮率 / p50・p99 レイテンシを1つの表にまとめ、図と JSON に保存。
  4. 「蒸留ありは素の student をどれだけ上回るか（暗黙知の移転）」を定量化する。

実行:
  uv run python lectures/38_knowledge_distillation/mini_project.py
結果は outputs/38_knowledge_distillation/ に保存される。
"""

from __future__ import annotations

import json
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from kd_lab import (  # noqa: E402
    ALPHA,
    STUDENT_EPOCHS,
    STUDENT_LR,
    TEMPERATURE,
    DualHeadStudent,
    StudentCNN,
    accuracy,
    benchmark,
    count_params,
    ensure_output_dir,
    freeze_module,
    get_data,
    get_trained_teacher,
    quiet_warnings,
    set_seed,
    train_deit_style,
    train_feature_kd,
    train_response_kd,
    train_supervised,
)

SAMPLE = torch.randn(128, 1, 32, 32)  # 公平比較のため全モデルを同一バッチで計測


def _build_students(d: dict, teacher) -> dict[str, torch.nn.Module]:
    """4 つの学習戦略で student を作る（毎回 set_seed(0) で初期値を揃えて公平に）。"""
    students: dict[str, torch.nn.Module] = {}

    set_seed(0)
    s_base = StudentCNN()
    train_supervised(s_base, d["x_sub"], d["y_sub_noisy"], epochs=STUDENT_EPOCHS, lr=STUDENT_LR)
    students["(A) baseline"] = s_base

    set_seed(0)
    s_resp = StudentCNN()
    train_response_kd(s_resp, teacher, d["x_sub"], d["y_sub_noisy"],
                      temperature=TEMPERATURE, alpha=ALPHA,
                      epochs=STUDENT_EPOCHS, lr=STUDENT_LR)
    students["(B) response KD"] = s_resp

    set_seed(0)
    s_feat = StudentCNN()
    train_feature_kd(s_feat, teacher, d["x_sub"], d["y_sub_noisy"],
                     teacher_ch=64, student_ch=16,
                     temperature=TEMPERATURE, alpha=ALPHA,
                     epochs=STUDENT_EPOCHS, lr=STUDENT_LR)
    students["(C) response+feature"] = s_feat

    set_seed(0)
    s_deit = DualHeadStudent()
    train_deit_style(s_deit, teacher, d["x_sub"], d["y_sub_noisy"])
    students["(D) DeiT distill token"] = s_deit

    return students


def _evaluate(students: dict, teacher, d: dict) -> dict[str, dict]:
    """各 student を accuracy / 圧縮率 / レイテンシで評価して表(dict)にする。"""
    teacher_params = count_params(teacher)
    rows: dict[str, dict] = {}
    for name, model in students.items():
        b = benchmark(model, SAMPLE)
        params = count_params(model)
        rows[name] = {
            "accuracy": round(accuracy(model, d["x_te"], d["y_te"]), 4),
            "params": params,
            "compression": round(teacher_params / max(params, 1), 1),
            "p50_ms": round(b["p50"], 4),
            "p99_ms": round(b["p99"], 4),
        }
    return rows


def _print_table(teacher_acc: float, teacher_params: int, rows: dict) -> None:
    """teacher と各 student を1つの表で表示する。"""
    print("\n=== 蒸留ベンチ（同じ小 student・少量ノイジーデータ） ===")
    print(f"    teacher: acc={teacher_acc:.3f}  params={teacher_params:,}\n")
    print(f"    {'strategy':<24}{'acc':>7}{'params':>9}{'compr':>8}{'p50(ms)':>10}{'p99(ms)':>10}")
    base_acc = rows["(A) baseline"]["accuracy"]
    for name, m in rows.items():
        gain = m["accuracy"] - base_acc
        gain_s = f"  ({gain:+.3f})" if name != "(A) baseline" else ""
        print(f"    {name:<24}{m['accuracy']:>7.3f}{m['params']:>9,}"
              f"{m['compression']:>7.1f}x{m['p50_ms']:>10.3f}{m['p99_ms']:>10.3f}{gain_s}")


def _save_figure(out: pathlib.Path, teacher_acc: float, rows: dict) -> None:
    """accuracy（teacher 基準線つき）と p50 レイテンシを棒グラフで保存。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    names = list(rows.keys())
    short = [n.split(") ")[1] if ") " in n else n for n in names]

    ax = axes[0]
    accs = [rows[n]["accuracy"] for n in names]
    colors = ["tab:gray", "tab:blue", "tab:green", "tab:purple"]
    bars = ax.bar(short, accs, color=colors)
    ax.axhline(teacher_acc, ls="--", color="tab:red", lw=1.2, label=f"teacher ({teacher_acc:.2f})")
    ax.axhline(accs[0], ls=":", color="black", lw=1.0, label=f"baseline ({accs[0]:.2f})")
    ax.set_title("distillation strategies: test accuracy")
    ax.set_ylabel("test accuracy")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", labelrotation=15, labelsize=8)
    ax.legend(fontsize=8)
    for bar, a in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{a:.2f}", ha="center", va="bottom", fontsize=8)

    ax = axes[1]
    p50 = [rows[n]["p50_ms"] for n in names]
    ax.bar(short, p50, color="tab:orange")
    ax.set_title(f"inference latency p50 (ms, batch={SAMPLE.shape[0]})")
    ax.set_ylabel("p50 (ms)")
    ax.tick_params(axis="x", labelrotation=15, labelsize=8)

    fig.tight_layout()
    fig.savefig(out / "mini_distill_bench.png", dpi=110)
    plt.close(fig)


def main() -> None:
    quiet_warnings()
    out = ensure_output_dir()

    d = get_data()
    teacher = freeze_module(get_trained_teacher())
    teacher_acc = accuracy(teacher, d["x_te"], d["y_te"])
    teacher_params = count_params(teacher)

    students = _build_students(d, teacher)
    rows = _evaluate(students, teacher, d)
    _print_table(teacher_acc, teacher_params, rows)

    # 図と JSON に保存
    _save_figure(out, teacher_acc, rows)
    payload = {
        "teacher": {"accuracy": round(teacher_acc, 4), "params": teacher_params},
        "config": {"temperature": TEMPERATURE, "alpha": ALPHA,
                   "student_epochs": STUDENT_EPOCHS, "student_lr": STUDENT_LR},
        "students": rows,
    }
    (out / "mini_distill_bench.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\n[保存]", (out / "mini_distill_bench.png").name, "/",
          (out / "mini_distill_bench.json").name)

    # 意思決定の手引き
    best = max(rows.items(), key=lambda kv: kv[1]["accuracy"])
    base_acc = rows["(A) baseline"]["accuracy"]
    print("\n[意思決定の手引き]")
    print(f"  - 最良戦略: {best[0]}  acc={best[1]['accuracy']:.3f}"
          f"（baseline 比 {best[1]['accuracy'] - base_acc:+.3f}）")
    print("  - student のサイズ(params/レイテンシ)は学習戦略を変えても不変 — 蒸留は『同じ小ささで")
    print("    精度を底上げする』手法。まずレスポンス蒸留、足りなければ特徴量/DeiT を足す。")
    print("  - teacher は凍結し soft target だけ供給。T/alpha はスイープで詰める。")
    print("\n完了。精度×圧縮率×速度で蒸留戦略を選べるベンチが動きました。")


if __name__ == "__main__":
    main()
