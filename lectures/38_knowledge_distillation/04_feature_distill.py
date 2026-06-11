"""04: 特徴量ベース蒸留(FitNets) — forward hook + 射影層 + MSE、関係蒸留(RKD)の概観。

学ぶこと:
  - レスポンス蒸留は「出力ロジット」だけを合わせる。特徴量蒸留はさらに「中間特徴」を合わせ、
    teacher の内部表現まで student に写す（FitNets, hint learning）。
  - 実装の肝:
      * forward hook(module.register_forward_hook) で student/teacher の中間テンソルを取り出す。
      * student と teacher の特徴次元(チャネル数)は違うので射影層(1x1 Conv か Linear)を必ず挟む。
      * チャネルごとのスケール差を消すため正規化(F.normalize)してから MSE を取る。
      * フックは使い終わったら必ず remove()（付けっぱなしは事故の元）。
  - 関係ベース蒸留(RKD): 個々の特徴ではなく「サンプル間の距離/角度の関係」を合わせる。
    特徴次元が大きく違っても関係(距離行列)なら比較できる、という発想を概観する。
  - 評価: レスポンス蒸留のみ vs レスポンス+特徴量蒸留 でテスト accuracy を比較する。

実行:
  uv run python lectures/38_knowledge_distillation/04_feature_distill.py
結果は outputs/38_knowledge_distillation/ に保存される。
"""

from __future__ import annotations

import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from kd_lab import (  # noqa: E402
    ALPHA,
    STUDENT_EPOCHS,
    STUDENT_LR,
    TEMPERATURE,
    FeatureDistiller,
    StudentCNN,
    accuracy,
    ensure_output_dir,
    freeze_module,
    get_data,
    get_trained_teacher,
    quiet_warnings,
    set_seed,
    train_feature_kd,
    train_response_kd,
)


def _demo_hook_shapes(teacher, student, x: torch.Tensor) -> None:
    """forward hook が中間特徴を捕まえる様子と、次元の不一致を確認する。"""
    print("[1] forward hook で中間特徴を取り出す（次元は一致しない）")
    fd = FeatureDistiller(teacher.features, student.features, teacher_ch=64, student_ch=16)
    with torch.no_grad():
        teacher(x[:4])    # フックが teacher 特徴を保存
        student(x[:4])    # フックが student 特徴を保存
    t_shape = tuple(fd._t_feat.shape)   # noqa: SLF001  デモ目的で内部を覗く
    s_shape = tuple(fd._s_feat.shape)   # noqa: SLF001
    print(f"    teacher 特徴 shape = {t_shape}  (64 ch)")
    print(f"    student 特徴 shape = {s_shape}  (16 ch)")
    proj = fd.projector(fd._s_feat)     # noqa: SLF001  射影で 16ch→64ch に持ち上げる
    print(f"    射影後 student     = {tuple(proj.shape)}  ← 1x1 Conv で teacher の 64ch に合わせた")
    print(f"    特徴量蒸留 MSE     = {float(fd.loss()):.5f}  （正規化してから MSE）")
    fd.remove()  # 後始末
    print("    → 射影層が無いと次元が合わず MSE が取れない。正規化はスケール差の吸収に必須。\n")


def _demo_rkd_concept(teacher, x: torch.Tensor) -> None:
    """関係ベース蒸留(RKD)の発想: サンプル間の距離行列を「関係」として比較する。"""
    print("[2] 関係ベース蒸留(RKD) の発想 — サンプル間の距離行列")
    with torch.no_grad():
        feat = teacher.features(x[:6])
        emb = teacher.pool(feat).flatten(1)        # (6, 64) の埋め込み
    # ペア間ユークリッド距離を平均で正規化（RKD-Distance の核）。
    dist = torch.cdist(emb, emb)
    dmat = dist / dist[dist > 0].mean()
    print("    teacher 埋め込みの正規化距離行列(6x6, 抜粋):")
    for i in range(3):
        row = "  ".join(f"{float(dmat[i, j]):4.2f}" for j in range(6))
        print(f"      {row}")
    print("    → student にこの『距離行列』を真似させるのが RKD。特徴次元が違っても比較できる。")
    print("      (本スクリプトの学習では FitNets 型の特徴 MSE を使い、RKD は概念紹介に留める。)\n")


def _compare_variants(d: dict, teacher) -> dict[str, float]:
    """レスポンス蒸留のみ vs レスポンス+特徴量蒸留 を比較する。"""
    print("[3] レスポンス蒸留のみ vs レスポンス+特徴量蒸留")

    set_seed(0)
    s_resp = StudentCNN()
    train_response_kd(s_resp, teacher, d["x_sub"], d["y_sub_noisy"],
                      temperature=TEMPERATURE, alpha=ALPHA,
                      epochs=STUDENT_EPOCHS, lr=STUDENT_LR)
    acc_resp = accuracy(s_resp, d["x_te"], d["y_te"])

    set_seed(0)
    s_feat = StudentCNN()
    train_feature_kd(s_feat, teacher, d["x_sub"], d["y_sub_noisy"],
                     teacher_ch=64, student_ch=16,
                     temperature=TEMPERATURE, alpha=ALPHA,
                     epochs=STUDENT_EPOCHS, lr=STUDENT_LR)
    acc_feat = accuracy(s_feat, d["x_te"], d["y_te"])

    print(f"    (B) response KD            acc = {acc_resp:.3f}")
    print(f"    (C) response + feature KD  acc = {acc_feat:.3f}")
    print(f"    → 中間特徴も合わせると改善: {acc_feat - acc_resp:+.3f}"
          "（teacher の内部表現まで写した効果）\n")
    return {"response": acc_resp, "response+feature": acc_feat}


def _baseline_acc(d: dict) -> float:
    """比較用に素の student（蒸留なし）の accuracy も測る。"""
    from kd_lab import train_supervised  # noqa: PLC0415

    set_seed(0)
    s = StudentCNN()
    train_supervised(s, d["x_sub"], d["y_sub_noisy"], epochs=STUDENT_EPOCHS, lr=STUDENT_LR)
    return accuracy(s, d["x_te"], d["y_te"])


def _save_figure(out: pathlib.Path, baseline: float, variants: dict) -> None:
    """素 / レスポンス / レスポンス+特徴 の accuracy を棒グラフで保存。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    names = ["baseline", "response KD", "response+feature KD"]
    vals = [baseline, variants["response"], variants["response+feature"]]
    bars = ax.bar(names, vals, color=["tab:gray", "tab:blue", "tab:green"])
    ax.set_title("adding feature distillation on top of response KD")
    ax.set_ylabel("test accuracy")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", labelrotation=10)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "04_feature_distill.png", dpi=110)
    plt.close(fig)


def main() -> None:
    quiet_warnings()
    out = ensure_output_dir()

    d = get_data()
    teacher = freeze_module(get_trained_teacher())

    _demo_hook_shapes(teacher, StudentCNN(), d["x_sub"])
    _demo_rkd_concept(teacher, d["x_sub"])
    variants = _compare_variants(d, teacher)
    baseline = _baseline_acc(d)

    _save_figure(out, baseline, variants)
    print("[4] 図を保存:", (out / "04_feature_distill.png").name)
    print("\n[まとめ]")
    print("  - 特徴量蒸留 = forward hook で中間特徴を取り、射影+正規化+MSE で合わせる。")
    print("  - 出力だけのレスポンス蒸留に中間特徴を足すと、内部表現まで写せて精度が伸びやすい。")
    print("  - 関係ベース(RKD)は『サンプル間の距離/角度』を合わせる別系統。次は 05 で DeiT を学ぶ。")


if __name__ == "__main__":
    main()
