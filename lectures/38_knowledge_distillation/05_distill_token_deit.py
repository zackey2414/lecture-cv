"""05: DeiT の distillation token — 専用ヘッドで teacher から hard distillation。

学ぶこと:
  - DeiT(Data-efficient image Transformer) は ViT に「distillation token」を1本足す。
      * class token   … 通常どおり正解ラベルを CE で学ぶ（head）
      * distill token … teacher(例 CNN)の予測を学ぶ専用トークン（head_dist）
    推論時は class ヘッドと distill ヘッドの出力を平均する。
  - DeiT は hard distillation（teacher の argmax を擬似ラベルにして CE）を既定にしている。
    soft(KL) より単純で、CNN teacher → ViT student で強く効くことが知られる。
  - timm は token distillation 学習をサポートし、`*_distilled` 系モデルは head と head_dist を
    持つ（eval 時に自動で平均）。ここでは構造を確認し、CPU では自前の小さな
    「2ヘッド student」で同じ仕組みをトイ再現する。

実行:
  uv run python lectures/38_knowledge_distillation/05_distill_token_deit.py
結果は outputs/38_knowledge_distillation/ に保存される。
"""

from __future__ import annotations

import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from kd_lab import (  # noqa: E402
    STUDENT_EPOCHS,
    STUDENT_LR,
    DualHeadStudent,
    StudentCNN,
    accuracy,
    ensure_output_dir,
    freeze_module,
    get_data,
    get_trained_teacher,
    quiet_warnings,
    set_seed,
    train_deit_style,
    train_supervised,
)


# === DeiT の構造を timm で確認（pretrained=False なのでダウンロード無し） ==========
def _inspect_timm_deit() -> None:
    """timm の deit_*_distilled に head と head_dist の2ヘッドがあることを確認する。"""
    print("[1] timm の DeiT distilled モデル構造（重みDLなし・構造だけ確認）")
    try:
        import timm  # noqa: PLC0415

        model = timm.create_model("deit_tiny_distilled_patch16_224",
                                  pretrained=False, num_classes=10)
        has = [a for a in ("cls_token", "dist_token", "head", "head_dist") if hasattr(model, a)]
        print(f"    保有する属性: {has}")
        model.eval()
        with torch.inference_mode():
            out = model(torch.randn(1, 3, 224, 224))
        print(f"    eval 推論の出力 shape = {tuple(out.shape)}  ← head と head_dist を平均した結果")
        print("    → cls token と distill token をそれぞれ別ヘッドへ通し、推論時に平均する。\n")
    except Exception as exc:  # noqa: BLE001
        print(f"    [info] timm 未導入/構築失敗のためスキップ（{type(exc).__name__}）。概念のみ。\n")


# === 自前の2ヘッド student（DeiT の distillation token をトイ再現） ===============
# DualHeadStudent / train_deit_style は kd_lab に置き、mini_project と共有する。


def _compare(d: dict, teacher) -> dict[str, float]:
    """単一ヘッド baseline vs DeiT 流2ヘッド student を比較する。"""
    print("[2] 単一ヘッド baseline vs DeiT 流 2ヘッド student（hard distillation）")

    set_seed(0)
    single = StudentCNN()
    train_supervised(single, d["x_sub"], d["y_sub_noisy"], epochs=STUDENT_EPOCHS, lr=STUDENT_LR)
    acc_single = accuracy(single, d["x_te"], d["y_te"])

    set_seed(0)
    deit = DualHeadStudent()
    train_deit_style(deit, teacher, d["x_sub"], d["y_sub_noisy"])
    acc_deit_avg = accuracy(deit, d["x_te"], d["y_te"])  # 平均ヘッドで評価

    # 各ヘッド単体の精度も見る（distill ヘッドはクリーンな teacher ラベルで学ぶので強い）。
    deit.eval()
    with torch.inference_mode():
        z = deit.embed(d["x_te"])
        acc_cls = float((deit.cls_head(z).argmax(1) == d["y_te"]).float().mean())
        acc_dist = float((deit.dist_head(z).argmax(1) == d["y_te"]).float().mean())

    print(f"    単一ヘッド baseline         acc = {acc_single:.3f}")
    print(f"    DeiT: class ヘッド単体       acc = {acc_cls:.3f}  ← ノイジー正解ラベルで学習")
    print(f"    DeiT: distill ヘッド単体     acc = {acc_dist:.3f}  ← teacher の argmax で学習(クリーン)")
    print(f"    DeiT: 2ヘッド平均(推論)      acc = {acc_deit_avg:.3f}  ← 推論時に両者を平均")
    print("    → distill ヘッドはクリーンな teacher ラベルで学ぶので強く、平均すると安定する。\n")
    return {"baseline": acc_single, "cls_head": acc_cls,
            "dist_head": acc_dist, "deit_avg": acc_deit_avg}


def _save_figure(out: pathlib.Path, res: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.4))
    names = ["baseline", "DeiT cls head", "DeiT dist head", "DeiT avg"]
    vals = [res["baseline"], res["cls_head"], res["dist_head"], res["deit_avg"]]
    colors = ["tab:gray", "tab:orange", "tab:green", "tab:blue"]
    bars = ax.bar(names, vals, color=colors)
    ax.set_title("DeiT-style distillation token (toy): cls vs dist vs average")
    ax.set_ylabel("test accuracy")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", labelrotation=10)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "05_deit_distill_token.png", dpi=110)
    plt.close(fig)


def main() -> None:
    quiet_warnings()
    out = ensure_output_dir()

    _inspect_timm_deit()

    d = get_data()
    teacher = freeze_module(get_trained_teacher())
    res = _compare(d, teacher)

    _save_figure(out, res)
    print("[3] 図を保存:", (out / "05_deit_distill_token.png").name)
    print("\n[まとめ]")
    print("  - DeiT は class token に加え distillation token を持ち、teacher から hard distillation。")
    print("  - 推論時は class ヘッドと distill ヘッドを平均する。")
    print("  - timm の *_distilled は head/head_dist の2ヘッドを持つ。本章の総合は mini_project へ。")


if __name__ == "__main__":
    main()
