"""03: セグメンテーション評価指標（pixel acc / IoU / mIoU / Dice / FWIoU）を自作し torchmetrics と一致確認。

評価の土台は「画素混同行列」だけ。GT と予測のクラスマップから KxK の行列 cm を作れば、
cm[g, p] = 「正解 g・予測 p の画素数」になり、全指標がそこから出る:
  - TP_c = cm[c, c]、FP_c = Σ_g cm[g, c] - TP_c、FN_c = Σ_p cm[c, p] - TP_c
  - per-class IoU = TP / (TP + FP + FN)          … クラス c の重なり具合
  - mIoU         = per-class IoU のクラス平均      … セグメの主指標
  - Dice(=F1)    = 2TP / (2TP + FP + FN)          … 医用で頻出
  - pixel acc    = ΣTP / 全画素                    … 直感的だが不均衡に弱い
  - FWIoU        = Σ (クラスのGT画素割合 × IoU)     … 頻度で重み付けした IoU

未出現クラスの扱いが要点:
  GT にも予測にも一度も出ないクラスは IoU/Dice が 0/0 で「未定義」。
  ベンチマーク慣行ではこれを NaN として平均から除外する（0 と数えると mIoU が不当に下がる）。
  torchmetrics は MeanIoU が -1.0 のセンチネル、DiceScore が NaN を返す。自作と突き合わせる。

実行: uv run python lectures/21_segmentation_intro/03_miou_dice_eval.py
出力: outputs/21_segmentation_intro/03_*.png / 03_miou_dice_eval.json
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from torchmetrics.segmentation import DiceScore, MeanIoU  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import seg_helpers as H  # noqa: E402


# =====================================================================
# 自作: 画素混同行列 → 各指標
# =====================================================================
def pixel_confusion_matrix(
    gt: np.ndarray, pred: np.ndarray, num_classes: int, ignore_index: int | None = None
) -> np.ndarray:
    """GT と予測のクラスマップから KxK の画素混同行列 cm[g, p] を作る。

    ignore_index を指定すると、その GT を持つ画素は集計から完全に除外する
    （“評価対象外”の領域。VOC の境界 255 などが典型）。
    np.bincount を使って二重ループ無しで一気に数える定石。
    """
    g = gt.ravel()
    p = pred.ravel()
    if ignore_index is not None:
        keep = g != ignore_index
        g, p = g[keep], p[keep]
    # (g, p) のペアを一意な整数 g*K + p に符号化し、bincount で頻度を数えて KxK に整形。
    idx = g * num_classes + p
    cm = np.bincount(idx, minlength=num_classes * num_classes)
    return cm.reshape(num_classes, num_classes).astype(np.int64)


def metrics_from_cm(cm: np.ndarray) -> dict:
    """混同行列から pixel acc / per-class IoU / mIoU / Dice / FWIoU を計算する。

    未出現クラス（GT にも予測にも無い = 行も列も全0）は IoU/Dice を NaN にし、
    平均では np.nanmean で除外する（ベンチマーク慣行）。
    """
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0) - tp  # 列方向の合計から TP を引く = そのクラスと誤予測した画素
    fn = cm.sum(axis=1) - tp  # 行方向の合計から TP を引く = そのクラスを取りこぼした画素
    total = cm.sum()

    present = (tp + fp + fn) > 0  # GT か予測のどちらかに登場したクラス
    iou = np.full(cm.shape[0], np.nan, dtype=np.float64)
    iou[present] = tp[present] / (tp[present] + fp[present] + fn[present])

    dice = np.full(cm.shape[0], np.nan, dtype=np.float64)
    dice[present] = 2 * tp[present] / (2 * tp[present] + fp[present] + fn[present])

    pixel_acc = float(tp.sum() / total) if total > 0 else float("nan")
    # FWIoU: 各クラスの GT 画素割合で IoU を重み付け（未出現クラスは割合0で寄与0）。
    gt_freq = cm.sum(axis=1) / total
    fwiou = float(np.nansum(gt_freq * iou))

    return {
        "per_class_iou": iou,
        "miou": float(np.nanmean(iou)),
        "per_class_dice": dice,
        "mean_dice": float(np.nanmean(dice)),
        "pixel_acc": pixel_acc,
        "fwiou": fwiou,
    }


# =====================================================================
# torchmetrics との一致確認
# =====================================================================
def torchmetrics_reference(gt: np.ndarray, pred: np.ndarray, num_classes: int) -> dict:
    """torchmetrics で per-class IoU / mIoU / Dice を計算する（データセット全体＝global 定義）。

    重要: torchmetrics の MeanIoU/DiceScore は既定で『画像ごとに計算して平均（samplewise）』する。
    一方、Cityscapes/ADE20K 等のベンチマーク mIoU は『全画素を1つの混同行列に貯めてから IoU』＝
    global 定義。両者は一般に一致しない（後述の比較で確認）。
    自作の混同行列は global なので、torchmetrics 側も「全画素を1枚に平坦化」して 1 サンプルとして
    渡すことで global に揃える。
    """
    P = torch.from_numpy(pred.ravel()).long().unsqueeze(0)  # (1, N)
    G = torch.from_numpy(gt.ravel()).long().unsqueeze(0)

    miou_pc = MeanIoU(num_classes=num_classes, include_background=True, per_class=True, input_format="index")
    miou_pc.update(P, G)
    iou_tm = miou_pc.compute().numpy().astype(np.float64)
    iou_tm[iou_tm < 0] = np.nan  # MeanIoU は未出現クラスを -1.0 で返す → NaN に正規化

    miou_scalar = MeanIoU(num_classes=num_classes, include_background=True, per_class=False, input_format="index")
    miou_scalar.update(P, G)

    dice_pc = DiceScore(num_classes=num_classes, include_background=True, average="none", input_format="index")
    dice_pc.update(P, G)
    dice_tm = dice_pc.compute().numpy().astype(np.float64)  # 未出現クラスは NaN

    dice_macro = DiceScore(num_classes=num_classes, include_background=True, average="macro", input_format="index")
    dice_macro.update(P, G)

    return {
        "per_class_iou": iou_tm,
        "miou": float(miou_scalar.compute()),
        "per_class_dice": dice_tm,
        "mean_dice": float(dice_macro.compute()),
    }


def show_aggregation_difference(gt: np.ndarray, pred: np.ndarray, num_classes: int) -> dict:
    """『画像ごと平均（samplewise）』と『全画素まとめ（global）』で mIoU が変わることを示す。

    同じ画素集合を上下2枚の“画像”に割って batch (2,H/2,W) として MeanIoU に渡すと、
    既定の samplewise では各画像の mIoU を平均する。これは global とは別物。
    """
    h = gt.shape[0]
    half = h // 2
    P = torch.stack([torch.from_numpy(pred[:half]).long(), torch.from_numpy(pred[half:]).long()])
    G = torch.stack([torch.from_numpy(gt[:half]).long(), torch.from_numpy(gt[half:]).long()])
    m = MeanIoU(num_classes=num_classes, include_background=True, per_class=False, input_format="index")
    m.update(P, G)
    per_image = float(m.compute())
    global_miou = metrics_from_cm(pixel_confusion_matrix(gt, pred, num_classes))["miou"]
    return {"samplewise_miou": round(per_image, 4), "global_miou": round(global_miou, 4)}


# =====================================================================
# 可視化
# =====================================================================
def save_confusion_and_iou(cm: np.ndarray, iou: np.ndarray, names: list[str], path: pathlib.Path) -> None:
    """混同行列ヒートマップ（行=GT, 列=予測）と per-class IoU 棒グラフを保存する。"""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    ax = axes[0]
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(names)), names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(names)), names, fontsize=8)
    ax.set_xlabel("predicted")
    ax.set_ylabel("ground truth")
    ax.set_title("pixel confusion matrix")
    for g in range(len(names)):
        for p in range(len(names)):
            ax.text(p, g, int(cm[g, p]), ha="center", va="center", fontsize=7,
                    color="white" if cm[g, p] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)

    ax2 = axes[1]
    x = np.arange(len(names))
    vals = np.nan_to_num(iou, nan=0.0)
    colors = ["#9ecae1" if np.isfinite(v) else "#eeeeee" for v in iou]
    ax2.bar(x, vals, color=colors)
    ax2.set_xticks(x, names, rotation=45, ha="right", fontsize=8)
    ax2.set_ylim(0, 1.05)
    ax2.set_title("per-class IoU (gray = absent/NaN, excluded from mIoU)")
    for xi, v in zip(x, iou):
        label = f"{v:.2f}" if np.isfinite(v) else "NaN"
        ax2.text(xi, (v if np.isfinite(v) else 0) + 0.02, label, ha="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _to_jsonable(arr: np.ndarray) -> list:
    """NaN を含む配列を JSON 可能な list（NaN は None）に変換する。"""
    return [None if not np.isfinite(v) else round(float(v), 6) for v in arr]


# =====================================================================
# メイン
# =====================================================================
def main() -> None:
    out = H.ensure_output_dir()
    names = H.TOY_CLASSES
    K = len(names)

    # 決定的な GT / 予測ペア（person=5 はどちらにも出さない＝未出現クラスの実験用）。
    gt, pred = H.make_toy_gt_pred()

    # --- 自作 ---
    cm = pixel_confusion_matrix(gt, pred, K)
    manual = metrics_from_cm(cm)

    # --- torchmetrics（global に揃えて）---
    tm = torchmetrics_reference(gt, pred, K)

    # --- 一致確認（NaN も含めて allclose, equal_nan=True）---
    iou_ok = np.allclose(manual["per_class_iou"], tm["per_class_iou"], atol=1e-6, equal_nan=True)
    dice_ok = np.allclose(manual["per_class_dice"], tm["per_class_dice"], atol=1e-6, equal_nan=True)
    miou_ok = abs(manual["miou"] - tm["miou"]) < 1e-6
    mdice_ok = abs(manual["mean_dice"] - tm["mean_dice"]) < 1e-6
    assert iou_ok and dice_ok and miou_ok and mdice_ok, "自作と torchmetrics が一致しません"

    print("=== 自作 vs torchmetrics（global 定義）===")
    print(f"  per-class IoU 一致: {iou_ok}   per-class Dice 一致: {dice_ok}")
    print(f"  mIoU     自作={manual['miou']:.4f}  tm={tm['miou']:.4f}  [{'OK' if miou_ok else 'NG'}]")
    print(f"  meanDice 自作={manual['mean_dice']:.4f}  tm={tm['mean_dice']:.4f}  [{'OK' if mdice_ok else 'NG'}]")
    print(f"  pixel acc={manual['pixel_acc']:.4f}   FWIoU={manual['fwiou']:.4f}")
    print("  per-class IoU:")
    for n, v in zip(names, manual["per_class_iou"]):
        tag = "  (未出現→NaN, mIoUから除外)" if not np.isfinite(v) else ""
        print(f"    {n:11s} {'NaN' if not np.isfinite(v) else f'{v:.3f}'}{tag}")

    # --- ignore_index の効果 ---
    # GT の左端 12 画素列を「評価対象外（255）」に塗り、無視した場合としない場合で比べる。
    gt_ign = gt.copy()
    gt_ign[:, :12] = 255  # 255 = 評価対象外（クラス範囲 0..K-1 の外）。必ず ignore_index で除外する。
    cm_ign = pixel_confusion_matrix(gt_ign, pred, K, ignore_index=255)
    m_ign = metrics_from_cm(cm_ign)
    print("\n=== ignore_index=255（評価対象外領域）===")
    print(f"  無視あり: pixel acc={m_ign['pixel_acc']:.4f}  mIoU={m_ign['miou']:.4f}")
    print("  無視した画素は分母から消えるので、その領域の誤りは指標に影響しない。")

    # --- 集計のクセ: samplewise vs global ---
    agg = show_aggregation_difference(gt, pred, K)
    print("\n=== mIoU の集計方法による差 ===")
    print(f"  画像ごと平均(samplewise, torchmetrics既定)={agg['samplewise_miou']}")
    print(f"  全画素まとめ(global, ベンチマーク慣行)    ={agg['global_miou']}")
    print("  論文の mIoU は global。torchmetrics を画像ごとに回して平均すると値が変わる。")

    # --- 保存 ---
    color_gt = H.colorize(gt, H.voc_colormap())
    color_pred = H.colorize(pred, H.voc_colormap())
    H.save_panel([color_gt, color_pred], ["toy GT", "toy prediction"],
                 out / "03_gt_vs_pred.png", ncol=2)
    save_confusion_and_iou(cm, manual["per_class_iou"], names, out / "03_confusion_iou.png")

    summary = {
        "classes": names,
        "manual": {
            "pixel_acc": round(manual["pixel_acc"], 6),
            "miou": round(manual["miou"], 6),
            "mean_dice": round(manual["mean_dice"], 6),
            "fwiou": round(manual["fwiou"], 6),
            "per_class_iou": _to_jsonable(manual["per_class_iou"]),
            "per_class_dice": _to_jsonable(manual["per_class_dice"]),
        },
        "torchmetrics": {
            "miou": round(tm["miou"], 6),
            "mean_dice": round(tm["mean_dice"], 6),
            "per_class_iou": _to_jsonable(tm["per_class_iou"]),
        },
        "match": {"iou": bool(iou_ok), "dice": bool(dice_ok), "miou": bool(miou_ok)},
        "ignore_index_demo": {"pixel_acc": round(m_ign["pixel_acc"], 6), "miou": round(m_ign["miou"], 6)},
        "aggregation": agg,
    }
    (out / "03_miou_dice_eval.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nsaved: {out / '03_gt_vs_pred.png'}, {out / '03_confusion_iou.png'}, {out / '03_miou_dice_eval.json'}")


if __name__ == "__main__":
    main()
