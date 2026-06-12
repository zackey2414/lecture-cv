"""第21回 章末ミニプロジェクト — 「セグメンテーション評価ベンチマーク」を一枚にまとめる総合課題。

この回で学んだ要素（クラスマップ → 画素混同行列 → pixel acc / mIoU / Dice / FWIoU、
未出現クラスの NaN 除外、global と samplewise の集計差、ignore_index）を1つに統合し、
「複数画像のデータセットを正しく評価するレポート」を生成する。
さらに 01〜03 には無い“マスター要素”を3つ足してある:

  (1) データセット集計（global）: 複数画像の画素を1つの混同行列に積み上げてから指標を出す。
                                  これがベンチマーク（Cityscapes/ADE20K/VOC）の正準な mIoU。
  (2) 指標の落とし穴（不均衡）  : 「強い予測器」と「多数派クラスだけを返す弱い予測器」を比較。
                                  弱い方は pixel accuracy が一見そこそこ高いのに mIoU は壊滅
                                  → accuracy だけで判断する危うさを数値で見せる。
  (3) しきい値最適化（マスク）  : 1つの前景クラスを2値マスク化し、確率予測のしきい値を掃いて
                                  mask-IoU / Dice 曲線を描き、IoU 最大のしきい値を選ぶ。

実装方針:
  - digit 始まりの既存スクリプト（01_*.py 等）は import できないので、評価コアは本ファイル内に
    自己完結で書く（可視化・パレットなど共通部品だけ seg_helpers から借用する）。
  - 自作した numpy の global mIoU / per-class IoU は torchmetrics と突き合わせて一致を assert する。
  - 完全 CPU・合成データ・数秒で完走。matplotlib は Agg、図のタイトルは ASCII。
  - 末尾で（任意）実モデル LR-ASPP の推論パイプラインも試すが、重みDLに失敗しても exit 0。

実行: uv run python lectures/21_segmentation_intro/mini_project.py
結果（ダッシュボード画像・レポート JSON）は outputs/21_segmentation_intro/ に保存される。
"""

from __future__ import annotations

import json
import pathlib
import sys

import cv2
import numpy as np
import torch

import matplotlib

matplotlib.use("Agg")  # headless でも図を保存できるバックエンド（pyplot より前に固定）
import matplotlib.pyplot as plt  # noqa: E402

from torchmetrics.segmentation import MeanIoU  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import seg_helpers as H  # noqa: E402  共通の道具箱（パレット・colorize・図保存など）

# このミニプロジェクト専用の小さなクラス体系（person を“まれで難しい”クラスにして不均衡を作る）。
CLASSES = ["background", "sky", "road", "car", "person"]
BACKGROUND, SKY, ROAD, CAR, PERSON = 0, 1, 2, 3, 4
NUM_CLASSES = len(CLASSES)


# =====================================================================
# 評価コア（numpy だけ。03 の再実装だが import 不可なので自己完結で持つ）
# =====================================================================
def pixel_confusion_matrix(gt: np.ndarray, pred: np.ndarray, num_classes: int) -> np.ndarray:
    """(g,p) を一意な整数に符号化し bincount で KxK 混同行列を作る。"""
    idx = gt.ravel() * num_classes + pred.ravel()
    cm = np.bincount(idx, minlength=num_classes * num_classes)
    return cm.reshape(num_classes, num_classes).astype(np.int64)


def metrics_from_cm(cm: np.ndarray) -> dict:
    """混同行列から pixel acc / per-class IoU / mIoU / Dice / FWIoU をまとめて計算する。"""
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    total = cm.sum()
    present = (tp + fp + fn) > 0  # GT か予測のどちらかに登場したクラス

    iou = np.full(cm.shape[0], np.nan)
    iou[present] = tp[present] / (tp[present] + fp[present] + fn[present])
    dice = np.full(cm.shape[0], np.nan)
    dice[present] = 2 * tp[present] / (2 * tp[present] + fp[present] + fn[present])

    freq = cm.sum(axis=1) / total if total > 0 else np.zeros(cm.shape[0])
    return {
        "per_class_iou": iou,
        "miou": float(np.nanmean(iou)) if present.any() else float("nan"),
        "per_class_dice": dice,
        "mean_dice": float(np.nanmean(dice)) if present.any() else float("nan"),
        "pixel_acc": float(tp.sum() / total) if total > 0 else 0.0,
        "fwiou": float(np.nansum(freq * iou)),
    }


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    """2値マスク同士の IoU。両方空なら 1.0（完全一致とみなす）。"""
    a = a.astype(bool)
    b = b.astype(bool)
    union = float(np.logical_or(a, b).sum())
    if union == 0:
        return 1.0
    return float(np.logical_and(a, b).sum()) / union


# =====================================================================
# 合成データセット（決定的）: 各画像に GT・強い予測・弱い予測を作る
# =====================================================================
def make_scene(size: int, seed: int) -> np.ndarray:
    """1枚の GT クラスマップ (size,size) を作る。sky が支配的＝クラス不均衡をわざと作る。"""
    rng = np.random.RandomState(seed)
    gt = np.zeros((size, size), dtype=np.uint8)  # background
    sky_h = int(size * (0.45 + 0.08 * rng.rand()))   # 空: 上 ~45-53%（最頻クラス）
    road_h = int(size * (0.78 + 0.05 * rng.rand()))  # 道: 下 ~22%
    gt[:sky_h, :] = SKY
    gt[road_h:, :] = ROAD
    # car: 道の上の横長の矩形
    cx = rng.randint(int(size * 0.15), int(size * 0.65))
    cv2.rectangle(gt, (cx, road_h - 8), (cx + int(size * 0.20), road_h + 6), CAR, thickness=-1)
    # person: 中央帯に立つ細い矩形（小さく＝まれで当てにくいクラス）
    px = rng.randint(int(size * 0.25), int(size * 0.7))
    py0 = road_h - int(size * 0.22)
    cv2.rectangle(gt, (px, py0), (px + int(size * 0.06), road_h - 4), PERSON, thickness=-1)
    return gt.astype(np.int64)


def make_strong_pred(gt: np.ndarray, seed: int) -> np.ndarray:
    """“強いがやや不完全な”予測。境界を数px ずらし、person を一部取りこぼす（IoU < 1）。"""
    rng = np.random.RandomState(seed + 100)
    dy = int(rng.choice([-2, -1, 1, 2]))
    dx = int(rng.choice([-2, -1, 1, 2]))
    pred = np.roll(np.roll(gt, dy, axis=0), dx, axis=1)  # 2px 程度の境界ズレ
    # person は難しい想定: 40% を background に落とす
    coords = np.argwhere(gt == PERSON)
    if len(coords):
        k = max(1, int(len(coords) * 0.4))
        sel = coords[rng.choice(len(coords), k, replace=False)]
        pred[sel[:, 0], sel[:, 1]] = BACKGROUND
    return pred


def make_majority_pred(gt: np.ndarray, majority_class: int) -> np.ndarray:
    """多数派クラスだけを全画素に返す“弱い”ベースライン（accuracy の落とし穴を見せる用）。"""
    return np.full_like(gt, majority_class)


def build_dataset(n_images: int = 6, size: int = 100) -> dict:
    """決定的な (GT, 強い予測, 弱い予測) を n_images 枚ぶん作る。"""
    gts, strongs = [], []
    for i in range(n_images):
        gt = make_scene(size, seed=i)
        gts.append(gt)
        strongs.append(make_strong_pred(gt, seed=i))
    # データセット全体での最頻クラス（弱い予測はこれを常に返す）
    all_gt = np.concatenate([g.ravel() for g in gts])
    majority = int(np.bincount(all_gt, minlength=NUM_CLASSES).argmax())
    weaks = [make_majority_pred(g, majority) for g in gts]
    return {"gts": gts, "strongs": strongs, "weaks": weaks, "majority": majority}


# =====================================================================
# データセット集計（global）
# =====================================================================
def evaluate_dataset(gts: list, preds: list) -> tuple[np.ndarray, dict]:
    """全画像の画素を1つの混同行列に積んでから（=global）指標を計算する。"""
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for gt, pred in zip(gts, preds):
        cm += pixel_confusion_matrix(gt, pred, NUM_CLASSES)
    return cm, metrics_from_cm(cm)


def torchmetrics_global_check(gts: list, preds: list, manual: dict) -> bool:
    """全画素を1サンプルに平坦化して MeanIoU（=global）に渡し、自作 mIoU/IoU と一致を確認。"""
    flat_g = np.concatenate([g.ravel() for g in gts])
    flat_p = np.concatenate([p.ravel() for p in preds])
    G = torch.from_numpy(flat_g).long().unsqueeze(0)  # (1, N) ← 全画素を1サンプルに
    P = torch.from_numpy(flat_p).long().unsqueeze(0)
    pc = MeanIoU(num_classes=NUM_CLASSES, include_background=True, per_class=True, input_format="index")
    pc.update(P, G)
    iou_tm = pc.compute().numpy().astype(np.float64)
    iou_tm[iou_tm < 0] = np.nan  # MeanIoU は未出現クラスを -1 で返す → NaN に正規化
    sc = MeanIoU(num_classes=NUM_CLASSES, include_background=True, per_class=False, input_format="index")
    sc.update(P, G)
    iou_ok = np.allclose(manual["per_class_iou"], iou_tm, atol=1e-6, equal_nan=True)
    miou_ok = abs(manual["miou"] - float(sc.compute())) < 1e-6
    return bool(iou_ok and miou_ok)


# =====================================================================
# しきい値最適化（前景マスクの2値評価）
# =====================================================================
def threshold_sweep(gt: np.ndarray, fg_class: int, seed: int = 0) -> dict:
    """前景クラスの GT マスクに対し、ぼかし＋ノイズで作った確率予測のしきい値を掃く。

    mask-IoU / Dice が最大になるしきい値を選ぶ。確率→2値の境目を最適化する実務の縮図。
    """
    rng = np.random.RandomState(seed)
    fg = (gt == fg_class).astype(np.float32)
    prob = cv2.GaussianBlur(fg, ksize=(0, 0), sigmaX=3.0)  # 連続的な確率マップを模す
    if prob.max() > 0:
        prob = prob / prob.max()
    prob = np.clip(prob + rng.normal(0, 0.05, prob.shape), 0, 1)

    thresholds = np.linspace(0.1, 0.9, 17)
    ious, dices = [], []
    fg_bool = fg.astype(bool)
    for t in thresholds:
        m = prob >= t
        iou = mask_iou(m, fg_bool)
        ious.append(iou)
        dices.append(2 * iou / (1 + iou) if (1 + iou) > 0 else 0.0)
    best = int(np.argmax(ious))
    return {
        "thresholds": thresholds,
        "ious": np.array(ious),
        "dices": np.array(dices),
        "best_threshold": float(thresholds[best]),
        "best_iou": float(ious[best]),
        "best_dice": float(dices[best]),
    }


# =====================================================================
# （任意）実モデルの推論パイプライン demo（重みDLに失敗しても exit 0）
# =====================================================================
def try_real_model_demo(device: torch.device) -> dict:
    """LR-ASPP を合成シーンで推論し out['out']→interpolate→argmax の骨格を再確認する。

    ネット非接続などで重み取得に失敗しても例外を握りつぶし、スキップ理由を返す。
    """
    try:
        import torch.nn.functional as F

        image = H.make_photo_like_scene(size=256, seed=0)
        model, weights = H.load_torchvision_seg("lraspp", device)
        categories = weights.meta["categories"]
        w0, h0 = image.size
        x = weights.transforms()(image).unsqueeze(0).to(device)
        with torch.inference_mode():
            out = model(x)  # ★ dict。out['out'] が (1, 21, H, W) のロジット
        logits = F.interpolate(out["out"], size=(h0, w0), mode="bilinear", align_corners=False)
        pred = logits.argmax(dim=1)[0].cpu().numpy()
        ids, counts = np.unique(pred, return_counts=True)
        top = sorted(
            ({"name": categories[int(i)], "ratio": round(float(c) / pred.size, 4)} for i, c in zip(ids, counts)),
            key=lambda r: r["ratio"], reverse=True,
        )[:5]
        return {"ran": True, "top_classes": top}
    except Exception as e:  # noqa: BLE001  ネット不通でも教材は exit 0 を守る
        return {"ran": False, "reason": f"{type(e).__name__}: {e}"}


# =====================================================================
# 可視化（ダッシュボード1枚）
# =====================================================================
def save_dashboard(
    dataset: dict, strong_m: dict, weak_m: dict, sweep: dict, path: pathlib.Path
) -> None:
    """GT/予測の見本・per-class IoU 比較・accuracy の罠・しきい値曲線を1枚にまとめる。"""
    palette = H.voc_colormap()
    gt0, strong0, weak0 = dataset["gts"][0], dataset["strongs"][0], dataset["weaks"][0]

    fig, axes = plt.subplots(2, 3, figsize=(12.5, 8))

    # 上段: GT / strong / weak のクラスマップ（image 0）
    for ax, arr, title in zip(
        axes[0], [gt0, strong0, weak0],
        ["GT (image 0)", "strong pred", "weak = majority-class only"],
    ):
        ax.imshow(H.colorize(arr, palette))
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    # 下段左: per-class IoU （strong vs weak）
    ax = axes[1, 0]
    x = np.arange(NUM_CLASSES)
    si = np.nan_to_num(strong_m["per_class_iou"], nan=0.0)
    wi = np.nan_to_num(weak_m["per_class_iou"], nan=0.0)
    ax.bar(x - 0.2, si, width=0.4, label="strong", color="#3182bd")
    ax.bar(x + 0.2, wi, width=0.4, label="weak", color="#fdae6b")
    ax.set_xticks(x, CLASSES, rotation=45, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_title("per-class IoU (global)", fontsize=10)
    ax.legend(fontsize=8)

    # 下段中: accuracy の罠（pixel acc vs mIoU）
    ax = axes[1, 1]
    labels = ["pixel acc", "mIoU"]
    xb = np.arange(2)
    ax.bar(xb - 0.2, [strong_m["pixel_acc"], strong_m["miou"]], width=0.4, label="strong", color="#3182bd")
    ax.bar(xb + 0.2, [weak_m["pixel_acc"], weak_m["miou"]], width=0.4, label="weak", color="#fdae6b")
    ax.set_xticks(xb, labels, fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_title("accuracy trap: weak looks OK on pixel acc,\nbut mIoU exposes it", fontsize=9)
    ax.legend(fontsize=8)
    for xi, (sv, wv) in enumerate(zip(
        [strong_m["pixel_acc"], strong_m["miou"]], [weak_m["pixel_acc"], weak_m["miou"]])):
        ax.text(xi - 0.2, sv + 0.02, f"{sv:.2f}", ha="center", fontsize=7)
        ax.text(xi + 0.2, wv + 0.02, f"{wv:.2f}", ha="center", fontsize=7)

    # 下段右: しきい値 vs mask IoU / Dice
    ax = axes[1, 2]
    ax.plot(sweep["thresholds"], sweep["ious"], "-o", ms=3, label="mask IoU", color="#3182bd")
    ax.plot(sweep["thresholds"], sweep["dices"], "-s", ms=3, label="mask Dice", color="#31a354")
    ax.axvline(sweep["best_threshold"], color="#d62728", ls="--", lw=1,
               label=f"best thr={sweep['best_threshold']:.2f}")
    ax.set_xlabel("threshold")
    ax.set_ylim(0, 1.05)
    ax.set_title("car mask: threshold vs IoU/Dice", fontsize=10)
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _jsonable(arr: np.ndarray) -> list:
    """NaN を None にして JSON 化可能な list にする。"""
    return [None if not np.isfinite(v) else round(float(v), 6) for v in arr]


# =====================================================================
# メイン
# =====================================================================
def main() -> None:
    out = H.ensure_output_dir()
    device = H.pick_device()

    # 1) データセット作成（決定的）
    dataset = build_dataset(n_images=6, size=100)

    # 2) データセット集計（global）で強い予測・弱い予測を評価
    _, strong_m = evaluate_dataset(dataset["gts"], dataset["strongs"])
    _, weak_m = evaluate_dataset(dataset["gts"], dataset["weaks"])

    # 3) torchmetrics（global）との一致を確認（自作の正しさの担保）
    tm_ok = torchmetrics_global_check(dataset["gts"], dataset["strongs"], strong_m)
    assert tm_ok, "自作の global mIoU/IoU が torchmetrics と一致しません"

    # 4) しきい値最適化（car の前景マスク）
    sweep = threshold_sweep(dataset["gts"][0], fg_class=CAR, seed=0)

    # 5) （任意）実モデルの推論パイプライン demo
    real = try_real_model_demo(device)

    # --- ダッシュボード保存 ---
    save_dashboard(dataset, strong_m, weak_m, sweep, out / "mini_project_dashboard.png")

    # --- コンソール出力 ---
    print("=== セグメンテーション評価ベンチマーク（6枚・global 集計）===")
    print(f"device={device}  classes={CLASSES}  majority={CLASSES[dataset['majority']]}")
    print(f"\n[強い予測]  pixel_acc={strong_m['pixel_acc']:.4f}  mIoU={strong_m['miou']:.4f}  "
          f"meanDice={strong_m['mean_dice']:.4f}  FWIoU={strong_m['fwiou']:.4f}")
    print(f"[弱い予測]  pixel_acc={weak_m['pixel_acc']:.4f}  mIoU={weak_m['miou']:.4f}  "
          f"（多数派 {CLASSES[dataset['majority']]} だけを返す）")
    print("  → 弱い予測は pixel acc が一見そこそこ高いのに mIoU は壊滅。"
          "accuracy だけで選ぶと不均衡で誤る。")
    print(f"  torchmetrics(global) と自作の一致: {tm_ok}")

    print("\n[per-class IoU] (strong / weak)")
    for c, name in enumerate(CLASSES):
        s = strong_m["per_class_iou"][c]
        w = weak_m["per_class_iou"][c]
        sf = "NaN" if not np.isfinite(s) else f"{s:.3f}"
        wf = "NaN" if not np.isfinite(w) else f"{w:.3f}"
        print(f"    {name:11s} strong={sf:>6s}   weak={wf:>6s}")

    print(f"\n[car マスクのしきい値最適化] best_threshold={sweep['best_threshold']:.2f}  "
          f"IoU={sweep['best_iou']:.4f}  Dice={sweep['best_dice']:.4f}")

    if real["ran"]:
        print(f"\n[実モデル LR-ASPP demo] 主なクラス: "
              f"{', '.join(f'{r['name']}({r['ratio']})' for r in real['top_classes'])}")
    else:
        print(f"\n[実モデル LR-ASPP demo] スキップ（{real['reason']}）"
              " ← 重みDL不可でも評価本体は完走。")

    # --- レポート JSON ---
    report = {
        "device": str(device),
        "classes": CLASSES,
        "n_images": len(dataset["gts"]),
        "majority_class": CLASSES[dataset["majority"]],
        "strong": {
            "pixel_acc": round(strong_m["pixel_acc"], 6),
            "miou": round(strong_m["miou"], 6),
            "mean_dice": round(strong_m["mean_dice"], 6),
            "fwiou": round(strong_m["fwiou"], 6),
            "per_class_iou": _jsonable(strong_m["per_class_iou"]),
        },
        "weak_majority_baseline": {
            "pixel_acc": round(weak_m["pixel_acc"], 6),
            "miou": round(weak_m["miou"], 6),
            "per_class_iou": _jsonable(weak_m["per_class_iou"]),
        },
        "torchmetrics_global_match": tm_ok,
        "threshold_optimization": {
            "best_threshold": sweep["best_threshold"],
            "best_iou": round(sweep["best_iou"], 6),
            "best_dice": round(sweep["best_dice"], 6),
        },
        "real_model_demo": real,
    }
    (out / "mini_project.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nsaved: {out / 'mini_project_dashboard.png'}, {out / 'mini_project.json'}")


if __name__ == "__main__":
    main()
