"""章末ミニプロジェクト: 2台の検出器を「自作 mAP 評価器」で比較し、運用しきい値まで決める。

これまでの 01〜03 で組み立てた部品（IoU → confidence 降順マッチング → PR → COCO 101点 AP）
を統合し、「現場で評価レポートを書く」ところまで一気通貫でやり切る総合課題。
やることは大きく4つ:

  1) GT（正解）を1セットだけ合成し、その同じ GT に対して品質の異なる2台の検出器
     （strong / weak）の予測を合成する。同じ GT で比べることが検出器比較の大前提。
  2) numpy だけの自作評価器で、各検出器の mAP@0.5 / mAP@0.75 / mAP@[.5:.95] と
     カテゴリ別 AP を計算する（03 のロジックを自己完結で再実装）。
  3) その自作値が pycocotools COCOeval と小数点以下まで一致することを検証する
     （自作が正しいという保証＝レポートの信頼性）。
  4) PR 機構を「運用」に落とす: strong 検出器について confidence しきい値を掃引し、
     F1 を最大化するしきい値を選ぶ（PR 曲線は評価だけでなく閾値決定の道具でもある）。

成果物（lectures/19_detection_map_from_scratch/outputs/ に保存）:
  - mini_detector_report.png : 4枚パネル（PR比較 / mAP-IoU曲線 / カテゴリ別AP50 / F1掃引）
  - mini_project_report.json : 両検出器の全指標・pycocotools 一致差分・推奨しきい値

実行: uv run python lectures/19_detection_map_from_scratch/mini_project.py

注意: digit始まりの 01〜03 は import 不可なので、評価ロジックは本ファイルで自己完結させ、
      合成データ生成・COCO変換・IoU・マッチングエンジンだけ det_helpers.py を借用する。
"""

from __future__ import annotations

import contextlib
import io
import json
import pathlib
import sys

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from det_helpers import (  # noqa: E402
    CATEGORIES,
    IMAGE_H,
    IMAGE_W,
    box_iou_numpy,
    match_image,
    output_dir,
    to_coco_dt,
    to_coco_gt,
)

IOU_THRS = np.linspace(0.5, 0.95, 10)  # COCO 既定: 0.50,0.55,...,0.95
REC_THRS = np.linspace(0.0, 1.0, 101)  # COCO 既定: 0,0.01,...,1.0
CAT_IDS = [c["id"] for c in CATEGORIES]


# =====================================================================
# 1. 同一 GT に対して品質の異なる2台の検出器の予測を合成する
# =====================================================================

def make_ground_truth(n_images: int = 16, seed: int = 7) -> list[dict]:
    """GT（正解）だけを合成して返す。両検出器はこの同じ GT を共有する（公平な比較の前提）。"""
    rng = np.random.default_rng(seed)
    gts: list[dict] = []
    for _ in range(n_images):
        n_obj = int(rng.integers(4, 9))
        boxes, labels = [], []
        for _ in range(n_obj):
            bw = float(rng.uniform(50, 130))
            bh = float(rng.uniform(50, 130))
            x1 = float(rng.uniform(0, IMAGE_W - bw))
            y1 = float(rng.uniform(0, IMAGE_H - bh))
            boxes.append([x1, y1, x1 + bw, y1 + bh])
            labels.append(int(rng.choice(CAT_IDS)))
        gts.append(
            {
                "gt_boxes": np.array(boxes, dtype=np.float64).reshape(-1, 4),
                "gt_labels": np.array(labels, dtype=np.int64).reshape(-1),
            }
        )
    return gts


def make_predictions(
    gts: list[dict],
    seed: int,
    *,
    miss_rate: float,
    shift_frac: float,
    scale_lo: float,
    scale_hi: float,
    score_lo: float,
    n_fp: tuple[int, int],
) -> list[dict]:
    """GT を基に1台分の予測を合成し、gt と pred を同居させた dataset 形式で返す。

    品質パラメータ:
      miss_rate  : GT を取りこぼす確率（高いほど recall が下がる＝FN 増）。
      shift_frac : 予測ボックスの中心ずらし幅の割合（大きいほど IoU が下がる＝位置が甘い）。
      scale_*    : 予測ボックスのサイズ倍率の範囲（1.0 から離れるほど IoU が下がる）。
      score_lo   : 当たり予測のスコア下限（高いほど自信のある予測になる）。
      n_fp       : 各画像にばらまく無関係な誤検出(FP)の個数レンジ。

    strong は miss/shift/scale が小さく score が高い。weak はその逆。同じ GT を共有する。
    """
    rng = np.random.default_rng(seed)
    out: list[dict] = []
    for g in gts:
        pred_boxes, pred_scores, pred_labels = [], [], []
        for box, label in zip(g["gt_boxes"], g["gt_labels"]):
            if rng.random() < miss_rate:
                continue  # 検出漏れ（FN）
            bw, bh = box[2] - box[0], box[3] - box[1]
            cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            dx = rng.uniform(-shift_frac, shift_frac) * bw
            dy = rng.uniform(-shift_frac, shift_frac) * bh
            nw = bw * rng.uniform(scale_lo, scale_hi)
            nh = bh * rng.uniform(scale_lo, scale_hi)
            px1 = float(np.clip(cx + dx - nw / 2, 0, IMAGE_W - 1))
            py1 = float(np.clip(cy + dy - nh / 2, 0, IMAGE_H - 1))
            px2 = float(np.clip(cx + dx + nw / 2, 1, IMAGE_W))
            py2 = float(np.clip(cy + dy + nh / 2, 1, IMAGE_H))
            pred_boxes.append([px1, py1, px2, py2])
            pred_scores.append(float(rng.uniform(score_lo, 0.99)))
            pred_labels.append(int(label))
        # 無関係な誤検出(FP)。スコアは低めだが当たり予測と値域が重なる
        for _ in range(int(rng.integers(n_fp[0], n_fp[1] + 1))):
            bw = float(rng.uniform(40, 110))
            bh = float(rng.uniform(40, 110))
            x1 = float(rng.uniform(0, IMAGE_W - bw))
            y1 = float(rng.uniform(0, IMAGE_H - bh))
            pred_boxes.append([x1, y1, x1 + bw, y1 + bh])
            pred_scores.append(float(rng.uniform(0.05, 0.55)))
            pred_labels.append(int(rng.choice(CAT_IDS)))
        out.append(
            {
                "gt_boxes": g["gt_boxes"],
                "gt_labels": g["gt_labels"],
                "pred_boxes": np.array(pred_boxes, dtype=np.float64).reshape(-1, 4),
                "pred_scores": np.array(pred_scores, dtype=np.float64).reshape(-1),
                "pred_labels": np.array(pred_labels, dtype=np.int64).reshape(-1),
            }
        )
    return out


# =====================================================================
# 2. numpy だけの自作評価器（03 のロジックを自己完結で再実装）
# =====================================================================

def coco101_ap(tp_sorted: np.ndarray, n_gt: int) -> float:
    """スコア降順 TP 列から COCO 101点 AP を計算（pycocotools.accumulate と同手順）。"""
    if n_gt == 0:
        return float("nan")  # GT 無しカテゴリは平均から除外
    tp_cum = np.cumsum(tp_sorted)
    fp_cum = np.cumsum(1.0 - tp_sorted)
    recall = tp_cum / n_gt
    precision = tp_cum / (tp_cum + fp_cum + np.spacing(1))
    precision = np.maximum.accumulate(precision[::-1])[::-1]  # 右から単調化
    inds = np.searchsorted(recall, REC_THRS, side="left")
    q = np.zeros(101)
    for ri, pi in enumerate(inds):
        if pi < len(precision):
            q[ri] = precision[pi]
    return float(q.mean())


def collect_tp(model_ds: list[dict], category_id: int, iou_thr: float) -> tuple[np.ndarray, int]:
    """1カテゴリ・1閾値で、画像横断のスコア降順 TP 列と GT 総数を返す（PR と AP の素材）。"""
    scores_all, tp_all, n_gt_total = [], [], 0
    for d in model_ds:
        gm = d["gt_labels"] == category_id
        dm = d["pred_labels"] == category_id
        tp, sc, n_gt = match_image(d["pred_boxes"][dm], d["pred_scores"][dm], d["gt_boxes"][gm], iou_thr)
        scores_all.append(sc)
        tp_all.append(tp)
        n_gt_total += n_gt
    scores_all = np.concatenate(scores_all) if scores_all else np.zeros(0)
    tp_all = np.concatenate(tp_all) if tp_all else np.zeros(0)
    order = np.argsort(-scores_all, kind="stable")  # 画像横断で改めて降順
    return tp_all[order], n_gt_total


def ap_table(model_ds: list[dict]) -> np.ndarray:
    """AP[カテゴリ, IoU閾値] の表を埋めて返す。"""
    ap = np.full((len(CAT_IDS), len(IOU_THRS)), np.nan)
    for ci, c in enumerate(CAT_IDS):
        for ti, thr in enumerate(IOU_THRS):
            tp_sorted, n_gt = collect_tp(model_ds, c, thr)
            ap[ci, ti] = coco101_ap(tp_sorted, n_gt)
    return ap


def pr_curve(model_ds: list[dict], category_id: int, iou_thr: float) -> tuple[np.ndarray, np.ndarray]:
    """描画用に PR 列（画像横断・スコア降順）を返す。"""
    scores_all, tp_all, n_gt_total = [], [], 0
    for d in model_ds:
        gm = d["gt_labels"] == category_id
        dm = d["pred_labels"] == category_id
        tp, sc, n_gt = match_image(d["pred_boxes"][dm], d["pred_scores"][dm], d["gt_boxes"][gm], iou_thr)
        scores_all.append(sc)
        tp_all.append(tp)
        n_gt_total += n_gt
    scores_all = np.concatenate(scores_all)
    tp_sorted = np.concatenate(tp_all)[np.argsort(-scores_all, kind="stable")]
    tp_cum = np.cumsum(tp_sorted)
    fp_cum = np.cumsum(1.0 - tp_sorted)
    recall = tp_cum / max(n_gt_total, 1)
    precision = tp_cum / np.clip(tp_cum + fp_cum, 1e-12, None)
    return precision, recall


def confidence_sweep(model_ds: list[dict], iou_thr: float) -> dict:
    """confidence しきい値を掃引し、各しきい値での全カテゴリ合算 precision/recall/F1 を返す。

    PR 曲線は『評価』だけでなく『運用しきい値の決定』にも使える。スコアがしきい値以上の
    予測だけ採用したとき TP/FP/FN がどう動くかを掃引し、F1 が最大のしきい値を推奨値とする。
    """
    # 全カテゴリ・全画像でマッチングし、(score, tp) の組と GT 総数を集める
    rows, n_gt_total = [], 0
    for d in model_ds:
        for c in CAT_IDS:
            gm = d["gt_labels"] == c
            dm = d["pred_labels"] == c
            tp, sc, n_gt = match_image(d["pred_boxes"][dm], d["pred_scores"][dm], d["gt_boxes"][gm], iou_thr)
            n_gt_total += n_gt
            for s, t in zip(sc, tp):
                rows.append((float(s), float(t)))
    rows = np.array(rows, dtype=np.float64).reshape(-1, 2)
    thrs = np.linspace(0.05, 0.95, 19)
    prec, rec, f1 = [], [], []
    for t in thrs:
        keep = rows[:, 0] >= t
        tp = float(rows[keep, 1].sum())
        fp = float(keep.sum() - tp)
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / n_gt_total if n_gt_total > 0 else 0.0
        prec.append(p)
        rec.append(r)
        f1.append(2 * p * r / (p + r) if (p + r) > 0 else 0.0)
    best = int(np.argmax(f1))
    return {
        "thresholds": thrs.tolist(),
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "best_threshold": float(thrs[best]),
        "best_f1": float(f1[best]),
    }


# =====================================================================
# 3. pycocotools との検算
# =====================================================================

def eval_pycocotools(model_ds: list[dict]) -> np.ndarray:
    """COCOeval(iouType='bbox') を回し 12個の stats を返す（自作値の検算用）。"""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    with contextlib.redirect_stdout(io.StringIO()):
        cg = COCO()
        cg.dataset = to_coco_gt(model_ds)
        cg.createIndex()
        cd = cg.loadRes(to_coco_dt(model_ds))
        ev = COCOeval(cg, cd, iouType="bbox")
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    return np.array(ev.stats, dtype=np.float64)


def summarize_model(name: str, model_ds: list[dict]) -> dict:
    """1台分の自作評価＋pycocotools 検算をまとめて返す。"""
    ap = ap_table(model_ds)
    map_50 = float(np.nanmean(ap[:, 0]))
    map_75 = float(np.nanmean(ap[:, 5]))
    map_5095 = float(np.nanmean(ap))
    stats = eval_pycocotools(model_ds)
    diff = max(abs(map_5095 - stats[0]), abs(map_50 - stats[1]), abs(map_75 - stats[2]))
    assert diff < 1e-6, f"{name}: 自作と pycocotools が不一致 (max diff {diff:.2e})"
    return {
        "name": name,
        "ap_table": ap,  # numpy のまま（JSON 化は呼び出し側で）
        "map@0.5": map_50,
        "map@0.75": map_75,
        "map@[.5:.95]": map_5095,
        "per_cat_ap50": {c["name"]: float(ap[ci, 0]) for ci, c in enumerate(CATEGORIES)},
        "pycoco_diff": float(diff),
        "pycoco_AR@100": float(stats[8]),
    }


# =====================================================================
# 4. レポート（図＋JSON）
# =====================================================================

def make_report_figure(strong_ds, weak_ds, strong, weak, sweep, out: pathlib.Path) -> None:
    """4枚パネルの評価レポートを1枚の PNG にまとめる。"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # (1) PR 曲線の比較（circle, IoU=0.5）。strong が右上＝高精度高再現
    ax = axes[0, 0]
    for ds, lab, color in [(strong_ds, "strong", "C0"), (weak_ds, "weak", "C3")]:
        p, r = pr_curve(ds, category_id=1, iou_thr=0.5)
        ax.plot(r, p, color=color, lw=2.0, label=lab)
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_title("PR curve (cat=circle, IoU=0.5)")
    ax.legend(loc="lower left", fontsize=9)

    # (2) mAP vs IoU 閾値。両者とも右肩下がり、strong が常に上
    ax = axes[0, 1]
    for s, color in [(strong, "C0"), (weak, "C3")]:
        ax.plot(IOU_THRS, np.nanmean(s["ap_table"], axis=0), "o-", color=color, label=s["name"])
    ax.set_xlabel("IoU threshold")
    ax.set_ylabel("mAP (mean over categories)")
    ax.set_ylim(0, 1.02)
    ax.set_title("mAP drops as IoU gets stricter")
    ax.legend(fontsize=9)

    # (3) カテゴリ別 AP50 の棒比較
    ax = axes[1, 0]
    names = [c["name"] for c in CATEGORIES]
    x = np.arange(len(names))
    ax.bar(x - 0.2, [strong["per_cat_ap50"][n] for n in names], 0.4, label="strong", color="C0")
    ax.bar(x + 0.2, [weak["per_cat_ap50"][n] for n in names], 0.4, label="weak", color="C3")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("AP50")
    ax.set_title("per-category AP50")
    ax.legend(fontsize=9)

    # (4) confidence しきい値の掃引（strong）。F1 最大点に縦線
    ax = axes[1, 1]
    thrs = sweep["thresholds"]
    ax.plot(thrs, sweep["precision"], "-", color="C0", label="precision")
    ax.plot(thrs, sweep["recall"], "-", color="C2", label="recall")
    ax.plot(thrs, sweep["f1"], "-", color="C1", lw=2.2, label="F1")
    ax.axvline(sweep["best_threshold"], color="k", ls="--", lw=1,
               label=f"best thr={sweep['best_threshold']:.2f}")
    ax.set_xlabel("confidence threshold")
    ax.set_ylabel("score")
    ax.set_ylim(0, 1.02)
    ax.set_title("strong: F1 vs confidence threshold (IoU=0.5)")
    ax.legend(fontsize=8, loc="lower center")

    fig.suptitle("Mini project: from-scratch mAP detector report", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out / "mini_detector_report.png", dpi=120)
    plt.close(fig)


def main() -> None:
    out = output_dir()

    # 1) 同一 GT に対して strong / weak の予測を合成
    gts = make_ground_truth(n_images=16, seed=7)
    strong_ds = make_predictions(
        gts, seed=11, miss_rate=0.05, shift_frac=0.06, scale_lo=0.92, scale_hi=1.08,
        score_lo=0.55, n_fp=(0, 2),
    )
    weak_ds = make_predictions(
        gts, seed=12, miss_rate=0.30, shift_frac=0.20, scale_lo=0.70, scale_hi=1.30,
        score_lo=0.20, n_fp=(2, 5),
    )

    # 2)+3) 自作評価＋pycocotools 検算
    strong = summarize_model("strong", strong_ds)
    weak = summarize_model("weak", weak_ds)
    print("=== 検出器比較（自作 mAP、pycocotools と一致検証済み）===")
    for m in (strong, weak):
        print(f"  [{m['name']:6s}] mAP@0.5={m['map@0.5']:.4f}  "
              f"mAP@0.75={m['map@0.75']:.4f}  mAP@[.5:.95]={m['map@[.5:.95]']:.4f}  "
              f"(pycoco 最大差 {m['pycoco_diff']:.1e})")
    winner = "strong" if strong["map@[.5:.95]"] > weak["map@[.5:.95]"] else "weak"
    print(f"  → mAP@[.5:.95] が高いのは『{winner}』。strong は位置精度が高く誤検出が少ない。")

    # 4) strong の運用しきい値を F1 最大で決める
    sweep = confidence_sweep(strong_ds, iou_thr=0.5)
    print("\n=== strong の confidence しきい値選び（IoU=0.5, F1 最大）===")
    print(f"  推奨しきい値 = {sweep['best_threshold']:.2f}  （そのとき F1 = {sweep['best_f1']:.3f}）")
    print("  注: mAP はしきい値非依存の総合力。運用では F1 等が最大の1点を選んで採用する。")

    # レポート図＋JSON
    make_report_figure(strong_ds, weak_ds, strong, weak, sweep, out)
    report = {
        "models": {
            m["name"]: {
                "map@0.5": m["map@0.5"],
                "map@0.75": m["map@0.75"],
                "map@[.5:.95]": m["map@[.5:.95]"],
                "per_cat_ap50": m["per_cat_ap50"],
                "pycoco_max_abs_diff": m["pycoco_diff"],
                "pycoco_AR@100": m["pycoco_AR@100"],
            }
            for m in (strong, weak)
        },
        "winner_by_map@[.5:.95]": winner,
        "strong_confidence_sweep": sweep,
    }
    (out / "mini_project_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print("\n完了。mini_detector_report.png / mini_project_report.json を保存しました。")


if __name__ == "__main__":
    main()
