"""03: 自作 mAP（mAP@0.5 / mAP@[.5:.95]）を組み上げ、pycocotools COCOeval と一致を検証する。

02 までで「1カテゴリ・1閾値の AP」を作れた。本スクリプトはそれを
  (a) 全カテゴリで平均 → mAP、
  (b) IoU 閾値 0.50:0.05:0.95 の10通りで平均 → mAP@[.5:.95]（COCO の主指標）、
へ拡張する。そして同じデータを pycocotools の COCOeval(iouType='bbox') に通し、
自作値と公式値が小数点以下まで一致することを assert で確かめる。「自作できる＝中身を
理解している」ことの最終確認であり、本講座の評価トラックの集大成にあたる。

学ぶこと:
  - カテゴリ×IoU閾値のループで AP を埋め、mAP@0.5 / mAP@0.75 / mAP@[.5:.95] を算出。
  - COCO の AP は 101点補間。自作の COCO-101 AP が pycocotools と一致することを検証。
  - COCO 形式の落とし穴: bbox は xywh、maxDets と areaRng の既定値が結果を左右する。
  - AP_S/M/L・AR（面積別 AP / Average Recall）は pycocotools の summarize から読む（概念紹介）。

実行: uv run python lectures/19_detection_map_from_scratch/03_map_vs_pycocotools.py
結果（一致レポートの図・JSON）は outputs/19_detection_map_from_scratch/ に保存。
初回は pycocotools のインデックス作成ログが出るが、ネット接続もモデルDLも不要。
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
    make_detection_dataset,
    match_image,
    output_dir,
    to_coco_dt,
    to_coco_gt,
)

# COCO の既定値に合わせる。IoU は 0.50,0.55,...,0.95 の10点、recall は 0,0.01,...,1.0 の101点。
IOU_THRS = np.linspace(0.5, 0.95, 10)
REC_THRS = np.linspace(0.0, 1.0, 101)


def coco101_ap(tp_sorted: np.ndarray, n_gt: int) -> float:
    """スコア降順に並んだ TP フラグ列から COCO 101点 AP を計算（pycocotools.accumulate と同手順）。"""
    if n_gt == 0:
        return float("nan")  # GT が無いカテゴリは平均から除外（COCO は -1 として無視する）
    tp_cum = np.cumsum(tp_sorted)
    fp_cum = np.cumsum(1.0 - tp_sorted)
    recall = tp_cum / n_gt
    precision = tp_cum / (tp_cum + fp_cum + np.spacing(1))  # COCO と同じく極小 eps を足す
    precision = np.maximum.accumulate(precision[::-1])[::-1]  # 右から単調化
    inds = np.searchsorted(recall, REC_THRS, side="left")  # 101個の recall 閾値の位置
    q = np.zeros(101)
    for ri, pi in enumerate(inds):
        if pi < len(precision):
            q[ri] = precision[pi]
    return float(q.mean())


def evaluate_scratch(dataset: list[dict]) -> np.ndarray:
    """自作評価器: AP[カテゴリ, IoU閾値] の表を埋めて返す。

    マッチングは画像ごと・閾値ごとに行い（GT の使用済み状態は画像内で完結）、
    PR の累積は画像をまたいでスコア降順に積む——この二段構えが COCO の評価と同じ構造。
    """
    cat_ids = [c["id"] for c in CATEGORIES]
    ap = np.full((len(cat_ids), len(IOU_THRS)), np.nan)
    for ci, c in enumerate(cat_ids):
        for ti, thr in enumerate(IOU_THRS):
            scores_all, tp_all, n_gt_total = [], [], 0
            for d in dataset:
                gm = d["gt_labels"] == c
                dm = d["pred_labels"] == c
                tp, sc, n_gt = match_image(
                    d["pred_boxes"][dm], d["pred_scores"][dm], d["gt_boxes"][gm], thr
                )
                scores_all.append(sc)
                tp_all.append(tp)
                n_gt_total += n_gt
            scores_all = np.concatenate(scores_all)
            tp_all = np.concatenate(tp_all)
            order = np.argsort(-scores_all, kind="stable")  # 画像横断のスコア降順
            ap[ci, ti] = coco101_ap(tp_all[order], n_gt_total)
    return ap


def evaluate_pycocotools(dataset: list[dict]) -> tuple[np.ndarray, dict]:
    """pycocotools COCOeval を回し、12個の標準統計（stats）を返す。

    stats の並びは summarize() と同じ:
      [0]=AP@[.5:.95] [1]=AP50 [2]=AP75 [3]=AP_S [4]=AP_M [5]=AP_L
      [6]=AR@1 [7]=AR@10 [8]=AR@100 [9]=AR_S [10]=AR_M [11]=AR_L
    """
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    gt_dict = to_coco_gt(dataset)
    dt_list = to_coco_dt(dataset)

    # pycocotools の冗長な print（インデックス作成・サマリ）を JSON 出力前に握り潰す
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        coco_gt = COCO()
        coco_gt.dataset = gt_dict
        coco_gt.createIndex()
        coco_dt = coco_gt.loadRes(dt_list)  # 予測リスト(xywh+score)を読み込む
        evaler = COCOeval(coco_gt, coco_dt, iouType="bbox")
        evaler.evaluate()  # 画像×カテゴリのマッチング
        evaler.accumulate()  # PR を積んで precision テンソルを作る
        evaler.summarize()  # 12個の stats を埋める
    summarize_log = buf.getvalue()
    return np.array(evaler.stats, dtype=np.float64), {"summarize_log": summarize_log}


def main() -> None:
    out = output_dir()
    dataset = make_detection_dataset(n_images=12, seed=0)

    # === 自作評価器 ============================================================
    ap = evaluate_scratch(dataset)
    map_5095 = float(np.nanmean(ap))  # 全カテゴリ・全閾値の平均 = COCO 主指標
    map_50 = float(np.nanmean(ap[:, 0]))  # IoU=0.50 のみ
    map_75 = float(np.nanmean(ap[:, 5]))  # IoU=0.75 のみ
    print("=== 自作 mAP（numpy で一から）===")
    print(f"  mAP@0.5      = {map_50:.4f}")
    print(f"  mAP@0.75     = {map_75:.4f}")
    print(f"  mAP@[.5:.95] = {map_5095:.4f}   ← IoU を厳しくするほど下がる")
    for ci, c in enumerate(CATEGORIES):
        print(f"    cat {c['id']} ({c['name']:8s}) AP50={ap[ci, 0]:.3f}  AP@[.5:.95]={np.nanmean(ap[ci]):.3f}")

    # === pycocotools と突き合わせ ==============================================
    stats, _ = evaluate_pycocotools(dataset)
    print("\n=== pycocotools COCOeval(iouType='bbox') ===")
    print(f"  AP@[.5:.95] = {stats[0]:.4f}   AP50 = {stats[1]:.4f}   AP75 = {stats[2]:.4f}")
    print(f"  面積別 AP_S/M/L = {stats[3]:.4f} / {stats[4]:.4f} / {stats[5]:.4f}"
          "   （小:<32^2, 中:32^2〜96^2, 大:>96^2 px）")
    print(f"  AR(maxDets 1/10/100) = {stats[6]:.4f} / {stats[7]:.4f} / {stats[8]:.4f}"
          "   （AR=各画像の上位N検出での平均再現率）")

    # 自作と公式が小数点以下まで一致することを検証（連続スコアなので丸め誤差レベルで一致する）
    assert np.isclose(map_5095, stats[0], atol=1e-6), (map_5095, stats[0])
    assert np.isclose(map_50, stats[1], atol=1e-6), (map_50, stats[1])
    assert np.isclose(map_75, stats[2], atol=1e-6), (map_75, stats[2])
    print("\n[検証OK] 自作 mAP@[.5:.95]/mAP@0.5/mAP@0.75 が pycocotools と一致 "
          f"(最大差 < 1e-6)。差: "
          f"{abs(map_5095 - stats[0]):.2e} / {abs(map_50 - stats[1]):.2e} / {abs(map_75 - stats[2]):.2e}")

    # === IoU 閾値ごとの mAP を可視化（厳しくするほど下がる曲線）====================
    map_per_thr = np.nanmean(ap, axis=0)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    axes[0].plot(IOU_THRS, map_per_thr, "o-", color="C0")
    axes[0].axhline(map_5095, color="C3", ls="--", lw=1, label=f"mAP@[.5:.95]={map_5095:.3f}")
    axes[0].set_xlabel("IoU threshold")
    axes[0].set_ylabel("mAP (mean over categories)")
    axes[0].set_title("mAP drops as IoU gets stricter")
    axes[0].set_ylim(0, 1.02)
    axes[0].legend(fontsize=8)

    # 自作 vs pycocotools の主要3指標を棒で重ねる（ぴったり重なる＝一致）
    labels = ["AP@[.5:.95]", "AP50", "AP75"]
    scratch_vals = [map_5095, map_50, map_75]
    coco_vals = [stats[0], stats[1], stats[2]]
    xpos = np.arange(3)
    axes[1].bar(xpos - 0.2, scratch_vals, width=0.4, label="scratch (numpy)", color="C0")
    axes[1].bar(xpos + 0.2, coco_vals, width=0.4, label="pycocotools", color="C1")
    axes[1].set_xticks(xpos)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylim(0, 1.02)
    axes[1].set_title("scratch == pycocotools")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "03_map_compare.png", dpi=120)
    plt.close(fig)

    # === JSON 保存 =============================================================
    summary = {
        "scratch": {"mAP@0.5": map_50, "mAP@0.75": map_75, "mAP@[.5:.95]": map_5095},
        "pycocotools": {
            "AP@[.5:.95]": float(stats[0]), "AP50": float(stats[1]), "AP75": float(stats[2]),
            "AP_S": float(stats[3]), "AP_M": float(stats[4]), "AP_L": float(stats[5]),
            "AR@1": float(stats[6]), "AR@10": float(stats[7]), "AR@100": float(stats[8]),
        },
        "max_abs_diff": {
            "mAP@[.5:.95]": float(abs(map_5095 - stats[0])),
            "mAP@0.5": float(abs(map_50 - stats[1])),
            "mAP@0.75": float(abs(map_75 - stats[2])),
        },
        "ap_table_per_category_per_iou": ap.tolist(),
    }
    (out / "03_map_vs_pycocotools.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\n完了。03_map_compare.png / 03_map_vs_pycocotools.json を保存しました。")


if __name__ == "__main__":
    main()
