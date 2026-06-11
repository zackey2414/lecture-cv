"""02: TP/FP の累積から PR 曲線を作り、AP を3つの補間方式（11点 / 全点 / COCO101点）で計算する。

01 で「各予測が TP か FP か」を決めた。次は、スコア閾値を高い方から下げていったときの
precision と recall の軌跡（PR 曲線）を描き、その“面積”を1つの数 AP にまとめる。
ここで重要なのは「AP の値は補間方式に依存する」こと。同じ PR 曲線でも、
PASCAL VOC 2007 の 11点・VOC2010+ の全点・COCO の 101点で少しずつ違う数字になる。
論文や他チームの mAP と突き合わせるとき、この“方式の違い”を知らないと値がズレて見える。

学ぶこと:
  - cumsum(tp)/cumsum(fp) → precision=TP/(TP+FP), recall=TP/n_gt の列を画像横断で構築。
  - 単調包絡（np.maximum.accumulate）で precision を右から非増加に均す＝補間の共通前処理。
  - 3つの AP: 11点(平均), 全点(monotone PR の面積), COCO 101点(平均)。値の差を実測する。
  - 「confidence 降順ソートを忘れる」と AP が壊れることを、わざと再現して確かめる。

実行: uv run python lectures/19_detection_map_from_scratch/02_pr_ap_interpolation.py
結果（PR 曲線の図・AP の JSON）は outputs/19_detection_map_from_scratch/ に保存。
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from det_helpers import make_detection_dataset, match_image, output_dir  # noqa: E402


def build_pr(dataset: list[dict], category_id: int, iou_thr: float) -> tuple[np.ndarray, np.ndarray]:
    """1カテゴリの全画像を貫いて TP/FP を集め、スコア降順に累積して PR 列を返す。

    ポイント: マッチング（TP/FP 判定）は画像ごとに行うが、PR 曲線は“画像をまたいだ”
    全予測をスコア降順に並べてから cumsum で積む。recall の分母 n_gt はカテゴリ全体の GT 数。
    """
    scores_all, tp_all = [], []
    n_gt_total = 0
    for d in dataset:
        gm = d["gt_labels"] == category_id
        dm = d["pred_labels"] == category_id
        tp, sc, n_gt = match_image(d["pred_boxes"][dm], d["pred_scores"][dm], d["gt_boxes"][gm], iou_thr)
        n_gt_total += n_gt
        scores_all.append(sc)
        tp_all.append(tp)
    scores_all = np.concatenate(scores_all)
    tp_all = np.concatenate(tp_all)

    order = np.argsort(-scores_all, kind="stable")  # 画像横断で改めてスコア降順に
    tp_sorted = tp_all[order]
    tp_cum = np.cumsum(tp_sorted)
    fp_cum = np.cumsum(1.0 - tp_sorted)
    recall = tp_cum / max(n_gt_total, 1)
    precision = tp_cum / np.clip(tp_cum + fp_cum, 1e-12, None)
    return precision, recall


def ap_11point(precision: np.ndarray, recall: np.ndarray) -> float:
    """PASCAL VOC 2007 方式: recall = 0,0.1,...,1.0 の11点で interpolated precision を平均。

    各 r について「recall >= r を満たす点の precision の最大値」を取り（=右側包絡）、
    満たす点が無ければ 0。その11個を平均する。古い検出論文でよく出る方式。
    """
    aps = []
    for r in np.linspace(0, 1, 11):
        mask = recall >= r
        p = precision[mask].max() if mask.any() else 0.0
        aps.append(p)
    return float(np.mean(aps))


def ap_all_point(precision: np.ndarray, recall: np.ndarray) -> float:
    """VOC2010+ 方式: precision を右から単調化し、recall の増分×precision を全点で積む。

    AP = Σ (recall_i − recall_{i−1}) × p_monotone_i。
    これは「単調化した PR 曲線の真下の面積」に等しい（11点のような粗い標本化をしない）。
    """
    # 端点 (recall=0, precision=1) と (recall=最終, precision=0) を足して面積を正しく閉じる
    mrec = np.concatenate([[0.0], recall, [recall[-1]]])
    mpre = np.concatenate([[1.0], precision, [0.0]])
    # 右から単調非増加に均す（recall を上げても precision は上がらない、という包絡）
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]
    idx = np.where(mrec[1:] != mrec[:-1])[0]  # recall が変化した位置だけ面積に寄与
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def ap_coco101(precision: np.ndarray, recall: np.ndarray) -> float:
    """COCO 方式: recall = 0,0.01,...,1.0 の101点で、単調化 precision を平均。

    COCO の公式実装（pycocotools.accumulate）と同じ手順。03 で pycocotools と一致を検証する。
    """
    mpre = np.maximum.accumulate(precision[::-1])[::-1]  # 右から単調化
    rec_thrs = np.linspace(0, 1, 101)
    inds = np.searchsorted(recall, rec_thrs, side="left")  # 各閾値を満たす最初の点
    q = np.zeros(101)
    for ri, pi in enumerate(inds):
        if pi < len(mpre):
            q[ri] = mpre[pi]  # 届かない recall 閾値は 0 のまま
    return float(q.mean())


def main() -> None:
    out = output_dir()
    dataset = make_detection_dataset(n_images=12, seed=0)
    category_id, iou_thr = 1, 0.5  # circle を例に IoU=0.5 の AP を3方式で比べる

    precision, recall = build_pr(dataset, category_id, iou_thr)

    ap11 = ap_11point(precision, recall)
    ap_all = ap_all_point(precision, recall)
    ap101 = ap_coco101(precision, recall)

    print(f"=== カテゴリ {category_id} の AP@{iou_thr}（補間方式で値が変わる）===")
    print(f"  PASCAL 11点補間   : {ap11:.4f}")
    print(f"  全点(VOC2010+)補間: {ap_all:.4f}")
    print(f"  COCO 101点補間    : {ap101:.4f}")
    print("  → どれも“同じ PR 曲線”の要約だが、標本化の粗さで小数が違う。"
          "『mAP いくつ』と言うときは方式(と IoU)を必ず添える。")

    # === ソートを忘れるとどうなるか（意図的なバグ）===============================
    # AP は『スコア降順』が大前提。並べ替えずに cumsum すると PR 列が無意味になり値が壊れる。
    scores_all, tp_all, n_gt_total = [], [], 0
    for d in dataset:
        gm, dm = d["gt_labels"] == category_id, d["pred_labels"] == category_id
        tp, sc, n_gt = match_image(d["pred_boxes"][dm], d["pred_scores"][dm], d["gt_boxes"][gm], iou_thr)
        scores_all.append(sc)
        tp_all.append(tp)
        n_gt_total += n_gt
    tp_unsorted = np.concatenate(tp_all)  # わざとスコアで並べ替えない
    tp_cum = np.cumsum(tp_unsorted)
    fp_cum = np.cumsum(1.0 - tp_unsorted)
    rec_bad = tp_cum / max(n_gt_total, 1)
    prec_bad = tp_cum / np.clip(tp_cum + fp_cum, 1e-12, None)
    ap101_bad = ap_coco101(prec_bad, rec_bad)
    print("\n=== よくある事故: confidence 降順ソートを忘れる ===")
    print(f"  正しい(ソートあり) COCO-AP = {ap101:.4f}")
    print(f"  間違い(ソート無し) COCO-AP = {ap101_bad:.4f}  ← 別物の数字になる")

    # === PR 曲線と単調包絡の可視化 ==============================================
    mpre = np.maximum.accumulate(precision[::-1])[::-1]
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    ax.plot(recall, precision, color="C0", alpha=0.45, lw=1.4, label="raw PR (zig-zag)")
    ax.plot(recall, mpre, color="C3", lw=2.0, label="monotone envelope")
    # 11点標本の位置を点で示す
    for r in np.linspace(0, 1, 11):
        mask = recall >= r
        p = precision[mask].max() if mask.any() else 0.0
        ax.plot(r, p, "o", color="C2", ms=4, label="11-pt samples" if r == 0 else None)
    ax.set_xlabel("recall = TP / (#GT)")
    ax.set_ylabel("precision = TP / (TP+FP)")
    ax.set_ylim(0, 1.02)
    ax.set_xlim(0, 1.02)
    ax.set_title(f"PR curve (cat={category_id}, IoU={iou_thr})\n"
                 f"AP: 11pt={ap11:.3f} / all={ap_all:.3f} / COCO101={ap101:.3f}")
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "02_pr_curve.png", dpi=120)
    plt.close(fig)

    # === JSON 保存 =============================================================
    summary = {
        "category_id": category_id, "iou_thr": iou_thr,
        "ap_11point": ap11, "ap_all_point": ap_all, "ap_coco101": ap101,
        "ap_coco101_unsorted_bug": ap101_bad,
    }
    (out / "02_pr_ap.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\n完了。02_pr_curve.png / 02_pr_ap.json を保存しました。")


if __name__ == "__main__":
    main()
