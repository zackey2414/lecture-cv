"""第19回 演習問題（物体検出 mAP の自力実装 — IoU・マッチング・PR・AP）。

使い方:
  1. 各 exN_*() の中の TODO を自分で実装する（最初は NotImplementedError で FAIL になる）。
  2. 自己採点を実行（未実装でも例外で落ちず、PASS/FAIL を一覧表示して必ず正常終了する）:
         uv run python lectures/19_detection_map_from_scratch/exercises.py
  3. どうしても分からない時は、模範解答の挙動を見る:
         SHOW_SOLUTION=1 uv run python lectures/19_detection_map_from_scratch/exercises.py

狙い: mAP は「ライブラリを呼ぶ」前に「式と手順で書ける」ことが大事。
      IoU → confidence 降順マッチング → PR の累積 → COCO 101点 AP を numpy だけで組み立てる。
"""

from __future__ import annotations

import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from det_helpers import (  # noqa: E402
    CATEGORIES,
    box_iou_numpy,
    make_detection_dataset,
    match_image,
    to_coco_dt,
    to_coco_gt,
)

REC_THRS = np.linspace(0.0, 1.0, 101)


# =====================================================================
# 演習（ここを実装する）
# =====================================================================

def ex1_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """演習1: 2つの箱(xyxy)の IoU を1つ返す。

    交差矩形の左上=max同士・右下=min同士、負なら0。IoU=交差/(面積A+面積B-交差)。和集合0なら0.0。
    """
    # TODO: 定義式どおりに IoU を計算して float で返す
    raise NotImplementedError


def ex2_match(pred_boxes: np.ndarray, pred_scores: np.ndarray,
              gt_boxes: np.ndarray, iou_thr: float) -> np.ndarray:
    """演習2: 貪欲マッチングで、各予測の TP=1.0/FP=0.0 フラグを『スコア降順の順序で』返す。

    手順: スコア降順(安定ソート)に予測を見て、未使用かつ IoU≥閾値の GT のうち IoU 最大へ割当て。
          割り当てたら TP・その GT を使用済みに。見つからなければ FP。1GT は1度だけ使える。
    ヒント: box_iou_numpy(pred_boxes[order], gt_boxes) で IoU 行列を作ると楽。
    """
    # TODO: スコア降順に並んだ TP/FP フラグ配列 (N,) を返す
    raise NotImplementedError


def ex3_pr(tp_sorted: np.ndarray, n_gt: int) -> tuple[np.ndarray, np.ndarray]:
    """演習3: スコア降順の TP フラグ列から precision 列・recall 列を作って返す。

    tp_cum=cumsum(tp), fp_cum=cumsum(1-tp)。recall=tp_cum/n_gt, precision=tp_cum/(tp_cum+fp_cum)。
    戻り値は (precision, recall) の順。
    """
    # TODO: (precision, recall) を返す
    raise NotImplementedError


def ex4_ap_coco101(precision: np.ndarray, recall: np.ndarray) -> float:
    """演習4: COCO 101点補間で AP を返す（pycocotools と一致させる）。

    手順: precision を右から単調化（np.maximum.accumulate(precision[::-1])[::-1]）。
          recall 閾値 0,0.01,...,1.0 の各点を np.searchsorted(recall, thr, side='left') で探し、
          その位置の単調 precision を取る（範囲外は 0）。101個の平均が AP。
    """
    # TODO: COCO 101点 AP を float で返す
    raise NotImplementedError


def ex5_ap_11point(precision: np.ndarray, recall: np.ndarray) -> float:
    """演習5: PASCAL VOC 2007 方式の 11点補間 AP を返す。

    recall = 0,0.1,...,1.0 の11点それぞれで「recall>=r を満たす点の precision の最大値」
    （満たす点が無ければ 0）を取り、その11個を平均する。同じ PR でも COCO 101点とは値が違う。
    ヒント: for r in np.linspace(0,1,11): mask=recall>=r; p=precision[mask].max() if mask.any() else 0
    """
    # TODO: PASCAL 11点 AP を float で返す
    raise NotImplementedError


def ex6_ap_all_point(precision: np.ndarray, recall: np.ndarray) -> float:
    """演習6: VOC2010+ 方式の「全点」AP（単調化した PR 曲線の真下の面積）を返す。

    端点 (recall=0, precision=1) と (recall=最終, precision=0) を足し、precision を右から
    単調化（np.maximum.accumulate）。recall が変化した位置だけ AP=Σ(Δrecall)×precision を積む。
    ヒント: mrec=concat([[0],recall,[recall[-1]]]); mpre=concat([[1],precision,[0]]) を単調化。
    """
    # TODO: 全点(VOC2010+) AP を float で返す
    raise NotImplementedError


def ex7_xyxy_to_xywh(boxes: np.ndarray) -> np.ndarray:
    """演習7: xyxy 形式の (N,4) ボックス群を COCO の xywh 形式へ変換して返す。

    xyxy=[x1,y1,x2,y2] → xywh=[x1, y1, x2-x1, y2-y1]。pycocotools へ渡すときに必須の変換で、
    ここを誤ると箱が歪んで AP が崩壊する典型バグになる。x1,y1 はそのまま、w,h を差で作る。
    """
    # TODO: (N,4) の xywh 配列を返す
    raise NotImplementedError


def ex8_nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> np.ndarray:
    """演習8: Non-Maximum Suppression。残す予測の index 配列（スコア降順）を返す。

    手順: スコア降順に並べ、先頭(最高スコア)を採用→残りのうち先頭との IoU>閾値 を捨てる、を繰り返す。
          マッチングと同じく IoU で重複を抑える操作で、torchvision.ops.nms と同じ集合になる。
    ヒント: box_iou_numpy(boxes[i][None], boxes[残り])[0] <= iou_thr で生き残りを絞る。
    """
    # TODO: 残す index 配列 (K,) を返す
    raise NotImplementedError


def ex9_map_aggregate(ap_table: np.ndarray) -> tuple[float, float, float]:
    """演習9: AP[カテゴリ, IoU閾値] の表から mAP@0.5 / mAP@0.75 / mAP@[.5:.95] を返す。

    列の並びは IoU=0.50,0.55,...,0.95。mAP@0.5=列0の平均, mAP@0.75=列5の平均,
    mAP@[.5:.95]=全要素の平均。GT 無しカテゴリは np.nan で入っているので np.nanmean で除外する。
    戻り値は (map50, map75, map5095) の順。
    """
    # TODO: (map50, map75, map5095) を返す
    raise NotImplementedError


# =====================================================================
# 自己採点ランナー（未実装でも例外で落とさず PASS/FAIL を表示。必ず exit 0）
# =====================================================================

def _scratch_map5095(dataset) -> float:
    """各演習関数を組み合わせて mAP@[.5:.95] を出す（採点用。ex2〜ex4 が正しければ pycocotools と一致）。"""
    iou_thrs = np.linspace(0.5, 0.95, 10)
    cat_ids = [c["id"] for c in CATEGORIES]
    aps = []
    for c in cat_ids:
        for thr in iou_thrs:
            scores_all, tp_all, n_gt_total = [], [], 0
            for d in dataset:
                gm, dm = d["gt_labels"] == c, d["pred_labels"] == c
                tp = ex2_match(d["pred_boxes"][dm], d["pred_scores"][dm], d["gt_boxes"][gm], thr)
                order = np.argsort(-d["pred_scores"][dm], kind="stable")
                scores_all.append(d["pred_scores"][dm][order])
                tp_all.append(np.asarray(tp, dtype=float))
                n_gt_total += int(gm.sum())
            scores_all = np.concatenate(scores_all)
            tp_all = np.concatenate(tp_all)
            order = np.argsort(-scores_all, kind="stable")
            precision, recall = ex3_pr(tp_all[order], n_gt_total)
            aps.append(ex4_ap_coco101(np.asarray(precision), np.asarray(recall)))
    return float(np.nanmean(aps))


def _grade() -> None:
    import contextlib
    import io

    dataset = make_detection_dataset(n_images=8, seed=3)
    rng = np.random.default_rng(0)
    corner = rng.uniform(0, 300, (2,))
    a = np.array([corner[0], corner[1], corner[0] + 80, corner[1] + 60])
    b = np.array([corner[0] + 20, corner[1] + 10, corner[0] + 100, corner[1] + 70])

    results: list[tuple[str, bool, str]] = []

    def check(name: str, fn) -> None:
        try:
            ok, detail = fn()
            results.append((name, bool(ok), detail))
        except NotImplementedError:
            results.append((name, False, "未実装（TODO を埋めてください）"))
        except Exception as e:  # noqa: BLE001
            results.append((name, False, f"例外: {type(e).__name__}: {e}"))

    # ex1: 自作 IoU が box_iou_numpy と一致
    check("ex1_iou", lambda: (
        np.isclose(ex1_iou(a, b), box_iou_numpy(a[None], b[None])[0, 0], atol=1e-9),
        "IoU が box_iou_numpy と一致",
    ))

    # ex2: マッチングが正準エンジン match_image と一致
    def _c2():
        d = dataset[0]
        dm, gm = d["pred_labels"] == 1, d["gt_labels"] == 1
        mine = np.asarray(ex2_match(d["pred_boxes"][dm], d["pred_scores"][dm], d["gt_boxes"][gm], 0.5))
        ref, _, _ = match_image(d["pred_boxes"][dm], d["pred_scores"][dm], d["gt_boxes"][gm], 0.5)
        return np.array_equal(mine.astype(bool), ref.astype(bool)), "TP/FP フラグが match_image と一致"
    check("ex2_match", _c2)

    # ex3: PR 列の終端 recall がGTの取りこぼし具合と整合（precision/recall の定義チェック）
    def _c3():
        tp = np.array([1.0, 0.0, 1.0, 1.0, 0.0])
        precision, recall = ex3_pr(tp, n_gt=4)
        ok = (np.isclose(recall[-1], 3 / 4) and np.isclose(precision[0], 1.0)
              and np.isclose(precision[-1], 3 / 5))
        return ok, "precision/recall 列が定義どおり"
    check("ex3_pr", _c3)

    # ex4: COCO-101 AP（単調・単純ケース）。完全な PR は ex2/ex3 経由で総合チェックする
    def _c4():
        recall = np.array([0.25, 0.5, 0.75, 1.0])
        precision = np.array([1.0, 1.0, 0.75, 0.6])
        ap = ex4_ap_coco101(precision, recall)
        # 単調化後、recall>=0 の全101点で最大precision=1.0...の平均。手計算の参照値と比較
        mpre = np.maximum.accumulate(precision[::-1])[::-1]
        inds = np.searchsorted(recall, REC_THRS, side="left")
        ref = np.mean([mpre[i] if i < len(mpre) else 0.0 for i in inds])
        return np.isclose(ap, ref, atol=1e-9), "COCO 101点 AP が参照値と一致"
    check("ex4_ap_coco101", _c4)

    # ex5: PASCAL 11点 AP（参照値はインラインで独立計算）
    def _c5_11pt():
        recall = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
        precision = np.array([1.0, 0.9, 0.7, 0.6, 0.5])
        ref = float(np.mean([precision[recall >= r].max() if (recall >= r).any() else 0.0
                             for r in np.linspace(0, 1, 11)]))
        return np.isclose(ex5_ap_11point(precision, recall), ref, atol=1e-9), "PASCAL 11点 AP が参照値と一致"
    check("ex5_ap_11point", _c5_11pt)

    # ex6: 全点(VOC2010+) AP
    def _c6_all():
        recall = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
        precision = np.array([1.0, 0.9, 0.7, 0.6, 0.5])
        mrec = np.concatenate([[0.0], recall, [recall[-1]]])
        mpre = np.maximum.accumulate(np.concatenate([[1.0], precision, [0.0]])[::-1])[::-1]
        idx = np.where(mrec[1:] != mrec[:-1])[0]
        ref = float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))
        return np.isclose(ex6_ap_all_point(precision, recall), ref, atol=1e-9), "全点(VOC2010+) AP が参照値と一致"
    check("ex6_ap_all_point", _c6_all)

    # ex7: xyxy→xywh 変換
    def _c7_fmt():
        boxes = np.array([[10, 20, 40, 60], [0, 0, 100, 50]], dtype=float)
        got = np.asarray(ex7_xyxy_to_xywh(boxes), dtype=float)
        ref = np.array([[10, 20, 30, 40], [0, 0, 100, 50]], dtype=float)
        return got.shape == ref.shape and np.allclose(got, ref), "xyxy→xywh(COCO) 変換が正しい"
    check("ex7_xyxy_to_xywh", _c7_fmt)

    # ex8: NMS が torchvision.ops.nms と同じ残存集合になる
    def _c8_nms():
        import torch
        from torchvision.ops import nms as tv_nms
        rng2 = np.random.default_rng(1)
        bb = rng2.uniform(0, 300, (15, 2))
        bb = np.concatenate([bb, bb + rng2.uniform(20, 80, (15, 2))], axis=1)
        ss = rng2.uniform(0, 1, 15)
        mine = np.asarray(ex8_nms(bb, ss, 0.5)).astype(int).tolist()
        ref = tv_nms(torch.tensor(bb), torch.tensor(ss), 0.5).numpy().tolist()
        return set(mine) == set(ref), "NMS の残存indexが torchvision.ops.nms と一致"
    check("ex8_nms", _c8_nms)

    # ex9: AP 表からの mAP 集約（NaN=空カテゴリを除外）
    def _c9_agg():
        ap = np.full((3, 10), np.nan)
        ap[0] = np.linspace(0.9, 0.4, 10)
        ap[1, :] = 0.5  # ap[2] は全 NaN（GT 無しカテゴリ）→ 平均から除外されるべき
        m50, m75, m5095 = ex9_map_aggregate(ap)
        ok = (np.isclose(m50, np.nanmean(ap[:, 0])) and np.isclose(m75, np.nanmean(ap[:, 5]))
              and np.isclose(m5095, np.nanmean(ap)))
        return ok, "mAP@0.5/0.75/[.5:.95] の集約(NaN除外)が正しい"
    check("ex9_map_aggregate", _c9_agg)

    # 総合: ex2〜ex4 を組み合わせた mAP@[.5:.95] が pycocotools と一致
    def _c5():
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
        mine = _scratch_map5095(dataset)
        with contextlib.redirect_stdout(io.StringIO()):
            cg = COCO()
            cg.dataset = to_coco_gt(dataset)
            cg.createIndex()
            cd = cg.loadRes(to_coco_dt(dataset))
            ev = COCOeval(cg, cd, "bbox")
            ev.evaluate()
            ev.accumulate()
            ev.summarize()
        return np.isclose(mine, ev.stats[0], atol=1e-6), "自作 mAP@[.5:.95] が pycocotools と一致"
    check("integration_map_vs_pycocotools", _c5)

    print("=== 採点結果 ===")
    all_ok = True
    for name, ok, detail in results:
        mark = "PASS" if ok else "FAIL"
        all_ok = all_ok and ok
        print(f"  [{mark}] {name:32s} {detail}")
    print("\nALL PASS 🎉" if all_ok else "\nまだ未達の演習があります。TODO を埋めましょう。")


# =====================================================================
# 模範解答（SHOW_SOLUTION=1 のときに本体へ差し替える）。まずは自力で！
# =====================================================================

def _sol_ex1(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1) + max(0.0, bx2 - bx1) * max(0.0, by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def _sol_ex2(pred_boxes, pred_scores, gt_boxes, iou_thr):
    pred_boxes = np.asarray(pred_boxes, float).reshape(-1, 4)
    pred_scores = np.asarray(pred_scores, float).reshape(-1)
    gt_boxes = np.asarray(gt_boxes, float).reshape(-1, 4)
    order = np.argsort(-pred_scores, kind="stable")
    tp = np.zeros(len(order))
    if len(gt_boxes) == 0:
        return tp
    ious = box_iou_numpy(pred_boxes[order], gt_boxes)
    used = np.zeros(len(gt_boxes), bool)
    for di in range(len(order)):
        best_iou, best = iou_thr, -1
        for gi in range(len(gt_boxes)):
            if used[gi] or ious[di, gi] < best_iou:
                continue
            best_iou, best = ious[di, gi], gi
        if best >= 0:
            used[best] = True
            tp[di] = 1.0
    return tp


def _sol_ex3(tp_sorted, n_gt):
    tp_sorted = np.asarray(tp_sorted, float)
    tp_cum = np.cumsum(tp_sorted)
    fp_cum = np.cumsum(1.0 - tp_sorted)
    recall = tp_cum / max(n_gt, 1)
    precision = tp_cum / np.clip(tp_cum + fp_cum, 1e-12, None)
    return precision, recall


def _sol_ex4(precision, recall):
    precision = np.asarray(precision, float)
    recall = np.asarray(recall, float)
    mpre = np.maximum.accumulate(precision[::-1])[::-1]
    inds = np.searchsorted(recall, REC_THRS, side="left")
    q = np.zeros(101)
    for ri, pi in enumerate(inds):
        if pi < len(mpre):
            q[ri] = mpre[pi]
    return float(q.mean())


def _sol_ex5(precision, recall):
    precision = np.asarray(precision, float)
    recall = np.asarray(recall, float)
    aps = []
    for r in np.linspace(0, 1, 11):
        mask = recall >= r
        aps.append(precision[mask].max() if mask.any() else 0.0)
    return float(np.mean(aps))


def _sol_ex6(precision, recall):
    precision = np.asarray(precision, float)
    recall = np.asarray(recall, float)
    mrec = np.concatenate([[0.0], recall, [recall[-1]]])
    mpre = np.maximum.accumulate(np.concatenate([[1.0], precision, [0.0]])[::-1])[::-1]
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def _sol_ex7(boxes):
    boxes = np.asarray(boxes, float).reshape(-1, 4)
    out = boxes.copy()
    out[:, 2] = boxes[:, 2] - boxes[:, 0]
    out[:, 3] = boxes[:, 3] - boxes[:, 1]
    return out


def _sol_ex8(boxes, scores, iou_thr):
    boxes = np.asarray(boxes, float).reshape(-1, 4)
    scores = np.asarray(scores, float).reshape(-1)
    order = np.argsort(-scores, kind="stable")
    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(int(i))
        if len(order) == 1:
            break
        ious = box_iou_numpy(boxes[i][None], boxes[order[1:]])[0]
        order = order[1:][ious <= iou_thr]  # 先頭との重なりが閾値以下のものだけ残す
    return np.array(keep, dtype=int)


def _sol_ex9(ap_table):
    ap = np.asarray(ap_table, float)
    return float(np.nanmean(ap[:, 0])), float(np.nanmean(ap[:, 5])), float(np.nanmean(ap))


def _install_solutions() -> None:
    g = globals()
    g["ex1_iou"] = _sol_ex1
    g["ex2_match"] = _sol_ex2
    g["ex3_pr"] = _sol_ex3
    g["ex4_ap_coco101"] = _sol_ex4
    g["ex5_ap_11point"] = _sol_ex5
    g["ex6_ap_all_point"] = _sol_ex6
    g["ex7_xyxy_to_xywh"] = _sol_ex7
    g["ex8_nms"] = _sol_ex8
    g["ex9_map_aggregate"] = _sol_ex9


if __name__ == "__main__":
    if os.environ.get("SHOW_SOLUTION") == "1":
        print("(模範解答モードで実行します)\n")
        _install_solutions()
    _grade()
