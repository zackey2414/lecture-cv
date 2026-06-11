"""04: インスタンス/パノプティックの評価指標 — mask AP と PQ=SQ×RQ。

この回の“核”。検出やセマンティックの指標を、マスクの世界へ拡張する。

(1) mask AP（インスタンス）:
    物体検出の mAP（第19回）で使った『box IoU』を、そのまま『mask IoU』に置き換える
    だけ。IoU=交差画素/和集合画素 で予測マスクと GT マスクを対応付け、confidence 降順に
    TP/FP を積んで PR 曲線→AP を出す。実装の正準は pycocotools の COCOeval(iouType="segm")
    で、予測・GT を RLE（Run-Length Encoding, pycocotools.mask）で渡す。ここでは
    『自作の単純な AP@0.5』と『COCOeval の segm AP』を並べ、考え方と公式実装を結ぶ。

(2) PQ（パノプティック）= SQ × RQ:
    パノプティックは things と stuff を全画素にもれなく割り当てるので、AP とは別の
    指標 PQ（Panoptic Quality）で測る。カテゴリごとに、IoU>0.5 を満たす GT-予測セグメントを
    一意マッチさせ、
        SQ（Segmentation Quality）= マッチした組の平均 IoU            … 形の良さ
        RQ（Recognition Quality） = TP / (TP + 0.5 FP + 0.5 FN)        … 検出の F1
        PQ = SQ × RQ
    をカテゴリごとに出し、全カテゴリで平均する。ここでは numpy で一から PQ を組み、
    torchmetrics.detection.PanopticQuality と一致することを assert で確認する。

合成 GT/予測を完全に制御して使うので、ネット接続もモデルも不要（評価は CPU で完結）。
実行: uv run python lectures/22_instance_panoptic_sam/04_maskap_pq_eval.py
出力: outputs/22_instance_panoptic_sam/04_eval_metrics.json
"""

from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import seg_helpers as H  # noqa: E402

HEIGHT, WIDTH = 200, 260


# =====================================================================
# 合成マスク生成（GT/予測を完全制御する）
# =====================================================================

def _disk(cy: int, cx: int, r: int) -> np.ndarray:
    yy, xx = np.ogrid[:HEIGHT, :WIDTH]
    return (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r


def _rect(y0: int, x0: int, y1: int, x1: int) -> np.ndarray:
    m = np.zeros((HEIGHT, WIDTH), dtype=bool)
    m[y0:y1, x0:x1] = True
    return m


# =====================================================================
# (1) mask AP
# =====================================================================

def evaluate_mask_ap() -> dict:
    """インスタンスの mask AP を『自作 AP@0.5』と『COCOeval segm』で出す。"""
    # GT: ball(cat1) と box(cat2)
    gt_masks = [_disk(80, 80, 35), _rect(40, 160, 120, 230)]
    gt_cats = [1, 2]
    # 予測: ①ほぼ一致(良) ②少しずれ(良) ③偽陽性(GTに対応なし・低スコア)
    preds = [
        (_disk(82, 82, 34), 1, 0.95),
        (_rect(45, 165, 118, 232), 2, 0.80),
        (_disk(150, 60, 20), 1, 0.40),
    ]

    # --- 自作 AP@0.5（クラス1のみ・PR を手で組む最小版） ---
    ap50_manual_cls1 = _manual_ap50(
        gt=[m for m, c in zip(gt_masks, gt_cats) if c == 1],
        preds=[(m, s) for (m, c, s) in preds if c == 1],
    )

    # --- COCOeval segm（公式実装で AP / AP50 / AP75） ---
    coco_stats = _cocoeval_segm(gt_masks, gt_cats, preds)

    print("[04] (1) mask AP（インスタンス）")
    print(f"   自作 AP@0.5 (class=1, ball)        = {ap50_manual_cls1:.3f}")
    print(f"   COCOeval segm  mAP@[.5:.95]        = {coco_stats['mAP']:.3f}")
    print(f"   COCOeval segm  AP@0.5              = {coco_stats['AP50']:.3f}")
    print(f"   COCOeval segm  AP@0.75             = {coco_stats['AP75']:.3f}")
    return {"manual_ap50_class1": round(ap50_manual_cls1, 4), "cocoeval_segm": coco_stats}


def _manual_ap50(gt: list[np.ndarray], preds: list[tuple[np.ndarray, float]]) -> float:
    """1クラス・IoU>=0.5 の AP を numpy で素朴に計算する（PR 全点積分）。

    手順は検出 mAP と同じ: 予測を score 降順に並べ、未マッチの GT に IoU>=0.5 で
    貪欲対応 → TP/FP を累積 → precision/recall 列 → 単調化して面積を取る。
    """
    order = sorted(range(len(preds)), key=lambda i: -preds[i][1])  # score 降順
    matched = [False] * len(gt)
    tp = np.zeros(len(preds))
    fp = np.zeros(len(preds))
    for rank, i in enumerate(order):
        pmask = preds[i][0]
        best_iou, best_j = 0.0, -1
        for j, gmask in enumerate(gt):
            if matched[j]:
                continue
            iou = H.mask_iou(pmask, gmask)
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_iou >= 0.5 and best_j >= 0:
            tp[rank] = 1.0
            matched[best_j] = True
        else:
            fp[rank] = 1.0
    tp_cum, fp_cum = np.cumsum(tp), np.cumsum(fp)
    recall = tp_cum / max(len(gt), 1)
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
    # precision を単調非増加に補正してから台形で積分（全点補間）。
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    recall = np.r_[0.0, recall]
    precision = np.r_[precision[0] if len(precision) else 1.0, precision]
    return float(np.trapezoid(precision, recall))


def _cocoeval_segm(gt_masks, gt_cats, preds) -> dict:
    """pycocotools の COCOeval(iouType='segm') で公式の mask AP を出す。"""
    from pycocotools import mask as mask_utils
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    def encode_rle(m: np.ndarray) -> dict:
        # マスクは Fortran 連続（列優先）で渡すのが pycocotools の約束。counts は bytes。
        return mask_utils.encode(np.asfortranarray(m.astype(np.uint8)))

    def to_json_rle(rle: dict) -> dict:
        # counts(bytes) を str 化（COCOeval は in-memory なら str/bytes どちらも扱える）。
        return {"size": rle["size"], "counts": rle["counts"].decode("ascii")}

    # GT を COCO 形式の dict に組む（images は height/width が必須）。
    gt_dict = {
        "images": [{"id": 1, "height": HEIGHT, "width": WIDTH}],
        "categories": [{"id": 1, "name": "ball"}, {"id": 2, "name": "box"}],
        "annotations": [],
    }
    for i, (m, c) in enumerate(zip(gt_masks, gt_cats), start=1):
        rle = encode_rle(m)  # area/bbox は bytes-RLE から計算する
        gt_dict["annotations"].append(
            {
                "id": i,
                "image_id": 1,
                "category_id": c,
                "segmentation": to_json_rle(rle),
                "area": float(mask_utils.area(rle)),
                "bbox": list(mask_utils.toBbox(rle)),
                "iscrowd": 0,
            }
        )
    coco_gt = COCO()
    coco_gt.dataset = gt_dict
    coco_gt.createIndex()

    dt = [{"image_id": 1, "category_id": c, "segmentation": to_json_rle(encode_rle(m)), "score": s}
          for (m, c, s) in preds]
    coco_dt = coco_gt.loadRes(dt)

    ev = COCOeval(coco_gt, coco_dt, iouType="segm")
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    return {"mAP": round(float(ev.stats[0]), 4), "AP50": round(float(ev.stats[1]), 4),
            "AP75": round(float(ev.stats[2]), 4)}


# =====================================================================
# (2) PQ = SQ × RQ
# =====================================================================

def evaluate_pq() -> dict:
    """パノプティック PQ を自作 numpy と torchmetrics で計算し一致を確認する。"""
    import torch
    from torchmetrics.detection import PanopticQuality

    # things: ball(1), box(2)。stuff: sky(10), ground(11)。
    things, stuffs = {1, 2}, {10, 11}
    ground = _rect(150, 0, 200, WIDTH)
    ball_gt, box_gt = _disk(80, 80, 35), _rect(40, 160, 120, 230)
    ball_pr, box_pr = _disk(82, 82, 34), _rect(45, 165, 118, 232)
    ball2_gt = _disk(30, 130, 15)        # GT にだけ在る2個目の ball → 取りこぼし=FN
    spurious_box = _rect(8, 8, 40, 55)   # 予測にだけ在る余計な box → 誤検出=FP

    # (category_id, instance_id) を全画素に1つだけ割り当てる（後勝ちで重なりを解消）。
    # わざと FN（ball2 を予測し損ねる）と FP（余計な box を予測）を入れ、RQ<1 を体感する。
    gt = _compose_panoptic([
        (np.ones((HEIGHT, WIDTH), bool), 10, 0),  # sky（最初に全面、後で上書き）
        (ground, 11, 0),
        (ball_gt, 1, 1),
        (ball2_gt, 1, 3),
        (box_gt, 2, 2),
    ])
    pred = _compose_panoptic([
        (np.ones((HEIGHT, WIDTH), bool), 10, 0),
        (ground, 11, 0),
        (ball_pr, 1, 1),
        (box_pr, 2, 2),
        (spurious_box, 2, 5),
    ])

    pq_manual = _manual_pq(pred, gt, things | stuffs)

    metric = PanopticQuality(things=things, stuffs=stuffs, return_sq_and_rq=True)
    out = metric(torch.tensor(pred)[None], torch.tensor(gt)[None])
    pq_tm, sq_tm, rq_tm = (float(out[0]), float(out[1]), float(out[2]))

    print("\n[04] (2) パノプティック PQ = SQ × RQ")
    print(f"   自作        PQ={pq_manual['PQ']:.4f}  SQ={pq_manual['SQ']:.4f}  RQ={pq_manual['RQ']:.4f}")
    print(f"   torchmetrics PQ={pq_tm:.4f}  SQ={sq_tm:.4f}  RQ={rq_tm:.4f}")
    ok = np.isclose(pq_manual["PQ"], pq_tm, atol=1e-3)
    print(f"   一致(atol=1e-3): {'PASS' if ok else 'FAIL'}")
    assert ok, "自作 PQ と torchmetrics が一致しません"
    return {"manual": pq_manual, "torchmetrics": {"PQ": round(pq_tm, 4), "SQ": round(sq_tm, 4), "RQ": round(rq_tm, 4)}}


def _compose_panoptic(specs: list[tuple[np.ndarray, int, int]]) -> np.ndarray:
    """(mask, category_id, instance_id) の列から (H, W, 2) の panoptic ラベルを作る。"""
    cat = np.zeros((HEIGHT, WIDTH), dtype=np.int64)
    inst = np.zeros((HEIGHT, WIDTH), dtype=np.int64)
    for m, c, iid in specs:
        cat[m] = c
        inst[m] = iid
    return np.stack([cat, inst], axis=-1)


def _manual_pq(pred: np.ndarray, gt: np.ndarray, valid_cats: set[int]) -> dict:
    """panopticapi 流に PQ をカテゴリ別集計 → 平均で求める（numpy 実装）。

    各カテゴリで、GT セグメントと予測セグメントを (cat, inst) 単位に切り出し、
    IoU>0.5 を満たす組を一意マッチ（>0.5 なので最大1組に限られる）。
    SQ=Σ(TPのIoU)/TP, RQ=TP/(TP+0.5FP+0.5FN), PQ=SQ×RQ を category 平均する。
    """
    per_class_pq: list[float] = []
    per_class_sq: list[float] = []
    per_class_rq: list[float] = []
    for c in sorted(valid_cats):
        gt_segments = _segments_of(gt, c)
        pr_segments = _segments_of(pred, c)
        if not gt_segments and not pr_segments:
            continue  # GT も予測も無いカテゴリは評価対象外（panopticapi と同じ）
        matched_gt: set[int] = set()
        iou_sum, tp = 0.0, 0
        for pr in pr_segments:
            best_iou, best_k = 0.0, -1
            for k, g in enumerate(gt_segments):
                if k in matched_gt:
                    continue
                iou = H.mask_iou(pr, g)
                if iou > best_iou:
                    best_iou, best_k = iou, k
            if best_iou > 0.5 and best_k >= 0:
                matched_gt.add(best_k)
                iou_sum += best_iou
                tp += 1
        fp = len(pr_segments) - tp
        fn = len(gt_segments) - tp
        sq = iou_sum / tp if tp > 0 else 0.0
        rq = tp / (tp + 0.5 * fp + 0.5 * fn) if (tp + fp + fn) > 0 else 0.0
        per_class_sq.append(sq)
        per_class_rq.append(rq)
        per_class_pq.append(sq * rq)
    pq = float(np.mean(per_class_pq)) if per_class_pq else 0.0
    sq = float(np.mean(per_class_sq)) if per_class_sq else 0.0
    rq = float(np.mean(per_class_rq)) if per_class_rq else 0.0
    return {"PQ": round(pq, 4), "SQ": round(sq, 4), "RQ": round(rq, 4)}


def _segments_of(panoptic: np.ndarray, category: int) -> list[np.ndarray]:
    """(H,W,2) の panoptic から、指定カテゴリの各インスタンスの bool マスクを列挙する。"""
    cat, inst = panoptic[..., 0], panoptic[..., 1]
    out = []
    for iid in np.unique(inst[cat == category]):
        out.append((cat == category) & (inst == iid))
    return out


def main() -> None:
    out_dir = H.ensure_output_dir()
    ap = evaluate_mask_ap()
    pq = evaluate_pq()
    payload = {"mask_ap": ap, "panoptic_quality": pq}
    path = out_dir / "04_eval_metrics.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  saved: {path.name}")


if __name__ == "__main__":
    main()
