"""第22回 演習問題（インスタンス/パノプティック/SAM の“肝”を手で書く）。

使い方:
  1. 各 exN_*() の中の TODO を自分で実装する（最初は NotImplementedError で FAIL）。
  2. 自己採点を実行（未実装でも例外で落ちず、PASS/FAIL を一覧表示して必ず正常終了する）:
         uv run python lectures/22_instance_panoptic_sam/exercises.py
  3. どうしても分からない時は、模範解答の挙動を見る:
         SHOW_SOLUTION=1 uv run python lectures/22_instance_panoptic_sam/exercises.py
     全問の完成形は exercises_solutions.py（同じ採点ロジックで ALL PASS する）。

狙い: この回の指標（mask IoU / mask AP のマッチング・AP 補間 / PQ）と、モデル出力の
      正しい後処理（Mask R-CNN の確率マスク二値化・SAM の3マスクからの選択・RLE 復元）を、
      ライブラリに頼らず numpy/torch だけで組み立てられるようにする。
      易→難で10問。とくに ex6〜ex8（AP 補間と mask AP）は物体検出 mAP（第19回）を
      マスクへ拡張する核心なので厚めに置いた。
本演習はモデルもネットも不要（合成マスクだけで完結する）。
"""

from __future__ import annotations

import os

import numpy as np


# =====================================================================
# 演習（ここを実装する）。易 → 難 の順に並んでいる。
# =====================================================================

def ex1_mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    """演習1（易）: 2つの bool マスク a, b の IoU = 交差画素 / 和集合画素 を返す。

    和集合が 0（両方とも空）のときは 0.0 を返す。
    ヒント: np.logical_and / np.logical_or の .sum()。
    """
    # TODO: IoU を計算して float で返す
    raise NotImplementedError


def ex2_binarize_masks(prob: np.ndarray, thr: float = 0.5) -> np.ndarray:
    """演習2（易）: Mask R-CNN の masks 出力 (N, 1, H, W) 確率を (N, H, W) bool に二値化する。

    Mask R-CNN の 'masks' は確率マップで形が (N, 1, H, W)。チャンネル次元(1)を潰し、
    thr を超える画素を True にする。これを忘れて argmax したり (N,1,H,W) のまま
    draw_segmentation_masks に渡すのが典型バグ。
    ヒント: prob[:, 0] で (N, H, W) を取り出し、> thr で bool 化。
    """
    # TODO: (N, H, W) の bool 配列を返す
    raise NotImplementedError


def ex3_match_counts(
    gt_masks: list[np.ndarray],
    preds: list[tuple[np.ndarray, float]],
    iou_thr: float = 0.5,
) -> tuple[int, int, int]:
    """演習3（中）: 予測を score 降順に GT へ貪欲マッチし (TP, FP, FN) を返す（mask AP の前段）。

    手順（検出 mAP と同じ。box IoU を mask IoU に置換しただけ）:
      1. preds を score の降順に並べる。
      2. 各予測について、未マッチの GT の中で IoU 最大の相手を探す。
      3. その IoU >= iou_thr なら TP かつ GT をマッチ済みにする。さもなくば FP。
      4. 最後まで誰にもマッチされなかった GT の数が FN。
    ヒント: ex1_mask_iou を使ってよい。matched = [False]*len(gt_masks)。
    """
    # TODO: (tp, fp, fn) を返す
    raise NotImplementedError


def ex4_pq_single_category(
    gt_masks: list[np.ndarray],
    pred_masks: list[np.ndarray],
    iou_thr: float = 0.5,
) -> tuple[float, float, float]:
    """演習4（中）: 1カテゴリの (SQ, RQ, PQ) を返す。

    IoU>iou_thr を満たす GT-予測を一意マッチ（>0.5 なので相手は高々1つ）。
      SQ = マッチ組の平均 IoU（TP が0なら0）
      RQ = TP / (TP + 0.5*FP + 0.5*FN)（分母0なら0）
      PQ = SQ * RQ
    ヒント: TP のときに IoU を足し込み、最後に TP で割って SQ。
    """
    # TODO: (sq, rq, pq) を返す
    raise NotImplementedError


def ex5_select_best_sam_mask(masks: np.ndarray, iou_scores: np.ndarray) -> np.ndarray:
    """演習5（易）: SAM が返す3枚のマスクから iou_scores 最大の1枚 (H, W) を返す。

    SAM は曖昧性を考慮して1プロンプトにつき3マスクを返す。iou_scores（予測品質）が
    最大のインデックスを選ぶ。masks 形状は (3, H, W)、iou_scores は (3,)。
    ヒント: np.argmax(iou_scores)。
    """
    # TODO: best な (H, W) マスクを返す
    raise NotImplementedError


def ex6_ap_all_point(recall: np.ndarray, precision: np.ndarray) -> float:
    """演習6（難）: PR 曲線から AP を「全点補間（all-point）」で求める。

    これは物体検出/インスタンスの AP の核。手順は:
      1. recall の両端に 0 と 1 を足す: mrec = [0, *recall, 1]
      2. precision の両端に 0 を足す:   mpre = [0, *precision, 0]
      3. precision を右から見て単調非増加にする（包絡線）:
             for i in range(len(mpre)-1, 0, -1): mpre[i-1] = max(mpre[i-1], mpre[i])
      4. recall が変化した点だけを取り、長方形の面積を足す:
             idx = where(mrec[1:] != mrec[:-1]); AP = Σ (mrec[idx+1]-mrec[idx]) * mpre[idx+1]
    recall は単調非減少で渡される前提（cumsum から作った列）。
    ヒント: np.concatenate / np.where / np.sum。
    """
    # TODO: 全点補間 AP を float で返す
    raise NotImplementedError


def ex7_ap_11point(recall: np.ndarray, precision: np.ndarray) -> float:
    """演習7（中）: PASCAL VOC2007 流の「11点補間」で AP を求める。

    recall の閾値を 0.0, 0.1, ..., 1.0 の11点で動かし、各 t について
    『recall >= t を満たす範囲での precision の最大値』を取り、その平均を返す。
        AP = (1/11) * Σ_{t in 0,0.1,...,1.0} max_{recall>=t} precision
    recall>=t を満たす点が無ければ、その t の寄与は 0。
    ヒント: np.linspace(0, 1, 11)、precision[recall >= t].max()。
    """
    # TODO: 11点補間 AP を float で返す
    raise NotImplementedError


def ex8_mask_ap50(
    gt: list[tuple[np.ndarray, int]],
    preds: list[tuple[np.ndarray, int, float]],
    iou_thr: float = 0.5,
) -> float:
    """演習8（難）: クラスを跨いだ mask AP@0.5（= mAP）を求める。

    gt    : [(mask, category_id), ...]
    preds : [(mask, category_id, score), ...]
    手順（COCO の mAP と同じ枠組み。box→mask に置換しただけ）:
      1. GT に登場する category ごとに評価する。
      2. そのクラスの予測を score 降順に並べ、未マッチ GT に IoU>=iou_thr で貪欲対応。
         TP なら tp=1、外したら fp=1（rank 順に 0/1 を記録）。
      3. tp/fp を累積して recall=tp_cum/len(gt_c), precision=tp_cum/(tp_cum+fp_cum)。
      4. その PR から AP（全点補間, ex6 でよい）を出す。
      5. 全カテゴリの AP を平均して mAP を返す（GT が無いクラスは評価対象外）。
    ヒント: ex1_mask_iou と ex6_ap_all_point を再利用してよい。
    """
    # TODO: mAP@0.5 を float で返す
    raise NotImplementedError


def ex9_rle_decode(counts: list[int], height: int, width: int) -> np.ndarray:
    """演習9（中）: COCO の「非圧縮 RLE」counts を (H, W) bool マスクへ復元する。

    pycocotools の RLE は列優先（Fortran 連続）で、0(背景) から始まる run の長さの列。
    例: counts=[7,3,3,3,14] は『背景7画素→前景3→背景3→前景3→背景14』を表す。
    復元手順:
      1. 長さ H*W の bool 配列を用意。
      2. counts を順に読み、False/True を交互に run の長さぶん書き込む（最初は False）。
      3. order='F'（列優先）で (H, W) に reshape する。
    ヒント: flat[idx:idx+c] = val; idx += c; val = not val。reshape(..., order='F')。
    """
    # TODO: (H, W) の bool マスクを返す
    raise NotImplementedError


def ex10_pq_overall(
    per_category: dict[int, tuple[list[np.ndarray], list[np.ndarray]]],
) -> tuple[float, float, float]:
    """演習10（難）: カテゴリ別の (gt_masks, pred_masks) から全体 (SQ, RQ, PQ) を求める。

    per_category: {category_id: (gt_masks, pred_masks), ...}
    各カテゴリで ex4 と同じ要領で (SQ, RQ, PQ) を出し、最後にカテゴリ平均する。
    『全体でまとめて集計』ではなく『カテゴリごとに出して最後に平均』が正解
    （PQ の定番の落とし穴）。GT も予測も無いカテゴリは平均から除く。
    ヒント: ex4_pq_single_category を各カテゴリに適用 → np.mean。
    """
    # TODO: (sq, rq, pq) を返す
    raise NotImplementedError


# =====================================================================
# 自己採点ランナー（未実装でも例外で落とさず PASS/FAIL を表示。必ず exit 0）
# =====================================================================

def _disk(h, w, cy, cx, r) -> np.ndarray:
    yy, xx = np.ogrid[:h, :w]
    return (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r


def _rect(h, w, y0, x0, y1, x1) -> np.ndarray:
    m = np.zeros((h, w), bool)
    m[y0:y1, x0:x1] = True
    return m


def _counts_of(mask: np.ndarray) -> list[int]:
    """採点用: bool マスクを COCO 非圧縮 RLE の counts（列優先・背景始まり）へ。"""
    flat = mask.reshape(-1, order="F")
    counts: list[int] = []
    val = False  # 最初の run は背景(0)
    run = 0
    for px in flat:
        if bool(px) == val:
            run += 1
        else:
            counts.append(run)
            val = not val
            run = 1
    counts.append(run)
    return counts


def _grade() -> None:
    H, W = 100, 120
    a = _disk(H, W, 50, 50, 25)
    b = _disk(H, W, 52, 52, 24)
    empty = np.zeros((H, W), bool)

    # 採点用 GT/予測（things 1カテゴリぶん）
    gt = [_disk(H, W, 40, 40, 20), _rect(H, W, 20, 80, 60, 110)]
    preds = [
        (_disk(H, W, 41, 41, 19), 0.9),   # gt0 にマッチ → TP
        (_rect(H, W, 22, 82, 58, 108), 0.8),  # gt1 にマッチ → TP
        (_disk(H, W, 80, 20, 10), 0.3),   # どれにも合わない → FP
    ]

    # ex6/ex7 用の PR 列（recall は単調非減少）
    pr_recall = np.array([0.2, 0.4, 0.4, 0.6, 0.8, 1.0])
    pr_precision = np.array([1.0, 1.0, 0.67, 0.75, 0.8, 0.83])

    # ex8 用のマルチクラス GT/予測
    gt_mc = [
        (_disk(H, W, 40, 40, 18), 1),
        (_rect(H, W, 20, 80, 60, 110), 2),
        (_disk(H, W, 75, 30, 12), 1),
    ]
    preds_mc = [
        (_disk(H, W, 41, 41, 17), 1, 0.90),   # cat1 gt0 → TP
        (_rect(H, W, 22, 82, 58, 108), 2, 0.85),  # cat2 gt1 → TP
        (_disk(H, W, 76, 31, 11), 1, 0.70),   # cat1 gt2 → TP
        (_disk(H, W, 10, 10, 8), 1, 0.30),    # cat1 FP
    ]

    # ex9 用のマスク（小さめ）
    rle_mask = _rect(6, 5, 1, 1, 4, 3)
    rle_counts = _counts_of(rle_mask)

    # ex10 用のカテゴリ別 (gt, pred)
    per_cat = {
        1: ([_disk(H, W, 40, 40, 18), _disk(H, W, 75, 30, 12)],
            [_disk(H, W, 41, 41, 17)]),                       # 1 TP, 1 FN
        2: ([_rect(H, W, 20, 80, 60, 110)],
            [_rect(H, W, 22, 82, 58, 108), _rect(H, W, 5, 5, 15, 15)]),  # 1 TP, 1 FP
    }

    results: list[tuple[str, bool, str]] = []

    def check(name: str, fn) -> None:
        try:
            ok, detail = fn()
            results.append((name, bool(ok), detail))
        except NotImplementedError:
            results.append((name, False, "未実装（TODO を埋めてください）"))
        except Exception as e:  # noqa: BLE001
            results.append((name, False, f"例外: {type(e).__name__}: {e}"))

    check("ex1_mask_iou", lambda: (
        np.isclose(ex1_mask_iou(a, b), _sol_ex1(a, b)) and ex1_mask_iou(empty, empty) == 0.0,
        f"IoU={_sol_ex1(a, b):.3f}（空×空=0 も確認）",
    ))

    def _c2():
        prob = np.stack([a.astype(np.float32) * 0.9, b.astype(np.float32) * 0.7])[:, None]  # (2,1,H,W)
        got = np.asarray(ex2_binarize_masks(prob, 0.5))
        ref = _sol_ex2(prob, 0.5)
        return (got.shape == ref.shape and got.dtype == bool and np.array_equal(got, ref),
                f"二値化マスク shape={got.shape} dtype={got.dtype}")
    check("ex2_binarize_masks", _c2)

    check("ex3_match_counts", lambda: (
        tuple(ex3_match_counts(gt, preds, 0.5)) == _sol_ex3(gt, preds, 0.5),
        f"(TP,FP,FN)={_sol_ex3(gt, preds, 0.5)}",
    ))

    def _c4():
        got = tuple(round(float(v), 4) for v in ex4_pq_single_category(gt, [p[0] for p in preds], 0.5))
        ref = tuple(round(float(v), 4) for v in _sol_ex4(gt, [p[0] for p in preds], 0.5))
        return (all(np.isclose(g, r, atol=1e-4) for g, r in zip(got, ref)),
                f"(SQ,RQ,PQ)={ref}")
    check("ex4_pq_single_category", _c4)

    def _c5():
        masks = np.stack([_disk(H, W, 50, 50, 10), a, b])  # (3,H,W)
        scores = np.array([0.2, 0.95, 0.5])
        got = np.asarray(ex5_select_best_sam_mask(masks, scores))
        return (np.array_equal(got, masks[1]), "iou_scores 最大(idx=1)のマスクを選べた")
    check("ex5_select_best_sam_mask", _c5)

    def _c6():
        got = float(ex6_ap_all_point(pr_recall, pr_precision))
        ref = float(_sol_ex6(pr_recall, pr_precision))
        return (np.isclose(got, ref, atol=1e-6), f"all-point AP={ref:.4f}")
    check("ex6_ap_all_point", _c6)

    def _c7():
        got = float(ex7_ap_11point(pr_recall, pr_precision))
        ref = float(_sol_ex7(pr_recall, pr_precision))
        return (np.isclose(got, ref, atol=1e-6), f"11-point AP={ref:.4f}")
    check("ex7_ap_11point", _c7)

    def _c8():
        got = float(ex8_mask_ap50(gt_mc, preds_mc, 0.5))
        ref = float(_sol_ex8(gt_mc, preds_mc, 0.5))
        return (np.isclose(got, ref, atol=1e-4), f"mAP@0.5={ref:.4f}")
    check("ex8_mask_ap50", _c8)

    def _c9():
        got = np.asarray(ex9_rle_decode(rle_counts, 6, 5))
        ref = _sol_ex9(rle_counts, 6, 5)
        sane = np.array_equal(ref, rle_mask)  # 参照復元が元マスクに一致（健全性）
        return (got.shape == ref.shape and got.dtype == bool and np.array_equal(got, ref) and sane,
                f"RLE 復元 shape={got.shape}（counts={rle_counts}）")
    check("ex9_rle_decode", _c9)

    def _c10():
        got = tuple(round(float(v), 4) for v in ex10_pq_overall(per_cat))
        ref = tuple(round(float(v), 4) for v in _sol_ex10(per_cat))
        return (all(np.isclose(g, r, atol=1e-4) for g, r in zip(got, ref)),
                f"全体(SQ,RQ,PQ)={ref}")
    check("ex10_pq_overall", _c10)

    print("=== 採点結果 ===")
    all_ok = True
    n_pass = 0
    for name, ok, detail in results:
        mark = "PASS" if ok else "FAIL"
        all_ok = all_ok and ok
        n_pass += int(ok)
        print(f"  [{mark}] {name:24s} {detail}")
    print(f"\n  {n_pass}/{len(results)} PASS")
    print("ALL PASS 🎉" if all_ok else "まだ未達の演習があります。TODO を埋めましょう。")


# =====================================================================
# 模範解答（SHOW_SOLUTION=1 のときに本体へ差し替える）。まずは自力で！
# exercises_solutions.py はこの _sol_* と _grade を再利用して ALL PASS を出す。
# =====================================================================

def _sol_ex1(a, b):
    a = a.astype(bool)
    b = b.astype(bool)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union > 0 else 0.0


def _sol_ex2(prob, thr=0.5):
    return prob[:, 0] > thr


def _sol_ex3(gt_masks, preds, iou_thr=0.5):
    order = sorted(range(len(preds)), key=lambda i: -preds[i][1])
    matched = [False] * len(gt_masks)
    tp = fp = 0
    for i in order:
        pmask = preds[i][0]
        best_iou, best_j = 0.0, -1
        for j, g in enumerate(gt_masks):
            if matched[j]:
                continue
            iou = _sol_ex1(pmask, g)
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_iou >= iou_thr and best_j >= 0:
            tp += 1
            matched[best_j] = True
        else:
            fp += 1
    fn = matched.count(False)
    return tp, fp, fn


def _sol_ex4(gt_masks, pred_masks, iou_thr=0.5):
    matched = [False] * len(gt_masks)
    iou_sum, tp = 0.0, 0
    for pmask in pred_masks:
        best_iou, best_j = 0.0, -1
        for j, g in enumerate(gt_masks):
            if matched[j]:
                continue
            iou = _sol_ex1(pmask, g)
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_iou > iou_thr and best_j >= 0:
            matched[best_j] = True
            iou_sum += best_iou
            tp += 1
    fp = len(pred_masks) - tp
    fn = len(gt_masks) - tp
    sq = iou_sum / tp if tp > 0 else 0.0
    rq = tp / (tp + 0.5 * fp + 0.5 * fn) if (tp + fp + fn) > 0 else 0.0
    return sq, rq, sq * rq


def _sol_ex5(masks, iou_scores):
    return masks[int(np.argmax(iou_scores))]


def _sol_ex6(recall, precision):
    recall = np.asarray(recall, dtype=float)
    precision = np.asarray(precision, dtype=float)
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(len(mpre) - 1, 0, -1):  # 右から左へ単調非増加（包絡線）にする
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]  # recall が変化した点だけ
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def _sol_ex7(recall, precision):
    recall = np.asarray(recall, dtype=float)
    precision = np.asarray(precision, dtype=float)
    ap = 0.0
    for t in np.linspace(0.0, 1.0, 11):
        m = recall >= t
        p = float(precision[m].max()) if np.any(m) else 0.0
        ap += p / 11.0
    return float(ap)


def _sol_ex8(gt, preds, iou_thr=0.5):
    cats = sorted({c for _, c in gt})
    aps: list[float] = []
    for cat in cats:
        gmasks = [m for m, c in gt if c == cat]
        cpreds = [(m, s) for (m, c, s) in preds if c == cat]
        if not gmasks:
            continue
        order = sorted(range(len(cpreds)), key=lambda i: -cpreds[i][1])
        matched = [False] * len(gmasks)
        tp = np.zeros(len(cpreds))
        fp = np.zeros(len(cpreds))
        for rank, i in enumerate(order):
            pmask = cpreds[i][0]
            best_iou, best_j = 0.0, -1
            for j, g in enumerate(gmasks):
                if matched[j]:
                    continue
                iou = _sol_ex1(pmask, g)
                if iou > best_iou:
                    best_iou, best_j = iou, j
            if best_iou >= iou_thr and best_j >= 0:
                tp[rank] = 1.0
                matched[best_j] = True
            else:
                fp[rank] = 1.0
        tp_cum, fp_cum = np.cumsum(tp), np.cumsum(fp)
        recall = tp_cum / max(len(gmasks), 1)
        precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
        aps.append(_sol_ex6(recall, precision))
    return float(np.mean(aps)) if aps else 0.0


def _sol_ex9(counts, height, width):
    flat = np.zeros(height * width, dtype=bool)
    idx = 0
    val = False
    for c in counts:
        flat[idx:idx + c] = val
        idx += c
        val = not val
    return flat.reshape((height, width), order="F")


def _sol_ex10(per_category):
    sqs: list[float] = []
    rqs: list[float] = []
    pqs: list[float] = []
    for _cat, (gmasks, pmasks) in per_category.items():
        if not gmasks and not pmasks:
            continue
        sq, rq, pq = _sol_ex4(gmasks, pmasks, 0.5)
        sqs.append(sq)
        rqs.append(rq)
        pqs.append(pq)
    mean = lambda xs: float(np.mean(xs)) if xs else 0.0  # noqa: E731
    return mean(sqs), mean(rqs), mean(pqs)


def _install_solutions() -> None:
    """模範解答を本体名へ差し替える（exercises_solutions.py からも呼ぶ）。"""
    g = globals()
    g["ex1_mask_iou"] = _sol_ex1
    g["ex2_binarize_masks"] = _sol_ex2
    g["ex3_match_counts"] = _sol_ex3
    g["ex4_pq_single_category"] = _sol_ex4
    g["ex5_select_best_sam_mask"] = _sol_ex5
    g["ex6_ap_all_point"] = _sol_ex6
    g["ex7_ap_11point"] = _sol_ex7
    g["ex8_mask_ap50"] = _sol_ex8
    g["ex9_rle_decode"] = _sol_ex9
    g["ex10_pq_overall"] = _sol_ex10


if __name__ == "__main__":
    if os.environ.get("SHOW_SOLUTION") == "1":
        print("(模範解答モードで実行します)\n")
        _install_solutions()
    _grade()
