"""第22回 演習問題（インスタンス/パノプティック/SAM の“肝”を手で書く）。

使い方:
  1. 各 exN_*() の中の TODO を自分で実装する（最初は NotImplementedError で FAIL）。
  2. 自己採点を実行（未実装でも例外で落ちず、PASS/FAIL を一覧表示して必ず正常終了する）:
         uv run python lectures/22_instance_panoptic_sam/exercises.py
  3. どうしても分からない時は、模範解答の挙動を見る:
         SHOW_SOLUTION=1 uv run python lectures/22_instance_panoptic_sam/exercises.py

狙い: この回の指標（mask IoU / mask AP のマッチング / PQ）と、モデル出力の正しい
      後処理（Mask R-CNN の確率マスク二値化・SAM の3マスクからの選択）を、
      ライブラリに頼らず numpy/torch だけで組み立てられるようにする。
本演習はモデルもネットも不要（合成マスクだけで完結する）。
"""

from __future__ import annotations

import os

import numpy as np


# =====================================================================
# 演習（ここを実装する）
# =====================================================================

def ex1_mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    """演習1: 2つの bool マスク a, b の IoU = 交差画素 / 和集合画素 を返す。

    和集合が 0（両方とも空）のときは 0.0 を返す。
    ヒント: np.logical_and / np.logical_or の .sum()。
    """
    # TODO: IoU を計算して float で返す
    raise NotImplementedError


def ex2_binarize_masks(prob: np.ndarray, thr: float = 0.5) -> np.ndarray:
    """演習2: Mask R-CNN の masks 出力 (N, 1, H, W) 確率を (N, H, W) bool に二値化する。

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
    """演習3: 予測を score 降順に GT へ貪欲マッチし (TP, FP, FN) を返す（mask AP の前段）。

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
    """演習4: 1カテゴリの (SQ, RQ, PQ) を返す。

    IoU>iou_thr を満たす GT-予測を一意マッチ（>0.5 なので相手は高々1つ）。
      SQ = マッチ組の平均 IoU（TP が0なら0）
      RQ = TP / (TP + 0.5*FP + 0.5*FN)（分母0なら0）
      PQ = SQ * RQ
    ヒント: TP のときに IoU を足し込み、最後に TP で割って SQ。
    """
    # TODO: (sq, rq, pq) を返す
    raise NotImplementedError


def ex5_select_best_sam_mask(masks: np.ndarray, iou_scores: np.ndarray) -> np.ndarray:
    """演習5: SAM が返す3枚のマスクから iou_scores 最大の1枚 (H, W) を返す。

    SAM は曖昧性を考慮して1プロンプトにつき3マスクを返す。iou_scores（予測品質）が
    最大のインデックスを選ぶ。masks 形状は (3, H, W)、iou_scores は (3,)。
    ヒント: np.argmax(iou_scores)。
    """
    # TODO: best な (H, W) マスクを返す
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

    print("=== 採点結果 ===")
    all_ok = True
    for name, ok, detail in results:
        mark = "PASS" if ok else "FAIL"
        all_ok = all_ok and ok
        print(f"  [{mark}] {name:26s} {detail}")
    print("\nALL PASS 🎉" if all_ok else "\nまだ未達の演習があります。TODO を埋めましょう。")


# =====================================================================
# 模範解答（SHOW_SOLUTION=1 のときに本体へ差し替える）。まずは自力で！
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


def _install_solutions() -> None:
    g = globals()
    g["ex1_mask_iou"] = _sol_ex1
    g["ex2_binarize_masks"] = _sol_ex2
    g["ex3_match_counts"] = _sol_ex3
    g["ex4_pq_single_category"] = _sol_ex4
    g["ex5_select_best_sam_mask"] = _sol_ex5


if __name__ == "__main__":
    if os.environ.get("SHOW_SOLUTION") == "1":
        print("(模範解答モードで実行します)\n")
        _install_solutions()
    _grade()