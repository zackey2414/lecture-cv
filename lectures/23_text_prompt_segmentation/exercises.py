"""第23回 演習問題（テキストプロンプト/参照セグメンテーション）。

使い方:
  1. 各 exN_*() の TODO を自分で実装する（最初は NotImplementedError だが、
     採点ランナーが拾うのでプロセスは落ちず、FAIL と表示されるだけ）。
  2. 自己採点:  uv run python lectures/23_text_prompt_segmentation/exercises.py
     全問 pass すれば "ALL PASS" と表示される。
  3. 詰まったら模範解答の挙動を見る:
        SHOW_SOLUTION=1 uv run python lectures/23_text_prompt_segmentation/exercises.py
     または全問の模範解答を一括実行:
        uv run python lectures/23_text_prompt_segmentation/exercises_solutions.py

この9問は本モジュールの核を易→難で1つずつ抜き出したもの（モデルDL不要・純計算）:
  ex1: ロジット → sigmoid → 閾値で2値マスク化（CLIPSeg の出力後処理）        … 01    [易]
  ex2: マスク IoU = |∩| / |∪|（参照セグメの主指標）                          … 01/03 [易]
  ex3: マスク Dice = 2|∩| / (|P|+|G|)（= F1, 医用頻出）                       … 01/03 [易]
  ex4: しきい値スイープで IoU 最大点を探す（CLIPSeg の最適閾値）              … 03    [中]
  ex5: box IoU（Grounding DINO の検出 box を GT に対応付ける土台）            … 02/03 [中]
  ex6: mIoU（複数オブジェクトの IoU 平均。セマンティックセグメの主指標）     … 全般  [中]
  ex7: pixel 単位の precision/recall（マスクの過剰/塗り残しを切り分ける）     … 03    [中]
  ex8: 貪欲マッチング（検出 box → GT を score 降順で対応付け TP/FP/FN）      … 02/03 [難]
  ex9: Average Precision（PR 曲線の全点補間。検出評価の総仕上げ）            … 19/03 [難]

ex8/ex9 は『検出を GT に対応付けて精度を測る』検出評価の核で、Grounded-SAM の段1
（box 閾値と recall のトレードオフ）を数式で裏打ちする良問。ex5(box IoU) を土台に、
貪欲マッチング → TP/FP/FN → precision/recall → AP補間 と積み上げる。
"""

from __future__ import annotations

import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))


# =====================================================================
# 演習（ここを実装する）
# =====================================================================

def ex1_logits_to_mask(logits: np.ndarray, threshold: float) -> np.ndarray:
    """演習1[易]: CLIPSeg のロジットを sigmoid で確率化し、threshold 以上を前景にした bool マスクを返す。

    - logits: 任意形の float 配列（CLIPSeg の生出力に相当）。
    - sigmoid(x) = 1 / (1 + exp(-x)) で 0〜1 の確率にする。
    - 返り値は logits と同じ形の bool 配列（prob >= threshold が True）。
    - ヒント: np.exp を使う。境界は『以上(>=)』で判定する。
    """
    # TODO: sigmoid を計算し、threshold 以上を True にした bool 配列を返す
    raise NotImplementedError


def ex2_mask_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    """演習2[易]: 2値マスクの IoU = |∩| / |∪| を返す。

    - pred, gt: 同じ形の bool（または 0/1）配列。
    - 交差 = 両方 True の画素数、和集合 = どちらか True の画素数。
    - 和集合が 0（両方とも空）のときは 1.0 を返す（完全一致扱い・ゼロ割回避）。
    """
    # TODO: np.logical_and / np.logical_or で交差と和集合を数え、その比を返す
    raise NotImplementedError


def ex3_mask_dice(pred: np.ndarray, gt: np.ndarray) -> float:
    """演習3[易]: 2値マスクの Dice = 2|∩| / (|P| + |G|) を返す（= F1 スコア）。

    - pred, gt: 同じ形の bool（または 0/1）配列。
    - 分母 |P|+|G| が 0 のときは 1.0 を返す（ゼロ割回避）。
    - IoU より「重なり」を甘く評価する（同じ重なりでも Dice >= IoU）。
    """
    # TODO: 交差を 2 倍し、(pred の画素数 + gt の画素数) で割って返す
    raise NotImplementedError


def ex4_best_threshold(prob: np.ndarray, gt: np.ndarray, thresholds: np.ndarray) -> tuple[float, float]:
    """演習4[中]: 確率マップ prob を thresholds で2値化し、IoU 最大のしきい値とその IoU を返す。

    - prob: 0〜1 の float 配列、gt: 同形の bool 配列、thresholds: 走査するしきい値の1次元配列。
    - 各 t について (prob >= t) と gt の IoU を計算し、最大の (t, IoU) を返す。
    - 返り値は (best_threshold: float, best_iou: float)。ex2 を使ってよい。
    - 同点のときは『先に出た（小さい）しきい値』を採用する。
    """
    # TODO: thresholds を走査し IoU 最大の (しきい値, IoU) を返す
    raise NotImplementedError


def ex5_box_iou(a: list[float], b: list[float]) -> float:
    """演習5[中]: 2つの box [x0, y0, x1, y1] の IoU を返す（Grounded-SAM の box→GT 対応付けに使う）。

    - 各 box は左上 (x0,y0)・右下 (x1,y1) の絶対座標。
    - 交差矩形の幅・高さは負にならないよう 0 でクランプする（重ならなければ交差 0）。
    - 和集合 = 面積A + 面積B - 交差。和集合が 0 なら 0.0 を返す。
    """
    # TODO: 交差矩形の面積を求め、IoU = 交差 / (A + B - 交差) を返す
    raise NotImplementedError


def ex6_mean_iou(preds: list[np.ndarray], gts: list[np.ndarray]) -> float:
    """演習6[中]: 複数オブジェクトの IoU 平均（mIoU）を返す。

    - preds, gts: 同じ長さのマスクのリスト（preds[i] と gts[i] が対応）。
    - 各ペアの IoU（ex2 と同じ定義）を計算し、その平均（単純平均）を返す。
    - 空リスト（要素 0）のときは 0.0 を返す。
    - mIoU はセマンティックセグメ（第21回）の主指標。クラス（オブジェクト）ごとの
      IoU を平均することで、得意/不得意クラスを均等に評価する。
    """
    # TODO: ex2 の IoU を各ペアで計算し、その平均を返す（空なら 0.0）
    raise NotImplementedError


def ex7_pixel_pr(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    """演習7[中]: 予測マスクの pixel 単位 precision / recall を返す。

    - TP = pred かつ gt（正しく塗れた画素）、FP = pred かつ not gt（塗り過ぎ）、
      FN = not pred かつ gt（塗り残し）。
    - precision = TP / (TP + FP) = TP / |pred|。予測が空(|pred|=0)なら 1.0。
    - recall    = TP / (TP + FN) = TP / |gt|。  正解が空(|gt|=0)なら 1.0。
    - 返り値は (precision, recall)。precision が低い=塗り過ぎ、recall が低い=塗り残し。
    """
    # TODO: TP/|pred| と TP/|gt| を計算して (precision, recall) を返す
    raise NotImplementedError


def ex8_greedy_match(
    det_boxes: list[list[float]], scores: list[float], gt_boxes: list[list[float]], iou_thr: float
) -> tuple[int, int, int]:
    """演習8[難]: 検出 box を GT に貪欲マッチングし (TP, FP, FN) を返す（検出評価の核）。

    手順（物体検出 mAP の前段処理そのもの）:
      1. 検出を score の降順に並べる。
      2. 上位から順に、まだ未マッチの GT の中で IoU が最大のものを探す。
      3. その最大 IoU が iou_thr 以上なら TP（その GT を『使用済み』にする）、
         未満なら FP（背景を検出した扱い）。
      4. 1つの GT に対応できる検出は1つだけ（残りは FP）。
      5. 最後まで未マッチで残った GT の数が FN。
    返り値: (tp, fp, fn)。ex5_box_iou を使ってよい。
    """
    # TODO: score 降順に貪欲マッチングして (tp, fp, fn) を返す
    raise NotImplementedError


def ex9_average_precision(scores: list[float], tp_flags: list[int], n_gt: int) -> float:
    """演習9[難]: PR 曲線の全点補間で Average Precision (AP) を返す。

    入力（ex8 のマッチング結果から作る想定）:
      - scores:   各検出の confidence。
      - tp_flags: 各検出が TP なら 1、FP なら 0（scores と同じ並び・同じ長さ）。
      - n_gt:     GT の総数（recall の分母）。
    手順:
      1. score 降順に並べ、tp/fp を累積（cumsum）する。
      2. recall    = tp累積 / n_gt、precision = tp累積 / (tp累積 + fp累積)。
      3. 全点補間: precision を右から見て単調非増加にする（envelope）。
         np.maximum.accumulate(precision[::-1])[::-1] が定石。
      4. AP = Σ (recall[i] - recall[i-1]) * precision_envelope[i]（recall[-1]=0 始まり）。
    n_gt が 0 のときは 0.0 を返す。
    """
    # TODO: PR 曲線を作り、全点補間で AP を返す（n_gt==0 なら 0.0）
    raise NotImplementedError


# =====================================================================
# 自己採点ランナー（exercises_solutions.py からも再利用される）
# =====================================================================

def _grade() -> bool:
    """全演習を採点し、結果を表示して all_ok を返す（exit code には使わない）。"""
    results: list[tuple[str, bool, str]] = []

    def check(name: str, fn) -> None:
        try:
            ok, detail = fn()
            results.append((name, bool(ok), detail))
        except NotImplementedError:
            results.append((name, False, "未実装（TODOを埋めてください）"))
        except Exception as e:  # noqa: BLE001
            results.append((name, False, f"例外: {type(e).__name__}: {e}"))

    # ex1: 既知ロジット → sigmoid → 閾値。sigmoid(0)=0.5 は『以上』で前景に入る。
    def _c1():
        logits = np.array([[-2.0, 0.0, 2.0]])
        got = ex1_logits_to_mask(logits, 0.5)
        ref = (1.0 / (1.0 + np.exp(-logits))) >= 0.5  # [[False, True, True]]
        return got.dtype == bool and np.array_equal(got, ref), f"mask={got.tolist()}"
    check("ex1_logits_to_mask", _c1)

    # ex2: 交差2・和集合6 → IoU=1/3。両方空 → 1.0。
    def _c2():
        pred = np.array([[1, 1, 0], [1, 1, 0], [0, 0, 0]], dtype=bool)  # 4 画素
        gt = np.array([[0, 1, 1], [0, 1, 1], [0, 0, 0]], dtype=bool)    # 4 画素, 交差2, 和6
        v = ex2_mask_iou(pred, gt)
        empty = ex2_mask_iou(np.zeros((2, 2), bool), np.zeros((2, 2), bool))
        return abs(v - 1 / 3) < 1e-6 and empty == 1.0, f"IoU={v:.3f}, empty={empty}"
    check("ex2_mask_iou", _c2)

    # ex3: 同じ配置で Dice=2*2/(4+4)=0.5。
    def _c3():
        pred = np.array([[1, 1, 0], [1, 1, 0], [0, 0, 0]], dtype=bool)
        gt = np.array([[0, 1, 1], [0, 1, 1], [0, 0, 0]], dtype=bool)
        v = ex3_mask_dice(pred, gt)
        empty = ex3_mask_dice(np.zeros((2, 2), bool), np.zeros((2, 2), bool))
        return abs(v - 0.5) < 1e-6 and empty == 1.0, f"Dice={v:.3f}, empty={empty}"
    check("ex3_mask_dice", _c3)

    # ex4: 中央が高確率の prob。閾値 0.5〜0.8 で中央のみ True になり IoU 最大になる。
    def _c4():
        prob = np.array([[0.1, 0.4, 0.1], [0.4, 0.9, 0.4], [0.1, 0.4, 0.1]])
        gt = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=bool)  # 中央1画素だけ正解
        ths = np.round(np.linspace(0.1, 0.9, 9), 3)
        bt, bv = ex4_best_threshold(prob, gt, ths)
        return abs(bv - 1.0) < 1e-9 and 0.5 <= bt <= 0.8, f"best_t={bt:.2f}, best_IoU={bv:.3f}"
    check("ex4_best_threshold", _c4)

    # ex5: 同サイズ box が半分重なる例 → IoU=1/3。離れていれば 0。
    def _c5():
        a = [0, 0, 2, 2]   # 面積4
        b = [1, 0, 3, 2]   # 面積4, 交差=1*2=2, 和=4+4-2=6 → 2/6=1/3
        v = ex5_box_iou(a, b)
        far = ex5_box_iou([0, 0, 1, 1], [5, 5, 6, 6])  # 重ならない → 0
        return abs(v - 1 / 3) < 1e-6 and far == 0.0, f"IoU={v:.3f}, far={far}"
    check("ex5_box_iou", _c5)

    # ex6: IoU=1.0 と IoU=1/3 の2ペア → mIoU=(1+1/3)/2=2/3。空リスト → 0.0。
    def _c6():
        p1 = np.array([[1, 1], [0, 0]], bool); g1 = np.array([[1, 1], [0, 0]], bool)       # IoU 1.0
        p2 = np.array([[1, 1, 0], [0, 0, 0]], bool); g2 = np.array([[0, 1, 1], [0, 0, 0]], bool)  # IoU 1/3
        v = ex6_mean_iou([p1, p2], [g1, g2])
        empty = ex6_mean_iou([], [])
        return abs(v - 2 / 3) < 1e-6 and empty == 0.0, f"mIoU={v:.3f}, empty={empty}"
    check("ex6_mean_iou", _c6)

    # ex7: pred 4画素・gt 4画素・交差2 → precision=0.5, recall=0.5。
    def _c7():
        pred = np.array([[1, 1, 0], [1, 1, 0], [0, 0, 0]], dtype=bool)
        gt = np.array([[0, 1, 1], [0, 1, 1], [0, 0, 0]], dtype=bool)
        prec, rec = ex7_pixel_pr(pred, gt)
        # 予測が空なら precision=1.0（FP なし）
        pe, _ = ex7_pixel_pr(np.zeros((2, 2), bool), np.ones((2, 2), bool))
        return abs(prec - 0.5) < 1e-6 and abs(rec - 0.5) < 1e-6 and pe == 1.0, \
            f"P={prec:.3f}, R={rec:.3f}, empty_pred_P={pe}"
    check("ex7_pixel_pr", _c7)

    # ex8: GT2個。det0=完全一致(TP), det1=gt0と被るが既に使用済み→FP, det2=gt1一致(TP)。
    def _c8():
        gt = [[0, 0, 10, 10], [20, 20, 30, 30]]
        det = [[0, 0, 10, 10], [0, 0, 9, 9], [20, 20, 30, 30]]
        sc = [0.9, 0.8, 0.7]
        tp, fp, fn = ex8_greedy_match(det, sc, gt, 0.5)
        return (tp, fp, fn) == (2, 1, 0), f"tp={tp}, fp={fp}, fn={fn}"
    check("ex8_greedy_match", _c8)

    # ex9: scores=[.9,.8,.7,.6], tp=[1,1,0,1], n_gt=3 → AP = 1/3+1/3+1/4 = 11/12 ≈ 0.9167。
    def _c9():
        ap = ex9_average_precision([0.9, 0.8, 0.7, 0.6], [1, 1, 0, 1], 3)
        zero = ex9_average_precision([0.9], [1], 0)
        return abs(ap - 11 / 12) < 1e-3 and zero == 0.0, f"AP={ap:.4f}, n_gt0={zero}"
    check("ex9_average_precision", _c9)

    print("=== 採点結果 ===")
    all_ok = True
    for name, ok, detail in results:
        mark = "PASS" if ok else "FAIL"
        all_ok = all_ok and ok
        print(f"  [{mark}] {name:22s} {detail}")
    print("\nALL PASS 🎉" if all_ok else "\nまだ未達の演習があります。TODO を埋めましょう。")
    return all_ok


# =====================================================================
# 模範解答（SHOW_SOLUTION=1 / exercises_solutions.py から本体へ差し替える）
# まずは自力で解いてから見ること。
# =====================================================================

def _sol_ex1(logits: np.ndarray, threshold: float) -> np.ndarray:
    prob = 1.0 / (1.0 + np.exp(-logits))
    return prob >= threshold


def _sol_ex2(pred: np.ndarray, gt: np.ndarray) -> float:
    p = pred.astype(bool)
    g = gt.astype(bool)
    union = np.logical_or(p, g).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(p, g).sum()) / float(union)


def _sol_ex3(pred: np.ndarray, gt: np.ndarray) -> float:
    p = pred.astype(bool)
    g = gt.astype(bool)
    denom = p.sum() + g.sum()
    if denom == 0:
        return 1.0
    return 2.0 * float(np.logical_and(p, g).sum()) / float(denom)


def _sol_ex4(prob: np.ndarray, gt: np.ndarray, thresholds: np.ndarray) -> tuple[float, float]:
    best_t, best_v = float(thresholds[0]), -1.0
    for t in thresholds:
        v = _sol_ex2(prob >= t, gt)
        if v > best_v:
            best_v, best_t = v, float(t)
    return best_t, best_v


def _sol_ex5(a: list[float], b: list[float]) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter) / float(union) if union > 0 else 0.0


def _sol_ex6(preds: list[np.ndarray], gts: list[np.ndarray]) -> float:
    if len(preds) == 0:
        return 0.0
    ious = [_sol_ex2(p, g) for p, g in zip(preds, gts)]
    return float(np.mean(ious))


def _sol_ex7(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    p = pred.astype(bool)
    g = gt.astype(bool)
    tp = float(np.logical_and(p, g).sum())
    pred_sum = float(p.sum())
    gt_sum = float(g.sum())
    precision = tp / pred_sum if pred_sum > 0 else 1.0
    recall = tp / gt_sum if gt_sum > 0 else 1.0
    return precision, recall


def _sol_ex8(det_boxes, scores, gt_boxes, iou_thr) -> tuple[int, int, int]:
    order = np.argsort(-np.asarray(scores, dtype=float))  # score 降順
    matched = [False] * len(gt_boxes)
    tp = fp = 0
    for i in order:
        best_j, best_iou = -1, 0.0
        for j, gb in enumerate(gt_boxes):
            if matched[j]:
                continue
            v = _sol_ex5(det_boxes[i], gb)
            if v > best_iou:
                best_iou, best_j = v, j
        if best_j >= 0 and best_iou >= iou_thr:
            tp += 1
            matched[best_j] = True
        else:
            fp += 1
    fn = matched.count(False)
    return tp, fp, fn


def _sol_ex9(scores, tp_flags, n_gt) -> float:
    if n_gt == 0:
        return 0.0
    order = np.argsort(-np.asarray(scores, dtype=float))
    tp = np.asarray(tp_flags, dtype=float)[order]
    fp = 1.0 - tp
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recall = tp_cum / float(n_gt)
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)
    # 全点補間: 右から見て precision を単調非増加にする
    prec_env = np.maximum.accumulate(precision[::-1])[::-1]
    ap, prev_r = 0.0, 0.0
    for r, pr in zip(recall, prec_env):
        ap += (r - prev_r) * pr
        prev_r = r
    return float(ap)


def _install_solutions() -> None:
    g = globals()
    g["ex1_logits_to_mask"] = _sol_ex1
    g["ex2_mask_iou"] = _sol_ex2
    g["ex3_mask_dice"] = _sol_ex3
    g["ex4_best_threshold"] = _sol_ex4
    g["ex5_box_iou"] = _sol_ex5
    g["ex6_mean_iou"] = _sol_ex6
    g["ex7_pixel_pr"] = _sol_ex7
    g["ex8_greedy_match"] = _sol_ex8
    g["ex9_average_precision"] = _sol_ex9


if __name__ == "__main__":
    if os.environ.get("SHOW_SOLUTION") == "1":
        print("(模範解答モードで実行します)\n")
        _install_solutions()
    _grade()
