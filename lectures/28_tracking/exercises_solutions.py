"""exercises_solutions.py — 28_tracking 演習の模範解答（実行すると全 PASS）

    uv run python lectures/28_tracking/exercises_solutions.py

exercises.py と同じ採点ハーネスを使い、全問が PASS することを確認する。
各解答は単一責務・早期 return を意識し、追跡/評価の定義式を素直に写している。
"""

from __future__ import annotations

import math


# 問1: 2 ボックスの IoU
def iou_xyxy(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return float(inter / union) if union > 0 else 0.0


# 問2: 中心間ユークリッド距離
def center_distance(a, b) -> float:
    acx, acy = (a[0] + a[2]) / 2, (a[1] + a[3]) / 2
    bcx, bcy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    return float(math.hypot(acx - bcx, acy - bcy))


# 問3: 等速カルマン予測 1 ステップ（中心が (vx,vy) 移動、サイズ不変）
def predict_constant_velocity(box, vx, vy):
    x1, y1, x2, y2 = box
    return [x1 + vx, y1 + vy, x2 + vx, y2 + vy]


# 問4: 貪欲法 + 閾値ゲートで対応付け
def gated_greedy_match(iou, thr):
    # 全セル (i,j,値) を IoU 降順に並べる
    cells = []
    for i, row in enumerate(iou):
        for j, v in enumerate(row):
            cells.append((float(v), i, j))
    cells.sort(reverse=True)

    used_r, used_c, pairs = set(), set(), []
    for v, i, j in cells:
        if v < thr:
            break  # これ以降はすべて閾値未満
        if i in used_r or j in used_c:
            continue
        used_r.add(i)
        used_c.add(j)
        pairs.append((i, j))
    pairs.sort()
    return pairs


# 問5: 1 フレームの FN / FP
def fn_fp_one_frame(gt_boxes, pred_boxes, thr):
    iou = [[iou_xyxy(g, p) for p in pred_boxes] for g in gt_boxes]
    pairs = gated_greedy_match(iou, thr)  # (gt_idx, pred_idx)
    matched_gt = {i for i, _ in pairs}
    matched_pred = {j for _, j in pairs}
    fn = len(gt_boxes) - len(matched_gt)
    fp = len(pred_boxes) - len(matched_pred)
    return fn, fp


# 問6: ID スイッチの総数
def count_idsw(frames):
    prev: dict[int, int] = {}
    idsw = 0
    for matches in frames:
        for gid, pid in matches:
            if gid in prev and prev[gid] != pid:
                idsw += 1
            prev[gid] = pid
    return idsw


# 問7: MOTA
def mota(fn, fp, idsw, gt) -> float:
    if gt == 0:
        return 0.0
    return 1.0 - (fn + fp + idsw) / gt


# 問8: IDF1
def idf1(idtp, idfp, idfn) -> float:
    denom = 2 * idtp + idfp + idfn
    return 2 * idtp / denom if denom > 0 else 0.0


# 問9: HOTA（単一閾値の簡易版）
def hota(tp, fn, fp, assa) -> float:
    denom = tp + fn + fp
    if denom == 0:
        return 0.0
    deta = tp / denom
    return float(math.sqrt(deta * assa))


# ---------------------------------------------------------------------------
# 採点ハーネス（exercises.py と同一）
# ---------------------------------------------------------------------------
def _approx(got, exp, tol=1e-3):
    try:
        return abs(float(got) - float(exp)) <= tol
    except Exception:
        return False


def _eq_pairs(got, exp):
    try:
        return sorted([tuple(map(int, p)) for p in got]) == sorted(exp)
    except Exception:
        return False


def _run(name, fn):
    try:
        ok = fn()
    except Exception as e:  # noqa: BLE001
        print(f"  [FAIL] {name}: 例外 {type(e).__name__}: {e}")
        return False
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    return ok


def main():
    print("=" * 60)
    print("28_tracking 演習 模範解答 自己採点")
    checks = [
        ("問1 iou_xyxy", lambda: _approx(iou_xyxy([0, 0, 10, 10], [5, 5, 15, 15]), 0.142857)
         and _approx(iou_xyxy([0, 0, 10, 10], [20, 20, 30, 30]), 0.0)),
        ("問2 center_distance",
         lambda: _approx(center_distance([0, 0, 10, 10], [10, 10, 20, 20]), math.sqrt(200))),
        ("問3 predict_constant_velocity",
         lambda: list(map(float, predict_constant_velocity([0, 0, 10, 10], 2, 3)))
         == [2.0, 3.0, 12.0, 13.0]),
        ("問4 gated_greedy_match",
         lambda: _eq_pairs(gated_greedy_match([[0.8, 0.1], [0.2, 0.7]], 0.5), [(0, 0), (1, 1)])
         and _eq_pairs(gated_greedy_match([[0.4, 0.6], [0.5, 0.1]], 0.5), [(0, 1), (1, 0)])),
        ("問5 fn_fp_one_frame",
         lambda: tuple(fn_fp_one_frame(
             [[0, 0, 10, 10], [20, 20, 30, 30], [40, 40, 50, 50]],
             [[0, 0, 9, 9], [100, 100, 110, 110]], 0.3)) == (2, 1)),
        ("問6 count_idsw",
         lambda: count_idsw([[(0, 10), (1, 11)], [(0, 10), (1, 12)], [(0, 13), (1, 12)]]) == 2),
        ("問7 mota", lambda: _approx(mota(10, 5, 3, 100), 0.82)),
        ("問8 idf1", lambda: _approx(idf1(80, 10, 20), 0.842105)),
        ("問9 hota", lambda: _approx(hota(80, 10, 10, 0.9), math.sqrt(0.8 * 0.9))),
    ]
    passed = sum(_run(n, f) for n, f in checks)
    print("-" * 60)
    print(f"合計: {passed}/{len(checks)} PASS")
    assert passed == len(checks), "模範解答が全 PASS しません（バグ）"
    print("全問 PASS。")


if __name__ == "__main__":
    main()
