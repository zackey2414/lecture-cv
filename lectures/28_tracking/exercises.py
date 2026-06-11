"""exercises.py — 28_tracking 演習（9 問・易→難・自己採点つき）

各 TODO 関数を実装し、保存して実行すると自動採点されます:

    uv run python lectures/28_tracking/exercises.py

- 未実装でも **exit 0** で終了します（PASS/FAIL の一覧が出るだけ）。
- 模範解答は exercises_solutions.py（実行すると全 PASS）。
- ねらい: IoU → 対応付け → カルマン予測 → 検出誤りの数え方 → ID スイッチ →
  MOTA/IDF1/HOTA まで、追跡と評価の核を自分の手で書けるようにする。
"""

from __future__ import annotations

import math


# ============================================================================
# 問1（易）: 2 ボックスの IoU
#   box は [x1,y1,x2,y2]。交差面積 / 和集合面積 を返す。重ならなければ 0。
# ============================================================================
def iou_xyxy(a, b) -> float:
    # TODO: 交差幅 = min(ax2,bx2)-max(ax1,bx1) を 0 でクリップ、高さも同様。
    #       union = areaA + areaB - inter。union<=0 なら 0.0 を返す。
    raise NotImplementedError


# ============================================================================
# 問2（易）: 2 ボックス中心間のユークリッド距離
#   中心 = ((x1+x2)/2, (y1+y2)/2)。距離 = sqrt(dx^2 + dy^2)。
# ============================================================================
def center_distance(a, b) -> float:
    # TODO: それぞれの中心を求めて距離を返す。
    raise NotImplementedError


# ============================================================================
# 問3（易〜中）: 等速モデルのカルマン「予測」1 ステップ
#   速度 (vx,vy) で中心が動く。サイズは不変。
#   box=[x1,y1,x2,y2] を (vx,vy) だけ平行移動した新しい box を返す。
# ============================================================================
def predict_constant_velocity(box, vx, vy):
    # TODO: 4 座標すべてに (vx,vy) を足す（x 座標に vx、y 座標に vy）。
    raise NotImplementedError


# ============================================================================
# 問4（中）: IoU 行列を「貪欲法 + 閾値ゲート」で対応付ける
#   iou は行=予測、列=検出 の 2 次元リスト/配列。
#   IoU が大きい組から順に確定し、すでに使った行/列は飛ばす。
#   IoU < thr の組は採用しない。返り値は (row, col) のリスト（row 昇順でソート）。
# ============================================================================
def gated_greedy_match(iou, thr):
    # TODO: 全セルを IoU 降順に並べ、未使用の行・列なら採用（iou>=thr のときだけ）。
    raise NotImplementedError


# ============================================================================
# 問5（中）: 1 フレームの FN / FP を数える
#   gt_boxes, pred_boxes は box のリスト。IoU>=thr で貪欲対応付け。
#   FN = 対応が付かなかった GT 数、FP = 対応が付かなかった予測数。
#   返り値: (fn, fp)。問1, 問4 を再利用してよい。
# ============================================================================
def fn_fp_one_frame(gt_boxes, pred_boxes, thr):
    # TODO: iou 行列を作り gated_greedy_match で対応付け、余りを数える。
    raise NotImplementedError


# ============================================================================
# 問6（中〜難）: ID スイッチ(IDSW) の総数
#   frames は各フレームの対応リスト: [[(gt_id, pred_id), ...], ...]
#   ある gt_id に対し、前回対応した pred_id と今回が違えば +1。
#   （対応が無いフレームを挟んでも「直近の pred_id」を覚えておく＝標準仕様）
# ============================================================================
def count_idsw(frames):
    # TODO: prev = {} に gt_id->pred_id を保持しつつ、変化を数える。
    raise NotImplementedError


# ============================================================================
# 問7（中）: MOTA
#   MOTA = 1 - (FN + FP + IDSW) / GT。GT==0 のときは 0.0。
# ============================================================================
def mota(fn, fp, idsw, gt) -> float:
    # TODO: 定義式どおりに。
    raise NotImplementedError


# ============================================================================
# 問8（難）: IDF1
#   IDF1 = 2*IDTP / (2*IDTP + IDFP + IDFN)。分母 0 のときは 0.0。
# ============================================================================
def idf1(idtp, idfp, idfn) -> float:
    # TODO: 定義式どおりに。
    raise NotImplementedError


# ============================================================================
# 問9（難）: HOTA（単一 IoU 閾値の簡易版）
#   DetA = tp / (tp + fn + fp)、HOTA = sqrt(DetA * AssA)。
#   tp+fn+fp==0 のときは 0.0。assa は引数で与えられる（0..1）。
# ============================================================================
def hota(tp, fn, fp, assa) -> float:
    # TODO: DetA を計算し、sqrt(DetA*AssA) を返す。
    raise NotImplementedError


# ============================================================================
# 自己採点ハーネス（編集不要）。未実装は FAIL になるが exit 0 で終了する。
# ============================================================================
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
    except NotImplementedError:
        print(f"  [TODO] {name}: 未実装")
        return False
    except Exception as e:  # noqa: BLE001
        print(f"  [FAIL] {name}: 例外 {type(e).__name__}: {e}")
        return False
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    return ok


def main():
    print("=" * 60)
    print("28_tracking 演習 自己採点")
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
    if passed == len(checks):
        print("全問正解！ 追跡と MOT 評価の核を自力で書けています。")


if __name__ == "__main__":
    main()
