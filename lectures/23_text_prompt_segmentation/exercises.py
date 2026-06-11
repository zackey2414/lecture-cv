"""第23回 演習問題（テキストプロンプト/参照セグメンテーション）。

使い方:
  1. 各 exN_*() の TODO を自分で実装する（最初は NotImplementedError だが、
     採点ランナーが拾うのでプロセスは落ちず、FAIL と表示されるだけ）。
  2. 自己採点:  uv run python lectures/23_text_prompt_segmentation/exercises.py
     全問 pass すれば "ALL PASS" と表示される。
  3. 詰まったら模範解答の挙動を見る:
        SHOW_SOLUTION=1 uv run python lectures/23_text_prompt_segmentation/exercises.py

この5問は本モジュール3スクリプトの核を1つずつ抜き出したもの（モデルDL不要・純計算）:
  ex1: ロジット → sigmoid → 閾値で2値マスク化（CLIPSeg の出力後処理）        … 01
  ex2: マスク IoU = |∩| / |∪|（参照セグメの主指標）                          … 01/03
  ex3: マスク Dice = 2|∩| / (|P|+|G|)（= F1, 医用頻出）                       … 01/03
  ex4: しきい値スイープで IoU 最大点を探す（CLIPSeg の最適閾値）              … 03
  ex5: box IoU（Grounding DINO の検出 box を GT に対応付ける土台）            … 02/03
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
    """演習1: CLIPSeg のロジットを sigmoid で確率化し、threshold 以上を前景にした bool マスクを返す。

    - logits: 任意形の float 配列（CLIPSeg の生出力に相当）。
    - sigmoid(x) = 1 / (1 + exp(-x)) で 0〜1 の確率にする。
    - 返り値は logits と同じ形の bool 配列（prob >= threshold が True）。
    - ヒント: np.exp を使う。境界は『以上(>=)』で判定する。
    """
    # TODO: sigmoid を計算し、threshold 以上を True にした bool 配列を返す
    raise NotImplementedError


def ex2_mask_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    """演習2: 2値マスクの IoU = |∩| / |∪| を返す。

    - pred, gt: 同じ形の bool（または 0/1）配列。
    - 交差 = 両方 True の画素数、和集合 = どちらか True の画素数。
    - 和集合が 0（両方とも空）のときは 1.0 を返す（完全一致扱い・ゼロ割回避）。
    """
    # TODO: np.logical_and / np.logical_or で交差と和集合を数え、その比を返す
    raise NotImplementedError


def ex3_mask_dice(pred: np.ndarray, gt: np.ndarray) -> float:
    """演習3: 2値マスクの Dice = 2|∩| / (|P| + |G|) を返す（= F1 スコア）。

    - pred, gt: 同じ形の bool（または 0/1）配列。
    - 分母 |P|+|G| が 0 のときは 1.0 を返す（ゼロ割回避）。
    - IoU より「重なり」を甘く評価する（同じ重なりでも Dice >= IoU）。
    """
    # TODO: 交差を 2 倍し、(pred の画素数 + gt の画素数) で割って返す
    raise NotImplementedError


def ex4_best_threshold(prob: np.ndarray, gt: np.ndarray, thresholds: np.ndarray) -> tuple[float, float]:
    """演習4: 確率マップ prob を thresholds で2値化し、IoU 最大のしきい値とその IoU を返す。

    - prob: 0〜1 の float 配列、gt: 同形の bool 配列、thresholds: 走査するしきい値の1次元配列。
    - 各 t について (prob >= t) と gt の IoU を計算し、最大の (t, IoU) を返す。
    - 返り値は (best_threshold: float, best_iou: float)。ex2 を使ってよい。
    - 同点のときは『先に出た（小さい）しきい値』を採用する。
    """
    # TODO: thresholds を走査し IoU 最大の (しきい値, IoU) を返す
    raise NotImplementedError


def ex5_box_iou(a: list[float], b: list[float]) -> float:
    """演習5: 2つの box [x0, y0, x1, y1] の IoU を返す（Grounded-SAM の box→GT 対応付けに使う）。

    - 各 box は左上 (x0,y0)・右下 (x1,y1) の絶対座標。
    - 交差矩形の幅・高さは負にならないよう 0 でクランプする（重ならなければ交差 0）。
    - 和集合 = 面積A + 面積B - 交差。和集合が 0 なら 0.0 を返す。
    """
    # TODO: 交差矩形の面積を求め、IoU = 交差 / (A + B - 交差) を返す
    raise NotImplementedError


# =====================================================================
# 自己採点ランナー
# =====================================================================

def _grade() -> None:
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

    # ex4: 中央が高確率の prob。閾値 0.5 付近で IoU 最大になる。
    def _c4():
        prob = np.array([[0.1, 0.4, 0.1], [0.4, 0.9, 0.4], [0.1, 0.4, 0.1]])
        gt = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=bool)  # 中央1画素だけ正解
        ths = np.round(np.linspace(0.1, 0.9, 9), 3)
        bt, bv = ex4_best_threshold(prob, gt, ths)
        # 閾値 0.5〜0.8 のどこかで中央のみ True になり IoU=1.0 になるはず。
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

    print("=== 採点結果 ===")
    all_ok = True
    for name, ok, detail in results:
        mark = "PASS" if ok else "FAIL"
        all_ok = all_ok and ok
        print(f"  [{mark}] {name:22s} {detail}")
    print("\nALL PASS 🎉" if all_ok else "\nまだ未達の演習があります。TODO を埋めましょう。")


# =====================================================================
# 模範解答（SHOW_SOLUTION=1 のときに本体へ差し替えて実行）
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


def _install_solutions() -> None:
    g = globals()
    g["ex1_logits_to_mask"] = _sol_ex1
    g["ex2_mask_iou"] = _sol_ex2
    g["ex3_mask_dice"] = _sol_ex3
    g["ex4_best_threshold"] = _sol_ex4
    g["ex5_box_iou"] = _sol_ex5


if __name__ == "__main__":
    if os.environ.get("SHOW_SOLUTION") == "1":
        print("(模範解答モードで実行します)\n")
        _install_solutions()
    _grade()
