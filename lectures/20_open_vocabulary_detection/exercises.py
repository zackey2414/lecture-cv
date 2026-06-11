"""第20回 演習問題（オープン語彙物体検出の核を純計算で）。

使い方:
  1. 各 exN_*() の TODO を実装する（最初は NotImplementedError だが、採点ランナーが
     拾うのでプロセスは落ちず、FAIL と表示されるだけ。必ず exit 0）。
  2. 自己採点:  uv run python lectures/20_open_vocabulary_detection/exercises.py
     全問 pass すれば "ALL PASS" と表示される。
  3. 詰まったら模範解答の挙動を見る:
        SHOW_SOLUTION=1 uv run python lectures/20_open_vocabulary_detection/exercises.py

この5問はモデル DL 不要・純計算で、本モジュールの肝を1つずつ抜き出したもの:
  ex1: IoU（検出評価の核。box の重なり）                                    … 03
  ex2: 正規化 cxcywh → 絶対 xyxy 変換（target_sizes=(H,W) の座標変換の心臓部）… 01/02
  ex3: クラス込み貪欲マッチング → TP/FP/FN                                  … 03
  ex4: precision / recall / F1（TP/FP/FN から）                              … 03
  ex5: Grounding DINO 用キャプション整形（小文字＋ピリオド区切り）          … 02
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


def ex1_iou(box_a: list[float], box_b: list[float]) -> float:
    """演習1: 2つの box [x1,y1,x2,y2] の IoU（Intersection over Union）を返す。

    - 交差領域の幅 = max(0, min(ax2,bx2) - max(ax1,bx1))、高さも同様。交差面積 = 幅×高さ。
    - 和集合 = areaA + areaB - 交差。IoU = 交差 / 和集合（和集合が 0 なら 0.0）。
    - 重ならなければ 0.0、完全一致なら 1.0。
    """
    # TODO: 交差面積と和集合面積から IoU を計算して返す
    raise NotImplementedError


def ex2_cxcywh_norm_to_xyxy(box_norm: list[float], height: int, width: int) -> list[float]:
    """演習2: 正規化 cxcywh を絶対 xyxy へ変換する（DETR/OWL の生出力→可視化座標）。

    - box_norm = [cx, cy, w, h]（すべて 0〜1 の正規化座標。中心(cx,cy)と幅高さ(w,h)）。
    - x 方向は width を、y 方向は height を掛ける（ここを取り違えるのが target_sizes=(H,W) の典型バグ）。
    - 返り値は [x1, y1, x2, y2]（絶対画素）。x1=cx*W - (w*W)/2 など。
    """
    # TODO: cx,cy,w,h を絶対化し、中心±半分で xyxy に直して返す
    raise NotImplementedError


def ex3_greedy_match(
    pred_boxes: np.ndarray,
    pred_scores: np.ndarray,
    pred_labels: list[str],
    gt_boxes: np.ndarray,
    gt_labels: list[str],
    iou_thr: float = 0.5,
) -> tuple[int, int, int]:
    """演習3: クラス込みの貪欲マッチングで (TP, FP, FN) を返す。

    手順:
      1. 予測を score 降順に並べる（np.argsort(-pred_scores)）。
      2. 各予測について『同じラベルかつ未マッチで IoU>=iou_thr』の GT のうち IoU 最大に対応づける。
      3. 対応すれば TP、できなければ FP。最後まで対応されなかった GT は FN。
    1つの GT に複数予測が当たっても TP は最初の1つだけ（残りは FP）。ex1 の IoU を使ってよい。
    """
    # TODO: 貪欲マッチングで TP/FP/FN を数えて返す
    raise NotImplementedError


def ex4_prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    """演習4: TP/FP/FN から (precision, recall, f1) を返す。

    - precision = TP / (TP + FP)、recall = TP / (TP + FN)。分母 0 のときは 0.0。
    - f1 = 2*P*R / (P + R)。P+R が 0 のときは 0.0。
    """
    # TODO: precision, recall, f1 を計算して返す（ゼロ割は 0.0）
    raise NotImplementedError


def ex5_labels_to_caption(labels: list[str]) -> str:
    """演習5: 候補ラベルのリストを Grounding DINO 用キャプションへ整形する。

    - 各ラベルを小文字化し、前後の空白と末尾のピリオドを除く。
    - 空文字は飛ばす。各ラベルの末尾に "." を付け、半角スペースで連結する。
    - 例: ["a Red circle", " a BLUE square. "] -> "a red circle. a blue square."
    """
    # TODO: 小文字化＋トリム＋ピリオド区切りで連結して返す
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

    # ex1: 既知の IoU（重なり1/7・完全一致1・disjoint0）。
    def _c1():
        a = [0.0, 0.0, 10.0, 10.0]
        b = [5.0, 5.0, 15.0, 15.0]  # 交差25 / 和集合175 = 1/7
        same = ex1_iou(a, a)
        part = ex1_iou(a, b)
        disj = ex1_iou(a, [20.0, 20.0, 30.0, 30.0])
        ok = abs(same - 1.0) < 1e-6 and abs(part - 25 / 175) < 1e-6 and disj == 0.0
        return ok, f"same={same:.3f}, partial={part:.3f}, disjoint={disj:.3f}"

    check("ex1_iou", _c1)

    # ex2: 中央・幅高さ半分の box を 100x200 へ。x は W=200、y は H=100 でスケール。
    def _c2():
        got = ex2_cxcywh_norm_to_xyxy([0.5, 0.5, 0.5, 0.5], height=100, width=200)
        exp = [50.0, 25.0, 150.0, 75.0]
        ok = len(got) == 4 and all(abs(g - e) < 1e-6 for g, e in zip(got, exp))
        return ok, f"got={[round(v, 1) for v in got]} (期待 {exp})"

    check("ex2_cxcywh_norm_to_xyxy", _c2)

    # ex3: 2 GT（cat/dog）に対し 3 予測（cat 当たり・dog 当たり・cat 余り）→ TP2 FP1 FN0。
    def _c3():
        gt_boxes = np.array([[0, 0, 10, 10], [20, 20, 30, 30]], dtype=np.float32)
        gt_labels = ["cat", "dog"]
        pred_boxes = np.array([[0, 0, 9, 10], [21, 20, 30, 30], [0, 0, 10, 10]], dtype=np.float32)
        pred_scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
        pred_labels = ["cat", "dog", "cat"]
        tp, fp, fn = ex3_greedy_match(
            pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels, 0.5
        )
        return (tp, fp, fn) == (2, 1, 0), f"TP={tp} FP={fp} FN={fn} (期待 2,1,0)"

    check("ex3_greedy_match", _c3)

    # ex4: TP4 FP1 FN0 → P=0.8 R=1.0 F1=0.888...
    def _c4():
        p, r, f1 = ex4_prf(4, 1, 0)
        p2, r2, f2 = ex4_prf(0, 0, 3)  # 検出ゼロ → 全部 0.0
        ok = abs(p - 0.8) < 1e-6 and abs(r - 1.0) < 1e-6 and abs(f1 - 2 * 0.8 / 1.8) < 1e-6
        ok = ok and p2 == 0.0 and r2 == 0.0 and f2 == 0.0
        return ok, f"P={p:.3f} R={r:.3f} F1={f1:.3f}"

    check("ex4_prf", _c4)

    # ex5: 大文字・余分な空白・末尾ピリオド・空文字を正規化。
    def _c5():
        got = ex5_labels_to_caption(["a Red circle", " a BLUE square. ", ""])
        exp = "a red circle. a blue square."
        return got == exp, f"got={got!r}"

    check("ex5_labels_to_caption", _c5)

    print("=== 採点結果 ===")
    all_ok = True
    for name, ok, detail in results:
        mark = "PASS" if ok else "FAIL"
        all_ok = all_ok and ok
        print(f"  [{mark}] {name:24s} {detail}")
    print("\nALL PASS 🎉" if all_ok else "\nまだ未達の演習があります。TODO を埋めましょう。")


# =====================================================================
# 模範解答（SHOW_SOLUTION=1 のときに本体へ差し替えて実行）
# まずは自力で解いてから見ること。
# =====================================================================


def _sol_ex1(box_a: list[float], box_b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return float(inter / union) if union > 0 else 0.0


def _sol_ex2(box_norm: list[float], height: int, width: int) -> list[float]:
    cx, cy, w, h = box_norm
    cx_a, w_a = cx * width, w * width  # x 方向は width
    cy_a, h_a = cy * height, h * height  # y 方向は height
    return [cx_a - w_a / 2, cy_a - h_a / 2, cx_a + w_a / 2, cy_a + h_a / 2]


def _sol_ex3(pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels, iou_thr=0.5):
    order = np.argsort(-pred_scores)
    matched: set[int] = set()
    tp = fp = 0
    for i in order:
        cand = [
            g for g in range(len(gt_boxes)) if g not in matched and gt_labels[g] == pred_labels[i]
        ]
        if not cand:
            fp += 1
            continue
        ious = [_sol_ex1(list(pred_boxes[i]), list(gt_boxes[g])) for g in cand]
        j = int(np.argmax(ious))
        if ious[j] >= iou_thr:
            matched.add(cand[j])
            tp += 1
        else:
            fp += 1
    fn = len(gt_boxes) - len(matched)
    return tp, fp, fn


def _sol_ex4(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1


def _sol_ex5(labels: list[str]) -> str:
    parts = [lab.strip().rstrip(".").strip().lower() for lab in labels]
    parts = [p for p in parts if p]
    return " ".join(f"{p}." for p in parts)


def _install_solutions() -> None:
    g = globals()
    g["ex1_iou"] = _sol_ex1
    g["ex2_cxcywh_norm_to_xyxy"] = _sol_ex2
    g["ex3_greedy_match"] = _sol_ex3
    g["ex4_prf"] = _sol_ex4
    g["ex5_labels_to_caption"] = _sol_ex5


if __name__ == "__main__":
    if os.environ.get("SHOW_SOLUTION") == "1":
        print("(模範解答モードで実行します)\n")
        _install_solutions()
    _grade()
