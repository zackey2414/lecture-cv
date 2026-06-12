"""第20回 演習問題（オープン語彙物体検出の核を純計算で）。

使い方:
  1. 各 exN_*() の TODO を実装する（最初は NotImplementedError だが、採点ランナーが
     拾うのでプロセスは落ちず、FAIL と表示されるだけ。必ず exit 0）。
  2. 自己採点:  uv run python lectures/20_open_vocabulary_detection/exercises.py
     全問 pass すれば "ALL PASS" と表示される。
  3. 詰まったら模範解答（全 PASS）を見る:
        uv run python lectures/20_open_vocabulary_detection/exercises_solutions.py
     もしくは同じ採点ロジックで模範解答を採点する:
        SHOW_SOLUTION=1 uv run python lectures/20_open_vocabulary_detection/exercises.py

この8問はモデル DL 不要・純計算で、本モジュールの肝を1つずつ抜き出したもの（易→難）:
  ex1: IoU（検出評価の核。box の重なり）                                    … 03
  ex2: 正規化 cxcywh → 絶対 xyxy 変換（target_sizes=(H,W) の座標変換の心臓部）… 01/02
  ex3: クラス込み貪欲マッチング → TP/FP/FN                                  … 03
  ex4: precision / recall / F1（TP/FP/FN から）                              … 03
  ex5: Grounding DINO 用キャプション整形（小文字＋ピリオド区切り）          … 02
  ex6: IoU 行列（N 予測 × M GT のペアワイズ IoU。マッチングの土台）         … 03/mini
  ex7: NMS（非最大抑制。過検出した重複ボックスを1つに畳む）                … 02 の過検出ケア
  ex8: AP（PR 曲線の全点補間。第19回 mAP 自作の心臓部）                     … 03/mini
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
    """演習1（易）: 2つの box [x1,y1,x2,y2] の IoU（Intersection over Union）を返す。

    - 交差領域の幅 = max(0, min(ax2,bx2) - max(ax1,bx1))、高さも同様。交差面積 = 幅×高さ。
    - 和集合 = areaA + areaB - 交差。IoU = 交差 / 和集合（和集合が 0 なら 0.0）。
    - 重ならなければ 0.0、完全一致なら 1.0。
    """
    # TODO: 交差面積と和集合面積から IoU を計算して返す
    raise NotImplementedError


def ex2_cxcywh_norm_to_xyxy(box_norm: list[float], height: int, width: int) -> list[float]:
    """演習2（易）: 正規化 cxcywh を絶対 xyxy へ変換する（DETR/OWL の生出力→可視化座標）。

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
    """演習3（中）: クラス込みの貪欲マッチングで (TP, FP, FN) を返す。

    手順:
      1. 予測を score 降順に並べる（np.argsort(-pred_scores)）。
      2. 各予測について『同じラベルかつ未マッチで IoU>=iou_thr』の GT のうち IoU 最大に対応づける。
      3. 対応すれば TP、できなければ FP。最後まで対応されなかった GT は FN。
    1つの GT に複数予測が当たっても TP は最初の1つだけ（残りは FP）。ex1 の IoU を使ってよい。
    """
    # TODO: 貪欲マッチングで TP/FP/FN を数えて返す
    raise NotImplementedError


def ex4_prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    """演習4（易）: TP/FP/FN から (precision, recall, f1) を返す。

    - precision = TP / (TP + FP)、recall = TP / (TP + FN)。分母 0 のときは 0.0。
    - f1 = 2*P*R / (P + R)。P+R が 0 のときは 0.0。
    """
    # TODO: precision, recall, f1 を計算して返す（ゼロ割は 0.0）
    raise NotImplementedError


def ex5_labels_to_caption(labels: list[str]) -> str:
    """演習5（易）: 候補ラベルのリストを Grounding DINO 用キャプションへ整形する。

    - 各ラベルを小文字化し、前後の空白と末尾のピリオドを除く。
    - 空文字は飛ばす。各ラベルの末尾に "." を付け、半角スペースで連結する。
    - 例: ["a Red circle", " a BLUE square. "] -> "a red circle. a blue square."
    """
    # TODO: 小文字化＋トリム＋ピリオド区切りで連結して返す
    raise NotImplementedError


def ex6_iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """演習6（中）: N 個の box 群と M 個の box 群のペアワイズ IoU 行列 (N, M) を返す。

    - boxes_a: (N,4) xyxy、boxes_b: (M,4) xyxy。返り値 M[i,j] = IoU(boxes_a[i], boxes_b[j])。
    - 貪欲マッチング/評価の土台。素朴な二重ループでも、numpy のブロードキャストでもよい。
    - どちらかが空なら shape (N,0) または (0,M) の配列を返す（zero 割は 0.0 に）。
    ヒント: torchvision.ops.box_iou と同じ意味。ex1 を全ペアに適用すれば作れる。
    """
    # TODO: 全ペアの IoU を (N, M) 行列にして返す
    raise NotImplementedError


def ex7_nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float = 0.5) -> list[int]:
    """演習7（中）: 単一クラスの NMS（非最大抑制）で『残す予測のインデックス』を返す。

    過検出（同じ物体に複数ボックス）を1つに畳む基本アルゴリズム。手順:
      1. スコア降順に並べる。
      2. 先頭（最高スコア）を採用し、その box と IoU>=iou_thr の残りを捨てる。
      3. 残りについて 2 を繰り返す。
    - 返り値は『採用したインデックスのリスト』（採用順＝スコア降順）。ex1 の IoU を使ってよい。
    - 何も重ならなければ全件が残る。
    """
    # TODO: スコア降順に貪欲に採用し、IoU>=iou_thr の重複を抑制したインデックス列を返す
    raise NotImplementedError


def ex8_average_precision(
    scores: np.ndarray, is_tp: np.ndarray, n_gt: int
) -> float:
    """演習8（難）: マッチ済み検出から AP（Average Precision・全点補間）を返す。

    第19回 mAP 自作の心臓部。各検出は『スコア』と『TP かどうか(1/0)』を持つ（マッチングは済み）。
    手順:
      1. スコア降順に並べる。
      2. tp_cum = cumsum(is_tp), fp_cum = cumsum(1-is_tp)。
      3. recall = tp_cum / n_gt、precision = tp_cum / (tp_cum + fp_cum)。
      4. 全点補間: precision を右側からの累積最大で単調非増加に均し、
         AP = Σ (recall[i] - recall[i-1]) * precision_envelope[i]（recall[-1]=0 を起点に）。
    - n_gt が 0、または TP が一つも無ければ AP=0.0。PASCAL の 11 点ではなく『全点（連続）』補間。
    """
    # TODO: PR 曲線を作り、全点補間で面積（AP）を求めて返す
    raise NotImplementedError


# =====================================================================
# 自己採点ランナー（未実装でも例外で落とさず PASS/FAIL を表示。必ず exit 0）
#   grade(funcs) は「関数名 -> 実装」の辞書を受け取って採点する。
#   exercises.py（TODO 版）も exercises_solutions.py（模範解答）も、この同じ grade を呼ぶ。
# =====================================================================

EX_NAMES = [
    "ex1_iou",
    "ex2_cxcywh_norm_to_xyxy",
    "ex3_greedy_match",
    "ex4_prf",
    "ex5_labels_to_caption",
    "ex6_iou_matrix",
    "ex7_nms",
    "ex8_average_precision",
]


def grade(funcs: dict) -> bool:
    """funcs（名前->実装）を採点し、PASS/FAIL を表示して all_ok を返す。

    grading は exercises 側にのみ存在し、模範解答ファイルもこれを再利用する（重複なし）。
    """
    results: list[tuple[str, bool, str]] = []

    def check(name: str, fn) -> None:
        try:
            ok, detail = fn()
            results.append((name, bool(ok), detail))
        except NotImplementedError:
            results.append((name, False, "未実装（TODOを埋めてください）"))
        except Exception as e:  # noqa: BLE001
            results.append((name, False, f"例外: {type(e).__name__}: {e}"))

    # ex1: 既知の IoU（重なり25/175・完全一致1・disjoint0）。
    def _c1():
        a = [0.0, 0.0, 10.0, 10.0]
        b = [5.0, 5.0, 15.0, 15.0]  # 交差25 / 和集合175 = 1/7
        iou = funcs["ex1_iou"]
        same = iou(a, a)
        part = iou(a, b)
        disj = iou(a, [20.0, 20.0, 30.0, 30.0])
        ok = abs(same - 1.0) < 1e-6 and abs(part - 25 / 175) < 1e-6 and disj == 0.0
        return ok, f"same={same:.3f}, partial={part:.3f}, disjoint={disj:.3f}"

    check("ex1_iou", _c1)

    # ex2: 中央・幅高さ半分の box を 100x200 へ。x は W=200、y は H=100 でスケール。
    def _c2():
        got = funcs["ex2_cxcywh_norm_to_xyxy"]([0.5, 0.5, 0.5, 0.5], height=100, width=200)
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
        tp, fp, fn = funcs["ex3_greedy_match"](
            pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels, 0.5
        )
        return (tp, fp, fn) == (2, 1, 0), f"TP={tp} FP={fp} FN={fn} (期待 2,1,0)"

    check("ex3_greedy_match", _c3)

    # ex4: TP4 FP1 FN0 → P=0.8 R=1.0 F1=0.888...、検出ゼロは全部 0.0。
    def _c4():
        prf = funcs["ex4_prf"]
        p, r, f1 = prf(4, 1, 0)
        p2, r2, f2 = prf(0, 0, 3)
        ok = abs(p - 0.8) < 1e-6 and abs(r - 1.0) < 1e-6 and abs(f1 - 2 * 0.8 / 1.8) < 1e-6
        ok = ok and p2 == 0.0 and r2 == 0.0 and f2 == 0.0
        return ok, f"P={p:.3f} R={r:.3f} F1={f1:.3f}"

    check("ex4_prf", _c4)

    # ex5: 大文字・余分な空白・末尾ピリオド・空文字を正規化。
    def _c5():
        got = funcs["ex5_labels_to_caption"](["a Red circle", " a BLUE square. ", ""])
        exp = "a red circle. a blue square."
        return got == exp, f"got={got!r}"

    check("ex5_labels_to_caption", _c5)

    # ex6: 2x2 の IoU 行列。a0=b0 で 1.0、a0 と b1=[5,5,15,15] で 1/7、a1 は両方と disjoint。
    def _c6():
        a = np.array([[0, 0, 10, 10], [20, 20, 30, 30]], dtype=np.float32)
        b = np.array([[0, 0, 10, 10], [5, 5, 15, 15]], dtype=np.float32)
        m = np.asarray(funcs["ex6_iou_matrix"](a, b), dtype=float)
        exp = np.array([[1.0, 25 / 175], [0.0, 0.0]])
        ok = m.shape == (2, 2) and np.allclose(m, exp, atol=1e-6)
        return ok, f"shape={m.shape}, [0,0]={m[0, 0]:.3f}, [0,1]={m[0, 1]:.3f}"

    check("ex6_iou_matrix", _c6)

    # ex7: box0,box1 は IoU≈0.68 で重なる。thr=0.5 で box1 抑制 →[0,2]、thr=0.9 で全残り[0,1,2]。
    def _c7():
        boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [20, 20, 30, 30]], dtype=np.float32)
        scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
        keep_strict = list(funcs["ex7_nms"](boxes, scores, 0.5))
        keep_loose = list(funcs["ex7_nms"](boxes, scores, 0.9))
        ok = keep_strict == [0, 2] and keep_loose == [0, 1, 2]
        return ok, f"thr0.5->{keep_strict} (期待[0,2]) / thr0.9->{keep_loose} (期待[0,1,2])"

    check("ex7_nms", _c7)

    # ex8: AP の全点補間。3 ケースで検算。
    def _c8():
        ap = funcs["ex8_average_precision"]
        # (A) is_tp=[1,0,1], n_gt=2 → PR は (0.5,1.0)->(0.5,0.5)->(1.0,2/3)、全点補間で AP=5/6。
        a = ap(np.array([0.9, 0.8, 0.7]), np.array([1, 0, 1]), 2)
        # (B) 全部 TP・全 GT 回収 → AP=1.0。
        b = ap(np.array([0.9, 0.8]), np.array([1, 1]), 2)
        # (C) TP ゼロ → AP=0.0。
        c = ap(np.array([0.9]), np.array([0]), 1)
        ok = abs(a - 5 / 6) < 1e-3 and abs(b - 1.0) < 1e-6 and abs(c - 0.0) < 1e-9
        return ok, f"AP(A)={a:.4f}(期待0.8333) AP(B)={b:.3f} AP(C)={c:.3f}"

    check("ex8_average_precision", _c8)

    print("=== 採点結果 ===")
    all_ok = True
    for name, ok, detail in results:
        mark = "PASS" if ok else "FAIL"
        all_ok = all_ok and ok
        print(f"  [{mark}] {name:24s} {detail}")
    print("\nALL PASS 🎉" if all_ok else "\nまだ未達の演習があります。TODO を埋めましょう。")
    return all_ok


def current_functions() -> dict:
    """このファイル内の exN 実装（最初は TODO）を辞書で集める。"""
    g = globals()
    return {name: g[name] for name in EX_NAMES}


if __name__ == "__main__":
    if os.environ.get("SHOW_SOLUTION") == "1":
        # 模範解答を別ファイルから読み込み、同じ grade で採点する（採点ロジックを共有）。
        print("(模範解答モードで実行します — exercises_solutions.py の実装を採点)\n")
        import exercises_solutions  # noqa: E402

        grade(exercises_solutions.solution_functions())
    else:
        grade(current_functions())
