"""第21回 演習問題（セマンティックセグメンテーションの評価指標）。

使い方:
  1. 各 exN_*() の TODO を自分で実装する（最初は NotImplementedError だが、
     採点ランナーが拾うのでプロセスは落ちず、FAIL と表示されるだけ＝必ず exit 0）。
  2. 自己採点:  uv run python lectures/21_segmentation_intro/exercises.py
     全問 pass すれば "ALL PASS" と表示される。
  3. 詰まったら模範解答の挙動を見る（採点ロジックは共有・実装だけ差し替え）:
        SHOW_SOLUTION=1 uv run python lectures/21_segmentation_intro/exercises.py
     これは exercises_solutions.py の実装を同じ grade() で採点する。

この 10 問は本モジュールの核「画素混同行列 → 各指標」を1段ずつ分解し、
さらに IoU・ignore・集計単位（global/samplewise）まで段階的に踏み込む
（モデルDL不要・純粋な numpy 計算。易→難）:
  ex1 : 画素混同行列 cm[g,p] を作る                          … 03
  ex2 : per-class IoU（未出現クラスは NaN）                  … 03
  ex3 : mIoU（per-class IoU の nanmean）                     … 03
  ex4 : per-class Dice（=F1, 未出現クラスは NaN）            … 03
  ex5 : pixel accuracy（ΣTP / 全画素）                       … 03
  ex6 : FWIoU（GT 画素割合で重み付けした IoU）               … 06/03
  ex7 : ignore_index 付き混同行列（評価対象外の画素を除外）  … 08
  ex8 : 2値マスク同士の IoU（マスクIoU の基礎・両方空なら1） … 06
  ex9 : IoU→Dice の換算（Dice = 2·IoU/(1+IoU)）              … 06
  ex10: global と samplewise の mIoU を両方出す（集計のクセ）… 09
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
def ex1_confusion_matrix(gt: np.ndarray, pred: np.ndarray, num_classes: int) -> np.ndarray:
    """演習1: GT と予測のクラスマップから KxK の画素混同行列 cm[g, p] を作る。

    - gt, pred: 同じ形 (H, W) の int 配列。値は 0..num_classes-1。
    - cm[g, p] = 「正解が g で予測が p の画素数」。形は (num_classes, num_classes)。
    - ヒント: idx = gt*num_classes + pred を平坦化し np.bincount(idx, minlength=K*K) → reshape。
    """
    # TODO: 二重ループでも良いが、bincount を使うと速い
    raise NotImplementedError


def ex2_per_class_iou(cm: np.ndarray) -> np.ndarray:
    """演習2: 混同行列から per-class IoU を返す（未出現クラスは np.nan）。

    - TP_c = cm[c, c], FP_c = 列和 - TP, FN_c = 行和 - TP。
    - IoU_c = TP / (TP + FP + FN)。
    - GT にも予測にも出ないクラス（TP+FP+FN == 0）は 0/0 なので np.nan にする。
    - 返り値は形 (num_classes,) の float 配列。
    """
    # TODO: tp/fp/fn を出し、分母0のクラスを nan にして IoU を返す
    raise NotImplementedError


def ex3_mean_iou(cm: np.ndarray) -> float:
    """演習3: mIoU（per-class IoU のクラス平均）を返す。

    - 未出現クラスの NaN は平均から除外する（np.nanmean を使う）。
    - ex2 を再利用してよい。
    """
    # TODO: np.nanmean(ex2_per_class_iou(cm)) を float で返す
    raise NotImplementedError


def ex4_per_class_dice(cm: np.ndarray) -> np.ndarray:
    """演習4: per-class Dice係数（= F1）を返す（未出現クラスは np.nan）。

    - Dice_c = 2*TP / (2*TP + FP + FN)。
    - 未出現クラス（分母0）は np.nan。
    - 返り値は形 (num_classes,) の float 配列。
    """
    # TODO: ex2 と同じ要領で 2TP/(2TP+FP+FN) を計算（分母0は nan）
    raise NotImplementedError


def ex5_pixel_accuracy(cm: np.ndarray) -> float:
    """演習5: pixel accuracy（正しく分類された画素の割合）を返す。

    - pixel acc = 対角の総和(ΣTP) / 全画素(cm.sum())。
    - cm.sum() == 0 のときは 0.0 を返す（ゼロ割回避）。
    """
    # TODO: np.diag(cm).sum() / cm.sum() を float で返す
    raise NotImplementedError


def ex6_fwiou(cm: np.ndarray) -> float:
    """演習6: FWIoU（Frequency-Weighted IoU）を返す。

    mIoU はクラスを対等に平均するが、FWIoU は「GT に占める画素割合」で重み付けする。
    面積の大きいクラスを重視したいとき（=画素誤差をそのまま気にするとき）に使う。
      - freq_c = (GT のクラス c の画素数) / 全画素 = 行和_c / cm.sum()
      - FWIoU  = Σ_c freq_c * IoU_c   （未出現クラスは freq=0 なので寄与しない）
    - 返り値は float。cm.sum()==0 のときは 0.0。
    - ヒント: ex2 の per-class IoU を再利用し、NaN は np.nansum で握りつぶす。
    """
    # TODO: 行和/総和の重み × per-class IoU の総和を返す（NaN は nansum で除外）
    raise NotImplementedError


def ex7_confusion_matrix_ignore(
    gt: np.ndarray, pred: np.ndarray, num_classes: int, ignore_index: int
) -> np.ndarray:
    """演習7: ignore_index 付きの画素混同行列を作る。

    データセットには「アノテーションが曖昧な境界」「評価対象外」を表す特別ラベル
    （VOC なら 255）があり、その画素は評価から完全に除外する。
      - GT が ignore_index の画素は、予測が何であれ cm に数えない。
      - 返り値は (num_classes, num_classes)。値域外の ignore ラベルが混じっても壊れないこと。
    - ヒント: keep = (gt != ignore_index) でマスクしてから ex1 と同じ符号化を行う。
    """
    # TODO: gt==ignore_index の画素を捨ててから bincount で cm を作る
    raise NotImplementedError


def ex8_binary_mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """演習8: 2値マスク同士の IoU を返す（マスク IoU の基礎）。

    bbox の IoU と同じ「交差 / 和集合」だが、ここでは画素マスク（bool 配列）で計算する。
    第22回のインスタンス mask AP・SAM の品質評価につながる土台。
      - IoU = |A∩B| / |A∪B|
      - 両方とも空（union==0）のときは「完全一致」とみなして 1.0 を返す。
    - mask_a, mask_b: 同じ形の bool（または 0/1）配列。返り値は float。
    """
    # TODO: logical_and / logical_or の画素数から IoU。union==0 は 1.0
    raise NotImplementedError


def ex9_dice_from_iou(iou):
    """演習9: IoU から Dice を換算する（Dice = 2·IoU / (1 + IoU)）。

    IoU と Dice は1対1で対応する（片方が上がれば必ずもう片方も上がる）。
    その関係式 Dice = 2·IoU/(1+IoU) を実装する。
      - iou はスカラでも numpy 配列でも動くようにする（同じ式で両対応できる）。
      - NaN が入れば NaN がそのまま返る（特別扱い不要）。
    """
    # TODO: 2*iou/(1+iou) を返す（スカラ/配列どちらも同じ式でよい）
    raise NotImplementedError


def ex10_global_vs_samplewise_miou(pairs: list, num_classes: int) -> tuple[float, float]:
    """演習10: 複数画像の mIoU を「global」と「samplewise」の両方で計算する。

    pairs = [(gt0, pred0), (gt1, pred1), ...]（各要素は (H,W) のクラスマップ）。
      - global     : 全画像の画素を1つの混同行列に積んでから mIoU を1回計算（ベンチマーク慣行）。
      - samplewise : 画像ごとに mIoU を出してから、その平均（torchmetrics の既定）。
    両者は一般に一致しない。本演習で「同じ予測でも集計単位で値が変わる」を体感する。
      - 返り値は (global_miou, samplewise_miou) の float タプル。
    - ヒント: ex1（混同行列）と ex3（mIoU）を再利用すると素直に書ける。
    """
    # TODO: 1) cm を積んで global mIoU、2) 画像ごとの mIoU を平均して samplewise
    raise NotImplementedError


# =====================================================================
# 採点対象の関数名（exercises_solutions.py からも参照する）
# =====================================================================
EX_NAMES = [
    "ex1_confusion_matrix",
    "ex2_per_class_iou",
    "ex3_mean_iou",
    "ex4_per_class_dice",
    "ex5_pixel_accuracy",
    "ex6_fwiou",
    "ex7_confusion_matrix_ignore",
    "ex8_binary_mask_iou",
    "ex9_dice_from_iou",
    "ex10_global_vs_samplewise_miou",
]


# =====================================================================
# 採点用の固定参照（解答実装を呼ばずに「真値」を直接与えて検証する）
# =====================================================================
#   gt = [[0,0,1],[2,2,1]], pred = [[0,1,1],[2,2,1]], num_classes=4（class3 は未出現）
#   cm = [[1,1,0,0],[0,2,0,0],[0,0,2,0],[0,0,0,0]]
#   IoU = [0.5, 2/3, 1.0, NaN]   Dice = [2/3, 0.8, 1.0, NaN]
#   mIoU = nanmean = 0.7222...   pixel acc = 5/6 = 0.8333...
REF_GT = np.array([[0, 0, 1], [2, 2, 1]], dtype=np.int64)
REF_PRED = np.array([[0, 1, 1], [2, 2, 1]], dtype=np.int64)
REF_CM = np.array([[1, 1, 0, 0], [0, 2, 0, 0], [0, 0, 2, 0], [0, 0, 0, 0]], dtype=np.int64)
REF_IOU = np.array([0.5, 2 / 3, 1.0, np.nan])
REF_DICE = np.array([2 / 3, 0.8, 1.0, np.nan])

# ex6 用（FWIoU が mIoU と一致しないよう、クラス頻度を非一様にした 3 クラス cm）
#   iou = [8/11, 0.5, 0.8], freq = [0.5, 0.25, 0.25]
#   FWIoU = 0.5*(8/11) + 0.25*0.5 + 0.25*0.8
FW_CM = np.array([[8, 2, 0], [1, 4, 0], [0, 1, 4]], dtype=np.int64)
REF_FWIOU = 0.5 * (8 / 11) + 0.25 * 0.5 + 0.25 * 0.8

# ex7 用（gt の 255 は評価対象外。除外後の cm が真値）
IGN_GT = np.array([[0, 0, 255], [1, 1, 2]], dtype=np.int64)
IGN_PRED = np.array([[0, 1, 0], [1, 1, 2]], dtype=np.int64)
REF_CM_IGNORE = np.array([[1, 1, 0], [0, 2, 0], [0, 0, 1]], dtype=np.int64)

# ex10 用（2 枚の小画像。global と samplewise が食い違う）
GT_A = np.array([[0, 0], [1, 1]], dtype=np.int64)
PRED_A = np.array([[0, 1], [1, 1]], dtype=np.int64)
GT_B = np.array([[1, 1], [1, 1]], dtype=np.int64)
PRED_B = np.array([[1, 1], [1, 0]], dtype=np.int64)
REF_GLOBAL_MIOU = ((1 / 3) + (5 / 7)) / 2          # 累積 cm = [[1,1],[1,5]]
REF_SAMPLE_MIOU = ((0.5 + 2 / 3) / 2 + (0.0 + 0.75) / 2) / 2  # 画像A=0.5833, 画像B=0.375


# =====================================================================
# 自己採点ランナー（未実装でも例外で落とさず PASS/FAIL を表示。必ず exit 0）
#   grade(funcs) は「関数名 -> 実装」の辞書を受け取って採点する。
#   exercises.py（TODO 版）も exercises_solutions.py（模範解答）も、この同じ grade を呼ぶ。
# =====================================================================
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
            results.append((name, False, "未実装（TODO を埋めてください）"))
        except Exception as e:  # noqa: BLE001
            results.append((name, False, f"例外: {type(e).__name__}: {e}"))

    def _c1():
        got = np.asarray(funcs["ex1_confusion_matrix"](REF_GT, REF_PRED, 4))
        return np.array_equal(got, REF_CM), f"cm=\n{got}"
    check("ex1_confusion_matrix", _c1)

    def _c2():
        got = funcs["ex2_per_class_iou"](REF_CM)
        return np.allclose(got, REF_IOU, atol=1e-6, equal_nan=True), f"iou={np.round(got, 3)}"
    check("ex2_per_class_iou", _c2)

    def _c3():
        got = funcs["ex3_mean_iou"](REF_CM)
        return abs(got - float(np.nanmean(REF_IOU))) < 1e-6, f"mIoU={got:.4f}"
    check("ex3_mean_iou", _c3)

    def _c4():
        got = funcs["ex4_per_class_dice"](REF_CM)
        return np.allclose(got, REF_DICE, atol=1e-6, equal_nan=True), f"dice={np.round(got, 3)}"
    check("ex4_per_class_dice", _c4)

    def _c5():
        got = funcs["ex5_pixel_accuracy"](REF_CM)
        zero = funcs["ex5_pixel_accuracy"](np.zeros((4, 4), dtype=np.int64))  # ゼロ割回避
        return abs(got - 5 / 6) < 1e-6 and zero == 0.0, f"pixel_acc={got:.4f}"
    check("ex5_pixel_accuracy", _c5)

    def _c6():
        got = funcs["ex6_fwiou"](FW_CM)
        zero = funcs["ex6_fwiou"](np.zeros((3, 3), dtype=np.int64))
        return abs(got - REF_FWIOU) < 1e-6 and zero == 0.0, f"FWIoU={got:.4f} (期待 {REF_FWIOU:.4f})"
    check("ex6_fwiou", _c6)

    def _c7():
        got = np.asarray(funcs["ex7_confusion_matrix_ignore"](IGN_GT, IGN_PRED, 3, 255))
        return np.array_equal(got, REF_CM_IGNORE), f"cm(ignore=255)=\n{got}"
    check("ex7_confusion_matrix_ignore", _c7)

    def _c8():
        a = np.array([[1, 1, 0], [1, 1, 0], [0, 0, 0]], dtype=bool)  # 左上 2x2
        b = np.array([[0, 1, 1], [0, 1, 1], [0, 0, 0]], dtype=bool)  # 右上 2x2 → 交差2/和6
        got = funcs["ex8_binary_mask_iou"](a, b)
        empty = np.zeros((3, 3), dtype=bool)
        both_empty = funcs["ex8_binary_mask_iou"](empty, empty)  # union==0 → 1.0
        return (abs(got - 1 / 3) < 1e-6 and abs(both_empty - 1.0) < 1e-6,
                f"IoU={got:.4f} (期待 0.3333), both_empty={both_empty}")
    check("ex8_binary_mask_iou", _c8)

    def _c9():
        arr = funcs["ex9_dice_from_iou"](np.array([0.5, 1.0, 0.0, np.nan]))
        scal = funcs["ex9_dice_from_iou"](0.5)
        ok = (np.allclose(np.asarray(arr), [2 / 3, 1.0, 0.0, np.nan], atol=1e-6, equal_nan=True)
              and abs(float(scal) - 2 / 3) < 1e-6)
        return ok, f"dice(array)={np.round(np.asarray(arr), 3)}, dice(0.5)={float(scal):.4f}"
    check("ex9_dice_from_iou", _c9)

    def _c10():
        pairs = [(GT_A, PRED_A), (GT_B, PRED_B)]
        g, s = funcs["ex10_global_vs_samplewise_miou"](pairs, 2)
        ok = abs(g - REF_GLOBAL_MIOU) < 1e-6 and abs(s - REF_SAMPLE_MIOU) < 1e-6
        return ok, f"global={g:.4f} (期待 {REF_GLOBAL_MIOU:.4f}), samplewise={s:.4f} (期待 {REF_SAMPLE_MIOU:.4f})"
    check("ex10_global_vs_samplewise_miou", _c10)

    print("=== 採点結果 ===")
    all_ok = True
    for name, ok, detail in results:
        mark = "PASS" if ok else "FAIL"
        all_ok = all_ok and ok
        print(f"  [{mark}] {name:30s} {detail}")
    print("\nALL PASS 🎉" if all_ok else "\nまだ未達の演習があります。TODO を埋めましょう。")
    return all_ok


def current_functions() -> dict:
    """このファイル内の ex*（学習者の実装）を「名前->関数」の辞書で返す。"""
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
