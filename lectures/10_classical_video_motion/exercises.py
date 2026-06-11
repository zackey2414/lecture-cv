"""第10回 演習問題（古典的な動画処理 — 動き解析）。

使い方:
  1. 各 exN_*() 関数の中の TODO を自分で実装する（最初は NotImplementedError が出る）。
  2. 実装できたら自己採点を実行:
         uv run python lectures/10_classical_video_motion/exercises.py
     全問が PASS すれば "ALL PASS" と表示される（未実装でも落ちず FAIL 表示で正常終了）。
  3. どうしても分からない時は、模範解答を実行して挙動を確認する:
         SHOW_SOLUTION=1 uv run python lectures/10_classical_video_motion/exercises.py
     （まずは自力で！）

ヒント: サンプルや部品は同じフォルダの cv_helpers.py にある。
  - make_motion_frames()   : 静止背景の上を物体が動く連番フレーム
  - make_translating_pair(): 既知シフトの画像ペア（EPE の答え合わせ用）
"""

from __future__ import annotations

import os
import pathlib
import sys

import cv2
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from cv_helpers import (  # noqa: E402
    make_motion_frames,
    make_translating_pair,
    output_dir,
)

# LK で使う共通パラメータ（演習3用）。
LK_PARAMS = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
)


# =====================================================================
# 演習（ここを実装する）
# =====================================================================

def ex1_frame_diff_mask(prev: np.ndarray, nxt: np.ndarray, thresh: int = 25
                        ) -> np.ndarray:
    """演習1: フレーム差分による前景マスクを作る。

    手順:
      1. cv2.absdiff で2枚の差分を取る（飽和演算なのでラップアラウンドしない）。
      2. cv2.cvtColor でグレースケールにする。
      3. cv2.threshold で thresh を境に2値化し、(H, W) uint8 のマスクを返す。
    返り値: 動いた所が 255・それ以外 0 の uint8 マスク（shape は (H, W)）。
    """
    # TODO: diff = cv2.absdiff(prev, nxt)
    #       gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    #       _, mask = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
    #       return mask
    raise NotImplementedError


def ex2_count_blobs(mask: np.ndarray, min_area: int = 150) -> int:
    """演習2: 二値マスクから、面積が min_area 以上の塊（動体）の数を数える。

    cv2.findContours は OpenCV 4 系では (contours, hierarchy) の2つ返し（3つではない）。
    各輪郭の cv2.contourArea が min_area 以上のものだけ数えて int で返す。
    """
    # TODO: contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #       return sum(1 for c in contours if cv2.contourArea(c) >= min_area)
    raise NotImplementedError


def ex3_lk_good_new(prev_gray: np.ndarray, next_gray: np.ndarray,
                    p0: np.ndarray) -> np.ndarray:
    """演習3: LK で p0 を追跡し、『追跡成功した点(status==1)の新位置』だけ返す。

    cv2.calcOpticalFlowPyrLK(prev_gray, next_gray, p0, None, **LK_PARAMS) を呼び、
    返り値 (p1, status, err) のうち status.ravel()==1 の p1 だけを取り出して返す。
    返り値の shape は (M, 2)（追跡できた点の数 M、各行が x, y）。
    """
    # TODO: p1, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, next_gray, p0, None, **LK_PARAMS)
    #       return p1[status.ravel() == 1].reshape(-1, 2)
    raise NotImplementedError


def ex4_flow_magnitude(flow: np.ndarray) -> np.ndarray:
    """演習4: 密フロー (H, W, 2) から、各画素の動きの大きさ (H, W) を返す。

    大きさ = sqrt(u^2 + v^2)。np.linalg.norm(flow, axis=2) でも cv2.cartToPolar でもよい。
    返り値の shape は (H, W)。
    """
    # TODO: return np.linalg.norm(flow, axis=2)
    raise NotImplementedError


def ex5_mean_epe(flow: np.ndarray, gt: np.ndarray) -> float:
    """演習5: 推定フロー flow と真のフロー gt の平均終点誤差(EPE)を返す。

    各画素について (flow - gt) のユークリッド距離を取り、その平均を float で返す。
    """
    # TODO: epe = np.linalg.norm(flow - gt, axis=2)
    #       return float(epe.mean())
    raise NotImplementedError


# =====================================================================
# 自己採点ランナー
# =====================================================================

def _grade() -> None:
    out = output_dir()  # noqa: F841  （存在確認だけ。保存はしない）
    frames = make_motion_frames()

    # ex1/ex2 用: 連続フレームと、その参照前景マスク。
    prev_bgr, next_bgr = frames[18], frames[19]
    ref_mask = cv2.threshold(
        cv2.cvtColor(cv2.absdiff(prev_bgr, next_bgr), cv2.COLOR_BGR2GRAY),
        25, 255, cv2.THRESH_BINARY,
    )[1]

    # ex2 用: 円2つ・面積既知の合成マスク（決定的に数えられる）。
    blob_mask = np.zeros((120, 200), np.uint8)
    cv2.circle(blob_mask, (50, 60), 20, 255, -1)   # 面積 ~1256（>150）
    cv2.circle(blob_mask, (150, 60), 18, 255, -1)  # 面積 ~1017（>150）
    cv2.circle(blob_mask, (100, 100), 3, 255, -1)  # 面積 ~28（ノイズ。数えない）
    ref_blob_count = sum(
        1 for c in cv2.findContours(blob_mask, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)[0]
        if cv2.contourArea(c) >= 150
    )

    # ex3 用: グレースケール対と特徴点。
    g0 = cv2.cvtColor(frames[10], cv2.COLOR_BGR2GRAY)
    g1 = cv2.cvtColor(frames[11], cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(g0, maxCorners=200, qualityLevel=0.05,
                                 minDistance=7, blockSize=7)
    p1_ref, st_ref, _ = cv2.calcOpticalFlowPyrLK(g0, g1, p0, None, **LK_PARAMS)
    ref_good_count = int((st_ref.ravel() == 1).sum())

    # ex4/ex5 用: 既知シフトの密フローと真値。
    a, b, shift = make_translating_pair()
    flow = cv2.calcOpticalFlowFarneback(
        cv2.cvtColor(a, cv2.COLOR_BGR2GRAY),
        cv2.cvtColor(b, cv2.COLOR_BGR2GRAY),
        None, 0.5, 3, 15, 3, 5, 1.2, 0,
    )
    gt = np.zeros_like(flow)
    gt[..., 0], gt[..., 1] = shift
    ref_mag = np.linalg.norm(flow, axis=2)
    ref_epe = float(np.linalg.norm(flow - gt, axis=2).mean())

    results: list[tuple[str, bool, str]] = []

    def check(name: str, fn) -> None:
        try:
            ok, detail = fn()
            results.append((name, bool(ok), detail))
        except NotImplementedError:
            results.append((name, False, "未実装（TODOを埋めてください）"))
        except Exception as e:  # noqa: BLE001
            results.append((name, False, f"例外: {type(e).__name__}: {e}"))

    def _c1():
        m = ex1_frame_diff_mask(prev_bgr, next_bgr, 25)
        ok = (isinstance(m, np.ndarray) and m.shape == ref_mask.shape
              and m.dtype == np.uint8 and np.array_equal(m, ref_mask))
        shp = None if not isinstance(m, np.ndarray) else m.shape
        return ok, f"shape={shp}, 参照マスクと一致={'OK' if ok else 'NG'}"

    check("ex1_frame_diff_mask", _c1)

    def _c2():
        n = ex2_count_blobs(blob_mask, 150)
        return (n == ref_blob_count, f"数えた塊={n}（期待 {ref_blob_count}）")

    check("ex2_count_blobs", _c2)

    def _c3():
        good = ex3_lk_good_new(g0, g1, p0)
        good = np.asarray(good)
        ok = good.ndim == 2 and good.shape[1] == 2 and len(good) == ref_good_count
        return ok, f"追跡成功 {len(good)} 点（期待 {ref_good_count}）"

    check("ex3_lk_good_new", _c3)

    def _c4():
        mag = ex4_flow_magnitude(flow)
        mag = np.asarray(mag)
        ok = mag.shape == ref_mag.shape and np.allclose(mag, ref_mag, atol=1e-4)
        return ok, f"shape={mag.shape}, 参照と一致={'OK' if ok else 'NG'}"

    check("ex4_flow_magnitude", _c4)

    def _c5():
        epe = ex5_mean_epe(flow, gt)
        ok = isinstance(epe, float) and abs(epe - ref_epe) < 1e-4
        return ok, f"EPE={epe:.4f}（期待 {ref_epe:.4f}）"

    check("ex5_mean_epe", _c5)

    print("=== 採点結果 ===")
    all_ok = True
    for name, ok, detail in results:
        mark = "PASS" if ok else "FAIL"
        all_ok = all_ok and ok
        print(f"  [{mark}] {name:22s} {detail}")
    print("\nALL PASS 🎉" if all_ok else "\nまだ未達の演習があります。TODO を埋めましょう。")


# =====================================================================
# 模範解答（SHOW_SOLUTION=1 のとき本体へ差し替えて実行）。まず自力で！
# =====================================================================

def _sol_ex1(prev, nxt, thresh=25):
    diff = cv2.absdiff(prev, nxt)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
    return mask


def _sol_ex2(mask, min_area=150):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return sum(1 for c in contours if cv2.contourArea(c) >= min_area)


def _sol_ex3(prev_gray, next_gray, p0):
    p1, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, next_gray, p0, None,
                                             **LK_PARAMS)
    return p1[status.ravel() == 1].reshape(-1, 2)


def _sol_ex4(flow):
    return np.linalg.norm(flow, axis=2)


def _sol_ex5(flow, gt):
    return float(np.linalg.norm(flow - gt, axis=2).mean())


def _install_solutions() -> None:
    g = globals()
    g["ex1_frame_diff_mask"] = _sol_ex1
    g["ex2_count_blobs"] = _sol_ex2
    g["ex3_lk_good_new"] = _sol_ex3
    g["ex4_flow_magnitude"] = _sol_ex4
    g["ex5_mean_epe"] = _sol_ex5


if __name__ == "__main__":
    if os.environ.get("SHOW_SOLUTION") == "1":
        print("(模範解答モードで実行します)\n")
        _install_solutions()
    _grade()
