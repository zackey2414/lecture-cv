"""第10回 演習問題（古典的な動画処理 — 動き解析）。

使い方:
  1. 各 exN_*() 関数の中の TODO を自分で実装する（最初は NotImplementedError が出る）。
  2. 実装できたら自己採点を実行:
         uv run python lectures/10_classical_video_motion/exercises.py
     全問が PASS すれば "ALL PASS" と表示される（未実装でも落ちず FAIL 表示で正常終了）。
  3. どうしても分からない時は、模範解答を実行して挙動を確認する:
         SHOW_SOLUTION=1 uv run python lectures/10_classical_video_motion/exercises.py
     もしくは完全な模範解答スクリプト:
         uv run python lectures/10_classical_video_motion/exercises_solutions.py
     （まずは自力で！）

演習は易→難の 9 問:
  ex1 フレーム差分マスク      ex2 動体（塊）の数え上げ      ex3 LK で追跡成功点の新位置
  ex4 密フローの大きさ        ex5 平均終点誤差 EPE          ex6 背景差分(MOG2)で動体数
  ex7 LK の平均移動量         ex8 バックプロジェクションの最尤位置
  ex9 CamShift の窓サイズ成長率

ヒント: サンプルや部品は同じフォルダの cv_helpers.py にある。
  - make_motion_frames()   : 静止背景の上を物体が動く連番フレーム
  - make_translating_pair(): 既知シフトの画像ペア（EPE の答え合わせ用）
  - green_window()         : 緑の円を囲む初期追跡窓 (x, y, w, h)
"""

from __future__ import annotations

import os
import pathlib
import sys

import cv2
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from cv_helpers import (  # noqa: E402
    green_window,
    make_motion_frames,
    make_translating_pair,
    output_dir,
)

# LK で使う共通パラメータ（演習3・7用）。
LK_PARAMS = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
)

# 背景差分の後処理に使う構造要素（演習6用）。
KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# CamShift / meanShift の反復停止条件（演習9用）。
TERM_CRIT = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)


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


def ex6_bgsub_box_count(frames: list[np.ndarray], min_area: int = 150) -> int:
    """演習6: 背景差分(MOG2)を全フレームに順適用し、最終フレームの動体（塊）数を返す。

    手順（01_frame_diff_bgsub.py の実用版と同じ）:
      1. cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=25, detectShadows=True)
      2. frames を順に subtractor.apply(f) して背景モデルを育てる（慣らし運転）。
      3. 最後のマスクを cv2.threshold(.., 200, 255, THRESH_BINARY) で影(127)を落とす。
      4. KERNEL で MORPH_OPEN → MORPH_CLOSE してノイズ除去＋穴埋め。
      5. findContours で輪郭を取り、面積が min_area 以上の数を int で返す。
    """
    # TODO: sub = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=25, detectShadows=True)
    #       last = None
    #       for f in frames: last = sub.apply(f)
    #       _, fg = cv2.threshold(last, 200, 255, cv2.THRESH_BINARY)
    #       fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, KERNEL)
    #       fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, KERNEL)
    #       contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #       return sum(1 for c in contours if cv2.contourArea(c) >= min_area)
    raise NotImplementedError


def ex7_lk_mean_displacement(prev_gray: np.ndarray, next_gray: np.ndarray,
                             p0: np.ndarray) -> float:
    """演習7: LK で p0 を追跡し、『追跡成功した点』の平均移動量(px)を float で返す。

    手順:
      1. cv2.calcOpticalFlowPyrLK(..., **LK_PARAMS) で p1, status を得る。
      2. ok = status.ravel() == 1 の点だけ取り出す。
      3. 各点の移動量 = ||p1[ok] - p0[ok]|| を計算し、その平均を float で返す。
    追跡成功点が0個なら 0.0 を返す（落とさない）。
    """
    # TODO: p1, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, next_gray, p0, None, **LK_PARAMS)
    #       ok = status.ravel() == 1
    #       if ok.sum() == 0: return 0.0
    #       disp = np.linalg.norm(p1[ok] - p0[ok], axis=2).ravel()
    #       return float(disp.mean())
    raise NotImplementedError


def ex8_backproject_peak(frame: np.ndarray, window: tuple[int, int, int, int]
                         ) -> tuple[int, int]:
    """演習8: 窓の色モデルでバックプロジェクションし、確率が最大の画素 (x, y) を返す。

    生のバックプロジェクションは背景の似た色がポツポツ最大値で並ぶので、
    そのまま argmax すると孤立画素を拾う。そこで cv2.GaussianBlur で平滑化して
    『密に色が集まっている塊（=対象）』の中心を最尤位置として拾うのが定石（meanShift の素地）。

    手順:
      1. frame を HSV にし、window 領域の色相(H)ヒストグラム(16ビン,[0,180])を作る。
         S/V が低い画素は inRange((0,60,60),(180,255,255)) で除外してから calcHist。
         作ったヒストは cv2.normalize(.., 0, 255, NORM_MINMAX) で正規化する。
      2. cv2.calcBackProject([hsv],[0],hist,[0,180],1) で確率マップ(H,W)を作る。
      3. cv2.GaussianBlur(prob, (0, 0), 5) で平滑化する（孤立点を均し、塊を際立たせる）。
      4. 確率が最大の位置を np.argmax + np.unravel_index で求め、(x, y) を返す。
         （配列は (行=y, 列=x) なので、返すのは (x, y) の順であることに注意）
    """
    # TODO: x, y, w, h = window
    #       hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #       roi = hsv[y:y+h, x:x+w]
    #       m = cv2.inRange(roi, (0, 60, 60), (180, 255, 255))
    #       hist = cv2.calcHist([roi], [0], m, [16], [0, 180])
    #       cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
    #       prob = cv2.calcBackProject([hsv], [0], hist, [0, 180], 1)
    #       prob = cv2.GaussianBlur(prob, (0, 0), 5)
    #       py, px = np.unravel_index(int(np.argmax(prob)), prob.shape)
    #       return (int(px), int(py))
    raise NotImplementedError


def ex9_camshift_grow_ratio(frames: list[np.ndarray], hist: np.ndarray,
                            init_win: tuple[int, int, int, int]) -> float:
    """演習9: CamShift で全フレームを追跡し、窓面積の成長率(last/first)を float で返す。

    手順:
      1. 各フレームを cv2.calcBackProject で確率マップにする（hist は引数で渡される色モデル）。
      2. cv2.CamShift(prob, win, TERM_CRIT) で回転矩形 rot_rect と新しい win を得る。
         rot_rect[1] = (w, h) が窓のサイズ。面積 = w*h。
      3. 最初のフレームの面積を first、最後のフレームの面積を last として last/first を返す。
    緑円は半径が増えるので成長率は 1.0 より大きくなるはず（meanShift では 1.0 のまま）。
    """
    # TODO: win = init_win; first = last = None
    #       for i, f in enumerate(frames):
    #           hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
    #           prob = cv2.calcBackProject([hsv], [0], hist, [0, 180], 1)
    #           rot, win = cv2.CamShift(prob, win, TERM_CRIT)
    #           w, h = rot[1]; area = w * h
    #           if i == 0: first = area
    #           last = area
    #       return float(last / first) if first else 0.0
    raise NotImplementedError


# =====================================================================
# 採点用の小道具（解答ロジックではなく、参照値を作るための足場）
# =====================================================================

def _hue_hist(frame: np.ndarray, window: tuple[int, int, int, int]) -> np.ndarray:
    """採点足場: 窓領域の色相ヒストグラム（色モデル）を作る。ex9 の入力に使う。"""
    x, y, w, h = window
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    roi = hsv[y:y + h, x:x + w]
    m = cv2.inRange(roi, (0, 60, 60), (180, 255, 255))
    hist = cv2.calcHist([roi], [0], m, [16], [0, 180])
    cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
    return hist


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

    # ex3/ex7 用: グレースケール対と特徴点。
    g0 = cv2.cvtColor(frames[10], cv2.COLOR_BGR2GRAY)
    g1 = cv2.cvtColor(frames[11], cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(g0, maxCorners=200, qualityLevel=0.05,
                                 minDistance=7, blockSize=7)
    p1_ref, st_ref, _ = cv2.calcOpticalFlowPyrLK(g0, g1, p0, None, **LK_PARAMS)
    ref_good_count = int((st_ref.ravel() == 1).sum())
    ok_ref = st_ref.ravel() == 1
    ref_disp = float(np.linalg.norm(p1_ref[ok_ref] - p0[ok_ref], axis=2).ravel().mean())

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

    # ex6 用: MOG2 を全フレームに適用した参照動体数。
    _sub = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=25,
                                              detectShadows=True)
    _last = None
    for f in frames:
        _last = _sub.apply(f)
    _, _fg = cv2.threshold(_last, 200, 255, cv2.THRESH_BINARY)
    _fg = cv2.morphologyEx(_fg, cv2.MORPH_OPEN, KERNEL)
    _fg = cv2.morphologyEx(_fg, cv2.MORPH_CLOSE, KERNEL)
    ref_box_count = sum(
        1 for c in cv2.findContours(_fg, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)[0]
        if cv2.contourArea(c) >= 150
    )

    # ex8/ex9 用: 緑円の初期窓・色モデル・参照値。
    # ex8 は「窓から色モデルを作り、その同じフレームを背景投影して最尤位置を返す」。
    # 初期窓に緑円が写っている frames[0] を使えば、最尤位置は円の上に乗る。
    init_win = green_window(frames[0])
    hist = _hue_hist(frames[0], init_win)
    _hsv8 = cv2.cvtColor(frames[0], cv2.COLOR_BGR2HSV)
    _prob8 = cv2.calcBackProject([_hsv8], [0], hist, [0, 180], 1)
    _prob8 = cv2.GaussianBlur(_prob8, (0, 0), 5)
    _py, _px = np.unravel_index(int(np.argmax(_prob8)), _prob8.shape)
    ref_peak = (int(_px), int(_py))

    _win9 = init_win
    _first9 = _last9 = None
    for i, f in enumerate(frames):
        _hsv9 = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
        _prob9 = cv2.calcBackProject([_hsv9], [0], hist, [0, 180], 1)
        _rot9, _win9 = cv2.CamShift(_prob9, _win9, TERM_CRIT)
        _w9, _h9 = _rot9[1]
        if i == 0:
            _first9 = _w9 * _h9
        _last9 = _w9 * _h9
    ref_grow = float(_last9 / _first9) if _first9 else 0.0

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

    def _c6():
        n = ex6_bgsub_box_count(frames, 150)
        return (n == ref_box_count, f"MOG2 動体数={n}（期待 {ref_box_count}）")

    check("ex6_bgsub_box_count", _c6)

    def _c7():
        d = ex7_lk_mean_displacement(g0, g1, p0)
        ok = isinstance(d, float) and abs(d - ref_disp) < 1e-4
        return ok, f"平均移動量={d:.4f}px（期待 {ref_disp:.4f}）"

    check("ex7_lk_mean_displacement", _c7)

    def _c8():
        peak = ex8_backproject_peak(frames[0], init_win)
        peak = (int(peak[0]), int(peak[1]))
        ok = peak == ref_peak
        return ok, f"最尤位置(x,y)={peak}（期待 {ref_peak}）"

    check("ex8_backproject_peak", _c8)

    def _c9():
        r = ex9_camshift_grow_ratio(frames, hist, init_win)
        ok = isinstance(r, float) and r > 1.0 and abs(r - ref_grow) < 1e-3
        return ok, f"窓成長率={r:.3f}x（期待 {ref_grow:.3f}, >1.0）"

    check("ex9_camshift_grow_ratio", _c9)

    print("=== 採点結果 ===")
    all_ok = True
    for name, ok, detail in results:
        mark = "PASS" if ok else "FAIL"
        all_ok = all_ok and ok
        print(f"  [{mark}] {name:24s} {detail}")
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


def _sol_ex6(frames, min_area=150):
    sub = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=25,
                                             detectShadows=True)
    last = None
    for f in frames:
        last = sub.apply(f)
    _, fg = cv2.threshold(last, 200, 255, cv2.THRESH_BINARY)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, KERNEL)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, KERNEL)
    contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return sum(1 for c in contours if cv2.contourArea(c) >= min_area)


def _sol_ex7(prev_gray, next_gray, p0):
    p1, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, next_gray, p0, None,
                                             **LK_PARAMS)
    ok = status.ravel() == 1
    if ok.sum() == 0:
        return 0.0
    disp = np.linalg.norm(p1[ok] - p0[ok], axis=2).ravel()
    return float(disp.mean())


def _sol_ex8(frame, window):
    x, y, w, h = window
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    roi = hsv[y:y + h, x:x + w]
    m = cv2.inRange(roi, (0, 60, 60), (180, 255, 255))
    hist = cv2.calcHist([roi], [0], m, [16], [0, 180])
    cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
    prob = cv2.calcBackProject([hsv], [0], hist, [0, 180], 1)
    prob = cv2.GaussianBlur(prob, (0, 0), 5)
    py, px = np.unravel_index(int(np.argmax(prob)), prob.shape)
    return (int(px), int(py))


def _sol_ex9(frames, hist, init_win):
    win = init_win
    first = last = None
    for i, f in enumerate(frames):
        hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
        prob = cv2.calcBackProject([hsv], [0], hist, [0, 180], 1)
        rot, win = cv2.CamShift(prob, win, TERM_CRIT)
        w, h = rot[1]
        area = w * h
        if i == 0:
            first = area
        last = area
    return float(last / first) if first else 0.0


def _install_solutions() -> None:
    g = globals()
    g["ex1_frame_diff_mask"] = _sol_ex1
    g["ex2_count_blobs"] = _sol_ex2
    g["ex3_lk_good_new"] = _sol_ex3
    g["ex4_flow_magnitude"] = _sol_ex4
    g["ex5_mean_epe"] = _sol_ex5
    g["ex6_bgsub_box_count"] = _sol_ex6
    g["ex7_lk_mean_displacement"] = _sol_ex7
    g["ex8_backproject_peak"] = _sol_ex8
    g["ex9_camshift_grow_ratio"] = _sol_ex9


if __name__ == "__main__":
    if os.environ.get("SHOW_SOLUTION") == "1":
        print("(模範解答モードで実行します)\n")
        _install_solutions()
    _grade()
