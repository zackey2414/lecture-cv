"""第5回 演習問題（特徴点検出とマッチング / テンプレート / Hough）。

使い方:
  1. 各 exN_*() 関数の中の TODO を自分で実装する（最初は NotImplementedError が出る）。
  2. 実装できたら自己採点を実行:
         uv run python lectures/05_classical_features_matching/exercises.py
     全問が pass すれば "ALL PASS" と表示される。
  3. どうしても分からない時は、模範解答を実行して挙動を確認する:
         SHOW_SOLUTION=1 uv run python lectures/05_classical_features_matching/exercises.py
     または別ファイルの完全な模範解答を実行する:
         uv run python lectures/05_classical_features_matching/exercises_solutions.py
     （まずは自力で！）

問題は易→難の 9 問:
  ex1 比率テスト / ex2 インライア率 / ex3 距離種別 / ex4 テンプレ最良位置 /
  ex5 線分カウント / ex6 対応点配列の構築 / ex7 crossCheck 対応数 /
  ex8 マルチスケール最良倍率 / ex9 円の個数カウント

ヒント: サンプル画像は cv_helpers の make_scene_bgr / warp_to_view / make_object_scene /
make_shapes_bgr で合成できる（すべて BGR uint8）。未実装でも採点は最後まで走る（落ちない）。
"""

from __future__ import annotations

import os
import pathlib
import sys

import cv2
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from cv_helpers import make_object_scene, make_scene_bgr, make_shapes_bgr, warp_to_view  # noqa: E402


# =====================================================================
# 演習（ここを実装する）
# =====================================================================

def ex1_ratio_test(knn_matches: list, ratio: float = 0.75) -> list:
    """演習1【易】: Lowe の比率テストで「良いマッチ」だけを残したリストを返す。

    knn_matches は各クエリ点ごとの近傍ペア [m, n]（cv2.DMatch が 2 個）のリスト。
    条件 m.distance < ratio * n.distance を満たす m（最近傍 DMatch）だけを集めて返す。
    近傍が 1 個しかない（len < 2）ペアは比較できないのでスキップすること。
    """
    # TODO: knn_matches を回し、len>=2 を確認してから比率条件を満たす m を good に append して返す
    raise NotImplementedError


def ex2_inlier_ratio(src_pts: np.ndarray, dst_pts: np.ndarray) -> float:
    """演習2【中】: 対応点から RANSAC でホモグラフィを推定し、インライア率を返す。

    src_pts / dst_pts は形状 (N, 1, 2) の float32 対応点。
    cv2.findHomography(..., cv2.RANSAC, 5.0) の返す mask の合計をインライア数とし、
    インライア数 / 全対応数 を float で返す。点数が 4 未満なら 0.0 を返す。
    """
    # TODO: findHomography で mask を得て、int(mask.sum())/len(src_pts) を返す（4 点未満は 0.0）
    raise NotImplementedError


def ex3_norm_for(detector_name: str) -> int:
    """演習3【易】: 検出器に合った BFMatcher の距離種別（norm type）を返す。

    "SIFT" は浮動小数記述子なので cv2.NORM_L2、"ORB" は 2 進記述子なので cv2.NORM_HAMMING。
    それ以外の名前なら ValueError を投げること。
    """
    # TODO: detector_name に応じて cv2.NORM_L2 / cv2.NORM_HAMMING を返す。未知なら ValueError
    raise NotImplementedError


def ex4_template_topleft(scene: np.ndarray, template: np.ndarray) -> tuple[int, int]:
    """演習4【中】: TM_CCOEFF_NORMED でテンプレートを探し、最良一致の左上座標 (x, y) を返す。

    cv2.matchTemplate(scene, template, cv2.TM_CCOEFF_NORMED) の結果に cv2.minMaxLoc を使い、
    CCOEFF 系は「最大」が答えなので max_loc（=左上の (x, y)）を返す。
    """
    # TODO: matchTemplate → minMaxLoc で max_loc を取り出して返す（(int, int)）
    raise NotImplementedError


def ex5_count_line_segments(edges: np.ndarray, threshold: int) -> int:
    """演習5【中】: エッジ画像に HoughLinesP をかけ、検出された線分の本数を返す。

    cv2.HoughLinesP(edges, 1, np.pi/180, threshold, minLineLength=40, maxLineGap=10) を使う。
    検出が 0 本のとき返り値は None になる点に注意（その場合は 0 を返す）。
    """
    # TODO: HoughLinesP を呼び、None なら 0、そうでなければ len(lines) を返す
    raise NotImplementedError


def ex6_matched_points(kp1: list, kp2: list, good: list) -> tuple[np.ndarray, np.ndarray]:
    """演習6【中】: 良マッチ列から findHomography 用の対応点配列 (src, dst) を作る。

    各 DMatch m について、画像1側の点は kp1[m.queryIdx].pt、画像2側の点は kp2[m.trainIdx].pt。
    返り値は (src_pts, dst_pts)。どちらも dtype=float32・形状 (N, 1, 2)。
    （この形が cv2.findHomography / cv2.perspectiveTransform が要求する正準形。）
    """
    # TODO: kp1/kp2 から queryIdx/trainIdx で座標を引き、float32 の (N,1,2) で返す
    raise NotImplementedError


def ex7_cross_check_count(des1: np.ndarray, des2: np.ndarray, norm_type: int) -> int:
    """演習7【中】: crossCheck マッチング（相互最近傍）で残る対応数を返す。

    cv2.BFMatcher(norm_type, crossCheck=True) を作り、match()（knnMatch ではない）を呼ぶ。
    crossCheck=True は「1→2 の最近傍」と「2→1 の最近傍」が一致した対応だけを返す。
    返り値はその対応数（len）。比率テストとは併用しない別系統の誤対応除去であることに注意。
    """
    # TODO: crossCheck=True の BFMatcher で match() し、対応数 len を返す
    raise NotImplementedError


def ex8_best_template_scale(scene: np.ndarray, template: np.ndarray,
                            scales: list[float]) -> float:
    """演習8【難】: マルチスケール探索で、最も一致スコアの高いテンプレ倍率を返す。

    各 scale について template を fx=fy=scale でリサイズし（縮小は INTER_AREA、
    拡大は INTER_CUBIC）、TM_CCOEFF_NORMED で scene を探索して最大スコアを得る。
    リサイズ後テンプレが scene より大きい倍率はスキップする。全 scale 中で最大スコアの倍率を返す。
    （有効な候補が 1 つも無ければ -1.0 を返す。）
    """
    # TODO: scales を回し、各倍率の最大スコアを比較して最良の scale を返す
    raise NotImplementedError


def ex9_count_hough_circles(gray: np.ndarray, param2: int) -> int:
    """演習9【中】: HoughCircles で円を検出し、その個数を返す。

    cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=40, param1=120,
                     param2=param2, minRadius=20, maxRadius=80) を使う。
    入力は「グレー画像」を渡すこと（エッジ画像を渡すと内部 Canny と二重になり失敗する）。
    検出ゼロのとき返り値は None になる点に注意（その場合は 0 を返す）。
    """
    # TODO: HoughCircles を呼び、None なら 0、そうでなければ res.shape[1] を返す
    raise NotImplementedError


# =====================================================================
# 自己採点ランナー
# =====================================================================

def _match_key(dm) -> tuple[int, int]:
    """DMatch を (queryIdx, trainIdx) のタプルにして比較しやすくする。"""
    return (dm.queryIdx, dm.trainIdx)


def _grade() -> None:
    # --- 採点用の入力を合成 ---
    scene = make_scene_bgr()
    view, _ = warp_to_view(scene)
    g1 = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(view, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(nfeatures=1500)
    k1, d1 = orb.detectAndCompute(g1, None)
    k2, d2 = orb.detectAndCompute(g2, None)
    knn = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(d1, d2, k=2)

    obj_scene, template, true_tl, _true_br = make_object_scene()
    # ex8 用: シーンを 1.3 倍に拡大（正しい倍率は 1.3）。
    big_scene = cv2.resize(obj_scene, None, fx=1.3, fy=1.3, interpolation=cv2.INTER_CUBIC)
    scale_grid = [round(s, 2) for s in np.linspace(0.6, 1.6, 21)]

    shapes_gray = cv2.cvtColor(make_shapes_bgr(), cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(shapes_gray, 50, 150)

    results: list[tuple[str, bool, str]] = []

    def check(name: str, fn) -> None:
        try:
            ok, detail = fn()
            results.append((name, ok, detail))
        except NotImplementedError:
            results.append((name, False, "未実装（TODOを埋めてください）"))
        except Exception as e:  # noqa: BLE001
            results.append((name, False, f"例外: {type(e).__name__}: {e}"))

    # ex1: 比率テストの結果が模範解答と一致するか（query/train インデックスの集合で比較）。
    def _c1():
        got = ex1_ratio_test(knn, 0.75)
        ref = _sol_ex1(knn, 0.75)
        got_keys = sorted(_match_key(m) for m in got)
        ref_keys = sorted(_match_key(m) for m in ref)
        return (got_keys == ref_keys and len(ref_keys) > 0,
                f"good={len(got)} (期待 {len(ref)})")
    check("ex1_ratio_test", _c1)

    # ex2: インライア率が [0,1] かつ十分高いか（合成ペアはほぼ全点が幾何整合する）。
    def _c2():
        good = _sol_ex1(knn, 0.75)
        src = np.float32([k1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([k2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        r = ex2_inlier_ratio(src, dst)
        return (isinstance(r, float) and 0.0 <= r <= 1.0 and r >= 0.6,
                f"inlier率={r:.2f}")
    check("ex2_inlier_ratio", _c2)

    # ex3: 距離種別が正しいか。
    def _c3():
        ok = (ex3_norm_for("SIFT") == cv2.NORM_L2
              and ex3_norm_for("ORB") == cv2.NORM_HAMMING)
        return (ok, "SIFT=L2 / ORB=HAMMING")
    check("ex3_norm_for", _c3)

    # ex4: テンプレートの検出位置が正解の左上に十分近いか。
    def _c4():
        tl = ex4_template_topleft(obj_scene, template)
        err = abs(int(tl[0]) - true_tl[0]) + abs(int(tl[1]) - true_tl[1])
        return (err <= 3, f"検出={tuple(int(v) for v in tl)} 正解={true_tl} ズレ={err}px")
    check("ex4_template_topleft", _c4)

    # ex5: 線分本数が模範解答と一致するか（HoughLinesP は決定的）。
    def _c5():
        got = ex5_count_line_segments(edges, 80)
        ref = _sol_ex5(edges, 80)
        return (got == ref and ref > 0, f"線分数={got} (期待 {ref})")
    check("ex5_count_line_segments", _c5)

    # ex6: 対応点配列の形・型・中身が模範解答と一致するか。
    def _c6():
        good = _sol_ex1(knn, 0.75)
        src, dst = ex6_matched_points(k1, k2, good)
        rsrc, rdst = _sol_ex6(k1, k2, good)
        shape_ok = (src.shape == rsrc.shape == (len(good), 1, 2)
                    and dst.shape == rdst.shape)
        dtype_ok = (src.dtype == np.float32 and dst.dtype == np.float32)
        val_ok = (np.allclose(src, rsrc) and np.allclose(dst, rdst))
        return (bool(shape_ok and dtype_ok and val_ok),
                f"src.shape={src.shape} dtype={src.dtype}")
    check("ex6_matched_points", _c6)

    # ex7: crossCheck の対応数が模範解答と一致するか（決定的）。
    def _c7():
        got = ex7_cross_check_count(d1, d2, cv2.NORM_HAMMING)
        ref = _sol_ex7(d1, d2, cv2.NORM_HAMMING)
        return (got == ref and ref > 0, f"対応数={got} (期待 {ref})")
    check("ex7_cross_check_count", _c7)

    # ex8: マルチスケール最良倍率が正しい 1.3 付近か（模範解答とも一致）。
    def _c8():
        got = ex8_best_template_scale(big_scene, template, scale_grid)
        ref = _sol_ex8(big_scene, template, scale_grid)
        return (abs(got - ref) < 1e-6 and abs(got - 1.3) <= 0.1,
                f"最良scale={got} (期待 {ref}, 真値1.3)")
    check("ex8_best_template_scale", _c8)

    # ex9: 円の個数が模範解答と一致するか（決定的）。
    def _c9():
        got = ex9_count_hough_circles(shapes_gray, 30)
        ref = _sol_ex9(shapes_gray, 30)
        return (got == ref and ref > 0, f"円数={got} (期待 {ref})")
    check("ex9_count_hough_circles", _c9)

    print("=== 採点結果 ===")
    all_ok = True
    for name, ok, detail in results:
        mark = "PASS" if ok else "FAIL"
        all_ok = all_ok and ok
        print(f"  [{mark}] {name:24s} {detail}")
    print("\nALL PASS 🎉" if all_ok else "\nまだ未達の演習があります。TODO を埋めましょう。")


# =====================================================================
# 模範解答（SHOW_SOLUTION=1 のときに本体へ差し替えて実行）
# まずは自力で解いてから見ること。完全版は exercises_solutions.py にもある。
# =====================================================================

def _sol_ex1(knn_matches, ratio=0.75):
    good = []
    for pair in knn_matches:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            good.append(m)
    return good


def _sol_ex2(src_pts, dst_pts):
    if len(src_pts) < 4:
        return 0.0
    _, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if mask is None:
        return 0.0
    return float(int(mask.sum()) / len(src_pts))


def _sol_ex3(detector_name):
    if detector_name == "SIFT":
        return cv2.NORM_L2
    if detector_name == "ORB":
        return cv2.NORM_HAMMING
    raise ValueError(f"未知の検出器: {detector_name}")


def _sol_ex4(scene, template):
    res = cv2.matchTemplate(scene, template, cv2.TM_CCOEFF_NORMED)
    _min_v, _max_v, _min_l, max_l = cv2.minMaxLoc(res)
    return (int(max_l[0]), int(max_l[1]))


def _sol_ex5(edges, threshold):
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold,
                            minLineLength=40, maxLineGap=10)
    return 0 if lines is None else len(lines)


def _sol_ex6(kp1, kp2, good):
    src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    return src, dst


def _sol_ex7(des1, des2, norm_type):
    bf = cv2.BFMatcher(norm_type, crossCheck=True)
    return len(bf.match(des1, des2))


def _sol_ex8(scene, template, scales):
    best_score, best_scale = -1.0, -1.0
    for scale in scales:
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
        tmpl = cv2.resize(template, None, fx=scale, fy=scale, interpolation=interp)
        if tmpl.shape[0] >= scene.shape[0] or tmpl.shape[1] >= scene.shape[1]:
            continue
        res = cv2.matchTemplate(scene, tmpl, cv2.TM_CCOEFF_NORMED)
        _min_v, max_v, _min_l, _max_l = cv2.minMaxLoc(res)
        if max_v > best_score:
            best_score, best_scale = float(max_v), float(scale)
    return best_scale


def _sol_ex9(gray, param2):
    res = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=40,
                           param1=120, param2=param2, minRadius=20, maxRadius=80)
    return 0 if res is None else res.shape[1]


def _install_solutions() -> None:
    """模範解答で TODO 関数を差し替える（教材検証・答え合わせ用）。"""
    g = globals()
    g["ex1_ratio_test"] = _sol_ex1
    g["ex2_inlier_ratio"] = _sol_ex2
    g["ex3_norm_for"] = _sol_ex3
    g["ex4_template_topleft"] = _sol_ex4
    g["ex5_count_line_segments"] = _sol_ex5
    g["ex6_matched_points"] = _sol_ex6
    g["ex7_cross_check_count"] = _sol_ex7
    g["ex8_best_template_scale"] = _sol_ex8
    g["ex9_count_hough_circles"] = _sol_ex9


if __name__ == "__main__":
    if os.environ.get("SHOW_SOLUTION") == "1":
        print("(模範解答モードで実行します)\n")
        _install_solutions()
    _grade()
