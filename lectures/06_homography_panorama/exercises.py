"""第6回 演習問題（ホモグラフィ推定とパノラマ合成）。

使い方:
  1. 各 exN_*() 関数の中の TODO を自分で実装する（最初は NotImplementedError が出る）。
  2. 実装できたら自己採点を実行:
         uv run python lectures/06_homography_panorama/exercises.py
     全問が PASS すれば "ALL PASS" と表示される。未実装でも例外で落ちず exit 0。
  3. どうしても分からない時は、模範解答を実行して全 PASS の挙動を確認する:
         uv run python lectures/06_homography_panorama/exercises_solutions.py
     （まずは自力で！ 模範解答は採点ロジックを共有して全問 PASS する）

問題は易→難の順に並んでいる（ex1〜ex8）。
  ex1〜ex5: 対応点・推定・誤差・四隅投影・キャンバス（基礎）
  ex6     : ホモグラフィの合成（複数枚を基準フレームへそろえる）
  ex7     : warpPerspective でキャンバスへ配置＋マスク作成
  ex8     : フェザー（加重平均）ブレンドでシームを消す（パノラマ合成の総仕上げ）

ヒント: サンプルや部品は同じフォルダの cv_helpers.py にある。
  - make_two_views() : 重なりのある2視点 (A, B, 真のホモグラフィ)
  - orb_match(), points_from_matches() : 対応点づくり
  - feather_weight() : 縁ほど軽い重みマップ（フェザー用）
"""

from __future__ import annotations

import pathlib
import sys

import cv2
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from cv_helpers import (  # noqa: E402
    feather_weight,
    make_two_views,
    orb_match,
    output_dir,
    points_from_matches,
)


# =====================================================================
# 演習（ここを実装する）
# =====================================================================

def ex1_ratio_test(knn_matches: list, ratio: float = 0.75) -> list:
    """演習1: Lowe の比率テストを実装する。

    knn_matches は knnMatch(k=2) の結果（各要素は最大2個の DMatch を持つリスト）。
    各要素について「最近傍 m の距離 < ratio × 2番目 n の距離」なら m を採用してリストで返す。
    近傍が2個に満たない要素は捨てること。
    """
    # TODO: for pair in knn_matches: 要素数チェック → m,n を取り出し
    #       m.distance < ratio * n.distance なら good に m を append
    raise NotImplementedError


def ex2_estimate_homography(pts_src: np.ndarray, pts_dst: np.ndarray
                            ) -> tuple[np.ndarray | None, int]:
    """演習2: findHomography(RANSAC) で pts_src→pts_dst を推定し (H, インライア数) を返す。

    点が4対未満なら (None, 0) を返すこと（ホモグラフィは最低4対応点が必要）。
    それ以外は cv2.findHomography(..., cv2.RANSAC, 3.0) を使い、mask の合計をインライア数とする。
    """
    # TODO: len(pts_src) < 4 なら (None, 0)
    #       H, mask = cv2.findHomography(pts_src, pts_dst, cv2.RANSAC, 3.0)
    #       return H, int(mask.sum())
    raise NotImplementedError


def ex3_reprojection_error(H: np.ndarray, pts_src: np.ndarray,
                           pts_dst: np.ndarray) -> float:
    """演習3: H で pts_src を写した点と pts_dst の平均ユークリッド距離(px)を返す。

    cv2.perspectiveTransform で写し、対応点ごとの距離を計算して平均する。
    """
    # TODO: proj = cv2.perspectiveTransform(pts_src.reshape(-1,1,2), H).reshape(-1,2)
    #       dst  = pts_dst.reshape(-1,2)
    #       return float(np.linalg.norm(proj - dst, axis=1).mean())
    raise NotImplementedError


def ex4_warp_corners(H: np.ndarray, width: int, height: int) -> np.ndarray:
    """演習4: 画像(width×height)の四隅を H で写した座標 (4,2) float32 を返す。

    四隅は [(0,0),(width,0),(width,height),(0,height)] の順。perspectiveTransform を使う。
    パノラマのキャンバス計算や平面物体の枠描画の土台になる処理。
    """
    # TODO: corners = np.float32([[0,0],[width,0],[width,height],[0,height]]).reshape(-1,1,2)
    #       return cv2.perspectiveTransform(corners, H).reshape(-1,2)
    raise NotImplementedError


def ex5_canvas_offset(shape0: tuple[int, int], shape1: tuple[int, int],
                      H: np.ndarray) -> tuple[tuple[int, int], tuple[int, int]]:
    """演習5: 画像0(基準/単位行列)と画像1(Hで写す)が両方収まるキャンバスを求める。

    返り値 ((width, height), (tx, ty))。
      - 画像0の四隅と、画像1の四隅をHで写した点を全部集め、外接矩形を求める。
      - tx, ty は最小座標を 0 に押し込む平行移動量（= -x_min, -y_min を整数化）。
    ヒント: ex4_warp_corners を H=単位行列でも使えば画像0の四隅も同じ流儀で得られる。
    """
    # TODO: 画像0四隅(np.eye(3)で写す) と 画像1四隅(Hで写す) を concatenate
    #       x_min,y_min = floor(min), x_max,y_max = ceil(max)
    #       size=(x_max-x_min, y_max-y_min), offset=(-x_min, -y_min)
    raise NotImplementedError


def ex6_compose_to_reference(pairwise_H: list[np.ndarray]) -> list[np.ndarray]:
    """演習6: 隣接ペアのホモグラフィを掛け合わせ、各画像を基準フレームへ写す行列を作る。

    pairwise_H[k] は「画像 k+1 → 画像 k」へ写す 3x3 行列（左→右に重なる並び）。
    返り値は長さ len(pairwise_H)+1 のリスト [M0, M1, M2, ...]。
      - M0 = 単位行列（基準=画像0 はそのまま）
      - M_k = M_{k-1} @ pairwise_H[k-1]    （ひとつ前までの合成に掛けてつなぐ）
    これが複数枚パノラマの「ホモグラフィの合成」の核心（第8節）。
    """
    # TODO: M_to_ref = [np.eye(3)]
    #       for H in pairwise_H: M_to_ref.append(M_to_ref[-1] @ H)
    #       return M_to_ref
    raise NotImplementedError


def ex7_warp_and_place(img: np.ndarray, H: np.ndarray, size: tuple[int, int]
                       ) -> tuple[np.ndarray, np.ndarray]:
    """演習7: 画像を H でキャンバス(size=(W,H))へ warp し、(warp画像, 中身マスク) を返す。

    返り値 (warped, mask)。
      - warped: cv2.warpPerspective(img, H, size) の結果（BGR uint8）。
      - mask  : 同じ H で全面 255 の画像を warp し、0 でない画素を 255 にした uint8 マスク。
    mask は「その画像がキャンバスのどこに中身を持つか」を表し、合成の重なり判定に使う。
    （フラグは指定せず既定の補間を使うこと。模範解答と画素が一致する。）
    """
    # TODO: warped = cv2.warpPerspective(img, H, size)
    #       ones   = np.full(img.shape[:2], 255, np.uint8)
    #       wm     = cv2.warpPerspective(ones, H, size)
    #       mask   = (wm > 0).astype(np.uint8) * 255
    #       return warped, mask
    raise NotImplementedError


def ex8_feather_blend(warp_a: np.ndarray, warp_b: np.ndarray,
                      w_a: np.ndarray, w_b: np.ndarray) -> np.ndarray:
    """演習8: 2枚の warp 済み画像を、重みマップで加重平均ブレンドして uint8 で返す。

    引数:
      - warp_a, warp_b : 同じキャンバスへ warp 済みの BGR 画像（uint8, 同サイズ）。
      - w_a, w_b       : それぞれの (H,W) 重みマップ（float, 縁ほど 0 に近い feather 重み）。
    重なり領域は重み付き平均: (warp_a*w_a + warp_b*w_b) / (w_a + w_b)。
    どちらの重みも 0 の空白画素はゼロ割りになるので、その画素だけ分母を 1 にして 0 を出す。
    最後に 0〜255 にクリップして uint8 にして返す（これでシームが滑らかに溶ける）。
    """
    # TODO: wsum = w_a + w_b
    #       wsum_safe = wsum.copy(); wsum_safe[wsum_safe == 0] = 1.0
    #       acc = warp_a*w_a[...,None] + warp_b*w_b[...,None]   （float で計算）
    #       return (acc / wsum_safe[...,None]).clip(0, 255).astype(np.uint8)
    raise NotImplementedError


# =====================================================================
# 採点コンテキスト（決定的データを1度だけ作る）
# =====================================================================

def _build_context() -> dict:
    """全演習の採点に使う決定的な入力をまとめて作る。"""
    cv2.setRNGSeed(0)
    output_dir()  # 出力先の存在確認だけ（保存はしない）
    view_a, view_b, _ = make_two_views()
    kp1, kp2, good_ref = orb_match(view_a, view_b, ratio=0.75)
    pts_a, pts_b = points_from_matches(kp1, kp2, good_ref)

    # ex1 用に knnMatch をそのまま用意（比率テスト前の生データ）。
    orb = cv2.ORB_create(nfeatures=1500)
    g1 = cv2.cvtColor(view_a, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(view_b, cv2.COLOR_BGR2GRAY)
    _, des1 = orb.detectAndCompute(g1, None)
    _, des2 = orb.detectAndCompute(g2, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    knn = bf.knnMatch(des1, des2, k=2)
    ref_good_count = sum(
        1 for p in knn if len(p) == 2 and p[0].distance < 0.75 * p[1].distance
    )

    # 固定ホモグラフィ（ex4〜ex8 の決定的チェック用）。
    fixed_h = np.array([[1.05, 0.02, 40.0],
                        [0.01, 0.98, 12.0],
                        [1e-4, 5e-5, 1.0]], dtype=np.float64)
    fixed_h2 = np.array([[0.97, -0.03, 30.0],
                         [0.02, 1.04, -8.0],
                         [-8e-5, 4e-5, 1.0]], dtype=np.float64)
    return {
        "view_a": view_a, "view_b": view_b,
        "pts_a": pts_a, "pts_b": pts_b,
        "knn": knn, "ref_good_count": ref_good_count,
        "fixed_h": fixed_h, "fixed_h2": fixed_h2,
    }


def _make_checks(ctx: dict) -> list[tuple[str, object]]:
    """(名前, 採点関数) のリストを返す。採点関数は (ok, detail) を返す。

    各採点関数はモジュールのグローバル ex*_*() を名前で呼ぶので、
    模範解答ファイルが ex*_*() を差し替えれば、そのまま全 PASS の確認に使える
    （採点ロジックを重複させない仕掛け）。
    """
    pts_a, pts_b = ctx["pts_a"], ctx["pts_b"]
    knn, ref_good_count = ctx["knn"], ctx["ref_good_count"]
    fixed_h, fixed_h2 = ctx["fixed_h"], ctx["fixed_h2"]
    view_a, view_b = ctx["view_a"], ctx["view_b"]

    def c1():
        good = ex1_ratio_test(knn, 0.75)
        return (len(good) == ref_good_count, f"good={len(good)} (期待 {ref_good_count})")

    def c2():
        H, n = ex2_estimate_homography(pts_b, pts_a)
        ok = H is not None and H.shape == (3, 3) and n >= 100
        return (ok, f"H={'(3,3)' if H is not None else None} / インライア={n}")

    def c3():
        # インライアの対応点だけを渡す（外れ値を含むと誤差は当然大きくなるため）。
        H, mask3 = cv2.findHomography(pts_b, pts_a, cv2.RANSAC, 3.0)
        keep = mask3.ravel().astype(bool)
        err = ex3_reprojection_error(H, pts_b[keep], pts_a[keep])
        return (err < 2.0, f"再投影誤差={err:.3f}px (<2 で合格)")

    def c4():
        got = np.asarray(ex4_warp_corners(fixed_h, 640, 480), dtype=np.float64)
        exp = cv2.perspectiveTransform(
            np.float32([[0, 0], [640, 0], [640, 480], [0, 480]]).reshape(-1, 1, 2),
            fixed_h,
        ).reshape(-1, 2)
        return (got.shape == (4, 2) and np.allclose(got, exp, atol=1e-3), "四隅の投影")

    def c5():
        size, offset = ex5_canvas_offset((480, 640), (480, 640), fixed_h)
        c0 = np.float32([[0, 0], [640, 0], [640, 480], [0, 480]])
        c1q = cv2.perspectiveTransform(c0.reshape(-1, 1, 2), fixed_h).reshape(-1, 2)
        pts = np.concatenate([c0, c1q], axis=0)
        x_min, y_min = np.floor(pts.min(0)).astype(int)
        x_max, y_max = np.ceil(pts.max(0)).astype(int)
        exp_size = (int(x_max - x_min), int(y_max - y_min))
        exp_off = (int(-x_min), int(-y_min))
        return (tuple(size) == exp_size and tuple(offset) == exp_off,
                f"size={tuple(size)} offset={tuple(offset)}")

    def c6():
        got = ex6_compose_to_reference([fixed_h, fixed_h2])
        exp = [np.eye(3), fixed_h, fixed_h @ fixed_h2]
        ok = (isinstance(got, list) and len(got) == 3
              and all(np.allclose(np.asarray(g, dtype=np.float64), e, atol=1e-6)
                      for g, e in zip(got, exp)))
        return (ok, f"M の枚数={len(got) if hasattr(got, '__len__') else '?'} (期待 3)")

    def c7():
        size = (820, 540)
        warped, mask = ex7_warp_and_place(view_a, fixed_h, size)
        exp_w = cv2.warpPerspective(view_a, fixed_h, size)
        ones = np.full(view_a.shape[:2], 255, np.uint8)
        exp_m = cv2.warpPerspective(ones, fixed_h, size)
        ok = (warped.shape == exp_w.shape and np.array_equal(warped, exp_w)
              and mask.shape[:2] == exp_m.shape[:2]
              and np.array_equal(mask > 0, exp_m > 0))
        return (ok, f"warp={warped.shape} 中身画素={int((mask > 0).sum())}")

    def c8():
        size = (900, 560)
        wa = cv2.warpPerspective(view_a, np.eye(3), size)
        wb = cv2.warpPerspective(view_b, fixed_h, size)
        ga = cv2.warpPerspective(feather_weight(view_a.shape[:2]), np.eye(3), size)
        gb = cv2.warpPerspective(feather_weight(view_b.shape[:2]), fixed_h, size)
        blended = ex8_feather_blend(wa, wb, ga, gb)
        wsum = (ga + gb).copy()
        wsum[wsum == 0] = 1.0
        exp = ((wa.astype(np.float64) * ga[:, :, None]
                + wb.astype(np.float64) * gb[:, :, None])
               / wsum[:, :, None]).clip(0, 255).astype(np.uint8)
        diff = float(np.abs(blended.astype(np.int32) - exp.astype(np.int32)).mean())
        ok = (blended.dtype == np.uint8 and blended.shape == exp.shape and diff < 1.0)
        return (ok, f"平均画素差={diff:.3f} (<1 で合格)")

    return [
        ("ex1_ratio_test", c1),
        ("ex2_estimate_homography", c2),
        ("ex3_reprojection_error", c3),
        ("ex4_warp_corners", c4),
        ("ex5_canvas_offset", c5),
        ("ex6_compose_to_reference", c6),
        ("ex7_warp_and_place", c7),
        ("ex8_feather_blend", c8),
    ]


# =====================================================================
# 自己採点ランナー（模範解答ファイルからも呼ばれる）
# =====================================================================

def grade() -> bool:
    """全演習を採点して結果を表示し、全 PASS かどうかを返す。

    未実装(NotImplementedError)や例外は握りつぶして FAIL 表示にするので、
    どれだけ未実装でもこの関数は例外を投げず正常終了する（exit 0）。
    """
    ctx = _build_context()
    results: list[tuple[str, bool, str]] = []
    for name, fn in _make_checks(ctx):
        try:
            ok, detail = fn()
            results.append((name, bool(ok), detail))
        except NotImplementedError:
            results.append((name, False, "未実装（TODOを埋めてください）"))
        except Exception as e:  # noqa: BLE001
            results.append((name, False, f"例外: {type(e).__name__}: {e}"))

    print("=== 採点結果 ===")
    all_ok = True
    for name, ok, detail in results:
        mark = "PASS" if ok else "FAIL"
        all_ok = all_ok and ok
        print(f"  [{mark}] {name:26s} {detail}")
    print("\nALL PASS 🎉" if all_ok else "\nまだ未達の演習があります。TODO を埋めましょう。")
    return all_ok


if __name__ == "__main__":
    grade()
