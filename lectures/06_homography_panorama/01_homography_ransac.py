"""01: ホモグラフィ推定の核心 — findHomography(RANSAC)・インライアmask・再投影誤差。

学ぶこと:
  - 対応点からホモグラフィ(3x3 射影変換行列)を推定する正準形 cv2.findHomography。
  - RANSAC が返すインライア mask の読み方と、その数/比による品質チェック。
  - 推定の良さを「再投影誤差(対応点を写した後の平均ユークリッド距離)」で数値評価する。
  - ホモグラフィは最低4対応点が必要。3点では None、ノイズだらけでは破綻することを実験で確認。
  - cv2.perspectiveTransform で点(四隅)を写す → 平面物体の位置合わせ(物体検出)へ応用。

実行:
  uv run python lectures/06_homography_panorama/01_homography_ransac.py
結果は lectures/06_homography_panorama/outputs/ に保存される（画面表示はしない）。
"""

from __future__ import annotations

import pathlib
import sys

import cv2
import numpy as np

# 同じフォルダの cv_helpers.py を import できるようにする（どこから実行してもぶれない）。
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from cv_helpers import (  # noqa: E402
    estimate_homography,
    make_scene,
    make_two_views,
    orb_match,
    output_dir,
    points_from_matches,
    reprojection_error,
    warp_corners,
)

# RANSAC は内部で乱数を使う。毎回同じ結果にするため種を固定（再現性のため）。
cv2.setRNGSeed(0)


def demo_estimate_and_evaluate(out: pathlib.Path) -> None:
    """2視点からホモグラフィを推定し、インライア数・再投影誤差・真値との一致を確認する。"""
    # 重なりのある2視点 A(左)・B(右) と、答え合わせ用の真のホモグラフィ H_BtoA を得る。
    view_a, view_b, h_true = make_two_views()
    cv2.imwrite(str(out / "01_view_a.png"), view_a)
    cv2.imwrite(str(out / "01_view_b.png"), view_b)

    # === 1. 対応点を作る（ORB + Lowe 比率テスト） =========================
    kp1, kp2, good = orb_match(view_a, view_b, ratio=0.75)
    print("[1] ORB マッチング")
    print(f"  キーポイント A={len(kp1)} / B={len(kp2)} / 比率テスト後の good={len(good)}")

    # マッチを可視化（A の点と B の点を線で結ぶ）。
    match_vis = cv2.drawMatches(
        view_a, kp1, view_b, kp2, good[:80], None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    cv2.imwrite(str(out / "01_matches_all.png"), match_vis)

    # === 2. findHomography(RANSAC) ========================================
    # B の点 → A の点 へ写すホモグラフィを推定する（pts_src=B, pts_dst=A）。
    pts_a, pts_b = points_from_matches(kp1, kp2, good)
    H, mask = estimate_homography(pts_b, pts_a, thresh=3.0)
    inliers = int(mask.sum())
    print("\n[2] findHomography(RANSAC)")
    print(f"  インライア数 = {inliers} / {len(good)}  (比 {inliers / len(good):.2%})")

    # インライア(RANSACが採用した対応)だけを緑で描く。外れ値は描かれない。
    inlier_vis = cv2.drawMatches(
        view_a, kp1, view_b, kp2, good, None,
        matchesMask=mask.ravel().tolist(),
        matchColor=(0, 255, 0),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    cv2.imwrite(str(out / "01_matches_inliers.png"), inlier_vis)

    # === 3. 再投影誤差で品質を測る =========================================
    # インライアだけで「B の点を H で写した先」と「実際の A の点」の距離平均を測る。
    err_inliers = reprojection_error(H, pts_b, pts_a, mask)
    err_all = reprojection_error(H, pts_b, pts_a, None)
    print("\n[3] 再投影誤差（小さいほど良い）")
    print(f"  インライアのみ : {err_inliers:.3f} px  ← これが実質的な精度")
    print(f"  全対応点含む   : {err_all:.3f} px  ← 外れ値が混ざると大きくなる")

    # === 4. 推定 H と真値 H の比較 =========================================
    # スケールを [2,2]=1 に正規化してから並べると、ほぼ一致しているのが分かる。
    h_norm = H / H[2, 2]
    print("\n[4] 推定ホモグラフィ vs 真値（[2,2]=1 に正規化）")
    print("  推定 H_BtoA =\n", np.round(h_norm, 4))
    print("  真値 H_BtoA =\n", np.round(h_true / h_true[2, 2], 4))
    frob = float(np.linalg.norm(h_norm - h_true / h_true[2, 2]))
    print(f"  行列差のフロベニウスノルム = {frob:.4f}（0 に近いほど一致）")


def demo_failure_modes(out: pathlib.Path) -> None:
    """「最低4点必要」「外れ値だらけだと破綻」という実務注意を実験で示す。"""
    view_a, view_b, _ = make_two_views()
    kp1, kp2, good = orb_match(view_a, view_b, ratio=0.75)
    pts_a, pts_b = points_from_matches(kp1, kp2, good)

    print("\n[5] 失敗モードの体験")
    # (a) 3点だけ → ホモグラフィは解けない（None が返る）。
    H3, _ = estimate_homography(pts_b[:3], pts_a[:3])
    print(f"  3点だけで推定 → H is None: {H3 is None}  （最低4点必要）")

    # (b) 比率テストを通さない誤対応だらけ（ランダム対応）→ インライアがほとんど出ない。
    rng = np.random.default_rng(0)
    h, w = view_a.shape[:2]
    rand_src = rng.uniform(0, w, size=(60, 1, 2)).astype(np.float32)
    rand_dst = rng.uniform(0, h, size=(60, 1, 2)).astype(np.float32)
    Hr, maskr = estimate_homography(rand_src, rand_dst)
    n_in = 0 if maskr is None else int(maskr.sum())
    print(f"  でたらめな60対応 → インライア {n_in}/60  （比率テスト+RANSACの重要性）")
    print("  → 良い対応(比率テスト)を入れ、mask のインライア数を必ず検証すること。")


def demo_perspective_corners(out: pathlib.Path) -> None:
    """perspectiveTransform で B 画像の四隅を A の座標系へ写し、重なり範囲を描く。"""
    view_a, view_b, h_true = make_two_views()
    kp1, kp2, good = orb_match(view_a, view_b)
    pts_a, pts_b = points_from_matches(kp1, kp2, good)
    H, _ = estimate_homography(pts_b, pts_a)

    h, w = view_b.shape[:2]
    # B の四隅 (0,0)-(w,h) を H で写すと、A 座標系での「B が占める四辺形」になる。
    corners = warp_corners(H, w, h)
    canvas = view_a.copy()
    cv2.polylines(canvas, [corners.astype(np.int32)], isClosed=True,
                  color=(0, 0, 255), thickness=3, lineType=cv2.LINE_AA)
    cv2.imwrite(str(out / "01_overlap_quad.png"), canvas)
    print("\n[6] perspectiveTransform で B の四隅を A 座標へ投影")
    print("  A 座標系での B の四隅:\n", np.round(corners, 1))
    print("  → 01_overlap_quad.png に赤枠で重なり位置を描画")


def _make_card(text: str = "TARGET") -> np.ndarray:
    """平面物体検出デモ用の「カード（看板）」を合成する。特徴になる文字・枠・図形入り。"""
    card = np.full((180, 280, 3), 250, dtype=np.uint8)
    cv2.rectangle(card, (8, 8), (271, 171), (180, 60, 60), thickness=6)
    cv2.putText(card, text, (24, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.4,
                (30, 30, 200), 3, cv2.LINE_AA)
    cv2.putText(card, "homography", (24, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                (60, 120, 30), 2, cv2.LINE_AA)
    cv2.circle(card, (240, 40), 18, (0, 140, 220), thickness=-1)
    return card


def demo_planar_object(out: pathlib.Path) -> None:
    """平面物体の位置合わせ（応用）: 雑然としたシーンに歪んで写ったカードを検出して枠を描く。"""
    card = _make_card()
    ch, cw = card.shape[:2]

    # 背景の散らかったシーンに、カードを射影変換して貼り付ける（既知の四辺形へ）。
    scene = make_scene(height=540, width=820, seed=3)
    dst_quad = np.float32([[120, 90], [560, 140], [600, 430], [150, 380]])
    src_quad = np.float32([[0, 0], [cw, 0], [cw, ch], [0, ch]])
    M = cv2.getPerspectiveTransform(src_quad, dst_quad)
    warped = cv2.warpPerspective(card, M, (scene.shape[1], scene.shape[0]))
    mask = cv2.warpPerspective(np.full((ch, cw), 255, np.uint8), M,
                               (scene.shape[1], scene.shape[0]))
    scene[mask > 0] = warped[mask > 0]
    cv2.imwrite(str(out / "01_object_scene.png"), scene)

    # テンプレート(card) → シーン へ対応を取り、ホモグラフィでカードの四隅を投影して枠を描く。
    kp1, kp2, good = orb_match(card, scene, ratio=0.75)
    print("\n[7] 平面物体の検出（テンプレート → シーン）")
    print(f"  good マッチ = {len(good)}")
    if len(good) < 4:
        print("  対応が足りず検出スキップ（exit は正常）")
        return
    pts_card, pts_scene = points_from_matches(kp1, kp2, good)
    H, maskm = estimate_homography(pts_card, pts_scene)
    if H is None:
        print("  ホモグラフィ推定に失敗（exit は正常）")
        return
    print(f"  インライア = {int(maskm.sum())}/{len(good)}")
    found = warp_corners(H, cw, ch)
    result = scene.copy()
    cv2.polylines(result, [found.astype(np.int32)], True, (0, 255, 0), 3, cv2.LINE_AA)
    cv2.imwrite(str(out / "01_object_detected.png"), result)
    print("  → 01_object_detected.png に緑枠で検出位置を描画")


def main() -> None:
    out = output_dir()
    demo_estimate_and_evaluate(out)
    demo_failure_modes(out)
    demo_perspective_corners(out)
    demo_planar_object(out)
    print("\n完了。lectures/06_homography_panorama/outputs/ の 01_*.png を確認してください。")


if __name__ == "__main__":
    main()
