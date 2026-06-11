"""02: 手作りパノラマ合成 — warpPerspective での位置合わせ・キャンバス計算・シーム/ブレンド。

学ぶこと:
  - 2枚を貼り合わせる最小パイプライン: マッチ → findHomography → warpPerspective → 合成。
  - warpPerspective は出力が原点(0,0)始まり。負へはみ出す画像は平行移動 T で押し込む。
  - 重なり領域をそのまま上書きすると段差(シーム)が出る。フェザー(加重平均)で滑らかに溶かす。
  - 3枚以上はホモグラフィを掛け合わせ、基準フレームへ順次そろえてつなぐ。

実行:
  uv run python lectures/06_homography_panorama/02_panorama_manual.py
結果は outputs/06_homography_panorama/ に保存される（画面表示はしない）。
"""

from __future__ import annotations

import pathlib
import sys

import cv2
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from cv_helpers import (  # noqa: E402
    build_panorama,
    canvas_for,
    estimate_homography,
    feather_weight,
    make_three_views,
    make_two_views,
    orb_match,
    output_dir,
    points_from_matches,
)

cv2.setRNGSeed(0)


def stitch_two_step_by_step(out: pathlib.Path) -> None:
    """2枚の貼り合わせを段階ごとに保存し、上書き合成とフェザー合成を比較する。"""
    # A(基準/左) と B(右)。B を A の座標系へ写してつなぐ。
    view_a, view_b, _ = make_two_views()

    # === 1. 対応点 → ホモグラフィ（B → A） ================================
    kp1, kp2, good = orb_match(view_a, view_b)
    pts_a, pts_b = points_from_matches(kp1, kp2, good)
    H, mask = estimate_homography(pts_b, pts_a)  # B の点 → A の点
    print("[1] 位置合わせ")
    print(f"  good={len(good)} / インライア={int(mask.sum())}")

    # === 2. キャンバスサイズと平行移動 T ===================================
    # A は単位行列(そのまま)、B は H で写す。両方が収まる外接矩形からキャンバスを決める。
    T, size = canvas_for([view_a, view_b], [np.eye(3), H])
    print(f"[2] キャンバス size={size}（負座標を T で原点側へ押し込む）")

    # === 3. それぞれをキャンバスへ warp ===================================
    # A も T だけ平行移動して置く。B は T @ H で「平行移動込みの位置合わせ」をする。
    warp_a = cv2.warpPerspective(view_a, T, size)
    warp_b = cv2.warpPerspective(view_b, T @ H, size)
    cv2.imwrite(str(out / "02_warp_a.png"), warp_a)
    cv2.imwrite(str(out / "02_warp_b.png"), warp_b)

    # 各画像が「中身を持つ」場所のマスク（warp 後に黒(0)でない領域）。
    mask_a = cv2.warpPerspective(np.full(view_a.shape[:2], 255, np.uint8), T, size)
    mask_b = cv2.warpPerspective(np.full(view_b.shape[:2], 255, np.uint8), T @ H, size)

    # === 4a. 単純な上書き合成（シームが出る悪い例） =======================
    naive = warp_a.copy()
    naive[mask_b > 0] = warp_b[mask_b > 0]  # B で上書き → 重なり境界に段差が出る
    cv2.imwrite(str(out / "02_pano_naive.png"), naive)

    # === 4b. フェザー(加重平均)合成（滑らかな良い例） =====================
    # 各画像の「縁ほど軽い」重みを warp し、重み付き平均でブレンドする。
    w_a = cv2.warpPerspective(feather_weight(view_a.shape[:2]), T, size)
    w_b = cv2.warpPerspective(feather_weight(view_b.shape[:2]), T @ H, size)
    wsum = w_a + w_b
    wsum[wsum == 0] = 1.0  # どちらも無い空白でのゼロ割り回避
    blended = (warp_a.astype(np.float64) * w_a[:, :, None]
               + warp_b.astype(np.float64) * w_b[:, :, None]) / wsum[:, :, None]
    feather = blended.clip(0, 255).astype(np.uint8)
    cv2.imwrite(str(out / "02_pano_feather.png"), feather)

    # 重なり領域だけを可視化（A と B 両方が中身を持つ画素）。
    overlap = ((mask_a > 0) & (mask_b > 0)).astype(np.uint8) * 255
    cv2.imwrite(str(out / "02_overlap_mask.png"), overlap)
    print("[3] 02_pano_naive.png（段差あり）と 02_pano_feather.png（滑らか）を比較")


def stitch_three(out: pathlib.Path) -> None:
    """3枚を順次つないでパノラマにする（ホモグラフィの合成）。"""
    views = make_three_views()
    pano, inliers = build_panorama(views)
    cv2.imwrite(str(out / "02_pano_three.png"), pano)
    print("\n[4] 3枚パノラマ")
    print(f"  出力サイズ = {pano.shape[1]}x{pano.shape[0]}")
    print(f"  ペアごとのインライア数 = {inliers}")
    print("  → 02_pano_three.png を確認")


def main() -> None:
    out = output_dir()
    stitch_two_step_by_step(out)
    stitch_three(out)
    print("\n完了。outputs/06_homography_panorama/ の 02_*.png を確認してください。")


if __name__ == "__main__":
    main()
