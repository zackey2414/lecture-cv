"""03: 高レベル cv2.Stitcher との比較 — 自動パノラマ vs 手作りパイプライン。

学ぶこと:
  - cv2.Stitcher_create で「マッチ→推定→露出補正→マルチバンド合成」までを丸ごと自動化する正準形。
  - PANORAMA モード(回転カメラ想定)と SCANS モード(平面/スキャン想定)の違いと使い分け。
  - 失敗時は status が 0 以外。必ず status を確認し、握りつぶさない（exit は常に正常に）。
  - 2つのパノラマを位置合わせし、重なり領域の SSIM で「結果がどれだけ一致するか」を定量比較。

実行:
  uv run python lectures/06_homography_panorama/03_stitcher_compare.py
結果は outputs/06_homography_panorama/ に保存される（画面表示はしない）。
"""

from __future__ import annotations

import pathlib
import sys
import time

import cv2
import numpy as np

# 表示はせず保存するので matplotlib は Agg バックエンドを使う（headless 安全）。
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from cv_helpers import (  # noqa: E402
    build_panorama,
    estimate_homography,
    make_three_views,
    orb_match,
    output_dir,
    points_from_matches,
    ssim,
)

cv2.setRNGSeed(0)

# Stitcher の状態コードを読みやすい文字列にする辞書。
_STATUS = {
    cv2.Stitcher_OK: "OK",
    cv2.Stitcher_ERR_NEED_MORE_IMGS: "ERR_NEED_MORE_IMGS",
    cv2.Stitcher_ERR_HOMOGRAPHY_EST_FAIL: "ERR_HOMOGRAPHY_EST_FAIL",
    cv2.Stitcher_ERR_CAMERA_PARAMS_ADJUST_FAIL: "ERR_CAMERA_PARAMS_ADJUST_FAIL",
}


def run_stitcher(images: list[np.ndarray]) -> tuple[np.ndarray | None, str, float]:
    """cv2.Stitcher で自動合成する。PANORAMA で失敗したら SCANS にフォールバック。"""
    for mode_name, mode in (("PANORAMA", cv2.Stitcher_PANORAMA),
                            ("SCANS", cv2.Stitcher_SCANS)):
        stitcher = cv2.Stitcher_create(mode)
        t0 = time.perf_counter()
        status, pano = stitcher.stitch(images)
        dt = time.perf_counter() - t0
        label = _STATUS.get(status, f"status={status}")
        print(f"  Stitcher({mode_name}): {label}  {dt:.2f}s")
        if status == cv2.Stitcher_OK:
            return pano, mode_name, dt
    return None, "FAILED", 0.0


def overlap_ssim(pano_ref: np.ndarray, pano_other: np.ndarray) -> float | None:
    """2つのパノラマを位置合わせし、両方が中身を持つ重なり領域で SSIM を測る。

    2つの出力はキャンバスの座標系が違うので、まず other を ref に重ねる(ORB+homography)。
    重なった領域だけをグレースケールで比較するのが「重なり領域の SSIM」。
    対応が取れなければ None を返す（評価不能でも exit は正常）。
    """
    kp1, kp2, good = orb_match(pano_ref, pano_other, ratio=0.8)
    if len(good) < 10:
        return None
    pts_ref, pts_other = points_from_matches(kp1, kp2, good)
    H, mask = estimate_homography(pts_other, pts_ref)  # other → ref
    if H is None:
        return None
    size = (pano_ref.shape[1], pano_ref.shape[0])
    aligned = cv2.warpPerspective(pano_other, H, size)
    valid = cv2.warpPerspective(np.full(pano_other.shape[:2], 255, np.uint8), H, size)
    both = (valid > 0) & (pano_ref.sum(axis=2) > 0)  # 両方が中身を持つ画素
    if both.sum() < 1000:
        return None
    g_ref = cv2.cvtColor(pano_ref, cv2.COLOR_BGR2GRAY).astype(np.float64)
    g_other = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY).astype(np.float64)
    # 重なり外を 0 にしてから SSIM（窓平均の影響を抑えるためマスク領域に限定）。
    g_ref = np.where(both, g_ref, 0)
    g_other = np.where(both, g_other, 0)
    return ssim(g_ref, g_other)


def save_comparison(out: pathlib.Path, manual: np.ndarray,
                    auto: np.ndarray | None, auto_mode: str) -> None:
    """手作り版と Stitcher 版を1枚に並べて保存（matplotlib は BGR→RGB に変換）。"""
    n = 1 if auto is None else 2
    fig, axes = plt.subplots(n, 1, figsize=(11, 4 * n))
    axes = np.atleast_1d(axes)
    axes[0].imshow(cv2.cvtColor(manual, cv2.COLOR_BGR2RGB))
    axes[0].set_title(f"Manual pipeline  {manual.shape[1]}x{manual.shape[0]}")
    axes[0].axis("off")
    if auto is not None:
        axes[1].imshow(cv2.cvtColor(auto, cv2.COLOR_BGR2RGB))
        axes[1].set_title(f"cv2.Stitcher ({auto_mode})  {auto.shape[1]}x{auto.shape[0]}")
        axes[1].axis("off")
    fig.tight_layout()
    fig.savefig(str(out / "03_compare.png"), dpi=110)
    plt.close(fig)


def main() -> None:
    out = output_dir()
    views = make_three_views()

    # === 1. 手作りパイプライン ============================================
    print("[1] 手作りパイプライン（build_panorama）")
    t0 = time.perf_counter()
    manual, inliers = build_panorama(views)
    manual_dt = time.perf_counter() - t0
    cv2.imwrite(str(out / "03_manual.png"), manual)
    print(f"  size={manual.shape[1]}x{manual.shape[0]} / インライア={inliers} / {manual_dt:.2f}s")

    # === 2. cv2.Stitcher（自動） ==========================================
    print("\n[2] cv2.Stitcher（自動合成）")
    auto, mode, auto_dt = run_stitcher(views)
    if auto is not None:
        cv2.imwrite(str(out / "03_stitcher.png"), auto)
        print(f"  採用モード={mode} / size={auto.shape[1]}x{auto.shape[0]} / {auto_dt:.2f}s")
    else:
        print("  Stitcher は全モードで失敗（合成画像は手作り版のみ保存）")

    # === 3. 重なり領域 SSIM で結果を比較 ===================================
    print("\n[3] 2つの結果を位置合わせして重なり領域 SSIM を測定")
    if auto is not None:
        score = overlap_ssim(manual, auto)
        if score is None:
            print("  位置合わせに失敗 → SSIM 比較はスキップ")
        else:
            print(f"  重なり領域 SSIM = {score:.4f}（1 に近いほど両者の見えが一致）")
    else:
        print("  Stitcher 出力が無いため比較スキップ")

    # === 4. 並べて保存 ====================================================
    save_comparison(out, manual, auto, mode)
    print("\n[まとめ] 手作りは中身が見え学習向き。Stitcher は露出補正+マルチバンド合成まで自動。")
    print("完了。outputs/06_homography_panorama/ の 03_*.png を確認してください。")


if __name__ == "__main__":
    main()
