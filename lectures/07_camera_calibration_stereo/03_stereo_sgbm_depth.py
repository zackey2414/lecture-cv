"""03: ステレオ視差と深度 — StereoSGBM/StereoBM・stereoRectify(Q)・reprojectImageTo3D。

学ぶこと:
  - 平行化(rectify)済みステレオでは視差 d と深度 Z が d = f*baseline/Z で結ばれること。
  - StereoSGBM と StereoBM で視差マップを計算する正準形（戻り値は ×16 の固定小数点）。
  - stereoRectify が返す再投影行列 Q と、その手作り版が一致すること。
  - cv2.reprojectImageTo3D で視差マップを 3D 点群（X,Y,Z）に変換する。
  - findFundamentalMat と、平行化済みペアではエピポーラ線が水平になること。

このスクリプトは「実データ無し」で完走する: 奥行きの違う物体を並べた平行化済み
ステレオ対を合成し、各物体の正解深度と SGBM 推定を突き合わせて妥当性を検証する。

実行:
  uv run python lectures/07_camera_calibration_stereo/03_stereo_sgbm_depth.py
結果は outputs/07_camera_calibration_stereo/ に保存される（画面表示はしない）。
"""

from __future__ import annotations

import pathlib
import sys

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from cv_helpers import (  # noqa: E402
    IMAGE_SIZE,
    STEREO_BASELINE,
    STEREO_CX,
    STEREO_CY,
    STEREO_F,
    TRUE_K,
    make_stereo_pair,
    output_dir,
    reproject_matrix,
)

cv2.setRNGSeed(0)  # findFundamentalMat の RANSAC を再現可能にする。

NUM_DISP = 96  # 視差の探索幅。16 の倍数で、想定最大視差(64)を十分カバーする。
BLOCK = 5      # SGBM のマッチング窓。小さいほど細部に強いがノイズも増える。


def compute_disparities(left: np.ndarray, right: np.ndarray):
    """SGBM と BM で視差マップを計算する。戻り値は float（×16 を割って実視差に直す）。"""
    # --- StereoSGBM（準大域マッチング・高品質） ---------------------------
    sgbm = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=NUM_DISP,
        blockSize=BLOCK,
        P1=8 * 3 * BLOCK ** 2,    # 視差の滑らかさペナルティ（小さな段差）。
        P2=32 * 3 * BLOCK ** 2,   # 大きな段差のペナルティ（P1<P2 が定石）。
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=2,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )
    disp_sgbm = sgbm.compute(left, right).astype(np.float32) / 16.0

    # --- StereoBM（局所ブロックマッチング・高速・グレースケール専用） -----
    gl = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    gr = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
    bm = cv2.StereoBM_create(numDisparities=NUM_DISP, blockSize=15)
    disp_bm = bm.compute(gl, gr).astype(np.float32) / 16.0
    return disp_sgbm, disp_bm


def colorize_disparity(disp: np.ndarray) -> np.ndarray:
    """視差マップを 0-255 に正規化してカラーマップで色づけ（無効画素は黒）。"""
    valid = disp > 0
    vis = np.zeros(disp.shape, np.uint8)
    if valid.any():
        lo = float(disp[valid].min())
        hi = float(disp[valid].max())
        norm = np.clip((disp - lo) / max(hi - lo, 1e-6) * 255.0, 0, 255).astype(np.uint8)
        vis = np.where(valid, norm, 0).astype(np.uint8)
    color = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
    color[~valid] = 0  # 無効画素は黒にする。
    return color


def validate_depths(disp: np.ndarray, gt: list[dict]) -> None:
    """各物体の中心で視差→深度を測り、正解と比べる（評価）。"""
    print("\n[3] 視差→深度の妥当性（各物体の中心で検証）")
    print("  物体         正解視差 正解深度   推定視差 推定深度")
    for g in gt:
        cx = (g["x0"] + g["x1"]) // 2
        cy = (g["y0"] + g["y1"]) // 2
        patch = disp[cy - 10:cy + 10, cx - 10:cx + 10]
        good = patch[patch > 0]
        if good.size == 0:
            print(f"  {g['name']:11s} 視差が取れず（オクルージョン領域）")
            continue
        md = float(np.median(good))
        depth = STEREO_F * STEREO_BASELINE / md
        print(f"  {g['name']:11s} {g['disp']:7d} {g['depth']:8.3f}   "
              f"{md:7.1f} {depth:8.3f}")


def show_Q(out: pathlib.Path) -> np.ndarray:
    """stereoRectify で再投影行列 Q を得て、手作り Q と一致することを確認する。"""
    # 平行化済みなので回転 R=単位、並進 T=(-baseline,0,0)、歪み=0 を与える。
    D0 = np.zeros(5)
    R = np.eye(3)
    T = np.array([-STEREO_BASELINE, 0.0, 0.0])
    _, _, _, _, Q_sr, _, _ = cv2.stereoRectify(
        TRUE_K, D0, TRUE_K, D0, IMAGE_SIZE, R, T,
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=0,
    )
    Q_manual = reproject_matrix()
    print("\n[4] 再投影行列 Q（視差→3D）")
    print("  stereoRectify の Q と手作り Q が一致:", np.allclose(Q_sr, Q_manual, atol=1e-6))
    return Q_manual


def epipolar_demo(out: pathlib.Path, left: np.ndarray, right: np.ndarray) -> None:
    """findFundamentalMat → エピポーラ線を描く。平行化済みなら線は水平になる。"""
    gl = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    gr = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(2000)
    k1, d1 = orb.detectAndCompute(gl, None)
    k2, d2 = orb.detectAndCompute(gr, None)
    print("\n[5] エピポーラ幾何（findFundamentalMat）")
    if d1 is None or d2 is None or len(k1) < 8 or len(k2) < 8:
        print("  特徴点が足りず基本行列の推定をスキップ（exit は正常）")
        return
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(bf.match(d1, d2), key=lambda x: x.distance)[:200]
    if len(matches) < 8:
        print("  対応が足りず推定をスキップ（exit は正常）")
        return
    pts1 = np.float32([k1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([k2[m.trainIdx].pt for m in matches])
    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 1.0, 0.99)
    if F is None or mask is None:
        print("  基礎行列を推定できず（exit は正常）")
        return
    inl = mask.ravel().astype(bool)
    dy = float(np.abs(pts1[inl][:, 1] - pts2[inl][:, 1]).mean())
    print(f"  対応 {len(matches)} / インライア {int(mask.sum())}")
    print(f"  インライアの平均 |y_L - y_R| = {dy:.2f} px（≈0 なら平行化済み＝線は水平）")

    # 左の数点に対応する右画像のエピポーラ線を描く（水平になるはず）。
    sel = pts1[inl][:8].reshape(-1, 1, 2)
    lines = cv2.computeCorrespondEpilines(sel, 1, F).reshape(-1, 3)
    vis = right.copy()
    w = right.shape[1]
    rng = np.random.default_rng(0)
    for (a, b, c) in lines:
        if abs(b) < 1e-6:
            continue
        y0 = int(-c / b)             # x=0 での y
        y1 = int(-(c + a * w) / b)   # x=w での y
        col = tuple(int(v) for v in rng.integers(60, 255, size=3))
        cv2.line(vis, (0, y0), (w, y1), col, 1, cv2.LINE_AA)
    cv2.imwrite(str(out / "03_epipolar_lines.png"), vis)
    print("  → 03_epipolar_lines.png にエピポーラ線（ほぼ水平）を描画")


def main() -> None:
    out = output_dir()

    # === 1. 平行化済みステレオ対を合成 ====================================
    left, right, gt = make_stereo_pair()
    cv2.imwrite(str(out / "03_left.png"), left)
    cv2.imwrite(str(out / "03_right.png"), right)
    print("[1] 平行化済みステレオ対を合成 → 03_left.png / 03_right.png")
    print(f"  f={STEREO_F} px, baseline={STEREO_BASELINE} m, 主点=({STEREO_CX},{STEREO_CY})")

    # === 2. 視差マップ（SGBM / BM） ======================================
    disp_sgbm, disp_bm = compute_disparities(left, right)
    cv2.imwrite(str(out / "03_disparity_sgbm.png"), colorize_disparity(disp_sgbm))
    cv2.imwrite(str(out / "03_disparity_bm.png"), colorize_disparity(disp_bm))
    sgbm_valid = float((disp_sgbm > 0).mean())
    bm_valid = float((disp_bm > 0).mean())
    print("\n[2] 視差マップ（戻り値 ×16 を実視差に直す）")
    print(f"  SGBM 有効画素率 = {sgbm_valid:.0%}（準大域・高品質）")
    print(f"  BM   有効画素率 = {bm_valid:.0%}（局所・高速だが穴が多い）")

    # === 3. 視差→深度の検証 ==============================================
    validate_depths(disp_sgbm, gt)

    # === 4. Q 行列 → 5. エピポーラ =======================================
    Q = show_Q(out)

    # reprojectImageTo3D で 3D 点群にし、深度マップを保存。
    pts3d = cv2.reprojectImageTo3D(disp_sgbm, Q)
    Z = pts3d[:, :, 2]
    valid = (disp_sgbm > 0) & np.isfinite(Z) & (Z > 0)
    depth_vis = np.zeros(Z.shape, np.uint8)
    if valid.any():
        zc = np.clip(Z[valid], 0, 6.0)  # 0〜6m を 0-255 に。近いほど明るく。
        norm = (255 * (1 - zc / 6.0)).astype(np.uint8)
        depth_vis[valid] = norm
    cv2.imwrite(str(out / "03_depth.png"),
                cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO))
    print("  reprojectImageTo3D → 03_depth.png（近いほど明るい深度マップ）")

    epipolar_demo(out, left, right)

    # === 6. matplotlib で一覧（BGR→RGB） =================================
    def rgb(im: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    panels = [
        (rgb(left), "left"),
        (rgb(right), "right"),
        (rgb(colorize_disparity(disp_sgbm)), "disparity (SGBM)"),
        (rgb(cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)), "depth (near=bright)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (im, title) in zip(axes.ravel(), panels):
        ax.imshow(im)
        ax.set_title(title)
    for ax in axes.ravel():
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(str(out / "03_stereo_overview.png"), dpi=110)
    plt.close(fig)
    print("\n[6] → 03_stereo_overview.png に一覧を保存")
    print("\n完了。outputs/07_camera_calibration_stereo/ の 03_*.png を確認してください。")


if __name__ == "__main__":
    main()
