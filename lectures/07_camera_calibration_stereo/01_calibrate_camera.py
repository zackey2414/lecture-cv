"""01: カメラキャリブレーション — findChessboardCorners → cornerSubPix → calibrateCamera。

学ぶこと:
  - 内部行列 K（fx,fy,cx,cy）と歪み係数 dist が「カメラの何を表すか」。
  - チェスボードを複数視点から撮り、内側角を検出して 3D-2D 対応を作る正準手順。
  - calibrateCamera の戻り値 RMS 再投影誤差(px)で校正品質を測る。
  - 同じ誤差を projectPoints で自前計算し、ライブラリ値と一致することを確認する。
  - 高次の歪み係数 k3 は固定するのが定石、という実務知。

このスクリプトは「実データ無し」で完走する: チェスボード画像は既知の真値カメラ
（cv_helpers.TRUE_K / TRUE_DIST）で合成生成し、それを「当てに行く」。
コーナー検出に失敗した視点はスキップし、視点が足りなくても結果を保存して終了する。

実行:
  uv run python lectures/07_camera_calibration_stereo/01_calibrate_camera.py
結果は lectures/07_camera_calibration_stereo/outputs/ に保存される（画面表示はしない）。
"""

from __future__ import annotations

import pathlib
import sys

import cv2
import numpy as np

# 同じフォルダの cv_helpers.py を import できるようにする（どこから実行してもぶれない）。
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from cv_helpers import (  # noqa: E402
    IMAGE_SIZE,
    PATTERN_SIZE,
    TRUE_DIST,
    TRUE_K,
    calibrate,
    chessboard_object_points,
    detect_corners,
    make_camera_poses,
    mean_reprojection_error,
    output_dir,
    render_chessboard_view,
)

N_VIEWS = 15  # 校正用に合成する視点数。実務でも 10〜20 枚が目安。


def collect_corners(out: pathlib.Path):
    """N_VIEWS 視点のチェスボードを合成し、コーナーを検出して 3D-2D 対応を集める。"""
    objp = chessboard_object_points()  # 盤の内側角の 3D 座標（Z=0 平面）。
    poses = make_camera_poses(N_VIEWS)

    objpoints: list[np.ndarray] = []   # 各視点の 3D 点（毎回同じ objp）。
    imgpoints: list[np.ndarray] = []   # 各視点で検出した 2D 角。
    thumbs: list[np.ndarray] = []      # 検出結果のサムネイル（モンタージュ用）。

    print("[1] チェスボードの合成とコーナー検出")
    for idx, (rvec, tvec) in enumerate(poses):
        # 真値カメラで「歪みを含む」チェスボード写真を合成する。
        img = render_chessboard_view(TRUE_K, TRUE_DIST, rvec, tvec)
        found, corners = detect_corners(img)
        if not found:
            # 検出に失敗しても落とさず、その視点だけ捨てて続行する。
            print(f"  view {idx:02d}: コーナー検出に失敗 → スキップ")
            continue
        objpoints.append(objp)
        imgpoints.append(corners)
        # 検出した角を描いたサムネイル（後で 1 枚に並べて目視確認できる）。
        vis = img.copy()
        cv2.drawChessboardCorners(vis, PATTERN_SIZE, corners, found)
        thumbs.append(cv2.resize(vis, (256, 192)))

    print(f"  検出成功 {len(objpoints)}/{N_VIEWS} 視点")

    # 最初の数視点を 1 枚のモンタージュにして保存（検出の様子が一目で分かる）。
    if thumbs:
        cols = 4
        rows = (len(thumbs) + cols - 1) // cols
        canvas = np.full((rows * 192, cols * 256, 3), 255, np.uint8)
        for k, th in enumerate(thumbs):
            r, c = divmod(k, cols)
            canvas[r * 192:(r + 1) * 192, c * 256:(c + 1) * 256] = th
        cv2.imwrite(str(out / "01_detected_corners.png"), canvas)
        print("  → 01_detected_corners.png に検出結果のモンタージュを保存")

    return objpoints, imgpoints


def run_calibration(out: pathlib.Path, objpoints, imgpoints) -> None:
    """集めた対応から calibrateCamera を回し、品質と真値との一致を確認する。"""
    if len(objpoints) < 3:
        # 校正には最低でも数視点が要る。足りなければ保存だけして正常終了。
        print("\n[2] 有効視点が少なすぎて校正をスキップ（exit は正常）")
        return

    # === 2. 校正本体（k3 は固定＝非魚眼の定石） ============================
    rms, K, dist, rvecs, tvecs = calibrate(objpoints, imgpoints, IMAGE_SIZE)
    print("\n[2] calibrateCamera")
    print(f"  RMS 再投影誤差 = {rms:.4f} px（小さいほど良い。1px 未満なら良好）")

    # === 3. 自前の再投影誤差が RMS と一致するか確認 =======================
    my_rms = mean_reprojection_error(objpoints, imgpoints, rvecs, tvecs, K, dist)
    print("\n[3] 再投影誤差を自前で計算（projectPoints）")
    print(f"  自前 RMS = {my_rms:.4f} px  ／ ライブラリ RMS = {rms:.4f} px")
    print(f"  一致: {np.isclose(my_rms, rms, atol=1e-3)}  ← 定義どおりなら一致する")

    # === 4. 推定 K・dist を真値と比較 =====================================
    print("\n[4] 推定パラメータ vs 真値")
    print(f"  fx: {K[0, 0]:7.2f} (真値 {TRUE_K[0, 0]:.1f})")
    print(f"  fy: {K[1, 1]:7.2f} (真値 {TRUE_K[1, 1]:.1f})")
    print(f"  cx: {K[0, 2]:7.2f} (真値 {TRUE_K[0, 2]:.1f})")
    print(f"  cy: {K[1, 2]:7.2f} (真値 {TRUE_K[1, 2]:.1f})")
    print(f"  dist 推定 = {np.round(dist.ravel(), 4)}")
    print(f"  dist 真値 = {TRUE_DIST}")
    print("  → fx,fy,cx,cy と k1 はよく一致する。k2 は盤が画像隅まで届かないため")
    print("    やや不定（高次ほど不定）。だから k3 は固定するのが実務の定石。")

    # 校正結果を保存（02_undistort.py が読み込んで使える）。
    np.savez(str(out / "calib.npz"), K=K, dist=dist, image_size=np.array(IMAGE_SIZE))
    print("\n  → calib.npz に K と dist を保存（02_undistort.py が利用）")


def show_k3_instability(out: pathlib.Path, objpoints, imgpoints) -> None:
    """おまけ: k3 を固定しないと高次係数が暴れることを実験で示す。"""
    if len(objpoints) < 3:
        return
    print("\n[5] k3 を固定する理由（高次歪み係数の不定性）")
    rms_fix, _, dist_fix, _, _ = cv2.calibrateCamera(
        objpoints, imgpoints, IMAGE_SIZE, None, None, flags=cv2.CALIB_FIX_K3
    )
    rms_full, _, dist_full, _, _ = cv2.calibrateCamera(
        objpoints, imgpoints, IMAGE_SIZE, None, None, flags=0
    )
    print(f"  k3 固定 : RMS={rms_fix:.4f}  k3={dist_fix.ravel()[4]:.4f}")
    print(f"  k3 自由 : RMS={rms_full:.4f}  k3={dist_full.ravel()[4]:.4f}")
    print("  → RMS はほぼ同じなのに k3 だけ大きく変わる＝k3 は不定。固定が無難。")


def main() -> None:
    out = output_dir()
    objpoints, imgpoints = collect_corners(out)
    run_calibration(out, objpoints, imgpoints)
    show_k3_instability(out, objpoints, imgpoints)
    print("\n完了。lectures/07_camera_calibration_stereo/outputs/ の 01_*.png と calib.npz を確認してください。")


if __name__ == "__main__":
    main()
