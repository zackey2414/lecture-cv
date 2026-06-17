"""02: 歪み補正 — undistort・getOptimalNewCameraMatrix(ROI)・remap 版。

学ぶこと:
  - 歪み係数 dist がレンズの「直線を曲げる」効果であること（格子画像で可視化）。
  - cv2.undistort の正準形。newCameraMatrix=K（既定）なら元の見え方に戻る。
  - getOptimalNewCameraMatrix の alpha と「有効領域 ROI」の意味（黒縁の扱い）。
  - initUndistortRectifyMap + remap という等価で高速（動画向き）な方法。
  - 補正後の格子線がまっすぐに戻り、元のクリーン画像とほぼ一致することを数値確認。

このスクリプトは「実データ無し」で完走する: 直線格子のクリーン画像を合成し、
既知の K・dist で歪みを加えた画像を作ってから補正する。K・dist は 01 が保存した
calib.npz があれば読み込み、無ければ cv_helpers の真値を使う（どちらでも動く）。

実行:
  uv run python lectures/07_camera_calibration_stereo/02_undistort.py
結果は lectures/07_camera_calibration_stereo/outputs/ に保存される（画面表示はしない）。
"""

from __future__ import annotations

import pathlib
import sys

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")  # headless でも確実に保存できる非対話バックエンド。
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from cv_helpers import (  # noqa: E402
    IMAGE_SIZE,
    TRUE_DIST,
    TRUE_K,
    make_calibration_grid,
    make_distorted_image,
    output_dir,
)


def load_camera(out: pathlib.Path) -> tuple[np.ndarray, np.ndarray, str]:
    """K と dist を入手する。01 の calib.npz があれば使い、無ければ真値を使う。"""
    npz = out / "calib.npz"
    if npz.exists():
        data = np.load(str(npz))
        return data["K"].astype(np.float64), data["dist"].astype(np.float64), "calib.npz(01の校正結果)"
    return TRUE_K.copy(), TRUE_DIST.copy(), "cv_helpers の真値"


def main() -> None:
    out = output_dir()
    W, H = IMAGE_SIZE

    print("[1] カメラ（K, dist）の入手")
    K, dist, source = load_camera(out)
    print(f"  使用元: {source}")
    print("  dist =", np.round(dist.ravel(), 4))

    # === 2. クリーン画像と歪み画像を合成 ===================================
    # 直線格子は、歪むと曲がるので歪みの効果が一目で分かる。
    clean = make_calibration_grid()
    distorted = make_distorted_image(clean, K, dist)
    cv2.imwrite(str(out / "02_clean.png"), clean)
    cv2.imwrite(str(out / "02_distorted.png"), distorted)
    print("\n[2] クリーン画像と歪み画像を合成 → 02_clean.png / 02_distorted.png")
    print("  歪み画像では外周ほど格子線が湾曲している（たる型歪み）。")

    # === 3. 正準の undistort（newCameraMatrix を省略＝K のまま） ===========
    # newCameraMatrix を省くと K がそのまま使われ、補正後は「歪みだけ消えた元の見え方」
    # に戻る。だから元のクリーン画像と画素単位で比較できる。
    undistorted = cv2.undistort(distorted, K, dist)
    cv2.imwrite(str(out / "02_undistorted.png"), undistorted)

    # 中央領域（両画像とも有効）の平均絶対差で「元に戻ったか」を数値確認。
    cy0, cy1, cx0, cx1 = 80, H - 80, 80, W - 80
    mad = float(cv2.absdiff(undistorted[cy0:cy1, cx0:cx1], clean[cy0:cy1, cx0:cx1]).mean())
    print("\n[3] undistort（K のまま）→ 02_undistorted.png")
    print(f"  補正後 vs 元クリーンの平均絶対差 = {mad:.2f}（0 に近いほど復元成功・255階調中）")
    print("  → 細い線の再標本化ぼけ以外はほぼ一致。歪みが消え直線が戻っている。")

    # === 4. getOptimalNewCameraMatrix と ROI（黒縁の扱い） ================
    # alpha=1: 元画素を全部残す（補正後に黒い余白が出る）。ROI が有効範囲。
    # alpha=0: 黒余白が出ないよう内側だけ残す（端は切り取られる）。
    newK_keep, roi_keep = cv2.getOptimalNewCameraMatrix(K, dist, (W, H), 1.0, (W, H))
    newK_crop, roi_crop = cv2.getOptimalNewCameraMatrix(K, dist, (W, H), 0.0, (W, H))
    und_keep = cv2.undistort(distorted, K, dist, None, newK_keep)
    und_crop = cv2.undistort(distorted, K, dist, None, newK_crop)
    print("\n[4] getOptimalNewCameraMatrix（alpha と ROI）")
    print(f"  alpha=1 の有効領域 ROI = {roi_keep}（この矩形の外は黒い無効領域）")
    print(f"  alpha=0 の有効領域 ROI = {roi_crop}（端を切って黒縁ゼロ）")

    # alpha=1 の結果に ROI 矩形を描いて「有効範囲」を見えるようにする。
    keep_vis = und_keep.copy()
    x, y, rw, rh = roi_keep
    if rw > 0 and rh > 0:
        cv2.rectangle(keep_vis, (x, y), (x + rw, y + rh), (0, 0, 255), 2)
    cv2.imwrite(str(out / "02_undistorted_keep_roi.png"), keep_vis)

    # alpha=0 の ROI で切り取った「使える範囲だけ」の画像も保存。
    x0, y0, rw0, rh0 = roi_crop
    und_crop_roi = und_crop[y0:y0 + rh0, x0:x0 + rw0] if rw0 > 0 and rh0 > 0 else und_crop
    cv2.imwrite(str(out / "02_undistorted_cropped.png"), und_crop_roi)

    # === 5. remap 版（initUndistortRectifyMap）— 動画では定石 ============
    # 補正マップを一度だけ作り、各フレームに remap を適用すれば毎回 undistort より速い。
    map1, map2 = cv2.initUndistortRectifyMap(K, dist, None, K, (W, H), cv2.CV_32FC1)
    und_remap = cv2.remap(distorted, map1, map2, cv2.INTER_LINEAR)
    maxdiff = int(cv2.absdiff(und_remap, undistorted).max())
    print("\n[5] initUndistortRectifyMap + remap（undistort と等価・動画向き）")
    print(f"  undistort との最大差 = {maxdiff}（マップの固定小数点丸めの差のみ。中身は同じ処理）")

    # === 6. matplotlib で 4 枚並べて保存（BGR→RGB に直す） ================
    def rgb(im: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    panels = [
        (rgb(clean), "clean (ideal, straight lines)"),
        (rgb(distorted), "distorted (barrel)"),
        (rgb(undistorted), "undistorted (newK=K)"),
        (rgb(keep_vis), "optimal alpha=1 + ROI(red)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, (im, title) in zip(axes.ravel(), panels):
        ax.imshow(im)
        ax.set_title(title)
    for ax in axes.ravel():
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(str(out / "02_undistort_compare.png"), dpi=110)
    plt.close(fig)
    print("\n[6] → 02_undistort_compare.png に 4 枚並べて保存")
    print("\n完了。lectures/07_camera_calibration_stereo/outputs/ の 02_*.png を確認してください。")


if __name__ == "__main__":
    main()
