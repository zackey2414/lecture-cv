"""mini_project.py — 章末ミニプロジェクト: 校正 → 歪み補正 → ステレオ深度の一気通貫パイプライン。

この回の学び（カメラモデル・チェスボード校正・undistort・StereoSGBM・視差→深度）を
**1 本の幾何パイプライン**に統合する「完成形」。外部データもネットも GPU も不要で、
すべて cv_helpers の合成データ（チェスボード・歪み格子・平行化ステレオ対）から組む。

  Stage A 校正    : 合成チェスボードを 15 視点撮り、findChessboardCorners → cornerSubPix
                    → calibrateCamera で K, dist を推定。RMS と真値ずれを評価する。
  Stage B 歪み補正: 真値カメラ(TRUE_K/TRUE_DIST)で歪ませた「実写相当」画像を、
                    **Stage A で校正した K, dist** で undistort し、元のまっすぐな格子に
                    戻るか（中央領域の平均絶対差）で end-to-end の校正品質を測る。
  Stage C ステレオ: 平行化済みステレオ対から StereoSGBM で視差を計算し、Z=f*baseline/d
                    で深度に変換。各物体の正解深度との平均絶対誤差(MAE)で検証する。
  Stage D 3D計測  : reprojectImageTo3D の 3D 点群を使い、最前面物体の「メートル幅」を
                    画素幅×Z/f で実寸推定する（画素→3D の鎖を最後まで通す）。

各ステージは PASS/FAIL を自己判定し、最後に総合判定・図・JSON を出力する。
digit 始まりの 01〜03 スクリプトは import できないので、必要な処理は本ファイル内に
（cv_helpers の合成・校正部品だけ借りて）自己完結で書く。

実行:
    uv run python lectures/07_camera_calibration_stereo/mini_project.py
結果（画像・JSON）は lectures/07_camera_calibration_stereo/outputs/ に保存される（画面表示はしない）。
"""

from __future__ import annotations

import json
import pathlib
import sys

import cv2
import numpy as np

# headless 環境でも図を保存できるよう、pyplot より前に Agg を指定する。
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# 同じフォルダの cv_helpers.py を確実に import できるようにする（どこから実行してもぶれない）。
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from cv_helpers import (  # noqa: E402
    IMAGE_SIZE,
    PATTERN_SIZE,
    STEREO_BASELINE,
    STEREO_CX,
    STEREO_CY,
    STEREO_F,
    TRUE_DIST,
    TRUE_K,
    calibrate,
    chessboard_object_points,
    detect_corners,
    make_calibration_grid,
    make_camera_poses,
    make_distorted_image,
    make_stereo_pair,
    mean_reprojection_error,
    output_dir,
    render_chessboard_view,
    reproject_matrix,
)

N_VIEWS = 15   # 校正に使う合成視点数（実務でも 10〜20 枚が目安）。
NUM_DISP = 96  # 視差探索幅（16 の倍数で、想定最大視差 64 を十分カバー）。
BLOCK = 5      # SGBM のマッチング窓。


# =====================================================================
# Stage A: チェスボード校正（K, dist を推定）
# =====================================================================

def stage_calibrate(out: pathlib.Path) -> dict:
    """合成チェスボード 15 視点から K, dist を推定し、品質を評価する。"""
    objp = chessboard_object_points()
    poses = make_camera_poses(N_VIEWS)

    objpoints: list[np.ndarray] = []
    imgpoints: list[np.ndarray] = []
    thumbs: list[np.ndarray] = []
    for rvec, tvec in poses:
        img = render_chessboard_view(TRUE_K, TRUE_DIST, rvec, tvec)
        found, corners = detect_corners(img)
        if not found:
            continue  # 検出失敗視点は握りつぶさず捨てて続行。
        objpoints.append(objp)
        imgpoints.append(corners)
        vis = img.copy()
        cv2.drawChessboardCorners(vis, PATTERN_SIZE, corners, found)
        thumbs.append(cv2.resize(vis, (200, 150)))

    # 検出モンタージュ（最初の数視点）。
    sample_vis = thumbs[0] if thumbs else np.full((150, 200, 3), 255, np.uint8)

    rms, K, dist, rvecs, tvecs = calibrate(objpoints, imgpoints, IMAGE_SIZE)
    my_rms = mean_reprojection_error(objpoints, imgpoints, rvecs, tvecs, K, dist)

    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    # K の主要パラメータが真値に十分近いか（px 単位の許容）。
    k_ok = (
        abs(fx - TRUE_K[0, 0]) < 5.0 and abs(fy - TRUE_K[1, 1]) < 5.0
        and abs(cx - TRUE_K[0, 2]) < 5.0 and abs(cy - TRUE_K[1, 2]) < 5.0
    )
    rms_ok = rms < 1.0
    consistent = bool(np.isclose(my_rms, rms, atol=1e-3))
    passed = bool(k_ok and rms_ok and consistent)

    print("=" * 64)
    print("Stage A: チェスボード校正（findChessboardCorners→calibrateCamera）")
    print(f"  検出成功 {len(objpoints)}/{N_VIEWS} 視点")
    print(f"  RMS 再投影誤差 = {rms:.4f} px（自前 {my_rms:.4f} px・一致={consistent}）")
    print(f"  fx={fx:.2f} fy={fy:.2f} cx={cx:.2f} cy={cy:.2f}（真値 600/600/320/240）")
    print(f"  判定: {'PASS' if passed else 'FAIL'}（RMS<1px・K が真値±5px・自前RMS一致）")

    metrics = {
        "n_views_used": len(objpoints),
        "rms_px": round(rms, 4),
        "rms_self_px": round(my_rms, 4),
        "rms_consistent": consistent,
        "fx": round(fx, 3), "fy": round(fy, 3),
        "cx": round(cx, 3), "cy": round(cy, 3),
        "dist": [round(float(v), 5) for v in dist.ravel()],
        "passed": passed,
    }
    return {"K": K, "dist": dist, "sample_vis": sample_vis, "metrics": metrics}


# =====================================================================
# Stage B: 歪み補正（校正した K, dist で実写相当の歪みを戻す）
# =====================================================================

def stage_undistort(out: pathlib.Path, K: np.ndarray, dist: np.ndarray) -> dict:
    """真値カメラで歪ませた画像を、校正済み K, dist で補正して end-to-end 品質を測る。"""
    clean = make_calibration_grid()
    # 「実写相当」= 真のレンズ(TRUE_K/TRUE_DIST)で歪んだ画像。
    distorted = make_distorted_image(clean, TRUE_K, TRUE_DIST)
    # 校正で“当てた” K, dist で補正する（end-to-end の校正品質テスト）。
    undist = cv2.undistort(distorted, K, dist)

    W, H = IMAGE_SIZE
    cy0, cy1, cx0, cx1 = 80, H - 80, 80, W - 80  # 両画像とも有効な中央領域。
    mad = float(cv2.absdiff(undist[cy0:cy1, cx0:cx1], clean[cy0:cy1, cx0:cx1]).mean())
    passed = mad < 8.0  # 細線の再標本化ぼけを許容しても十分小さいはず。

    cv2.imwrite(str(out / "mini_project_distorted.png"), distorted)
    cv2.imwrite(str(out / "mini_project_undistorted.png"), undist)

    print("\nStage B: 歪み補正（校正した K, dist で実写相当の歪みを戻す）")
    print(f"  補正後 vs 元クリーンの中央 MAD = {mad:.2f}（0 に近いほど良・255 階調中）")
    print(f"  判定: {'PASS' if passed else 'FAIL'}（MAD<8 なら直線が直線に戻った）")

    metrics = {"center_mad": round(mad, 3), "passed": passed}
    return {"clean": clean, "distorted": distorted, "undistorted": undist, "metrics": metrics}


# =====================================================================
# Stage C: ステレオ深度（StereoSGBM → 視差 → 深度 → MAE）
# =====================================================================

def compute_sgbm_disparity(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """StereoSGBM で視差マップを計算する（戻り値 ×16 を割って実視差に直す）。自己完結。"""
    sgbm = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=NUM_DISP,
        blockSize=BLOCK,
        P1=8 * 3 * BLOCK ** 2,    # 小さな視差変化のペナルティ。
        P2=32 * 3 * BLOCK ** 2,   # 大きな段差のペナルティ（P1<P2 が定石）。
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=2,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )
    return sgbm.compute(left, right).astype(np.float32) / 16.0


def colorize_disparity(disp: np.ndarray) -> np.ndarray:
    """視差マップを 0-255 に正規化してカラーマップで色づけ（無効画素は黒）。"""
    valid = disp > 0
    vis = np.zeros(disp.shape, np.uint8)
    if valid.any():
        lo, hi = float(disp[valid].min()), float(disp[valid].max())
        norm = np.clip((disp - lo) / max(hi - lo, 1e-6) * 255.0, 0, 255).astype(np.uint8)
        vis = np.where(valid, norm, 0).astype(np.uint8)
    color = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
    color[~valid] = 0
    return color


def stage_stereo(out: pathlib.Path) -> dict:
    """平行化済みステレオ対から深度を復元し、各物体の正解深度と突き合わせる。"""
    left, right, gt = make_stereo_pair()
    disp = compute_sgbm_disparity(left, right)

    # 各物体の中心で視差→深度を測り、正解と比較。
    rows = []
    abs_errors = []
    for g in gt:
        cx = (g["x0"] + g["x1"]) // 2
        cy = (g["y0"] + g["y1"]) // 2
        patch = disp[cy - 10:cy + 10, cx - 10:cx + 10]
        good = patch[patch > 0]
        if good.size == 0:
            rows.append({"name": g["name"], "gt_depth": round(g["depth"], 4),
                         "est_depth": None, "abs_err": None})
            continue
        md = float(np.median(good))
        depth = STEREO_F * STEREO_BASELINE / md
        err = abs(depth - g["depth"])
        abs_errors.append(err)
        rows.append({"name": g["name"], "gt_disp": g["disp"], "est_disp": round(md, 1),
                     "gt_depth": round(g["depth"], 4), "est_depth": round(depth, 4),
                     "abs_err": round(err, 4)})

    mae = float(np.mean(abs_errors)) if abs_errors else float("inf")
    passed = mae < 0.05  # 5cm 未満なら合成シーンでは良好。

    cv2.imwrite(str(out / "mini_project_left.png"), left)
    cv2.imwrite(str(out / "mini_project_right.png"), right)
    cv2.imwrite(str(out / "mini_project_disparity.png"), colorize_disparity(disp))

    print("\nStage C: ステレオ深度（StereoSGBM → Z=f*baseline/d）")
    print("  物体         正解深度 推定深度  絶対誤差[m]")
    for r in rows:
        if r["est_depth"] is None:
            print(f"  {r['name']:11s} {r['gt_depth']:8.4f} （視差取れず）")
        else:
            print(f"  {r['name']:11s} {r['gt_depth']:8.4f} {r['est_depth']:8.4f}  {r['abs_err']:8.4f}")
    print(f"  深度 MAE = {mae:.4f} m")
    print(f"  判定: {'PASS' if passed else 'FAIL'}（MAE<0.05m）")

    metrics = {"depth_mae_m": round(mae, 4), "objects": rows, "passed": passed}
    return {"left": left, "right": right, "disp": disp, "gt": gt, "metrics": metrics}


# =====================================================================
# Stage D: 3D 計測（reprojectImageTo3D → 最前面物体のメートル幅）
# =====================================================================

def stage_measure(out: pathlib.Path, disp: np.ndarray, gt: list[dict]) -> dict:
    """最前面物体の実寸幅(m)を「画素幅 × Z / f」で推定する（画素→3D の鎖の総仕上げ）。"""
    Q = reproject_matrix()
    pts3d = cv2.reprojectImageTo3D(disp.astype(np.float32), Q)

    # gt のうち最も手前（視差最大）の物体を選ぶ（background は除く）。
    objs = [g for g in gt if g["name"] != "background"]
    front = max(objs, key=lambda g: g["disp"])
    cx = (front["x0"] + front["x1"]) // 2
    cy = (front["y0"] + front["y1"]) // 2

    # 物体中心の 3D 点（X,Y,Z）と、画素幅から実寸幅を逆算する。
    Z = float(pts3d[cy, cx, 2])
    px_w = front["x1"] - front["x0"]
    metric_w = px_w * Z / STEREO_F  # x = (u-cx)*Z/f を端どうしで引いた幅。
    gt_metric_w = px_w * front["depth"] / STEREO_F  # 正解深度から計算した参照値。
    err = abs(metric_w - gt_metric_w)
    passed = err < 0.02  # 2cm 未満なら良好。

    print("\nStage D: 3D 計測（reprojectImageTo3D → 最前面物体のメートル幅）")
    print(f"  対象={front['name']} 画素幅={px_w}px 中心Z={Z:.4f}m")
    print(f"  推定メートル幅 = {metric_w:.4f} m（正解深度からの参照値 {gt_metric_w:.4f} m）")
    print(f"  判定: {'PASS' if passed else 'FAIL'}（参照値との差<0.02m）")

    metrics = {
        "target": front["name"], "pixel_width": int(px_w),
        "center_Z_m": round(Z, 4), "metric_width_m": round(metric_w, 4),
        "ref_metric_width_m": round(gt_metric_w, 4),
        "passed": passed,
    }
    return {"metrics": metrics}


# =====================================================================
# メイン
# =====================================================================

def save_summary_figure(out: pathlib.Path, a: dict, b: dict, c: dict) -> None:
    """6 枚（校正・歪み・補正・クリーン・視差・深度）を 1 枚にまとめて保存する。"""
    def rgb(im: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    # 深度マップ（近いほど明るい）。
    Q = reproject_matrix()
    pts3d = cv2.reprojectImageTo3D(c["disp"].astype(np.float32), Q)
    Zc = pts3d[:, :, 2]
    valid = (c["disp"] > 0) & np.isfinite(Zc) & (Zc > 0)
    depth_vis = np.zeros(Zc.shape, np.uint8)
    if valid.any():
        zc = np.clip(Zc[valid], 0, 6.0)
        depth_vis[valid] = (255 * (1 - zc / 6.0)).astype(np.uint8)
    depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)

    panels = [
        (rgb(a["sample_vis"]), "A: chessboard + corners"),
        (rgb(b["distorted"]), "B: distorted (real lens)"),
        (rgb(b["undistorted"]), "B: undistorted (calibrated K)"),
        (rgb(c["left"]), "C: stereo left"),
        (rgb(colorize_disparity(c["disp"])), "C: disparity (SGBM)"),
        (rgb(depth_color), "D: depth (near=bright)"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, (im, title) in zip(axes.ravel(), panels):
        ax.imshow(im)
        ax.set_title(title)
    for ax in axes.ravel():
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(str(out / "mini_project_summary.png"), dpi=110)
    plt.close(fig)


def main() -> None:
    out = output_dir()
    print("章末ミニプロジェクト: 校正 → 歪み補正 → ステレオ深度の一気通貫")

    a = stage_calibrate(out)
    b = stage_undistort(out, a["K"], a["dist"])
    c = stage_stereo(out)
    d = stage_measure(out, c["disp"], c["gt"])

    save_summary_figure(out, a, b, c)

    # 総合判定と指標を JSON 化（再現性・自動評価のため）。
    overall = bool(
        a["metrics"]["passed"] and b["metrics"]["passed"]
        and c["metrics"]["passed"] and d["metrics"]["passed"]
    )
    metrics = {
        "stage_a_calibrate": a["metrics"],
        "stage_b_undistort": b["metrics"],
        "stage_c_stereo": c["metrics"],
        "stage_d_measure": d["metrics"],
        "overall_pass": overall,
    }
    json_path = out / "mini_project_metrics.json"
    json_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 64)
    print(f"総合判定: {'ALL PASS' if overall else '一部 FAIL（上のログを確認）'}")
    print("  保存物:")
    print("    - まとめ図  : mini_project_summary.png")
    print("    - 中間画像  : mini_project_distorted/undistorted/left/right/disparity.png")
    print(f"    - 指標 JSON : {json_path.name}")
    print("\n完成: K の推定 → 歪み補正 → 視差→深度 → 実寸計測、を外部依存ゼロで通せた。")
    print("発展: 合成チェスボードを実写に、合成ステレオを実 2 眼カメラに差し替えれば実運用へ。")


if __name__ == "__main__":
    main()
