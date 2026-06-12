"""use_case.py — 実践ユースケース: 1 枚の写真から実寸(mm)を測る「平面メジャー」ツール。

このモジュールの学び（カメラ校正・歪み補正・平面射影=ホモグラフィ・solvePnP）を、
現実に役立つ **1 台のカメラ × 既知サイズの基準物** だけで成立する小ツールに落とす。

  やりたいこと:
    机や紙の上に「印刷したチェスボード（マス幅が既知=スケール基準）」を一緒に置いて
    1 枚だけ撮る。すると盤の各角は『画像画素』と『盤平面上の実寸(mm)座標』の対応が付く。
    平面どうしの対応は **ホモグラフィ H** で結べるので、H を使えば画像上の任意の 2 点を
    盤平面の mm 座標へ写して、その間の **実寸距離(mm)** を測れる。さらに校正済み K があれば
    solvePnP で **カメラ〜対象面の距離(おおよその奥行き)** も出せる。
    （家具・部品・傷・葉っぱのサイズを「定規を当てずに写真から測る」現実の道具そのもの。）

  mini_project.py との違い:
    - mini_project は **2 眼(ステレオ)** で StereoSGBM の視差から 3D 復元する“奥行き計測”。
    - 本ツールは **単眼 + 平面拘束(ホモグラフィ)** で“平面上の実寸計測”。2 台もステレオも要らず、
      基準物のマス幅(mm)さえ分かれば 1 枚で mm が出る。intrinsics(K) は歪み補正と距離推定に
      使うだけで、mm 計測そのものは K 無しでも成立する（平面ホモグラフィの強み）。

  実データの置き方（あれば優先・無ければ合成で必ず完走）:
    data/07_camera_calibration_stereo/ に計測したい写真を 1 枚置く（例 measure.jpg / *.png）。
    写真には「マス幅が既知のチェスボード」を被写体と同一平面に写し込むこと。
      環境変数で基準を指定できる:
        CHESSBOARD_SQUARE_MM=25     # 盤 1 マスの実寸(mm)。既定 25.0
        CHESSBOARD_COLS=9 ROWS=6    # 内側角の数(マス数ではない)。既定 9x6
        CALIB_NPZ=/path/calib.npz   # 自前カメラの K,dist があれば歪み補正＋距離推定に使う
    データが無ければ、基準ボード＋既知サイズの対象物を合成した 1 枚を作って計測する
    （正解 mm が分かるので、計測値の誤差まで検証できる）。

  拡張アイデア（ここを出発点に練習）:
    - MEASURE_GUI=1（かつ DISPLAY 有り）で Tkinter のクリック計測 GUI を起動。
      2 点クリック→その実寸(mm)を表示。headless では自動で予定点の計測にフォールバック。
    - 基準を ArUco/ChArUco マーカに変えて一部が隠れても基準を取れるようにする。
    - 盤の代わりに A4 用紙(210x297mm)の四隅を基準点にして findHomography する。
    - 鳥瞰図(bird's-eye view)を warpPerspective で作り、画素↔mm を一定スケールにする。
    - 複数枚を平均して計測ノイズを下げる／物体検出と組んで自動採寸する。

  実行:
    uv run python lectures/07_camera_calibration_stereo/use_case.py
  出力（画面表示はしない・headless 安全）は outputs/07_camera_calibration_stereo/ に保存。
"""

from __future__ import annotations

import json
import os
import pathlib
import sys

import cv2
import numpy as np

# headless でも図を保存できるよう、pyplot より前に Agg を指定する。
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# 同じフォルダの cv_helpers.py を確実に import する（どこから実行してもぶれない）。
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from cv_helpers import (  # noqa: E402
    IMAGE_SIZE,
    PATTERN_SIZE,
    PROJECT_ROOT,
    TRUE_DIST,
    TRUE_K,
    calibrate,
    chessboard_object_points,
    detect_corners,
    make_camera_poses,
    output_dir,
    render_chessboard_view,
)

MODULE_ID = "07_camera_calibration_stereo"
DATA_DIR = PROJECT_ROOT / "data" / MODULE_ID

# 合成シーンの「基準 1 マス = 何 mm か」。校正で得る K はマスの絶対サイズに依存しない
# （= 画素幾何だけで決まる）ので、校正は単位スケール(square=1.0)で行い、mm はここで与える。
SQUARE_MM = float(os.environ.get("CHESSBOARD_SQUARE_MM", "25.0"))


# =====================================================================
# 1) カメラ（K, dist）を用意する: calib.npz があれば使い、無ければ合成校正
# =====================================================================

def load_or_calibrate_camera() -> tuple[np.ndarray | None, np.ndarray | None, str]:
    """歪み補正・距離推定に使う K, dist を用意する。

    実運用では「一度カメラを校正 → 何度も計測に使い回す」流れ。ここでは
      (a) CALIB_NPZ もしくは outputs/.../calib.npz（01_calibrate_camera.py の保存物）があれば読む。
      (b) 無ければ合成チェスボードを多視点で撮って calibrateCamera で即席に校正する。
    どちらも失敗したら (None, None) を返し、mm 計測はホモグラフィのみで続行する。
    """
    # (a) 既存の校正結果を探す。
    candidates = []
    env_npz = os.environ.get("CALIB_NPZ")
    if env_npz:
        candidates.append(pathlib.Path(env_npz))
    candidates.append(output_dir() / "calib.npz")
    for npz in candidates:
        if npz.exists():
            try:
                data = np.load(str(npz))
                K = data["K"].astype(np.float64)
                dist = data["dist"].astype(np.float64)
                return K, dist, f"calib.npz ({npz.name})"
            except Exception as exc:  # 壊れた npz は無視して次へ。
                print(f"  [calib] {npz} を読めなかったのでスキップ: {exc}")

    # (b) 合成チェスボードで即席に校正する（単位スケールで十分: K は絶対サイズに非依存）。
    objp = chessboard_object_points(PATTERN_SIZE, 1.0)
    poses = make_camera_poses(12, PATTERN_SIZE, 1.0)
    objpoints: list[np.ndarray] = []
    imgpoints: list[np.ndarray] = []
    for rvec, tvec in poses:
        img = render_chessboard_view(TRUE_K, TRUE_DIST, rvec, tvec, PATTERN_SIZE, 1.0)
        found, corners = detect_corners(img)
        if found:
            objpoints.append(objp)
            imgpoints.append(corners)
    if len(objpoints) < 4:
        return None, None, "uncalibrated"
    rms, K, dist, _, _ = calibrate(objpoints, imgpoints, IMAGE_SIZE)
    return K, dist, f"synthetic calibration (RMS={rms:.3f}px, {len(objpoints)} views)"


# =====================================================================
# 2) 計測写真を用意する: data/ にあれば優先、無ければ合成シーンを作る
# =====================================================================

def find_real_image() -> pathlib.Path | None:
    """data/<module_id>/ から計測対象の写真を 1 枚探す（無ければ None）。"""
    if not DATA_DIR.exists():
        return None
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    for p in sorted(DATA_DIR.iterdir()):
        if p.suffix.lower() in exts:
            return p
    return None


# 合成シーンの幾何（すべて「単位 = SQUARE_MM mm」で定義）。
# 盤の内側角は x:0..8, y:0..5。盤の下に既知サイズの対象物を同一平面(Z=0)に置く。
OBJ_UNIT = np.array(
    [[2.0, 7.0, 0.0],   # 左上 TL
     [6.0, 7.0, 0.0],   # 右上 TR
     [6.0, 9.0, 0.0],   # 右下 BR
     [2.0, 9.0, 0.0]],  # 左下 BL
    dtype=np.float32,
)
SCENE_RVEC = np.array([[0.20], [-0.16], [0.05]], dtype=np.float64)  # 斜め見下ろし。
SCENE_CENTER = np.array([4.0, 4.0, 0.0])  # 盤＋対象の概ねの中心（単位）。
SCENE_Z = 15.0  # この距離(単位)なら盤(約10単位)が白縁付きで 640x480 に収まる。


def scene_pose() -> tuple[np.ndarray, np.ndarray]:
    """合成シーンの真の姿勢 (rvec, tvec)。中心が光軸上 Z に来るよう平行移動を逆算。"""
    R, _ = cv2.Rodrigues(SCENE_RVEC)
    t = np.array([0.0, 0.0, SCENE_Z]) - R @ SCENE_CENTER
    return SCENE_RVEC, t.reshape(3, 1)


def make_synthetic_scene() -> tuple[np.ndarray, dict]:
    """基準チェスボード＋既知サイズ対象物を 1 枚に合成（歪み込み）し、正解情報も返す。

    盤は cv_helpers.render_chessboard_view で（白い静音域付きで）描き、対象物は同一姿勢・
    同一 K,dist で projectPoints して塗る。歪んだ写真なので、計測前に undistort が効く。
    """
    rvec, tvec = scene_pose()
    # 盤を描画（単位スケール）。歪みは TRUE_DIST。
    img = render_chessboard_view(TRUE_K, TRUE_DIST, rvec, tvec, PATTERN_SIZE, 1.0)

    # 対象物（既知サイズの長方形）を同一平面に塗る。辺を細分して歪み曲線に沿わせる。
    ring = []
    for a, b in [(0, 1), (1, 2), (2, 3), (3, 0)]:
        p0, p1 = OBJ_UNIT[a], OBJ_UNIT[b]
        for t in np.linspace(0.0, 1.0, 8, endpoint=False):
            ring.append(p0 + (p1 - p0) * t)
    ring_w = np.array(ring, dtype=np.float32)
    proj, _ = cv2.projectPoints(ring_w, rvec, tvec, TRUE_K, TRUE_DIST)
    poly = proj.reshape(-1, 2).astype(np.int32)
    cv2.fillPoly(img, [poly], (70, 140, 210), lineType=cv2.LINE_AA)  # 暖色の板。
    cv2.polylines(img, [poly], True, (40, 90, 150), 2, cv2.LINE_AA)

    info = {
        "kind": "synthetic",
        "square_mm": SQUARE_MM,
        "pattern": PATTERN_SIZE,
        "rvec": rvec,
        "tvec": tvec,
        "obj_unit": OBJ_UNIT,
        # 正解の実寸(mm): 幅= TL-TR=4単位, 高さ= TL-BL=2単位。
        "gt_width_mm": 4.0 * SQUARE_MM,
        "gt_height_mm": 2.0 * SQUARE_MM,
    }
    return img, info


# =====================================================================
# 3) 計測の中核: 画像画素 → 盤平面 mm のホモグラフィと距離計算
# =====================================================================

def to_world_mm(H: np.ndarray, pts_px) -> np.ndarray:
    """画像画素 (x,y) の並びを、ホモグラフィ H で盤平面 mm 座標へ写す。"""
    pts = np.asarray(pts_px, dtype=np.float32).reshape(-1, 1, 2)
    world = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
    return world


def segment_mm(H: np.ndarray, p1, p2) -> float:
    """画像上の 2 点 p1,p2 の「盤平面上の実寸距離(mm)」を返す。"""
    w = to_world_mm(H, [p1, p2])
    return float(np.linalg.norm(w[0] - w[1]))


def draw_label_segment(img: np.ndarray, p1, p2, mm: float, color) -> None:
    """計測線と mm ラベルを描く（ラベルは ASCII のみ）。"""
    p1 = (int(round(p1[0])), int(round(p1[1])))
    p2 = (int(round(p2[0])), int(round(p2[1])))
    cv2.line(img, p1, p2, color, 2, cv2.LINE_AA)
    for c in (p1, p2):
        cv2.circle(img, c, 4, color, -1, cv2.LINE_AA)
    mid = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2 - 8)
    text = f"{mm:.1f} mm"
    cv2.putText(img, text, mid, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, mid, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)


# =====================================================================
# 4) 任意: Tkinter のクリック計測 GUI（headless では自動スキップ）
# =====================================================================

def interactive_measure(photo_bgr: np.ndarray, H: np.ndarray) -> bool:
    """MEASURE_GUI=1 かつ DISPLAY 有りのときだけ、2 点クリック計測の窓を出す。

    headless（DISPLAY 無し・Tk 不可）では何もせず False を返す（自動計測にフォールバック）。
    """
    if os.environ.get("MEASURE_GUI") != "1":
        return False
    if not os.environ.get("DISPLAY"):
        print("  [GUI] DISPLAY が無いため対話計測はスキップ（headless フォールバック）。")
        return False
    try:
        import tkinter as tk

        from PIL import Image, ImageTk
    except Exception as exc:  # Tk/Pillow が無い環境。
        print(f"  [GUI] Tkinter/Pillow が使えないためスキップ: {exc}")
        return False

    try:
        rgb = cv2.cvtColor(photo_bgr, cv2.COLOR_BGR2RGB)
        root = tk.Tk()
        root.title("planar mm ruler - click two points (q to quit)")
        pil = Image.fromarray(rgb)
        tkimg = ImageTk.PhotoImage(pil)
        canvas = tk.Canvas(root, width=pil.width, height=pil.height)
        canvas.pack()
        canvas.create_image(0, 0, anchor="nw", image=tkimg)
        pts: list[tuple[int, int]] = []

        def on_click(ev) -> None:
            pts.append((ev.x, ev.y))
            canvas.create_oval(ev.x - 3, ev.y - 3, ev.x + 3, ev.y + 3, outline="red", width=2)
            if len(pts) == 2:
                mm = segment_mm(H, pts[0], pts[1])
                canvas.create_line(pts[0][0], pts[0][1], pts[1][0], pts[1][1], fill="lime", width=2)
                tx = (pts[0][0] + pts[1][0]) // 2
                ty = (pts[0][1] + pts[1][1]) // 2 - 10
                canvas.create_text(tx, ty, text=f"{mm:.1f} mm", fill="yellow",
                                   font=("Arial", 14, "bold"))
                print(f"  [GUI] 計測: {mm:.1f} mm")
                pts.clear()

        canvas.bind("<Button-1>", on_click)
        root.bind("q", lambda _e: root.destroy())
        root.mainloop()
        return True
    except Exception as exc:  # GUI 中の例外でも全体は止めない。
        print(f"  [GUI] 対話計測中にエラー（スキップ）: {exc}")
        return False


# =====================================================================
# メイン処理
# =====================================================================

def run() -> dict:
    out = output_dir()
    print("=" * 64)
    print("実践ユースケース: 1 枚写真の平面メジャー（基準チェスボードで mm 計測）")

    K, dist, cam_src = load_or_calibrate_camera()
    print(f"  カメラ: {cam_src}")

    # --- 計測写真を用意（実データ優先） ---
    real = find_real_image()
    if real is not None:
        photo = cv2.imread(str(real))
        if photo is None:
            print(f"  [warn] {real} を読めなかったので合成シーンに切替。")
            photo, info = make_synthetic_scene()
        else:
            if photo.ndim == 2:
                photo = cv2.cvtColor(photo, cv2.COLOR_GRAY2BGR)
            cols = int(os.environ.get("CHESSBOARD_COLS", PATTERN_SIZE[0]))
            rows = int(os.environ.get("CHESSBOARD_ROWS", PATTERN_SIZE[1]))
            info = {"kind": "real", "path": str(real), "square_mm": SQUARE_MM,
                    "pattern": (cols, rows)}
            print(f"  入力: 実データ {real.name}（盤 {cols}x{rows}・マス {SQUARE_MM}mm）")
    else:
        photo, info = make_synthetic_scene()
        print(f"  入力: 合成シーン（盤 {PATTERN_SIZE[0]}x{PATTERN_SIZE[1]}・"
              f"マス {SQUARE_MM}mm・既知サイズの対象物つき）")

    pattern = info["pattern"]
    square_mm = info["square_mm"]

    # --- 歪み補正（K があれば。実データで自前 K が無いと正確には合わない点に注意） ---
    if K is not None and dist is not None:
        photo_u = cv2.undistort(photo, K, dist)
        undistorted = True
    else:
        photo_u = photo.copy()
        undistorted = False
        print("  [info] K 不明のため undistort は省略（ホモグラフィ mm 計測は続行可能）。")

    # --- 基準チェスボードを検出 → 画素↔mm のホモグラフィ ---
    found, corners = detect_corners(photo_u, pattern)
    if not found:
        print("  [warn] チェスボードを検出できませんでした。")
        print("         実写では『盤が画面内に収まり・マス数(内側角)が正しく・"
              "十分なコントラスト』を確認してください。")
        cv2.imwrite(str(out / "use_case_input.png"), photo_u)
        metrics = {"camera_source": cam_src, "undistorted": undistorted,
                   "board_detected": False, "passed": False}
        (out / "use_case_metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        return metrics

    # 盤内側角の mm 座標（Z=0 平面）。検出順は chessboard_object_points と一致する。
    objp_unit = chessboard_object_points(pattern, 1.0)
    objp_mm_xy = (objp_unit[:, :2] * square_mm).astype(np.float32)
    # findHomography(src=画素, dst=mm) → 画像→盤平面 mm の射影 H。
    H, _mask = cv2.findHomography(
        corners.reshape(-1, 2), objp_mm_xy, cv2.RANSAC, 3.0
    )
    if H is None:
        print("  [warn] ホモグラフィ推定に失敗しました。")
        metrics = {"camera_source": cam_src, "board_detected": True,
                   "homography": False, "passed": False}
        (out / "use_case_metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        return metrics

    vis = photo_u.copy()
    cv2.drawChessboardCorners(vis, pattern, corners, found)

    # --- 自己検証: 盤の対角線(既知 mm)を H 経由で測る（どんな写真でも効くサニティ） ---
    cpx = corners.reshape(-1, 2)
    diag_gt = float(np.linalg.norm(objp_mm_xy[0] - objp_mm_xy[-1]))
    diag_meas = segment_mm(H, cpx[0], cpx[-1])
    diag_err = abs(diag_meas - diag_gt)
    draw_label_segment(vis, cpx[0], cpx[-1], diag_meas, (0, 200, 255))
    print(f"  盤の対角線: 計測 {diag_meas:.1f} mm / 既知 {diag_gt:.1f} mm "
          f"(誤差 {diag_err:.2f} mm)")

    metrics: dict = {
        "camera_source": cam_src,
        "undistorted": undistorted,
        "square_mm": square_mm,
        "pattern": list(pattern),
        "board_detected": True,
        "board_diagonal_mm": round(diag_meas, 2),
        "board_diagonal_gt_mm": round(diag_gt, 2),
        "board_diagonal_err_mm": round(diag_err, 3),
    }

    # --- 合成シーンなら、既知サイズ対象物の幅・高さを計測して誤差検証 ---
    if info["kind"] == "synthetic":
        rvec, tvec = info["rvec"], info["tvec"]
        # 対象物 4 隅を「歪んだ写真の画素」へ投影 → undistort して『ユーザが押す点』を作る。
        obj_d, _ = cv2.projectPoints(info["obj_unit"], rvec, tvec, TRUE_K, TRUE_DIST)
        if K is not None and dist is not None:
            clicks = cv2.undistortPoints(obj_d.reshape(-1, 1, 2), K, dist, P=K).reshape(-1, 2)
        else:
            clicks = obj_d.reshape(-1, 2)
        w_meas = segment_mm(H, clicks[0], clicks[1])   # TL-TR = 幅
        h_meas = segment_mm(H, clicks[0], clicks[3])   # TL-BL = 高さ
        w_err = abs(w_meas - info["gt_width_mm"])
        h_err = abs(h_meas - info["gt_height_mm"])
        draw_label_segment(vis, clicks[0], clicks[1], w_meas, (60, 220, 60))
        draw_label_segment(vis, clicks[0], clicks[3], h_meas, (220, 120, 60))
        print(f"  対象物: 幅 計測 {w_meas:.1f}mm / 正解 {info['gt_width_mm']:.1f}mm "
              f"(誤差 {w_err:.2f}mm)")
        print(f"          高 計測 {h_meas:.1f}mm / 正解 {info['gt_height_mm']:.1f}mm "
              f"(誤差 {h_err:.2f}mm)")
        metrics.update({
            "object_width_mm": round(w_meas, 2),
            "object_width_gt_mm": round(info["gt_width_mm"], 2),
            "object_width_err_mm": round(w_err, 3),
            "object_height_mm": round(h_meas, 2),
            "object_height_gt_mm": round(info["gt_height_mm"], 2),
            "object_height_err_mm": round(h_err, 3),
        })

        # --- おおよその奥行き: solvePnP でカメラ〜盤面の距離(mm) ---
        if K is not None and dist is not None:
            objp_mm3 = np.hstack([objp_mm_xy, np.zeros((len(objp_mm_xy), 1), np.float32)])
            ok, _rv, tv = cv2.solvePnP(objp_mm3, corners, K, np.zeros(5))
            if ok:
                dist_est = float(np.linalg.norm(tv))
                dist_true = float(np.linalg.norm(tvec)) * square_mm  # 真値(単位)→mm
                d_err_pct = abs(dist_est - dist_true) / dist_true * 100.0
                print(f"  カメラ距離: 推定 {dist_est:.0f}mm (~{dist_est / 10:.1f}cm) / "
                      f"真値 {dist_true:.0f}mm (誤差 {d_err_pct:.1f}%)")
                metrics.update({
                    "camera_distance_mm": round(dist_est, 1),
                    "camera_distance_gt_mm": round(dist_true, 1),
                    "camera_distance_err_pct": round(d_err_pct, 2),
                })

        # 合成は正解があるので合否判定する（実写は正解物体が無いので判定しない）。
        passed = bool(w_err < 3.0 and h_err < 3.0 and diag_err < 3.0)
        metrics["passed"] = passed
        print(f"  判定: {'PASS' if passed else 'FAIL'}"
              "（対象物 幅/高さ・盤対角の誤差 < 3mm）")
    else:
        metrics["passed"] = bool(diag_err < 3.0)
        print("  実データのため対象物の正解は無し（盤対角のサニティのみ）。"
              "MEASURE_GUI=1 で 2 点クリック計測ができます。")

    # --- 保存 ---
    cv2.imwrite(str(out / "use_case_input.png"), photo)
    cv2.imwrite(str(out / "use_case_measured.png"), vis)
    save_summary_figure(out, photo, vis, undistorted)
    (out / "use_case_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    # --- 任意の対話計測（既定では何もしない） ---
    interactive_measure(vis, H)

    print("  保存:")
    print("    - use_case_input.png    入力写真（歪みあり）")
    print("    - use_case_measured.png 検出＋計測線（mm ラベル付き）")
    print("    - use_case_summary.png  入力 vs 計測のまとめ図")
    print("    - use_case_metrics.json 計測値と誤差のログ")
    return metrics


def save_summary_figure(out: pathlib.Path, photo: np.ndarray, vis: np.ndarray,
                        undistorted: bool) -> None:
    """入力写真と計測結果を 2 枚並べて保存（ASCII タイトル）。"""
    def rgb(im: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].imshow(rgb(photo))
    axes[0].set_title("input photo (lens distortion)")
    title2 = "undistorted + measured (mm)" if undistorted else "measured (mm)"
    axes[1].imshow(rgb(vis))
    axes[1].set_title(title2)
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(str(out / "use_case_summary.png"), dpi=110)
    plt.close(fig)


def main() -> None:
    metrics = run()
    print("\n完了: 1 枚の写真 + 既知サイズの基準物だけで実寸(mm)を測れた。")
    print("発展: 基準を ArUco/A4 に変える・鳥瞰図化・MEASURE_GUI=1 でクリック計測、を試そう。")
    # 計測が成立していれば 0、検出失敗などでも例外は出さず 0 で終える（headless 安全）。
    _ = metrics


if __name__ == "__main__":
    main()
