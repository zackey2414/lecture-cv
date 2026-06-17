"""mini_project.py — 章末ミニプロジェクト: 合成動画の総合「動き解析レポート」。

この回の四本柱（背景差分／疎フロー LK／密フロー Farneback／色追跡 CamShift）を
1 本のパイプラインに統合する完成形。合成した連番フレーム列に対して、次を一気に走らせる:

  ① 背景差分(MOG2): 各フレームの前景マスク→形態学→外接矩形で「動体検出」する。
  ② 疎フロー(LK)  : goodFeaturesToTrack で選んだ角を全フレーム追跡し、軌跡を描く。
  ③ 密フロー(FB)  : 既知シフトのペアで EPE を測り、シーンのフローを HSV で可視化する。
  ④ 色追跡(CamShift): 緑円を色ヒストグラムで追い、窓サイズが成長することを数値化する。
  ⑤ 統合可視化   : 各フレームに「動体枠＋LK点＋追跡窓」を重ねた動画を書き出す。
  ⑥ 数値レポート : 検出数・追跡点数・EPE・窓成長率・処理FPS を JSON に保存する。

外部データもネットも GPU もカメラも不要（合成のみ・CPU 完結・headless 安全）。
digit 始まりのスクリプト(01〜04)は import できないため、各アルゴリズムの核心は
この中に自己完結で実装する（合成データ生成と可視化部品だけ cv_helpers から借りる）。
実運用では make_motion_frames() を実際の VideoCapture ループに差し替えれば、
そのまま「固定カメラの動体検出＋追跡」のひな形になる。

実行:
    uv run python lectures/10_classical_video_motion/mini_project.py
結果（動画・画像・JSON）は lectures/10_classical_video_motion/outputs/ に保存される（画面表示はしない）。
"""

from __future__ import annotations

import collections
import json
import pathlib
import sys
import time

import cv2
import numpy as np

# headless 環境でも図を保存できるよう、pyplot より前に Agg を指定する。
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# 同じフォルダの cv_helpers.py（合成データ生成・可視化部品）だけを import する。
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from cv_helpers import (  # noqa: E402
    endpoint_error,
    flow_to_bgr,
    green_window,
    make_motion_frames,
    make_translating_pair,
    output_dir,
)

# --- 共通パラメータ（本文・各スクリプトと揃える） -----------------------------
KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
FEATURE_PARAMS = dict(maxCorners=200, qualityLevel=0.05, minDistance=7, blockSize=7)
LK_PARAMS = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
)
FB_PARAMS = dict(pyr_scale=0.5, levels=3, winsize=15, iterations=3,
                 poly_n=5, poly_sigma=1.2, flags=0)
TERM_CRIT = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)


# =====================================================================
# 部品（単一責務で小さく分ける。digit スクリプトに依存せず自己完結）
# =====================================================================

def detect_foreground_boxes(
    subtractor, frame: np.ndarray, min_area: int = 150
) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    """背景差分器に1フレーム与え、前景マスクと動体の外接矩形を返す。

    subtractor.apply は内部で背景モデルを更新しつつ前景マスクを返す。
    影(127)を閾値で落とし、形態学で整え、面積の小さい輪郭をノイズとして捨てる。
    findContours は OpenCV 4 系では (contours, hierarchy) の2つ返し。
    """
    raw = subtractor.apply(frame)
    _, fg = cv2.threshold(raw, 200, 255, cv2.THRESH_BINARY)  # 影(127)を 0 に
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, KERNEL)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, KERNEL)
    contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) >= min_area]
    return fg, boxes


def build_hue_hist(frame: np.ndarray, window: tuple[int, int, int, int]) -> np.ndarray:
    """初期窓の領域から、色相(H)ヒストグラム＝色モデルを作って正規化して返す。

    S/V が低い画素は H が当てにならないので inRange で除外してから集計する。
    """
    x, y, w, h = window
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    roi = hsv[y:y + h, x:x + w]
    roi_mask = cv2.inRange(roi, (0, 60, 60), (180, 255, 255))
    hist = cv2.calcHist([roi], [0], roi_mask, [16], [0, 180])
    cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
    return hist


def back_project(frame: np.ndarray, hist: np.ndarray) -> np.ndarray:
    """フレームを『対象の色らしさ』の確率マップ（uint8, 0-255）に変換する。"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    return cv2.calcBackProject([hsv], [0], hist, [0, 180], scale=1)


def open_writer(path: pathlib.Path, size: tuple[int, int], fps: int = 20):
    """mp4v コーデックで VideoWriter を開く。開けなければ None を返す（呼び出し側で連番保存に退避）。"""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, size)
    return writer if writer.isOpened() else None


# =====================================================================
# メイン: 全手法を1ループに統合し、動画・図・JSON を出力する
# =====================================================================

def main() -> None:
    out = output_dir()
    frames = make_motion_frames()
    n = len(frames)
    h, w = frames[0].shape[:2]

    print("=" * 64)
    print("章末ミニプロジェクト: 合成動画の総合 動き解析レポート")
    print(f"  フレーム数={n} / サイズ={w}x{h} / 全手法を1ループに統合")

    # --- 各手法の初期化 -----------------------------------------------------
    mog2 = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=25,
                                              detectShadows=True)
    prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(prev_gray, mask=None, **FEATURE_PARAMS)
    n_init_points = 0 if p0 is None else len(p0)
    trace = frames[0].copy()  # LK 軌跡を描き続ける土台

    cam_win = green_window(frames[0])  # 緑円の初期窓
    init_win = cam_win
    hist = build_hue_hist(frames[0], cam_win)
    first_area = last_area = float(cam_win[2] * cam_win[3])

    # --- 出力動画（mp4v）。開けなければ連番 PNG にフォールバック --------------
    writer = open_writer(out / "mini_project_overlay.mp4", (w, h), fps=20)
    video_ok = writer is not None

    fps_window: collections.deque[float] = collections.deque(maxlen=30)
    fps_log: list[float] = []
    max_boxes = 0
    last_boxes = 0
    last_fg = np.zeros((h, w), np.uint8)
    cam_snapshot = frames[0].copy()
    key_indices = {0, n // 2, n - 1}

    # --- メインループ: 1フレームで「検出＋疎フロー＋色追跡」を全部やる ---------
    for i, frame in enumerate(frames):
        t0 = time.perf_counter()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        overlay = frame.copy()

        # ① 背景差分で動体の外接矩形（緑枠）。
        last_fg, boxes = detect_foreground_boxes(mog2, frame)
        max_boxes = max(max_boxes, len(boxes))
        last_boxes = len(boxes)
        for (bx, by, bw, bh) in boxes:
            cv2.rectangle(overlay, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)

        # ② 疎フロー(LK): 前フレームの点を追跡し、軌跡を描き足す（青点＝現在位置）。
        if p0 is not None and len(p0) > 0:
            p1, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None,
                                                     **LK_PARAMS)
            if p1 is not None:
                good_new = p1[status.ravel() == 1]
                good_old = p0[status.ravel() == 1]
                for new, old in zip(good_new, good_old):
                    x1, y1 = new.ravel()
                    x0, y0 = old.ravel()
                    cv2.line(trace, (int(x1), int(y1)), (int(x0), int(y0)),
                             (0, 0, 255), 2)
                    cv2.circle(overlay, (int(x1), int(y1)), 3, (255, 60, 0), -1)
                p0 = good_new.reshape(-1, 1, 2)

        # ③ 色追跡(CamShift): 確率マップ上で窓のサイズ・回転を対象に合わせる（黄枠）。
        prob = back_project(frame, hist)
        rot_rect, cam_win = cv2.CamShift(prob, cam_win, TERM_CRIT)
        cw, ch = rot_rect[1]
        area = float(cw * ch)
        if i == 0:
            first_area = area
        last_area = area
        pts = cv2.boxPoints(rot_rect).astype(np.int32)
        cv2.polylines(overlay, [pts], True, (0, 255, 255), 2)

        # FPS 計測（処理パイプライン自体の速さ。ソース FPS とは別物）。
        dt = time.perf_counter() - t0
        fps_window.append(dt)
        cur_fps = 1.0 / (sum(fps_window) / len(fps_window) + 1e-9)
        fps_log.append(cur_fps)
        cv2.putText(overlay, f"frame {i:02d}  ~{cur_fps:5.1f} FPS",
                    (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                    cv2.LINE_AA)

        # 出力（動画 or 連番 PNG）。要所フレームは別途 PNG でも残す。
        if video_ok:
            writer.write(overlay)
        else:
            cv2.imwrite(str(out / f"mini_project_overlay_{i:03d}.png"), overlay)
        if i in key_indices:
            cv2.imwrite(str(out / f"mini_project_overlay_key{i:03d}.png"), overlay)
            cam_snapshot = overlay

        prev_gray = gray

    if video_ok:
        writer.release()

    n_tracked = 0 if p0 is None else len(p0)
    grow_ratio = last_area / first_area if first_area > 0 else 0.0

    # --- 密フロー(Farneback): 既知シフトで EPE、シーンで HSV 可視化 ----------
    shift = (6, 3)
    a, b, true_shift = make_translating_pair(shift=shift)
    flow_pair = cv2.calcOpticalFlowFarneback(
        cv2.cvtColor(a, cv2.COLOR_BGR2GRAY),
        cv2.cvtColor(b, cv2.COLOR_BGR2GRAY), None, **FB_PARAMS)
    gt = np.zeros_like(flow_pair)
    gt[..., 0], gt[..., 1] = true_shift
    epe_mean = float(endpoint_error(flow_pair, gt).mean())

    flow_scene = cv2.calcOpticalFlowFarneback(
        cv2.cvtColor(frames[18], cv2.COLOR_BGR2GRAY),
        cv2.cvtColor(frames[19], cv2.COLOR_BGR2GRAY), None, **FB_PARAMS)
    flow_hsv = flow_to_bgr(flow_scene)
    scene_max_mag = float(np.linalg.norm(flow_scene, axis=2).max())
    cv2.imwrite(str(out / "mini_project_flow_hsv.png"), flow_hsv)

    # --- まとめ図（2x2, matplotlib・Agg・BGR→RGB に注意） --------------------
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].imshow(last_fg, cmap="gray")
    axes[0, 0].set_title("MOG2 foreground (last frame)")
    axes[0, 1].imshow(cv2.cvtColor(trace, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title("LK sparse tracks")
    axes[1, 0].imshow(cv2.cvtColor(flow_hsv, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title("Farneback dense flow (HSV)")
    axes[1, 1].imshow(cv2.cvtColor(cam_snapshot, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title("overlay: boxes + LK points + CamShift")
    for ax in axes.ravel():
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(str(out / "mini_project_summary.png"), dpi=110)
    plt.close(fig)

    # --- 数値レポート（再現性・自動評価のため JSON） ------------------------
    metrics = {
        "frames": n,
        "size": [w, h],
        "bgsub_mog2": {"last_boxes": last_boxes, "max_boxes": max_boxes},
        "lk_sparse": {"initial_points": n_init_points, "tracked_to_end": n_tracked},
        "farneback_dense": {
            "known_shift": list(true_shift),
            "epe_mean_px": round(epe_mean, 4),
            "scene_max_mag_px": round(scene_max_mag, 3),
        },
        "camshift": {
            "init_window": [int(v) for v in init_win],
            "first_area": round(first_area, 1),
            "last_area": round(last_area, 1),
            "grow_ratio": round(grow_ratio, 3),
        },
        "processing_fps": {
            "mean": round(float(np.mean(fps_log)), 1),
            "min": round(float(np.min(fps_log)), 1),
        },
        "video_written": video_ok,
    }
    json_path = out / "mini_project_metrics.json"
    json_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2),
                         encoding="utf-8")

    # --- コンソール要約 -----------------------------------------------------
    print("\n  [結果サマリ]")
    print(f"  ① 背景差分 MOG2 : 最終フレーム動体={last_boxes} 個 / 最大={max_boxes} 個")
    print(f"  ② 疎フロー  LK  : 初期点={n_init_points} → 最後まで追跡={n_tracked} 点")
    print(f"  ③ 密フロー  FB  : 既知シフト{true_shift} の EPE 平均={epe_mean:.4f}px "
          f"/ シーン最大動き={scene_max_mag:.2f}px")
    print(f"  ④ 色追跡 CamShift: 窓面積 {first_area:.0f}→{last_area:.0f}px^2 "
          f"(成長率 {grow_ratio:.2f}x)")
    print(f"  処理FPS（パイプライン）: 平均 {metrics['processing_fps']['mean']} "
          f"/ 最小 {metrics['processing_fps']['min']}")
    print("\n  保存物:")
    if video_ok:
        print("    - 統合動画: mini_project_overlay.mp4（動体枠＋LK点＋CamShift窓）")
    else:
        print("    - 統合連番: mini_project_overlay_000.png ...（VideoWriter 不可のため）")
    print("    - 要所フレーム: mini_project_overlay_key*.png")
    print("    - 密フロー   : mini_project_flow_hsv.png")
    print("    - まとめ図   : mini_project_summary.png")
    print(f"    - 指標 JSON  : {json_path.name}")
    print("\n完成: 背景差分→疎フロー→密フロー→色追跡 の動き解析一式を外部依存ゼロで回せた。")
    print("発展: make_motion_frames() を実カメラの VideoCapture ループに替えれば実運用へ。")


if __name__ == "__main__":
    main()
