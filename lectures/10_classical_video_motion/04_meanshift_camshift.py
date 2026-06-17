"""04: meanshift と camshift — 色ヒストグラムで物体を追いかける。

学ぶこと:
  - 追跡対象の色ヒストグラム（HSV の色相）を1回だけ作る（= 物体の色モデル）。
  - calcBackProject で各フレームを『その色らしさ』の確率マップに変換する。
  - meanShift: 固定サイズの窓を確率の山の頂上へ滑らせて追跡（窓サイズは不変）。
  - CamShift: meanshift を毎回回した上で窓のサイズ・回転まで対象に合わせる（拡縮・回転対応）。
  - フロー(02,03)が『動き』を追うのに対し、こちらは『色（見た目）』で追う対比を理解する。

実行:
  uv run python lectures/10_classical_video_motion/04_meanshift_camshift.py
結果は lectures/10_classical_video_motion/outputs/ に保存される（画面表示はしない）。
"""

from __future__ import annotations

import pathlib
import sys

import cv2
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from cv_helpers import green_window, make_motion_frames, output_dir  # noqa: E402

# 追跡の反復停止条件（10回まわす or 中心移動が1px 未満になったら止める）。
TERM_CRIT = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)


def build_hue_histogram(frame: np.ndarray, window: tuple[int, int, int, int]
                        ) -> np.ndarray:
    """初期窓の領域から、色相(H)のヒストグラム＝色モデルを作って正規化して返す。

    S(彩度)/V(明度)が低い画素はノイズになりやすいので inRange で除外してから集計する。
    """
    x, y, w, h = window
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    roi = hsv[y:y + h, x:x + w]
    # 暗すぎ・薄すぎる画素を捨てる（H が当てにならないため）。
    roi_mask = cv2.inRange(roi, (0, 60, 60), (180, 255, 255))
    hist = cv2.calcHist([roi], [0], roi_mask, [16], [0, 180])
    cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
    return hist


def back_project(frame: np.ndarray, hist: np.ndarray) -> np.ndarray:
    """フレームを『対象の色らしさ』の確率マップ（uint8, 0-255）に変換する。"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    return cv2.calcBackProject([hsv], [0], hist, [0, 180], scale=1)


def run_meanshift(frames: list[np.ndarray], hist: np.ndarray,
                  init_win: tuple[int, int, int, int], out: pathlib.Path) -> None:
    """固定サイズ窓の meanShift で全フレームを追跡し、軌跡と要所を保存する。"""
    win = init_win
    trace = frames[0].copy()
    prev_center = None
    snapshots = []
    for i, f in enumerate(frames):
        prob = back_project(f, hist)
        # ★ 核心: 確率マップ上で窓を密度の高い方へ動かす。窓サイズは変わらない。
        _, win = cv2.meanShift(prob, win, TERM_CRIT)
        x, y, w, h = win
        center = (x + w // 2, y + h // 2)
        if prev_center is not None:
            cv2.line(trace, prev_center, center, (0, 0, 255), 2)
        prev_center = center
        if i in (0, len(frames) // 2, len(frames) - 1):
            snap = f.copy()
            cv2.rectangle(snap, (x, y), (x + w, y + h), (0, 0, 255), 2)
            snapshots.append(snap)

    print("[1] meanShift（固定サイズ窓で色を追跡）")
    print(f"  初期窓 {init_win} → 最終窓 {tuple(int(v) for v in win)}")
    print("  窓のサイズは最後まで一定。対象が拡大しても枠は大きくならない（弱点）。")
    cv2.imwrite(str(out / "04_meanshift_trace.png"), trace)
    cv2.imwrite(str(out / "04_meanshift_snapshots.png"), cv2.hconcat(snapshots))


def run_camshift(frames: list[np.ndarray], hist: np.ndarray,
                 init_win: tuple[int, int, int, int], out: pathlib.Path) -> None:
    """CamShift で全フレームを追跡し、回転矩形（サイズ追従）を描いて保存する。"""
    win = init_win
    snapshots = []
    first_size = last_size = None
    for i, f in enumerate(frames):
        prob = back_project(f, hist)
        # ★ 核心: CamShift は meanshift を回した後、窓を対象の広がり・傾きへ合わせ直す。
        #   返り値 rot_rect = ((cx,cy),(w,h),angle) は回転を含む矩形。
        rot_rect, win = cv2.CamShift(prob, win, TERM_CRIT)
        size = rot_rect[1]
        if i == 0:
            first_size = size
        last_size = size
        if i in (0, len(frames) // 2, len(frames) - 1):
            snap = f.copy()
            pts = cv2.boxPoints(rot_rect).astype(np.int32)  # 回転矩形の四隅
            cv2.polylines(snap, [pts], True, (0, 255, 255), 2)
            snapshots.append(snap)

    print("\n[2] CamShift（窓のサイズ・回転も対象に合わせる）")
    print(f"  最初の窓サイズ ≈ {tuple(round(v, 1) for v in first_size)}")
    print(f"  最後の窓サイズ ≈ {tuple(round(v, 1) for v in last_size)}")
    print("  → 円が大きくなるにつれ枠も大きくなる。これが meanShift との決定的な差。")
    cv2.imwrite(str(out / "04_camshift_snapshots.png"), cv2.hconcat(snapshots))


def main() -> None:
    out = output_dir()
    frames = make_motion_frames()

    # 1枚目で追跡対象（緑の円）の初期窓を決め、その色ヒストグラムを作る。
    init_win = green_window(frames[0])
    hist = build_hue_histogram(frames[0], init_win)

    # 参考として、確率マップ（back projection）が物体だけ白く光ることを保存して見せる。
    cv2.imwrite(str(out / "04_backproject.png"), back_project(frames[20], hist))

    run_meanshift(frames, hist, init_win, out)
    run_camshift(frames, hist, init_win, out)

    print("\n[3] フロー追跡との違い")
    print("  - 02/03 のフロー: 連続フレーム間の『動き』から追う（色が同じでも動けば追える）。")
    print("  - meanshift/camshift: 『色（見た目）』で追う。似た色が背景にあると引っ張られる。")
    print("\n完了。lectures/10_classical_video_motion/outputs/ の 04_*.png を確認してください。")


if __name__ == "__main__":
    main()
