"""02: Lucas-Kanade 疎オプティカルフロー — 少数の特徴点だけを追いかける。

学ぶこと:
  - goodFeaturesToTrack で『追跡しやすい角（コーナー）』を選ぶ（Shi-Tomasi コーナー）。
  - calcOpticalFlowPyrLK でその点が次フレームのどこへ動いたかを推定する正準形。
  - 返り値 status（追えたか 0/1）と err（誤差）で、見失った点を必ず捨てること。
  - 軌跡（トラック）を描いて、動く物体の点は長く・静止背景の点は短いことを目で見る。
  - 古典フローの前提「輝度一定」と弱点「開口問題（aperture problem）」に触れる。

実行:
  uv run python lectures/10_classical_video_motion/02_optical_flow_lk.py
結果は lectures/10_classical_video_motion/outputs/ に保存される（画面表示はしない）。
"""

from __future__ import annotations

import pathlib
import sys

import cv2
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from cv_helpers import make_motion_frames, output_dir  # noqa: E402

# Shi-Tomasi コーナー検出のパラメータ（追跡の起点にする良い点を選ぶ）。
FEATURE_PARAMS = dict(maxCorners=200, qualityLevel=0.05, minDistance=7, blockSize=7)

# LK のパラメータ。winSize=探索窓、maxLevel=ピラミッド段数（大きい動きに強くなる）、
# criteria=反復停止条件（回数 or 変化量）。
LK_PARAMS = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
)


def track_sequence(frames: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, int]:
    """連番フレームを通して特徴点を LK で追跡し、軌跡を重ねた画像を返す。

    返り値: (軌跡を描いた画像, 最終フレームに点を描いた画像, 追跡し続けた点の数)。
    """
    gray_prev = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    # 最初のフレームで追跡の起点になる角を選ぶ。形は (N, 1, 2) float32。
    p0 = cv2.goodFeaturesToTrack(gray_prev, mask=None, **FEATURE_PARAMS)

    # 軌跡を描き込むキャンバス（最初のフレームを土台にする）。
    trace = frames[0].copy()
    rng = np.random.default_rng(0)
    colors = rng.integers(0, 255, (len(p0), 3)).tolist()  # 点ごとに色を変える

    for i in range(1, len(frames)):
        gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        # ★ 核心: 前フレームの点 p0 が、今のフレームでどこへ動いたか p1 を求める。
        p1, status, err = cv2.calcOpticalFlowPyrLK(
            gray_prev, gray, p0, None, **LK_PARAMS
        )
        if p1 is None:
            break
        # status==1（追えた点）だけ残す。見失った点を使い続けると軌跡が暴れる。
        good_new = p1[status.ravel() == 1]
        good_old = p0[status.ravel() == 1]
        keep_colors = [colors[j] for j in range(len(status)) if status.ravel()[j] == 1]

        # 旧位置→新位置の線分を引いて軌跡を伸ばす。
        for (new, old, col) in zip(good_new, good_old, keep_colors):
            x1, y1 = new.ravel()
            x0, y0 = old.ravel()
            cv2.line(trace, (int(x1), int(y1)), (int(x0), int(y0)), col, 2)

        # 次の反復へ: 今フレームを「前」に、生き残った点を次の起点に更新する。
        gray_prev = gray
        p0 = good_new.reshape(-1, 1, 2)
        colors = keep_colors

    # 最終フレームに、生き残った点を丸で重ねる。
    last = frames[-1].copy()
    for new, col in zip(p0.reshape(-1, 2), colors):
        cv2.circle(last, (int(new[0]), int(new[1])), 4, col, -1)
    return trace, last, len(p0)


def demo_displacement_stats(frames: list[np.ndarray]) -> None:
    """1ステップ分のフローを取り、点ごとの移動量を集計して『動/静』を切り分ける。"""
    g0 = cv2.cvtColor(frames[10], cv2.COLOR_BGR2GRAY)
    g1 = cv2.cvtColor(frames[11], cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(g0, mask=None, **FEATURE_PARAMS)
    p1, status, _ = cv2.calcOpticalFlowPyrLK(g0, g1, p0, None, **LK_PARAMS)

    ok = status.ravel() == 1
    disp = np.linalg.norm(p1[ok] - p0[ok], axis=2).ravel()  # 各点の移動量(px)
    moving = int((disp > 0.7).sum())
    static = int((disp <= 0.7).sum())
    print("[2] 1フレーム分の移動量で『動/静』を切り分ける")
    print(f"  追跡できた点 = {ok.sum()} / {len(p0)}")
    print(f"  動いた点(>0.7px) = {moving} 個 / ほぼ静止 = {static} 個")
    print("  → 物体上の点は大きく動き、静止ランドマーク上の点はほぼ動かない（移動量で分かる）。")


def main() -> None:
    out = output_dir()
    frames = make_motion_frames()

    trace, last, n_kept = track_sequence(frames)
    print("[1] Lucas-Kanade で特徴点を全フレーム追跡")
    print(f"  最後まで追跡できた点 = {n_kept} 個。軌跡の長い線が『動いた物体』。")
    cv2.imwrite(str(out / "02_lk_tracks.png"), trace)
    cv2.imwrite(str(out / "02_lk_last_points.png"), last)

    demo_displacement_stats(frames)

    print("\n[3] 古典フローの前提と弱点")
    print("  - 前提『輝度一定』: 同じ点は明るさが変わらない、という仮定の上で成り立つ。")
    print("  - 弱点『開口問題』: のっぺりした線上では“線に沿う動き”が決められない。")
    print("    だから角(コーナー)を選ぶ goodFeaturesToTrack と相性が良い。")
    print("\n完了。lectures/10_classical_video_motion/outputs/ の 02_lk_tracks.png を見てください。")


if __name__ == "__main__":
    main()
