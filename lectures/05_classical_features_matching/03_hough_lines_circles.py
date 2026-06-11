"""03: Hough 変換による直線・円検出 — Canny→HoughLinesP/HoughCircles とパラメータ過敏性。

学ぶこと:
  - パラメータ空間への「投票」という古典の発想。エッジ点が直線/円のパラメータに票を入れる。
  - 直線検出の定石: まず Canny でエッジ画像を作り、HoughLinesP に渡す（前段が必須）。
  - HoughLinesP は線「分」（端点付き）を返す。閾値・minLineLength・maxLineGap に極端に敏感。
  - 円検出 HoughCircles はエッジ画像でなく「グレー画像」を渡す（内部で Canny+勾配を使う）。
  - param2（投票の閾値）を上げ過ぎると 0 個、下げ過ぎると誤検出だらけになる過敏さを体感する。

実行:
  uv run python lectures/05_classical_features_matching/03_hough_lines_circles.py
結果は outputs/05_classical_features_matching/ に保存される（画面表示はしない）。
"""

from __future__ import annotations

import pathlib
import sys

import cv2
import numpy as np

# headless 環境でも図を保存できるよう、pyplot より前に Agg バックエンドを指定する。
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from cv_helpers import make_shapes_bgr, output_dir  # noqa: E402


def main() -> None:
    out = output_dir()

    # === 1. 入力と前処理（Canny エッジ） ==================================
    shapes = make_shapes_bgr()
    gray = cv2.cvtColor(shapes, cv2.COLOR_BGR2GRAY)
    # Canny: 低しきい値・高しきい値の 2 値でエッジを決める。Hough 直線検出の前段に必須。
    edges = cv2.Canny(gray, 50, 150)
    cv2.imwrite(str(out / "03_input.png"), shapes)
    cv2.imwrite(str(out / "03_edges.png"), edges)
    print("[1] Canny でエッジ画像を作成（直線 Hough はこのエッジ画像を入力にする）")

    # === 2. HoughLinesP（確率的ハフ）で線分検出 ===========================
    # 返り値は (N, 1, 4) で各行が [x1, y1, x2, y2]（線分の端点）。
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=80,
                            minLineLength=40, maxLineGap=10)
    line_vis = shapes.copy()
    n_lines = 0 if lines is None else len(lines)
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0, :]:
            cv2.line(line_vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.imwrite(str(out / "03_lines.png"), line_vis)
    print(f"[2] HoughLinesP（threshold=80）: {n_lines} 本の線分を検出")

    # === 3. 直線検出のパラメータスイープ（過敏性を数値で見る） =============
    # threshold（最低投票数）を上げると検出が減り、minLineLength を上げると短い線が消える。
    print("\n[3] HoughLinesP パラメータスイープ（検出された線分数）")
    print(f"  {'threshold':>10s} {'minLen':>7s} {'maxGap':>7s} {'線分数':>7s}")
    for threshold in (30, 80, 150):
        for min_len in (20, 60):
            res = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold,
                                  minLineLength=min_len, maxLineGap=10)
            count = 0 if res is None else len(res)
            print(f"  {threshold:10d} {min_len:7d} {10:7d} {count:7d}")
    print("  → わずかな閾値変更で本数が大きく変わる。Hough は『当たりのパラメータ探し』が肝。")

    # === 4. HoughCircles で円検出 =========================================
    # 注意: 円検出には「グレー画像」を渡す（内部で Canny を呼ぶため、自分でエッジ化しない）。
    # param1=Canny の上側しきい値、param2=中心の投票しきい値（小さいほど検出が緩い）。
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=40,
                               param1=120, param2=30, minRadius=20, maxRadius=80)
    circle_vis = shapes.copy()
    n_circ = 0 if circles is None else circles.shape[1]
    if circles is not None:
        # 返り値は float の (1, N, 3) で各行 [cx, cy, r]。描画用に整数化する。
        for cx, cy, r in np.uint16(np.around(circles))[0]:
            cv2.circle(circle_vis, (int(cx), int(cy)), int(r), (0, 255, 0), 2)
            cv2.circle(circle_vis, (int(cx), int(cy)), 2, (0, 0, 255), 3)  # 中心点
    cv2.imwrite(str(out / "03_circles.png"), circle_vis)
    print(f"\n[4] HoughCircles（param2=30）: {n_circ} 個の円を検出（正解は 3 個）")

    # === 5. 円検出のパラメータスイープ ====================================
    # param2 を上げ過ぎると 0 個、下げ過ぎると誤検出が出る「ちょうどよさ」の難しさを見る。
    print("\n[5] HoughCircles param2 スイープ（小さいほど検出が緩い）")
    print(f"  {'param2':>7s} {'円数':>6s}")
    for param2 in (15, 25, 30, 40, 60):
        res = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=40,
                               param1=120, param2=param2, minRadius=20, maxRadius=80)
        count = 0 if res is None else res.shape[1]
        print(f"  {param2:7d} {count:6d}")
    print("  → param2 が高過ぎると 1 個も出ず、低過ぎると幻の円が出る。実画像では調整必須。")

    # === 6. まとめ図（matplotlib, Agg） ===================================
    # cv2 画像は BGR。matplotlib は RGB なので変換してから渡す（忘れると色が化ける）。
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes[0, 0].imshow(cv2.cvtColor(shapes, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("input")
    axes[0, 1].imshow(edges, cmap="gray")
    axes[0, 1].set_title("Canny edges")
    axes[1, 0].imshow(cv2.cvtColor(line_vis, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title(f"HoughLinesP ({n_lines} segments)")
    axes[1, 1].imshow(cv2.cvtColor(circle_vis, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title(f"HoughCircles ({n_circ} circles)")
    for ax in axes.ravel():
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(str(out / "03_summary.png"), dpi=110)
    plt.close(fig)

    print("\n完了。outputs/05_classical_features_matching/ の 03_*.png（特に 03_summary.png）を確認。")


if __name__ == "__main__":
    main()
