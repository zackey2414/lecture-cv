"""01: フレーム差分と背景差分 — 動いている前景を抜き出す一番素朴な方法。

学ぶこと:
  - フレーム差分: 連続2枚の引き算 cv2.absdiff → 閾値で前景マスク。実装が一番簡単。
  - 「静止背景との差分」: 背景を1枚固定して引く。前景全体を取れるが背景の更新に弱い。
  - 背景差分の実用版 MOG2/KNN: 背景を統計的に学習し、照明変化や微小ゆれに頑健。
  - 形態学（オープニング）でごま塩ノイズを消し、輪郭→外接矩形で動体を矩形化する流れ。
  - 影検出（MOG2 は影をグレー値127で返す）と、影を消すかどうかの扱い。

実行:
  uv run python lectures/10_classical_video_motion/01_frame_diff_bgsub.py
結果は outputs/10_classical_video_motion/ に保存される（画面表示はしない＝headless 安全）。
"""

from __future__ import annotations

import pathlib
import sys

import cv2
import numpy as np

# 同じフォルダの cv_helpers.py を import できるようにする（どこから実行してもぶれない）。
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from cv_helpers import make_motion_frames, output_dir  # noqa: E402

# 形態学で使う小さな構造要素（前景マスクのごま塩ノイズを掃除する）。
KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


def mask_to_boxes(mask: np.ndarray, min_area: int = 150) -> list[tuple[int, int, int, int]]:
    """二値の前景マスクから、面積がしきい値以上の輪郭の外接矩形を返す。

    findContours は OpenCV 4 系では (contours, hierarchy) の2つ返し（3系の3つ返しと違う）。
    小さすぎる輪郭はノイズなので min_area で捨てる。
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        if cv2.contourArea(c) < min_area:
            continue
        boxes.append(cv2.boundingRect(c))  # (x, y, w, h)
    return boxes


def draw_boxes(frame: np.ndarray, boxes: list[tuple[int, int, int, int]]) -> np.ndarray:
    """前景の外接矩形を緑で描いたコピーを返す（元のフレームは壊さない）。"""
    vis = frame.copy()
    for (x, y, w, h) in boxes:
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return vis


def demo_frame_diff(frames: list[np.ndarray], out: pathlib.Path) -> None:
    """連続2フレームの差分（最も素朴な動体検出）。"""
    a, b = frames[18], frames[19]
    # absdiff は |a - b| を飽和演算で計算（uint8 のラップアラウンドを起こさない）。
    diff = cv2.absdiff(a, b)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    # 大津の自動閾値で「動いた所(明)／動かない所(暗)」を2値化。
    _, fg = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, KERNEL)  # ごま塩ノイズ除去
    boxes = mask_to_boxes(fg)

    print("[1] フレーム差分（連続2枚の引き算）")
    print(f"  差分の最大値 = {gray.max()}（動きが大きいほど大）/ 検出した動体 = {len(boxes)} 個")
    print("  → 物体の『動いた縁』だけが光り、中身は背景と同じ色なので穴が空きやすい。")

    cv2.imwrite(str(out / "01_framediff_diff.png"), diff)
    cv2.imwrite(str(out / "01_framediff_mask.png"), fg)
    cv2.imwrite(str(out / "01_framediff_boxes.png"), draw_boxes(b, boxes))


def demo_static_bg_diff(frames: list[np.ndarray], out: pathlib.Path) -> None:
    """「最初のフレーム＝背景」と固定して引く（前景の中身まで取れる）。"""
    background = frames[0]  # 物体が左端にいる最初の絵を背景の代用にする
    target = frames[25]
    diff = cv2.absdiff(target, background)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, fg = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, KERNEL)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, KERNEL)  # 穴を埋めて塊にする
    boxes = mask_to_boxes(fg)

    print("\n[2] 静止背景との差分（背景を1枚に固定）")
    print(f"  検出した動体 = {len(boxes)} 個。前景の『中身』まで塗れるのが差分2枚より良い点。")
    print("  ただし背景が変わる（照明・木の揺れ）と途端に破綻する → だから MOG2/KNN へ。")

    cv2.imwrite(str(out / "01_bgdiff_mask.png"), fg)
    cv2.imwrite(str(out / "01_bgdiff_boxes.png"), draw_boxes(target, boxes))


def run_subtractor(frames: list[np.ndarray], subtractor, name: str,
                   out: pathlib.Path) -> None:
    """背景差分器（MOG2 or KNN）を全フレームに順に適用し、最後のマスクを保存する。

    背景差分器は「過去フレームから背景モデルを学習」するので、何枚か apply して
    『慣らし運転』してから前景が安定する。途中で履歴を渡さず1枚だけ apply してはいけない。
    """
    last_mask = None
    last_frame = None
    for f in frames:
        fgmask = subtractor.apply(f)  # 内部で背景モデルを更新しつつ前景マスクを返す
        last_mask, last_frame = fgmask, f

    # MOG2 は影を 127（グレー）で返す設定があるので、127 は影として 0 に落とす。
    _, fg = cv2.threshold(last_mask, 200, 255, cv2.THRESH_BINARY)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, KERNEL)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, KERNEL)
    boxes = mask_to_boxes(fg)

    print(f"\n[3] 背景差分 {name}（背景を統計的に学習）")
    print(f"  最終フレームで検出した動体 = {len(boxes)} 個。")
    cv2.imwrite(str(out / f"01_{name.lower()}_mask.png"), fg)
    cv2.imwrite(str(out / f"01_{name.lower()}_boxes.png"), draw_boxes(last_frame, boxes))


def demo_background_subtractors(frames: list[np.ndarray], out: pathlib.Path) -> None:
    """MOG2 と KNN の2つを比較する（どちらも opencv 本体同梱・contrib 不要）。"""
    # detectShadows=True にすると影をグレー(127)で出してくれる（後で閾値で落とせる）。
    mog2 = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=25,
                                              detectShadows=True)
    knn = cv2.createBackgroundSubtractorKNN(history=200, dist2Threshold=400.0,
                                            detectShadows=True)
    run_subtractor(frames, mog2, "MOG2", out)
    run_subtractor(frames, knn, "KNN", out)
    print("  → MOG2 は混合ガウス、KNN は近傍法。どちらも背景の微小変動に頑健で実用的。")


def main() -> None:
    out = output_dir()
    frames = make_motion_frames()
    demo_frame_diff(frames, out)
    demo_static_bg_diff(frames, out)
    demo_background_subtractors(frames, out)
    print("\n完了。outputs/10_classical_video_motion/ の 01_*.png を見比べてください。")
    print("  framediff（縁だけ）→ bgdiff（中身も）→ MOG2/KNN（頑健）の順に賢くなります。")


if __name__ == "__main__":
    main()
