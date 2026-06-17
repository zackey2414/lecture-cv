"""01: 背景差分による動体検出（MOG2 / KNN ＋ モルフォロジー）。

リアルタイム/ストリーム処理の入口。固定カメラの映像では「背景」はほとんど動かない。
そこで背景を統計モデル（各画素の色の分布）として学習し、そこから外れた画素を
『前景＝動いているもの』とみなす。これが背景差分（background subtraction）で、
学習済みモデルを毎フレーム更新しながらCPUだけで軽快に動くため、ストリーム実演の最初の一歩に最適。

このスクリプトで体験すること:
  - cv2.createBackgroundSubtractorMOG2 / KNN で前景マスクを作る（apply）
  - 立ち上がり（warm-up）: 最初の数フレームは背景未学習で前景だらけになる
  - 生マスクのノイズ・影(127)をモルフォロジー(open/close)で掃除する
  - 掃除後マスクから輪郭→外接矩形を取り、動体にバウンディングボックスを描く
  - MOG2 と KNN を並べて比較する

入力はカメラ不要の合成フレーム（静止背景の上を円が動く）。結果はすべて
lectures/11_realtime_stream/outputs/ に保存する（headless でも後から確認できる）。

実行:
  uv run python lectures/11_realtime_stream/01_background_subtraction.py
"""

from __future__ import annotations

import pathlib
import sys

import cv2
import numpy as np

# matplotlib は import 前に Agg（画面不要のバックエンド）へ固定する。
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from stream_helpers import output_dir, synthetic_frames  # noqa: E402


def build_subtractor(kind: str):
    """背景差分器を作る。kind は "MOG2" か "KNN"。

    detectShadows=True にすると影と判断した画素を前景(255)ではなく『影(127)』として返す。
    影を前景に混ぜたくない場面が多いので、後段の clean_mask で 127 を捨てて 2値化する。
    """
    if kind == "MOG2":
        # history: 背景学習に使う過去フレーム数。varThreshold: 背景/前景を分ける感度。
        return cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=24, detectShadows=True)
    if kind == "KNN":
        return cv2.createBackgroundSubtractorKNN(history=200, dist2Threshold=400.0, detectShadows=True)
    raise ValueError(f"unknown kind: {kind}")


def clean_mask(raw_mask: np.ndarray) -> np.ndarray:
    """生の前景マスクを掃除して 0/255 の2値マスクにする。

    手順:
      1. 影(127)を捨てる: 値が 255 の画素だけを前景として残す（>200 で2値化）。
      2. オープニング(収縮→膨張)で『ぽつぽつした誤検出』を消す。
      3. クロージング(膨張→収縮)で『物体内部の穴』を埋める。
    構造要素は楕円形(MORPH_ELLIPSE)。サイズを上げるほど強く掃除されるが、細部も潰れる。
    """
    # 127(影)を 0 に落とし、255(確実な前景)だけ残す。
    _, binary = cv2.threshold(raw_mask, 200, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=2)
    return closed


def detect_boxes(mask: np.ndarray, min_area: int = 150) -> list[tuple[int, int, int, int]]:
    """2値マスクから輪郭を取り、面積が min_area 以上のものだけ外接矩形にして返す。

    cv2.findContours は OpenCV 4 系では (contours, hierarchy) の2つ返し（3系の3つ返しと違う）。
    小さすぎる領域はノイズとみなして捨てる。返すのは (x, y, w, h) のリスト。
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        if cv2.contourArea(c) < min_area:
            continue
        boxes.append(cv2.boundingRect(c))
    return boxes


def annotate(frame_bgr: np.ndarray, boxes: list[tuple[int, int, int, int]]) -> np.ndarray:
    """検出した矩形を緑枠でフレームに描いて返す（元フレームは壊さずコピーに描く）。"""
    out = frame_bgr.copy()
    for (x, y, w, h) in boxes:
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(out, f"objects: {len(boxes)}", (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    return out


def gray_to_bgr(mask: np.ndarray) -> np.ndarray:
    """1ch マスクを3ch にして、カラーフレームと横に並べられるようにする。"""
    return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)


def main() -> None:
    out = output_dir()
    # 80フレーム生成。最初の方は背景未学習なので、十分に学習が進んだフレームで可視化する。
    frames = list(synthetic_frames(num_frames=80, size=(360, 640), seed=1))

    mog2 = build_subtractor("MOG2")
    knn = build_subtractor("KNN")

    # 前景率（前景画素の割合）の推移を記録して warm-up を可視化する。
    fg_ratio_mog2: list[float] = []
    snapshot_at = 60  # この番号のフレームを4枚パネルで保存する（十分に学習済み）
    panel_saved = False

    for idx, frame in frames:
        # apply: 1フレーム入れると、その画素を背景モデルと比較した前景マスクが返る。
        # 同時にモデルも内部で更新される（次フレームのために背景を少しずつ学習）。
        raw_mog2 = mog2.apply(frame)
        raw_knn = knn.apply(frame)

        clean_mog2 = clean_mask(raw_mog2)
        fg_ratio_mog2.append(float((clean_mog2 > 0).mean()))

        if idx == snapshot_at and not panel_saved:
            # 4枚パネル: [入力 | 生マスク(MOG2) | 掃除後マスク | 検出結果]
            boxes = detect_boxes(clean_mog2)
            panel = np.hstack([
                frame,
                gray_to_bgr(raw_mog2),
                gray_to_bgr(clean_mog2),
                annotate(frame, boxes),
            ])
            cv2.imwrite(str(out / "01_mog2_pipeline.png"), panel)
            print(f"[panel] frame#{idx}: 検出 {len(boxes)} 個 -> 01_mog2_pipeline.png")

            # MOG2 と KNN の掃除後マスクを並べて比較（matplotlib, グレー表示）。
            clean_knn = clean_mask(raw_knn)
            fig, axes = plt.subplots(1, 2, figsize=(9, 3))
            axes[0].imshow(clean_mog2, cmap="gray")
            axes[0].set_title("MOG2 (cleaned)")
            axes[1].imshow(clean_knn, cmap="gray")
            axes[1].set_title("KNN (cleaned)")
            for ax in axes:
                ax.axis("off")
            fig.suptitle(f"background subtraction comparison @frame {idx}")
            fig.savefig(out / "01_mog2_vs_knn.png", dpi=120, bbox_inches="tight")
            plt.close(fig)
            print("[compare] MOG2 vs KNN -> 01_mog2_vs_knn.png")
            panel_saved = True

    # 前景率の推移グラフ（warm-up: 序盤は背景未学習で前景率が高い）。
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(fg_ratio_mog2, marker=".")
    ax.set_xlabel("frame index")
    ax.set_ylabel("foreground ratio (MOG2, cleaned)")
    # 注: matplotlib の既定フォントは日本語グリフを持たないため、図中の文字は英語にする。
    ax.set_title("warm-up: high FG ratio early (background not learned), then only movers remain")
    ax.grid(True, alpha=0.3)
    fig.savefig(out / "01_warmup_ratio.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("[warm-up] 前景率の推移 -> 01_warmup_ratio.png")

    print("\n完了。lectures/11_realtime_stream/outputs/ に背景差分の結果を保存しました。")


if __name__ == "__main__":
    main()
