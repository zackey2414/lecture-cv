"""01: VideoCapture の正準ループ — isOpened/read の ret 判定/release とフレームの基本操作。

学ぶこと:
  - 動画 = 連続した静止画（フレーム）であること。1フレームずつ read() で取り出す。
  - VideoCapture を開いて while ループで回し、ret（成功フラグ）でループ終了を判定する正準形。
  - 使い終わったら必ず release() する（ファイルハンドル/デバイスを解放）。
  - read() が返す frame は (H, W, 3) uint8 の BGR numpy 配列。
    cvtColor(グレー化)・resize(縮小)・numpy スライス(ROI 切り出し) を体験する。
  - OpenCV は BGR・matplotlib/Pillow は RGB。変換を忘れると赤と青が入れ替わる罠を確認。

実行:
  uv run python lectures/09_video_io_basics/01_videocapture_loop.py
結果は lectures/09_video_io_basics/outputs/ に保存される（cv2.imshow は呼ばない＝headless 安全）。
"""

from __future__ import annotations

import pathlib
import sys

import cv2
import numpy as np

# 描画は必ず Agg バックエンド（GUI 不要）に固定してから pyplot を import する。
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# 同じフォルダの cv_helpers.py を import できるようにする（どこから実行してもぶれない）。
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from cv_helpers import make_demo_video, output_dir  # noqa: E402


def canonical_capture_loop(source: str, out: pathlib.Path) -> tuple[int, np.ndarray]:
    """VideoCapture を開く → while で read() → ret 判定 → release の「正準パターン」。"""
    # 文字列パス（or 画像シーケンスのパターン、or カメラ番号）を渡して開く。
    cap = cv2.VideoCapture(source)

    # 開けたかを必ず確認する。失敗時に read() し続けると無限ループになりやすい。
    if not cap.isOpened():
        raise RuntimeError(f"動画を開けませんでした: {source}")

    count = 0
    first_frame: np.ndarray | None = None
    while True:
        # read() は (ret, frame) を返す。ret=False で「もう次のフレームは無い」を表す。
        ret, frame = cap.read()
        if not ret:
            break  # ★ ret 判定がループ終了条件。フレーム数を前提にループしない。
        if first_frame is None:
            first_frame = frame  # 1枚目を後で目視確認のため取っておく。
        count += 1

    # ★ 使い終わったら必ず解放する（with 文が無いので明示的に呼ぶ）。
    cap.release()

    if first_frame is None:
        raise RuntimeError("1フレームも読めませんでした")

    print("[1] VideoCapture の正準ループ")
    print(f"  読み込んだフレーム総数 = {count}")
    print(f"  1フレームの shape={first_frame.shape} dtype={first_frame.dtype}"
          f"（= 高さ×幅×3チャンネルの uint8 配列）")
    cv2.imwrite(str(out / "01_first_frame.png"), first_frame)
    return count, first_frame


def frame_basic_ops(frame: np.ndarray, out: pathlib.Path) -> None:
    """1枚のフレームに対し cvtColor/resize/ROI という前処理の基本3点を試す。"""
    h, w = frame.shape[:2]

    # (a) グレースケール化。出力は (H, W) の2次元になる（チャンネル軸が消える点に注意）。
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(str(out / "01_gray.png"), gray)

    # (b) 縮小（半分のサイズへ）。dsize は (幅, 高さ) の順。縮小には INTER_AREA が定石。
    small = cv2.resize(frame, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(out / "01_resized_half.png"), small)

    # (c) ROI（関心領域）の切り出し。numpy スライスは [y0:y1, x0:x1] の順。
    roi = frame[h // 4: 3 * h // 4, w // 4: 3 * w // 4]
    cv2.imwrite(str(out / "01_roi_center.png"), roi)

    print("\n[2] フレームの基本操作（前処理の土台）")
    print(f"  グレー化: shape {frame.shape} → {gray.shape}（チャンネル軸が消える）")
    print(f"  リサイズ(半分): {frame.shape[:2][::-1]} → {small.shape[:2][::-1]} (W,H)")
    print(f"  ROI(中央): shape {roi.shape}")


def bgr_vs_rgb_figure(frame: np.ndarray, out: pathlib.Path) -> None:
    """matplotlib(RGB前提)に BGR のまま渡すと色が崩れることを並べて可視化する。"""
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))

    # 左: BGR のまま渡す → 赤と青が入れ替わって見える（オレンジが青っぽくなる）。
    axes[0].imshow(frame)
    axes[0].set_title("WRONG: BGR as-is")
    axes[0].axis("off")

    # 右: cvtColor で BGR→RGB に直してから渡す → 正しい色。
    axes[1].imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    axes[1].set_title("OK: BGR2RGB")
    axes[1].axis("off")

    fig.tight_layout()
    fig.savefig(str(out / "01_bgr_vs_rgb.png"), dpi=110)
    plt.close(fig)
    print("\n[3] BGR/RGB の確認")
    print("  → 01_bgr_vs_rgb.png: 左(変換なし)は色が崩れ、右(BGR2RGB)が正しい。")


def note_webcam() -> None:
    """Webカメラの開き方だけ説明する（本講座では開かない＝カメラ不要・headless）。"""
    print("\n[4] Webカメラについて（このスクリプトでは開きません）")
    print("  ファイルの代わりに cv2.VideoCapture(0) でカメラ#0 を開ける（API は read/release で同じ）。")
    print("  ただし本講座はカメラ不要・headless 前提のため、合成した動画ファイルで学びます。")


def main() -> None:
    out = output_dir()

    # 学習用の動画ファイルを合成生成する（mp4v が無理なら MJPG / 連番PNG に自動フォールバック）。
    source, fps, num, mode = make_demo_video(out, name="demo_source", fps=24.0, num=60)
    print(f"合成動画を用意: source={source} (mode={mode}, fps={fps}, frames={num})\n")

    count, first = canonical_capture_loop(source, out)
    frame_basic_ops(first, out)
    bgr_vs_rgb_figure(first, out)
    note_webcam()

    print("\n完了。lectures/09_video_io_basics/outputs/ の 01_*.png を確認してください。")


if __name__ == "__main__":
    main()
