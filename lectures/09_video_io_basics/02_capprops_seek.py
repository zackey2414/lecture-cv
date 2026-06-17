"""02: 動画メタデータ取得（CAP_PROP）と POS_FRAMES シーク・ライブでの注意。

学ぶこと:
  - cap.get(cv2.CAP_PROP_*) で FPS / 総フレーム数 / 幅 / 高さ / FOURCC を取得する。
  - FOURCC は 32bit 整数で返るので 4 文字のコーデック名にデコードする。
    （要求した "mp4v" と、コンテナに記録される名前が違うことがある点も確認）。
  - cap.set(cv2.CAP_PROP_POS_FRAMES, i) で任意フレームへシーク（ファイルのみ可能）。
  - ライブ/ストリームでは総フレーム数が 0 や不正値になり得る → 総数前提のループを書かない判断。

実行:
  uv run python lectures/09_video_io_basics/02_capprops_seek.py
結果は lectures/09_video_io_basics/outputs/ に保存される（cv2.imshow は呼ばない）。
"""

from __future__ import annotations

import pathlib
import sys

import cv2
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from cv_helpers import fourcc_to_str, make_demo_video, output_dir  # noqa: E402


def read_metadata(source: str) -> dict:
    """CAP_PROP_* でメタデータをまとめて取得して表示・返却する。"""
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"動画を開けませんでした: {source}")

    # get() は常に float を返す。整数で欲しいものは int() で丸める。
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
    cap.release()

    meta = {
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "fourcc": fourcc_to_str(fourcc_int),
    }
    print("[1] 動画メタデータ（cap.get / CAP_PROP_*）")
    print(f"  FPS            = {fps}")
    print(f"  総フレーム数    = {frame_count}")
    print(f"  幅 x 高さ       = {width} x {height}")
    print(f"  FOURCC(記録名)  = {meta['fourcc']!r}（要求した 'mp4v' と異なる場合がある）")

    # 総フレーム数 / FPS から「動画の長さ（秒）」が計算できる。
    if fps > 0:
        print(f"  推定の長さ      = {frame_count / fps:.2f} 秒")
    return meta


def seek_frames(source: str, indices: list[int], out: pathlib.Path) -> None:
    """POS_FRAMES で指定フレームへ飛んで取り出し、横に並べて保存する（ファイルのみ可）。"""
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"動画を開けませんでした: {source}")

    grabbed: list[tuple[int, np.ndarray]] = []
    for idx in indices:
        # 次に read() するフレーム位置を idx にセットしてから read() する。
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            print(f"  フレーム {idx} は読めませんでした（範囲外?）")
            continue
        # set 直後の read で位置は idx+1 に進む。描かれた "frame NNN" と照合できる。
        pos_now = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        grabbed.append((idx, frame))
        print(f"  シーク idx={idx:3d} → 取得成功（read 後の位置={pos_now}）")
    cap.release()

    print("\n[2] POS_FRAMES シーク（ファイルだからできる。動く円の位置で確認）")
    if not grabbed:
        print("  シーク不可（画像シーケンス等）。スキップして継続。")
        return

    # 取り出した各フレームを matplotlib で横並びにして保存（BGR→RGB を忘れない）。
    fig, axes = plt.subplots(1, len(grabbed), figsize=(3.4 * len(grabbed), 3.0))
    if len(grabbed) == 1:
        axes = [axes]
    for ax, (idx, frame) in zip(axes, grabbed):
        ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ax.set_title(f"seek to {idx}")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(str(out / "02_seek_grid.png"), dpi=110)
    plt.close(fig)
    print("  → 02_seek_grid.png: 円の位置とテキスト 'frame NNN' が指定idxと一致するはず。")


def live_stream_caution(source: str) -> None:
    """「ライブでは FRAME_COUNT が不正値になり得る」を踏まえた安全なループの考え方。"""
    cap = cv2.VideoCapture(source)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print("\n[3] ライブ/ストリームでの注意")
    print(f"  ファイルなら総フレーム数={total} が信頼できるが、Webカメラや RTSP では")
    print("  総数が 0 や負値・巨大値になり得る。そこで総数に依存せず ret 判定で回すのが安全。")

    # 総数を信用せず、read() の ret だけでループする「ライブ安全」な書き方の例。
    safe_count = 0
    while True:
        ret, _frame = cap.read()
        if not ret:
            break
        safe_count += 1
        if safe_count >= 100000:  # 暴走よけの上限だけ付ける（ライブは自然には止まらない）。
            break
    cap.release()
    print(f"  ret 判定だけで数えたフレーム数={safe_count}（総数前提の for ループを避ける）")


def main() -> None:
    out = output_dir()
    source, fps, num, mode = make_demo_video(out, name="demo_source", fps=24.0, num=60)
    print(f"合成動画を用意: source={source} (mode={mode}, 要求fps={fps}, frames={num})\n")

    read_metadata(source)
    # 先頭・中ほど・終端付近へ飛んでみる（円の位置で正しさが目で分かる）。
    seek_frames(source, [0, num // 2, num - 1], out)
    live_stream_caution(source)

    print("\n完了。lectures/09_video_io_basics/outputs/ の 02_*.png を確認してください。")


if __name__ == "__main__":
    main()
