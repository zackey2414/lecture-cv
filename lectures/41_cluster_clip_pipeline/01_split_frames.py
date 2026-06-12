"""01 Split — 動画を連番フレームに分割する（パイプラインの入口）。

参考実装対応: src/adaptive_cluster_clip/split/processor.py
  実リポは data/videos の mp4 を cv2.VideoCapture で開き、1 フレームずつ JPEG 保存する。
  ここでは入力動画も合成し、「VideoWriter で書く → VideoCapture で読み戻す」で分割を体験する。

学ぶこと:
  - 動画はフレームのループとして扱える（read が False になったら終わり）。
  - cv2.VideoCapture / cv2.imread は失敗時に例外ではなく False/None を返す → 必ずガードする。
  - 連番ファイル名（{video}_frame{n:06d}.jpg）が後段 Build/Search の ID の素になる。

実行:
    uv run python lectures/41_cluster_clip_pipeline/01_split_frames.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cluster_clip_helpers as H  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

N_FRAMES = 12
FPS = 10.0


def make_synthetic_video(video_path: Path, n_frames: int) -> bool:
    """合成フレームを mp4 として書き出す。書き出しに失敗したら False を返す。"""
    h, w = 180, 240
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, FPS, (w, h))
    if not writer.isOpened():
        return False
    for i in range(n_frames):
        writer.write(H.synth_frame(i, n_frames, h=h, w=w))
    writer.release()
    return True


def split_video_to_frames(video_path: Path, frames_dir: Path, video_name: str) -> list[Path]:
    """cv2.VideoCapture で動画を開き、フレームを連番 JPEG として保存する。"""
    frames_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():  # ★ 開けない場合は例外でなく False が返る
        raise RuntimeError(f"動画を開けませんでした: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"[split] video opened: total_frames={total} fps={fps:.1f}")

    saved: list[Path] = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:  # ★ これ以上フレームが無い
            break
        out = frames_dir / f"{video_name}_frame{idx:06d}.jpg"
        cv2.imwrite(str(out), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        saved.append(out)
        idx += 1
    cap.release()
    return saved


def save_montage(frame_paths: list[Path], out_name: str) -> Path:
    """分割したフレームを並べたモンタージュ図を保存する（BGR→RGB 変換に注意）。"""
    n = len(frame_paths)
    cols = min(6, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(2.0 * cols, 2.0 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, p in zip(axes, frame_paths):
        img = cv2.imread(str(p))  # BGR で読まれる
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(p.name[-13:], fontsize=7)
        ax.axis("off")
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle("Split: synthetic video -> numbered frames", fontsize=12)
    return H.save_fig(fig, out_name)


def main() -> int:
    H.set_seed(0)
    base = H.outputs_dir("capstone")
    video_path = base / "synthA.mp4"
    frames_dir = base / "frames"
    video_name = "synthA"

    # 1) 合成動画を作る（実運用では既に mp4 がある所からスタート）
    print("[split] 合成動画を作成中 ...")
    ok = make_synthetic_video(video_path, N_FRAMES)

    # 2) 動画 → フレーム（VideoWriter が使えない環境では合成フレーム直書きにフォールバック）
    if ok:
        frames = split_video_to_frames(video_path, frames_dir, video_name)
    else:
        print("[split] VideoWriter が使えないため、合成フレームを直接書き出します")
        frames = H.write_synth_frames(frames_dir, N_FRAMES, video_name=video_name)

    print(f"[split] {len(frames)} フレームを保存: {frames_dir}")

    # 3) 可視化
    montage = save_montage(frames, "01_split_montage.png")
    print(f"[plot] saved -> {montage}")

    # 4) 自己検証
    assert len(frames) == N_FRAMES, f"フレーム数が想定と違う: {len(frames)}"
    assert all(p.exists() for p in frames), "保存されていないフレームがある"
    print("[done] Split 完了。次は 02_build_index.py で領域ベクトル化＋索引化を行う。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
