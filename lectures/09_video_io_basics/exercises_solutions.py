"""第9回 演習の模範解答（実行すると全 PASS）。

設計方針:
  - 模範解答の「関数本体」だけをここに置く。**採点ロジックは exercises.py の _grade を
    そのまま再利用**し、重複させない（DRY）。
  - install(module) で exercises モジュールの各 exN_* を本ファイルの解答へ差し替える。
  - exercises.py は SHOW_SOLUTION=1 のとき本ファイルの install() を呼ぶ（循環 import を避けるため
    exercises 側の import は __main__ 内の遅延 import にしてある）。

実行:
  uv run python lectures/09_video_io_basics/exercises_solutions.py
    → 模範解答を組み込んで採点し、全問 PASS することを確認できる。
"""

from __future__ import annotations

import pathlib
import sys
from collections import deque

import cv2
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))


# =====================================================================
# 模範解答（各 exN_* の完成形）
# =====================================================================

def ex1_count_frames(source: str) -> int:
    cap = cv2.VideoCapture(source)
    count = 0
    while True:
        ret, _frame = cap.read()
        if not ret:
            break
        count += 1
    cap.release()
    return count


def ex2_read_meta(source: str) -> tuple[float, int, int, int]:
    cap = cv2.VideoCapture(source)
    fps = cap.get(cv2.CAP_PROP_FPS)
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return (fps, count, width, height)


def ex3_fourcc_to_str(fourcc_int: int) -> str:
    code = int(fourcc_int)
    return "".join(chr((code >> (8 * i)) & 0xFF) for i in range(4))


def ex4_seek_and_read(source: str, index: int) -> np.ndarray | None:
    cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


def ex5_moving_average_fps(dts: list[float], window: int = 20) -> float:
    buf: deque[float] = deque(maxlen=window)
    for d in dts:
        buf.append(d)
    if not buf:
        return 0.0
    avg = sum(buf) / len(buf)
    return 1.0 / avg if avg > 0 else 0.0


def ex6_estimate_duration_sec(source: str) -> float:
    cap = cv2.VideoCapture(source)
    fps = cap.get(cv2.CAP_PROP_FPS)
    count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    return float(count) / float(fps) if fps > 0 else 0.0


def ex7_write_video_roundtrip(frames: list[np.ndarray], fps: float, out_path: str) -> int:
    if not frames:
        return 0
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        writer.release()
        return 0  # 開けない環境ではスキップ（採点側で PASS 扱い）。
    for f in frames:
        writer.write(f)
    writer.release()

    cap = cv2.VideoCapture(out_path)
    count = 0
    while True:
        ret, _frame = cap.read()
        if not ret:
            break
        count += 1
    cap.release()
    return count


def ex8_subsample_indices(source: str, step: int) -> list[int]:
    cap = cv2.VideoCapture(source)
    indices: list[int] = []
    idx = 0
    while True:
        ret, _frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            indices.append(idx)
        idx += 1
    cap.release()
    return indices


def ex9_count_motion_pixels(source: str, thresh: int = 25) -> int:
    cap = cv2.VideoCapture(source)
    prev_gray: np.ndarray | None = None
    total = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            diff = cv2.absdiff(gray, prev_gray)
            total += int(np.count_nonzero(diff > thresh))
        prev_gray = gray
    cap.release()
    return total


# 関数名 → 模範解答 の対応表（install で exercises モジュールへ流し込む）。
_SOLUTIONS = {
    "ex1_count_frames": ex1_count_frames,
    "ex2_read_meta": ex2_read_meta,
    "ex3_fourcc_to_str": ex3_fourcc_to_str,
    "ex4_seek_and_read": ex4_seek_and_read,
    "ex5_moving_average_fps": ex5_moving_average_fps,
    "ex6_estimate_duration_sec": ex6_estimate_duration_sec,
    "ex7_write_video_roundtrip": ex7_write_video_roundtrip,
    "ex8_subsample_indices": ex8_subsample_indices,
    "ex9_count_motion_pixels": ex9_count_motion_pixels,
}


def install(module) -> None:
    """exercises モジュールの各 exN_* を本ファイルの模範解答で上書きする。"""
    for name, fn in _SOLUTIONS.items():
        setattr(module, name, fn)


def main() -> None:
    # 採点ロジックは exercises 側を再利用（ここでは複製しない）。
    import exercises
    install(exercises)
    exercises._grade()


if __name__ == "__main__":
    main()
