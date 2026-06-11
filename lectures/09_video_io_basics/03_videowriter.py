"""03: VideoWriter で書き出す + ソースFPSと処理FPSの違い（この回の完成物）。

学ぶこと:
  - cv2.VideoWriter を FOURCC・FPS・出力サイズ(W,H) を指定して作る正準形。
  - writer.isOpened() で「本当に開けたか」を必ず検証する（開けないと無言で空ファイルになる）。
  - 書き込むフレームのサイズが VideoWriter の出力サイズと一致しないと壊れる/書けない罠。
  - ソースFPS（cap.get で得る動画本来のFPS）と、自分の処理パイプラインの処理FPSは別物。
    time.perf_counter + collections.deque で処理FPSの移動平均をとって表示する。
  - 完成物: 1フレームずつ読み、Nフレーム間引きで縮小画像を連番保存しつつ処理FPSを出し、
    処理結果を VideoWriter で動画に再書き出しする（書けない環境では連番PNGにフォールバック）。

実行:
  uv run python lectures/09_video_io_basics/03_videowriter.py
結果は outputs/09_video_io_basics/ に保存される（cv2.imshow は呼ばない）。
"""

from __future__ import annotations

import pathlib
import sys
import time
from collections import deque

import cv2
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from cv_helpers import make_demo_video, output_dir  # noqa: E402


def open_writer(path: pathlib.Path, fps: float, size: tuple[int, int]) -> cv2.VideoWriter | None:
    """VideoWriter を mp4v→MJPG の順で開き、開けたものを返す（両方ダメなら None）。"""
    # size は (幅, 高さ) の順。書き込むフレームの (W, H) と必ず一致させること。
    for ext, cc in ((".mp4", "mp4v"), (".avi", "MJPG")):
        target = path.with_suffix(ext)
        writer = cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*cc), fps, size)
        if writer.isOpened():  # ★ 開けたかを必ず確認する。
            print(f"  VideoWriter OK: {target.name} (fourcc={cc}, fps={fps}, size={size})")
            return writer
        writer.release()
    print("  VideoWriter を開けませんでした → 連番PNGにフォールバックします。")
    return None


def process_and_rewrite(source: str, src_fps: float, out: pathlib.Path,
                        skip: int = 5, scale: float = 0.5) -> None:
    """完成物: 読み→縮小処理→処理FPS計測→連番PNG保存→VideoWriter で再書き出し。"""
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"動画を開けませんでした: {source}")

    print("[1] ソースFPS vs 処理FPS")
    print(f"  ソースFPS（動画本来のFPS, cap.get）= {cap.get(cv2.CAP_PROP_FPS)}")

    writer: cv2.VideoWriter | None = None
    out_size: tuple[int, int] | None = None
    png_dir = out / "03_processed_frames"  # フォールバック用の連番PNG置き場
    fps_history: list[float] = []           # 各フレームの処理FPS（最後にグラフ化）
    recent = deque(maxlen=20)               # 直近の処理時間で移動平均をとる
    saved = 0
    idx = 0
    prev = time.perf_counter()

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # ret 判定で終了（総フレーム数に依存しない）。

        # --- ここが「処理」: 縮小する（実務の前処理の最小例） ---
        proc = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

        # 出力サイズは最初のフレームで確定し、以降ずれないようにする（サイズ不一致は禁物）。
        if out_size is None:
            h, w = proc.shape[:2]
            out_size = (w, h)  # VideoWriter は (幅, 高さ)
            writer = open_writer(out / "03_processed", src_fps, out_size)

        # --- 処理FPSの計測（time.perf_counter + deque の移動平均） ---
        now = time.perf_counter()
        dt = now - prev
        prev = now
        recent.append(dt)
        avg_dt = sum(recent) / len(recent)
        proc_fps = 1.0 / avg_dt if avg_dt > 0 else 0.0
        fps_history.append(proc_fps)

        # --- VideoWriter があれば動画へ、無ければ連番PNGへ ---
        if writer is not None:
            writer.write(proc)
        else:
            png_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(png_dir / f"{idx:04d}.png"), proc)

        # --- Nフレームに1回だけ、縮小画像を連番で保存（間引き） ---
        if idx % skip == 0:
            cv2.imwrite(str(out / f"03_thumb_{saved:03d}.png"), proc)
            saved += 1

        idx += 1

    cap.release()
    if writer is not None:
        writer.release()

    # 処理FPSの平均（最初の1フレームは dt が不安定なので 2 枚目以降で平均）。
    body = fps_history[1:] if len(fps_history) > 1 else fps_history
    mean_fps = float(np.mean(body)) if body else 0.0
    print(f"  処理FPS（このパイプラインの実測, 平均）≈ {mean_fps:,.0f} fps")
    print(f"  → 処理FPS > ソースFPS({src_fps}) なら実時間に追いつける（間に合う）。")
    print("\n[2] 完成物の出力")
    print(f"  読み込みフレーム数 = {idx} / 間引き保存(縮小サムネ) = {saved} 枚 (skip={skip})")
    print(f"  再書き出し: {'動画ファイル' if writer is not None else '連番PNG(フォールバック)'}")

    _plot_fps(fps_history, src_fps, out)


def _plot_fps(fps_history: list[float], src_fps: float, out: pathlib.Path) -> None:
    """フレームごとの処理FPSと、ソースFPSの水平線を1枚のグラフにする。"""
    if not fps_history:
        return
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.plot(range(len(fps_history)), fps_history, label="processing FPS", color="tab:blue")
    ax.axhline(src_fps, color="tab:red", linestyle="--", label=f"source FPS ({src_fps:g})")
    ax.set_xlabel("frame index")
    ax.set_ylabel("FPS")
    ax.set_title("processing FPS vs source FPS")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(str(out / "03_fps_plot.png"), dpi=110)
    plt.close(fig)
    print("  → 03_fps_plot.png: 処理FPS(青)が ソースFPS(赤破線) を上回れば実時間処理が可能。")


def verify_rewritten(out: pathlib.Path) -> None:
    """再書き出しした動画を VideoCapture で開き直し、メタデータを確認する。"""
    for name in ("03_processed.mp4", "03_processed.avi"):
        path = out / name
        if not path.exists():
            continue
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            cap.release()
            continue
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        print(f"\n[3] 再書き出し動画を読み戻して検証: {name}")
        print(f"  フレーム数={n} サイズ={w}x{h}（書き出した縮小フレームと一致するはず）")
        return
    print("\n[3] 動画は書けず連番PNGで保存（環境のコーデック制約）。検証はスキップ。")


def main() -> None:
    out = output_dir()
    source, fps, num, mode = make_demo_video(out, name="demo_source", fps=24.0, num=60)
    print(f"合成動画を用意: source={source} (mode={mode}, fps={fps}, frames={num})\n")

    process_and_rewrite(source, src_fps=fps, out=out, skip=5, scale=0.5)
    verify_rewritten(out)

    print("\n完了。outputs/09_video_io_basics/ の 03_*.png / 03_processed.* を確認してください。")


if __name__ == "__main__":
    main()
