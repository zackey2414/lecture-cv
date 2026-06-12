"""第9回 章末ミニプロジェクト — 合成動画の「動体ハイライト」パイプライン＋レポート。

この回で学んだ要素を 1 本に統合する総合課題の「完成形」です。具体的には:
  - 動画を numpy/cv2 で**その場で合成生成**（2つの動く物体＋フレーム番号テキスト）。
  - cv2.VideoCapture の**正準ループ**（isOpened → while read → ret 判定 → release）。
  - フレームは BGR numpy 配列。cvtColor でグレー化し、**連続フレーム差分**で動体を検出。
  - 動いた画素を赤でハイライトした結果を cv2.VideoWriter で**再書き出し**（mp4v→MJPG→連番PNG）。
  - time.perf_counter + collections.deque で**処理FPS の移動平均**を計測（ソースFPSと対比）。
  - cap.get(CAP_PROP_*) で**メタデータ**取得＋ FOURCC を 4 文字へデコード。
  - POS_FRAMES で**シーク**し、代表フレームのモンタージュを作る（headless 安全に保存）。
  - 図(PNG)・JSONレポートを outputs/09_video_io_basics/ に出力する。

設計上の注意:
  - digit始まりの既存スクリプト(01_/02_/03_)は import できないため、必要な処理はすべて
    この 1 ファイル内に**自己完結**で実装する（出力先 output_dir だけ cv_helpers を借用）。
  - cv2.imshow は一切呼ばない（opencv-python-headless では存在せず cv2.error になる）。
  - matplotlib は Agg バックエンド固定。フレームを描くときは必ず BGR→RGB に直す。

実行:
  uv run python lectures/09_video_io_basics/mini_project.py
"""

from __future__ import annotations

import json
import pathlib
import sys
import time
from collections import deque

import cv2
import numpy as np

import matplotlib
matplotlib.use("Agg")  # GUI 不要のバックエンドに固定してから pyplot を import する。
import matplotlib.pyplot as plt  # noqa: E402

# 出力先の解決だけ同モジュールの cv_helpers を借りる（非digitヘルパなので import 可）。
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from cv_helpers import output_dir  # noqa: E402

# --- このミニプロジェクト固有の定数（決定的に動かすため固定） ----------------
SRC_FPS = 24.0          # 合成するソース動画の FPS（= ソースFPS）
NUM_FRAMES = 72         # 合成フレーム数
WIDTH, HEIGHT = 320, 240
MOTION_THRESHOLD = 25   # グレースケール差分がこの値を超えた画素を「動いた」とみなす
THUMB_SKIP = 12         # 何フレームに1回サムネを保存するか（間引き）
FPS_WINDOW = 20         # 処理FPS 移動平均の窓幅


# =====================================================================
# 1) ソース動画の合成（2つの動く物体＋フレーム番号）
# =====================================================================

def make_scene_frame(i: int, num: int) -> np.ndarray:
    """フレーム番号 i の合成フレームを 1 枚作る（BGR uint8 (H,W,3)）。

    左→右に動くオレンジの円と、右→左に動く緑の四角を描く。2 物体が動くので、
    後段の連続フレーム差分（動体検出）で必ず変化画素が出る。番号テキストも入れ、
    シーク結果の答え合わせができるようにする。
    """
    frame = np.full((HEIGHT, WIDTH, 3), 30, dtype=np.uint8)  # 濃いグレー背景
    cv2.rectangle(frame, (8, 8), (WIDTH - 8, HEIGHT - 8), (70, 70, 70), 2)  # 固定枠

    t = i / max(num - 1, 1)  # 0.0 → 1.0 の進行度（ゼロ除算回避）

    # オレンジの円: 左→右（BGR なのでオレンジは (0,165,255)）。
    cx = 24 + int(t * (WIDTH - 48))
    cv2.circle(frame, (cx, HEIGHT // 3), 16, (0, 165, 255), thickness=-1)

    # 緑の四角: 右→左（逆向きに動かして動体を増やす）。
    sx = (WIDTH - 48) - int(t * (WIDTH - 80))
    cv2.rectangle(frame, (sx, 2 * HEIGHT // 3 - 16),
                  (sx + 32, 2 * HEIGHT // 3 + 16), (0, 200, 0), thickness=-1)

    cv2.putText(frame, f"frame {i:03d}", (12, HEIGHT - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def open_writer(path: pathlib.Path, fps: float, size: tuple[int, int]
                ) -> tuple[cv2.VideoWriter | None, str]:
    """VideoWriter を mp4v→MJPG の順に試して開く。両方ダメなら (None, "png-sequence")。

    size は (幅, 高さ) の順。書き込むフレームの (W, H) と必ず一致させること。
    isOpened() で「本当に開けたか」を検証してから返す（開けないと無言で空ファイルになる）。
    """
    for ext, cc in ((".mp4", "mp4v"), (".avi", "MJPG")):
        target = path.with_suffix(ext)
        writer = cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*cc), fps, size)
        if writer.isOpened():
            return writer, f"{cc}{ext}"
        writer.release()
    return None, "png-sequence"


def synth_source_video(out: pathlib.Path) -> tuple[str, str]:
    """合成フレーム列をソース動画として書き出し、(source, mode) を返す。

    mp4v/MJPG が両方開けない環境では連番PNG を書き、VideoCapture が読める
    "%04d.png" パターンを source として返す（どんな環境でも必ず素材が残る）。
    """
    frames = [make_scene_frame(i, NUM_FRAMES) for i in range(NUM_FRAMES)]
    writer, mode = open_writer(out / "mini_project_source", SRC_FPS, (WIDTH, HEIGHT))

    if writer is not None:
        for f in frames:
            writer.write(f)
        writer.release()
        suffix = ".mp4" if mode.endswith(".mp4") else ".avi"
        return str((out / "mini_project_source").with_suffix(suffix)), mode

    # フォールバック: 連番PNG（画像シーケンスとして VideoCapture が読める）。
    seq = out / "mini_project_source_frames"
    seq.mkdir(parents=True, exist_ok=True)
    for i, f in enumerate(frames):
        cv2.imwrite(str(seq / f"{i:04d}.png"), f)
    return str(seq / "%04d.png"), mode


# =====================================================================
# 2) メタデータ取得（CAP_PROP）と FOURCC デコード
# =====================================================================

def fourcc_to_str(code: int) -> str:
    """CAP_PROP_FOURCC の 32bit 整数を 4 文字のコーデック名に復元する。"""
    code = int(code)
    return "".join(chr((code >> (8 * i)) & 0xFF) for i in range(4))


def probe_metadata(source: str) -> dict:
    """cap.get(CAP_PROP_*) で FPS/総数/幅/高さ/FOURCC を取り、長さ(秒)まで計算して返す。"""
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"動画を開けませんでした: {source}")

    fps = cap.get(cv2.CAP_PROP_FPS)  # get() は常に float。整数で欲しいものは int() で丸める。
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = fourcc_to_str(int(cap.get(cv2.CAP_PROP_FOURCC)))
    cap.release()

    duration = frame_count / fps if fps > 0 else 0.0
    meta = {
        "fps": round(float(fps), 3),
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "fourcc": fourcc,
        "duration_sec": round(duration, 3),
    }
    print("[1] メタデータ（cap.get / CAP_PROP_*）")
    print(f"  FPS={meta['fps']}  総数={frame_count}  {width}x{height}  "
          f"FOURCC={fourcc!r}  長さ={meta['duration_sec']}秒")
    return meta


# =====================================================================
# 3) 本体パイプライン: 読み→動体検出→ハイライト→再書き出し＋処理FPS計測
# =====================================================================

def process_pipeline(source: str, out: pathlib.Path
                     ) -> tuple[dict, list[float], list[int]]:
    """正準ループで読みながら連続フレーム差分で動体を赤くハイライトし、動画へ再書き出す。

    同時に処理FPS の移動平均・動体画素数の推移を集計し、(stats, fps_history, motion_counts)
    を返す。完成物の中核。
    """
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"動画を開けませんでした: {source}")

    writer, rewrite_mode = open_writer(out / "mini_project_processed", SRC_FPS, (WIDTH, HEIGHT))
    png_dir = out / "mini_project_processed_frames"  # 動画が書けないときの連番PNG置き場

    prev_gray: np.ndarray | None = None
    recent = deque(maxlen=FPS_WINDOW)  # 直近の処理時間 → 移動平均で滑らかな処理FPS
    fps_history: list[float] = []
    motion_counts: list[int] = []
    thumbs_saved = 0
    idx = 0
    prev_t = time.perf_counter()

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # ★ 終了は ret 判定だけに任せる（総フレーム数に依存しない）。

        # --- 動体検出: 連続フレームのグレースケール差分が閾値を超えた画素 ---
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # (H,W,3) → (H,W)
        if prev_gray is None:
            mask = np.zeros(gray.shape, dtype=bool)  # 1枚目は比較対象が無いので動体なし
        else:
            diff = cv2.absdiff(gray, prev_gray)
            mask = diff > MOTION_THRESHOLD
        prev_gray = gray
        motion_px = int(np.count_nonzero(mask))
        motion_counts.append(motion_px)

        # 動いた画素を赤(BGR=(0,0,255))で塗ってハイライトしたフレームを作る。
        vis = frame.copy()
        vis[mask] = (0, 0, 255)

        # --- 処理FPS（time.perf_counter + deque の移動平均） ---
        now = time.perf_counter()
        recent.append(now - prev_t)
        prev_t = now
        avg_dt = sum(recent) / len(recent)
        fps_history.append(1.0 / avg_dt if avg_dt > 0 else 0.0)

        # --- 結果の書き出し（動画があれば動画へ、無ければ連番PNGへ） ---
        if writer is not None:
            writer.write(vis)
        else:
            png_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(png_dir / f"{idx:04d}.png"), vis)

        # --- 間引きサムネ保存（Nフレームに1回だけ） ---
        if idx % THUMB_SKIP == 0:
            cv2.imwrite(str(out / f"mini_project_thumb_{thumbs_saved:03d}.png"), vis)
            thumbs_saved += 1

        idx += 1

    cap.release()
    if writer is not None:
        writer.release()

    # 1枚目は dt が不安定なので 2枚目以降で代表値を取る。
    body = fps_history[1:] if len(fps_history) > 1 else fps_history
    stats = {
        "frames_processed": idx,
        "thumbnails_saved": thumbs_saved,
        "thumb_skip": THUMB_SKIP,
        "motion_threshold": MOTION_THRESHOLD,
        "mean_processing_fps": round(float(np.mean(body)), 1) if body else 0.0,
        "min_processing_fps": round(float(np.min(body)), 1) if body else 0.0,
        "total_motion_pixels": int(sum(motion_counts)),
        "rewrite_mode": rewrite_mode,
    }
    print("\n[2] 動体ハイライト＆再書き出し")
    print(f"  処理フレーム数={idx}  間引きサムネ={thumbs_saved}枚  再書き出し={rewrite_mode}")
    print(f"  処理FPS(平均)≈{stats['mean_processing_fps']:,.0f}  "
          f"（ソースFPS {SRC_FPS:g} を上回れば実時間に追いつける）")
    print(f"  動体画素 合計={stats['total_motion_pixels']:,}（2物体が動くので 0 にならない）")
    return stats, fps_history, motion_counts


# =====================================================================
# 4) シーク・モンタージュ（POS_FRAMES でファイルの好きな位置へ飛ぶ）
# =====================================================================

def build_seek_montage(source: str, out: pathlib.Path) -> bool:
    """先頭・1/4・中央・3/4・終端へシークし、原フレームを横並びにして保存する。"""
    indices = [0, NUM_FRAMES // 4, NUM_FRAMES // 2, 3 * NUM_FRAMES // 4, NUM_FRAMES - 1]
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        return False

    grabbed: list[tuple[int, np.ndarray]] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)  # 次に read() する位置を idx にする
        ret, frame = cap.read()
        if ret:
            grabbed.append((idx, frame))
    cap.release()

    if not grabbed:
        print("\n[3] シーク不可（画像シーケンス等）。モンタージュはスキップ。")
        return False

    fig, axes = plt.subplots(1, len(grabbed), figsize=(2.6 * len(grabbed), 2.6))
    if len(grabbed) == 1:
        axes = [axes]
    for ax, (idx, frame) in zip(axes, grabbed):
        ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # BGR→RGB を忘れない
        ax.set_title(f"seek {idx}")
        ax.axis("off")
    fig.suptitle("POS_FRAMES seek montage (circle moves L->R, square R->L)")
    fig.tight_layout()
    fig.savefig(str(out / "mini_project_seek_montage.png"), dpi=110)
    plt.close(fig)
    print("\n[3] シーク・モンタージュ → mini_project_seek_montage.png")
    return True


# =====================================================================
# 5) レポート図（処理FPS と 動体画素の推移）
# =====================================================================

def plot_report(fps_history: list[float], motion_counts: list[int], out: pathlib.Path) -> None:
    """上段=処理FPS vs ソースFPS、下段=フレームごとの動体画素数 の 2 段グラフ。"""
    if not fps_history:
        return
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.2, 5.2))

    ax1.plot(range(len(fps_history)), fps_history, color="tab:blue", label="processing FPS")
    ax1.axhline(SRC_FPS, color="tab:red", linestyle="--", label=f"source FPS ({SRC_FPS:g})")
    ax1.set_ylabel("FPS")
    ax1.set_title("processing FPS vs source FPS")
    ax1.legend(loc="best")

    ax2.plot(range(len(motion_counts)), motion_counts, color="tab:green")
    ax2.set_xlabel("frame index")
    ax2.set_ylabel("motion pixels")
    ax2.set_title("moving pixels per frame (frame differencing)")

    fig.tight_layout()
    fig.savefig(str(out / "mini_project_report.png"), dpi=110)
    plt.close(fig)
    print("  → mini_project_report.png（処理FPSと動体画素の推移）")


# =====================================================================
# 6) JSON レポート
# =====================================================================

def write_json_report(out: pathlib.Path, source: str, src_mode: str,
                      meta: dict, stats: dict, has_montage: bool) -> pathlib.Path:
    """メタデータ＋処理統計＋出力一覧を 1 つの JSON にまとめて保存する。"""
    report = {
        "module_id": "09_video_io_basics",
        "project": "motion-highlight video pipeline",
        "source": {
            "path": pathlib.Path(source).name,
            "write_mode": src_mode,
            "requested_fps": SRC_FPS,
            "frames_synthesized": NUM_FRAMES,
            "size": [WIDTH, HEIGHT],
        },
        "metadata": meta,
        "processing": stats,
        "outputs": sorted(p.name for p in out.glob("mini_project_*")
                          if p.is_file()),
        "seek_montage": has_montage,
    }
    path = out / "mini_project_report.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[4] JSONレポート → {path.name}")
    return path


def main() -> None:
    out = output_dir()
    print("=== 第9回 章末ミニプロジェクト: 動体ハイライト動画パイプライン ===\n")

    source, src_mode = synth_source_video(out)
    print(f"合成ソース動画: {pathlib.Path(source).name} (mode={src_mode})\n")

    meta = probe_metadata(source)
    stats, fps_history, motion_counts = process_pipeline(source, out)
    has_montage = build_seek_montage(source, out)
    plot_report(fps_history, motion_counts, out)
    write_json_report(out, source, src_mode, meta, stats, has_montage)

    print("\n完了。outputs/09_video_io_basics/ の mini_project_* を確認してください。")


if __name__ == "__main__":
    main()
