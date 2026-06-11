"""04: ネットワークストリーム（RTSP / ライブ配信）— 低遅延設定と再接続ループ。

ネットワーク越しの映像（防犯カメラの RTSP、ライブ配信）は、ローカルファイルと違って
『途中で切れる』のが普通。だから本番のストリームアプリは、read() 失敗を即クラッシュにせず、
『開き直して続ける』再接続ループとして書く。本スクリプトは:

  - 既定では『たまに切断する合成ソース』で再接続ループとバックオフを実演する（ネット/カメラ不要）。
  - RTSP の低遅延設定（CAP_FFMPEG ＋ rtsp_transport=tcp ＋ CAP_PROP_BUFFERSIZE=1）の
    正準コードを示す（CV_RTSP=<url> を指定したときだけ実際に接続）。
  - yt-dlp でライブ配信URLを直URLに解決して VideoCapture へ渡す手順を示す
    （CV_YOUTUBE=<url> かつ yt-dlp がある場合だけ実行）。
  - Webカメラ経路は CV_CAM=1 のときだけ（前スクリプト同様）。

いずれの外部接続も既定では行わないので、引数なしで必ず exit 0 になる。

実行（合成ソースで再接続を実演）:
  uv run python lectures/11_realtime_stream/04_rtsp_youtube_stream.py
実カメラ/RTSP/ライブ配信を使う場合（任意・ネット必要）:
  CV_CAM=1     uv run python lectures/11_realtime_stream/04_rtsp_youtube_stream.py
  CV_RTSP=rtsp://...  uv run python lectures/11_realtime_stream/04_rtsp_youtube_stream.py
  CV_YOUTUBE=https://...  uv run python lectures/11_realtime_stream/04_rtsp_youtube_stream.py
"""

from __future__ import annotations

import os
import pathlib
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from stream_helpers import FpsMeter, output_dir, synthetic_frames  # noqa: E402


class FlakySource:
    """『たまに切断する』ネットワークソースを模擬する（再接続ループの練習台）。

    通常は (True, frame) を返すが、指定した区間（fail_windows）では (False, None) を返して
    切断を再現する。open()/release() を持ち、本物の VideoCapture と同じ感覚で扱える。
    再接続しても切断区間が過ぎるまでは失敗し続ける＝『復旧待ち』を体験できる。
    """

    def __init__(
        self,
        num_frames: int = 200,
        size: tuple[int, int] = (360, 640),
        seed: int = 3,
        fail_starts: tuple[int, ...] = (25, 70),
        fail_len: int = 8,
    ) -> None:
        self._gen = synthetic_frames(num_frames, size, seed)
        self._opened = False
        self._fail_windows = [(s, s + fail_len) for s in fail_starts]

    def open(self) -> bool:
        """接続を開く（本物なら VideoCapture を作る箇所）。成功で True。"""
        self._opened = True
        return True

    def read(self) -> tuple[bool, np.ndarray | None]:
        if not self._opened:
            return False, None
        try:
            idx, frame = next(self._gen)
        except StopIteration:
            return False, None
        for (a, b) in self._fail_windows:
            if a <= idx < b:
                return False, None      # 切断区間: フレームが取れない
        return True, frame

    def release(self) -> None:
        self._opened = False


def reconnecting_loop(
    source: FlakySource,
    out_dir: pathlib.Path,
    target_good: int = 110,
    give_up_after: int = 20,
) -> dict:
    """再接続ループ本体: read() が失敗したら release→バックオフ→open し直して継続する。

    要点:
      - 1枚読めないだけで落とさない。ネットワークは切れて当然、という前提で書く。
      - 連続失敗時はバックオフ（待ち時間を少しずつ延ばす）で再接続の連打を避ける。
      - 復旧したらバックオフをリセット。背景差分で動体を検出しながら良フレームを処理する。
      - ライブ入力には『終端』が無いので、良フレームを target_good 枚さばけたら正常終了。
        逆に give_up_after 回連続で失敗したら『ストリーム断』と判断して諦める（無限リトライ回避）。
    返り値は集計（良フレーム数・再接続回数・処理FPS）。
    """
    bgsub = cv2.createBackgroundSubtractorMOG2(history=120, varThreshold=24, detectShadows=False)
    meter = FpsMeter(alpha=0.2)
    good = 0
    reconnects = 0
    consecutive_fail = 0     # 連続失敗回数（良フレームが来たらリセット）
    backoff = 0.01           # 初期バックオフ秒（学習用に短く）
    saved_sample = False

    source.open()
    while good < target_good:
        ret, frame = source.read()
        if not ret:
            # --- 切断検知 → 再接続シーケンス ---
            source.release()
            reconnects += 1
            consecutive_fail += 1
            if consecutive_fail == 1:
                print("  [warn] 接続が切れました。バックオフしながら再接続を試みます…")
            if consecutive_fail >= give_up_after:
                print(f"  [error] {give_up_after}回連続で失敗。ストリーム断と判断して終了します。")
                break
            time.sleep(backoff)
            backoff = min(backoff * 2, 0.2)     # 指数バックオフ（上限つき）
            source.open()                       # 本物なら VideoCapture を作り直す箇所
            continue

        # --- 正常フレーム: 復旧したのでバックオフ等をリセットして処理 ---
        if consecutive_fail > 0:
            print(f"  [ok] {consecutive_fail} 回の再接続試行ののち復旧しました。処理を継続します。")
        consecutive_fail = 0
        backoff = 0.01
        fg = bgsub.apply(frame)
        meter.tick()
        good += 1

        if good == 30 and not saved_sample:
            # 1枚だけ『入力＋前景マスク』を保存して、処理が回っている証拠を残す。
            vis = np.hstack([frame, cv2.cvtColor(fg, cv2.COLOR_GRAY2BGR)])
            cv2.putText(vis, f"good frame #{good}", (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imwrite(str(out_dir / "04_stream_sample.png"), vis)
            saved_sample = True

    source.release()
    return {"good": good, "reconnects": reconnects, "fps": meter.fps}


# =====================================================================
# 実接続パス（既定では呼ばれない。環境変数で明示したときだけ実行）
# =====================================================================

def open_rtsp(url: str) -> cv2.VideoCapture:
    """RTSP を低遅延で開く正準コード（CV_RTSP=<url> のときだけ使う）。

    要点:
      - OPENCV_FFMPEG_CAPTURE_OPTIONS=rtsp_transport;tcp で UDP のパケロスを避け TCP に固定。
        （この環境変数は VideoCapture を作る『前』に設定する必要がある）
      - cv2.CAP_FFMPEG バックエンドを明示する。
      - CAP_PROP_BUFFERSIZE=1 で内部バッファを最小化し、古いフレームの溜め込み＝遅延を抑える。
    """
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)     # 溜めない（最新に追従して低遅延化）
    return cap


def resolve_youtube(url: str) -> str | None:
    """yt-dlp でライブ配信URLを『直接の配信URL』に解決する（CV_YOUTUBE=<url> のときだけ）。

    yt-dlp -g は、ページURLから VideoCapture に渡せるストリームURLを取り出す。
    yt-dlp 未インストールや解決失敗なら None を返す（呼び出し側でスキップ）。
    ※ yt-dlp はサイト変更で壊れやすいので、本番では定期更新が前提。
    """
    import shutil
    import subprocess

    if shutil.which("yt-dlp") is None:
        print("  [skip] yt-dlp が見つかりません（uv add --group video yt-dlp 等で導入）。")
        return None
    try:
        # -f は『映像つきの手頃な形式』。-g で直URLだけを出力させる。
        proc = subprocess.run(
            ["yt-dlp", "-g", "-f", "best[height<=480]/best", url],
            capture_output=True, text=True, timeout=30, check=True,
        )
        direct = proc.stdout.strip().splitlines()
        return direct[0] if direct else None
    except Exception as e:  # noqa: BLE001  失敗してもアプリは落とさない
        print(f"  [skip] yt-dlp 解決に失敗: {type(e).__name__}: {e}")
        return None


def run_real_capture(cap: cv2.VideoCapture, label: str, out_dir: pathlib.Path, limit: int = 200) -> None:
    """実 VideoCapture（カメラ/RTSP/ライブ）を read で回す最小ループ（再接続なしの簡易版）。"""
    if not cap.isOpened():
        print(f"  [error] {label} を開けませんでした。スキップします。")
        return
    meter = FpsMeter(alpha=0.2)
    n = 0
    while n < limit:
        ret, frame = cap.read()
        if not ret:
            print(f"  [info] {label}: フレーム取得が途切れたので終了します。")
            break
        meter.tick()
        n += 1
        if n == 20:
            cv2.imwrite(str(out_dir / "04_real_sample.png"), frame)
    cap.release()
    print(f"  [done] {label}: {n} フレーム処理（処理FPS≈{meter.fps:.1f}）。")


def main() -> None:
    out = output_dir()

    # --- 実接続パス（明示時のみ）---
    if os.environ.get("CV_CAM") == "1":
        print("[real] Webカメラ(0) を使用します（CV_CAM=1）")
        run_real_capture(cv2.VideoCapture(0), "webcam", out)
        return

    rtsp_url = os.environ.get("CV_RTSP")
    if rtsp_url:
        print(f"[real] RTSP に接続します（低遅延設定）: {rtsp_url}")
        run_real_capture(open_rtsp(rtsp_url), "rtsp", out)
        return

    yt_url = os.environ.get("CV_YOUTUBE")
    if yt_url:
        print(f"[real] ライブ配信URLを yt-dlp で解決します: {yt_url}")
        direct = resolve_youtube(yt_url)
        if direct is None:
            print("[real] 解決できなかったので合成デモにフォールバックします。")
        else:
            run_real_capture(cv2.VideoCapture(direct, cv2.CAP_FFMPEG), "youtube-live", out)
            return

    # --- 既定: 合成の『切断するソース』で再接続ループを実演 ---
    print("[demo] 合成ソースで再接続ループを実演します（ネット/カメラ不要）。")
    print("       途中2回わざと切断し、バックオフ付きで再接続して処理を続けます。")
    stats = reconnecting_loop(FlakySource(), out)
    print("\n=== 再接続デモのまとめ ===")
    print(f"  良フレーム処理数 : {stats['good']}")
    print(f"  再接続回数       : {stats['reconnects']}")
    print(f"  処理FPS(EMA)     : {stats['fps']:.1f}")
    print("  → 切断が起きても落ちずに復旧して処理を継続できた（本番ストリームの必須要件）。")
    print("\n完了。outputs/11_realtime_stream/ に 04_stream_sample.png を保存しました。")


if __name__ == "__main__":
    main()
