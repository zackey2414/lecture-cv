"""第11回 演習問題（リアルタイム・ストリーム処理）。

使い方:
  1. 各 exN_*() 関数の中の TODO を自分で実装する（最初は NotImplementedError が出るが、
     採点ランナーが拾うのでプロセスは落ちず、FAIL と表示されるだけ）。
  2. 実装できたら自己採点を実行:
         uv run python lectures/11_realtime_stream/exercises.py
     全問 pass すれば "ALL PASS" と表示される。
  3. どうしても分からない時は、模範解答の挙動を見る:
         SHOW_SOLUTION=1 uv run python lectures/11_realtime_stream/exercises.py

この5問は、本モジュールの4スクリプトの核を1つずつ抜き出したもの:
  ex1: 背景差分マスクの掃除（影127除去＋モルフォロジー）        … 01
  ex2: 早期縮小（アスペクト比を保って最大辺を縮める）           … 02
  ex3: キュー満杯時のフレームドロップ（put_nowait）             … 03
  ex4: 処理FPSの指数移動平均(EMA)                               … helper / 全体
  ex5: 再接続ループの集計（連続失敗で諦める）                   … 04
"""

from __future__ import annotations

import os
import pathlib
import queue
import sys

import cv2
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))


# =====================================================================
# 演習（ここを実装する）
# =====================================================================

def ex1_clean_mask(raw_mask: np.ndarray) -> np.ndarray:
    """演習1: 生の前景マスクを 0/255 の2値マスクへ掃除する。

    手順（この通りに実装すれば模範解答と一致する）:
      1. cv2.threshold(raw, 200, 255, THRESH_BINARY) で影(127)を捨て、255 だけ残す。
      2. 5x5 楕円(MORPH_ELLIPSE)の構造要素を作る。
      3. MORPH_OPEN を iterations=1（ぽつぽつノイズ除去）。
      4. 続けて MORPH_CLOSE を iterations=2（物体内部の穴埋め）。
    返り値は uint8 で値が 0 か 255 のみ。
    """
    # TODO: 上の手順を実装する
    raise NotImplementedError


def ex2_resize_keep_aspect(frame: np.ndarray, max_side: int) -> np.ndarray:
    """演習2: アスペクト比を保ったまま、長辺が max_side 以下になるよう縮小する。

    - 既に長辺 <= max_side なら、そのまま返す（拡大はしない）。
    - 縮小の補間は cv2.INTER_AREA（モアレが出にくい定石）。
    - cv2.resize の dsize は (幅, 高さ) の順である点に注意（shape は (高さ, 幅)）。
    """
    # TODO: scale = max_side / max(h, w) を計算し、scale < 1 のときだけ INTER_AREA で縮小して返す
    raise NotImplementedError


def ex3_drop_when_full(items: list, maxsize: int) -> int:
    """演習3: maxsize のキューへ items を put_nowait で詰め、満杯で落ちた数を返す。

    消費者はいない前提。queue.Queue(maxsize) を作り、各 item を put_nowait で入れる。
    queue.Full 例外が出たら『ドロップ』としてカウントする。ドロップ総数を返すこと。
    （maxsize=1 で items が5個なら、最初の1個だけ入り、残り4個がドロップ＝4を返す）
    """
    # TODO: q = queue.Queue(maxsize) を作り、for item in items: try put_nowait / except queue.Full: dropped++
    raise NotImplementedError


def ex4_ema_fps(timestamps: list[float], alpha: float) -> float:
    """演習4: フレーム処理時刻の列から、処理FPSの指数移動平均(EMA)を計算して返す。

    - 隣り合う時刻の差 dt = t[i]-t[i-1] から瞬間FPS = 1/dt を出す（dt<=0 はスキップ）。
    - EMA: 最初の値はそのまま、以降は ema = (1-alpha)*ema + alpha*fps。
    - timestamps が2個未満なら 0.0 を返す。
    """
    # TODO: 上の漸化式で EMA を更新し、最終値を返す
    raise NotImplementedError


def ex5_reconnect_summary(read_results: list[bool], give_up_after: int) -> tuple[int, int]:
    """演習5: read 結果(True=成功/False=失敗)の列を再接続ループとして集計する。

    - True なら good += 1、連続失敗カウンタを 0 に戻す。
    - False なら reconnects += 1、連続失敗カウンタ += 1。
      連続失敗が give_up_after に達したら、その時点で打ち切って終了する。
    - 返り値は (good, reconnects)。
    """
    # TODO: good / reconnects / consecutive を回し、consecutive>=give_up_after で break
    raise NotImplementedError


# =====================================================================
# 自己採点ランナー
# =====================================================================

def _make_raw_mask() -> np.ndarray:
    """採点用の生マスクを作る: 大きな前景(255)＋影(127)＋孤立ノイズ(255)。"""
    m = np.zeros((80, 120), dtype=np.uint8)
    m[20:60, 30:80] = 255       # 大きな物体（残ってほしい）
    m[10:20, 90:110] = 127      # 影（127。捨てたい）
    m[5, 5] = 255               # 孤立ノイズ（open で消えてほしい）
    m[70, 100] = 255
    return m


def _grade() -> None:
    results: list[tuple[str, bool, str]] = []

    def check(name: str, fn) -> None:
        try:
            ok, detail = fn()
            results.append((name, bool(ok), detail))
        except NotImplementedError:
            results.append((name, False, "未実装（TODOを埋めてください）"))
        except Exception as e:  # noqa: BLE001
            results.append((name, False, f"例外: {type(e).__name__}: {e}"))

    # ex1: 模範解答と完全一致するか（手順が指定どおりなら一致するはず）。
    def _c1():
        raw = _make_raw_mask()
        got = ex1_clean_mask(raw)
        ref = _sol_ex1(raw)
        same = isinstance(got, np.ndarray) and got.dtype == np.uint8 and np.array_equal(got, ref)
        # 補足チェック: 影(127)領域が 0 に、値が 0/255 のみ
        shadow_cleared = bool((got[10:20, 90:110] == 0).all()) if isinstance(got, np.ndarray) else False
        return same and shadow_cleared, "影除去＋モルフォロジー"
    check("ex1_clean_mask", _c1)

    # ex2: 540x960 を max_side=320 に → 長辺320・アスペクト比維持。
    def _c2():
        frame = np.zeros((540, 960, 3), dtype=np.uint8)
        got = ex2_resize_keep_aspect(frame, 320)
        ok = got.shape[1] == 320 and got.shape[0] == 180   # 960:540 = 16:9 → 320x180
        # 既に小さい場合はそのまま返す
        small = np.zeros((100, 150, 3), dtype=np.uint8)
        got2 = ex2_resize_keep_aspect(small, 320)
        ok2 = got2.shape[:2] == (100, 150)
        return ok and ok2, "アスペクト比維持の縮小"
    check("ex2_resize_keep_aspect", _c2)

    # ex3: maxsize=1 に5個 → 4ドロップ。maxsize=3 に5個 → 2ドロップ。
    def _c3():
        d1 = ex3_drop_when_full([1, 2, 3, 4, 5], maxsize=1)
        d3 = ex3_drop_when_full([1, 2, 3, 4, 5], maxsize=3)
        return d1 == 4 and d3 == 2, "put_nowait ドロップ計数"
    check("ex3_drop_when_full", _c3)

    # ex4: 等間隔(0.05s刻み)なら EMA は 20fps に収束する。
    def _c4():
        ts = [i * 0.05 for i in range(10)]
        got = ex4_ema_fps(ts, alpha=0.3)
        ref = _sol_ex4(ts, 0.3)
        short = ex4_ema_fps([0.0], 0.3)
        return abs(got - ref) < 1e-6 and short == 0.0, f"EMA≈{got:.2f}fps"
    check("ex4_ema_fps", _c4)

    # ex5: 成功3・失敗2(連続)・成功2、give_up_after=5 → (5, 2)。
    #      連続失敗で打ち切り: [T,F,F,F]，give_up_after=3 → (1, 3) で終了。
    def _c5():
        a = ex5_reconnect_summary([True, True, True, False, False, True, True], give_up_after=5)
        b = ex5_reconnect_summary([True, False, False, False, True, True], give_up_after=3)
        return a == (5, 2) and b == (1, 3), f"集計={a}"
    check("ex5_reconnect_summary", _c5)

    print("=== 採点結果 ===")
    all_ok = True
    for name, ok, detail in results:
        mark = "PASS" if ok else "FAIL"
        all_ok = all_ok and ok
        print(f"  [{mark}] {name:22s} {detail}")
    print("\nALL PASS 🎉" if all_ok else "\nまだ未達の演習があります。TODO を埋めましょう。")


# =====================================================================
# 模範解答（SHOW_SOLUTION=1 のときに本体へ差し替えて実行）
# まずは自力で解いてから見ること。
# =====================================================================

def _sol_ex1(raw_mask: np.ndarray) -> np.ndarray:
    _, binary = cv2.threshold(raw_mask, 200, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    return cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=2)


def _sol_ex2(frame: np.ndarray, max_side: int) -> np.ndarray:
    h, w = frame.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1.0:
        return frame
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _sol_ex3(items: list, maxsize: int) -> int:
    q: "queue.Queue" = queue.Queue(maxsize=maxsize)
    dropped = 0
    for item in items:
        try:
            q.put_nowait(item)
        except queue.Full:
            dropped += 1
    return dropped


def _sol_ex4(timestamps: list[float], alpha: float) -> float:
    if len(timestamps) < 2:
        return 0.0
    ema: float | None = None
    for i in range(1, len(timestamps)):
        dt = timestamps[i] - timestamps[i - 1]
        if dt <= 0:
            continue
        fps = 1.0 / dt
        ema = fps if ema is None else (1 - alpha) * ema + alpha * fps
    return ema or 0.0


def _sol_ex5(read_results: list[bool], give_up_after: int) -> tuple[int, int]:
    good = 0
    reconnects = 0
    consecutive = 0
    for ok in read_results:
        if ok:
            good += 1
            consecutive = 0
        else:
            reconnects += 1
            consecutive += 1
            if consecutive >= give_up_after:
                break
    return good, reconnects


def _install_solutions() -> None:
    """模範解答で TODO 関数を差し替える（教材検証・答え合わせ用）。"""
    g = globals()
    g["ex1_clean_mask"] = _sol_ex1
    g["ex2_resize_keep_aspect"] = _sol_ex2
    g["ex3_drop_when_full"] = _sol_ex3
    g["ex4_ema_fps"] = _sol_ex4
    g["ex5_reconnect_summary"] = _sol_ex5


if __name__ == "__main__":
    if os.environ.get("SHOW_SOLUTION") == "1":
        print("(模範解答モードで実行します)\n")
        _install_solutions()
    _grade()
