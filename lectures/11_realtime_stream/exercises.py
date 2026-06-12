"""第11回 演習問題（リアルタイム・ストリーム処理）。

使い方:
  1. 各 exN_*() 関数の中の TODO を自分で実装する（最初は NotImplementedError が出るが、
     採点ランナーが拾うのでプロセスは落ちず、FAIL と表示されるだけ）。
  2. 実装できたら自己採点を実行:
         uv run python lectures/11_realtime_stream/exercises.py
     全問 pass すれば "ALL PASS" と表示される。
  3. どうしても分からない時は、模範解答の挙動を見る:
         SHOW_SOLUTION=1 uv run python lectures/11_realtime_stream/exercises.py
     （完全な解答ファイルは exercises_solutions.py にもある）

この10問は易→難。本モジュールの4スクリプト＋プロファイルの核を1つずつ抜き出した:
  ex1  背景差分マスクの掃除（影127除去＋モルフォロジー）          … 01
  ex2  早期縮小（アスペクト比を保って最大辺を縮める）             … 02
  ex3  キュー満杯時のフレームドロップ（put_nowait）               … 03
  ex4  処理FPSの指数移動平均(EMA)                                 … helper / 全体
  ex5  再接続ループの集計（連続失敗で諦める）                     … 04
  ex6  レイテンシのパーセンタイル(p50/p99)                        … プロファイル
  ex7  フレームスキップで実際に処理する枚数                       … 02
  ex8  前景マスクから動体の個数を数える（輪郭＋面積しきい値）     … 01
  ex9  指数バックオフのスケジュール（上限つき）                   … 04
  ex10 maxsize 付きキューの再生（put_nowait ドロップ＋FIFO get）  … 03（総合）
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


def ex6_latency_percentiles(latencies_ms: list[float]) -> tuple[float, float]:
    """演習6: レイテンシ(ミリ秒)のリストから (p50, p99) を返す。

    - 平均だけ見ると『たまの詰まり』を見落とすので p99（上位1%の遅さ）も測る。
    - np.percentile(arr, 50) と np.percentile(arr, 99) を使い、float で返す。
    - 空リストなら (0.0, 0.0) を返す。
    """
    # TODO: np.asarray にして np.percentile で 50 と 99 を計算し (p50, p99) を返す
    raise NotImplementedError


def ex7_processed_count(num_seen: int, every_n: int) -> int:
    """演習7: 「N枚に1回だけ処理する」フレームスキップで、実際に処理する枚数を返す。

    - 1..num_seen のうち seen % every_n == 0 となる回数（= num_seen // every_n）。
    - every_n=1 なら全部処理（= num_seen）、every_n=3・num_seen=120 なら 40。
    - every_n <= 0 は不正なので ValueError を投げる。
    """
    # TODO: every_n<=0 をはじき、num_seen // every_n を返す
    raise NotImplementedError


def ex8_count_motion_boxes(mask: np.ndarray, min_area: int) -> int:
    """演習8: 2値の前景マスクから、面積 min_area 以上の動体の個数を数える。

    - cv2.findContours(mask, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE) で輪郭を取る
      （OpenCV 4 系は (contours, hierarchy) の2つ返し）。
    - cv2.contourArea(c) >= min_area の輪郭だけ数える（小さいノイズは無視）。
    - 返り値はその個数(int)。
    """
    # TODO: findContours → contourArea でしきい値以上の輪郭をカウントして返す
    raise NotImplementedError


def ex9_backoff_schedule(num_failures: int, initial: float, factor: float, cap: float) -> list[float]:
    """演習9: 連続失敗 num_failures 回ぶんの指数バックオフ待ち時間リストを返す。

    - i 回目(0 始まり)の待ち時間は min(initial * factor**i, cap)。
    - 返り値は長さ num_failures の float リスト。num_failures<=0 なら空リスト。
    例: backoff_schedule(5, 0.01, 2.0, 0.2) -> [0.01, 0.02, 0.04, 0.08, 0.16]
        6個目は 0.32 だが cap=0.2 で頭打ちになり 0.2。
    """
    # TODO: for i in range(num_failures): min(initial*factor**i, cap) を積む
    raise NotImplementedError


def ex10_queue_replay(events: list[tuple], maxsize: int) -> dict:
    """演習10: maxsize 付きキューの動作を再生する（put_nowait ドロップ＋FIFO get）。

    events は次のいずれかのタプルの列:
      ("put", value) : value を入れる。満杯なら put_nowait のように『新しい方を捨てる』(drop++)。
      ("get",)       : 最も古い要素を取り出して outputs に追加。空なら None を outputs に追加。
    返り値は {"outputs": [...], "dropped": int}。
    （FIFO=先入れ先出し。queue.Queue(maxsize) と put_nowait/get_nowait で実装してよい）
    """
    # TODO: q=queue.Queue(maxsize)。put は満杯で queue.Full を握って drop++、get は空で None を積む
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


def _make_two_blob_mask() -> np.ndarray:
    """採点用の2値マスク: 大きなブロブ2個（残す）＋小さな点1個（無視させる）。"""
    m = np.zeros((100, 160), dtype=np.uint8)
    m[20:60, 20:70] = 255    # ブロブ1（面積 50*40=2000）
    m[20:55, 100:140] = 255  # ブロブ2（面積 40*35=1400）
    m[90, 150] = 255         # 微小ノイズ（面積≒1）
    return m


def _grade() -> bool:
    """全演習を採点して結果を表示し、全問 PASS なら True を返す。

    exercises_solutions.py はこの関数をそのまま再利用する（採点ロジックを重複させない）。
    """
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

    # ex6: 0..100 を 1 刻みで並べた列なら p50≈50.0・p99≈99.0。空なら (0,0)。
    def _c6():
        vals = [float(i) for i in range(101)]
        got = ex6_latency_percentiles(vals)
        ref = _sol_ex6(vals)
        empty = ex6_latency_percentiles([])
        close = abs(got[0] - ref[0]) < 1e-6 and abs(got[1] - ref[1]) < 1e-6
        return close and empty == (0.0, 0.0), f"p50={got[0]:.1f} p99={got[1]:.1f}"
    check("ex6_latency_percentiles", _c6)

    # ex7: (120,3)->40, (10,1)->10, (7,3)->2。every_n<=0 は ValueError。
    def _c7():
        ok = ex7_processed_count(120, 3) == 40 and ex7_processed_count(10, 1) == 10 \
            and ex7_processed_count(7, 3) == 2
        raised = False
        try:
            ex7_processed_count(10, 0)
        except ValueError:
            raised = True
        return ok and raised, "frameskip 処理枚数"
    check("ex7_processed_count", _c7)

    # ex8: 大ブロブ2個（min_area=100）→ 2。min_area=1900 にすると面積1400のブロブが落ちて 1。
    def _c8():
        mask = _make_two_blob_mask()
        n2 = ex8_count_motion_boxes(mask, min_area=100)
        n1 = ex8_count_motion_boxes(mask, min_area=1900)
        return n2 == 2 and n1 == 1, f"検出数={n2}(min100)/{n1}(min1900)"
    check("ex8_count_motion_boxes", _c8)

    # ex9: (5,0.01,2,0.2) -> [.01,.02,.04,.08,.16]。6個目は cap=0.2。空は []。
    def _c9():
        got = ex9_backoff_schedule(5, 0.01, 2.0, 0.2)
        ref = _sol_ex9(5, 0.01, 2.0, 0.2)
        capped = ex9_backoff_schedule(6, 0.01, 2.0, 0.2)
        empty = ex9_backoff_schedule(0, 0.01, 2.0, 0.2)
        ok = len(got) == 5 and all(abs(a - b) < 1e-9 for a, b in zip(got, ref))
        return ok and abs(capped[5] - 0.2) < 1e-9 and empty == [], f"先頭={got[:3]}"
    check("ex9_backoff_schedule", _c9)

    # ex10: maxsize=1 で put1,put2(drop),get->1,put3,put4(drop),get->3,get->None,get->None。
    def _c10():
        events = [("put", 1), ("put", 2), ("get",), ("put", 3), ("put", 4),
                  ("get",), ("get",), ("get",)]
        got = ex10_queue_replay(events, maxsize=1)
        ref = _sol_ex10(events, 1)
        return got == ref and got == {"outputs": [1, 3, None, None], "dropped": 2}, \
            f"outputs={got.get('outputs')} drop={got.get('dropped')}"
    check("ex10_queue_replay", _c10)

    print("=== 採点結果 ===")
    all_ok = True
    for name, ok, detail in results:
        mark = "PASS" if ok else "FAIL"
        all_ok = all_ok and ok
        print(f"  [{mark}] {name:24s} {detail}")
    print("\nALL PASS 🎉" if all_ok else "\nまだ未達の演習があります。TODO を埋めましょう。")
    return all_ok


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


def _sol_ex6(latencies_ms: list[float]) -> tuple[float, float]:
    if not latencies_ms:
        return 0.0, 0.0
    arr = np.asarray(latencies_ms, dtype=np.float64)
    return float(np.percentile(arr, 50)), float(np.percentile(arr, 99))


def _sol_ex7(num_seen: int, every_n: int) -> int:
    if every_n <= 0:
        raise ValueError("every_n must be >= 1")
    return num_seen // every_n


def _sol_ex8(mask: np.ndarray, min_area: int) -> int:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return sum(1 for c in contours if cv2.contourArea(c) >= min_area)


def _sol_ex9(num_failures: int, initial: float, factor: float, cap: float) -> list[float]:
    return [min(initial * (factor ** i), cap) for i in range(num_failures)]


def _sol_ex10(events: list[tuple], maxsize: int) -> dict:
    q: "queue.Queue" = queue.Queue(maxsize=maxsize)
    outputs: list = []
    dropped = 0
    for ev in events:
        if ev[0] == "put":
            try:
                q.put_nowait(ev[1])
            except queue.Full:
                dropped += 1          # 満杯 → 新しい方を捨てる（put_nowait の挙動）
        else:  # ("get",)
            try:
                outputs.append(q.get_nowait())
            except queue.Empty:
                outputs.append(None)  # 空 → None を積む
    return {"outputs": outputs, "dropped": dropped}


def _install_solutions() -> None:
    """模範解答で TODO 関数を差し替える（教材検証・答え合わせ用）。"""
    g = globals()
    g["ex1_clean_mask"] = _sol_ex1
    g["ex2_resize_keep_aspect"] = _sol_ex2
    g["ex3_drop_when_full"] = _sol_ex3
    g["ex4_ema_fps"] = _sol_ex4
    g["ex5_reconnect_summary"] = _sol_ex5
    g["ex6_latency_percentiles"] = _sol_ex6
    g["ex7_processed_count"] = _sol_ex7
    g["ex8_count_motion_boxes"] = _sol_ex8
    g["ex9_backoff_schedule"] = _sol_ex9
    g["ex10_queue_replay"] = _sol_ex10


if __name__ == "__main__":
    if os.environ.get("SHOW_SOLUTION") == "1":
        print("(模範解答モードで実行します)\n")
        _install_solutions()
    _grade()
