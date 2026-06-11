"""第9回 演習問題（動画I/Oの基礎）。

使い方:
  1. 各 exN_*() 関数の中の TODO を自分で実装する（最初は NotImplementedError が出る）。
  2. 実装できたら自己採点を実行:
         uv run python lectures/09_video_io_basics/exercises.py
     全問が PASS すれば "ALL PASS" と表示される。
  3. どうしても分からない時は、模範解答を実行して挙動を確認する:
         SHOW_SOLUTION=1 uv run python lectures/09_video_io_basics/exercises.py
     （まずは自力で！）

注意: 未実装でも例外で落とさず PASS/FAIL を表示して正常終了（exit 0）します。
ヒント: 採点用の合成動画は同じフォルダの cv_helpers.make_demo_video() で作っています。
"""

from __future__ import annotations

import os
import pathlib
import sys
from collections import deque

import cv2
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from cv_helpers import make_demo_video, output_dir  # noqa: E402


# =====================================================================
# 演習（ここを実装する）
# =====================================================================

def ex1_count_frames(source: str) -> int:
    """演習1: VideoCapture の正準ループでフレーム総数を数える。

    cv2.VideoCapture(source) を開き、while ループで read() し、ret=False で抜ける。
    数えた枚数を返し、最後に必ず release() すること（総フレーム数 cap.get に頼らない）。
    """
    # TODO: cap=cv2.VideoCapture(source); while True: ret,frame=cap.read();
    #       if not ret: break; count+=1 → cap.release(); return count
    raise NotImplementedError


def ex2_read_meta(source: str) -> tuple[float, int, int, int]:
    """演習2: cap.get で (fps, frame_count, width, height) を取得して返す。

    それぞれ cv2.CAP_PROP_FPS / FRAME_COUNT / FRAME_WIDTH / FRAME_HEIGHT を使う。
    frame_count / width / height は int に丸めること。fps は float のままで良い。
    """
    # TODO: cap=cv2.VideoCapture(source)
    #       fps=cap.get(cv2.CAP_PROP_FPS); count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) ...
    #       cap.release(); return (fps, count, width, height)
    raise NotImplementedError


def ex3_fourcc_to_str(fourcc_int: int) -> str:
    """演習3: CAP_PROP_FOURCC の 32bit 整数を 4 文字のコーデック名に復元する。

    fourcc は 4 つの ASCII 文字を 1 バイトずつ詰めた整数。下位 8bit から順に取り出す。
    例: cv2.VideoWriter_fourcc(*"mp4v") を渡すと "mp4v" が返る。
    """
    # TODO: return "".join(chr((int(fourcc_int) >> (8*i)) & 0xFF) for i in range(4))
    raise NotImplementedError


def ex4_seek_and_read(source: str, index: int) -> np.ndarray | None:
    """演習4: POS_FRAMES で index フレームへシークし、その1枚を返す（無ければ None）。

    cap.set(cv2.CAP_PROP_POS_FRAMES, index) してから read()。ret=False なら None を返す。
    最後に release() すること。
    """
    # TODO: cap=cv2.VideoCapture(source); cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    #       ret,frame=cap.read(); cap.release(); return frame if ret else None
    raise NotImplementedError


def ex5_moving_average_fps(dts: list[float], window: int = 20) -> float:
    """演習5: フレーム毎の処理時間 dts(秒) から、末尾 window 個の移動平均FPSを返す。

    deque(maxlen=window) に dt を順に入れ、最後に平均 dt の逆数(=FPS)を返す。
    dts が空、または平均が 0 のときは 0.0 を返すこと。
    """
    # TODO: buf=deque(maxlen=window); [buf.append(d) for d in dts]
    #       avg=sum(buf)/len(buf); return 1.0/avg if avg>0 else 0.0
    raise NotImplementedError


# =====================================================================
# 自己採点ランナー
# =====================================================================

def _grade() -> None:
    out = output_dir()
    # 採点用の合成動画を作る（決定的: fps=24, 60フレーム, 320x240）。
    source, fps, num, mode = make_demo_video(out, name="ex_grade", fps=24.0,
                                             num=60, width=320, height=240)

    results: list[tuple[str, bool, str]] = []

    def check(name: str, fn) -> None:
        try:
            ok, detail = fn()
            results.append((name, bool(ok), detail))
        except NotImplementedError:
            results.append((name, False, "未実装（TODOを埋めてください）"))
        except Exception as e:  # noqa: BLE001
            results.append((name, False, f"例外: {type(e).__name__}: {e}"))

    def _c1():
        n = ex1_count_frames(source)
        # コーデックにより ±1 のずれが出ることがあるため許容する。
        return (abs(int(n) - num) <= 1, f"count={n} (期待 {num})")
    check("ex1_count_frames", _c1)

    def _c2():
        meta = ex2_read_meta(source)
        f, c, w, h = meta
        ok = (abs(f - fps) < 1.0 and abs(int(c) - num) <= 1 and w == 320 and h == 240)
        return (ok, f"fps={f} count={c} {w}x{h}")
    check("ex2_read_meta", _c2)

    def _c3():
        code = cv2.VideoWriter_fourcc(*"mp4v")
        s = ex3_fourcc_to_str(code)
        return (s == "mp4v", f"decoded={s!r} (期待 'mp4v')")
    check("ex3_fourcc_to_str", _c3)

    def _c4():
        f0 = ex4_seek_and_read(source, 0)
        fk = ex4_seek_and_read(source, num // 2)
        if f0 is None or fk is None:
            return (False, "シーク結果が None（画像シーケンス環境ならスキップ可）")
        shape_ok = f0.shape == (240, 320, 3)
        differ = bool(np.any(f0.astype(int) - fk.astype(int)))  # 円が動くので差が出る
        return (shape_ok and differ, f"shape={f0.shape} 別フレームと差あり={differ}")
    check("ex4_seek_and_read", _c4)

    def _c5():
        dts = [0.01] * 50  # 0.01秒/フレーム → 100 fps になるはず
        got = ex5_moving_average_fps(dts, window=20)
        empty = ex5_moving_average_fps([], window=20)
        ok = abs(got - 100.0) < 1e-6 and empty == 0.0
        return (ok, f"fps={got:.3f} (期待 100.0), 空入力={empty}")
    check("ex5_moving_average_fps", _c5)

    print("=== 採点結果 ===")
    print(f"(採点用動画: mode={mode}, fps={fps}, frames={num})")
    all_ok = True
    for name, ok, detail in results:
        mark = "PASS" if ok else "FAIL"
        all_ok = all_ok and ok
        print(f"  [{mark}] {name:22s} {detail}")
    print("\nALL PASS 🎉" if all_ok else "\nまだ未達の演習があります。TODO を埋めましょう。")


# =====================================================================
# 模範解答（SHOW_SOLUTION=1 のとき本体へ差し替えて実行）。まず自力で！
# =====================================================================

def _sol_ex1(source):
    cap = cv2.VideoCapture(source)
    count = 0
    while True:
        ret, _frame = cap.read()
        if not ret:
            break
        count += 1
    cap.release()
    return count


def _sol_ex2(source):
    cap = cv2.VideoCapture(source)
    fps = cap.get(cv2.CAP_PROP_FPS)
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return (fps, count, width, height)


def _sol_ex3(fourcc_int):
    code = int(fourcc_int)
    return "".join(chr((code >> (8 * i)) & 0xFF) for i in range(4))


def _sol_ex4(source, index):
    cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


def _sol_ex5(dts, window=20):
    buf = deque(maxlen=window)
    for d in dts:
        buf.append(d)
    if not buf:
        return 0.0
    avg = sum(buf) / len(buf)
    return 1.0 / avg if avg > 0 else 0.0


def _install_solutions() -> None:
    g = globals()
    g["ex1_count_frames"] = _sol_ex1
    g["ex2_read_meta"] = _sol_ex2
    g["ex3_fourcc_to_str"] = _sol_ex3
    g["ex4_seek_and_read"] = _sol_ex4
    g["ex5_moving_average_fps"] = _sol_ex5


if __name__ == "__main__":
    if os.environ.get("SHOW_SOLUTION") == "1":
        print("(模範解答モードで実行します)\n")
        _install_solutions()
    _grade()
