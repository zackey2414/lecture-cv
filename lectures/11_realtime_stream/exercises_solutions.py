"""exercises_solutions.py — 第11回 演習の完全な模範解答（実行すると全 PASS）。

    uv run python lectures/11_realtime_stream/exercises_solutions.py

exercises.py と同じ採点ハーネス（_grade）をそのまま再利用する（採点ロジックを重複させない）。
全 TODO 関数を模範解答(_sol_*)で差し替えてから採点し、全 10 問が PASS することを assert で
保証する。まずは自力で exercises.py を解いてから参照すること。

各解答は単一責務・早期 return・本章の定石を素直に写したもの:
  ex1  clean_mask         : threshold(>200)→OPEN(1)→CLOSE(2) で影127除去＋掃除
  ex2  resize_keep_aspect : scale=max_side/max(h,w)、scale<1 のときだけ INTER_AREA で縮小
  ex3  drop_when_full     : put_nowait し queue.Full を数える
  ex4  ema_fps            : dt から瞬間FPS、ema=(1-a)*ema+a*fps
  ex5  reconnect_summary  : good/reconnects、連続失敗が give_up_after で break
  ex6  latency_percentiles: np.percentile(arr, 50/99)（空は (0,0)）
  ex7  processed_count    : num_seen // every_n（every_n<=0 は ValueError）
  ex8  count_motion_boxes : findContours→contourArea>=min_area を数える
  ex9  backoff_schedule   : min(initial*factor**i, cap) を num_failures 個
  ex10 queue_replay       : put_nowait ドロップ＋get_nowait（空は None）の再生
"""

from __future__ import annotations

import pathlib
import sys

# exercises.py の採点ハーネス(_grade)・模範解答(_sol_*)・差し替え(_install_solutions)を再利用する。
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import exercises  # noqa: E402


def main() -> None:
    print("=" * 60)
    print("11_realtime_stream 演習 模範解答 自己採点")
    print("=" * 60)

    # 全 TODO を模範解答へ差し替えてから、exercises 側の採点ロジックをそのまま回す。
    exercises._install_solutions()
    all_ok = exercises._grade()   # 採点結果を表示し、全問 PASS かどうかを返す

    # --- 模範解答が全 10 問 PASS することを機械的に保証する ---
    assert all_ok, "模範解答が PASS しませんでした（exercises.py の採点条件を確認してください）"
    print("\n全 10 問の模範解答が PASS することを確認しました。")


if __name__ == "__main__":
    main()
