"""第19回 演習の模範解答ランナー（全問 PASS）。

このファイルは「答え合わせ専用」。採点ロジックは exercises.py の _grade() を、模範解答は
exercises.py の _sol_* を、そのまま再利用する（重複実装はしない）。やっていることは
「exercises 側の本体関数(ex1〜ex9)を模範解答に差し替えてから採点を回す」だけ。

  - exercises.py を自力で解く前のカンニングにせず、詰まったときの“正解の挙動確認”に使う。
  - SHOW_SOLUTION=1 uv run python .../exercises.py と同じ結果（ALL PASS）になる。

実行: uv run python lectures/19_detection_map_from_scratch/exercises_solutions.py
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import exercises  # noqa: E402  採点ロジックと模範解答をそのまま借りる（重複なし）


def main() -> None:
    print("(模範解答モード: exercises.py の ex1〜ex9 を解答に差し替えて採点します)\n")
    exercises._install_solutions()  # 本体関数を _sol_* に差し替え
    exercises._grade()              # exercises.py と同一の採点を実行 → ALL PASS


if __name__ == "__main__":
    main()
