"""第45回 演習の模範解答ランナー（全問 PASS）。

exercises.py の TODO をすべて埋めた状態を再現する。重複を避けるため、採点ロジック
(_grade) も模範解答(_sol_*) も exercises 側のものをそのまま再利用し、ここでは
「解答を差し込んで採点を回すだけ」に徹する。

実行:
  uv run python lectures/45_sketch_emoji_search/exercises_solutions.py
  → 8/8 PASS（ALL PASS）になることを確認する。

まずは exercises.py を自力で埋めること。これは答え合わせ用。
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import exercises  # noqa: E402


def main() -> None:
    # ex*_ 名に模範解答を差し込んでから、exercises 側の採点ランナーを呼ぶ。
    # （採点ロジックを二重に持たない＝唯一の正解定義は exercises._grade に集約）
    exercises._install_solutions()
    exercises._grade()


if __name__ == "__main__":
    main()
