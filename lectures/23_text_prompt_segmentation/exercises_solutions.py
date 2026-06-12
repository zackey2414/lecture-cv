"""第23回 演習の模範解答ランナー（全問 PASS）。

exercises.py の採点ロジック・模範解答（_sol_*）をそのまま再利用し、TODO 関数を
模範解答に差し替えてから採点する。採点ロジックも解答本体も exercises 側に1本化し、
ここでは重複実装しない（DRY）。学習者が「正解だと採点がどう出るか」を確認する用途。

実行: uv run python lectures/23_text_prompt_segmentation/exercises_solutions.py
  → 全問 PASS（ALL PASS）になり exit 0。

まずは exercises.py の TODO を自力で埋めること。本ファイルは答え合わせ用。
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import exercises as ex  # noqa: E402


def main() -> None:
    print("(exercises.py の模範解答を全問に適用して採点します)\n")
    ex._install_solutions()  # ex1..ex9 を _sol_* へ差し替え
    all_ok = ex._grade()     # 採点ロジックは exercises 側を再利用
    if not all_ok:
        # 模範解答なら必ず全 PASS のはず。万一 FAIL があれば気付けるよう明示する。
        print("\n[警告] 模範解答で FAIL が出ました。期待値・解答を確認してください。")


if __name__ == "__main__":
    main()
