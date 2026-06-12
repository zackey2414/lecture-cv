"""第22回 演習の模範解答（全問 PASS）。

exercises.py の TODO をすべて埋めた完成形に相当する。ただし解答コードはここに
再掲しない——exercises.py 側に置いた `_sol_ex*`（模範解答）と `_grade`（採点ロジック）を
そのまま再利用し、本体名へ差し替えてから採点を回すだけ。こうすることで
「採点ロジックや解答ロジックの二重管理」を避ける（DRY）。

実行:
  uv run python lectures/22_instance_panoptic_sam/exercises_solutions.py
  → 10問すべて PASS（ALL PASS 🎉）になることを確認する。

学習の流れ:
  1. まず exercises.py の TODO を自力で埋める（最初は全 FAIL でも exit 0）。
  2. 詰まったら SHOW_SOLUTION=1 で挙動を見る、または本ファイルで完成形の採点を見る。
"""

from __future__ import annotations

import pathlib
import sys

# digit 始まりの 01_*.py 等は import 不可だが、exercises.py は import 可能。
# 同ディレクトリを import パスに通してから読み込む。
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import exercises as ex  # noqa: E402


def main() -> None:
    """模範解答を全問に注入し、exercises.py の採点ランナーをそのまま回す。"""
    ex._install_solutions()  # ex1..ex10 を _sol_* に差し替え（採点は exercises 側を再利用）
    ex._grade()


if __name__ == "__main__":
    main()
