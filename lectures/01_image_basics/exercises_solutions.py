"""第1回 演習の模範解答（実行すると全問 PASS する）。

このファイルは exercises.py の TODO をすべて埋めた版です。解答ロジックは
exercises.py 内に _sol_exN として既に用意してあるので、ここでは

  1. exercises モジュールを読み込み、
  2. 模範解答を本体関数へ差し替え（_install_solutions）、
  3. 同じ自己採点ランナー（_grade）を実行する

という最小構成にしています（解答ロジックの重複を避けるため）。

実行:
    uv run python lectures/01_image_basics/exercises_solutions.py
期待:
    全 10 問が PASS し "ALL PASS 🎉" と表示され、exit 0 で終了する。

学習の順序としては、まず exercises.py を自力で解き、どうしても詰まったときに
このファイル（または exercises.py の _sol_exN）を読むのがおすすめです。
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import exercises as E  # noqa: E402  同じフォルダの演習本体（採点ランナーを共有する）


def main() -> None:
    print("(模範解答: すべての ex を解答実装に差し替えて採点します)\n")
    E._install_solutions()  # 各 exN を _sol_exN に差し替え
    E._grade()  # exercises.py と同じ採点ランナーを実行 → ALL PASS になる


if __name__ == "__main__":
    main()
