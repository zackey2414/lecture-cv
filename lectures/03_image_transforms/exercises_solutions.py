"""第3回 演習の模範解答（このファイルを実行すると全問 PASS する）。

exercises.py の各 TODO を埋めた参照実装。まずは自力で exercises.py を解き、
詰まったらここで答え合わせをすること。

設計: 採点ロジック（_grade）と各模範解答（_sol_*）は exercises.py に1つだけ持たせ、
ここではそれを import → 模範解答を本体へ差し替え → 採点、と再利用する（重複を作らない）。

実行: uv run python lectures/03_image_transforms/exercises_solutions.py
  → "10/10 問 PASS" / "ALL PASS 🎉" が表示されれば、教材の解答が現行版で正しいことの確認。
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import exercises  # noqa: E402（採点ランナーと模範解答の置き場）


def main() -> None:
    print("(模範解答 exercises_solutions: 全問が PASS することを確認します)\n")
    # exercises.py 内の TODO 関数を、同ファイルに用意した模範解答へ差し替える。
    exercises._install_solutions()
    # 差し替え後の関数で自己採点を実行する（全 PASS になるはず）。
    exercises._grade()


if __name__ == "__main__":
    main()
