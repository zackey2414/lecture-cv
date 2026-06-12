"""第18回 演習の模範解答ランナー（全10問 PASS を確認する）。

設計方針: 採点ロジックは exercises.py に一本化し、ここでは重複させない。
exercises._install_solutions() で ex1〜ex10 を模範解答に差し替えてから、
exercises._grade() をそのまま呼ぶだけ。学習者が自力で詰まったときの「答え合わせ」用。

実行: uv run python lectures/18_object_detection_intro/exercises_solutions.py
期待: 全問 PASS（ALL PASS）して exit 0。
"""

from __future__ import annotations

import pathlib
import sys

# 同フォルダの exercises を import できるようにパスを通す（スクリプト単体実行のため）。
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import exercises as E  # noqa: E402  採点ロジックと模範解答の本体


def main() -> None:
    print("(模範解答を全問に差し込んで採点します)\n")
    E._install_solutions()      # ex1〜ex10 を _sol_* に置き換え
    all_ok = E._grade()         # exercises 側の採点を再利用（重複なし）
    if not all_ok:
        # 模範解答は全問通る前提。通らなければ解答側のバグなので明示的に失敗させる。
        raise SystemExit("模範解答が全問 PASS しませんでした（解答側の不具合）。")
    print("\n模範解答: 全10問 PASS を確認しました。")


if __name__ == "__main__":
    main()
