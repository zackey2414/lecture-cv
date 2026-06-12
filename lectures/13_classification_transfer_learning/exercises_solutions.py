"""exercises_solutions.py — 第13回 演習の完全な模範解答（実行すると全 PASS）。

    uv run python lectures/13_classification_transfer_learning/exercises_solutions.py

exercises.py と同じ採点ハーネス（_grade）を再利用する。各 TODO 関数を模範解答(_sol_*)で
差し替え、全 9 問が PASS することを assert で機械的に保証する。まずは自力で exercises.py を
解いてから参照すること（採点ロジックはこちらには重複させず、exercises 側を import して使う）。

各解答は単一責務・早期 return・本章の定石を素直に写したもの:
  ex1 device 判定      : cuda → mps → cpu の早期 return 3 分岐
  ex2 ImageNet 正規化  : /255 → CHW 転置 → (x-mean)/std（mean/std は (3,1,1) で broadcast）
  ex3 転移学習モデル   : backbone を requires_grad_(False) → nn.Linear ヘッド → Sequential
  ex4 top-k accuracy   : topk.indices と targets.unsqueeze(1) の一致を any → 平均（N=0 は 0.0）
  ex5 コサイン類似度   : F.normalize(p=2) してから内積
  ex6 top-k ラベル名   : topk.indices を labels で名前に変換（id2label 後処理）
  ex7 混同行列         : zip(targets, preds) で cm[t, p] += 1（row=正解, col=予測）
  ex8 macro recall     : 各クラス diag/行和 を行和 0 を除いて平均
  ex9 最近傍重心分類   : クラス重心を作り正規化 → コサイン類似 → argmax
"""

from __future__ import annotations

import pathlib
import sys

# exercises.py の採点ハーネスと模範解答(_sol_*)・差し替え関数(_install_solutions)を再利用する。
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import exercises  # noqa: E402


def main() -> None:
    print("=" * 60)
    print("13_classification_transfer_learning 演習 模範解答 自己採点")
    print("=" * 60)

    # 全 TODO を模範解答へ差し替えてから採点する。
    exercises._install_solutions()

    # _grade() は採点結果を表示し、全問 PASS なら "ALL PASS" を出す（ビュー）。
    exercises._grade()

    # --- 念のため模範解答が全 PASS することを assert で機械的に保証する ---
    _assert_all_pass()
    print("\n全 9 問の模範解答が PASS することを確認しました。")


def _assert_all_pass() -> None:
    """exercises.py の模範解答(_sol_*)を直接呼び、全問の出力を検証する。"""
    import numpy as np
    import torch
    import torch.nn as nn

    s = exercises  # 模範解答が差し込まれた名前空間（_sol_* も同居）

    # ex1: device 優先順位
    assert s._sol_ex1(True, True) == "cuda"
    assert s._sol_ex1(False, True) == "mps"
    assert s._sol_ex1(False, False) == "cpu"

    # ex2: ImageNet 正規化（形・dtype）
    px = (np.arange(2 * 2 * 3).reshape(2, 2, 3) % 255).astype(np.uint8)
    out2 = s._sol_ex2(px)
    assert out2.shape == (3, 2, 2) and out2.dtype == np.float32

    # ex3: 転移学習モデル（head=15 のみ学習可・backbone 凍結）
    backbone = nn.Sequential(nn.Linear(8, 4), nn.ReLU(), nn.Linear(4, 4))
    model = s._sol_ex3(backbone, 4, 3)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    assert trainable == 15 and frozen > 0

    # ex4: top-k accuracy
    logits = torch.tensor([[3.0, 1.0, 0.0, 2.0], [0.0, 1.0, 2.0, 0.5], [1.0, 0.0, 0.0, 0.0]])
    targets = torch.tensor([0, 1, 3])
    assert abs(s._sol_ex4(logits, targets, 1) - 1 / 3) < 1e-6
    assert abs(s._sol_ex4(logits, targets, 2) - 2 / 3) < 1e-6
    assert s._sol_ex4(torch.zeros(0, 4), torch.zeros(0, dtype=torch.long), 1) == 0.0

    # ex5: コサイン類似度
    assert abs(s._sol_ex5(torch.tensor([3.0, 0.0, 0.0]), torch.tensor([10.0, 0.0, 0.0])) - 1.0) < 1e-6
    assert abs(s._sol_ex5(torch.tensor([3.0, 0.0, 0.0]), torch.tensor([0.0, 5.0, 0.0]))) < 1e-6

    # ex6: top-k ラベル名
    assert s._sol_ex6(torch.tensor([0.1, 5.0, 2.0, 3.0]), ["a", "b", "c", "d"], 2) == ["b", "d"]

    # ex7: 混同行列
    cm = s._sol_ex7(np.array([0, 1, 2, 1, 0]), np.array([0, 1, 1, 1, 2]), 3)
    assert cm.shape == (3, 3) and int(cm.sum()) == 5
    assert np.array_equal(cm, np.array([[1, 0, 0], [0, 2, 1], [1, 0, 0]]))

    # ex8: macro recall
    mr = s._sol_ex8(np.array([[8, 2, 0], [0, 10, 0], [1, 1, 8]]))
    assert abs(mr - (0.8 + 1.0 + 0.8) / 3) < 1e-6
    assert abs(s._sol_ex8(np.array([[5, 0], [0, 0]])) - 1.0) < 1e-6

    # ex9: 最近傍重心分類
    train_emb = torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]])
    train_labels = torch.tensor([0, 0, 1, 1])
    query = torch.tensor([[2.0, 0.0], [0.0, 3.0], [0.8, 0.2]])
    pred = s._sol_ex9(train_emb, train_labels, 2, query)
    assert torch.equal(pred.long(), torch.tensor([0, 1, 0]))


if __name__ == "__main__":
    main()
