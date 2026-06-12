"""第16回 演習の模範解答（全問 PASS）。

設計方針:
  - 採点ロジック（テストケース）は exercises.py 側にだけ置き、ここでは『解答の実装』だけを持つ。
    重複を避けるため、本ファイルは exercises._grade() をそのまま再利用して採点する。
  - install(module) で exercises（または SHOW_SOLUTION 実行中の __main__）へ解答を差し込む。

実行: uv run python lectures/16_clip_zeroshot_retrieval/exercises_solutions.py
  → exercises のテストを解答実装で走らせ、ALL PASS を表示する（exit 0）。
"""

from __future__ import annotations

import pathlib
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import exercises  # noqa: E402  採点ランナー（_grade）と TODO 雛形を提供する本体


# =====================================================================
# 模範解答（各 ex の正しい実装）
# =====================================================================

def solve_ex1_l2_normalize(x: torch.Tensor) -> torch.Tensor:
    # 各行を L2 ノルムで割る。F.normalize が eps つきで安全にやってくれる。
    return F.normalize(x, p=2, dim=-1)


def solve_ex2_cosine_sim_matrix(text_emb: torch.Tensor, image_emb: torch.Tensor) -> torch.Tensor:
    # 両方を正規化してから内積 → コサイン類似度（-1〜1）。
    t = F.normalize(text_emb, p=2, dim=-1)
    im = F.normalize(image_emb, p=2, dim=-1)
    return t @ im.t()


def solve_ex3_zeroshot_probs(logits: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "softmax":
        return torch.softmax(logits, dim=-1)  # 行合計1（相互排他）
    if mode == "sigmoid":
        return torch.sigmoid(logits)          # 各要素独立（合計は1にならない）
    raise ValueError(f"未知の mode: {mode}")


def solve_ex4_topk_indices(scores: torch.Tensor, k: int) -> list[int]:
    return torch.topk(scores, k).indices.tolist()


def solve_ex5_recall_at_k(ranking: list[int], relevant: set[int], k: int) -> float:
    if not relevant:
        return 0.0
    hit = len(set(ranking[:k]) & relevant)
    return hit / len(relevant)


def solve_ex6_reciprocal_rank(ranking: list[int], relevant: set[int]) -> float:
    for i, idx in enumerate(ranking):
        if idx in relevant:
            return 1.0 / (i + 1)  # 最初の正解が見つかった順位の逆数
    return 0.0


def solve_ex7_average_precision(ranking: list[int], relevant: set[int]) -> float:
    if not relevant:
        return 0.0
    hits = 0
    precision_sum = 0.0
    for i, idx in enumerate(ranking):
        if idx in relevant:
            hits += 1
            precision_sum += hits / (i + 1)  # この正解ヒット位置での precision
    return precision_sum / len(relevant)


def solve_ex8_clip_logits_from_embeds(
    image_emb: torch.Tensor, text_emb: torch.Tensor, logit_scale_raw: torch.Tensor
) -> torch.Tensor:
    img_n = F.normalize(image_emb, p=2, dim=-1)
    txt_n = F.normalize(text_emb, p=2, dim=-1)
    return (img_n @ txt_n.t()) * logit_scale_raw.exp()  # 温度は exp してから掛ける


def solve_ex9_abstain_or_pick(scores: torch.Tensor, threshold: float) -> int:
    best = int(torch.argmax(scores))
    if float(scores[best]) >= threshold:
        return best
    return -1  # 最大でも閾値未満 → 該当なし（softmax には出せない芸当）


# exercises 側の関数名 → 解答実装 の対応表（重複なく一箇所で管理）。
SOLUTIONS = {
    "ex1_l2_normalize": solve_ex1_l2_normalize,
    "ex2_cosine_sim_matrix": solve_ex2_cosine_sim_matrix,
    "ex3_zeroshot_probs": solve_ex3_zeroshot_probs,
    "ex4_topk_indices": solve_ex4_topk_indices,
    "ex5_recall_at_k": solve_ex5_recall_at_k,
    "ex6_reciprocal_rank": solve_ex6_reciprocal_rank,
    "ex7_average_precision": solve_ex7_average_precision,
    "ex8_clip_logits_from_embeds": solve_ex8_clip_logits_from_embeds,
    "ex9_abstain_or_pick": solve_ex9_abstain_or_pick,
}


def install(module) -> None:
    """指定モジュールの ex*_ 関数を模範解答で上書きする。"""
    for name, fn in SOLUTIONS.items():
        setattr(module, name, fn)


def main() -> None:
    # exercises の TODO を解答で差し替えてから、exercises 側の採点ロジックを再利用する。
    install(exercises)
    all_ok = exercises._grade()
    if not all_ok:  # 解答が正しければここには来ない（保険）。
        print("（想定外）解答で FAIL が出ています。実装を見直してください。")


if __name__ == "__main__":
    main()
