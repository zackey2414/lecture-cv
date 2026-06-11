"""第38回 演習の模範解答（全 PASS）。

実行:
  uv run python lectures/38_knowledge_distillation/exercises_solutions.py

exercises.py の採点ランナー grade() をそのまま import して使い、
ここでは各 exN_*() を「正しく実装した版」に差し替えて採点する。
モデル DL も学習も不要・決定的なので一瞬で全 PASS する。
"""

from __future__ import annotations

import pathlib
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from exercises import grade  # noqa: E402  （採点ランナーを再利用）


# =====================================================================
# 模範解答
# =====================================================================

def ex1_soften_logits(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """温度 T で軟化した確率分布 softmax(logits / T)。"""
    return F.softmax(logits / temperature, dim=1)


def ex2_kd_kl_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor,
                   temperature: float) -> torch.Tensor:
    """温度付き KL（ソフト損失）。入力=log_softmax(student)/ターゲット=softmax(teacher)・T^2。"""
    log_p = F.log_softmax(student_logits / temperature, dim=1)
    q = F.softmax(teacher_logits / temperature, dim=1)
    return F.kl_div(log_p, q, reduction="batchmean") * (temperature ** 2)


def ex3_combined_kd_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor,
                         targets: torch.Tensor, temperature: float,
                         alpha: float) -> torch.Tensor:
    """合成損失 = alpha*ソフト + (1-alpha)*ハード(CE)。"""
    soft = ex2_kd_kl_loss(student_logits, teacher_logits, temperature)
    hard = F.cross_entropy(student_logits, targets)
    return alpha * soft + (1.0 - alpha) * hard


def ex4_compression_ratio(teacher_params: int, student_params: int) -> float:
    """圧縮率 = teacher / student（student=0 は 0.0）。"""
    if student_params == 0:
        return 0.0
    return teacher_params / student_params


def ex5_soft_target_entropy(logits: torch.Tensor, temperature: float) -> float:
    """温度ソフトターゲットの平均エントロピー(nats)。"""
    p = F.softmax(logits / temperature, dim=1)
    entropy = -(p * (p + 1e-12).log()).sum(dim=1)
    return float(entropy.mean())


def ex6_feature_distill_mse(student_feat: torch.Tensor, teacher_feat: torch.Tensor,
                            proj_weight: torch.Tensor) -> torch.Tensor:
    """特徴量蒸留 MSE: 射影で次元を合わせ、L2 正規化してから MSE。"""
    proj = student_feat @ proj_weight.T          # (N,Cs)@(Cs,Ct) = (N,Ct)
    s = F.normalize(proj, dim=1)
    t = F.normalize(teacher_feat, dim=1)
    return F.mse_loss(s, t)


def ex7_deit_inference_logits(cls_logits: torch.Tensor,
                              dist_logits: torch.Tensor) -> torch.Tensor:
    """DeiT 推論: class ヘッドと distill ヘッドの平均。"""
    return (cls_logits + dist_logits) / 2.0


def ex8_rkd_distance_matrix(emb: torch.Tensor) -> torch.Tensor:
    """RKD の正規化距離行列: cdist を「0 を除いた平均」で割る。"""
    dist = torch.cdist(emb, emb)
    mu = dist[dist > 0].mean()
    return dist / mu


def solution_funcs() -> dict:
    return {
        "ex1": ex1_soften_logits,
        "ex2": ex2_kd_kl_loss,
        "ex3": ex3_combined_kd_loss,
        "ex4": ex4_compression_ratio,
        "ex5": ex5_soft_target_entropy,
        "ex6": ex6_feature_distill_mse,
        "ex7": ex7_deit_inference_logits,
        "ex8": ex8_rkd_distance_matrix,
    }


if __name__ == "__main__":
    grade(solution_funcs())
