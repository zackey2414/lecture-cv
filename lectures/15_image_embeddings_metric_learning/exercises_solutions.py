"""第15回 演習の模範解答（全 PASS）。

実行:
  uv run python lectures/15_image_embeddings_metric_learning/exercises_solutions.py

exercises.py の採点ランナー grade() をそのまま import して使い（採点ロジックの重複なし）、
ここでは各 exN_*() を「正しく実装した版」に差し替えて採点する。
モデル DL 不要・決定的なので数秒で全 PASS する。
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from exercises import grade  # noqa: E402  （採点ランナーを再利用）


# =====================================================================
# 模範解答
# =====================================================================

def ex1_l2_normalize(x: np.ndarray) -> np.ndarray:
    """各行を L2 ノルム 1 に正規化（eps で 0 割回避）。正規化後は内積=コサイン。"""
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norm, 1e-12)


def ex2_cosine_sim_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a, b を行ごと L2 正規化してから内積（= 全ペアのコサイン類似度）。"""
    return ex1_l2_normalize(a) @ ex1_l2_normalize(b).T


def ex3_knn_predict(
    gallery: np.ndarray, gallery_labels: np.ndarray, query: np.ndarray, k: int
) -> np.ndarray:
    """コサイン上位 k 件の多数決ラベル。"""
    sim = ex2_cosine_sim_matrix(query, gallery)
    topk = np.argsort(-sim, axis=1)[:, :k]
    neigh = gallery_labels[topk]
    return np.array([np.bincount(row).argmax() for row in neigh], dtype=np.int64)


def ex4_recall_at_k(
    gallery: np.ndarray, gallery_labels: np.ndarray,
    query: np.ndarray, query_labels: np.ndarray, k: int
) -> float:
    """上位 k 件に同クラスが 1 件でも入る割合。"""
    sim = ex2_cosine_sim_matrix(query, gallery)
    topk = np.argsort(-sim, axis=1)[:, :k]
    hit = (gallery_labels[topk] == query_labels[:, None]).any(axis=1)
    return float(hit.mean())


def ex5_triplet_margin_loss(
    anchor: np.ndarray, positive: np.ndarray, negative: np.ndarray, margin: float
) -> float:
    """hinge = max(0, ||a-p|| - ||a-n|| + margin) の平均。"""
    d_ap = np.linalg.norm(anchor - positive, axis=1)
    d_an = np.linalg.norm(anchor - negative, axis=1)
    return float(np.maximum(0.0, d_ap - d_an + margin).mean())


def ex6_vit_pool(last_hidden_state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """先頭 CLS と、残りパッチの系列平均（mean pooling）。"""
    return last_hidden_state[:, 0], last_hidden_state[:, 1:].mean(axis=1)


def ex7_batch_hard_triplet_loss(z: np.ndarray, labels: np.ndarray, margin: float) -> float:
    """各アンカーで最遠正例 d_pos と最近負例 d_neg を選んだ hinge の平均。"""
    d = np.sqrt(((z[:, None, :] - z[None, :, :]) ** 2).sum(-1))  # 全ペア L2 距離
    same = labels[:, None] == labels[None, :]
    d_pos = np.where(same, d, -np.inf).max(1)  # 最も遠い正例（自分自身 d=0 は max に無影響）
    d_neg = np.where(~same, d, np.inf).min(1)  # 最も近い負例
    return float(np.maximum(0.0, d_pos - d_neg + margin).mean())


def ex8_supervised_infonce(z: np.ndarray, labels: np.ndarray, temperature: float) -> float:
    """同クラスを正例集合とした温度付き softmax（教師あり InfoNCE）の平均損失。"""
    zn = ex1_l2_normalize(z)
    sim = (zn @ zn.T) / temperature
    np.fill_diagonal(sim, -1e9)  # 自分自身は除外
    same = (labels[:, None] == labels[None, :]).astype(np.float64)
    np.fill_diagonal(same, 0.0)
    m = sim.max(axis=1, keepdims=True)  # オーバーフロー回避
    logp = sim - (m + np.log(np.exp(sim - m).sum(axis=1, keepdims=True)))  # 行 log_softmax
    loss = -(same * logp).sum(1) / np.maximum(same.sum(1), 1.0)
    return float(loss.mean())


def ex9_retrieval_map(
    gallery: np.ndarray, gallery_labels: np.ndarray,
    query: np.ndarray, query_labels: np.ndarray
) -> float:
    """各クエリの Average Precision を平均した mAP（順位を考慮した検索精度）。"""
    sim = ex2_cosine_sim_matrix(query, gallery)
    order = np.argsort(-sim, axis=1)  # 類似度降順のランキング
    rel = (gallery_labels[order] == query_labels[:, None]).astype(np.float64)
    cum = np.cumsum(rel, axis=1)
    ranks = np.arange(1, gallery.shape[0] + 1)
    precision = cum / ranks  # 各順位での precision@r
    nrel = rel.sum(axis=1)
    ap = np.where(nrel > 0, (precision * rel).sum(axis=1) / np.maximum(nrel, 1.0), 0.0)
    return float(ap.mean())


def solution_funcs() -> dict:
    return {
        "ex1": ex1_l2_normalize,
        "ex2": ex2_cosine_sim_matrix,
        "ex3": ex3_knn_predict,
        "ex4": ex4_recall_at_k,
        "ex5": ex5_triplet_margin_loss,
        "ex6": ex6_vit_pool,
        "ex7": ex7_batch_hard_triplet_loss,
        "ex8": ex8_supervised_infonce,
        "ex9": ex9_retrieval_map,
    }


if __name__ == "__main__":
    grade(solution_funcs())
