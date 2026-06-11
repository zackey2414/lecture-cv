"""第33回 演習の模範解答（全 PASS）。

実行:
  uv run python lectures/33_multimodal_embeddings/exercises_solutions.py

exercises.py の採点ランナー grade() をそのまま import して使い、
ここでは各 exN_*() を「正しく実装した版」に差し替えて採点する。
モデル DL 不要・決定的なので数秒で全 PASS する。
"""

from __future__ import annotations

import pathlib
import sys

import faiss
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from exercises import grade  # noqa: E402  （採点ランナーを再利用）


# =====================================================================
# 模範解答
# =====================================================================

def ex1_sigmoid(x: np.ndarray) -> np.ndarray:
    """要素ごとの sigmoid。各要素が独立に 0〜1（SigLIP の確率化）。"""
    return 1.0 / (1.0 + np.exp(-x))


def ex2_softmax_rows(x: np.ndarray) -> np.ndarray:
    """行ごとの softmax（各行の和が 1）。行max を引いてからexp（オーバーフロー回避）。"""
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def ex3_l2_normalize_rows(x: np.ndarray) -> np.ndarray:
    """各行を L2 正規化（float32・C連続）。正規化後の内積=コサイン。"""
    x = np.ascontiguousarray(x, dtype=np.float32)
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return np.ascontiguousarray(x / np.maximum(norm, 1e-12), dtype=np.float32)


def ex4_cosine_sim_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """2 行列の全ペアコサイン類似度 (A,B)。各行 L2 正規化 → 内積。"""
    a_n = ex3_l2_normalize_rows(a)
    b_n = ex3_l2_normalize_rows(b)
    return a_n @ b_n.T


def ex5_build_cosine_index(xb: np.ndarray) -> faiss.Index:
    """コサイン検索用 IndexFlatIP（正規化して add 済み）。"""
    xb_n = ex3_l2_normalize_rows(xb)
    index = faiss.IndexFlatIP(xb_n.shape[1])
    index.add(xb_n)
    return index


def ex6_text_to_image_top1(text_emb: np.ndarray, image_emb: np.ndarray) -> list[int]:
    """各テキストに最も近い画像の行インデックス（クロスモーダル top1）。"""
    sims = ex4_cosine_sim_matrix(text_emb, image_emb)  # (T, I)
    return [int(i) for i in np.argmax(sims, axis=1)]


def ex7_recall_at_k(retrieved_ids: list[int], relevant_ids: set[int], k: int) -> float:
    """Recall@k = 上位k件に入った正解数 / 正解総数。"""
    if not relevant_ids:
        return 0.0
    topk = [i for i in retrieved_ids[:k] if i != -1]
    return len(set(topk) & relevant_ids) / len(relevant_ids)


def ex8_average_precision_at_k(retrieved_ids: list[int], relevant_ids: set[int], k: int) -> float:
    """AP@k（順位を考慮した適合率の平均, mAP の素）。"""
    if not relevant_ids:
        return 0.0
    hits = 0
    score = 0.0
    for rank, idx in enumerate(retrieved_ids[:k], start=1):
        if idx in relevant_ids:
            hits += 1
            score += hits / rank
    denom = min(len(relevant_ids), k)
    return score / denom if denom > 0 else 0.0


def ex9_crossmodal_recall_at_k(
    query_emb: np.ndarray,
    db_emb: np.ndarray,
    query_concepts: list[str],
    db_concepts: list[str],
    k: int,
) -> float:
    """クロスモーダル検索（音→画像など）の平均 Recall@k。"""
    index = ex5_build_cosine_index(db_emb)
    q_n = ex3_l2_normalize_rows(query_emb)
    _scores, ids = index.search(q_n, index.ntotal)  # 全件ランキング
    recalls: list[float] = []
    for qi, qc in enumerate(query_concepts):
        relevant = {j for j, c in enumerate(db_concepts) if c == qc}
        ranked = [int(i) for i in ids[qi].tolist()]
        recalls.append(ex7_recall_at_k(ranked, relevant, k))
    return float(np.mean(recalls)) if recalls else 0.0


def solution_funcs() -> dict:
    return {
        "ex1": ex1_sigmoid,
        "ex2": ex2_softmax_rows,
        "ex3": ex3_l2_normalize_rows,
        "ex4": ex4_cosine_sim_matrix,
        "ex5": ex5_build_cosine_index,
        "ex6": ex6_text_to_image_top1,
        "ex7": ex7_recall_at_k,
        "ex8": ex8_average_precision_at_k,
        "ex9": ex9_crossmodal_recall_at_k,
    }


if __name__ == "__main__":
    grade(solution_funcs())
