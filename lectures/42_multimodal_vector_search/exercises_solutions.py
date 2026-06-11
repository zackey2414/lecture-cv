"""第42回 演習の模範解答（全 PASS）。

実行:
  uv run python lectures/42_multimodal_vector_search/exercises_solutions.py
→ exercises.py と同じ採点ランナーで全問 PASS することを確認できる。

まずは自力で exercises.py を解き、詰まったときだけここを見ること。
"""

from __future__ import annotations

import os
import pathlib
import sys

import faiss
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from exercises import grade  # noqa: E402  採点ロジックは exercises.py を再利用（DRY）


# =====================================================================
# 模範解答
# =====================================================================

def ex1_l2_normalize_rows(x: np.ndarray) -> np.ndarray:
    """各行を L2 正規化（float32・C連続）。ゼロ割は 1e-12 でクリップ。"""
    x = np.ascontiguousarray(x, dtype=np.float32)
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return np.ascontiguousarray(x / np.maximum(norm, 1e-12), dtype=np.float32)


def ex2_build_cosine_index(xb: np.ndarray) -> faiss.Index:
    """正規化 + IndexFlatIP = コサイン検索の index を返す（add 済み）。"""
    xn = ex1_l2_normalize_rows(xb)
    index = faiss.IndexFlatIP(xn.shape[1])
    index.add(xn)
    return index


def ex3_idmap_search(xb: np.ndarray, ids: np.ndarray, xq: np.ndarray, k: int) -> np.ndarray:
    """IndexIDMap で任意IDを付けて検索し、ID 行列 I を返す。"""
    xb_n = ex1_l2_normalize_rows(xb)
    xq_n = ex1_l2_normalize_rows(xq)
    index = faiss.IndexIDMap(faiss.IndexFlatIP(xb_n.shape[1]))
    index.add_with_ids(xb_n, ids.astype(np.int64))
    _, ids_out = index.search(xq_n, k)
    return ids_out


def ex4_filter_by_modality(
    ids_row: list[int], scores_row: list[float], meta: dict[int, dict], modality: str, k: int
) -> list[tuple[int, float]]:
    """検索結果を modality で絞り、上位k件 (id, score) を返す。-1 は除外。"""
    out: list[tuple[int, float]] = []
    for i, s in zip(ids_row, scores_row):
        if i == -1:
            continue
        m = meta.get(int(i))
        if m is None or m.get("modality") != modality:
            continue
        out.append((int(i), float(s)))
        if len(out) >= k:
            break
    return out


def ex5_save_reload(index: faiss.Index, path: pathlib.Path) -> int:
    """write_index で保存 → read_index で読み戻し → ntotal を返す。"""
    faiss.write_index(index, str(path))
    return faiss.read_index(str(path)).ntotal


def ex6_recall_at_k(retrieved_ids: list[int], relevant_ids: set[int], k: int) -> float:
    """Recall@k = 上位k件の正解数 / 正解総数。"""
    if not relevant_ids:
        return 0.0
    hit = len(set(retrieved_ids[:k]) & relevant_ids)
    return hit / len(relevant_ids)


def ex7_average_precision_at_k(retrieved_ids: list[int], relevant_ids: set[int], k: int) -> float:
    """AP@k = 正解を引くたびの precision の平均（min(正解総数,k) で割る）。"""
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


def ex8_crossmodal_topk(query_vec: np.ndarray, db_matrix: np.ndarray, k: int) -> list[int]:
    """正規化 → コサイン(=内積) → 上位k件の行インデックスを返す（FAISS 不使用）。"""
    q = ex1_l2_normalize_rows(np.atleast_2d(query_vec))[0]
    db = ex1_l2_normalize_rows(db_matrix)
    sims = db @ q
    return np.argsort(-sims)[:k].astype(int).tolist()


def solution_funcs() -> dict:
    return {
        "ex1": ex1_l2_normalize_rows,
        "ex2": ex2_build_cosine_index,
        "ex3": ex3_idmap_search,
        "ex4": ex4_filter_by_modality,
        "ex5": ex5_save_reload,
        "ex6": ex6_recall_at_k,
        "ex7": ex7_average_precision_at_k,
        "ex8": ex8_crossmodal_topk,
    }


if __name__ == "__main__":
    os.makedirs("outputs/42_multimodal_vector_search", exist_ok=True)
    ok = grade(solution_funcs())
    if not ok:
        raise SystemExit("模範解答が全PASSになっていません（バグです）")
