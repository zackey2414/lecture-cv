"""第42回 演習問題（マルチモーダル・ベクトル検索 / CLIP + FAISS）。

使い方:
  1. 各 exN_*() の中の TODO を自分で実装する（最初は NotImplementedError で FAIL）。
  2. 自己採点（未実装でも例外で落とさず PASS/FAIL を表示し、必ず exit 0）:
         uv run python lectures/42_multimodal_vector_search/exercises.py
  3. 全問の模範解答（全 PASS）はこちら:
         uv run python lectures/42_multimodal_vector_search/exercises_solutions.py

テーマ（易→難）:
  1 L2正規化  2 コサインindex  3 IndexIDMap  4 modality後フィルタ
  5 保存・読込  6 Recall@k  7 AP@k  8 クロスモーダル top-k（numpy）

CLIP は使わず、決定的な合成ベクトルだけで採点するので速くて再現的。
本物の CLIP 連携は 01〜04・mini_project.py を参照。
"""

from __future__ import annotations

import os
import pathlib

import faiss
import numpy as np


# =====================================================================
# 演習（ここを実装する）
# =====================================================================

def ex1_l2_normalize_rows(x: np.ndarray) -> np.ndarray:
    """演習1: 各行を L2 正規化した float32・C連続 配列を返す。

    正規化後に内積を取ると = コサイン類似度になる（FAISS のコサイン検索の前提）。
    ゼロ割を避けるため、ノルムは下限 1e-12 でクリップすること。
    """
    # TODO: float32化 → 行ごとのノルムで割る（np.maximum(norm,1e-12)）→ C連続で返す
    raise NotImplementedError


def ex2_build_cosine_index(xb: np.ndarray) -> faiss.Index:
    """演習2: コサイン類似度で検索できる Flat インデックスを作って返す（add 済み）。

    手順: xb を L2 正規化 → faiss.IndexFlatIP(d) を作る → add する。
    （正規化 + 内積(IP) = コサイン類似度）
    """
    # TODO: ex1 で正規化 → IndexFlatIP(d) に add した index を返す
    raise NotImplementedError


def ex3_idmap_search(xb: np.ndarray, ids: np.ndarray, xq: np.ndarray, k: int) -> np.ndarray:
    """演習3: IndexIDMap で任意IDを付けて検索し、近傍の「ID行列」I を返す。

    手順: 正規化 → IndexIDMap(IndexFlatIP(d)) → add_with_ids(xb, ids)
          → search(xq, k) の ID 行列 I を返す。xb・xq は同じ前処理で正規化。
    ids は int64 にすること。
    """
    # TODO: IndexIDMap + add_with_ids + search で I を返す
    raise NotImplementedError


def ex4_filter_by_modality(
    ids_row: list[int], scores_row: list[float], meta: dict[int, dict], modality: str, k: int
) -> list[tuple[int, float]]:
    """演習4: 1クエリの検索結果を modality で絞り、上位k件 (id, score) を返す。

    - ids_row / scores_row は search の1行（同じ並び）。
    - -1（無効ID）は飛ばす。meta[id]['modality'] が引数 modality と一致するものだけ残す。
    - 上位から見て k 件そろったら打ち切る。
    """
    # TODO: zip(ids_row, scores_row) を走査し、-1除外・modality一致のみを k 件集める
    raise NotImplementedError


def ex5_save_reload(index: faiss.Index, path: pathlib.Path) -> int:
    """演習5: index を path に保存し、読み戻した index の ntotal を返す。

    faiss.write_index で保存 → faiss.read_index で別オブジェクトとして読込 → ntotal を返す。
    """
    # TODO: write_index / read_index
    raise NotImplementedError


def ex6_recall_at_k(retrieved_ids: list[int], relevant_ids: set[int], k: int) -> float:
    """演習6: 1クエリの Recall@k = 上位k件に入った正解数 / 正解総数 を返す。

    relevant_ids が空なら 0.0（0除算回避）。retrieved_ids[:k] と relevant の積集合サイズで計算。
    """
    # TODO: len(set(retrieved_ids[:k]) & relevant_ids) / len(relevant_ids)
    raise NotImplementedError


def ex7_average_precision_at_k(retrieved_ids: list[int], relevant_ids: set[int], k: int) -> float:
    """演習7: 1クエリの AP@k を返す（mAP の素）。

    上位から順に見て、正解を引くたびに「その時点の precision = hits/rank」を加算し、
    最後に min(正解総数, k) で割る。relevant_ids が空なら 0.0。
    """
    # TODO: rank=1..k を走査し hits を数えながら hits/rank を加算、min(len(rel),k) で割る
    raise NotImplementedError


def ex8_crossmodal_topk(query_vec: np.ndarray, db_matrix: np.ndarray, k: int) -> list[int]:
    """演習8: 生(未正規化)のクエリと DB 行列から、コサイン上位k件の「行インデックス」を返す。

    FAISS を使わず numpy だけで「クロスモーダル検索の中身」を書く問題。
    手順: query と db を各行 L2 正規化 → コサイン = 内積 → 大きい順に上位 k 件のインデックス。
    （query_vec は (d,) でも (1,d) でも可。返り値は長さ k の int リスト）
    """
    # TODO: 正規化して内積 → np.argsort(-sims)[:k] を list で返す
    raise NotImplementedError


# =====================================================================
# 自己採点ランナー（exercises_solutions.py からも import して再利用）
# =====================================================================

def make_test_data():
    """採点用の決定的データ（5クラスタの合成ベクトル）を作る。"""
    rng = np.random.default_rng(42)
    d, n_clusters, per = 16, 5, 20
    centers = rng.standard_normal((n_clusters, d)).astype(np.float32) * 3.0
    xb, labels = [], []
    for c in range(n_clusters):
        pts = centers[c] + rng.standard_normal((per, d)).astype(np.float32) * 0.5
        xb.append(pts)
        labels += [c] * per
    xb = np.ascontiguousarray(np.concatenate(xb, axis=0), dtype=np.float32)
    return xb, np.array(labels), d


def grade(funcs: dict) -> bool:
    """funcs（名前→関数）を採点して PASS/FAIL を表示。全PASSなら True。"""
    xb, labels, d = make_test_data()
    ids = np.arange(1000, 1000 + len(xb)).astype(np.int64)
    xq = xb[:5].copy()
    results: list[tuple[str, bool, str]] = []

    def check(name: str, fn) -> None:
        try:
            ok, detail = fn()
            results.append((name, bool(ok), detail))
        except NotImplementedError:
            results.append((name, False, "未実装（TODOを埋めてください）"))
        except Exception as e:  # noqa: BLE001
            results.append((name, False, f"例外: {type(e).__name__}: {e}"))

    def c1():
        a = funcs["ex1"](np.arange(6, dtype=np.float64).reshape(2, 3))
        norms = np.linalg.norm(a, axis=1)
        return (a.dtype == np.float32 and a.flags["C_CONTIGUOUS"] and np.allclose(norms, 1.0),
                "float32・C連続・行ノルム1")

    def c2():
        idx = funcs["ex2"](xb)
        qn = funcs["ex1"](xb[:3])
        D, _ = idx.search(np.ascontiguousarray(qn, dtype=np.float32), 1)
        return (idx.ntotal == len(xb) and float(D[0, 0]) > 0.99, "IP+正規化=コサイン(self≈1)")

    def c3():
        ii = funcs["ex3"](xb, ids, xq, 5)
        return (ii.shape == (5, 5) and int(ii[0, 0]) == int(ids[0]), "IDMapで任意ID検索(top1=self)")

    def c4():
        meta = {10: {"modality": "image"}, 11: {"modality": "text"}, 12: {"modality": "image"}}
        out = funcs["ex4"]([-1, 11, 10, 12], [0.9, 0.8, 0.7, 0.6], meta, "image", 2)
        return (out == [(10, 0.7), (12, 0.6)], f"modality後フィルタ -> {out}")

    def c5():
        base = faiss.IndexFlatIP(d)
        base.add(np.ascontiguousarray(xb, dtype=np.float32))
        nt = funcs["ex5"](base, pathlib.Path("outputs/42_multimodal_vector_search/ex_index.faiss"))
        return (nt == len(xb), "write/read 永続化")

    def c6():
        val = funcs["ex6"]([1, 2, 9, 8, 5], {1, 2, 3, 4, 5}, 5)  # 上位5に正解3個 / 正解総数5
        return (abs(val - 3 / 5) < 1e-9, f"Recall@5={val:.3f}")

    def c7():
        # 上位: [rel, x, rel, x] → AP = (1/1 + 2/3)/min(2,4) = (1.6667)/2 = 0.8333
        val = funcs["ex7"]([1, 9, 2, 8], {1, 2}, 4)
        return (abs(val - ((1 / 1 + 2 / 3) / 2)) < 1e-9, f"AP@4={val:.3f}")

    def c8():
        # クエリ=クラスタ0の中心付近 → 上位k はクラスタ0の点（label==0）が占めるはず
        q = xb[0]
        top = funcs["ex8"](q, xb, 5)
        return (len(top) == 5 and all(labels[i] == labels[0] for i in top),
                f"crossmodal top-k 同クラスタ={[int(labels[i]) for i in top]}")

    check("ex1_l2_normalize_rows", c1)
    check("ex2_build_cosine_index", c2)
    check("ex3_idmap_search", c3)
    check("ex4_filter_by_modality", c4)
    check("ex5_save_reload", c5)
    check("ex6_recall_at_k", c6)
    check("ex7_average_precision_at_k", c7)
    check("ex8_crossmodal_topk", c8)

    print("=== 採点結果 ===")
    all_ok = True
    for name, ok, detail in results:
        mark = "PASS" if ok else "FAIL"
        all_ok = all_ok and ok
        print(f"  [{mark}] {name:28s} {detail}")
    print("\nALL PASS 🎉" if all_ok else "\nまだ未達の演習があります。TODO を埋めましょう。")
    return all_ok


def current_funcs() -> dict:
    """このファイルで定義された ex* を採点ランナーへ渡す形にまとめる。"""
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
    grade(current_funcs())
