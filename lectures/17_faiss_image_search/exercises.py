"""第17回 演習問題（FAISS ベクトルDBと画像検索）。

使い方:
  1. 各 exN_*() の中の TODO を自分で実装する（最初は NotImplementedError で FAIL）。
  2. 自己採点:
         uv run python lectures/17_faiss_image_search/exercises.py
     未実装でも例外で落とさず PASS/FAIL を表示し、必ず exit 0 で終わる。
  3. どうしても分からない時だけ模範解答の挙動を見る:
         SHOW_SOLUTION=1 uv run python lectures/17_faiss_image_search/exercises.py

テーマ: 正規化→IP=コサイン / float32・C連続 / IndexIDMap / write・read / Recall@k。
"""

from __future__ import annotations

import os
import pathlib
import sys

import faiss
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from search_helpers import make_clustered_vectors, output_dir  # noqa: E402


# =====================================================================
# 演習（ここを実装する）
# =====================================================================

def ex1_to_faiss_array(x: np.ndarray) -> np.ndarray:
    """演習1: 任意の配列を FAISS 仕様（float32・C連続）に変換して返す。

    例: float64 や非連続なスライスを渡されても、dtype==float32 かつ
    flags['C_CONTIGUOUS']==True の配列にして返すこと。
    """
    # TODO: np.ascontiguousarray(x, dtype=np.float32) を返す
    raise NotImplementedError


def ex2_cosine_index(xb: np.ndarray) -> faiss.Index:
    """演習2: コサイン類似度で検索できる Flat インデックスを作って返す。

    手順: xb を L2 正規化 → IndexFlatIP(d) を作る → add する。
    （正規化した上で内積を使うと、内積=コサイン類似度になる）
    返り値は add 済みの index。
    """
    # TODO: xb を float32 化し、各行を L2 正規化してから IndexFlatIP に add する
    raise NotImplementedError


def ex3_idmap_search(xb: np.ndarray, ids: np.ndarray, xq: np.ndarray, k: int) -> np.ndarray:
    """演習3: IndexIDMap で任意IDを付けて検索し、近傍の「ID行列」を返す。

    手順: 正規化 → IndexFlatIP を IndexIDMap で包む → add_with_ids(xb, ids)
          → search(xq, k) の返り値のうち ID 行列 I を返す。
    （xb, xq は同じ前処理で正規化すること）
    """
    # TODO: IndexIDMap(IndexFlatIP(d)) を作り add_with_ids、search して I を返す
    raise NotImplementedError


def ex4_save_load(index: faiss.Index, path: pathlib.Path) -> int:
    """演習4: index を path に保存し、読み戻した index の ntotal を返す。

    write_index で保存 → read_index で別オブジェクトとして読み込み → その ntotal を返す。
    """
    # TODO: faiss.write_index / faiss.read_index を使う
    raise NotImplementedError


def ex5_recall_at_k(gt_ids: np.ndarray, ann_ids: np.ndarray, k: int) -> float:
    """演習5: Recall@k を集合一致で計算する。

    各クエリ行について「gt_ids[:k] と ann_ids[:k] の共通要素数 / k」を求め、平均する。
    np.intersect1d を使うと -1（無効ID）も自然に無視できる。
    """
    # TODO: 各行で np.intersect1d(gt[:k], ann[:k]).size / k を平均して返す
    raise NotImplementedError


# =====================================================================
# 自己採点ランナー
# =====================================================================

def _grade() -> None:
    out = output_dir()
    xb, _ = make_clustered_vectors(n=500, d=16, n_clusters=5, noise=0.7, seed=10)
    ids = np.arange(100, 100 + len(xb)).astype(np.int64)
    xq = xb[:5].copy()
    k = 5
    results: list[tuple[str, bool, str]] = []

    def check(name: str, fn) -> None:
        try:
            ok, detail = fn()
            results.append((name, bool(ok), detail))
        except NotImplementedError:
            results.append((name, False, "未実装（TODOを埋めてください）"))
        except Exception as e:  # noqa: BLE001
            results.append((name, False, f"例外: {type(e).__name__}: {e}"))

    def _c1():
        a = ex1_to_faiss_array(np.arange(6, dtype=np.float64).reshape(2, 3))
        return (a.dtype == np.float32 and a.flags["C_CONTIGUOUS"], "float32・C連続")

    def _c2():
        idx = ex2_cosine_index(xb)
        D, _ = idx.search(np.ascontiguousarray(xb[:3] / np.linalg.norm(xb[:3], axis=1, keepdims=True),
                                               dtype=np.float32), 1)
        # 自分自身が最近傍で、コサイン≈1.0 になるはず
        return (idx.ntotal == len(xb) and float(D[0, 0]) > 0.99, "IP+正規化=コサイン")

    def _c3():
        ii = ex3_idmap_search(xb, ids, xq, k)
        # 先頭の近傍が自分自身の ID（=ids[クエリ番号]）になっているはず
        return (ii.shape == (5, k) and int(ii[0, 0]) == int(ids[0]), "IDMapで任意ID検索")

    def _c4():
        base = faiss.IndexFlatIP(16)
        base.add(np.ascontiguousarray(xb, dtype=np.float32))
        nt = ex4_save_load(base, out / "ex_index.faiss")
        return (nt == len(xb), "write/read 永続化")

    def _c5():
        gt = np.array([[1, 2, 3, 4, 5], [10, 11, 12, 13, 14]])
        ann = np.array([[1, 2, 9, 8, 5], [10, 99, 12, -1, -1]])  # 行0:3一致, 行1:2一致
        val = ex5_recall_at_k(gt, ann, 5)
        return (abs(val - ((3 / 5 + 2 / 5) / 2)) < 1e-6, f"Recall@5={val:.3f}")

    check("ex1_to_faiss_array", _c1)
    check("ex2_cosine_index", _c2)
    check("ex3_idmap_search", _c3)
    check("ex4_save_load", _c4)
    check("ex5_recall_at_k", _c5)

    print("=== 採点結果 ===")
    all_ok = True
    for name, ok, detail in results:
        mark = "PASS" if ok else "FAIL"
        all_ok = all_ok and ok
        print(f"  [{mark}] {name:20s} {detail}")
    print("\nALL PASS 🎉" if all_ok else "\nまだ未達の演習があります。TODO を埋めましょう。")


# =====================================================================
# 模範解答（SHOW_SOLUTION=1 のとき本体へ差し替えて実行）
# まずは自力で！
# =====================================================================

def _sol_ex1(x):
    return np.ascontiguousarray(x, dtype=np.float32)


def _sol_ex2(xb):
    xb = np.ascontiguousarray(xb, dtype=np.float32)
    xn = xb / np.maximum(np.linalg.norm(xb, axis=1, keepdims=True), 1e-12)
    xn = np.ascontiguousarray(xn, dtype=np.float32)
    index = faiss.IndexFlatIP(xn.shape[1])
    index.add(xn)
    return index


def _sol_ex3(xb, ids, xq, k):
    def norm(a):
        a = np.ascontiguousarray(a, dtype=np.float32)
        return np.ascontiguousarray(a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12),
                                    dtype=np.float32)

    index = faiss.IndexIDMap(faiss.IndexFlatIP(xb.shape[1]))
    index.add_with_ids(norm(xb), ids.astype(np.int64))
    _, I = index.search(norm(xq), k)  # noqa: E741
    return I


def _sol_ex4(index, path):
    faiss.write_index(index, str(path))
    loaded = faiss.read_index(str(path))
    return loaded.ntotal


def _sol_ex5(gt_ids, ann_ids, k):
    vals = [np.intersect1d(g[:k], a[:k]).size / k for g, a in zip(gt_ids, ann_ids)]
    return float(np.mean(vals))


def _install_solutions() -> None:
    g = globals()
    g["ex1_to_faiss_array"] = _sol_ex1
    g["ex2_cosine_index"] = _sol_ex2
    g["ex3_idmap_search"] = _sol_ex3
    g["ex4_save_load"] = _sol_ex4
    g["ex5_recall_at_k"] = _sol_ex5


if __name__ == "__main__":
    if os.environ.get("SHOW_SOLUTION") == "1":
        print("(模範解答モードで実行します)\n")
        _install_solutions()
    _grade()
