"""第17回 演習問題（FAISS ベクトルDBと画像検索）。全10問・易→難。

使い方:
  1. 各 exN_*() の中の TODO を自分で実装する（最初は NotImplementedError で FAIL）。
  2. 自己採点:
         uv run python lectures/17_faiss_image_search/exercises.py
     未実装でも例外で落とさず PASS/FAIL を表示し、必ず exit 0 で終わる。
  3. どうしても分からない時だけ模範解答の挙動を見る（2通り）:
         SHOW_SOLUTION=1 uv run python lectures/17_faiss_image_search/exercises.py
         uv run python lectures/17_faiss_image_search/exercises_solutions.py

テーマ（易→難）:
  ex1  float32・C連続への変換（FAISS の入力作法）
  ex2  正規化 + IndexFlatIP = コサイン検索
  ex3  IndexIDMap で任意 int64 ID を付けて検索
  ex4  write_index / read_index による永続化
  ex5  Recall@k を集合一致で自前計算
  ex6  IVFFlat（train 必須・nprobe スイープ）
  ex7  HNSW（train 不要・efSearch ダイヤル）
  ex8  検索結果 ID → メタデータ参照と -1 ガード
  ex9  index_factory で PQ 圧縮インデックスを組む
  ex10 retrieval mAP の素 — Average Precision@k
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


def ex6_ivf_search(
    xb: np.ndarray, xq: np.ndarray, k: int, nlist: int, nprobe: int
) -> np.ndarray:
    """演習6: IVFFlat（転置インデックス）を train してから検索し、ID 行列を返す。

    手順: xb・xq を L2 正規化 → quantizer=IndexFlatIP(d) → IndexIVFFlat(quantizer, d,
          nlist, METRIC_INNER_PRODUCT) → train(xb) → add(xb) → nprobe を設定 → search。
    ポイント: train を忘れて add すると例外。nprobe が精度↔速度のダイヤル。
    """
    # TODO: IVFFlat を train→add→search し、ID 行列 I を返す（nprobe を設定すること）
    raise NotImplementedError


def ex7_hnsw_search(xb: np.ndarray, xq: np.ndarray, k: int, ef: int) -> np.ndarray:
    """演習7: HNSW（近傍グラフ・train 不要）で検索し、ID 行列を返す。

    手順: 正規化 → IndexHNSWFlat(d, 32, METRIC_INNER_PRODUCT) → efConstruction=40 →
          add(xb)（train 不要） → hnsw.efSearch=ef → search(xq, k)。
    ポイント: efSearch が速度↔精度の主ダイヤル（IVF の nprobe に相当）。
    """
    # TODO: HNSW を add→efSearch 設定→search し、ID 行列 I を返す
    raise NotImplementedError


def ex8_guard_lookup(result_ids, id_to_label: dict) -> list[str]:
    """演習8: 検索結果の ID 列をラベルに変換する。-1 は '(none)' にガードする。

    各 id について: id < 0 なら '(none)'、それ以外は id_to_label から引く
    （未登録は '(missing)'）。-1 をそのまま dict 参照に使わないこと。
    """
    # TODO: i<0 を早期 return（continue）でガードしつつラベル列を作って返す
    raise NotImplementedError


def ex9_factory_index(xb: np.ndarray, spec: str) -> faiss.Index:
    """演習9: index_factory で PQ 圧縮インデックスを組み、train→add して返す。

    手順: xb を L2 正規化 → faiss.index_factory(d, spec, METRIC_INNER_PRODUCT) →
          train(xb) → add(xb) → 返す。
    例の spec: "IVF16,PQ8"（PQ でベクトルを量子化してメモリ削減）。train 必須。
    """
    # TODO: index_factory で index を作り、train→add して返す
    raise NotImplementedError


def ex10_average_precision(ranked_labels, query_label, k: int) -> float:
    """演習10: retrieval mAP の素となる Average Precision@k を計算する。

    ranked_labels: 類似度降順に並べた候補のラベル列（クエリ自身は除いてある）。
    query_label と一致する候補が「正例」。
    AP = (各正例位置までの precision の平均)。k 内に正例が無ければ 0.0。
    例: labels=[1,0,1,1,0], query=1, k=5 → ranks1,3,4 が正例
        → (1/1 + 2/3 + 3/4)/3 ≈ 0.806。
    """
    # TODO: 上位k を走査し、正例に当たるたび hits/rank を足して hits で割る
    raise NotImplementedError


# =====================================================================
# 自己採点ランナー（exercises_solutions.py からも再利用される）
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

    def _c6():
        # nprobe=nlist=16 なら実質全セル探索 → 自分自身が top-1 になるはず
        ii = ex6_ivf_search(xb, xq, k, nlist=16, nprobe=16)
        return (ii.shape == (5, k) and int(ii[0, 0]) == 0, "IVFFlat train+nprobe")

    def _c7():
        ii = ex7_hnsw_search(xb, xq, k, ef=64)
        return (ii.shape == (5, k) and int(ii[0, 0]) == 0, "HNSW efSearch")

    def _c8():
        rids = np.array([100, 101, -1, 103])
        mapping = {100: "a", 101: "b", 103: "d"}
        labs = ex8_guard_lookup(rids, mapping)
        return (labs == ["a", "b", "(none)", "d"], "-1 ガード付き lookup")

    def _c9():
        idx = ex9_factory_index(xb, "IVF16,PQ8")
        return (idx.is_trained and idx.ntotal == len(xb), "index_factory(PQ) train+add")

    def _c10():
        val = ex10_average_precision([1, 0, 1, 1, 0], 1, 5)
        # (1/1 + 2/3 + 3/4)/3
        return (abs(val - ((1.0 + 2 / 3 + 3 / 4) / 3)) < 1e-6, f"AP@5={val:.3f}")

    check("ex1_to_faiss_array", _c1)
    check("ex2_cosine_index", _c2)
    check("ex3_idmap_search", _c3)
    check("ex4_save_load", _c4)
    check("ex5_recall_at_k", _c5)
    check("ex6_ivf_search", _c6)
    check("ex7_hnsw_search", _c7)
    check("ex8_guard_lookup", _c8)
    check("ex9_factory_index", _c9)
    check("ex10_average_precision", _c10)

    print("=== 採点結果 ===")
    all_ok = True
    for name, ok, detail in results:
        mark = "PASS" if ok else "FAIL"
        all_ok = all_ok and ok
        print(f"  [{mark}] {name:24s} {detail}")
    n_pass = sum(1 for _, ok, _ in results if ok)
    print(f"\n{n_pass}/{len(results)} PASS")
    print("ALL PASS 🎉" if all_ok else "まだ未達の演習があります。TODO を埋めましょう。")


# =====================================================================
# 模範解答（SHOW_SOLUTION=1 / exercises_solutions.py で本体へ差し替え）
# まずは自力で！
# =====================================================================

def _norm(a: np.ndarray) -> np.ndarray:
    """L2 正規化した float32・C連続コピー（模範解答の共通前処理）。"""
    a = np.ascontiguousarray(a, dtype=np.float32)
    n = np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
    return np.ascontiguousarray(a / n, dtype=np.float32)


def _sol_ex1(x):
    return np.ascontiguousarray(x, dtype=np.float32)


def _sol_ex2(xb):
    xn = _norm(xb)
    index = faiss.IndexFlatIP(xn.shape[1])
    index.add(xn)
    return index


def _sol_ex3(xb, ids, xq, k):
    index = faiss.IndexIDMap(faiss.IndexFlatIP(xb.shape[1]))
    index.add_with_ids(_norm(xb), ids.astype(np.int64))
    _, I = index.search(_norm(xq), k)  # noqa: E741
    return I


def _sol_ex4(index, path):
    faiss.write_index(index, str(path))
    loaded = faiss.read_index(str(path))
    return loaded.ntotal


def _sol_ex5(gt_ids, ann_ids, k):
    vals = [np.intersect1d(g[:k], a[:k]).size / k for g, a in zip(gt_ids, ann_ids)]
    return float(np.mean(vals))


def _sol_ex6(xb, xq, k, nlist, nprobe):
    xb_n, xq_n = _norm(xb), _norm(xq)
    d = xb_n.shape[1]
    quantizer = faiss.IndexFlatIP(d)
    ivf = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
    ivf.train(xb_n)  # ← これを忘れて add すると例外
    ivf.add(xb_n)
    ivf.nprobe = nprobe
    _, I = ivf.search(xq_n, k)  # noqa: E741
    return I


def _sol_ex7(xb, xq, k, ef):
    xb_n, xq_n = _norm(xb), _norm(xq)
    hnsw = faiss.IndexHNSWFlat(xb_n.shape[1], 32, faiss.METRIC_INNER_PRODUCT)
    hnsw.hnsw.efConstruction = 40
    hnsw.add(xb_n)  # train 不要
    hnsw.hnsw.efSearch = ef
    _, I = hnsw.search(xq_n, k)  # noqa: E741
    return I


def _sol_ex8(result_ids, id_to_label):
    out = []
    for i in result_ids:
        i = int(i)
        if i < 0:  # -1 ガード
            out.append("(none)")
            continue
        out.append(id_to_label.get(i, "(missing)"))
    return out


def _sol_ex9(xb, spec):
    xb_n = _norm(xb)
    idx = faiss.index_factory(xb_n.shape[1], spec, faiss.METRIC_INNER_PRODUCT)
    idx.train(xb_n)
    idx.add(xb_n)
    return idx


def _sol_ex10(ranked_labels, query_label, k):
    hits = 0
    score = 0.0
    for rank, lab in enumerate(ranked_labels[:k], start=1):
        if lab == query_label:
            hits += 1
            score += hits / rank
    return score / hits if hits else 0.0


def _install_solutions() -> None:
    """模範解答を ex*_ 名に差し込む（採点ロジック _grade はそのまま再利用される）。"""
    g = globals()
    g["ex1_to_faiss_array"] = _sol_ex1
    g["ex2_cosine_index"] = _sol_ex2
    g["ex3_idmap_search"] = _sol_ex3
    g["ex4_save_load"] = _sol_ex4
    g["ex5_recall_at_k"] = _sol_ex5
    g["ex6_ivf_search"] = _sol_ex6
    g["ex7_hnsw_search"] = _sol_ex7
    g["ex8_guard_lookup"] = _sol_ex8
    g["ex9_factory_index"] = _sol_ex9
    g["ex10_average_precision"] = _sol_ex10


if __name__ == "__main__":
    if os.environ.get("SHOW_SOLUTION") == "1":
        print("(模範解答モードで実行します)\n")
        _install_solutions()
    _grade()
