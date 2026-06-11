"""第33回 演習問題（マルチモーダル埋め込み / SigLIP・SigLIP2・クロスモーダル検索）。

使い方:
  1. 各 exN_*() の中の TODO を自分で実装する（最初は NotImplementedError で FAIL）。
  2. 自己採点（未実装でも例外で落とさず PASS/FAIL を表示し、必ず exit 0）:
         uv run python lectures/33_multimodal_embeddings/exercises.py
  3. 全問の模範解答（全 PASS）はこちら:
         uv run python lectures/33_multimodal_embeddings/exercises_solutions.py

テーマ（易→難）:
  1 sigmoid   2 行softmax   3 L2正規化   4 コサイン類似度行列
  5 コサインindex(FAISS)    6 text->image top1   7 Recall@k
  8 AP@k(mAPの素)          9 クロスモーダル(音->画像) 平均Recall@k

採点は決定的な合成ベクトルだけで行うので、モデル DL 不要で速く再現的。
本物の SigLIP/SigLIP2 連携は 01〜05・mini_project.py を参照。
"""

from __future__ import annotations

import faiss
import numpy as np


# =====================================================================
# 演習（ここを実装する）
# =====================================================================

def ex1_sigmoid(x: np.ndarray) -> np.ndarray:
    """演習1: 要素ごとの sigmoid = 1/(1+exp(-x)) を返す（SigLIP の確率化）。

    各要素が独立に 0〜1 に写る。ラベル間で和は 1 にならない（多ラベル可）のが CLIP との違い。
    """
    # TODO: 1.0 / (1.0 + np.exp(-x)) を返す
    raise NotImplementedError


def ex2_softmax_rows(x: np.ndarray) -> np.ndarray:
    """演習2: 行ごとの softmax（各行の和が 1）を返す（CLIP の確率化）。

    オーバーフロー回避のため、各行から行最大値を引いてから exp すること。
    """
    # TODO: x - 行max → exp → 行和で割る
    raise NotImplementedError


def ex3_l2_normalize_rows(x: np.ndarray) -> np.ndarray:
    """演習3: 各行を L2 正規化した float32・C連続 配列を返す。

    正規化後に内積を取ると = コサイン類似度。ゼロ割回避にノルムは下限 1e-12 でクリップ。
    """
    # TODO: float32化 → 行ノルム(np.maximum(.,1e-12))で割る → C連続で返す
    raise NotImplementedError


def ex4_cosine_sim_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """演習4: 2つの行列 a(A,d), b(B,d) の全ペアのコサイン類似度行列 (A,B) を返す。

    手順: a・b をそれぞれ行ごと L2 正規化 → a_n @ b_n.T。
    （これがテキスト×画像のクロスモーダル類似度の中身）
    """
    # TODO: ex3 で正規化 → 内積
    raise NotImplementedError


def ex5_build_cosine_index(xb: np.ndarray) -> faiss.Index:
    """演習5: コサイン類似度で検索できる Flat インデックスを作って返す（add 済み）。

    手順: xb を L2 正規化 → faiss.IndexFlatIP(d) → add。（正規化 + 内積 = コサイン）
    """
    # TODO: 正規化 → IndexFlatIP(d) に add した index を返す
    raise NotImplementedError


def ex6_text_to_image_top1(text_emb: np.ndarray, image_emb: np.ndarray) -> list[int]:
    """演習6: 各テキスト埋め込みに最も近い画像の「行インデックス」を返す（クロスモーダル top1）。

    手順: 双方を L2 正規化 → コサイン = text_n @ image_n.T → 各行 argmax。
    返り値は長さ = テキスト件数 の int リスト。FAISS でも numpy でもよい。
    """
    # TODO: 正規化 → 内積 → 行ごと argmax を list で返す
    raise NotImplementedError


def ex7_recall_at_k(retrieved_ids: list[int], relevant_ids: set[int], k: int) -> float:
    """演習7: 1クエリの Recall@k = 上位k件に入った正解数 / 正解総数 を返す。

    relevant_ids が空なら 0.0（0除算回避）。retrieved_ids[:k] と relevant の積集合サイズで計算。
    """
    # TODO: len(set(retrieved_ids[:k]) & relevant_ids) / len(relevant_ids)
    raise NotImplementedError


def ex8_average_precision_at_k(retrieved_ids: list[int], relevant_ids: set[int], k: int) -> float:
    """演習8: 1クエリの AP@k を返す（mAP の素）。

    上位から順に見て、正解を引くたびに「その時点の precision = hits/rank」を加算し、
    最後に min(正解総数, k) で割る。relevant_ids が空なら 0.0。
    """
    # TODO: rank=1..k を走査し hits を数えながら hits/rank を加算、min(len(rel),k) で割る
    raise NotImplementedError


def ex9_crossmodal_recall_at_k(
    query_emb: np.ndarray,
    db_emb: np.ndarray,
    query_concepts: list[str],
    db_concepts: list[str],
    k: int,
) -> float:
    """演習9: クロスモーダル検索（例: 音→画像）の「平均 Recall@k」を返す。

    - query_emb(Q,d) を db_emb(N,d) のコサイン最近傍で検索する。
    - クエリ q の正解 = db のうち concept が一致する行すべて。
    - 各クエリの Recall@k を ex7 で計算し、その平均（mean Recall@k）を返す。
    手順: db を正規化して IndexFlatIP に add → query を正規化して search(k=全件)
          → 各クエリで relevant 集合を作り ex7_recall_at_k を適用 → 平均。
    """
    # TODO: index 構築 → 検索 → 各クエリ relevant を concept 一致で作り Recall@k を平均
    raise NotImplementedError


# =====================================================================
# 自己採点ランナー（exercises_solutions.py からも import して再利用）
# =====================================================================

def make_trimodal_toy(concepts: list[str], dim: int = 32, noise: float = 0.12, seed: int = 0):
    """採点用の決定的トイ三モーダル空間（概念アンカー + モダリティ別ノイズ）。"""
    rng = np.random.default_rng(seed)
    n = len(concepts)
    anchor = rng.standard_normal((n, dim)).astype(np.float32)
    anchor /= np.maximum(np.linalg.norm(anchor, axis=1, keepdims=True), 1e-12)

    def enc():
        v = anchor + noise * rng.standard_normal((n, dim)).astype(np.float32)
        return np.ascontiguousarray(v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12),
                                    dtype=np.float32)

    return anchor, enc(), enc()  # anchor, image, audio


def grade(funcs: dict) -> bool:
    """funcs（名前→関数）を採点して PASS/FAIL を表示。全PASSなら True。"""
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
        out = funcs["ex1"](np.array([0.0, 20.0, -20.0], dtype=np.float32))
        return (abs(out[0] - 0.5) < 1e-6 and out[1] > 0.999 and out[2] < 1e-3,
                f"sigmoid([0,+,-])={np.round(out, 3).tolist()}")

    def c2():
        out = funcs["ex2"](np.array([[1.0, 2.0, 3.0]], dtype=np.float32))
        return (abs(float(out.sum()) - 1.0) < 1e-6 and int(np.argmax(out[0])) == 2,
                f"softmax 行和={float(out.sum()):.3f}")

    def c3():
        a = funcs["ex3"](np.arange(6, dtype=np.float64).reshape(2, 3))
        norms = np.linalg.norm(a, axis=1)
        return (a.dtype == np.float32 and a.flags["C_CONTIGUOUS"] and np.allclose(norms, 1.0),
                "float32・C連続・行ノルム1")

    def c4():
        a = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        m = funcs["ex4"](a, a)
        return (m.shape == (2, 2) and abs(m[0, 0] - 1.0) < 1e-6 and abs(m[0, 1]) < 1e-6,
                "コサイン行列(自分=1,直交=0)")

    def c5():
        _anchor, img, _aud = make_trimodal_toy(["a", "b", "c", "d", "e"], seed=1)
        idx = funcs["ex5"](img)
        qn = funcs["ex3"](img[:3]) if "ex3" in funcs else img[:3]
        D, _ = idx.search(np.ascontiguousarray(qn, dtype=np.float32), 1)
        return (idx.ntotal == len(img) and float(D[0, 0]) > 0.99, "IP+正規化=コサイン(self≈1)")

    def c6():
        _anchor, img, aud = make_trimodal_toy(["a", "b", "c", "d", "e", "f"], seed=2)
        top = funcs["ex6"](aud, img)  # audio クエリで画像 top1
        return (top == list(range(len(img))), f"text/audio->image top1={top}")

    def c7():
        val = funcs["ex7"]([1, 2, 9, 8, 5], {1, 2, 3, 4, 5}, 5)  # 上位5に正解3個 / 正解総数5
        return (abs(val - 3 / 5) < 1e-9, f"Recall@5={val:.3f}")

    def c8():
        # 上位: [rel, x, rel, x] → AP=(1/1 + 2/3)/min(2,4)=0.8333
        val = funcs["ex8"]([1, 9, 2, 8], {1, 2}, 4)
        return (abs(val - ((1 / 1 + 2 / 3) / 2)) < 1e-9, f"AP@4={val:.3f}")

    def c9():
        # 各概念2枚ずつの画像DB + 同概念の音クエリ → Recall@2 は満点に近いはず
        concepts = ["dog", "bird", "bell", "drum"]
        db_concepts = [c for c in concepts for _ in range(2)]
        rng = np.random.default_rng(3)
        dim = 32
        base = {c: rng.standard_normal(dim).astype(np.float32) for c in concepts}
        db = np.stack([base[c] + 0.05 * rng.standard_normal(dim).astype(np.float32) for c in db_concepts])
        q = np.stack([base[c] + 0.05 * rng.standard_normal(dim).astype(np.float32) for c in concepts])
        val = funcs["ex9"](q, db, concepts, db_concepts, k=2)
        return (val > 0.99, f"音->画像 mean Recall@2={val:.3f}")

    check("ex1_sigmoid", c1)
    check("ex2_softmax_rows", c2)
    check("ex3_l2_normalize_rows", c3)
    check("ex4_cosine_sim_matrix", c4)
    check("ex5_build_cosine_index", c5)
    check("ex6_text_to_image_top1", c6)
    check("ex7_recall_at_k", c7)
    check("ex8_average_precision_at_k", c8)
    check("ex9_crossmodal_recall_at_k", c9)

    print("=== 採点結果 ===")
    all_ok = True
    for name, ok, detail in results:
        mark = "PASS" if ok else "FAIL"
        all_ok = all_ok and ok
        print(f"  [{mark}] {name:30s} {detail}")
    print("\nALL PASS 🎉" if all_ok else "\nまだ未達の演習があります。TODO を埋めましょう。")
    return all_ok


def current_funcs() -> dict:
    """このファイルで定義された ex* を採点ランナーへ渡す形にまとめる。"""
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
    grade(current_funcs())
