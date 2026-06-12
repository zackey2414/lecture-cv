"""第45回 演習問題（手書きスケッチで絵文字検索 SBIR）。全8問・易→難。

使い方:
  1. 各 exN_*() の中の TODO を自分で実装する（最初は NotImplementedError で FAIL）。
  2. 自己採点:
         uv run python lectures/45_sketch_emoji_search/exercises.py
     未実装でも例外で落とさず PASS/FAIL を表示し、必ず exit 0 で終わる。
  3. どうしても分からない時だけ模範解答の挙動を見る（2通り）:
         SHOW_SOLUTION=1 uv run python lectures/45_sketch_emoji_search/exercises.py
         uv run python lectures/45_sketch_emoji_search/exercises_solutions.py

テーマ（易→難）:
  ex1  float32・C連続への変換（FAISS の入力作法）
  ex2  各行を L2 正規化（コサイン検索の前処理）
  ex3  グレースケール化して 3ch に戻す（スケッチへドメインを寄せる）
  ex4  正方化の白パディング（縦横比を保ったまま正方形へ）
  ex5  正規化 + IndexFlatIP + IndexIDMap で任意IDのコサイン検索
  ex6  検索結果 ID → ラベル（-1 を (none) でガード）
  ex7  top-N ヒット判定（正しいラベルが上位N件に入ったか）
  ex8  Recall@N（= top-N ヒット率）を複数クエリで平均
"""

from __future__ import annotations

import os
import pathlib
import sys

import faiss
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))


# =====================================================================
# 演習（ここを実装する）
# =====================================================================


def ex1_to_faiss_array(x: np.ndarray) -> np.ndarray:
    """演習1: 任意の配列を FAISS 仕様（float32・C連続）に変換して返す。

    float64 や非連続なスライスを渡されても、dtype==float32 かつ
    flags['C_CONTIGUOUS']==True の配列にして返すこと。
    """
    # TODO: np.ascontiguousarray(x, dtype=np.float32) を返す
    raise NotImplementedError


def ex2_l2_normalize_rows(x: np.ndarray) -> np.ndarray:
    """演習2: 2次元配列の各行を L2 ノルム1に正規化して返す（float32）。

    内積（IndexFlatIP）でコサイン類似度を測るための前処理。0除算に注意
    （ノルムを下限 1e-12 でクリップする）。
    """
    # TODO: 各行を norm で割る。np.linalg.norm(x, axis=1, keepdims=True) を使う
    raise NotImplementedError


def ex3_grayscale_rgb(img):
    """演習3: PIL 画像をグレースケール化し、3ch（RGB）に戻して返す。

    スケッチ（白地に黒線）と絵文字のドメインを寄せる定番処理。返り値の各画素は
    R==G==B（グレー）になっていること。mode は 'RGB'。
    """
    # TODO: img.convert("L") してから convert("RGB") して返す
    raise NotImplementedError


def ex4_square_pad(arr: np.ndarray, fill: int = 255) -> np.ndarray:
    """演習4: 2次元配列を長辺基準で正方形に中央パディングして返す。

    返り値は (s, s)（s=max(h,w)）。元の領域は中央に置き、余白は fill で埋める。
    スケッチを縦横比を保ったまま正方化する前処理に使う。
    """
    # TODO: s=max(h,w) の白(=fill)配列を作り、中央に arr を貼る
    raise NotImplementedError


def ex5_cosine_search_ids(xb: np.ndarray, ids: np.ndarray, xq: np.ndarray, k: int) -> np.ndarray:
    """演習5: 正規化 + IndexFlatIP + IndexIDMap で任意IDのコサイン検索を行い、ID行列を返す。

    手順: xb・xq を L2 正規化 → IndexIDMap(IndexFlatIP(d)) → add_with_ids(xb, ids)
          → search(xq, k) の ID 行列 I を返す。xb と xq は同じ正規化を通すこと。
    """
    # TODO: 正規化 → IndexIDMap(IndexFlatIP) → add_with_ids → search して I を返す
    raise NotImplementedError


def ex6_guard_lookup(result_ids, id_to_label: dict) -> list:
    """演習6: 検索結果の ID 列をラベルに変換する。-1 は '(none)' にガードする。

    各 id について: id < 0 なら '(none)'、それ以外は id_to_label から引く
    （未登録は '(missing)'）。-1 をそのまま dict 参照に使わないこと。
    """
    # TODO: i<0 を早期にガードしつつラベル列を作って返す
    raise NotImplementedError


def ex7_topn_hit(ranked_labels, true_label, n: int) -> bool:
    """演習7: 正しいラベルが上位 n 件に入っているか（top-N ヒット）を判定する。

    ranked_labels: 類似度降順に並んだラベル列。true_label がその先頭 n 件に
    含まれていれば True。
    """
    # TODO: true_label in ranked_labels[:n] を返す
    raise NotImplementedError


def ex8_recall_at_n(pairs, n: int) -> float:
    """演習8: 複数クエリの Recall@N（= top-N ヒット率）を平均して返す。

    pairs: (ranked_labels, true_label) のリスト。各クエリで「true_label が上位 n 件に
    入っていれば 1、なければ 0」とし、その平均（0〜1）を返す。pairs が空なら 0.0。
    """
    # TODO: 各 pair で top-N ヒットを判定し、平均する（ex7 を使ってよい）
    raise NotImplementedError


# =====================================================================
# 自己採点ランナー（exercises_solutions.py からも再利用される）
# =====================================================================


def _grade() -> None:
    from PIL import Image

    rng = np.random.default_rng(0)
    xb = rng.standard_normal((12, 8)).astype(np.float64)
    ids = np.arange(200, 212).astype(np.int64)
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
        a = ex2_l2_normalize_rows(xb)
        norms = np.linalg.norm(a, axis=1)
        return (np.allclose(norms, 1.0, atol=1e-5), "各行ノルム≈1")

    def _c3():
        out = ex3_grayscale_rgb(Image.new("RGB", (8, 8), (200, 50, 50)))
        arr = np.asarray(out)
        eq = arr.ndim == 3 and arr.shape[2] == 3 and np.array_equal(arr[..., 0], arr[..., 1]) \
            and np.array_equal(arr[..., 1], arr[..., 2])
        return (out.mode == "RGB" and eq, "RGBでR==G==B")

    def _c4():
        arr = np.zeros((2, 6), dtype=np.uint8)
        out = ex4_square_pad(arr, fill=255)
        # 6x6 になり、中央2行が0、上下2行が255で埋まっているはず
        ok = out.shape == (6, 6) and out[0, 0] == 255 and out[2, 0] == 0 and out[3, 0] == 0
        return (ok, f"{arr.shape}->{out.shape}")

    def _c5():
        I = ex5_cosine_search_ids(xb, ids, xb[:3], 1)  # noqa: E741
        # 自分自身がコサイン最大 → 先頭の近傍 ID は ids[クエリ番号]
        ok = I.shape == (3, 1) and int(I[0, 0]) == int(ids[0]) and int(I[1, 0]) == int(ids[1])
        return (ok, "自分が top-1")

    def _c6():
        labs = ex6_guard_lookup([200, -1, 211], {200: "happy", 211: "sad"})
        return (labs == ["happy", "(none)", "sad"], "-1 ガード lookup")

    def _c7():
        hit = ex7_topn_hit(["a", "b", "c", "d"], "c", 3)
        miss = ex7_topn_hit(["a", "b", "c", "d"], "c", 2)
        return (hit is True and miss is False, "top-N ヒット判定")

    def _c8():
        pairs = [(["a", "b"], "a"), (["a", "b"], "b"), (["c", "d"], "a")]
        val = ex8_recall_at_n(pairs, 1)  # hit, miss, miss → 1/3
        return (abs(val - 1 / 3) < 1e-9, f"Recall@1={val:.3f}")

    check("ex1_to_faiss_array", _c1)
    check("ex2_l2_normalize_rows", _c2)
    check("ex3_grayscale_rgb", _c3)
    check("ex4_square_pad", _c4)
    check("ex5_cosine_search_ids", _c5)
    check("ex6_guard_lookup", _c6)
    check("ex7_topn_hit", _c7)
    check("ex8_recall_at_n", _c8)

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


def _sol_ex1(x):
    return np.ascontiguousarray(x, dtype=np.float32)


def _sol_ex2(x):
    x = np.ascontiguousarray(x, dtype=np.float32)
    n = np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    return np.ascontiguousarray(x / n, dtype=np.float32)


def _sol_ex3(img):
    return img.convert("L").convert("RGB")


def _sol_ex4(arr, fill=255):
    h, w = arr.shape[:2]
    s = max(h, w)
    canvas = np.full((s, s), fill, dtype=arr.dtype)
    y, x = (s - h) // 2, (s - w) // 2
    canvas[y:y + h, x:x + w] = arr
    return canvas


def _sol_ex5(xb, ids, xq, k):
    xb_n, xq_n = _sol_ex2(xb), _sol_ex2(xq)
    index = faiss.IndexIDMap(faiss.IndexFlatIP(xb_n.shape[1]))
    index.add_with_ids(xb_n, ids.astype(np.int64))
    _, I = index.search(xq_n, k)  # noqa: E741
    return I


def _sol_ex6(result_ids, id_to_label):
    out = []
    for i in result_ids:
        i = int(i)
        if i < 0:  # -1 ガード
            out.append("(none)")
            continue
        out.append(id_to_label.get(i, "(missing)"))
    return out


def _sol_ex7(ranked_labels, true_label, n):
    return true_label in list(ranked_labels)[:n]


def _sol_ex8(pairs, n):
    if not pairs:
        return 0.0
    hits = [1.0 if _sol_ex7(rl, tl, n) else 0.0 for rl, tl in pairs]
    return float(np.mean(hits))


def _install_solutions() -> None:
    """模範解答を ex*_ 名に差し込む（採点ロジック _grade はそのまま再利用される）。"""
    g = globals()
    g["ex1_to_faiss_array"] = _sol_ex1
    g["ex2_l2_normalize_rows"] = _sol_ex2
    g["ex3_grayscale_rgb"] = _sol_ex3
    g["ex4_square_pad"] = _sol_ex4
    g["ex5_cosine_search_ids"] = _sol_ex5
    g["ex6_guard_lookup"] = _sol_ex6
    g["ex7_topn_hit"] = _sol_ex7
    g["ex8_recall_at_n"] = _sol_ex8


if __name__ == "__main__":
    if os.environ.get("SHOW_SOLUTION") == "1":
        print("(模範解答モードで実行します)\n")
        _install_solutions()
    _grade()
