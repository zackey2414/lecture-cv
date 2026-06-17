"""01: FAISS の基本ループ — IndexFlatL2 / IndexFlatIP と「正規化でコサイン」。

学ぶこと:
  - ベクトル検索の最小ループ: index を作る → add でDBを入れる → search で近傍を引く。
  - 入力は必ず float32 かつ C連続（np.ascontiguousarray）。これを破ると落ちる/誤動作。
  - IndexFlatL2 は L2 距離（小さいほど近い）、IndexFlatIP は内積（大きいほど近い）。
  - コサイン類似度は「L2 正規化してから内積(IP)」で得る。DB側もクエリ側も正規化する。
  - search(xq, k) は距離 D と ID I の2つの (nq, k) 行列を返す。

実行:
  uv run python lectures/17_faiss_image_search/01_flat_ip_cosine.py
結果は lectures/17_faiss_image_search/outputs/ に保存される。
"""

from __future__ import annotations

import pathlib
import sys

import faiss
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from search_helpers import as_faiss_array, l2_normalize, make_clustered_vectors, output_dir  # noqa: E402


def main() -> None:
    out = output_dir()
    print(f"faiss {faiss.__version__}")

    # === 1. データを用意（float32・C連続が絶対条件） ======================
    # 8クラスタに分かれた 2000 本の 32次元ベクトル。実務ではここが画像/テキストの埋め込み。
    d = 32
    xb, labels = make_clustered_vectors(n=2000, d=d, n_clusters=8, noise=0.8, seed=1)
    # FAISS は float32・C連続を要求する。make_clustered_vectors は既にその形だが、
    # 「index に渡す前に必ず通す」習慣として明示しておく。
    xb = as_faiss_array(xb)
    print("\n[1] DBベクトル", xb.shape, xb.dtype, "C連続:", xb.flags["C_CONTIGUOUS"])

    # クエリは DB の先頭5本を使う（自分自身が最近傍に来るはず＝動作確認になる）。
    xq = as_faiss_array(xb[:5].copy())
    k = 5

    # === 2. IndexFlatL2: 総当たりの L2 距離検索 ===========================
    # Flat = 近似なしの全件スキャン（厳密）。train 不要で、add した瞬間から検索できる。
    index_l2 = faiss.IndexFlatL2(d)
    print("\n[2] IndexFlatL2  is_trained:", index_l2.is_trained, "（Flatは学習不要）")
    index_l2.add(xb)
    print("    ntotal(登録件数):", index_l2.ntotal)

    # search は (距離D, ID I) を返す。L2 は「小さいほど近い」ので D は昇順。
    D, I = index_l2.search(xq, k)  # noqa: E741 （FAISSの正準名: D=距離, I=ID）
    print("    返り値の形: D", D.shape, " I", I.shape)
    print("    クエリ0 の近傍ID:", I[0].tolist())
    print("    クエリ0 の L2距離:", np.round(D[0], 3).tolist(), "← 先頭は自分自身で 0.0")

    # === 3. 素の内積(IP)とコサインの違い =================================
    # IndexFlatIP は内積で並べる（大きいほど近い）。ただし正規化しないと
    # 「ベクトルの長さ」に引っ張られ、コサイン類似度にはならない。
    index_ip_raw = faiss.IndexFlatIP(d)
    index_ip_raw.add(xb)  # 未正規化のまま add → これは「素の内積」検索
    Dr, Ir = index_ip_raw.search(xq, k)
    print("\n[3] 未正規化の IndexFlatIP（素の内積。コサインではない）")
    print("    クエリ0 の近傍ID:", Ir[0].tolist())
    print("    クエリ0 の内積  :", np.round(Dr[0], 3).tolist(), "← 大きいほど近い（降順）")

    # === 4. 正規化してコサイン類似度にする ===============================
    # L2 正規化すると ‖x‖=1 になり、内積 = コサイン類似度（-1〜1）になる。
    # DB・クエリの「両方」を正規化するのが必須。片方だけだと崩れる。
    xb_n = l2_normalize(xb)
    xq_n = l2_normalize(xq)
    index_cos = faiss.IndexFlatIP(d)
    index_cos.add(xb_n)
    Dc, Ic = index_cos.search(xq_n, k)
    print("\n[4] 正規化 + IndexFlatIP = コサイン類似度検索")
    print("    クエリ0 の近傍ID:", Ic[0].tolist())
    print("    クエリ0 のコサイン:", np.round(Dc[0], 3).tolist(), "← 先頭は自分自身で ≈1.0")

    # faiss.normalize_L2 は in-place 版。下のように使えるが「元配列を破壊する」点に注意。
    # （本講座では学習用に非破壊の l2_normalize を使うが、実務では in-place が定番。）
    demo = as_faiss_array(xb[:3].copy())
    faiss.normalize_L2(demo)  # demo がその場で正規化される
    print("    faiss.normalize_L2 後のノルム:", np.round(np.linalg.norm(demo, axis=1), 3).tolist())

    # === 5. L2 と コサイン の「並び順」を比べる ===========================
    # 正規化済みベクトル同士なら L2距離 と コサイン には単調な関係があり、
    # 近傍の集合は一致する（ソート方向だけ逆: L2は昇順, IPは降順）。
    index_l2n = faiss.IndexFlatL2(d)
    index_l2n.add(xb_n)
    _, Il2n = index_l2n.search(xq_n, k)
    same = np.array_equal(np.sort(Il2n, axis=1), np.sort(Ic, axis=1))
    print("\n[5] 正規化後: L2 と IP(コサイン) の近傍集合は一致:", same)
    print("    L2  の近傍ID(クエリ0):", Il2n[0].tolist())
    print("    IP  の近傍ID(クエリ0):", Ic[0].tolist())

    # === 6. 図に保存（距離スコアの並び方を可視化） =======================
    _save_score_figure(out, Dr[0], Dc[0], D[0])

    print("\n完了。lectures/17_faiss_image_search/outputs/ を確認してください。")


def _save_score_figure(out: pathlib.Path, ip_raw, cos, l2) -> None:
    """近傍スコアの並び（IP素・コサイン・L2）を1枚の図にする。"""
    import matplotlib

    matplotlib.use("Agg")  # headless（画面なし）で savefig を確実に動かす
    import matplotlib.pyplot as plt

    ranks = np.arange(1, len(cos) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))
    for ax, vals, title, note in zip(
        axes,
        [ip_raw, cos, l2],
        ["IP (raw inner product)", "IP after L2-normalize (cosine)", "L2 distance"],
        ["bigger = nearer", "bigger = nearer (<=1)", "smaller = nearer"],
    ):
        ax.bar(ranks, vals, color="#3b7dd8")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(f"rank (k)  /  {note}")
        ax.set_xticks(ranks)
    fig.suptitle("Flat search scores for query #0 (self is rank 1)", fontsize=11)
    fig.tight_layout()
    path = out / "01_flat_scores.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print("    図を保存:", path.name)


if __name__ == "__main__":
    main()
