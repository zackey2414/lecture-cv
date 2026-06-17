"""03: 近似最近傍(ANN) — IVFFlat / HNSW / IVFPQ・OPQ(index_factory)。

学ぶこと:
  - Flat(総当たり)は厳密だが O(N)。規模が増えると ANN（近似）で速度を稼ぐ。
  - IVFFlat: 空間を nlist 個のセルに分け、クエリ近傍の nprobe セルだけ探す転置インデックス。
    add の前に train（代表データでセル中心を学習）が必須。nprobe で精度↔速度を調整。
  - HNSW: 近傍グラフを辿る方式。train 不要。efSearch が速度↔精度の主ダイヤル。
  - index_factory: 文字列で index を組み立てる。"IVF100,PQ8" や "OPQ8_64,IVF100,PQ8"。
    PQ(Product Quantization) はベクトルを量子化してメモリを激減させる（精度と引き換え）。
  - 各 index のメモリ（保存ファイルサイズ）を比較し、圧縮の効果を体感する。

実行:
  uv run python lectures/17_faiss_image_search/03_ivf_hnsw_pq.py
"""

from __future__ import annotations

import pathlib
import sys
import time

import faiss
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from search_helpers import (  # noqa: E402
    as_faiss_array,
    l2_normalize,
    make_clustered_vectors,
    output_dir,
    recall_at_k,
)


def _search_timed(index, xq, k, repeats: int = 5):
    """検索して (ID行列, 1回あたり経過秒) を返す。QPS=クエリ数/経過秒。

    初回呼び出しのオーバーヘッドを除くためウォームアップし、複数回の平均で
    計測ブレを抑える（小規模データだと1回計測は揺れやすい）。
    """
    index.search(xq, k)  # ウォームアップ
    t = time.perf_counter()
    for _ in range(repeats):
        _, ids = index.search(xq, k)
    return ids, (time.perf_counter() - t) / repeats


def main() -> None:
    out = output_dir()
    print(f"faiss {faiss.__version__}")

    # === 1. データ（コサイン検索なので正規化して使う） ===================
    d = 64
    n = 10000
    xb, _ = make_clustered_vectors(n=n, d=d, n_clusters=50, noise=1.0, seed=2)
    xb = l2_normalize(xb)  # IP=コサインにするため正規化
    rng = np.random.default_rng(3)
    qidx = rng.choice(n, 200, replace=False)
    xq = as_faiss_array(xb[qidx].copy())
    k = 10
    metric = faiss.METRIC_INNER_PRODUCT
    print(f"\n[1] DB={xb.shape} クエリ={xq.shape} k={k}（metric=内積/コサイン）")

    # === 2. ground truth は必ず Flat(厳密)で作る =========================
    # ANN の良し悪し(Recall)は「厳密検索の答え」と比べて測る。
    # ground truth を ANN 自身で作るのは典型的な誤り（高く見えるが意味がない）。
    flat = faiss.IndexFlatIP(d)
    flat.add(xb)
    gt_ids, flat_sec = _search_timed(flat, xq, k)
    print(f"\n[2] IndexFlatIP(厳密) を ground truth に  QPS={len(xq) / flat_sec:,.0f}")

    results = []  # (名前, recall@k, QPS, バイト数)

    # === 3. IVFFlat（train 必須・nprobe スイープ） ========================
    nlist = 100  # セル数の目安は 4√N〜16√N（N=1万なら 100〜1600 くらい）
    quantizer = faiss.IndexFlatIP(d)  # セル中心を探すための内部 index
    ivf = faiss.IndexIVFFlat(quantizer, d, nlist, metric)
    print(f"\n[3] IVFFlat nlist={nlist}  is_trained(初期)={ivf.is_trained}")
    ivf.train(xb)  # ← これを忘れて add すると例外。代表データでセル中心を学習。
    ivf.add(xb)
    print(f"    train 後 is_trained={ivf.is_trained}  ntotal={ivf.ntotal}")
    for nprobe in (1, 5, 20):
        ivf.nprobe = nprobe  # 探索セル数。1 のままだと recall が大きく落ちる（既定の罠）
        ids, sec = _search_timed(ivf, xq, k)
        rec = recall_at_k(gt_ids, ids, k)
        qps = len(xq) / sec
        print(f"    nprobe={nprobe:2d}  recall@{k}={rec:.3f}  QPS={qps:,.0f}")
        results.append((f"IVFFlat np={nprobe}", rec, qps, None))

    # === 4. HNSW（train 不要・efSearch スイープ） ========================
    M = 32  # グラフの次数（大きいほど精度↑・メモリ↑）
    hnsw = faiss.IndexHNSWFlat(d, M, metric)
    hnsw.hnsw.efConstruction = 40  # 構築品質（大きいほど高品質・構築遅い）
    hnsw.add(xb)  # train 不要。add だけでグラフが構築される。
    print(f"\n[4] HNSW M={M} efConstruction=40（train不要）")
    for ef in (16, 64):
        hnsw.hnsw.efSearch = ef  # 検索幅。速度↔精度の主ダイヤル。
        ids, sec = _search_timed(hnsw, xq, k)
        rec = recall_at_k(gt_ids, ids, k)
        qps = len(xq) / sec
        print(f"    efSearch={ef:3d}  recall@{k}={rec:.3f}  QPS={qps:,.0f}")
        results.append((f"HNSW ef={ef}", rec, qps, None))

    # === 5. index_factory と PQ 圧縮 =====================================
    # 文字列レシピで index を組む。PQ8 = 64次元を8個の部分ベクトルに割って各8bitで量子化。
    # → 1ベクトル 8 バイト（float32 生の 64*4=256 バイトの 1/32）。メモリ激減・精度は低下。
    print("\n[5] index_factory（PQ 圧縮でメモリ削減）")
    factory_specs = ["IVF100,Flat", "IVF100,PQ8", "OPQ8_64,IVF100,PQ8", "HNSW32"]
    for spec in factory_specs:
        idx = faiss.index_factory(d, spec, metric)
        idx.train(xb)  # PQ/IVF 系は train 必須（HNSW32 は train が no-op）
        idx.add(xb)
        if hasattr(idx, "nprobe"):
            idx.nprobe = 10
        ids, sec = _search_timed(idx, xq, k)
        rec = recall_at_k(gt_ids, ids, k)
        qps = len(xq) / sec
        # 保存してファイルサイズ（≒メモリ量）を測る
        p = out / f"03_factory_{spec.replace(',', '_').replace(' ', '')}.faiss"
        faiss.write_index(idx, str(p))
        nbytes = p.stat().st_size
        print(f"    {spec:22s} recall@{k}={rec:.3f}  QPS={qps:,.0f}  size={nbytes / 1024:6.1f}KB")
        results.append((spec, rec, qps, nbytes))

    # Flat の生サイズも比較用に
    pflat = out / "03_flat.faiss"
    faiss.write_index(flat, str(pflat))
    print(
        f"    {'Flat(無圧縮)':22s} {'':<22}size={pflat.stat().st_size / 1024:6.1f}KB ← PQ と比べる"
    )

    _save_tradeoff_figure(out, results)
    print("\n完了。lectures/17_faiss_image_search/outputs/ を確認してください。")


def _save_tradeoff_figure(out: pathlib.Path, results) -> None:
    """recall と QPS（と PQ のメモリ）をまとめた散布図を保存。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    for name, rec, qps, _ in results:
        ax.scatter(rec, qps, s=60)
        ax.annotate(name, (rec, qps), fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("Recall@10 (vs exact Flat)")
    ax.set_ylabel("QPS (queries/sec, higher=faster)")
    ax.set_title("ANN trade-off: accuracy vs speed")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = out / "03_ann_tradeoff.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print("    図を保存:", path.name)


if __name__ == "__main__":
    main()
