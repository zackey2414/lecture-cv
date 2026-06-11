"""04: ANN品質の定量評価 — Recall@k、QPS-recall曲線、retrieval mAP。

学ぶこと:
  - Recall@k の定義と自前計算（集合一致 np.intersect1d）。ground truth は Flat(厳密)で作る。
  - nprobe / efSearch をスイープして「QPS（速度）vs recall（精度）」の曲線を描く。
    用途に応じてどの設定を選ぶかを定量的に判断できるようにする。
  - ラベル付きデータでは retrieval mAP（同ラベルを正例とした平均適合率）も測る。
  - 結果は JSON と PNG に保存し、後から数値で再現・比較できるようにする。

実行:
  uv run python lectures/17_faiss_image_search/04_recall_qps_eval.py
"""

from __future__ import annotations

import json
import pathlib
import sys
import time

import faiss
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from search_helpers import (  # noqa: E402
    as_faiss_array,
    average_precision_at_k,
    l2_normalize,
    make_clustered_vectors,
    output_dir,
    recall_at_k,
)


def sweep_index(index, xq, gt_ids, k, param_name, param_values, setter):
    """1つの index について param をスイープし、(param, recall, QPS) のリストを返す。

    setter(index, value) で nprobe や efSearch を設定する関数を渡す（差分はここだけ）。
    """
    rows = []
    for v in param_values:
        setter(index, v)
        index.search(xq, k)  # ウォームアップ（初回呼び出しのオーバーヘッドを計測から除く）
        # QPS は1回計測だとブレるので、複数回まわした合計時間から算出して安定させる。
        repeats = 5
        t = time.perf_counter()
        for _ in range(repeats):
            _, ann_ids = index.search(xq, k)
        sec = (time.perf_counter() - t) / repeats
        rec = recall_at_k(gt_ids, ann_ids, k)
        qps = len(xq) / sec
        rows.append({param_name: v, "recall": rec, "qps": qps})
        print(f"    {param_name}={v:<4} recall@{k}={rec:.3f}  QPS={qps:,.0f}")
    return rows


def retrieval_map(index, xq, q_labels, db_labels, k):
    """ラベルを正解とした retrieval mAP@k を計算する。

    各クエリで上位(k+1)件を引き（自分自身が混ざるので+1して先頭を除く）、
    同ラベルを正例として AP@k を出し、全クエリで平均する。
    """
    # クエリ自身がDBに含まれるので、先頭1件(=自分)を捨てるため k+1 件取る。
    _, ann_ids = index.search(xq, k + 1)
    aps = []
    for qi in range(len(xq)):
        ranked = ann_ids[qi]
        ranked = ranked[ranked != -1]
        # 先頭(自分自身)を除いた候補のラベル列
        cand_labels = db_labels[ranked[1:]]
        aps.append(average_precision_at_k(cand_labels, q_labels[qi], k))
    return float(np.mean(aps))


def main() -> None:
    out = output_dir()
    print(f"faiss {faiss.__version__}")

    # === 1. ラベル付きデータ（同ラベル=関連あり） ========================
    d = 64
    n = 10000
    n_clusters = 50
    xb, db_labels = make_clustered_vectors(n=n, d=d, n_clusters=n_clusters, noise=1.0, seed=4)
    xb = l2_normalize(xb)
    rng = np.random.default_rng(5)
    qidx = rng.choice(n, 300, replace=False)
    xq = as_faiss_array(xb[qidx].copy())
    q_labels = db_labels[qidx]
    k = 10
    metric = faiss.METRIC_INNER_PRODUCT
    print(f"\n[1] DB={xb.shape} クエリ={len(xq)} クラスタ={n_clusters} k={k}")

    # === 2. ground truth を Flat(厳密)で作る =============================
    flat = faiss.IndexFlatIP(d)
    flat.add(xb)
    _, gt_ids = flat.search(xq, k)
    print("[2] ground truth = IndexFlatIP の厳密 top-k（ANN自身では作らない）")

    report: dict = {"k": k, "n_db": n, "dim": d}

    # === 3. IVF の nprobe スイープ ======================================
    nlist = 100
    quantizer = faiss.IndexFlatIP(d)
    ivf = faiss.IndexIVFFlat(quantizer, d, nlist, metric)
    ivf.train(xb)
    ivf.add(xb)
    print(f"\n[3] IVFFlat nlist={nlist} の nprobe スイープ")

    def set_nprobe(ix, v):
        ix.nprobe = int(v)

    report["ivf"] = sweep_index(ivf, xq, gt_ids, k, "nprobe", [1, 2, 5, 10, 20, 50], set_nprobe)

    # === 4. HNSW の efSearch スイープ ====================================
    hnsw = faiss.IndexHNSWFlat(d, 32, metric)
    hnsw.hnsw.efConstruction = 40
    hnsw.add(xb)
    print("\n[4] HNSW M=32 の efSearch スイープ")

    def set_ef(ix, v):
        ix.hnsw.efSearch = int(v)

    report["hnsw"] = sweep_index(hnsw, xq, gt_ids, k, "efSearch", [8, 16, 32, 64, 128], set_ef)

    # === 5. retrieval mAP（埋め込み品質をラベルで測る） ==================
    # Recall@k は「ANN が厳密検索にどれだけ近いか」。mAP は「同カテゴリを上位に集められるか」。
    # 観点が違うので両方見る。mAP は厳密検索(Flat)上で測り、埋め込み自体の良さを評価する。
    map_flat = retrieval_map(flat, xq, q_labels, db_labels, k)
    report["retrieval_map@k_flat"] = map_flat
    print(f"\n[5] retrieval mAP@{k}（Flat上, ラベル一致が正例）= {map_flat:.3f}")

    # === 6. 保存 ========================================================
    json_path = out / "04_eval_report.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[6] 数値を保存: {json_path.name}")

    _save_curve(out, report["ivf"], report["hnsw"])
    print("\n完了。outputs/17_faiss_image_search/ を確認してください。")


def _save_curve(out: pathlib.Path, ivf_rows, hnsw_rows) -> None:
    """QPS(横, 対数) vs Recall(縦) の曲線を保存。右上ほど良い（速くて正確）。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(
        [r["qps"] for r in ivf_rows],
        [r["recall"] for r in ivf_rows],
        "o-",
        label="IVFFlat (sweep nprobe)",
    )
    for r in ivf_rows:
        ax.annotate(
            f"np={r['nprobe']}",
            (r["qps"], r["recall"]),
            fontsize=7,
            xytext=(3, -8),
            textcoords="offset points",
        )
    ax.plot(
        [r["qps"] for r in hnsw_rows],
        [r["recall"] for r in hnsw_rows],
        "s-",
        label="HNSW (sweep efSearch)",
    )
    for r in hnsw_rows:
        ax.annotate(
            f"ef={r['efSearch']}",
            (r["qps"], r["recall"]),
            fontsize=7,
            xytext=(3, 6),
            textcoords="offset points",
        )
    ax.set_xscale("log")
    ax.set_xlabel("QPS (queries/sec, log scale)  — faster →")
    ax.set_ylabel("Recall@10  — more accurate ↑")
    ax.set_title("Speed-accuracy trade-off (top-right is best)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = out / "04_qps_recall_curve.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print("    図を保存:", path.name)


if __name__ == "__main__":
    main()
