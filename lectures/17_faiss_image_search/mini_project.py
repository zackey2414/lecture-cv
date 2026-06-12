"""章末ミニプロジェクト: 小さな画像検索エンジンを作って評価する（capstone）。

この回で学んだ要素を1本に統合する総合課題の「完成形」。

統合する学び（本章の全部入り）:
  1. 埋め込み(CLIP) → L2正規化 → IndexIDMap2 + add_with_ids（任意ID付与）
  2. メタデータ(ID→ラベル) を SQLite で別管理し、検索結果の ID から引く（-1 ガード付き）
  3. write_index / read_index で永続化し、まっさらな状態から読み戻して検索（再起動を模す）
  4. 画像→画像 / テキスト→画像（CLIP 共有空間）のマルチモーダル検索
  5. retrieval mAP（同ラベル=正例）で「埋め込み自体の良さ」を測る
  6. 規模が増えたときの ANN（IVFFlat / HNSW）を Recall@k と QPS で定量比較し、
     目標 Recall を満たす中で最速の設定を「推薦」する

設計メモ:
  - digit 始まりの 01〜04 は import 不可なので、必要処理は自己完結させるか、
    digit で始まらない共通ヘルパ search_helpers から借用する（本ファイルは後者）。
  - すべて CPU・合成データで数十秒以内に完走。CLIP を取得できない環境では
    色記述子へ自動フォールバックし、テキスト検索だけスキップして exit 0 で完走する。
  - 出力は outputs/17_faiss_image_search/ に保存（matplotlib=Agg, imshow は呼ばない）。

実行:
  uv run python lectures/17_faiss_image_search/mini_project.py
"""

from __future__ import annotations

import json
import pathlib
import sqlite3
import sys
import time

import faiss
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from search_helpers import (  # noqa: E402
    Embedder,
    as_faiss_array,
    average_precision_at_k,
    images_from_data_or_synthetic,
    l2_normalize,
    make_clustered_vectors,
    output_dir,
    recall_at_k,
)

# 画像に付与する ID の起点（連番 0..N-1 と区別できることを示すため 1000 始まり）。
ID_BASE = 1000


# =====================================================================
# Part A: CLIP 埋め込みで本物の画像検索エンジンを組み立て・永続化・検索
# =====================================================================


def build_engine(
    db_vecs: np.ndarray, labels: list[str], index_path: pathlib.Path, db_path: pathlib.Path
) -> tuple[faiss.Index, np.ndarray]:
    """正規化済みベクトルから IndexIDMap2 のエンジンを作り、メタDBと一緒に永続化する。

    返り値: (index, ids)。ids は ID_BASE 始まりの int64 配列。
    """
    d = db_vecs.shape[1]
    # IndexFlatIP（コサイン用）を IndexIDMap2 で包み、任意の int64 ID を割り当てる。
    index = faiss.IndexIDMap2(faiss.IndexFlatIP(d))
    ids = np.arange(ID_BASE, ID_BASE + len(db_vecs)).astype(np.int64)
    index.add_with_ids(as_faiss_array(db_vecs), ids)

    # メタデータ(ID→ラベル)は FAISS の外（SQLite）で管理する。
    if db_path.exists():
        db_path.unlink()
    con = sqlite3.connect(db_path)
    con.execute("CREATE TABLE items (id INTEGER PRIMARY KEY, label TEXT)")
    con.executemany(
        "INSERT INTO items (id, label) VALUES (?, ?)",
        [(int(i), lab) for i, lab in zip(ids, labels)],
    )
    con.commit()
    con.close()

    # .faiss（ベクトル+内部ID）と .db（メタデータ）は「セットで1つの検索DB」。
    faiss.write_index(index, str(index_path))
    return index, ids


def lookup_labels(db_path: pathlib.Path, ids) -> list[str]:
    """検索で得た ID 列 → ラベル列。-1（無効ID）は '(none)' に変換してガードする。"""
    con = sqlite3.connect(db_path)
    out: list[str] = []
    for i in ids:
        i = int(i)
        if i < 0:  # ← -1 ガード。そのまま DB 参照するとクラッシュ/誤参照する。
            out.append("(none)")
            continue
        row = con.execute("SELECT label FROM items WHERE id = ?", (i,)).fetchone()
        out.append(row[0] if row else "(missing)")
    con.close()
    return out


def retrieval_map_on_labels(
    index: faiss.Index, db_vecs: np.ndarray, labels: list[str], k: int
) -> float:
    """同ラベル=正例とした retrieval mAP@k（埋め込み品質の指標）。

    各ベクトルをクエリにして上位(k+1)件を引き、自分自身を除いた候補のラベル列で
    AP@k を計算し、全クエリで平均する。
    """
    _, I = index.search(as_faiss_array(db_vecs), k + 1)  # noqa: E741
    aps = []
    for qi in range(len(db_vecs)):
        ranked = I[qi]
        ranked = ranked[ranked != -1]
        # ID → 位置(pos) → ラベル。自分自身(pos==qi)は候補から除外する。
        cand = [labels[int(r) - ID_BASE] for r in ranked if int(r) - ID_BASE != qi]
        aps.append(average_precision_at_k(cand[:k], labels[qi], k))
    return float(np.mean(aps))


def run_image_search_engine(out: pathlib.Path) -> dict:
    """Part A 本体: 画像エンジンを組み立て・永続化・再読込・画像/テキスト検索・mAP。"""
    index_path = out / "mini_project_engine.faiss"
    db_path = out / "mini_project_meta.db"

    # --- 1. 画像を用意して埋め込み（CLIP、無ければ色記述子にフォールバック） ---
    images, labels, source = images_from_data_or_synthetic(repeat=4)
    embedder = Embedder()
    print(f"[A-1] 検索対象: {source} / {len(images)} 枚")
    print(f"      エンコーダ: {embedder.backend} (dim={embedder.dim}, text={embedder.supports_text})")
    feats = embedder.encode_images(images)  # (N, dim) 未正規化
    db_vecs = l2_normalize(feats)  # コサイン用に L2 正規化

    # --- 2. エンジン構築 + 永続化 ---
    index, ids = build_engine(db_vecs, labels, index_path, db_path)
    print(f"[A-2] エンジン構築: ntotal={index.ntotal}  ID={ids[0]}..{ids[-1]}")
    print(f"      永続化: {index_path.name} ({index_path.stat().st_size / 1024:.1f}KB) + {db_path.name}")

    # --- 3. まっさらな状態から読み戻す（=アプリ再起動を模す） ---
    index = faiss.read_index(str(index_path))
    assert index.ntotal == len(images), "read_index 後の件数が一致しない（永続化失敗）"
    print(f"[A-3] read_index 後 ntotal={index.ntotal} ← 一致＝永続化OK")

    # --- 4. 画像→画像 検索 ---
    query_idx = labels.index("赤い円") if "赤い円" in labels else 0
    q_img = l2_normalize(embedder.encode_images([images[query_idx]]))
    Di, Ii = index.search(as_faiss_array(q_img), 5)
    img_names = lookup_labels(db_path, Ii[0])
    print(f"[A-4] 画像クエリ=「{labels[query_idx]}」 → {img_names}")

    # --- 5. テキスト→画像 検索（CLIP のときのみ） ---
    text_results: dict[str, list[str]] = {}
    text_queries = ["a photo of a red circle", "a blue square", "a green triangle"]
    if embedder.supports_text:
        t_vecs = l2_normalize(embedder.encode_texts(text_queries))
        Dt, It = index.search(as_faiss_array(t_vecs), 3)
        for q, row in zip(text_queries, It):
            text_results[q] = lookup_labels(db_path, row)
            print(f"[A-5] '{q}' → {text_results[q]}")
    else:
        print("[A-5] テキスト検索はスキップ（色記述子フォールバックのため）。")

    # --- 6. retrieval mAP（同ラベル=正例） ---
    map_k = 5
    mapk = retrieval_map_on_labels(index, db_vecs, labels, map_k)
    print(f"[A-6] retrieval mAP@{map_k} = {mapk:.3f}（同ラベルを上位に集められるか）")

    _save_search_figure(out, images, labels, query_idx, Ii[0], img_names, text_results)

    return {
        "source": source,
        "backend": embedder.backend,
        "n_images": len(images),
        "dim": int(db_vecs.shape[1]),
        "supports_text": embedder.supports_text,
        "image_query": {"label": labels[query_idx], "results": img_names},
        "text_query": text_results,
        "retrieval_map@5": mapk,
    }


# =====================================================================
# Part B: 規模が増えたときの ANN を Recall@k / QPS で評価し設定を推薦
# =====================================================================


def _search_timed(index, xq, k, repeats: int = 3) -> tuple[np.ndarray, float]:
    """検索して (ID行列, 1回あたり経過秒) を返す。ウォームアップ+平均でブレを抑える。"""
    index.search(xq, k)  # ウォームアップ（初回オーバーヘッドを計測から除く）
    t = time.perf_counter()
    for _ in range(repeats):
        _, ids = index.search(xq, k)
    return ids, (time.perf_counter() - t) / repeats


def evaluate_ann_scaling(out: pathlib.Path, target_recall: float = 0.95) -> dict:
    """大きめの合成埋め込みで IVF/HNSW をスイープし、Recall@k と QPS を測って推薦する。"""
    d, n, k = 64, 20000, 10
    xb, _ = make_clustered_vectors(n=n, d=d, n_clusters=100, noise=1.0, seed=7)
    xb = l2_normalize(xb)
    rng = np.random.default_rng(8)
    qidx = rng.choice(n, 300, replace=False)
    xq = as_faiss_array(xb[qidx].copy())
    metric = faiss.METRIC_INNER_PRODUCT
    print(f"\n[B-1] 規模評価: DB={xb.shape} クエリ={len(xq)} k={k}")

    # ground truth は必ず厳密 Flat で作る（ANN 自身で作らない）。
    flat = faiss.IndexFlatIP(d)
    flat.add(xb)
    gt_ids, flat_sec = _search_timed(flat, xq, k)
    flat_qps = len(xq) / flat_sec
    print(f"[B-2] Flat(厳密) を ground truth に  QPS={flat_qps:,.0f}")

    candidates: list[dict] = []

    # IVFFlat: nprobe スイープ。
    ivf = faiss.IndexIVFFlat(faiss.IndexFlatIP(d), d, 200, metric)
    ivf.train(xb)
    ivf.add(xb)
    ivf_rows = []
    for nprobe in (1, 2, 5, 10, 20, 50):
        ivf.nprobe = nprobe
        ids, sec = _search_timed(ivf, xq, k)
        rec = recall_at_k(gt_ids, ids, k)
        qps = len(xq) / sec
        row = {"method": "IVFFlat", "param": f"nprobe={nprobe}", "recall": rec, "qps": qps}
        ivf_rows.append({"nprobe": nprobe, "recall": rec, "qps": qps})
        candidates.append(row)
        print(f"      IVF nprobe={nprobe:2d}  recall@{k}={rec:.3f}  QPS={qps:,.0f}")

    # HNSW: efSearch スイープ。
    hnsw = faiss.IndexHNSWFlat(d, 32, metric)
    hnsw.hnsw.efConstruction = 40
    hnsw.add(xb)
    hnsw_rows = []
    for ef in (8, 16, 32, 64, 128):
        hnsw.hnsw.efSearch = ef
        ids, sec = _search_timed(hnsw, xq, k)
        rec = recall_at_k(gt_ids, ids, k)
        qps = len(xq) / sec
        row = {"method": "HNSW", "param": f"efSearch={ef}", "recall": rec, "qps": qps}
        hnsw_rows.append({"efSearch": ef, "recall": rec, "qps": qps})
        candidates.append(row)
        print(f"      HNSW ef={ef:3d}    recall@{k}={rec:.3f}  QPS={qps:,.0f}")

    # 推薦: 目標 Recall を満たす中で最速(QPS最大)。満たすものが無ければ Recall 最大。
    feasible = [c for c in candidates if c["recall"] >= target_recall]
    if feasible:
        best = max(feasible, key=lambda c: c["qps"])
        reason = f"recall>={target_recall} を満たす中で最速"
    else:
        best = max(candidates, key=lambda c: c["recall"])
        reason = f"recall>={target_recall} を満たす設定なし→Recall最大を採用"
    print(f"[B-3] 推薦設定: {best['method']} {best['param']} "
          f"(recall={best['recall']:.3f}, QPS={best['qps']:,.0f}) — {reason}")

    _save_curve_figure(out, ivf_rows, hnsw_rows, best)

    return {
        "k": k,
        "n_db": n,
        "dim": d,
        "target_recall": target_recall,
        "flat_qps": flat_qps,
        "ivf": ivf_rows,
        "hnsw": hnsw_rows,
        "recommended": {**best, "reason": reason},
    }


# =====================================================================
# 図の保存（タイトルは ASCII、日本語ラベルはローマ字化）
# =====================================================================

_ROMAJI = {
    "赤": "red", "緑": "green", "青": "blue", "黄": "yellow",
    "円": "circle", "四角": "square", "三角": "triangle", "い": " ",
}


def _ascii_label(text: str) -> str:
    for jp, en in _ROMAJI.items():
        text = text.replace(jp, en)
    return text.encode("ascii", "ignore").decode().strip() or text


def _save_search_figure(out, images, labels, query_idx, result_ids, result_names, text_results) -> None:
    """画像クエリの結果サムネ（+テキストクエリ結果のラベル）を1枚に。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(result_ids)
    fig, axes = plt.subplots(1, n + 1, figsize=(2.0 * (n + 1), 2.6))
    axes[0].imshow(images[query_idx])
    axes[0].set_title(f"query\n{_ascii_label(labels[query_idx])}", fontsize=9)
    axes[0].axis("off")
    for ax, rid, name in zip(axes[1:], result_ids, result_names):
        local = int(rid) - ID_BASE
        if 0 <= local < len(images):
            ax.imshow(images[local])
        ax.set_title(_ascii_label(name), fontsize=9)
        ax.axis("off")
    # テキスト検索の結果はタイトルに添える（CLIP のときのみ中身がある）。
    subtitle = "Image search engine (left: query)"
    if text_results:
        first_q = next(iter(text_results))
        subtitle += f"\ntext '{_ascii_label(first_q)}' -> {_ascii_label(text_results[first_q][0])}"
    fig.suptitle(subtitle, fontsize=10)
    fig.tight_layout()
    path = out / "mini_project_search.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print("      図を保存:", path.name)


def _save_curve_figure(out, ivf_rows, hnsw_rows, best) -> None:
    """QPS(横, 対数) vs Recall(縦) の曲線。推薦設定を星印で強調。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot([r["qps"] for r in ivf_rows], [r["recall"] for r in ivf_rows],
            "o-", label="IVFFlat (sweep nprobe)")
    ax.plot([r["qps"] for r in hnsw_rows], [r["recall"] for r in hnsw_rows],
            "s-", label="HNSW (sweep efSearch)")
    ax.scatter([best["qps"]], [best["recall"]], marker="*", s=320,
               color="gold", edgecolor="black", zorder=5,
               label=f"recommended ({best['method']} {best['param']})")
    ax.set_xscale("log")
    ax.set_xlabel("QPS (queries/sec, log scale) -- faster ->")
    ax.set_ylabel("Recall@10 -- more accurate ^")
    ax.set_title("Mini project: speed-accuracy trade-off (top-right is best)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = out / "mini_project_curve.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print("      図を保存:", path.name)


def main() -> None:
    out = output_dir()
    print(f"=== ミニプロジェクト: 画像検索エンジン (faiss {faiss.__version__}) ===")

    engine_report = run_image_search_engine(out)
    ann_report = evaluate_ann_scaling(out)

    report = {"faiss": faiss.__version__, "engine": engine_report, "ann_scaling": ann_report}
    json_path = out / "mini_project_report.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[done] レポートを保存: {json_path.name}")
    print("完了。outputs/17_faiss_image_search/ を確認してください。")


if __name__ == "__main__":
    main()
