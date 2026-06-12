"""章末ミニプロジェクト: ミニ Cluster-CLIP（Split -> Build -> Search -> Stream）。

統合課題:
  これまで学んだ全要素を 1 本に束ねる、講座の総仕上げ。
    Split  : 合成動画をフレーム JPEG に分解（参考実装 split/processor.py 相当）
    Build  : 各フレームを dense CLIP -> 空間連結クラスタリング -> 代表ベクトル化し、
             FAISS IndexIDMap(IndexFlatIP) と SQLite メタDB を構築
             （build/models.py + indexer.py + db_writer.py 相当）
    Search : テキストクエリ -> CLIP text -> FAISS -> SQLite join -> ヒット領域を重畳可視化
             （search/engine.py + visualizer.py 相当）
    Stream : multiprocessing で capture(取得) / consumer(推論) / writer(記録) を分離し、
             キュー満杯時はフレームをドロップしてリアルタイム性を担保
             （stream/pipeline.py 相当）

出力:
  outputs/40_cluster_clip_dense_cluster/mini_project/ に
    frames/ ・ build/(local_index.faiss, local_metadata.db, vectors/, cluster_maps/)
    ・ search/ ・ mini_project_search.png

実行:
  uv run python lectures/40_cluster_clip_dense_cluster/mini_project.py
"""

from __future__ import annotations

import multiprocessing as mp
import pathlib
import queue
import sqlite3
import sys
import time

import matplotlib

matplotlib.use("Agg")
import cv2  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import cc_common as C  # noqa: E402

POISON = "STOP"
N_CLUSTERS = 6


# =====================================================================
# Split: 合成動画 -> フレーム JPEG
# =====================================================================
def stage_split(frames_dir: pathlib.Path, n: int = 6) -> list[dict]:
    frames_dir.mkdir(parents=True, exist_ok=True)
    seq = C.make_frame_sequence(n=n, seed=0)
    manifest = []
    for i, (img_bgr, regions) in enumerate(seq):
        frame_id = f"frame{i:04d}"
        path = frames_dir / f"{frame_id}.jpg"
        cv2.imwrite(str(path), img_bgr)  # OpenCV は BGR をそのまま保存
        manifest.append({"frame_id": frame_id, "path": str(path), "regions": regions})
    print(f"[split] {len(manifest)} フレームを {frames_dir} に書き出し")
    return manifest


# =====================================================================
# Build: dense -> cluster -> FAISS + SQLite カプセル
# =====================================================================
def stage_build(enc, manifest: list[dict], build_dir: pathlib.Path) -> None:
    import faiss

    vec_dir = build_dir / "vectors"
    cmap_dir = build_dir / "cluster_maps"
    vec_dir.mkdir(parents=True, exist_ok=True)
    cmap_dir.mkdir(parents=True, exist_ok=True)

    db_path = build_dir / "local_metadata.db"
    if db_path.exists():
        db_path.unlink()
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    # Frames: フレーム単位のメタ / VectorMapping: faiss_id <-> 領域 の対応表
    cur.execute(
        "CREATE TABLE Frames (frame_id TEXT PRIMARY KEY, image_path TEXT, cmap_path TEXT)"
    )
    cur.execute(
        "CREATE TABLE VectorMapping ("
        "faiss_id INTEGER PRIMARY KEY AUTOINCREMENT, frame_id TEXT, cluster_idx INTEGER, "
        "x1 INTEGER, y1 INTEGER, x2 INTEGER, y2 INTEGER)"
    )

    dim = enc.feat_dim
    index = faiss.IndexIDMap(faiss.IndexFlatIP(dim))

    for m in manifest:
        img_bgr = cv2.imread(m["path"])  # imread は失敗時 None を返す
        assert img_bgr is not None, f"画像が読めない: {m['path']}"
        x = torch.stack([enc.preprocess_bgr(img_bgr)])
        fmap = enc.dense(x)[0]
        reps, label_map = C.cluster_regions(fmap, N_CLUSTERS, connectivity=True)

        cmap_path = cmap_dir / f"{m['frame_id']}_cmap.npy"
        vec_path = vec_dir / f"{m['frame_id']}_vecs.npy"
        np.save(cmap_path, label_map)
        np.save(vec_path, reps)
        cur.execute(
            "INSERT INTO Frames VALUES (?,?,?)",
            (m["frame_id"], m["path"], str(cmap_path)),
        )

        # クラスタごとの代表 bbox（領域マスクの外接矩形）を計算して登録
        H, W = img_bgr.shape[:2]
        lab_big = C.upscale_nearest(label_map, (H, W))
        ids, vecs = [], []
        for cid in range(len(reps)):
            ys, xs = np.where(lab_big == cid)
            if len(xs) == 0:
                continue
            bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
            cur.execute(
                "INSERT INTO VectorMapping (frame_id, cluster_idx, x1, y1, x2, y2) VALUES (?,?,?,?,?,?)",
                (m["frame_id"], cid, *bbox),
            )
            ids.append(cur.lastrowid)  # SQLite の rowid を faiss_id に採用
            vecs.append(reps[cid])
        index.add_with_ids(C.as_faiss(np.stack(vecs)), np.array(ids, dtype=np.int64))

    conn.commit()
    conn.close()
    faiss.write_index(index, str(build_dir / "local_index.faiss"))
    print(f"[build] FAISS={index.ntotal} 本の領域ベクトル + SQLite メタDB を構築")


# =====================================================================
# Search: text -> FAISS -> SQLite -> 重畳可視化
# =====================================================================
def stage_search(enc, build_dir: pathlib.Path, query: str, out_png: pathlib.Path, top_k: int = 4):
    import faiss

    index = faiss.read_index(str(build_dir / "local_index.faiss"))
    conn = sqlite3.connect(build_dir / "local_metadata.db")
    cur = conn.cursor()

    qv = enc.encode_text([query])
    scores, ids = index.search(C.as_faiss(qv), top_k)

    results = []
    for rank, (fid, score) in enumerate(zip(ids[0], scores[0]), 1):
        if fid == -1:  # 近傍が足りない場合 FAISS は -1 を返す。必ずガードする。
            continue
        cur.execute(
            "SELECT v.frame_id, v.cluster_idx, v.x1, v.y1, v.x2, v.y2, f.image_path, f.cmap_path "
            "FROM VectorMapping v JOIN Frames f ON v.frame_id = f.frame_id WHERE v.faiss_id=?",
            (int(fid),),
        )
        row = cur.fetchone()
        assert row is not None, "faiss_id に対応するメタが SQLite に無い（整合性破壊）"
        results.append({"rank": rank, "score": float(score), "frame_id": row[0],
                        "cluster": row[1], "bbox": row[2:6], "image_path": row[6],
                        "cmap_path": row[7]})
    conn.close()

    print(f"\n[search] query={query!r} -> {len(results)} 件:")
    for r in results:
        print(f"  #{r['rank']} {r['frame_id']} cluster={r['cluster']} "
              f"bbox={tuple(r['bbox'])} score={r['score']:.3f}")

    # 可視化: ヒット領域のマスクを重畳
    fig, axes = plt.subplots(1, len(results), figsize=(3.4 * len(results), 3.6))
    if len(results) == 1:
        axes = [axes]
    for ax, r in zip(axes, results):
        img_rgb = cv2.imread(r["image_path"])[:, :, ::-1]
        label_map = np.load(r["cmap_path"])
        H, W = img_rgb.shape[:2]
        mask = C.upscale_nearest(label_map, (H, W)) == r["cluster"]
        ov = C.overlay_region(img_rgb, mask, color=(255, 40, 40), alpha=0.55)
        ax.imshow(ov)
        ax.set_title(f"#{r['rank']} {r['frame_id']}\nc{r['cluster']} ({r['score']:.2f})")
        ax.axis("off")
    fig.suptitle(f"Mini Cluster-CLIP search: '{query}'", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
    print(f"[search] {out_png}")
    return results


# =====================================================================
# Stream: capture / consumer / writer の 3 プロセス（キュー満杯はドロップ）
# =====================================================================
def run_capture(task_q, n_frames: int, dropped):
    """[CPU] フレームを生成し、満杯なら put_nowait で即ドロップ（リアルタイム保証）。"""
    seq = C.make_frame_sequence(n=n_frames, seed=7)
    for i, (img_bgr, _r) in enumerate(seq):
        try:
            task_q.put_nowait((i, img_bgr))
        except queue.Full:
            with dropped.get_lock():
                dropped.value += 1  # 追いつけないフレームは捨てる
        time.sleep(0.03)  # 約 30fps の到着を模擬
    task_q.put(POISON)


def run_consumer(task_q, result_q):
    """[CPU bound] dense CLIP 推論 + クラスタリング（取得とは別プロセスで実行）。"""
    enc = C.load_encoder(verbose=False)
    while True:
        item = task_q.get()
        if item == POISON:
            result_q.put(POISON)
            return
        idx, img_bgr = item
        x = torch.stack([enc.preprocess_bgr(img_bgr)])
        fmap = enc.dense(x)[0]
        reps, _lab = C.cluster_regions(fmap, N_CLUSTERS, connectivity=True)
        result_q.put({"idx": idx, "n_vectors": int(reps.shape[0])})


def run_writer(result_q, processed):
    """[CPU] 結果を受け取り、増分でメタを記録する（ここでは件数を数えるだけ）。"""
    while True:
        item = result_q.get()
        if item == POISON:
            return
        with processed.get_lock():
            processed.value += 1


def stage_stream(n_frames: int = 8, queue_size: int = 2) -> None:
    ctx = mp.get_context("spawn")  # torch を子で安全にロードするため spawn を使う
    task_q = ctx.Queue(maxsize=queue_size)
    result_q = ctx.Queue()
    dropped = ctx.Value("i", 0)
    processed = ctx.Value("i", 0)

    writer = ctx.Process(target=run_writer, args=(result_q, processed))
    consumer = ctx.Process(target=run_consumer, args=(task_q, result_q))
    capture = ctx.Process(target=run_capture, args=(task_q, n_frames, dropped))

    t0 = time.time()
    writer.start()
    consumer.start()
    capture.start()
    capture.join()
    consumer.join()
    writer.join()
    elapsed = time.time() - t0

    eff_fps = processed.value / elapsed if elapsed > 0 else 0.0
    print(f"\n[stream] 投入={n_frames}  処理={processed.value}  ドロップ={dropped.value}  "
          f"実効FPS={eff_fps:.2f} (queue_size={queue_size})")
    print("  -> 取得(capture)が推論(consumer)を追い越すと、満杯キューでフレームを捨てる。")


# =====================================================================
def main() -> None:
    out = C.output_dir() / "mini_project"
    out.mkdir(parents=True, exist_ok=True)
    frames_dir = out / "frames"
    build_dir = out / "build"

    print("=" * 64)
    print("ミニ Cluster-CLIP: Split -> Build -> Search -> Stream")
    print("=" * 64)

    enc = C.load_encoder()
    manifest = stage_split(frames_dir, n=6)
    stage_build(enc, manifest, build_dir)
    results = stage_search(enc, build_dir, "a yellow ball", out / "mini_project_search.png")
    assert all(r["frame_id"] for r in results), "検索結果のメタ解決に失敗"
    stage_stream(n_frames=8, queue_size=2)

    print("\n[done] 全ステージ完了。学んだ CLIP/クラスタリング/FAISS/SQLite/ストリームが 1 本に繋がった。")


if __name__ == "__main__":
    main()
