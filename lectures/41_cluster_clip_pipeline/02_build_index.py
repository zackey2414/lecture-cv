"""02 Build — 各フレームを dense CLIP → 領域クラスタ → 代表ベクトル化し、FAISS+SQLite を作る。

参考実装対応:
  build/consumer.py（推論+クラスタリング+npy保存）
  + build/indexer.py（FAISS IndexFlatIP+IDMap 構築）
  + build/db_writer.py（SQLite に Frames/VectorMapping 書き込み）
  を 1 つの run_build に統合している（helper 参照）。

学ぶこと:
  - encode_image は CLS=画像1ベクトルだが、dense_clip_embeddings はパッチ特徴 [C,H,W] を返す。
  - 空間連結ありの凝集型クラスタリングで「意味の似た隣接パッチ」を領域にまとめ、平均→正規化で代表化。
  - 代表ベクトルを faiss(IndexFlatIP+IDMap) に登録し、faiss_id↔(frame,cluster) を SQLite で引けるようにする。

実行:
    uv run python lectures/41_cluster_clip_pipeline/02_build_index.py
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cluster_clip_helpers as H  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

N_FRAMES = 12
N_CLUSTERS = 6


def visualize_regions(frame_paths: list[Path], build_dir: Path, out_name: str) -> Path:
    """先頭 3 フレームについて「原画像 | 色分けクラスタマップ | 代表クラスタ重畳」を並べる。"""
    show = frame_paths[:3]
    fig, axes = plt.subplots(len(show), 3, figsize=(9, 3.0 * len(show)))
    axes = np.atleast_2d(axes)
    for r, fpath in enumerate(show):
        frame = cv2.imread(str(fpath))
        h, w = frame.shape[:2]
        frame_id = f"synthA_frame{r:06d}"
        cmap = np.load(build_dir / "cluster_maps" / f"{frame_id}_cmap.npy")

        axes[r, 0].imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        axes[r, 0].set_title(f"original {fpath.name[-13:]}", fontsize=8)

        axes[r, 1].imshow(cv2.cvtColor(H.colorize_cmap(cmap, h, w), cv2.COLOR_BGR2RGB))
        axes[r, 1].set_title(f"cluster map (k={int(cmap.max()) + 1})", fontsize=8)

        # 面積が最大のクラスタを 1 つ選んで重畳（領域の見え方を確認）
        big = int(np.bincount(cmap.ravel()).argmax())
        overlay = H.overlay_cluster_mask(frame, cmap, big)
        axes[r, 2].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        axes[r, 2].set_title(f"region overlay (cl{big})", fontsize=8)
        for c in range(3):
            axes[r, c].axis("off")
    fig.suptitle("Build: dense CLIP -> spatial clustering -> region representatives", fontsize=12)
    return H.save_fig(fig, out_name)


def main() -> int:
    H.set_seed(0)
    device = H.pick_device()
    base = H.outputs_dir("capstone")
    frames_dir = base / "frames"
    build_dir = base / "build"

    # 1) フレームを用意（01 を実行していなければここで合成）
    frames = H.run_split(frames_dir, N_FRAMES)
    print(f"[build] frames={len(frames)}")

    # 2) CLIP をロード（ViT-B-32, force_quick_gelu, 正方形リサイズ前処理）
    print("[build] CLIP (ViT-B-32) をロード中 ...")
    model, preprocess, _ = H.load_clip(device)

    # 3) Build: dense → cluster → 代表ベクトル → FAISS + SQLite
    print("[build] dense CLIP + 空間連結クラスタリング + 索引化 ...")
    summary = H.run_build(
        build_dir, frames, model, preprocess, n_clusters=N_CLUSTERS, device=device
    )
    print(
        f"[build] frames={summary['n_frames']} vectors={summary['n_vectors']} "
        f"dim={summary['dim']} (= {summary['n_frames']} frames x {N_CLUSTERS} clusters)"
    )

    # 4) 索引と DB の中身を確認
    import faiss

    index = faiss.read_index(str(build_dir / H.INDEX_NAME))
    conn = sqlite3.connect(build_dir / H.DB_NAME)
    n_frames_db = conn.execute("SELECT COUNT(*) FROM Frames").fetchone()[0]
    n_vecs_db = conn.execute("SELECT COUNT(*) FROM VectorMapping").fetchone()[0]
    sample = conn.execute(
        "SELECT faiss_id, frame_id, cluster_idx FROM VectorMapping ORDER BY faiss_id LIMIT 3"
    ).fetchall()
    conn.close()
    print(f"[build] FAISS ntotal={index.ntotal}  SQLite Frames={n_frames_db} VectorMapping={n_vecs_db}")
    print(f"[build] VectorMapping 先頭3件 (faiss_id, frame_id, cluster_idx): {sample}")

    # 5) 領域分割の可視化
    fig_path = visualize_regions(frames, build_dir, "02_build_regions.png")
    print(f"[plot] saved -> {fig_path}")

    # 6) 自己検証
    assert index.ntotal == n_vecs_db == summary["n_vectors"], "FAISS と DB の件数が不整合"
    assert summary["dim"] == H.EMBED_DIM, f"次元が想定外: {summary['dim']}"
    print("[done] Build 完了。次は 03_search_regions.py でテキスト/領域クエリ検索を行う。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
