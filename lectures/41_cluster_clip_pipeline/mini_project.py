"""章末ミニプロジェクト（総仕上げ）: Cluster-CLIP を Split→Build→Search→Stream で一気通貫。

統合課題:
  合成動画を 1) フレーム分割し（Split）、2) 各フレームを dense CLIP + 空間連結クラスタリングで
  領域代表ベクトル化して FAISS+SQLite を構築（Build）、3) テキスト/領域クエリで領域検索（Search）、
  4) 同じ部品で multiprocessing のリアルタイム・ストリーム索引も作る（Stream）。
  学んできた CLIP / クラスタリング / FAISS / SQLite / ストリームを 1 本に束ねる。

出力:
  - コンソールに各ステージのサマリ
  - lectures/41_cluster_clip_pipeline/outputs/mini_project_summary.png（検索結果ギャラリー）

実行:
    uv run python lectures/41_cluster_clip_pipeline/mini_project.py
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

N_FRAMES = 10
N_CLUSTERS = 6
TOP_K = 4
STREAM_FRAMES = 18


def region_mean_bgr(res: dict) -> np.ndarray:
    """検索ヒット領域の元画像での平均色（BGR）を返す（赤判定などに使う）。"""
    fr = cv2.imread(res["image_path"])
    cm = np.load(res["cmap_path"])
    cmu = cv2.resize(cm.astype(np.uint8), (fr.shape[1], fr.shape[0]), interpolation=cv2.INTER_NEAREST)
    return fr[cmu == res["cluster_idx"]].reshape(-1, 3).mean(axis=0)


def summary_figure(text_q, text_res, region_res, out_name) -> Path:
    """テキスト検索（上段）と領域検索（下段）の結果ギャラリーを 1 枚にまとめる。"""
    rows = [("text: " + text_q, text_res), ("image: frame0 red region", region_res)]
    ncol = TOP_K
    fig, axes = plt.subplots(2, ncol, figsize=(3.0 * ncol, 6.2))
    for r, (title, results) in enumerate(rows):
        for c in range(ncol):
            ax = axes[r, c]
            ax.axis("off")
            if c >= len(results):
                continue
            res = results[c]
            frame = cv2.imread(res["image_path"])
            cmap = np.load(res["cmap_path"])
            masked = H.overlay_cluster_mask(frame, cmap, res["cluster_idx"])
            ax.imshow(cv2.cvtColor(masked, cv2.COLOR_BGR2RGB))
            ax.set_title(f"{res['frame_id'][-9:]} cl{res['cluster_idx']}\nscore {res['score']:.3f}", fontsize=8)
        axes[r, 0].set_ylabel(title)
    fig.suptitle("Cluster-CLIP capstone: text vs region search", fontsize=13)
    return H.save_fig(fig, out_name)


def main() -> int:
    H.set_seed(0)
    device = H.pick_device()
    base = H.outputs_dir("capstone_mini")

    # ============ 1) SPLIT ============
    print("[1/4] Split: 合成動画 -> 連番フレーム")
    frames = H.run_split(base / "frames", N_FRAMES, video_name="synthA")
    print(f"      frames={len(frames)}")

    # ============ 2) BUILD ============
    print("[2/4] Build: dense CLIP + 空間連結クラスタリング + FAISS/SQLite")
    model, preprocess, tokenizer = H.load_clip(device)
    build_dir = base / "build"
    summary = H.run_build(build_dir, frames, model, preprocess, n_clusters=N_CLUSTERS, device=device)
    conn = sqlite3.connect(build_dir / H.DB_NAME)
    n_frames_db = conn.execute("SELECT COUNT(*) FROM Frames").fetchone()[0]
    conn.close()
    print(f"      vectors={summary['n_vectors']} (= {n_frames_db} frames x {N_CLUSTERS} clusters), dim={summary['dim']}")

    # ============ 3) SEARCH ============
    print("[3/4] Search: テキストクエリと領域クエリ")
    text_q = "a green region"
    text_res = H.search_index(build_dir, H.encode_text(model, tokenizer, [text_q], device), top_k=TOP_K)
    print(f"      text  '{text_q}' -> top: {[(r['frame_id'][-9:], r['cluster_idx'], round(r['score'], 3)) for r in text_res]}")

    f0 = cv2.imread(str(frames[0]))
    cmap0 = np.load(build_dir / "cluster_maps" / "synthA_frame000000_cmap.npy")
    red_cluster = H.pick_region_by_color(f0, cmap0, H.OBJECT_COLORS_BGR["red"])
    region_q = H.region_vector(build_dir, "synthA_frame000000", red_cluster)
    region_res = H.search_index(build_dir, region_q, top_k=TOP_K)
    print(f"      region frame0-red -> top: {[(r['frame_id'][-9:], r['cluster_idx'], round(r['score'], 3)) for r in region_res]}")

    fig_path = summary_figure(text_q, text_res, region_res, "mini_project_summary.png")
    print(f"      [plot] saved -> {fig_path}")

    # ============ 4) STREAM ============
    print("[4/4] Stream: multiprocessing 取得/推論分離 + 適応サンプリング")
    st = H.run_stream_pipeline(
        base / "stream", n_frames=STREAM_FRAMES, sampler="histogram_delta", threshold=0.1,
        queue_size=8, fps_cap=30, batch_size=2, video_name="live",
    )
    print(f"      total={st['total']} sampled_out={st['sampled_out']} processed={st['processed']} "
          f"dropped={st['dropped']} effective_fps={st['effective_fps']:.2f}")
    stream_hits = H.search_index(base / "stream", H.encode_text(model, tokenizer, ["a yellow region"], device), top_k=2)
    print(f"      streamed index も検索可能: 'a yellow region' -> {[(r['frame_id'][-9:], round(r['score'], 3)) for r in stream_hits]}")

    # ============ 自己検証 ============
    print("\n[verify] 統合パイプラインの健全性チェック ...")
    assert summary["n_vectors"] == n_frames_db * N_CLUSTERS, "ベクトル数 = フレーム x クラスタ になっていない"
    assert region_res[0]["score"] > 0.99, "領域クエリで自分自身が最上位でない"
    n_red = sum(1 for r in region_res[:3] if region_mean_bgr(r)[2] > region_mean_bgr(r)[0])
    assert n_red >= 2, "領域検索の上位が赤領域に偏らない"
    assert st["processed"] >= 1 and len(stream_hits) >= 1, "ストリーム索引が検索可能でない"
    assert st["sampled_out"] >= 1, "適応サンプラーが間引いていない"
    print("[done] Split→Build→Search→Stream を一気通貫で再構築できた。これが Cluster-CLIP の全体像。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
