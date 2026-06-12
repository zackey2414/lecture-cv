"""03 Search — テキスト/領域クエリで「フレームのどの領域が合致するか」を検索・可視化する。

参考実装対応:
  search/engine.py（CLIP text encode → FAISS 検索 → SQLite でメタ引き）
  + search/visualizer.py（該当クラスタ領域を重畳した可視化）。

学ぶこと:
  - テキストクエリは CLIP text encoder → L2 正規化 → IndexFlatIP 検索（=コサイン類似）。
  - 領域クエリ（ある領域の代表ベクトル）で「他フレームの似た領域」を高精度に引ける。
  - 検索結果 ID から SQLite で frame と cluster を引き、cmap を重畳して結果を見せる。
  - 画像全体 1 ベクトル（global）の検索と比べ、領域検索は「どこが効いたか」を局在化できる。

注意（合成データの限界）:
  本講座のフレームは平坦な色領域の合成画像なので、テキスト→領域の意味整合はノイジー。
  一方「領域→領域」検索は埋め込みが素直に効くため安定して当たる。実写動画では両方とも良くなる。

実行:
    uv run python lectures/41_cluster_clip_pipeline/03_search_regions.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cluster_clip_helpers as H  # noqa: E402

N_FRAMES = 12
N_CLUSTERS = 6
TOP_K = 5


def ensure_build(base: Path, device):
    """build カプセルが無ければ Split→Build を実行して用意する（03 単体実行でも動くように）。"""
    frames_dir = base / "frames"
    build_dir = base / "build"
    model, preprocess, tokenizer = H.load_clip(device)
    if not (build_dir / H.INDEX_NAME).exists():
        print("[search] build カプセルが無いので作成します ...")
        frames = H.run_split(frames_dir, N_FRAMES)
        H.run_build(build_dir, frames, model, preprocess, n_clusters=N_CLUSTERS, device=device)
    return build_dir, frames_dir, model, preprocess, tokenizer


def global_text_search(model, preprocess, tokenizer, frames_dir, query, device):
    """ベースライン: 画像全体の CLIP 埋め込みに対するテキスト検索（フレーム単位、局在なし）。"""
    from PIL import Image

    frames = H.read_frame_paths(frames_dir)
    pil = [Image.open(p) for p in frames]
    gvecs = H.encode_image_global(model, preprocess, pil, device)  # [n, C]
    qv = H.encode_text(model, tokenizer, [query], device)[0]
    scores = gvecs @ qv  # コサイン類似（どちらも正規化済み）
    order = np.argsort(-scores)[:3]
    return [(frames[i].name[-13:], float(scores[i])) for i in order]


def main() -> int:
    H.set_seed(0)
    device = H.pick_device()
    base = H.outputs_dir("capstone")
    build_dir, frames_dir, model, preprocess, tokenizer = ensure_build(base, device)

    # ---------------------------------------------------------------
    # (A) テキスト → 領域 検索（Cluster-CLIP の看板機能）
    # ---------------------------------------------------------------
    text_query = "a red region"
    qv = H.encode_text(model, tokenizer, [text_query], device)
    text_results = H.search_index(build_dir, qv, top_k=TOP_K)
    print(f"[search][text] query='{text_query}'")
    for r in text_results:
        print(f"   {r['frame_id'][-9:]} cluster={r['cluster_idx']} score={r['score']:.3f}")
    gallery_t = H.save_search_gallery(text_query, text_results, "03_search_text.png")
    print(f"[plot] saved -> {gallery_t}")

    # ---------------------------------------------------------------
    # (B) 領域 → 領域 検索（画像クエリ）: フレーム0の「赤い領域」で他フレームの赤領域を探す
    # ---------------------------------------------------------------
    frame0 = H.read_frame_paths(frames_dir)[0]
    f0_bgr = cv2.imread(str(frame0))
    cmap0 = np.load(build_dir / "cluster_maps" / "synthA_frame000000_cmap.npy")
    red_cluster = H.pick_region_by_color(f0_bgr, cmap0, H.OBJECT_COLORS_BGR["red"])
    region_q = H.region_vector(build_dir, "synthA_frame000000", red_cluster)
    region_results = H.search_index(build_dir, region_q, top_k=TOP_K)
    print(f"\n[search][region] query=frame0 の赤領域 (cluster {red_cluster})")
    for r in region_results:
        print(f"   {r['frame_id'][-9:]} cluster={r['cluster_idx']} score={r['score']:.3f}")
    gallery_r = H.save_search_gallery("frame0 red region (image query)", region_results, "03_search_region.png")
    print(f"[plot] saved -> {gallery_r}")

    # ---------------------------------------------------------------
    # (C) ベースライン: 画像全体ベクトルでのテキスト検索（局在しない）
    # ---------------------------------------------------------------
    global_hits = global_text_search(model, preprocess, tokenizer, frames_dir, text_query, device)
    print(f"\n[search][global] query='{text_query}' -> フレーム単位 top3: {global_hits}")
    print("   ※ global は『どのフレームか』までで、『どの領域か』は出せない（領域検索との違い）。")

    # ---------------------------------------------------------------
    # 自己検証: 領域→領域 検索は、自分自身（score≈1）を含み、上位が赤領域に偏る
    # ---------------------------------------------------------------
    assert region_results[0]["score"] > 0.99, "領域クエリで自分自身が最上位に来ていない"

    def region_mean_bgr(res):
        fr = cv2.imread(res["image_path"])
        cm = np.load(res["cmap_path"])
        cmu = cv2.resize(cm.astype(np.uint8), (fr.shape[1], fr.shape[0]), interpolation=cv2.INTER_NEAREST)
        return fr[cmu == res["cluster_idx"]].reshape(-1, 3).mean(axis=0)

    top_is_red = sum(1 for r in region_results[:3] if region_mean_bgr(r)[2] > region_mean_bgr(r)[0])
    assert top_is_red >= 2, "領域クエリ上位が赤領域に偏っていない"
    print("[done] Search 完了。次は 04_stream_pipeline.py でリアルタイム・ストリーム処理を見る。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
