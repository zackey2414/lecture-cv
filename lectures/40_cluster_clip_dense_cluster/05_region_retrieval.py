"""05: 領域検索 — 代表ベクトルを FAISS に入れてクエリでヒット領域を引く。

学ぶこと:
  - 複数フレームの「領域代表ベクトル」を FAISS IndexIDMap(IndexFlatIP) に登録し、
    検索結果の faiss_id から「どのフレームのどのクラスタか」をメタ表で引く。
  - 画像領域クエリ（ある領域の代表ベクトルでそのまま検索）は機構として確実に動く:
    自分自身が rank-1 に来る。これが領域検索の最小サニティ。
  - テキストクエリ（CLIP text）は実写では領域をうまく当てるが、本講座の合成フラット
    画像では CLIP の意味的識別が弱く順位はあてにならない（重要な落とし穴）。
    ここでは end-to-end の機構と可視化を学ぶことを主目的にする。

実行:
  uv run python lectures/40_cluster_clip_dense_cluster/05_region_retrieval.py
"""

from __future__ import annotations

import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import cc_common as C  # noqa: E402


def build_region_db(enc, frames, n_clusters=6):
    """各フレームを dense->cluster し、代表ベクトルと領域メタを集める（Build 相当）。"""
    reps_all, meta = [], []
    label_maps = []
    for fi, (img_bgr, _regions) in enumerate(frames):
        x = torch.stack([enc.preprocess_bgr(img_bgr)])
        fmap = enc.dense(x)[0]
        reps, lab = C.cluster_regions(fmap, n_clusters, connectivity=True)
        label_maps.append(lab)
        for cid in range(len(reps)):
            reps_all.append(reps[cid])
            meta.append({"frame": fi, "cluster": cid})
    return np.stack(reps_all).astype("float32"), meta, label_maps


def main() -> None:
    out = C.output_dir()
    enc = C.load_encoder()

    frames = C.make_frame_sequence(n=4, seed=0)
    reps_all, meta, label_maps = build_region_db(enc, frames)
    faiss_ids = np.arange(len(reps_all), dtype=np.int64)
    index = C.build_ip_idmap(reps_all, faiss_ids)
    print(f"[build] {len(frames)} フレーム -> 領域ベクトル {reps_all.shape} を FAISS に登録")

    # === 1. 画像領域クエリ（機構サニティ: 自分が rank-1） ================
    q_id = 0  # フレーム0の最初の領域を使う
    _, Iself = index.search(C.as_faiss(reps_all[q_id : q_id + 1]), 5)
    print(f"\n[image-query] 領域#{q_id} で検索 -> top5 faiss_id={Iself[0].tolist()}")
    assert Iself[0][0] == q_id, "自分自身が rank-1 に来ない（FAISS 配線ミス）"
    print(f"  rank-1 が自分自身 (id={q_id}) -> 領域検索の機構 OK")

    # === 2. テキストクエリ（実写向けの本来の使い方） ====================
    query = "a yellow ball"
    qv = enc.encode_text([query])
    Dt, It = index.search(C.as_faiss(qv), 4)
    print(f"\n[text-query] {query!r} -> top4:")
    for rank, (fid, score) in enumerate(zip(It[0], Dt[0]), 1):
        m = meta[int(fid)]
        print(f"  #{rank} frame={m['frame']} cluster={m['cluster']} score={score:.3f}")
    if enc.backend == "clip":
        print("  注: 合成フラット画像では順位は不安定。実写ではこの仕組みが領域を当てる。")

    # === 3. 可視化: テキストクエリの top4 ヒット領域を重畳 ===============
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.8))
    H, W = frames[0][0].shape[:2]
    for rank in range(4):
        fid = int(It[0][rank])
        m = meta[fid]
        img_rgb = frames[m["frame"]][0][:, :, ::-1]
        lab_big = C.upscale_nearest(label_maps[m["frame"]], (H, W))
        mask = lab_big == m["cluster"]
        ov = C.overlay_region(img_rgb, mask, color=(255, 40, 40), alpha=0.55)
        axes[rank].imshow(ov)
        axes[rank].set_title(f"#{rank + 1} f{m['frame']} c{m['cluster']} ({Dt[0][rank]:.2f})")
        axes[rank].axis("off")
    fig.suptitle(f"Text query: '{query}' -> top-4 regions", fontsize=13)
    fig.tight_layout()
    path = out / "05_region_retrieval.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f"\n[save] {path}")


if __name__ == "__main__":
    main()
