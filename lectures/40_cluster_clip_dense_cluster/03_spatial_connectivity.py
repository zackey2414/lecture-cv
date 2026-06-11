"""03: 空間連結クラスタリング — なぜ connectivity グラフを付けるのか。

学ぶこと:
  - 普通の AgglomerativeClustering は特徴空間の近さだけで併合するので、画面の
    バラバラな位置のパッチが同じクラスタに入り、領域が飛び地になりがち。
  - sklearn の grid_to_graph で「隣り合うパッチ同士だけ」を結ぶ connectivity を渡すと、
    空間的に隣接したパッチしか併合されない => 各クラスタが必ず連結した1領域になる。
  - 連結成分の数で定量確認: connectivity あり -> 成分数 == クラスタ数（全部つながる）。
    connectivity なし -> 成分数 > クラスタ数（飛び地で断片化）。

実行:
  uv run python lectures/40_cluster_clip_dense_cluster/03_spatial_connectivity.py
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


def main() -> None:
    out = C.output_dir()
    enc = C.load_encoder()

    img_bgr, _ = C.make_scene(seed=2)
    x = torch.stack([enc.preprocess_bgr(img_bgr)])
    fmap = enc.dense(x)[0]  # [C, gh, gw]
    _, gh, gw = fmap.shape

    # === 1. connectivity グラフの中身を確認 =============================
    graph = C.grid_connectivity(gh, gw).tocsr()
    # 内部ノード（端でないパッチ）の隣接数は 4 のはず（上下左右）。対角成分込みで 5。
    center = (gh // 2) * gw + (gw // 2)
    deg = graph[center].nnz  # 自己ループ込みの非ゼロ数
    print(f"[graph] grid_to_graph({gh}x{gw}) の中央パッチの非ゼロ隣接数={deg} (自分+上下左右=5)")

    n_clusters = 6
    # === 2. connectivity あり / なし でクラスタリング =====================
    reps_on, lab_on = C.cluster_regions(fmap, n_clusters, connectivity=True)
    reps_off, lab_off = C.cluster_regions(fmap, n_clusters, connectivity=False)

    comp_on = C.total_components(lab_on)
    comp_off = C.total_components(lab_off)
    print(f"\n[connectivity ON ] クラスタ数={len(np.unique(lab_on))}  連結成分数={comp_on}")
    print(f"[connectivity OFF] クラスタ数={len(np.unique(lab_off))}  連結成分数={comp_off}")
    print(
        "\n=> ON は『成分数==クラスタ数』で各領域が1つながり。"
        "\n   OFF は成分数がクラスタ数を上回りやすく、飛び地（断片化）が起きる。"
    )
    assert comp_on == len(np.unique(lab_on)), "connectivity ありなのに領域が連結していない"

    # === 3. 可視化: 原画像 | 連結あり | 連結なし =========================
    img_rgb = img_bgr[:, :, ::-1]
    H, W = img_rgb.shape[:2]
    col_on = C.upscale_nearest(lab_on, (H, W))
    col_off = C.upscale_nearest(lab_off, (H, W))
    vis_on = C.label_map_to_color(col_on)
    vis_off = C.label_map_to_color(col_off)

    fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.4))
    axes[0].imshow(img_rgb)
    axes[0].set_title("scene")
    axes[1].imshow(vis_on)
    axes[1].set_title(f"connectivity ON ({comp_on} parts)")
    axes[2].imshow(vis_off)
    axes[2].set_title(f"connectivity OFF ({comp_off} parts)")
    for ax in axes:
        ax.axis("off")
    fig.suptitle("Spatial connectivity makes regions contiguous", fontsize=13)
    fig.tight_layout()
    path = out / "03_connectivity.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f"\n[save] {path}")
    _ = (reps_on, reps_off)


if __name__ == "__main__":
    main()
