"""04: 領域クラスタリングと代表ベクトル — Cluster-CLIP の Build 中核。

学ぶこと:
  - dense 特徴 [C,H,W] を空間連結クラスタリングで k 個の領域に割り、各クラスタの
    平均 -> L2 正規化を「クラスタ代表ベクトル」にする（参照リポ cluster_agglomerative）。
  - 代表ベクトルは k 本だけ。49 パッチを全部 index に入れるより軽く、領域単位で引ける。
  - 小物体（黄色いボール）が自分専用のクラスタを得られているか、bbox とのカバレッジで確認。

実行:
  uv run python lectures/40_cluster_clip_dense_cluster/04_agglomerative_region_cluster.py
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


def coverage_ratio(mask_bool: np.ndarray, bbox: tuple[int, int, int, int]) -> float:
    """クラスタマスクが bbox をどれだけ覆うか = (マスク∩bbox) / bbox面積。

    参照リポ eval/coverage.py のカバレッジ評価の最小版。
    """
    x1, y1, x2, y2 = bbox
    box_area = max(1, (x2 - x1) * (y2 - y1))
    inter = mask_bool[y1:y2, x1:x2].sum()
    return float(inter) / float(box_area)


def main() -> None:
    out = C.output_dir()
    enc = C.load_encoder()

    img_bgr, regions = C.make_scene(seed=0)
    x = torch.stack([enc.preprocess_bgr(img_bgr)])
    fmap = enc.dense(x)[0]  # [C, gh, gw]
    c, gh, gw = fmap.shape

    n_clusters = 6
    reps, label_map = C.cluster_regions(fmap, n_clusters, connectivity=True)
    print(f"[build] dense {fmap.shape} -> {len(reps)} 領域 / 代表ベクトル {reps.shape}")

    # 代表ベクトルが単位ベクトルか
    rn = np.linalg.norm(reps, axis=1)
    assert np.allclose(rn, 1.0, atol=1e-3), "代表ベクトルが正規化されていない"
    print(f"[check] 代表ベクトルのノルム in [{rn.min():.3f}, {rn.max():.3f}] -> OK")

    # === 小物体が専用クラスタを得たか（bbox カバレッジ） =================
    H, W = img_bgr.shape[:2]
    label_big = C.upscale_nearest(label_map, (H, W))
    print("\n[coverage] 各物体を最もよく覆うクラスタ:")
    for r in regions:
        best_cid, best_cov = -1, 0.0
        for cid in np.unique(label_map):
            cov = coverage_ratio(label_big == cid, r["bbox"])
            if cov > best_cov:
                best_cid, best_cov = int(cid), cov
        print(f"  {r['label']:14s} -> cluster {best_cid} (coverage={best_cov:.2f})")

    # === 可視化: 原画像 | クラスタ色分け | 各クラスタ重畳 ================
    img_rgb = img_bgr[:, :, ::-1]
    color_map = C.label_map_to_color(label_big)

    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    axes[0, 0].imshow(img_rgb)
    axes[0, 0].set_title("scene")
    axes[0, 1].imshow(color_map)
    axes[0, 1].set_title(f"{len(reps)} region clusters")
    # PCA-RGB も並べる
    feat_rgb = C.features_to_rgb(fmap)
    feat_vis = np.stack(
        [C.upscale_nearest(feat_rgb[..., k], (H, W)) for k in range(3)], axis=-1
    ).astype(np.uint8)
    axes[0, 2].imshow(feat_vis)
    axes[0, 2].set_title("dense (PCA->RGB)")
    axes[0, 3].axis("off")

    # 下段: クラスタごとの重畳（最大4つ）
    cids = list(np.unique(label_map))[:4]
    for i, cid in enumerate(cids):
        mask = label_big == cid
        ov = C.overlay_region(img_rgb, mask, color=(255, 30, 30), alpha=0.55)
        axes[1, i].imshow(ov)
        axes[1, i].set_title(f"cluster {int(cid)}")
    for j in range(len(cids), 4):
        axes[1, j].axis("off")
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Region clustering -> representative vectors", fontsize=13)
    fig.tight_layout()
    path = out / "04_region_clusters.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f"\n[save] {path}")
    _ = (c, gh, gw)


if __name__ == "__main__":
    main()
